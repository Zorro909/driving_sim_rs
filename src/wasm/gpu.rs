//! The WebGPU raycaster: `rays.wgsl` runs the scene's BSP `RayTree` queries
//! (raycasts and closest wall points) in a compute shader, serving the ray
//! sensors of `TrainingRunner`'s ray window and batch queries for JavaScript.
//!
//! `float.wgsl` preserves the CPU's rounding boundaries and divides binary32
//! significands with integers, avoiding WGSL fusion and approximate division.
//! `verify` compares a device with the CPU on random queries, and
//! `gpuVerifyEvery` samples the sensor rays of every window. These checks
//! diagnose hardware differences without replacing the GPU's results.

use super::{Shared, Simulation};
use crate::math::godot_math::F2;
use crate::math::vec2::V2;
use crate::track::bsp::RayTree;
use js_sys::{ArrayBuffer, Float32Array, Promise};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::{future_to_promise, JsFuture};
use web_sys::{
    gpu_buffer_usage as usage, gpu_map_mode, GpuAdapter, GpuBindGroup, GpuBindGroupDescriptor, GpuBindGroupEntry,
    GpuBuffer, GpuBufferBinding, GpuBufferDescriptor, GpuCompilationInfo, GpuCompilationMessage, GpuComputePipeline,
    GpuComputePipelineDescriptor, GpuDevice, GpuPipelineLayout, GpuProgrammableStage, GpuQueue, GpuShaderModule,
    GpuShaderModuleDescriptor,
};

/// The BSP arrays of `RayTree::gpu_arrays`.
pub type TreeArrays = (Vec<([u32; 4], [f32; 4], [f32; 4])>, Vec<[f32; 4]>, f32, usize);

const FLOAT_SHADER: &str = include_str!("float.wgsl");
const RAY_SHADER: &str = include_str!("rays.wgsl");
/// `MAX_DEPTH` of rays.wgsl: the deepest BSP the shader's stack holds.
pub const MAX_DEPTH: usize = 48;
const WORKGROUP: u32 = 64;
/// `params.op` of rays.wgsl.
pub const OP_RAYCAST: u32 = 0;
pub const OP_CLOSEST_WALL: u32 = 1;

fn error(message: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&message.to_string()).into()
}

fn describe(e: JsValue) -> String {
    e.dyn_ref::<js_sys::Error>()
        .map_or_else(|| format!("{e:?}"), |e| String::from(e.message()))
}

/// The bytes of plain arrays of 32-bit words; WebAssembly is little-endian, as GPU buffers are.
fn bytes<T: Copy>(rows: &[T]) -> &[u8] {
    // SAFETY: T is `[u32; 12]` or `[f32; 4]`, without padding; the slice
    // covers exactly the rows' memory for reading.
    unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), std::mem::size_of_val(rows)) }
}

/// The CPU raycast in the shader's result layout.
pub(crate) fn cpu_raycast(tree: &RayTree, q: &[f32; 4]) -> [f32; 4] {
    match tree.raycast(V2::new(q[0] as f64, q[1] as f64), V2::new(q[2] as f64, q[3] as f64)) {
        Some(hit) => {
            let h = F2::from(hit);
            [h.x, h.y, 1.0, 0.0]
        }
        None => [0.0; 4],
    }
}

/// Whether two results denote the same hit (bit for bit) or the same miss.
pub(crate) fn same_result(a: &[f32; 4], b: &[f32; 4]) -> bool {
    a[2] == b[2] && (a[2] == 0.0 || (a[0].to_bits() == b[0].to_bits() && a[1].to_bits() == b[1].to_bits()))
}

struct Batch {
    capacity: u32,
    queries: GpuBuffer,
    results: GpuBuffer,
    readback: GpuBuffer,
    bind_group: GpuBindGroup,
}

/// The device resources of one scene's ray tree.
pub struct Raycaster {
    device: GpuDevice,
    queue: GpuQueue,
    pipeline: GpuComputePipeline,
    params: GpuBuffer,
    nodes: GpuBuffer,
    walls: GpuBuffer,
    batch: RefCell<Option<Batch>>,
    in_flight: Cell<bool>,
    pub node_count: u32,
    pub wall_count: u32,
    pub depth: u32,
    magnitude: f32,
}

/// The shader compiler's messages, one per line.
async fn compilation_messages(module: &GpuShaderModule) -> String {
    let Ok(info) = JsFuture::from(module.get_compilation_info()).await else {
        return String::new();
    };
    let info: GpuCompilationInfo = info.unchecked_into();
    info.messages()
        .iter()
        .map(|m| {
            let m: GpuCompilationMessage = JsValue::from(m).unchecked_into();
            format!("\n  rays.wgsl:{}:{}: {}", m.line_num(), m.line_pos(), m.message())
        })
        .collect()
}

impl Raycaster {
    /// Compiles the shader and uploads the tree of `arrays` (`RayTree::gpu_arrays`).
    pub async fn create(device: GpuDevice, arrays: TreeArrays) -> Result<Raycaster, JsValue> {
        let (nodes, walls, magnitude, depth) = arrays;
        if depth > MAX_DEPTH {
            return Err(error(format!(
                "the BSP depth {depth} exceeds the shader's stack of {MAX_DEPTH}"
            )));
        }
        let rows: Vec<[u32; 12]> = nodes
            .iter()
            .map(|(l, o, s)| {
                [
                    l[0],
                    l[1],
                    l[2],
                    l[3],
                    o[0].to_bits(),
                    o[1].to_bits(),
                    o[2].to_bits(),
                    o[3].to_bits(),
                    s[0].to_bits(),
                    s[1].to_bits(),
                    s[2].to_bits(),
                    s[3].to_bits(),
                ]
            })
            .collect();
        let queue = device.queue();
        let shader = format!("{FLOAT_SHADER}\n{RAY_SHADER}");
        let module = device.create_shader_module(&GpuShaderModuleDescriptor::new(&shader));
        let stage = GpuProgrammableStage::new(&module);
        stage.set_entry_point("main");
        // `layout: "auto"`: the string stands in for a pipeline layout object.
        let auto: GpuPipelineLayout = JsValue::from_str("auto").unchecked_into();
        let pipeline = match JsFuture::from(
            device.create_compute_pipeline_async(&GpuComputePipelineDescriptor::new(&auto, &stage)),
        )
        .await
        {
            Ok(pipeline) => pipeline.unchecked_into::<GpuComputePipeline>(),
            Err(e) => {
                return Err(error(format!(
                    "the ray shader did not compile: {}{}",
                    describe(e),
                    compilation_messages(&module).await
                )))
            }
        };
        let params = Self::create_buffer(&device, 32, usage::UNIFORM | usage::COPY_DST)?;
        let wall_count = walls.len() as u32;
        let nodes = Self::upload(&device, &queue, bytes(&rows))?;
        let walls = Self::upload(&device, &queue, bytes(&walls))?;
        Ok(Raycaster {
            device,
            queue,
            pipeline,
            params,
            nodes,
            walls,
            batch: RefCell::new(None),
            in_flight: Cell::new(false),
            node_count: rows.len() as u32,
            wall_count,
            depth: depth as u32,
            magnitude,
        })
    }

    fn create_buffer(device: &GpuDevice, size: u32, usage: u32) -> Result<GpuBuffer, JsValue> {
        device.create_buffer(&GpuBufferDescriptor::new(size.max(16).next_multiple_of(16), usage))
    }

    fn upload(device: &GpuDevice, queue: &GpuQueue, data: &[u8]) -> Result<GpuBuffer, JsValue> {
        let buffer = Self::create_buffer(device, data.len() as u32, usage::STORAGE | usage::COPY_DST)?;
        if !data.is_empty() {
            queue.write_buffer_with_u32_and_u8_slice(&buffer, 0, data)?;
        }
        Ok(buffer)
    }

    /// Query buffers for at least `count` queries, grown in powers of two.
    fn ensure_batch(&self, count: u32) -> Result<(), JsValue> {
        if self.batch.borrow().as_ref().is_some_and(|b| b.capacity >= count) {
            return Ok(());
        }
        let capacity = count.next_power_of_two().max(1024);
        let size = capacity * 16;
        let queries = Self::create_buffer(&self.device, size, usage::STORAGE | usage::COPY_DST)?;
        let results = Self::create_buffer(&self.device, size, usage::STORAGE | usage::COPY_SRC)?;
        let readback = Self::create_buffer(&self.device, size, usage::MAP_READ | usage::COPY_DST)?;
        let entries: Vec<GpuBindGroupEntry> = [&self.params, &self.nodes, &self.walls, &queries, &results]
            .iter()
            .enumerate()
            .map(|(i, buffer)| GpuBindGroupEntry::new_with_gpu_buffer_binding(i as u32, &GpuBufferBinding::new(buffer)))
            .collect();
        let bind_group = self.device.create_bind_group(&GpuBindGroupDescriptor::new(
            &entries,
            &self.pipeline.get_bind_group_layout(0),
        ));
        *self.batch.borrow_mut() = Some(Batch {
            capacity,
            queries,
            results,
            readback,
            bind_group,
        });
        Ok(())
    }

    /// Runs `op` over `input` (`[f32; 4]` per query; see rays.wgsl) and reads
    /// the results back. Queries run one at a time per raycaster.
    pub async fn query(&self, op: u32, input: &[[f32; 4]]) -> Result<Vec<[f32; 4]>, JsValue> {
        let count = input.len() as u32;
        if count == 0 {
            return Ok(Vec::new());
        }
        if self.in_flight.replace(true) {
            return Err(error("a query is in flight; await it first"));
        }
        let result = self.run(op, input, count).await;
        self.in_flight.set(false);
        result
    }

    async fn run(&self, op: u32, input: &[[f32; 4]], count: u32) -> Result<Vec<[f32; 4]>, JsValue> {
        self.ensure_batch(count)?;
        let size = count * 16;
        let readback = {
            let batch = self.batch.borrow();
            let batch = batch.as_ref().expect("batch buffers");
            let params = [self.node_count, count, op, 0, self.magnitude.to_bits(), 0, 0, 0];
            self.queue
                .write_buffer_with_u32_and_u8_slice(&self.params, 0, bytes(&params))?;
            self.queue
                .write_buffer_with_u32_and_u8_slice(&batch.queries, 0, bytes(input))?;
            let encoder = self.device.create_command_encoder();
            let pass = encoder.begin_compute_pass();
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&batch.bind_group));
            pass.dispatch_workgroups(count.div_ceil(WORKGROUP));
            pass.end();
            encoder.copy_buffer_to_buffer_with_u32_and_u32_and_u32(&batch.results, 0, &batch.readback, 0, size)?;
            self.queue.submit(&[encoder.finish()]);
            batch.readback.clone()
        };
        JsFuture::from(readback.map_async_with_u32_and_u32(gpu_map_mode::READ, 0, size)).await?;
        let range: ArrayBuffer = readback.get_mapped_range_with_u32_and_u32(0, size)?;
        let floats = Float32Array::new(&range).to_vec();
        readback.unmap();
        Ok(floats.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect())
    }
}

/// `navigator.gpu` of the window or worker, if WebGPU is available.
fn navigator_gpu() -> Result<web_sys::Gpu, JsValue> {
    let navigator: JsValue = match web_sys::window() {
        Some(window) => window.navigator().into(),
        None => js_sys::global()
            .dyn_into::<web_sys::WorkerGlobalScope>()
            .map_err(|_| error("no window or worker scope"))?
            .navigator()
            .into(),
    };
    let gpu = js_sys::Reflect::get(&navigator, &"gpu".into())?;
    if gpu.is_undefined() || gpu.is_null() {
        return Err(error("WebGPU is not available in this browser"));
    }
    Ok(gpu.unchecked_into())
}

/// Requests a WebGPU adapter and device: a promise of a `GPUDevice`.
#[wasm_bindgen(js_name = requestGpuDevice)]
pub fn request_gpu_device(require_hardware: Option<bool>) -> Promise {
    future_to_promise(async move {
        let gpu = navigator_gpu()?;
        let adapter = JsFuture::from(gpu.request_adapter()).await?;
        if adapter.is_null() || adapter.is_undefined() {
            return Err(error("WebGPU: no adapter"));
        }
        let adapter: GpuAdapter = adapter.unchecked_into();
        if require_hardware.unwrap_or(false) {
            let info = js_sys::Reflect::get(adapter.as_ref(), &JsValue::from_str("info"))?;
            let fallback = js_sys::Reflect::get(&info, &JsValue::from_str("isFallbackAdapter"))?
                .as_bool()
                .unwrap_or(false);
            if fallback {
                return Err(error(
                    "WebGPU training needs a hardware adapter; the software adapter runs on the CPU",
                ));
            }
        }
        JsFuture::from(adapter.request_device()).await.map(JsValue::from)
    })
}

/// A scene's BSP walls on a WebGPU device.
#[wasm_bindgen]
pub struct GpuRaycaster {
    inner: Rc<Raycaster>,
}

fn flatten(results: &[[f32; 4]]) -> JsValue {
    let flat: Vec<f32> = results.iter().flatten().copied().collect();
    Float32Array::from(&flat[..]).into()
}

fn queries(values: &[f32]) -> Result<Vec<[f32; 4]>, JsValue> {
    if !values.len().is_multiple_of(4) {
        return Err(error("queries take four values each"));
    }
    Ok(values.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect())
}

#[wasm_bindgen]
impl GpuRaycaster {
    /// Compiles the shader and uploads the simulation's track to `device`
    /// (from `requestGpuDevice()` or the host's own `GPUDevice`); resolves to
    /// the raycaster, or rejects with the shader diagnostics.
    pub fn create(device: GpuDevice, simulation: &Simulation) -> Promise {
        let arrays = {
            let session = simulation.shared.session.borrow();
            session
                .runner
                .world
                .track
                .ray_tree()
                .map(RayTree::gpu_arrays)
                .ok_or_else(|| error("the scene has no BSP raycaster"))
        };
        future_to_promise(async move {
            let raycaster = Raycaster::create(device, arrays?).await?;
            Ok(GpuRaycaster {
                inner: Rc::new(raycaster),
            }
            .into())
        })
    }

    pub(crate) fn inner(&self) -> Rc<Raycaster> {
        self.inner.clone()
    }

    #[wasm_bindgen(getter, js_name = nodeCount)]
    pub fn node_count(&self) -> u32 {
        self.inner.node_count
    }

    #[wasm_bindgen(getter, js_name = wallCount)]
    pub fn wall_count(&self) -> u32 {
        self.inner.wall_count
    }

    #[wasm_bindgen(getter)]
    pub fn depth(&self) -> u32 {
        self.inner.depth
    }

    /// Casts rays `[startX, startY, endX, endY]...`; resolves to a
    /// Float32Array of `[hitX, hitY, hit ? 1 : 0, 0]` per ray.
    pub fn raycast(&self, rays: &[f32]) -> Promise {
        let inner = self.inner.clone();
        let rays = queries(rays);
        future_to_promise(async move { Ok(flatten(&inner.query(OP_RAYCAST, &rays?).await?)) })
    }

    /// The closest wall point of `[x, y, 0, 0]...`; resolves to
    /// `[x, y, 1, 0]` per point.
    #[wasm_bindgen(js_name = closestWall)]
    pub fn closest_wall(&self, points: &[f32]) -> Promise {
        let inner = self.inner.clone();
        let points = queries(points);
        future_to_promise(async move { Ok(flatten(&inner.query(OP_CLOSEST_WALL, &points?).await?)) })
    }

    /// Compares the device with the CPU on `rays` random rays and `points`
    /// random points over the track's bounds: resolves to
    /// `{rays, rayMismatches, points, pointMismatches}`.
    pub fn verify(&self, simulation: &Simulation, rays: u32, points: u32, seed: u32) -> Promise {
        let inner = self.inner.clone();
        let shared = simulation.shared.clone();
        future_to_promise(async move {
            let (ray_queries, point_queries, expected_rays, expected_points) = {
                let session = shared.session.borrow();
                let track = &session.runner.world.track;
                let tree = track.ray_tree().ok_or_else(|| error("no BSP raycaster"))?;
                let bounds = session.track_bounds();
                let mut rng = Xorshift(seed.max(1));
                let ray_queries: Vec<[f32; 4]> = (0..rays)
                    .map(|_| {
                        let start = rng.point(&bounds);
                        let angle = rng.unit() * std::f64::consts::TAU;
                        let length = 50.0 + rng.unit() * 750.0;
                        let end = V2::new(start.x + angle.cos() * length, start.y + angle.sin() * length);
                        let (s, e) = (F2::from(start), F2::from(end));
                        [s.x, s.y, e.x, e.y]
                    })
                    .collect();
                let point_queries: Vec<[f32; 4]> = (0..points)
                    .map(|_| {
                        let p = F2::from(rng.point(&bounds));
                        [p.x, p.y, 0.0, 0.0]
                    })
                    .collect();
                let expected_rays: Vec<[f32; 4]> = ray_queries.iter().map(|q| cpu_raycast(tree, q)).collect();
                let expected_points: Vec<[f32; 4]> = point_queries
                    .iter()
                    .map(|q| match track.closest_wall(V2::new(q[0] as f64, q[1] as f64)) {
                        Some(p) => {
                            let p = F2::from(p);
                            [p.x, p.y, 1.0, 0.0]
                        }
                        None => [0.0; 4],
                    })
                    .collect();
                (ray_queries, point_queries, expected_rays, expected_points)
            };
            let got_rays = inner.query(OP_RAYCAST, &ray_queries).await?;
            let got_points = inner.query(OP_CLOSEST_WALL, &point_queries).await?;
            let ray_mismatches = got_rays
                .iter()
                .zip(&expected_rays)
                .filter(|(a, b)| !same_result(a, b))
                .count();
            let point_mismatches = got_points
                .iter()
                .zip(&expected_points)
                .filter(|(a, b)| !same_result(a, b))
                .count();
            let first_mismatch = |queries: &[[f32; 4]], got: &[[f32; 4]], expected: &[[f32; 4]]| {
                got.iter().zip(expected).position(|(a, b)| !same_result(a, b)).map(|i| {
                    serde_json::json!({
                        "index": i, "query": queries[i], "got": got[i], "expected": expected[i],
                        "gotBits": got[i].map(f32::to_bits), "expectedBits": expected[i].map(f32::to_bits),
                    })
                })
            };
            let report = serde_json::json!({
                "rays": rays, "rayMismatches": ray_mismatches, "points": points, "pointMismatches": point_mismatches,
                "firstRayMismatch": first_mismatch(&ray_queries, &got_rays, &expected_rays),
                "firstPointMismatch": first_mismatch(&point_queries, &got_points, &expected_points),
            });
            js_sys::JSON::parse(&report.to_string())
        })
    }
}

/// A small deterministic generator for `verify`.
struct Xorshift(u32);

impl Xorshift {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f64 {
        self.next() as f64 / 4294967296.0
    }
    fn point(&mut self, bounds: &[f64; 4]) -> V2 {
        let pad = 64.0;
        V2::new(
            bounds[0] - pad + self.unit() * (bounds[2] - bounds[0] + 2.0 * pad),
            bounds[1] - pad + self.unit() * (bounds[3] - bounds[1] + 2.0 * pad),
        )
    }
}

/// `TrainingRunner`'s ray window with the ray sensors served by `raycaster`:
/// per tick, the statistics and the queries run on the CPU, the rays on the
/// GPU, then inference and the physics step on the CPU. With
/// `time_limit_ticks`, the window is the rest of the generation.
pub(crate) fn advance_window(
    shared: Rc<Shared>,
    raycaster: Rc<Raycaster>,
    ticks: u64,
    stop_when_inactive: bool,
    time_limit_ticks: Option<u64>,
) -> Promise {
    future_to_promise(async move {
        if shared.busy.replace(true) {
            return Err(error("a WebGPU window is already in flight"));
        }
        let result = run_window(&shared, &raycaster, ticks, stop_when_inactive, time_limit_ticks).await;
        shared.busy.set(false);
        result.map(|executed| JsValue::from(executed as u32))
    })
}

async fn run_window(
    shared: &Shared,
    raycaster: &Raycaster,
    ticks: u64,
    stop_when_inactive: bool,
    time_limit_ticks: Option<u64>,
) -> Result<u64, JsValue> {
    shared.revision.set(shared.revision.get().wrapping_add(1));
    let (mut window, verify_every) = {
        let mut session = shared.session.borrow_mut();
        if !session.started() {
            return Err(error("the session has not started"));
        }
        let (ticks, time_limit) = match time_limit_ticks {
            Some(limit) => {
                if session.runner.stats_phase != 0 {
                    return Err(error("advanceGeneration needs statistics phase 0"));
                }
                let (ticks, seconds) = session.runner.generation_window(limit);
                (ticks, Some(seconds))
            }
            None => (ticks, None),
        };
        let window = session
            .runner
            .begin_ray_window(ticks, stop_when_inactive, time_limit)
            .map_err(error)?;
        (window, session.options.gpu_verify_every as usize)
    };
    let mut queries = Vec::new();
    loop {
        {
            let mut session = shared.session.borrow_mut();
            if !session.runner.prepare_ray_tick(&mut window) {
                break;
            }
            session.runner.ray_queries(&window, &mut queries);
        }
        let hits = raycaster.query(OP_RAYCAST, &queries).await?;
        if verify_every > 0 {
            let session = shared.session.borrow();
            let tree = session
                .runner
                .world
                .track
                .ray_tree()
                .ok_or_else(|| error("no BSP raycaster"))?;
            let (mut checked, mut mismatches) = (0, 0);
            for (query, hit) in queries.iter().zip(&hits).step_by(verify_every) {
                checked += 1;
                if !same_result(&cpu_raycast(tree, query), hit) {
                    mismatches += 1;
                }
            }
            shared.gpu_rays_checked.set(shared.gpu_rays_checked.get() + checked);
            shared
                .gpu_ray_mismatches
                .set(shared.gpu_ray_mismatches.get() + mismatches);
        }
        shared
            .session
            .borrow_mut()
            .runner
            .finish_ray_tick(&window, &hits)
            .map_err(error)?;
    }
    Ok(shared.session.borrow_mut().runner.end_ray_window(window))
}
