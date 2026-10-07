//! Full WebGPU training windows. Evolution and checkpoint serialization stay
//! in Rust; sensing, inference, statistics and physics run entirely in WGSL.
use super::{js_error, Shared, Simulation};
use crate::gpu::simulation::{self, words, GpuAgent, GpuCar, RayNode, TrackArrays};
use js_sys::{Promise, Uint32Array};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{future_to_promise, JsFuture};
use web_sys::GpuDevice;
mod world;
struct BusyGuard(Rc<Shared>);
impl Drop for BusyGuard {
    fn drop(&mut self) {
        self.0.busy.set(false);
    }
}

#[wasm_bindgen(module = "/src/wasm/sim/runtime.js")]
extern "C" {
    #[wasm_bindgen(catch, js_name = createSimulator)]
    async fn create_simulator(
        device: &GpuDevice,
        source: &str,
        world: &Uint32Array,
        base: &Uint32Array,
        progress: Option<js_sys::Function>,
    ) -> Result<JsValue, JsValue>;
    #[wasm_bindgen(typescript_type = "object")]
    type Backend;
    #[wasm_bindgen(method)]
    fn upload(
        this: &Backend,
        cars: &Uint32Array,
        agents: &Uint32Array,
        networks: &Uint32Array,
        base: &Uint32Array,
    ) -> Promise;
    #[wasm_bindgen(method)]
    fn advance(this: &Backend, window: &Uint32Array, reset: bool) -> Promise;
    #[wasm_bindgen(method, catch, js_name = setWorld)]
    fn set_world(this: &Backend, world: &Uint32Array) -> Result<(), JsValue>;
    #[wasm_bindgen(method)]
    fn destroy(this: &Backend);
    #[wasm_bindgen(method, catch, js_name = setLoad)]
    fn set_load(this: &Backend, load: f64) -> Result<(), JsValue>;
}
const SHADER: &str = concat!(
    include_str!("f64.wgsl"),
    "\n",
    include_str!("helpers.wgsl"),
    "\n",
    include_str!("../float.wgsl"),
    "\n",
    include_str!("generated.wgsl"),
    "\n",
    include_str!("window.wgsl")
);
const MAX_WINDOW_TICKS: u32 = 120;

// These mirrors contain only numbers and arrays of numbers. The explicit
// little-endian word buffers match the generated raw WGSL loaders. Numeric
// fields and initialized, explicit padding cover each copied type's layout.
const _: () = {
    assert!(std::mem::size_of::<GpuCar>() == 3200);
    assert!(std::mem::size_of::<GpuAgent>() == 376);
    assert!(std::mem::size_of::<simulation::VehicleDesc>() == 336);
    assert!(std::mem::size_of::<simulation::SensorDesc>() == 24);
    assert!(std::mem::offset_of!(GpuCar, contacts) == 384);
    assert!(std::mem::offset_of!(GpuAgent, flags) == 368);
};
fn append<T: Copy>(data: &mut Vec<u32>, values: &[T]) -> Result<u32, String> {
    let offset = u32::try_from(data.len()).map_err(|_| "WebGPU world exceeds address range")?;
    data.extend(words(values));
    Ok(offset)
}
unsafe fn append_ptr<T: Copy>(data: &mut Vec<u32>, pointer: *const T, count: usize) -> Result<u32, String> {
    if count == 0 {
        return Ok(0);
    }
    // SAFETY: WorldDesc points into its owned TrackArrays for these counts.
    append(data, unsafe { std::slice::from_raw_parts(pointer, count) })
}

struct Inner {
    backend: Backend,
    shared: Rc<Shared>,
    surfaces: RefCell<simulation::SurfaceTable>,
    /// The session track the world buffer encodes.
    world: RefCell<std::sync::Arc<crate::track::world::World>>,
    base: RefCell<Vec<u32>>,
    tag: Cell<Option<(u64, u64)>>,
    population: Cell<usize>,
    busy: Cell<bool>,
    shape: Vec<usize>,
}
impl Drop for Inner {
    fn drop(&mut self) {
        self.backend.destroy();
    }
}

#[wasm_bindgen]
pub struct GpuSimulation {
    inner: Rc<Inner>,
}
#[wasm_bindgen]
impl GpuSimulation {
    #[wasm_bindgen(js_name = create)]
    /// Compiles the simulator for `simulation` and uploads its state.
    /// `progress`, if given, is called as `progress(done, total, kernel)`
    /// before each of the compute pipelines compiles and once when all have.
    pub async fn create(
        device: GpuDevice,
        simulation: &Simulation,
        progress: Option<js_sys::Function>,
    ) -> Result<GpuSimulation, JsError> {
        if simulation.shared.busy.get() {
            return Err(js_error("a GPU window is in flight"));
        }
        simulation.shared.busy.set(true);
        let _guard = BusyGuard(simulation.shared.clone());
        let encoded = {
            let session = simulation.shared.session.borrow();
            encode(&session).map_err(js_error)?
        };
        let world = encoded.world;
        let base = encoded.base;
        let (surfaces, shape) = (encoded.surfaces, encoded.shape);
        let track = simulation.shared.session.borrow().runner.world.clone();
        let backend = create_simulator(
            &device,
            &SHADER
                .replace("i < 55u", "i < 55u + params.pad")
                .replace("i < 23u", "i < 23u + params.pad"),
            &Uint32Array::from(world.as_slice()),
            &Uint32Array::from(base.as_slice()),
            progress,
        )
        .await
        .map_err(|e| js_error(format!("WebGPU compilation: {e:?}")))?
        .unchecked_into();
        let inner = Rc::new(Inner {
            backend,
            shared: simulation.shared.clone(),
            surfaces: RefCell::new(surfaces),
            world: RefCell::new(track),
            base: RefCell::new(base),
            tag: Cell::new(None),
            population: Cell::new(0),
            busy: Cell::new(false),
            shape,
        });
        inner.upload_if_needed().await.map_err(js_error)?;
        Ok(Self { inner })
    }
    /// Runs device-only ticks with one readback per bounded window.
    pub fn advance(&self, ticks: u32, stop_when_inactive: bool) -> Promise {
        advance_window(self.inner.clone(), ticks, stop_when_inactive, None)
    }
    #[wasm_bindgen(js_name = advanceGeneration)]
    pub fn advance_generation(&self, time_limit_ticks: u32) -> Promise {
        advance_window(self.inner.clone(), 0, true, Some(time_limit_ticks))
    }
    /// Limits the share of wall time the simulator keeps the GPU busy, in
    /// (0, 1]; the rest is left idle for the desktop and other applications.
    #[wasm_bindgen(js_name = setLoad)]
    pub fn set_load(&self, load: f64) -> Result<(), JsError> {
        self.inner
            .backend
            .set_load(load)
            .map_err(|e| js_error(format!("{e:?}")))
    }
    /// Compares a short window against WASM and restores its initial state.
    /// Checks remain inside Rust; the host receives only their count.
    pub fn verify(&self, ticks: u32) -> Promise {
        let inner = self.inner.clone();
        if ticks == 0 || ticks > 60 {
            return Promise::reject(&js_error("verification needs 1 to 60 ticks").into());
        }
        if inner.busy.get() || inner.shared.busy.get() {
            return Promise::reject(&js_error("a GPU window is in flight").into());
        }
        inner.busy.set(true);
        inner.shared.busy.set(true);
        future_to_promise(async move {
            if let Err(e) = inner.sync_world() {
                inner.busy.set(false);
                inner.shared.busy.set(false);
                return Err(js_error(e).into());
            }
            let (backup, tick, batch, expected) = {
                let mut s = inner.shared.session.borrow_mut();
                let r = &mut s.runner;
                let backup = r.agents.clone();
                let tick = r.tick;
                let batch = r.batch_index;
                r.advance(ticks as u64, false);
                let expected = r.canonical_state(&inner.surfaces.borrow());
                r.agents = backup.clone();
                r.tick = tick;
                r.batch_index = batch;
                (backup, tick, batch, expected)
            };
            inner.tag.set(None);
            let result = match expected {
                Err(e) => Err(e),
                Ok(expected) => match run_window(&inner, ticks, false, None).await {
                    Err(e) => Err(e),
                    Ok(_) => inner
                        .shared
                        .session
                        .borrow()
                        .runner
                        .canonical_state(&inner.surfaces.borrow())
                        .and_then(|actual| {
                            if actual != expected {
                                Err("WebGPU simulation verification differs from WASM".into())
                            } else {
                                Ok(expected.len() as u32)
                            }
                        }),
                },
            };
            {
                let mut s = inner.shared.session.borrow_mut();
                s.runner.agents = backup;
                s.runner.tick = tick;
                s.runner.batch_index = batch;
            }
            inner.tag.set(None);
            inner.shared.revision.set(inner.shared.revision.get().wrapping_add(1));
            inner.busy.set(false);
            inner.shared.busy.set(false);
            result.map(JsValue::from).map_err(|e| js_error(e).into())
        })
    }
}
struct Encoded {
    world: Vec<u32>,
    base: Vec<u32>,
    surfaces: simulation::SurfaceTable,
    shape: Vec<usize>,
}
/// The world buffer (track, vehicle, sensors, network shape and controls)
/// and the uniform base describing its layout.
fn encode(session: &crate::training::session::Session) -> Result<Encoded, String> {
    let runner = &session.runner;

    if runner.world.track.native_broadphase || runner.world.track.shapes.is_empty() {
        return Err("WebGPU simulation needs independent cars and physics shapes".into());
    }
    let arrays = TrackArrays::new(&runner.world)?;
    let (nodes, walls, magnitude, depth) = runner
        .world
        .track
        .ray_tree()
        .ok_or("WebGPU simulation needs a BSP ray tree")?
        .gpu_arrays();
    if depth > 48 {
        return Err("BSP exceeds WebGPU ray stack depth".into());
    }
    let rays = (
        nodes
            .into_iter()
            .map(|(links, own, subtree)| RayNode { links, own, subtree })
            .collect(),
        walls,
        magnitude,
        depth,
    );
    let mut world = world::encode_world(&arrays.desc(&rays))?;
    let vehicle = simulation::vehicle_desc(&runner.world.vehicle)?;
    let vehicle_offset = append(&mut world, &[vehicle])?;
    let sensors: Vec<_> = runner
        .layout
        .sensors
        .iter()
        .map(|sensor| simulation::with_ray_offset(simulation::sensor_desc(sensor), runner.world.math))
        .collect();
    if sensors.len() > 64 {
        return Err("WebGPU supports at most 64 sensor inputs".into());
    }
    let sensors_offset = append(&mut world, &sensors)?;
    let first = runner.agents.first().ok_or("the session has not started")?;
    if first.network.shape.iter().any(|&n| n > 64) {
        return Err("WebGPU supports layer widths up to 64".into());
    }
    let shape: Vec<u32> = first.network.shape.iter().map(|&n| n as u32).collect();
    let shape_offset = append(&mut world, &shape)?;
    let sources = simulation::control_sources(session.output_names());
    let control_offset = append(&mut world, &sources)?;
    let mut base = vec![0u32; 24];
    base[12] = 0;
    base[13] = vehicle_offset;
    base[14] = sensors_offset;
    base[15] = sensors.len() as u32;
    base[16] = shape.len() as u32;
    base[17] = first.network.params.len() as u32;
    base[18] = shape_offset;
    base[19] = control_offset;
    base[20] = sensors.iter().any(|s| s.kind == 11 || s.kind == 12) as u32;
    base[23] = MAX_WINDOW_TICKS;
    if base[20] != 0 && runner.world.track.curve.is_none() && runner.world.track.path.len() >= 2 {
        return Err("WebGPU path sensors need the scene's Curve2D control points".into());
    }
    Ok(Encoded {
        world,
        base,
        surfaces: arrays.surfaces,
        shape: first.network.shape.clone(),
    })
}
impl Inner {
    /// Re-encodes the world buffer after `Simulation.replaceTrack`.
    fn sync_world(&self) -> Result<(), String> {
        let session = self.shared.session.borrow();
        if std::sync::Arc::ptr_eq(&session.runner.world, &self.world.borrow()) {
            return Ok(());
        }
        let encoded = encode(&session)?;
        if encoded.shape != self.shape {
            return Err("network shape changed; recreate the GPU simulator".into());
        }
        // The sensor layout, which specializes the kernels, belongs to the
        // session and stays the same.
        let mut base = self.base.borrow_mut();
        let population = base[0];
        *base = encoded.base;
        base[0] = population;
        self.backend
            .set_world(&Uint32Array::from(encoded.world.as_slice()))
            .map_err(|e| format!("GPU world: {e:?}"))?;
        *self.surfaces.borrow_mut() = encoded.surfaces;
        *self.world.borrow_mut() = session.runner.world.clone();
        // Car state refers to the surface table: upload it again.
        self.tag.set(None);
        Ok(())
    }
    async fn upload_if_needed(&self) -> Result<(), String> {
        self.sync_world()?;
        let (upload, generation, population) = {
            let session = self.shared.session.borrow();
            let r = &session.runner;
            if self.tag.get() == Some((r.generation, self.shared.revision.get()))
                && self.population.get() == r.agents.len()
            {
                return Ok(());
            }
            let mut cars = Vec::new();
            let mut agents = Vec::new();
            crate::training::export_state_into(
                &r.agents,
                &r.world.vehicle,
                &self.surfaces.borrow(),
                &mut cars,
                &mut agents,
            )?;
            if r.is_paused() {
                return Err("WebGPU does not model frozen cars".into());
            }
            let first = r.agents.first().ok_or("no agents")?;
            if first.network.shape != self.shape {
                return Err("network shape changed; recreate the GPU simulator".into());
            }
            let mut params = Vec::new();
            for a in &r.agents {
                if a.network.shape != first.network.shape {
                    return Err("population network shapes differ".into());
                }
                params.extend_from_slice(&a.network.params);
            }
            let mut base = self.base.borrow_mut();
            base[0] = r.agents.len() as u32;
            let generation = r.generation;
            let population = r.agents.len();
            let upload = self.backend.upload(
                &Uint32Array::from(words(&cars).as_slice()),
                &Uint32Array::from(words(&agents).as_slice()),
                &Uint32Array::from(words(&params).as_slice()),
                &Uint32Array::from(base.as_slice()),
            );
            (upload, generation, population)
        };
        JsFuture::from(upload).await.map_err(|e| format!("GPU upload: {e:?}"))?;
        self.tag.set(Some((generation, self.shared.revision.get())));
        self.population.set(population);
        Ok(())
    }
}
fn advance_window(inner: Rc<Inner>, ticks: u32, stop: bool, limit: Option<u32>) -> Promise {
    if inner.busy.get() || inner.shared.busy.get() {
        return Promise::reject(&js_error("a GPU window is in flight").into());
    }
    inner.busy.set(true);
    inner.shared.busy.set(true);
    future_to_promise(async move {
        let result = run_window(&inner, ticks, stop, limit).await;
        if result.is_err() {
            inner.tag.set(None);
        }
        inner.busy.set(false);
        inner.shared.busy.set(false);
        result.map(JsValue::from).map_err(|e| js_error(e).into())
    })
}
async fn run_window(inner: &Inner, ticks: u32, stop: bool, limit: Option<u32>) -> Result<u32, String> {
    let total = {
        let s = inner.shared.session.borrow();
        let r = &s.runner;
        if limit.is_some() && r.stats_phase != 0 {
            return Err("generation needs statistics phase zero".into());
        }
        let total = u32::try_from(limit.map_or(ticks as u64, |t| ((t as u64 / 6 + 2) * 6).saturating_sub(r.tick)))
            .map_err(|_| "GPU generation window exceeds u32 range")?;
        let tick = u32::try_from(r.tick).map_err(|_| "GPU tick exceeds u32 range")?;
        tick.checked_add(total)
            .filter(|&n| n < u32::MAX)
            .ok_or("GPU tick overflow")?;
        total
    };
    if total == 0 {
        return Ok(0);
    }
    inner.upload_if_needed().await?;
    let mut executed = 0;
    while executed < total {
        // Long submissions can trip a browser's GPU watchdog. Commit bounded
        // windows while keeping many device-only ticks between readbacks.
        let count = (total - executed).min(MAX_WINDOW_TICKS);
        let (ran, transition) = run_chunk(inner, count, stop, limit, executed == 0).await?;
        executed += ran;
        if transition {
            break;
        }
    }
    Ok(executed)
}
async fn run_chunk(
    inner: &Inner,
    ticks: u32,
    stop: bool,
    limit: Option<u32>,
    reset: bool,
) -> Result<(u32, bool), String> {
    let window = {
        let s = inner.shared.session.borrow();
        let r = &s.runner;
        if limit.is_some() && r.stats_phase != 0 {
            return Err("generation needs statistics phase zero".into());
        }
        let tick = u32::try_from(r.tick).map_err(|_| "GPU tick exceeds u32 range")?;
        tick.checked_add(ticks)
            .filter(|&n| n < u32::MAX)
            .ok_or("GPU tick overflow")?;
        let bits = (limit.unwrap_or(0) as f64 / 60.0).to_bits();
        vec![
            ticks,
            tick,
            r.batch_index as u32,
            r.batches_per_tick() as u32,
            r.stats_phase as u32,
            stop as u32,
            r.eliminate_on_wall as u32,
            r.eliminate_when_idle as u32,
            limit.is_some() as u32,
            bits as u32,
            (bits >> 32) as u32,
        ]
    };
    let value = JsFuture::from(inner.backend.advance(&Uint32Array::from(window.as_slice()), reset))
        .await
        .map_err(|e| format!("WebGPU window: {e:?}"))?;
    let data = Uint32Array::new(&value).to_vec();
    let n = inner.population.get();
    let car_words = std::mem::size_of::<GpuCar>() / 4;
    let agent_words = std::mem::size_of::<GpuAgent>() / 4;
    if data.len() != n * (car_words + agent_words) + 4 {
        return Err("GPU returned invalid state size".into());
    }
    fn decode<T: Copy>(data: &[u32], stride: usize) -> Vec<T> {
        data.chunks_exact(stride)
            .map(|row| {
                // SAFETY: the shader writes every numeric field of the POD mirrors.
                // Read unaligned because u32 arrays do not promise f64 alignment.
                unsafe { std::ptr::read_unaligned(row.as_ptr().cast::<T>()) }
            })
            .collect()
    }
    let cars = decode::<GpuCar>(&data[..n * car_words], car_words);
    let agents = decode::<GpuAgent>(&data[n * car_words..n * (car_words + agent_words)], agent_words);
    let header = &data[n * (car_words + agent_words)..];
    let executed = header[0];
    let transition = header[1];
    if executed == 0
        || executed > window[0]
        || transition > 1
        || (transition == 0 && executed != window[0])
        || (transition != 0 && (window[1] + executed) % 6 != window[4])
    {
        return Err(format!(
            "GPU returned stopping tick {executed}, transition {transition}, for {} ticks from {}",
            window[0], window[1]
        ));
    }
    // Check the whole result before mutating the CPU session, so fallback can
    // resume from its last valid window if shader domain checks fail.
    if let Some(c) = cars.iter().find(|c| c.error != 0) {
        return Err(format!("GPU simulation error bits {:#x}", c.error));
    }
    let mut session = inner.shared.session.borrow_mut();
    let r = &mut session.runner;
    if cars
        .iter()
        .any(|c| c.pair_count as usize > simulation::MAX_PAIRS || c.contact_count as usize > simulation::MAX_CONTACTS)
        || agents
            .iter()
            .any(|a| a.recent_len as usize > simulation::RECENT || a.recent_start as usize >= simulation::RECENT)
    {
        return Err("GPU returned invalid collision or statistics counts".into());
    }
    if cars.iter().any(|c| {
        c.wheel_surface[..r.world.vehicle.wheels.len()]
            .iter()
            .any(|&i| i as usize >= inner.surfaces.borrow().values.len())
    }) {
        return Err("GPU returned an invalid wheel surface".into());
    }
    r.import_state(&cars, &agents, &inner.surfaces.borrow())?;
    r.tick += executed as u64;
    r.batch_index = (r.batch_index + ((executed - transition) as usize % 8) * r.batches_per_tick()) % 8;
    inner.shared.revision.set(inner.shared.revision.get().wrapping_add(1));
    inner.tag.set(Some((r.generation, inner.shared.revision.get())));
    Ok((executed, transition != 0))
}
