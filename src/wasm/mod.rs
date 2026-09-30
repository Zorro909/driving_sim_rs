//! The WebAssembly library: wasm-bindgen exports of `session::Session` for
//! JavaScript, and the WebGPU raycaster in `gpu`. Build with
//! `wasm/build.sh` (see wasm/README.md).
//!
//! Every method borrows the session for its own duration only; a WebGPU
//! window in flight (`advanceWithGpuRays`) rejects further mutations until
//! its promise settles. Invalid scene or model files abort the module
//! through the library's assertions, which the console panic hook reports;
//! options, network exports, indices and shapes are validated and reported
//! as errors.

use crate::session::{Session, SessionOptions};
use crate::training::{CAR_STATE_FIELDS, CAR_STATE_STRIDE};
use js_sys::{Float32Array, Float64Array, Promise};
use std::cell::{Cell, RefCell, RefMut};
use std::rc::Rc;
use wasm_bindgen::prelude::*;

pub mod gpu;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// The crate version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

fn js_error(message: impl std::fmt::Display) -> JsError {
    JsError::new(&message.to_string())
}

fn parse(text: &str, what: &str) -> Result<serde_json::Value, JsError> {
    serde_json::from_str(text).map_err(|e| js_error(format!("{what}: {e}")))
}

fn to_js(value: &serde_json::Value) -> Result<JsValue, JsError> {
    js_sys::JSON::parse(&value.to_string()).map_err(|e| js_error(format!("{e:?}")))
}

/// State shared between a `Simulation` handle and its in-flight WebGPU windows.
pub(crate) struct Shared {
    pub session: RefCell<Session>,
    /// A WebGPU window is between its promise's creation and settlement.
    pub busy: Cell<bool>,
    pub gpu_rays_checked: Cell<u64>,
    pub gpu_ray_mismatches: Cell<u64>,
}

/// A population of cars on one track: the `TrainingRunner` of the native
/// trainer, driven from JavaScript.
#[wasm_bindgen]
pub struct Simulation {
    pub(crate) shared: Rc<Shared>,
}

#[wasm_bindgen]
impl Simulation {
    /// `sceneJson`, `networkJson` (inputs, outputs, optional weights) and
    /// `modelJson` are the files the CLI takes; `options` is an object or
    /// JSON string of `SessionOptions` (see wasm/README.md).
    #[wasm_bindgen(constructor)]
    pub fn new(scene_json: &str, network_json: &str, model_json: &str, options: JsValue) -> Result<Simulation, JsError> {
        let options = if options.is_undefined() || options.is_null() {
            SessionOptions::default()
        } else if let Some(text) = options.as_string() {
            SessionOptions::from_json(&text).map_err(js_error)?
        } else {
            let text = js_sys::JSON::stringify(&options).map_err(|e| js_error(format!("options: {e:?}")))?;
            SessionOptions::from_json(&String::from(text)).map_err(js_error)?
        };
        let session = Session::new(&parse(scene_json, "scene")?, &parse(network_json, "network")?, &parse(model_json, "model")?, options)
            .map_err(js_error)?;
        Ok(Simulation {
            shared: Rc::new(Shared { session: RefCell::new(session), busy: Cell::new(false), gpu_rays_checked: Cell::new(0), gpu_ray_mismatches: Cell::new(0) }),
        })
    }

    fn session(&self) -> std::cell::Ref<'_, Session> {
        self.shared.session.borrow()
    }

    fn session_mut(&self) -> Result<RefMut<'_, Session>, JsError> {
        if self.shared.busy.get() {
            return Err(js_error("a WebGPU window is in flight; await its promise first"));
        }
        Ok(self.shared.session.borrow_mut())
    }

    /// Installs the first generation from the network export's weights, or
    /// from a Xavier network of its shape.
    pub fn start(&self) -> Result<(), JsError> {
        self.session_mut()?.start().map_err(js_error)
    }

    /// Installs the first generation from a Xavier network of `shape`
    /// (inputs first), drawn from the session's seeded RNG.
    #[wasm_bindgen(js_name = startWithShape)]
    pub fn start_with_shape(&self, shape: &[u32]) -> Result<(), JsError> {
        let shape: Vec<usize> = shape.iter().map(|&n| n as usize).collect();
        self.session_mut()?.start_with_shape(&shape).map_err(js_error)
    }

    /// Advances up to `ticks` physics ticks (60 per second); with
    /// `stopWhenInactive` the window ends once every car is inactive.
    /// Returns the executed ticks.
    pub fn advance(&self, ticks: u32, stop_when_inactive: bool) -> Result<u32, JsError> {
        self.session_mut()?.advance(ticks as u64, stop_when_inactive).map(|n| n as u32).map_err(js_error)
    }

    /// Runs the rest of the generation under a time limit in ticks, as the
    /// game does. Returns the executed ticks.
    #[wasm_bindgen(js_name = advanceGeneration)]
    pub fn advance_generation(&self, time_limit_ticks: u32) -> Result<u32, JsError> {
        self.session_mut()?.advance_generation(time_limit_ticks as u64).map(|n| n as u32).map_err(js_error)
    }

    /// Reproduces and installs the next generation:
    /// `{preservedCount, rewards: Float64Array}`.
    #[wasm_bindgen(js_name = nextGeneration)]
    pub fn next_generation(&self) -> Result<JsValue, JsError> {
        let (preserved_count, rewards) = self.session_mut()?.next_generation().map_err(js_error)?;
        let result = js_sys::Object::new();
        js_sys::Reflect::set(&result, &"preservedCount".into(), &JsValue::from(preserved_count as u32)).map_err(|e| js_error(format!("{e:?}")))?;
        js_sys::Reflect::set(&result, &"rewards".into(), &Float64Array::from(&rewards[..])).map_err(|e| js_error(format!("{e:?}")))?;
        Ok(result.into())
    }

    /// `advance` with the ray sensors cast on the GPU (tick-major). Resolves
    /// to the executed ticks. See `GpuRaycaster`.
    #[wasm_bindgen(js_name = advanceWithGpuRays)]
    pub fn advance_with_gpu_rays(&self, raycaster: &gpu::GpuRaycaster, ticks: u32, stop_when_inactive: bool) -> Promise {
        gpu::advance_window(self.shared.clone(), raycaster.inner(), ticks as u64, stop_when_inactive, None)
    }

    /// `advanceGeneration` with the ray sensors cast on the GPU.
    #[wasm_bindgen(js_name = advanceGenerationWithGpuRays)]
    pub fn advance_generation_with_gpu_rays(&self, raycaster: &gpu::GpuRaycaster, time_limit_ticks: u32) -> Promise {
        gpu::advance_window(self.shared.clone(), raycaster.inner(), 0, true, Some(time_limit_ticks as u64))
    }

    /// Ray queries that were also cast on the CPU (`gpuVerifyEvery`).
    #[wasm_bindgen(js_name = gpuRaysChecked)]
    pub fn gpu_rays_checked(&self) -> f64 {
        self.shared.gpu_rays_checked.get() as f64
    }

    /// Checked ray queries whose GPU result differed from the CPU raycast.
    #[wasm_bindgen(js_name = gpuRayMismatches)]
    pub fn gpu_ray_mismatches(&self) -> f64 {
        self.shared.gpu_ray_mismatches.get() as f64
    }

    #[wasm_bindgen(getter)]
    pub fn started(&self) -> bool {
        self.session().started()
    }

    #[wasm_bindgen(getter)]
    pub fn generation(&self) -> u32 {
        self.session().runner.generation as u32
    }

    #[wasm_bindgen(getter)]
    pub fn tick(&self) -> u32 {
        self.session().runner.tick as u32
    }

    #[wasm_bindgen(getter)]
    pub fn population(&self) -> u32 {
        self.session().runner.agents.len() as u32
    }

    /// Whether a WebGPU window is in flight.
    #[wasm_bindgen(getter)]
    pub fn busy(&self) -> bool {
        self.shared.busy.get()
    }

    /// The number of ray sensors per car.
    #[wasm_bindgen(js_name = rayCount)]
    pub fn ray_count(&self) -> u32 {
        self.session().runner.layout.ray_count() as u32
    }

    /// `carStateStride()` values per car, `carStateFields()` order.
    #[wasm_bindgen(js_name = carStates)]
    pub fn car_states(&self) -> Float64Array {
        Float64Array::from(&self.session().car_states()[..])
    }

    #[wasm_bindgen(js_name = carStateStride)]
    pub fn car_state_stride() -> u32 {
        CAR_STATE_STRIDE as u32
    }

    #[wasm_bindgen(js_name = carStateFields)]
    pub fn car_state_fields() -> Vec<String> {
        CAR_STATE_FIELDS.iter().map(|&s| s.to_owned()).collect()
    }

    /// The training metrics of a car (`metricNames()` order; NaN when unset).
    pub fn metrics(&self, index: u32) -> Result<Float64Array, JsError> {
        self.session().metrics(index as usize).map(|m| Float64Array::from(&m[..])).map_err(js_error)
    }

    #[wasm_bindgen(js_name = metricNames)]
    pub fn metric_names() -> Vec<String> {
        Session::metric_names()
    }

    /// The sensor inputs a car would read now (`sensorNames()` order).
    pub fn sensors(&self, index: u32) -> Result<Float64Array, JsError> {
        self.session().sensors(index as usize).map(|s| Float64Array::from(&s[..])).map_err(js_error)
    }

    #[wasm_bindgen(js_name = sensorNames)]
    pub fn sensor_names(&self) -> Vec<String> {
        self.session().sensor_names().to_vec()
    }

    /// `[acceleration, steering, brake, handbrake, boost]` of a car.
    pub fn controls(&self, index: u32) -> Result<Float64Array, JsError> {
        self.session().controls(index as usize).map(|c| Float64Array::from(&c[..])).map_err(js_error)
    }

    #[wasm_bindgen(js_name = outputNames)]
    pub fn output_names(&self) -> Vec<String> {
        self.session().output_names().to_vec()
    }

    /// A car's network as `{"shape", "weights", "biases"}` JSON.
    #[wasm_bindgen(js_name = networkJson)]
    pub fn network_json(&self, index: u32) -> Result<String, JsError> {
        self.session().network_json(index as usize).map_err(js_error)
    }

    /// Replaces a car's network with an export of the same shape.
    #[wasm_bindgen(js_name = setNetworkJson)]
    pub fn set_network_json(&self, index: u32, json: &str) -> Result<(), JsError> {
        self.session_mut()?.set_network_json(index as usize, json).map_err(js_error)
    }

    /// The best lap of the generation as `{index, time}`, or null.
    #[wasm_bindgen(js_name = bestLap)]
    pub fn best_lap(&self) -> Result<JsValue, JsError> {
        match self.session().best_lap() {
            Some((index, time)) => to_js(&serde_json::json!({"index": index, "time": time})),
            None => Ok(JsValue::NULL),
        }
    }

    /// A generation-boundary checkpoint (generation, RNG, networks) as JSON.
    #[wasm_bindgen(js_name = checkpointJson)]
    pub fn checkpoint_json(&self) -> Result<String, JsError> {
        self.session().checkpoint().map(|v| v.to_string()).map_err(js_error)
    }

    /// Installs a checkpoint's networks as its generation and restores its RNG.
    #[wasm_bindgen(js_name = restoreCheckpointJson)]
    pub fn restore_checkpoint_json(&self, json: &str) -> Result<(), JsError> {
        self.session_mut()?.restore_checkpoint(&parse(json, "checkpoint")?).map_err(js_error)
    }

    /// GameManager.OnPause: frozen cars keep their pose.
    #[wasm_bindgen(js_name = setPaused)]
    pub fn set_paused(&self, paused: bool) -> Result<(), JsError> {
        self.session_mut()?.runner.set_paused(paused);
        Ok(())
    }

    /// Wall segments `[x0, y0, x1, y1]...`.
    #[wasm_bindgen(js_name = trackWalls)]
    pub fn track_walls(&self) -> Float32Array {
        Float32Array::from(&self.session().track_walls()[..])
    }

    /// The baked centre path `[x, y]...`.
    #[wasm_bindgen(js_name = trackPath)]
    pub fn track_path(&self) -> Float64Array {
        Float64Array::from(&self.session().track_path()[..])
    }

    /// `[minX, minY, maxX, maxY]` of the walls.
    #[wasm_bindgen(js_name = trackBounds)]
    pub fn track_bounds(&self) -> Float64Array {
        Float64Array::from(&self.session().track_bounds()[..])
    }

    /// `[x, y, rotation]` of the spawn pose.
    pub fn spawn(&self) -> Float64Array {
        let s = self.session().spawn();
        Float64Array::from(&[s.position[0], s.position[1], s.rotation][..])
    }
}
