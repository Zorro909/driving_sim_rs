//! A simulation session for embedding hosts: the WebAssembly library
//! (src/wasm) and examples/wasm_reference.rs share it, so a host runs the
//! same code as the native reference. It wraps `TrainingRunner` with JSON
//! options, seeding, flat state export and generation-boundary checkpoints.

use crate::math::vec2::V2;
use crate::nn::network::{parameter_count, Network};
use crate::track::world::World;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use crate::training::evolution::METRIC_NAMES;
use crate::training::evolution::{Breeding, EvolutionSettings, CROSSOVERS};
use crate::training::pyrandom::PyRandom;
use crate::training::{Lineage, Mode, SensorLayout, TrainingAgent, TrainingRandom, TrainingRunner, CAR_STATE_STRIDE};
#[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
use serde_json::json;
use serde_json::Value;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
mod hip;
#[cfg(not(target_arch = "wasm32"))]
pub use hip::device as hip_device;

/// The spawn pose of every car.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Spawn {
    pub position: [f64; 2],
    pub rotation: f64,
}

/// Where a session's driving windows run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// The CPU runner (rayon).
    #[default]
    Cpu,
    /// The native HIP simulator (libaltd_gpu.so); reproduction stays on the CPU.
    Hip,
}

/// Options of `Session::new`, as the JSON object hosts pass (camelCase keys,
/// every field optional).
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub struct SessionOptions {
    /// Cars per generation; overrides `settings.population`. Default 32.
    pub(crate) population: Option<usize>,
    /// Seed of the Python-compatible training RNG (selection, mutation, Xavier networks).
    pub(crate) seed: i64,
    /// Inference batches per tick divisor (`--batch-count`).
    pub(crate) batch_count: usize,
    pub(crate) stats_phase: u64,
    pub(crate) eliminate_on_wall: bool,
    pub(crate) eliminate_when_idle: bool,
    /// Spawn pose; without it the scene's `reset_position`/`reset_rotation` are used.
    pub spawn: Option<Spawn>,
    /// `"independent"` or `"lockstep"` (`--mode`).
    pub mode: String,
    /// Evolution settings as `EvolutionSettings::from_mcp` reads them.
    pub(crate) settings: Option<Value>,
    /// WebGPU only: every n-th ray query is also cast on the CPU and compared
    /// (0 disables the check).
    pub(crate) gpu_verify_every: u32,
    /// `"cpu"` or `"hip"`. A native session falls back to the CPU when HIP
    /// cannot run it (see `Session::backend_note`).
    pub backend: Backend,
}

impl Default for SessionOptions {
    fn default() -> Self {
        SessionOptions {
            population: None,
            seed: 0,
            batch_count: 1,
            stats_phase: 0,
            eliminate_on_wall: false,
            eliminate_when_idle: false,
            spawn: None,
            mode: "independent".into(),
            settings: None,
            gpu_verify_every: 0,
            backend: Backend::Cpu,
        }
    }
}

impl SessionOptions {
    pub fn from_json(text: &str) -> Result<SessionOptions, String> {
        serde_json::from_str(text).map_err(|e| format!("invalid session options: {e}"))
    }
}

pub struct Session {
    pub runner: TrainingRunner,
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub(crate) options: SessionOptions,
    #[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
    network: Value,
    output_names: Vec<String>,
    boundary: Option<Boundary>,
    #[cfg(not(target_arch = "wasm32"))]
    hip: Option<hip::HipState>,
    /// Why the session runs on another backend than its options asked for.
    backend_note: Option<String>,
}

/// The generation that just began, as a checkpoint saves it. Kept apart from
/// the running cars so saving never touches them.
#[derive(Clone)]
struct Boundary {
    generation: u64,
    tick: u64,
    saved: Saved,
}

#[derive(Clone)]
enum Saved {
    /// After a full restore (format 1 or JSON): every car's parameters and
    /// the generator after them. Saved as format 1.
    Population {
        shape: Vec<usize>,
        rng: TrainingRandom,
        params: Vec<f64>,
    },
    /// After a start or reproduction: what bred the cars. Saved as format 2.
    Lineage(Lineage),
}

/// Per-generation statistics hosts read instead of the full car states.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationSummary {
    /// The first car with the highest distance score.
    pub(crate) best_index: usize,
    pub(crate) best_score: f64,
    /// Cars with at least one completed lap.
    pub(crate) lapped: usize,
    pub(crate) active: usize,
    /// The best lap of the generation: `(car, seconds)`.
    pub(crate) lap_index: Option<usize>,
    pub(crate) lap_time: Option<f64>,
}

/// `checkpoint_bytes` magic: format 1, every car's networks.
const CHECKPOINT_MAGIC: &[u8; 8] = b"ALTDCKP1";
/// Format 2: the parents a generation was bred from (`Lineage`).
const PARENTS_MAGIC: &[u8; 8] = b"ALTDCKP2";

fn strings(value: &Value, what: &str) -> Result<Vec<String>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("missing {what}"))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{what} must be strings"))
        })
        .collect()
}

impl Session {
    /// `scene`, `network` (a game network export with `inputs` and `outputs`,
    /// optionally `weights`/`biases`) and `model` (sensor parameters) as the
    /// CLI loads them.
    pub fn new(scene: &Value, network: &Value, model: &Value, options: SessionOptions) -> Result<Session, String> {
        let spawn = match options.spawn {
            Some(s) => s,
            None => match (
                scene.get("reset_position"),
                scene.get("reset_rotation").and_then(Value::as_f64),
            ) {
                (Some(p), Some(rotation)) => {
                    let p = crate::track::world::vector(p);
                    Spawn {
                        position: [p.x, p.y],
                        rotation,
                    }
                }
                _ => return Err("options.spawn is required: the scene has no reset_position".into()),
            },
        };
        let mode = match options.mode.as_str() {
            "independent" => Mode::Independent,
            "lockstep" => Mode::Lockstep,
            other => return Err(format!("unknown mode {other:?} (independent or lockstep)")),
        };
        if options.batch_count == 0 {
            return Err("batchCount must be positive".into());
        }
        #[cfg(target_arch = "wasm32")]
        if options.backend == Backend::Hip {
            return Err("the HIP backend is only available in the native simulator".into());
        }
        let output_names = strings(&network["outputs"], "network outputs")?;
        strings(&network["inputs"], "network inputs")?;
        let mut settings = options
            .settings
            .as_ref()
            .map_or_else(EvolutionSettings::default, EvolutionSettings::from_mcp);
        if let Some(population) = options.population {
            settings.population = population;
        }
        if settings.population == 0 {
            return Err("population must be positive".into());
        }
        let world = Arc::new(World::from_scene(scene));
        let layout = SensorLayout::from_exports(network, model);
        let mut runner = TrainingRunner::new(
            world,
            V2::new(spawn.position[0], spawn.position[1]),
            spawn.rotation,
            layout,
            &output_names,
            settings,
            PyRandom::new(options.seed),
            options.batch_count,
            options.stats_phase,
            options.eliminate_on_wall,
            options.eliminate_when_idle,
        );
        runner.mode = mode;
        #[cfg(not(target_arch = "wasm32"))]
        let (hip, backend_note) = match options.backend {
            Backend::Cpu => (None, None),
            Backend::Hip => match hip::HipState::new(&runner) {
                Ok(state) => (Some(state), None),
                Err(reason) => (None, Some(reason)),
            },
        };
        #[cfg(target_arch = "wasm32")]
        let backend_note = None;
        Ok(Session {
            runner,
            #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
            options,
            #[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
            network: network.clone(),
            output_names,
            boundary: None,
            #[cfg(not(target_arch = "wasm32"))]
            hip,
            backend_note,
        })
    }

    /// The network export's `shape` (or `summary.shape`), if any.
    #[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
    pub(crate) fn template_shape(&self) -> Option<Vec<usize>> {
        let summary = self.network.get("summary").unwrap_or(&self.network);
        summary
            .get("shape")?
            .as_array()?
            .iter()
            .map(|v| v.as_u64().map(|n| n as usize))
            .collect()
    }

    /// Installs the first generation: mutations of the export's weights when
    /// it has any, else of a Xavier network of its shape.
    #[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
    pub(crate) fn start(&mut self) -> Result<(), String> {
        if self.network.get("weights").is_some() {
            let seed = Network::try_from_game_export(&self.network)?;
            self.check_shape(&seed.shape)?;
            let lineage = self.runner.start_traced(&seed);
            self.record(lineage);
            self.networks_changed();
            Ok(())
        } else {
            let shape = self
                .template_shape()
                .ok_or("the network export has neither weights nor a shape; use start_with_shape")?;
            self.start_with_shape(&shape)
        }
    }

    /// Installs the first generation from a Xavier network of `shape`, drawn
    /// from the session RNG as `train-scratch` does.
    pub fn start_with_shape(&mut self, shape: &[usize]) -> Result<(), String> {
        self.check_shape(shape)?;
        let seed = self.runner.rng.xavier(shape);
        let lineage = self.runner.start_traced(&seed);
        self.record(lineage);
        self.networks_changed();
        Ok(())
    }

    fn check_shape(&self, shape: &[usize]) -> Result<(), String> {
        let inputs = self.runner.layout.sensors.len();
        if shape.len() < 2 || shape.contains(&0) {
            return Err(format!("invalid network shape {shape:?}"));
        }
        if checked_parameter_count(shape)
            .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err("the network shape exceeds the parameter buffer size".into());
        }
        if shape[0] != inputs {
            return Err(format!(
                "the network has {} inputs but the sensor layout has {inputs}",
                shape[0]
            ));
        }
        if *shape.last().unwrap() < self.output_names.len() {
            return Err(format!(
                "the network has {} outputs but the export names {}",
                shape.last().unwrap(),
                self.output_names.len()
            ));
        }
        Ok(())
    }

    pub(crate) fn started(&self) -> bool {
        !self.runner.agents.is_empty()
    }

    fn require_started(&self) -> Result<(), String> {
        if self.started() {
            Ok(())
        } else {
            Err("the session has not started".into())
        }
    }

    pub(crate) fn agent(&self, index: usize) -> Result<&TrainingAgent, String> {
        self.runner
            .agents
            .get(index)
            .ok_or_else(|| format!("car {index} is outside the population of {}", self.runner.agents.len()))
    }

    /// The backend that runs the driving windows.
    pub fn backend(&self) -> Backend {
        #[cfg(not(target_arch = "wasm32"))]
        if self.hip.is_some() {
            return Backend::Hip;
        }
        Backend::Cpu
    }

    /// Why the session runs on the CPU although its options asked for HIP.
    pub fn backend_note(&self) -> Option<&str> {
        self.backend_note.as_deref()
    }

    /// The networks changed without a new generation number, so a GPU
    /// simulator must upload them again.
    fn networks_changed(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(hip) = &mut self.hip {
            hip.invalidate_networks();
        }
    }

    /// The HIP simulator, after its startup comparison with the CPU; `None`
    /// on the CPU backend. A failed comparison moves the session to the CPU.
    #[cfg(not(target_arch = "wasm32"))]
    fn hip(&mut self) -> Option<&mut hip::HipState> {
        let state = self.hip.as_mut()?;
        if state.needs_verify() {
            if let Err(reason) = state.verify(&mut self.runner) {
                self.hip = None;
                self.backend_note = Some(reason);
                return None;
            }
        }
        self.hip.as_mut()
    }

    pub fn advance(&mut self, ticks: u64, stop_when_inactive: bool) -> Result<u64, String> {
        self.require_started()?;
        #[cfg(not(target_arch = "wasm32"))]
        if self.hip().is_some() {
            let hip = self.hip.as_mut().unwrap();
            return hip.advance(&mut self.runner, ticks, stop_when_inactive);
        }
        Ok(self.runner.advance(ticks, stop_when_inactive))
    }

    pub fn advance_generation(&mut self, time_limit_ticks: u64) -> Result<u64, String> {
        self.require_started()?;
        if self.runner.stats_phase != 0 {
            return Err("advance_generation needs statistics phase 0".into());
        }
        #[cfg(not(target_arch = "wasm32"))]
        if self.hip().is_some() {
            let hip = self.hip.as_mut().unwrap();
            return hip.advance_generation(&mut self.runner, time_limit_ticks);
        }
        Ok(self.runner.advance_generation(time_limit_ticks))
    }

    /// Reproduces and installs the next generation; returns the preserved
    /// parent count and every car's reward.
    pub fn next_generation(&mut self) -> Result<(usize, Vec<f64>), String> {
        self.require_started()?;
        let (turnover, lineage) = self.runner.next_generation_traced();
        self.record(lineage);
        Ok((turnover.preserved_count, turnover.rewards))
    }

    /// Change the settings used by the next reproduction without resetting
    /// cars, generation statistics or the training RNG. Population changes
    /// take effect when `next_generation` installs the new population.
    #[cfg(any(test, feature = "server", all(target_arch = "wasm32", feature = "wasm")))]
    pub fn set_evolution_settings(&mut self, text: &str) -> Result<(), String> {
        let settings: EvolutionSettings =
            serde_json::from_str(text).map_err(|e| format!("invalid evolution settings: {e}"))?;
        if settings.population == 0 || settings.selection_size == 0 {
            return Err("population and selection_size must be positive".into());
        }
        if !["best", "tournament", "roulette"].contains(&settings.selection_algorithm.as_str())
            || !["none", "single_point", "uniform"].contains(&settings.crossover.as_str())
            || !["off", "on_selection_size", "on_custom"].contains(&settings.preserve_parents.as_str())
        {
            return Err("unknown selection, crossover or preservation mode".into());
        }
        if !settings.mutation_rate.is_finite()
            || !(0.0..=10.0).contains(&settings.mutation_rate)
            || !settings.weight_decay.is_finite()
            || !(0.0..=1.0).contains(&settings.weight_decay)
        {
            return Err("mutation_rate must be in 0..10 and weight_decay in 0..1".into());
        }
        if settings.rewards.is_empty()
            || settings.rewards.iter().any(|r| {
                crate::training::evolution::metric_index(&r.metric).is_none()
                    || !["default", "average"].contains(&r.kind.as_str())
            })
        {
            return Err("invalid reward metric or type".into());
        }
        self.runner.settings = settings;
        Ok(())
    }

    /// `CAR_STATE_STRIDE` values per car (`training::CAR_STATE_FIELDS`).
    pub fn car_states(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.runner.agents.len() * CAR_STATE_STRIDE);
        self.runner.car_states(&mut out);
        out
    }

    /// The training metrics of car `index` in `METRIC_NAMES` order (NaN when unset).
    pub fn metrics(&self, index: usize) -> Result<Vec<f64>, String> {
        Ok(self
            .agent(index)?
            .stats
            .metrics()
            .iter()
            .map(|m| m.unwrap_or(f64::NAN))
            .collect())
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub(crate) fn metric_names() -> Vec<String> {
        METRIC_NAMES.iter().map(|&s| s.to_owned()).collect()
    }

    /// The sensor inputs car `index` would read now.
    pub fn sensors(&self, index: usize) -> Result<Vec<f64>, String> {
        let agent = self.agent(index)?;
        let mut scratch = crate::physics::car::SensorScratch::default();
        let mut out = Vec::new();
        self.runner
            .layout
            .read_into(&self.runner.world, &agent.car, &mut scratch, &mut out);
        Ok(out)
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub(crate) fn sensor_names(&self) -> &[String] {
        &self.runner.layout.names
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub(crate) fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// `[acceleration, steering, brake, handbrake, boost]` of car `index`.
    pub fn controls(&self, index: usize) -> Result<[f64; 5], String> {
        let c = self.agent(index)?.controls;
        Ok([c.acceleration, c.steering, c.brake, c.handbrake, c.boost])
    }

    /// The network of car `index` as `{"shape", "weights", "biases"}`.
    #[cfg(any(test, feature = "server", all(target_arch = "wasm32", feature = "wasm")))]
    pub(crate) fn network_json(&self, index: usize) -> Result<String, String> {
        Ok(self.agent(index)?.network.to_json().to_string())
    }

    /// Replaces the network of car `index` with a game export of the same shape.
    #[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
    pub(crate) fn set_network_json(&mut self, index: usize, text: &str) -> Result<(), String> {
        let value: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let network = Network::try_from_game_export(&value)?;
        let agent = self
            .runner
            .agents
            .get_mut(index)
            .ok_or_else(|| format!("car {index} is outside the population"))?;
        if network.shape != agent.network.shape {
            return Err(format!(
                "shape {:?} differs from the population's {:?}",
                network.shape, agent.network.shape
            ));
        }
        agent.network = network;
        self.networks_changed();
        Ok(())
    }

    /// The best lap of the current generation: `(car, seconds)`.
    pub(crate) fn best_lap(&self) -> Option<(usize, f64)> {
        self.runner
            .agents
            .iter()
            .enumerate()
            .filter_map(|(i, a)| a.stats.best_lap_time.map(|t| (i, t)))
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
    }

    /// A generation-boundary checkpoint: generation, RNG state and networks,
    /// restorable with `restore_checkpoint` on a session with the same
    /// options and files.
    #[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
    pub(crate) fn checkpoint(&self) -> Result<Value, String> {
        self.require_started()?;
        let shape = &self.runner.agents[0].network.shape;
        Ok(json!({
            "generation": self.runner.generation,
            "tick": self.runner.tick,
            "shape": shape,
            "rng": self.runner.rng.to_json(),
            "networks": self.runner.agents.iter().map(|a| Value::from(a.network.params.clone())).collect::<Vec<_>>(),
        }))
    }

    /// Installs the checkpoint's networks on reset cars as its generation
    /// (`TrainingRunner::resume`) and restores its RNG.
    #[cfg(any(test, feature = "server", all(target_arch = "wasm32", feature = "wasm")))]
    pub(crate) fn restore_checkpoint(&mut self, value: &Value) -> Result<(), String> {
        let shape: Vec<usize> = value["shape"]
            .as_array()
            .ok_or("missing shape")?
            .iter()
            .map(|v| v.as_u64().map(|n| n as usize).ok_or("invalid shape"))
            .collect::<Result<_, _>>()?;
        let generation = value["generation"].as_u64().ok_or("missing generation")?;
        self.check_shape(&shape)?;
        let size = crate::nn::network::parameter_count(&shape);
        let mut params = Vec::new();
        for (i, row) in value["networks"]
            .as_array()
            .ok_or("missing networks")?
            .iter()
            .enumerate()
        {
            let row = row.as_array().ok_or("invalid network")?;
            if row.len() != size {
                return Err(format!(
                    "network {i} has {} parameters, shape {shape:?} needs {size}",
                    row.len()
                ));
            }
            for v in row {
                params.push(v.as_f64().ok_or("invalid parameter")?);
            }
        }
        let rng = (!value["rng"].is_null()).then(|| TrainingRandom::from_json(&value["rng"]));
        self.restore(&shape, generation, params, rng)
    }

    fn restore(
        &mut self,
        shape: &[usize],
        generation: u64,
        params: Vec<f64>,
        rng: Option<TrainingRandom>,
    ) -> Result<(), String> {
        self.check_shape(shape)?;
        let size = crate::nn::network::parameter_count(shape);
        if params.is_empty() {
            return Err("the checkpoint has no networks".into());
        }
        if !params.len().is_multiple_of(size) {
            return Err(format!(
                "the checkpoint has {} parameters, not a multiple of the {size} of shape {shape:?}",
                params.len()
            ));
        }
        let networks: Vec<Network> = params
            .chunks(size)
            .map(|p| Network::from_vector(shape, p.to_vec()))
            .collect();
        if let Some(rng) = rng {
            self.runner.rng = rng;
        }
        self.runner.resume(&networks, generation);
        self.snapshot_population();
        self.networks_changed();
        Ok(())
    }

    /// Records the generation that just began, with every car's parameters,
    /// as the checkpoint boundary (after a full restore).
    fn snapshot_population(&mut self) {
        let agents = &self.runner.agents;
        let Some(first) = agents.first() else { return };
        let mut params = Vec::with_capacity(agents.len() * first.network.params.len());
        for a in agents {
            params.extend_from_slice(&a.network.params);
        }
        let saved = Saved::Population {
            shape: first.network.shape.clone(),
            rng: self.runner.rng.clone(),
            params,
        };
        self.boundary = Some(Boundary {
            generation: self.runner.generation,
            tick: self.runner.tick,
            saved,
        });
    }

    /// Records the reproduction that bred the generation that just began.
    fn record(&mut self, lineage: Lineage) {
        self.boundary = Some(Boundary {
            generation: self.runner.generation,
            tick: self.runner.tick,
            saved: Saved::Lineage(lineage),
        });
    }

    fn restore_lineage(&mut self, generation: u64, lineage: Lineage) -> Result<(), String> {
        lineage.validate()?;
        self.check_shape(&lineage.parents[0].shape)?;
        let (networks, rng) = lineage.rebuild();
        self.runner.rng = rng;
        self.runner.resume_owned(networks, generation);
        self.record(lineage);
        self.networks_changed();
        Ok(())
    }

    /// The generation boundary `checkpoint_bytes` saves: `(generation, tick)`.
    #[cfg(any(test, feature = "server", all(target_arch = "wasm32", feature = "wasm")))]
    pub(crate) fn boundary(&self) -> Option<(u64, u64)> {
        self.boundary.as_ref().map(|b| (b.generation, b.tick))
    }

    /// The boundary of the current generation in binary: format 2 (see
    /// `parent_bytes`) after a start or reproduction, format 1 (see
    /// `population_bytes`) after restoring a full checkpoint.
    pub fn checkpoint_bytes(&self) -> Result<Vec<u8>, String> {
        let b = self.boundary.as_ref().ok_or("the session has not started")?;
        match &b.saved {
            Saved::Population { shape, rng, params } => population_bytes(b.generation, b.tick, shape, rng, params),
            Saved::Lineage(lineage) => parent_bytes(b.generation, b.tick, lineage),
        }
    }

    /// Restores a `checkpoint_bytes` checkpoint of either format.
    pub fn restore_checkpoint_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        match bytes.get(..8) {
            Some(m) if m == CHECKPOINT_MAGIC => {
                let (shape, generation, params, rng) = read_population(bytes)?;
                self.restore(&shape, generation, params, rng)
            }
            Some(m) if m == PARENTS_MAGIC => {
                let (generation, lineage) = read_parents(bytes)?;
                self.restore_lineage(generation, lineage)
            }
            _ => Err("invalid binary checkpoint".into()),
        }
    }

    /// Statistics of the current generation (see `GenerationSummary`).
    pub fn generation_summary(&self) -> GenerationSummary {
        let agents = &self.runner.agents;
        let mut best = 0;
        for (i, a) in agents.iter().enumerate() {
            if a.stats.total_score > agents[best].stats.total_score {
                best = i;
            }
        }
        let lap = self.best_lap();
        GenerationSummary {
            best_index: best,
            best_score: agents.get(best).map_or(f64::NAN, |a| a.stats.total_score),
            lapped: agents.iter().filter(|a| a.stats.score.lap_count >= 1).count(),
            active: self.active_count(),
            lap_index: lap.map(|l| l.0),
            lap_time: lap.map(|l| l.1),
        }
    }

    /// Cars that still drive.
    pub(crate) fn active_count(&self) -> usize {
        self.runner.agents.iter().filter(|a| a.car.active).count()
    }

    /// Wall segments `[x0, y0, x1, y1]...` for drawing.
    #[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
    pub(crate) fn track_walls(&self) -> Vec<f32> {
        self.runner
            .world
            .track
            .walls
            .iter()
            .flat_map(|w| [w.start.x as f32, w.start.y as f32, w.end.x as f32, w.end.y as f32])
            .collect()
    }

    /// The baked centre path `[x, y]...`.
    #[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
    pub(crate) fn track_path(&self) -> Vec<f64> {
        self.runner.world.track.path.iter().flat_map(|p| [p.x, p.y]).collect()
    }

    /// `[min_x, min_y, max_x, max_y]` over the walls, or the path without walls.
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub(crate) fn track_bounds(&self) -> [f64; 4] {
        let track = &self.runner.world.track;
        let points: Vec<V2> = if track.walls.is_empty() {
            track.path.clone()
        } else {
            track.walls.iter().flat_map(|w| [w.start, w.end]).collect()
        };
        points.iter().fold(
            [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
            |b, p| [b[0].min(p.x), b[1].min(p.y), b[2].max(p.x), b[3].max(p.y)],
        )
    }

    #[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
    pub(crate) fn spawn(&self) -> Spawn {
        Spawn {
            position: [self.runner.position.x, self.runner.position.y],
            rotation: self.runner.rotation,
        }
    }

    /// Selects the track the next installed generation drives, spawning at
    /// the scene's reset pose. Call it at a generation boundary, before
    /// `next_generation`; the vehicle must not change.
    #[cfg(any(test, feature = "server", all(target_arch = "wasm32", feature = "wasm")))]
    pub fn replace_track(&mut self, scene: &Value) -> Result<(), String> {
        let position = scene
            .get("reset_position")
            .map(crate::track::world::vector)
            .ok_or("the scene has no reset_position")?;
        let rotation = scene
            .get("reset_rotation")
            .and_then(Value::as_f64)
            .ok_or("the scene has no reset_rotation")?;
        let world = Arc::new(World::from_scene(scene));
        if world.track.native_broadphase != self.runner.world.track.native_broadphase {
            return Err("a replacement track must keep the broadphase mode".into());
        }
        self.runner
            .replace_track(world, V2::new(position.x, position.y), rotation);
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(hip) = &mut self.hip {
            if let Err(reason) = hip.replace_world(&self.runner) {
                self.hip = None;
                self.backend_note = Some(reason);
            }
        }
        Ok(())
    }
}

/// `parameter_count` for a shape from untrusted bytes: `None` on overflow
/// (usize is 32 bits on wasm).
fn checked_parameter_count(shape: &[usize]) -> Option<usize> {
    shape.windows(2).try_fold(0usize, |sum, w| {
        w[0].checked_add(1)?.checked_mul(w[1])?.checked_add(sum)
    })
}

/// Format 1: `ALTDCKP1`, then little-endian u32 generation, tick, population,
/// layer count and RNG JSON length, the u32 shape, the RNG JSON, zero
/// padding to a multiple of 8 bytes and every network's f64 parameters.
fn population_bytes(
    generation: u64,
    tick: u64,
    shape: &[usize],
    rng: &TrainingRandom,
    params: &[f64],
) -> Result<Vec<u8>, String> {
    let rng = rng.to_json().to_string();
    let size = parameter_count(shape);
    let header = [
        generation,
        tick,
        (params.len() / size) as u64,
        shape.len() as u64,
        rng.len() as u64,
    ];
    let mut out = Vec::with_capacity(64 + rng.len() + params.len() * 8);
    out.extend_from_slice(CHECKPOINT_MAGIC);
    for n in header.iter().copied().chain(shape.iter().map(|&n| n as u64)) {
        out.extend_from_slice(
            &u32::try_from(n)
                .map_err(|_| "checkpoint field exceeds u32")?
                .to_le_bytes(),
        );
    }
    out.extend_from_slice(rng.as_bytes());
    out.resize(out.len().next_multiple_of(8), 0);
    for p in params {
        out.extend_from_slice(&p.to_le_bytes());
    }
    Ok(out)
}

type PopulationCheckpoint = (Vec<usize>, u64, Vec<f64>, Option<TrainingRandom>);

/// Reads format 1: `(shape, generation, params, rng)`.
fn read_population(bytes: &[u8]) -> Result<PopulationCheckpoint, String> {
    let invalid = || "invalid binary checkpoint".to_string();
    if bytes.len() < 28 || &bytes[..8] != CHECKPOINT_MAGIC {
        return Err(invalid());
    }
    let word = |i: usize| -> Result<usize, String> {
        let at = i.checked_mul(4).and_then(|n| n.checked_add(8)).ok_or_else(invalid)?;
        bytes
            .get(at..at.checked_add(4).ok_or_else(invalid)?)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()) as usize)
            .ok_or_else(invalid)
    };
    let (generation, population, layers, rng_len) = (word(0)? as u64, word(2)?, word(3)?, word(4)?);
    let shape: Vec<usize> = (0..layers).map(|i| word(5 + i)).collect::<Result<_, _>>()?;
    let rng_at = layers
        .checked_add(5)
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| n.checked_add(8))
        .ok_or_else(invalid)?;
    let rng_end = rng_at.checked_add(rng_len).ok_or_else(invalid)?;
    let rng_text = bytes.get(rng_at..rng_end).ok_or_else(invalid)?;
    let rng: Value = serde_json::from_slice(rng_text).map_err(|e| format!("invalid checkpoint RNG: {e}"))?;
    let params_at = rng_end.next_multiple_of(8);
    let size = if shape.len() >= 2 {
        checked_parameter_count(&shape).ok_or_else(invalid)?
    } else {
        0
    };
    let data = bytes.get(params_at..).ok_or_else(invalid)?;
    let expected = population.checked_mul(size).and_then(|n| n.checked_mul(8));
    if expected != Some(data.len()) {
        return Err(format!(
            "the checkpoint holds {} bytes of parameters, {population} networks of shape {shape:?} need {}",
            data.len(),
            expected.map_or("more".to_string(), |n| n.to_string())
        ));
    }
    let params = data
        .chunks_exact(8)
        .map(|p| f64::from_le_bytes(p.try_into().unwrap()))
        .collect();
    Ok((
        shape,
        generation,
        params,
        (!rng.is_null()).then(|| TrainingRandom::from_json(&rng)),
    ))
}

/// Format 2, `ALTDCKP2`, little-endian: u32 words generation, tick,
/// population, layer count L, RNG JSON length R, parent count P, selected
/// count S, preserved count K, crossover (index in `CROSSOVERS`) and flags
/// (bit 0: adaptive mutation); the u32 shape, selected and preserved parent
/// indices; the RNG JSON; zero padding to a multiple of 8 bytes; f64
/// mutation rate and weight decay; then the P parents' f64 parameters.
/// Words 0 to 4 sit where format 1 has them.
fn parent_bytes(generation: u64, tick: u64, l: &Lineage) -> Result<Vec<u8>, String> {
    let rng = l.rng.to_json().to_string();
    let shape = &l.parents[0].shape;
    let b = &l.breeding;
    let crossover = CROSSOVERS
        .iter()
        .position(|&c| c == b.crossover)
        .ok_or("unknown crossover")? as u64;
    let header = [
        generation,
        tick,
        b.population as u64,
        shape.len() as u64,
        rng.len() as u64,
        l.parents.len() as u64,
        l.selected.len() as u64,
        l.preserved.len() as u64,
        crossover,
        b.adaptive_mutation as u64,
    ];
    let size = parameter_count(shape);
    let mut out = Vec::with_capacity(
        64 + 4 * (shape.len() + l.selected.len() + l.preserved.len()) + rng.len() + 16 + l.parents.len() * size * 8,
    );
    out.extend_from_slice(PARENTS_MAGIC);
    let words = header
        .into_iter()
        .chain(shape.iter().map(|&n| n as u64))
        .chain(l.selected.iter().chain(&l.preserved).map(|&i| i as u64));
    for n in words {
        out.extend_from_slice(
            &u32::try_from(n)
                .map_err(|_| "checkpoint field exceeds u32")?
                .to_le_bytes(),
        );
    }
    out.extend_from_slice(rng.as_bytes());
    out.resize(out.len().next_multiple_of(8), 0);
    out.extend_from_slice(&b.mutation_rate.to_le_bytes());
    out.extend_from_slice(&b.weight_decay.to_le_bytes());
    for p in l.parents.iter().flat_map(|n| &n.params) {
        out.extend_from_slice(&p.to_le_bytes());
    }
    Ok(out)
}

/// Reads format 2 (see `parent_bytes`) and validates it: `(generation, lineage)`.
fn read_parents(bytes: &[u8]) -> Result<(u64, Lineage), String> {
    let invalid = || "invalid binary checkpoint".to_string();
    let word = |i: usize| -> Result<usize, String> {
        let at = i.checked_mul(4).and_then(|n| n.checked_add(8)).ok_or_else(invalid)?;
        bytes
            .get(at..at.checked_add(4).ok_or_else(invalid)?)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()) as usize)
            .ok_or_else(invalid)
    };
    let (generation, population, layers, rng_len) = (word(0)? as u64, word(2)?, word(3)?, word(4)?);
    let (parents, selected, preserved, crossover, flags) = (word(5)?, word(6)?, word(7)?, word(8)?, word(9)?);
    let first_index = 10usize.checked_add(layers).ok_or_else(invalid)?;
    let rng_at = first_index
        .checked_add(selected)
        .and_then(|n| n.checked_add(preserved))
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| n.checked_add(8))
        .ok_or_else(invalid)?;
    // Bound every count by the bytes present before allocating for it.
    let rng_end = rng_at
        .checked_add(rng_len)
        .filter(|&end| end <= bytes.len())
        .ok_or_else(invalid)?;
    let shape: Vec<usize> = (0..layers).map(|i| word(10 + i)).collect::<Result<_, _>>()?;
    if shape.len() < 2 || shape.contains(&0) {
        return Err(invalid());
    }
    let index = |i: usize| word(i).map(|n| n as u32);
    let selected_ix = (0..selected)
        .map(|i| index(first_index + i))
        .collect::<Result<Vec<_>, _>>()?;
    let preserved_ix = (0..preserved)
        .map(|i| index(first_index + selected + i))
        .collect::<Result<Vec<_>, _>>()?;
    let rng: Value =
        serde_json::from_slice(&bytes[rng_at..rng_end]).map_err(|e| format!("invalid checkpoint RNG: {e}"))?;
    if rng.is_null() {
        return Err("the checkpoint has no random state".into());
    }
    let rates_at = rng_end.next_multiple_of(8);
    let float = |at: usize| {
        bytes
            .get(at..at.checked_add(8).ok_or_else(invalid)?)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .ok_or_else(invalid)
    };
    let (mutation_rate, weight_decay) = (float(rates_at)?, float(rates_at + 8)?);
    let size = checked_parameter_count(&shape).ok_or_else(invalid)?;
    let stride = size.checked_mul(8).ok_or_else(invalid)?;
    let data = bytes.get(rates_at + 16..).ok_or_else(invalid)?;
    if parents.checked_mul(stride) != Some(data.len()) {
        return Err(format!(
            "the checkpoint holds {} bytes of parameters, {parents} networks of shape {shape:?} need more or fewer",
            data.len()
        ));
    }
    let crossover = CROSSOVERS
        .get(crossover)
        .ok_or("the checkpoint has an unknown crossover")?;
    if flags > 1 {
        return Err(invalid());
    }
    let networks = data
        .chunks_exact(stride)
        .map(|chunk| {
            Network::from_vector(
                &shape,
                chunk
                    .chunks_exact(8)
                    .map(|p| f64::from_le_bytes(p.try_into().unwrap()))
                    .collect(),
            )
        })
        .collect();
    let lineage = Lineage {
        parents: networks,
        selected: selected_ix,
        preserved: preserved_ix,
        breeding: Breeding {
            population,
            crossover: (*crossover).into(),
            mutation_rate,
            adaptive_mutation: flags == 1,
            weight_decay,
        },
        rng: TrainingRandom::from_json(&rng),
    };
    lineage.validate()?;
    Ok((generation, lineage))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(path: &str) -> Value {
        let root = env!("CARGO_MANIFEST_DIR");
        serde_json::from_str(&std::fs::read_to_string(format!("{root}/{path}")).unwrap()).unwrap()
    }

    fn generated_scene() -> Value {
        let settings = serde_json::from_value(load("assets/random_track_settings.json")).unwrap();
        crate::track::training_tracks::training_scene_at(
            &load("assets/scenes/formula_template.json"),
            &settings,
            None,
            1729,
            0,
            0,
        )
        .unwrap()
        .1
    }

    fn generated_session(options: &str) -> Session {
        Session::new(
            &generated_scene(),
            &load("assets/networks/formula.json"),
            &load("assets/models/formula.json"),
            SessionOptions::from_json(options).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn options_parse_with_defaults_and_reject_unknown_keys() {
        let o =
            SessionOptions::from_json(r#"{"population": 8, "seed": 3, "eliminateOnWall": true, "mode": "lockstep"}"#)
                .unwrap();
        assert_eq!(
            (
                o.population,
                o.seed,
                o.batch_count,
                o.eliminate_on_wall,
                o.eliminate_when_idle,
                o.mode.as_str()
            ),
            (Some(8), 3, 1, true, false, "lockstep")
        );
        assert!(SessionOptions::from_json(r#"{"populaton": 8}"#).is_err());
        assert!(SessionOptions::from_json("{}").unwrap().population.is_none());
    }

    #[test]
    fn oversized_shapes_fail_before_allocating_parameters() {
        let mut session = generated_session(r#"{"population":2}"#);
        assert!(session.start_with_shape(&[20, usize::MAX, 5]).is_err());
        assert!(!session.started());
        session.start_with_shape(&[20, 8, 5]).unwrap();
        assert!(session.started());
    }

    #[test]
    fn evolution_updates_preserve_statistics_and_rng_until_turnover() {
        let mut s = generated_session(r#"{"population":6,"seed":5,"eliminateOnWall":true}"#);
        s.start_with_shape(&[20, 8, 5]).unwrap();
        s.advance_generation(120).unwrap();
        let states = s.car_states();
        let rng = s.runner.rng.to_json();
        let generation = s.runner.generation;
        s.set_evolution_settings(r#"{"population":9,"mutation_rate":0.1,"weight_decay":0}"#)
            .unwrap();
        assert_eq!(s.car_states(), states);
        assert_eq!(s.runner.rng.to_json(), rng);
        assert_eq!(s.runner.generation, generation);
        assert!(s.set_evolution_settings(r#"{"mutation_rate":-1}"#).is_err());
        assert!(s
            .set_evolution_settings(r#"{"selection_algorithm":"unknown"}"#)
            .is_err());
        assert!(s
            .set_evolution_settings(r#"{"rewards":[{"metric":"unknown","weight":100,"type":"default"}]}"#)
            .is_err());
        assert!(s.set_evolution_settings(r#"{"population":0}"#).is_err());
        assert!(s.set_evolution_settings(r#"{"mutaton_rate":0.1}"#).is_err());
        assert_eq!(s.runner.settings.mutation_rate, 0.1);
        let (_, rewards) = s.next_generation().unwrap();
        assert_eq!(rewards.len(), 6);
        assert_eq!(s.runner.agents.len(), 9);
        assert_eq!(s.runner.generation, generation + 1);
    }

    #[test]
    fn session_runs_generations_and_checkpoints_round_trip() {
        let mut s = generated_session(
            r#"{"population": 6, "seed": 5, "batchCount": 2, "eliminateOnWall": true, "eliminateWhenIdle": true,
            "settings": {"selection_size": 3, "preserve_parents_size": 2, "mutation_rate": 0.3, "weight_decay": 0.0}}"#,
        );
        assert!(s.advance(1, false).is_err());
        assert!(s.start_with_shape(&[19, 5]).is_err(), "input count mismatch");
        s.start_with_shape(&[20, 8, 5]).unwrap();
        assert_eq!(s.car_states().len(), 6 * CAR_STATE_STRIDE);
        assert_eq!(s.sensors(0).unwrap().len(), 20);
        assert_eq!(s.metrics(0).unwrap().len(), 14);
        assert_eq!(s.advance(120, false).unwrap(), 120);
        let (preserved, rewards) = s.next_generation().unwrap();
        // `on_selection_size` preserves the three selected parents.
        assert_eq!((preserved, rewards.len()), (3, 6));
        let checkpoint = s.checkpoint().unwrap();
        s.advance_generation(300).unwrap();
        let states = s.car_states();
        assert_eq!(s.track_walls().len() % 4, 0);
        assert!(!s.track_path().is_empty());
        // The same generation replays from the checkpoint in a fresh session.
        let mut t = generated_session(
            r#"{"population": 6, "seed": 5, "batchCount": 2, "eliminateOnWall": true, "eliminateWhenIdle": true,
            "settings": {"selection_size": 3, "preserve_parents_size": 2, "mutation_rate": 0.3, "weight_decay": 0.0}}"#,
        );
        t.restore_checkpoint(&checkpoint).unwrap();
        assert_eq!(t.runner.generation, 1);
        t.advance_generation(300).unwrap();
        assert!(states
            .iter()
            .zip(&t.car_states())
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        assert_eq!(s.network_json(0).unwrap(), t.network_json(0).unwrap());
        s.set_network_json(1, &t.network_json(0).unwrap()).unwrap();
        assert_eq!(s.network_json(1).unwrap(), t.network_json(0).unwrap());
        // A malformed export (one weight row) and a well-formed one of another shape are both rejected.
        assert!(s
            .set_network_json(
                1,
                r#"{"shape":[20,5],"weights":[[[0.0,0,0,0,0]]],"biases":[[0,0,0,0,0]]}"#
            )
            .is_err());
        assert!(s
            .set_network_json(
                1,
                &Network::xavier(&[20, 5], &mut PyRandom::new(1)).to_json().to_string()
            )
            .is_err());
        assert_eq!(s.network_json(1).unwrap(), t.network_json(0).unwrap());
    }

    #[test]
    fn binary_checkpoints_save_the_generation_boundary() {
        let options = r#"{"population": 6, "seed": 5, "eliminateOnWall": true, "eliminateWhenIdle": true,
            "settings": {"selection_size": 3, "mutation_rate": 0.3, "weight_decay": 0.0}}"#;
        let mut s = generated_session(options);
        assert!(s.checkpoint_bytes().is_err());
        s.start_with_shape(&[20, 8, 5]).unwrap();
        s.advance_generation(240).unwrap();
        let summary = s.generation_summary();
        let states = s.car_states();
        let best = (0..6).fold(0, |b, i| {
            if states[i * CAR_STATE_STRIDE + 7] > states[b * CAR_STATE_STRIDE + 7] {
                i
            } else {
                b
            }
        });
        assert_eq!(
            (summary.best_index, summary.best_score),
            (best, states[best * CAR_STATE_STRIDE + 7])
        );
        assert_eq!(
            summary.active,
            (0..6).filter(|i| states[i * CAR_STATE_STRIDE + 6] == 1.0).count()
        );
        assert_eq!(summary.active, s.active_count());
        s.next_generation().unwrap();
        // The boundary is the checkpoint of the generation that just began,
        // however far that generation has since advanced.
        let json = s.checkpoint().unwrap();
        s.advance(120, false).unwrap();
        let bytes = s.checkpoint_bytes().unwrap();
        assert_eq!(&bytes[..8], b"ALTDCKP2");
        assert_eq!(s.boundary(), Some((1, 0)));
        let mut t = generated_session(options);
        t.restore_checkpoint_bytes(&bytes).unwrap();
        assert_eq!(t.checkpoint().unwrap(), json);
        assert_eq!(t.checkpoint_bytes().unwrap(), bytes);
        let mut u = generated_session(options);
        u.restore_checkpoint(&json).unwrap();
        let full = u.checkpoint_bytes().unwrap();
        assert_eq!(&full[..8], b"ALTDCKP1");
        let mut w = generated_session(options);
        w.restore_checkpoint_bytes(&full).unwrap();
        assert_eq!(w.checkpoint().unwrap(), json);
        assert_eq!(w.checkpoint_bytes().unwrap(), full);
        assert!(t.restore_checkpoint_bytes(&bytes[..bytes.len() - 8]).is_err());
        assert!(t.restore_checkpoint_bytes(b"ALTDCKP0").is_err());
    }

    fn state_bits(s: &Session) -> Vec<u64> {
        s.car_states().iter().map(|v| v.to_bits()).collect()
    }

    const PARENT_OPTIONS: [&str; 3] = [
        r#"{"population": 12, "seed": 5, "eliminateOnWall": true, "settings": {"selection_algorithm": "tournament", "selection_size": 4, "crossover": "uniform", "mutation_rate": 0.3, "adaptive_mutation": true, "weight_decay": 0.001, "preserve_parents": "on_custom", "preserve_parents_size": 2}}"#,
        r#"{"population": 12, "seed": 6, "eliminateOnWall": true, "settings": {"selection_algorithm": "roulette", "selection_size": 3, "crossover": "single_point", "mutation_rate": 0.2, "preserve_parents": "off"}}"#,
        r#"{"population": 12, "seed": 7, "eliminateOnWall": true, "settings": {"selection_size": 3, "mutation_rate": 0.2}}"#,
    ];

    /// A parent checkpoint holds the parents, not the generation, yet it
    /// restores every network, the generator and later generations exactly,
    /// even after the session's settings have moved on.
    #[test]
    fn parent_checkpoints_rebuild_the_generation_exactly() {
        for options in PARENT_OPTIONS {
            let mut s = generated_session(options);
            s.start_with_shape(&[20, 8, 5]).unwrap();
            let first = s.checkpoint_bytes().unwrap();
            assert_eq!(&first[..8], b"ALTDCKP2");
            let mut t = generated_session(options);
            t.restore_checkpoint_bytes(&first).unwrap();
            assert_eq!(t.checkpoint().unwrap(), s.checkpoint().unwrap(), "generation 0");
            assert_eq!(t.checkpoint_bytes().unwrap(), first);

            s.advance_generation(120).unwrap();
            s.next_generation().unwrap();
            s.advance_generation(120).unwrap();
            // A schedule or stage change: the next reproduction breeds 9 cars
            // at another rate, which a session built with `options` lacks.
            s.set_evolution_settings(r#"{"population": 9, "selection_algorithm": "tournament", "selection_size": 2, "crossover": "uniform", "mutation_rate": 0.05, "weight_decay": 0.01, "preserve_parents": "on_custom", "preserve_parents_size": 1}"#).unwrap();
            s.next_generation().unwrap();
            let bytes = s.checkpoint_bytes().unwrap();
            let json = s.checkpoint().unwrap();
            let mut u = generated_session(options);
            u.restore_checkpoint_bytes(&bytes).unwrap();
            assert_eq!(u.runner.agents.len(), 9);
            assert_eq!(u.runner.generation, 2);
            assert_eq!(u.checkpoint().unwrap(), json, "every network and the generator");
            assert_eq!(u.checkpoint_bytes().unwrap(), bytes);
            u.set_evolution_settings(r#"{"population": 9, "selection_algorithm": "tournament", "selection_size": 2, "crossover": "uniform", "mutation_rate": 0.05, "weight_decay": 0.01, "preserve_parents": "on_custom", "preserve_parents_size": 1}"#).unwrap();
            for _ in 0..2 {
                s.advance_generation(120).unwrap();
                u.advance_generation(120).unwrap();
                assert_eq!(state_bits(&u), state_bits(&s));
                assert_eq!(u.next_generation().unwrap(), s.next_generation().unwrap());
                assert_eq!(u.checkpoint_bytes().unwrap(), s.checkpoint_bytes().unwrap());
            }
        }
    }

    /// The checkpoint grows with the parents, not the cars.
    #[test]
    fn parent_checkpoints_scale_with_the_selection() {
        let mut s = generated_session(
            r#"{"population": 60, "seed": 5, "settings": {"selection_size": 3, "mutation_rate": 0.2}}"#,
        );
        s.start_with_shape(&[20, 8, 5]).unwrap();
        s.advance_generation(60).unwrap();
        s.next_generation().unwrap();
        let bytes = s.checkpoint_bytes().unwrap();
        let word = |i: usize| u32::from_le_bytes(bytes[8 + i * 4..12 + i * 4].try_into().unwrap()) as usize;
        let params = crate::nn::network::parameter_count(&[20, 8, 5]);
        assert_eq!((word(2), word(5)), (60, 3), "60 cars bred from the 3 best, preserved");
        // The generator's JSON (about 7 KB) is the fixed part, so compare with
        // a quarter of the 60 cars' parameters rather than a few networks.
        assert!(bytes.len() < 60 * params * 8 / 4, "{} bytes", bytes.len());
        assert!(bytes.len() >= 3 * params * 8, "the 3 parents are all there");
    }

    /// Format 1 checkpoints of earlier versions still restore.
    #[test]
    fn full_checkpoints_still_restore() {
        let options = PARENT_OPTIONS[0];
        let mut s = generated_session(options);
        s.start_with_shape(&[20, 8, 5]).unwrap();
        s.advance_generation(120).unwrap();
        s.next_generation().unwrap();
        let mut v = generated_session(options);
        v.restore_checkpoint(&s.checkpoint().unwrap()).unwrap();
        let full = v.checkpoint_bytes().unwrap();
        assert_eq!(&full[..8], b"ALTDCKP1");
        let mut w = generated_session(options);
        w.restore_checkpoint_bytes(&full).unwrap();
        assert_eq!(w.checkpoint().unwrap(), s.checkpoint().unwrap());
        w.advance_generation(120).unwrap();
        s.advance_generation(120).unwrap();
        w.next_generation().unwrap();
        s.next_generation().unwrap();
        assert_eq!(
            w.checkpoint_bytes().unwrap(),
            s.checkpoint_bytes().unwrap(),
            "the next save is format 2"
        );
    }

    /// Corrupt parent checkpoints are errors, never panics (a panic traps WASM).
    #[test]
    fn malformed_parent_checkpoints_are_rejected() {
        let options = PARENT_OPTIONS[0];
        let mut s = generated_session(options);
        s.start_with_shape(&[20, 8, 5]).unwrap();
        s.advance_generation(60).unwrap();
        s.next_generation().unwrap();
        let bytes = s.checkpoint_bytes().unwrap();
        let word = |b: &[u8], i: usize| u32::from_le_bytes(b[8 + i * 4..12 + i * 4].try_into().unwrap());
        let set = |i: usize, value: u32| {
            let mut b = bytes.clone();
            b[8 + i * 4..12 + i * 4].copy_from_slice(&value.to_le_bytes());
            b
        };
        let (layers, parents, selected, preserved) = (
            word(&bytes, 3) as usize,
            word(&bytes, 5),
            word(&bytes, 6),
            word(&bytes, 7),
        );
        let first_index = 10 + layers;
        let rates_at = {
            let rng_at = 8 + (first_index + (selected + preserved) as usize) * 4;
            (rng_at + word(&bytes, 4) as usize).next_multiple_of(8)
        };
        let mut nan_rate = bytes.clone();
        nan_rate[rates_at..rates_at + 8].copy_from_slice(&f64::NAN.to_le_bytes());
        let mut no_selection = set(6, 0);
        // Drop the selected indices so the rest of the layout stays valid.
        no_selection.drain(8 + first_index * 4..8 + (first_index + selected as usize) * 4);
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("truncated", bytes[..bytes.len() - 8].to_vec()),
            ("header only", bytes[..40].to_vec()),
            ("index past the parents", set(first_index, parents)),
            ("huge selection", set(6, u32::MAX)),
            ("huge parent count", set(5, u32::MAX)),
            ("no cars", set(2, 0)),
            ("more kept than cars", set(2, preserved - 1)),
            ("unknown crossover", set(8, 3)),
            ("unknown flags", set(9, 2)),
            ("NaN mutation rate", nan_rate),
            ("selects nothing", no_selection),
            ("one-layer shape", set(3, 1)),
        ];
        for (what, case) in cases {
            let mut t = generated_session(options);
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| t
                    .restore_checkpoint_bytes(&case)
                    .is_err()))
                .unwrap_or(false),
                "{what}"
            );
        }
    }

    /// HIP tests share the process's one device claim, so they run one at a time.
    #[cfg(not(target_arch = "wasm32"))]
    static HIP_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The HIP test lock, or `None` (the test is skipped) without a device.
    #[cfg(not(target_arch = "wasm32"))]
    fn hip_or_skip() -> Option<std::sync::MutexGuard<'static, ()>> {
        let guard = HIP_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        match hip_device() {
            Ok(_) => Some(guard),
            Err(e) => {
                eprintln!("skipping the HIP test: {e}");
                None
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn with_backend(options: &str, backend: &str) -> String {
        let mut value: Value = serde_json::from_str(options).unwrap();
        value["backend"] = Value::from(backend);
        value.to_string()
    }

    #[test]
    fn backend_option_parses_and_defaults_to_cpu() {
        assert_eq!(SessionOptions::from_json("{}").unwrap().backend, Backend::Cpu);
        assert_eq!(
            SessionOptions::from_json(r#"{"backend":"hip"}"#).unwrap().backend,
            Backend::Hip
        );
        assert!(SessionOptions::from_json(r#"{"backend":"webgpu"}"#).is_err());
        let s = generated_session(r#"{"population": 2}"#);
        assert_eq!((s.backend(), s.backend_note()), (Backend::Cpu, None));
    }

    /// A HIP session drives exactly like a CPU session through windows,
    /// whole generations and turnovers, and their checkpoints are
    /// interchangeable.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn hip_sessions_match_cpu_sessions() {
        let Some(_hip) = hip_or_skip() else { return };
        let options = PARENT_OPTIONS[0];
        let mut cpu = generated_session(options);
        let mut gpu = generated_session(&with_backend(options, "hip"));
        assert_eq!((gpu.backend(), gpu.backend_note()), (Backend::Hip, None));
        cpu.start_with_shape(&[20, 8, 5]).unwrap();
        gpu.start_with_shape(&[20, 8, 5]).unwrap();
        for (ticks, stop) in [(1, false), (37, false), (90, true)] {
            assert_eq!(cpu.advance(ticks, stop).unwrap(), gpu.advance(ticks, stop).unwrap());
            assert_eq!(state_bits(&gpu), state_bits(&cpu), "window of {ticks}");
        }
        assert_eq!(gpu.backend(), Backend::Hip, "{:?}", gpu.backend_note());
        assert_eq!(
            cpu.advance_generation(600).unwrap(),
            gpu.advance_generation(600).unwrap()
        );
        assert_eq!(state_bits(&gpu), state_bits(&cpu));
        for generation in 1..=2 {
            assert_eq!(gpu.next_generation().unwrap(), cpu.next_generation().unwrap());
            assert_eq!(gpu.checkpoint_bytes().unwrap(), cpu.checkpoint_bytes().unwrap());
            assert_eq!(
                cpu.advance_generation(600).unwrap(),
                gpu.advance_generation(600).unwrap()
            );
            assert_eq!(state_bits(&gpu), state_bits(&cpu), "generation {generation}");
            assert_eq!(gpu.generation_summary(), cpu.generation_summary());
        }
        // Checkpoints move between the backends.
        let saved = gpu.checkpoint_bytes().unwrap();
        drop(gpu);
        let mut gpu = generated_session(&with_backend(options, "hip"));
        let mut from_gpu = generated_session(options);
        gpu.restore_checkpoint_bytes(&cpu.checkpoint_bytes().unwrap()).unwrap();
        from_gpu.restore_checkpoint_bytes(&saved).unwrap();
        for s in [&mut gpu, &mut from_gpu] {
            assert_eq!(s.advance_generation(600).unwrap(), cpu.runner.tick);
            assert_eq!(state_bits(s), state_bits(&cpu));
        }
        assert_eq!(gpu.backend(), Backend::Hip, "{:?}", gpu.backend_note());
        // A stage change to another generated track.
        let settings = serde_json::from_value(load("assets/random_track_settings.json")).unwrap();
        let (_, next) = crate::track::training_tracks::training_scene_at(
            &load("assets/scenes/formula_template.json"),
            &settings,
            None,
            4242,
            0,
            0,
        )
        .unwrap();
        for s in [&mut cpu, &mut gpu] {
            s.replace_track(&next).unwrap();
            s.next_generation().unwrap();
            s.advance_generation(600).unwrap();
        }
        assert_eq!(state_bits(&gpu), state_bits(&cpu), "the replaced track");
        assert_eq!(gpu.backend(), Backend::Hip, "{:?}", gpu.backend_note());

        // An already verified session can restart with a different network
        // shape and restore an earlier generation without stale GPU networks.
        for s in [&mut cpu, &mut gpu] {
            s.start_with_shape(&[20, 16, 5]).unwrap();
        }
        assert!(gpu.hip.as_ref().unwrap().needs_verify());
        for s in [&mut cpu, &mut gpu] {
            s.advance(30, false).unwrap();
        }
        assert_eq!(state_bits(&gpu), state_bits(&cpu), "the restarted session");
        assert!(!gpu.hip.as_ref().unwrap().needs_verify());
        let checkpoint = cpu.checkpoint_bytes().unwrap();
        for s in [&mut cpu, &mut gpu] {
            s.restore_checkpoint_bytes(&checkpoint).unwrap();
        }
        assert!(gpu.hip.as_ref().unwrap().needs_verify());
        for s in [&mut cpu, &mut gpu] {
            s.advance(30, false).unwrap();
            s.set_evolution_settings(r#"{"population":24,"selection_size":3}"#)
                .unwrap();
            s.next_generation().unwrap();
            s.advance_generation(120).unwrap();
        }
        assert_eq!(state_bits(&gpu), state_bits(&cpu), "the larger population");
        assert_eq!(gpu.checkpoint_bytes().unwrap(), cpu.checkpoint_bytes().unwrap());
        assert_eq!(gpu.backend(), Backend::Hip, "{:?}", gpu.backend_note());
    }

    /// Scenes HIP cannot run, and a second concurrent HIP session, fall back
    /// to the CPU with a reason; the device is free again once a HIP session drops.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn hip_falls_back_to_the_cpu_with_a_reason() {
        let Some(_hip) = hip_or_skip() else { return };
        let options = with_backend(r#"{"population": 4, "seed": 3}"#, "hip");
        let session = |scene: &Value| {
            Session::new(
                scene,
                &load("assets/networks/formula.json"),
                &load("assets/models/formula.json"),
                SessionOptions::from_json(&options).unwrap(),
            )
            .unwrap()
        };
        let mut broadphase = generated_scene();
        broadphase["track"]["native_broadphase"] = Value::Bool(true);
        let mut no_curve = generated_scene();
        no_curve["track"].as_object_mut().unwrap().remove("curve");
        for (scene, reason) in [(&broadphase, "native broadphase"), (&no_curve, "track.curve")] {
            let s = session(scene);
            assert_eq!(s.backend(), Backend::Cpu);
            assert!(s.backend_note().unwrap().contains(reason), "{:?}", s.backend_note());
        }
        let mut first = session(&generated_scene());
        assert_eq!(first.backend(), Backend::Hip);
        let mut second = session(&generated_scene());
        assert_eq!(
            (second.backend(), second.backend_note()),
            (Backend::Cpu, Some("another session is using the GPU"))
        );
        // The fallback still trains.
        first.start_with_shape(&[20, 8, 5]).unwrap();
        second.start_with_shape(&[20, 8, 5]).unwrap();
        first.advance(30, false).unwrap();
        second.advance(30, false).unwrap();
        assert_eq!(state_bits(&first), state_bits(&second));
        drop(first);
        assert_eq!(session(&generated_scene()).backend(), Backend::Hip);
    }

    #[test]
    fn export_weights_seed_the_first_generation() {
        let scene = generated_scene();
        let spawn = serde_json::json!({"position":scene["reset_position"],"rotation":scene["reset_rotation"]});
        let options = SessionOptions {
            population: Some(4),
            spawn: Some(Spawn {
                position: [
                    spawn["position"][0].as_f64().unwrap(),
                    spawn["position"][1].as_f64().unwrap(),
                ],
                rotation: spawn["rotation"].as_f64().unwrap(),
            }),
            ..Default::default()
        };
        let network = Network::xavier(&[20, 7, 5], &mut PyRandom::new(1));
        let mut export = network.to_json();
        for (key, value) in load("assets/networks/formula.json").as_object().unwrap() {
            export[key] = value.clone();
        }
        let mut s = Session::new(&scene, &export, &load("assets/models/formula.json"), options).unwrap();
        assert_eq!(s.template_shape(), Some(vec![20, 7, 5]));
        s.start().unwrap();
        assert_eq!(s.runner.agents[0].network.shape, vec![20, 7, 5]);
        assert_eq!(s.spawn().rotation, spawn["rotation"].as_f64().unwrap());
    }
}
