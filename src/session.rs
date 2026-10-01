//! A simulation session for embedding hosts: the WebAssembly library
//! (src/wasm) and examples/wasm_reference.rs share it, so a host runs the
//! same code as the native reference. It wraps `TrainingRunner` with JSON
//! options, seeding, flat state export and generation-boundary checkpoints.

use crate::evolution::{EvolutionSettings, METRIC_NAMES};
use crate::network::Network;
use crate::pyrandom::PyRandom;
use crate::training::{Mode, SensorLayout, TrainingAgent, TrainingRandom, TrainingRunner, CAR_STATE_STRIDE};
use crate::vec2::V2;
use crate::world::World;
use serde_json::{json, Value};
use std::sync::Arc;

/// The spawn pose of every car.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Spawn {
    pub position: [f64; 2],
    pub rotation: f64,
}

/// Options of `Session::new`, as the JSON object hosts pass (camelCase keys,
/// every field optional).
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub struct SessionOptions {
    /// Cars per generation; overrides `settings.population`. Default 32.
    pub population: Option<usize>,
    /// Seed of the Python-compatible training RNG (selection, mutation, Xavier networks).
    pub seed: i64,
    /// Inference batches per tick divisor (`--batch-count`).
    pub batch_count: usize,
    pub stats_phase: u64,
    pub eliminate_on_wall: bool,
    pub eliminate_when_idle: bool,
    /// Spawn pose; without it the scene's `reset_position`/`reset_rotation` are used.
    pub spawn: Option<Spawn>,
    /// `"independent"` or `"lockstep"` (`--mode`).
    pub mode: String,
    /// Evolution settings as `EvolutionSettings::from_mcp` reads them.
    pub settings: Option<Value>,
    /// WebGPU only: every n-th ray query is also cast on the CPU and compared
    /// (0 disables the check).
    pub gpu_verify_every: u32,
}

impl Default for SessionOptions {
    fn default() -> Self {
        SessionOptions {
            population: None, seed: 0, batch_count: 1, stats_phase: 0, eliminate_on_wall: false, eliminate_when_idle: false,
            spawn: None, mode: "independent".into(), settings: None, gpu_verify_every: 0,
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
    pub options: SessionOptions,
    network: Value,
    output_names: Vec<String>,
    boundary: Option<Boundary>,
}

/// The population and RNG as the current generation began: what a
/// checkpoint saves. Kept as a copy so saving never touches the running cars.
#[derive(Clone)]
struct Boundary {
    generation: u64,
    tick: u64,
    shape: Vec<usize>,
    rng: TrainingRandom,
    /// Every network's parameters, car after car.
    params: Vec<f64>,
}

/// Per-generation statistics hosts read instead of the full car states.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationSummary {
    /// The first car with the highest distance score.
    pub best_index: usize,
    pub best_score: f64,
    /// Cars with at least one completed lap.
    pub lapped: usize,
    pub active: usize,
    /// The best lap of the generation: `(car, seconds)`.
    pub lap_index: Option<usize>,
    pub lap_time: Option<f64>,
}

/// `checkpoint_bytes` magic: format 1.
const CHECKPOINT_MAGIC: &[u8; 8] = b"ALTDCKP1";

fn strings(value: &Value, what: &str) -> Result<Vec<String>, String> {
    value.as_array().ok_or_else(|| format!("missing {what}"))?
        .iter().map(|v| v.as_str().map(str::to_owned).ok_or_else(|| format!("{what} must be strings"))).collect()
}

impl Session {
    /// `scene`, `network` (a game network export with `inputs` and `outputs`,
    /// optionally `weights`/`biases`) and `model` (sensor parameters) as the
    /// CLI loads them.
    pub fn new(scene: &Value, network: &Value, model: &Value, options: SessionOptions) -> Result<Session, String> {
        let spawn = match options.spawn {
            Some(s) => s,
            None => match (scene.get("reset_position"), scene.get("reset_rotation").and_then(Value::as_f64)) {
                (Some(p), Some(rotation)) => { let p = crate::world::vector(p); Spawn { position: [p.x, p.y], rotation } }
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
        let output_names = strings(&network["outputs"], "network outputs")?;
        strings(&network["inputs"], "network inputs")?;
        let mut settings = options.settings.as_ref().map_or_else(EvolutionSettings::default, EvolutionSettings::from_mcp);
        if let Some(population) = options.population {
            settings.population = population;
        }
        if settings.population == 0 {
            return Err("population must be positive".into());
        }
        let world = Arc::new(World::from_scene(scene));
        let layout = SensorLayout::from_exports(network, model);
        let mut runner = TrainingRunner::new(
            world, V2::new(spawn.position[0], spawn.position[1]), spawn.rotation, layout, &output_names, settings,
            PyRandom::new(options.seed), options.batch_count, options.stats_phase, options.eliminate_on_wall, options.eliminate_when_idle,
        );
        runner.mode = mode;
        Ok(Session { runner, options, network: network.clone(), output_names, boundary: None })
    }

    /// The network export's `shape` (or `summary.shape`), if any.
    pub fn template_shape(&self) -> Option<Vec<usize>> {
        let summary = self.network.get("summary").unwrap_or(&self.network);
        summary.get("shape")?.as_array()?.iter().map(|v| v.as_u64().map(|n| n as usize)).collect()
    }

    /// Installs the first generation: mutations of the export's weights when
    /// it has any, else of a Xavier network of its shape.
    pub fn start(&mut self) -> Result<(), String> {
        if self.network.get("weights").is_some() {
            let seed = Network::try_from_game_export(&self.network)?;
            self.check_shape(&seed.shape)?;
            self.runner.start(&seed);
            self.snapshot();
            Ok(())
        } else {
            let shape = self.template_shape().ok_or("the network export has neither weights nor a shape; use start_with_shape")?;
            self.start_with_shape(&shape)
        }
    }

    /// Installs the first generation from a Xavier network of `shape`, drawn
    /// from the session RNG as `train-scratch` does.
    pub fn start_with_shape(&mut self, shape: &[usize]) -> Result<(), String> {
        self.check_shape(shape)?;
        let seed = self.runner.rng.xavier(shape);
        self.runner.start(&seed);
        self.snapshot();
        Ok(())
    }

    fn check_shape(&self, shape: &[usize]) -> Result<(), String> {
        let inputs = self.runner.layout.sensors.len();
        if shape.len() < 2 || shape.iter().any(|&n| n == 0) {
            return Err(format!("invalid network shape {shape:?}"));
        }
        if shape[0] != inputs {
            return Err(format!("the network has {} inputs but the sensor layout has {inputs}", shape[0]));
        }
        if *shape.last().unwrap() < self.output_names.len() {
            return Err(format!("the network has {} outputs but the export names {}", shape.last().unwrap(), self.output_names.len()));
        }
        Ok(())
    }

    pub fn started(&self) -> bool {
        !self.runner.agents.is_empty()
    }

    fn require_started(&self) -> Result<(), String> {
        if self.started() { Ok(()) } else { Err("the session has not started".into()) }
    }

    pub fn agent(&self, index: usize) -> Result<&TrainingAgent, String> {
        self.runner.agents.get(index).ok_or_else(|| format!("car {index} is outside the population of {}", self.runner.agents.len()))
    }

    pub fn advance(&mut self, ticks: u64, stop_when_inactive: bool) -> Result<u64, String> {
        self.require_started()?;
        Ok(self.runner.advance(ticks, stop_when_inactive))
    }

    pub fn advance_generation(&mut self, time_limit_ticks: u64) -> Result<u64, String> {
        self.require_started()?;
        if self.runner.stats_phase != 0 {
            return Err("advance_generation needs statistics phase 0".into());
        }
        Ok(self.runner.advance_generation(time_limit_ticks))
    }

    /// Reproduces and installs the next generation; returns the preserved
    /// parent count and every car's reward.
    pub fn next_generation(&mut self) -> Result<(usize, Vec<f64>), String> {
        self.require_started()?;
        let generation = self.runner.next_generation();
        self.snapshot();
        Ok((generation.preserved_count, generation.rewards))
    }

    /// Change the settings used by the next reproduction without resetting
    /// cars, generation statistics or the training RNG. Population changes
    /// take effect when `next_generation` installs the new population.
    pub fn set_evolution_settings(&mut self, text: &str) -> Result<(), String> {
        let settings: EvolutionSettings = serde_json::from_str(text)
            .map_err(|e| format!("invalid evolution settings: {e}"))?;
        if settings.population == 0 || settings.selection_size == 0 {
            return Err("population and selection_size must be positive".into());
        }
        if !["best", "tournament", "roulette"].contains(&settings.selection_algorithm.as_str())
            || !["none", "single_point", "uniform"].contains(&settings.crossover.as_str())
            || !["off", "on_selection_size", "on_custom"].contains(&settings.preserve_parents.as_str()) {
            return Err("unknown selection, crossover or preservation mode".into());
        }
        if !settings.mutation_rate.is_finite() || !(0.0..=10.0).contains(&settings.mutation_rate)
            || !settings.weight_decay.is_finite() || !(0.0..=1.0).contains(&settings.weight_decay) {
            return Err("mutation_rate must be in 0..10 and weight_decay in 0..1".into());
        }
        if settings.rewards.is_empty() || settings.rewards.iter().any(|r|
            crate::evolution::metric_index(&r.metric).is_none()
            || !["default", "average"].contains(&r.kind.as_str())) {
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
        Ok(self.agent(index)?.stats.metrics().iter().map(|m| m.unwrap_or(f64::NAN)).collect())
    }

    pub fn metric_names() -> Vec<String> {
        METRIC_NAMES.iter().map(|&s| s.to_owned()).collect()
    }

    /// The sensor inputs car `index` would read now.
    pub fn sensors(&self, index: usize) -> Result<Vec<f64>, String> {
        let agent = self.agent(index)?;
        let mut scratch = crate::car::SensorScratch::default();
        let mut out = Vec::new();
        self.runner.layout.read_into(&self.runner.world, &agent.car, &mut scratch, &mut out);
        Ok(out)
    }

    pub fn sensor_names(&self) -> &[String] {
        &self.runner.layout.names
    }

    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// `[acceleration, steering, brake, handbrake, boost]` of car `index`.
    pub fn controls(&self, index: usize) -> Result<[f64; 5], String> {
        let c = self.agent(index)?.controls;
        Ok([c.acceleration, c.steering, c.brake, c.handbrake, c.boost])
    }

    /// The network of car `index` as `{"shape", "weights", "biases"}`.
    pub fn network_json(&self, index: usize) -> Result<String, String> {
        Ok(self.agent(index)?.network.to_json().to_string())
    }

    /// Replaces the network of car `index` with a game export of the same shape.
    pub fn set_network_json(&mut self, index: usize, text: &str) -> Result<(), String> {
        let value: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let network = Network::try_from_game_export(&value)?;
        let agent = self.runner.agents.get_mut(index).ok_or_else(|| format!("car {index} is outside the population"))?;
        if network.shape != agent.network.shape {
            return Err(format!("shape {:?} differs from the population's {:?}", network.shape, agent.network.shape));
        }
        agent.network = network;
        Ok(())
    }

    /// The best lap of the current generation: `(car, seconds)`.
    pub fn best_lap(&self) -> Option<(usize, f64)> {
        self.runner.agents.iter().enumerate()
            .filter_map(|(i, a)| a.stats.best_lap_time.map(|t| (i, t)))
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
    }

    /// A generation-boundary checkpoint: generation, RNG state and networks,
    /// restorable with `restore_checkpoint` on a session with the same
    /// options and files.
    pub fn checkpoint(&self) -> Result<Value, String> {
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
    pub fn restore_checkpoint(&mut self, value: &Value) -> Result<(), String> {
        let shape: Vec<usize> = value["shape"].as_array().ok_or("missing shape")?
            .iter().map(|v| v.as_u64().map(|n| n as usize).ok_or("invalid shape")).collect::<Result<_, _>>()?;
        let generation = value["generation"].as_u64().ok_or("missing generation")?;
        self.check_shape(&shape)?;
        let size = crate::network::parameter_count(&shape);
        let mut params = Vec::new();
        for (i, row) in value["networks"].as_array().ok_or("missing networks")?.iter().enumerate() {
            let row = row.as_array().ok_or("invalid network")?;
            if row.len() != size {
                return Err(format!("network {i} has {} parameters, shape {shape:?} needs {size}", row.len()));
            }
            for v in row {
                params.push(v.as_f64().ok_or("invalid parameter")?);
            }
        }
        let rng = (!value["rng"].is_null()).then(|| TrainingRandom::from_json(&value["rng"]));
        self.restore(&shape, generation, params, rng)
    }

    fn restore(&mut self, shape: &[usize], generation: u64, params: Vec<f64>, rng: Option<TrainingRandom>) -> Result<(), String> {
        self.check_shape(shape)?;
        let size = crate::network::parameter_count(shape);
        if params.is_empty() {
            return Err("the checkpoint has no networks".into());
        }
        if params.len() % size != 0 {
            return Err(format!("the checkpoint has {} parameters, not a multiple of the {size} of shape {shape:?}", params.len()));
        }
        let networks: Vec<Network> = params.chunks(size).map(|p| Network::from_vector(shape, p.to_vec())).collect();
        if let Some(rng) = rng {
            self.runner.rng = rng;
        }
        self.runner.resume(&networks, generation);
        self.snapshot();
        Ok(())
    }

    /// Records the generation that just began as the checkpoint boundary.
    fn snapshot(&mut self) {
        let agents = &self.runner.agents;
        let Some(first) = agents.first() else { return };
        let mut params = Vec::with_capacity(agents.len() * first.network.params.len());
        for a in agents {
            params.extend_from_slice(&a.network.params);
        }
        self.boundary = Some(Boundary {
            generation: self.runner.generation, tick: self.runner.tick, shape: first.network.shape.clone(),
            rng: self.runner.rng.clone(), params,
        });
    }

    /// The generation boundary `checkpoint_bytes` saves: `(generation, tick)`.
    pub fn boundary(&self) -> Option<(u64, u64)> {
        self.boundary.as_ref().map(|b| (b.generation, b.tick))
    }

    /// The boundary of the current generation in a compact binary form:
    /// `ALTDCKP1`, then little-endian u32 generation, tick, population,
    /// layer count and RNG JSON length, the u32 shape, the RNG JSON, zero
    /// padding to a multiple of 8 bytes and every network's f64 parameters.
    pub fn checkpoint_bytes(&self) -> Result<Vec<u8>, String> {
        let b = self.boundary.as_ref().ok_or("the session has not started")?;
        let rng = b.rng.to_json().to_string();
        let size = crate::network::parameter_count(&b.shape);
        let header = [b.generation, b.tick, (b.params.len() / size) as u64, b.shape.len() as u64, rng.len() as u64];
        let mut out = Vec::with_capacity(64 + rng.len() + b.params.len() * 8);
        out.extend_from_slice(CHECKPOINT_MAGIC);
        for n in header.iter().copied().chain(b.shape.iter().map(|&n| n as u64)) {
            out.extend_from_slice(&u32::try_from(n).map_err(|_| "checkpoint field exceeds u32")?.to_le_bytes());
        }
        out.extend_from_slice(rng.as_bytes());
        out.resize(out.len().next_multiple_of(8), 0);
        for p in &b.params {
            out.extend_from_slice(&p.to_le_bytes());
        }
        Ok(out)
    }

    /// Restores a `checkpoint_bytes` checkpoint (see `restore_checkpoint`).
    pub fn restore_checkpoint_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        let invalid = || "invalid binary checkpoint".to_string();
        if bytes.len() < 28 || &bytes[..8] != CHECKPOINT_MAGIC {
            return Err(invalid());
        }
        let word = |i: usize| -> Result<usize, String> {
            let at = 8 + i * 4;
            bytes.get(at..at + 4).map(|w| u32::from_le_bytes(w.try_into().unwrap()) as usize).ok_or_else(invalid)
        };
        let (generation, population, layers, rng_len) = (word(0)? as u64, word(2)?, word(3)?, word(4)?);
        let shape: Vec<usize> = (0..layers).map(|i| word(5 + i)).collect::<Result<_, _>>()?;
        let rng_at = 8 + (5 + layers) * 4;
        let rng_text = bytes.get(rng_at..rng_at + rng_len).ok_or_else(invalid)?;
        let rng: Value = serde_json::from_slice(rng_text).map_err(|e| format!("invalid checkpoint RNG: {e}"))?;
        let params_at = (rng_at + rng_len).next_multiple_of(8);
        let size = if shape.len() >= 2 { crate::network::parameter_count(&shape) } else { 0 };
        let data = bytes.get(params_at..).ok_or_else(invalid)?;
        if data.len() != population * size * 8 {
            return Err(format!("the checkpoint holds {} bytes of parameters, {population} networks of shape {shape:?} need {}", data.len(), population * size * 8));
        }
        let params = data.chunks_exact(8).map(|p| f64::from_le_bytes(p.try_into().unwrap())).collect();
        self.restore(&shape, generation, params, (!rng.is_null()).then(|| TrainingRandom::from_json(&rng)))
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
    pub fn active_count(&self) -> usize {
        self.runner.agents.iter().filter(|a| a.car.active).count()
    }

    /// Wall segments `[x0, y0, x1, y1]...` for drawing.
    pub fn track_walls(&self) -> Vec<f32> {
        self.runner.world.track.walls.iter().flat_map(|w| [w.start.x as f32, w.start.y as f32, w.end.x as f32, w.end.y as f32]).collect()
    }

    /// The baked centre path `[x, y]...`.
    pub fn track_path(&self) -> Vec<f64> {
        self.runner.world.track.path.iter().flat_map(|p| [p.x, p.y]).collect()
    }

    /// `[min_x, min_y, max_x, max_y]` over the walls, or the path without walls.
    pub fn track_bounds(&self) -> [f64; 4] {
        let track = &self.runner.world.track;
        let points: Vec<V2> = if track.walls.is_empty() { track.path.clone() } else { track.walls.iter().flat_map(|w| [w.start, w.end]).collect() };
        points.iter().fold([f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY], |b, p| {
            [b[0].min(p.x), b[1].min(p.y), b[2].max(p.x), b[3].max(p.y)]
        })
    }

    pub fn spawn(&self) -> Spawn {
        Spawn { position: [self.runner.position.x, self.runner.position.y], rotation: self.runner.rotation }
    }

    /// Selects the track the next installed generation drives, spawning at
    /// the scene's reset pose. Call it at a generation boundary, before
    /// `next_generation`; the vehicle must not change.
    pub fn replace_track(&mut self, scene: &Value) -> Result<(), String> {
        let position = scene.get("reset_position").map(crate::world::vector).ok_or("the scene has no reset_position")?;
        let rotation = scene.get("reset_rotation").and_then(Value::as_f64).ok_or("the scene has no reset_rotation")?;
        let world = Arc::new(World::from_scene(scene));
        if world.track.native_broadphase != self.runner.world.track.native_broadphase {
            return Err("a replacement track must keep the broadphase mode".into());
        }
        self.runner.replace_track(world, V2::new(position.x, position.y), rotation);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(path: &str) -> Value {
        let root = env!("CARGO_MANIFEST_DIR");
        serde_json::from_str(&std::fs::read_to_string(format!("{root}/{path}")).unwrap()).unwrap()
    }

    fn autumn(options: &str) -> Session {
        let spawn = &load("traces/autumn_04_spawn.json")["frames"][0];
        let mut options = SessionOptions::from_json(options).unwrap();
        options.spawn = Some(Spawn { position: [spawn["position"][0].as_f64().unwrap(), spawn["position"][1].as_f64().unwrap()], rotation: spawn["rotation"].as_f64().unwrap() });
        Session::new(&load("scenes_exact/autumn_04_formula_scene.json"), &load("formula_network_template.json"), &load("formula_trained_model_exact.json"), options).unwrap()
    }

    #[test]
    fn options_parse_with_defaults_and_reject_unknown_keys() {
        let o = SessionOptions::from_json(r#"{"population": 8, "seed": 3, "eliminateOnWall": true, "mode": "lockstep"}"#).unwrap();
        assert_eq!((o.population, o.seed, o.batch_count, o.eliminate_on_wall, o.eliminate_when_idle, o.mode.as_str()), (Some(8), 3, 1, true, false, "lockstep"));
        assert!(SessionOptions::from_json(r#"{"populaton": 8}"#).is_err());
        assert!(SessionOptions::from_json("{}").unwrap().population.is_none());
    }

    #[test]
    fn evolution_updates_preserve_statistics_and_rng_until_turnover() {
        let mut s = autumn(r#"{"population":6,"seed":5,"eliminateOnWall":true}"#);
        s.start_with_shape(&[20, 8, 5]).unwrap();
        s.advance_generation(120).unwrap();
        let states = s.car_states();
        let rng = s.runner.rng.to_json();
        let generation = s.runner.generation;
        s.set_evolution_settings(r#"{"population":9,"mutation_rate":0.1,"weight_decay":0}"#).unwrap();
        assert_eq!(s.car_states(), states);
        assert_eq!(s.runner.rng.to_json(), rng);
        assert_eq!(s.runner.generation, generation);
        assert!(s.set_evolution_settings(r#"{"mutation_rate":-1}"#).is_err());
        assert!(s.set_evolution_settings(r#"{"selection_algorithm":"unknown"}"#).is_err());
        assert!(s.set_evolution_settings(r#"{"rewards":[{"metric":"unknown","weight":100,"type":"default"}]}"#).is_err());
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
        let mut s = autumn(r#"{"population": 6, "seed": 5, "batchCount": 2, "eliminateOnWall": true, "eliminateWhenIdle": true,
            "settings": {"selection_size": 3, "preserve_parents_size": 2, "mutation_rate": 0.3, "weight_decay": 0.0}}"#);
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
        let mut t = autumn(r#"{"population": 6, "seed": 5, "batchCount": 2, "eliminateOnWall": true, "eliminateWhenIdle": true,
            "settings": {"selection_size": 3, "preserve_parents_size": 2, "mutation_rate": 0.3, "weight_decay": 0.0}}"#);
        t.restore_checkpoint(&checkpoint).unwrap();
        assert_eq!(t.runner.generation, 1);
        t.advance_generation(300).unwrap();
        assert!(states.iter().zip(&t.car_states()).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert_eq!(s.network_json(0).unwrap(), t.network_json(0).unwrap());
        s.set_network_json(1, &t.network_json(0).unwrap()).unwrap();
        assert_eq!(s.network_json(1).unwrap(), t.network_json(0).unwrap());
        // A malformed export (one weight row) and a well-formed one of another shape are both rejected.
        assert!(s.set_network_json(1, r#"{"shape":[20,5],"weights":[[[0.0,0,0,0,0]]],"biases":[[0,0,0,0,0]]}"#).is_err());
        assert!(s.set_network_json(1, &Network::xavier(&[20, 5], &mut PyRandom::new(1)).to_json().to_string()).is_err());
        assert_eq!(s.network_json(1).unwrap(), t.network_json(0).unwrap());
    }

    #[test]
    fn binary_checkpoints_save_the_generation_boundary() {
        let options = r#"{"population": 6, "seed": 5, "eliminateOnWall": true, "eliminateWhenIdle": true,
            "settings": {"selection_size": 3, "mutation_rate": 0.3, "weight_decay": 0.0}}"#;
        let mut s = autumn(options);
        assert!(s.checkpoint_bytes().is_err());
        s.start_with_shape(&[20, 8, 5]).unwrap();
        s.advance_generation(240).unwrap();
        let summary = s.generation_summary();
        let states = s.car_states();
        let best = (0..6).fold(0, |b, i| if states[i * CAR_STATE_STRIDE + 7] > states[b * CAR_STATE_STRIDE + 7] { i } else { b });
        assert_eq!((summary.best_index, summary.best_score), (best, states[best * CAR_STATE_STRIDE + 7]));
        assert_eq!(summary.active, (0..6).filter(|i| states[i * CAR_STATE_STRIDE + 6] == 1.0).count());
        assert_eq!(summary.active, s.active_count());
        s.next_generation().unwrap();
        // The boundary is the checkpoint of the generation that just began,
        // however far that generation has since advanced.
        let json = s.checkpoint().unwrap();
        s.advance(120, false).unwrap();
        let bytes = s.checkpoint_bytes().unwrap();
        assert_eq!(&bytes[..8], b"ALTDCKP1");
        assert_eq!(s.boundary(), Some((1, 0)));
        let mut t = autumn(options);
        t.restore_checkpoint_bytes(&bytes).unwrap();
        assert_eq!(t.checkpoint().unwrap(), json);
        assert_eq!(t.checkpoint_bytes().unwrap(), bytes);
        let mut u = autumn(options);
        u.restore_checkpoint(&json).unwrap();
        assert_eq!(u.checkpoint_bytes().unwrap(), bytes);
        assert!(t.restore_checkpoint_bytes(&bytes[..bytes.len() - 8]).is_err());
        assert!(t.restore_checkpoint_bytes(b"ALTDCKP0").is_err());
    }

    #[test]
    fn export_weights_seed_the_first_generation() {
        let mut options = SessionOptions::default();
        options.population = Some(4);
        options.spawn = Some(Spawn { position: [0.0, 0.0], rotation: 0.0 });
        let spawn = &load("traces/autumn_04_spawn.json")["frames"][0];
        options.spawn = Some(Spawn { position: [spawn["position"][0].as_f64().unwrap(), spawn["position"][1].as_f64().unwrap()], rotation: spawn["rotation"].as_f64().unwrap() });
        let network = Network::xavier(&[20, 7, 5], &mut PyRandom::new(1));
        let mut export = network.to_json();
        for (key, value) in load("formula_network_template.json").as_object().unwrap() {
            export[key] = value.clone();
        }
        let mut s = Session::new(&load("scenes_exact/autumn_04_formula_scene.json"), &export, &load("formula_trained_model_exact.json"), options).unwrap();
        assert_eq!(s.template_shape(), Some(vec![20, 7, 5]));
        s.start().unwrap();
        assert_eq!(s.runner.agents[0].network.shape, vec![20, 7, 5]);
        assert_eq!(s.spawn().rotation, spawn["rotation"].as_f64().unwrap());
    }
}
