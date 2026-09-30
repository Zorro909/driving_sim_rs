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
}

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
        Ok(Session { runner, options, network: network.clone(), output_names })
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
        Ok((generation.preserved_count, generation.rewards))
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
        self.check_shape(&shape)?;
        let generation = value["generation"].as_u64().ok_or("missing generation")?;
        let size = crate::network::parameter_count(&shape);
        let mut networks = Vec::new();
        for (i, row) in value["networks"].as_array().ok_or("missing networks")?.iter().enumerate() {
            let params: Vec<f64> = row.as_array().ok_or("invalid network")?.iter().map(|v| v.as_f64().ok_or("invalid parameter")).collect::<Result<_, _>>()?;
            if params.len() != size {
                return Err(format!("network {i} has {} parameters, shape {shape:?} needs {size}", params.len()));
            }
            networks.push(Network::from_vector(&shape, params));
        }
        if networks.is_empty() {
            return Err("the checkpoint has no networks".into());
        }
        if !value["rng"].is_null() {
            self.runner.rng = TrainingRandom::from_json(&value["rng"]);
        }
        self.runner.resume(&networks, generation);
        Ok(())
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
