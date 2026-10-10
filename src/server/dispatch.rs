//! Operations of `altd-sim serve` on one connection's `Session`.

use super::budget::{self, Lease, Plan, Pool};
use crate::math::profile::MathProfile;
use crate::training::session::{Backend, Session, SessionOptions};
use serde_json::{json, Value};
use std::sync::Arc;

/// A reply: a text frame, or a binary frame with a payload.
#[derive(Debug)]
pub enum Reply {
    Text(Value),
    Binary(Value, Vec<u8>),
}

/// The session of one connection; `create` installs it.
pub struct Connection {
    session: Option<Session>,
    /// The math profile of sessions whose options name none.
    math: MathProfile,
    pool: Arc<Pool>,
    plan: Option<Plan>,
    lease: Option<Lease>,
}

fn arg<'a>(args: &'a Value, name: &str) -> Result<&'a Value, String> {
    args.get(name).ok_or_else(|| format!("missing argument {name}"))
}

fn string_arg<'a>(args: &'a Value, name: &str) -> Result<&'a str, String> {
    arg(args, name)?
        .as_str()
        .ok_or_else(|| format!("argument {name} must be a string"))
}

fn u64_arg(args: &Value, name: &str) -> Result<u64, String> {
    arg(args, name)?
        .as_u64()
        .ok_or_else(|| format!("argument {name} must be a non-negative integer"))
}

fn bool_arg(args: &Value, name: &str) -> Result<bool, String> {
    arg(args, name)?
        .as_bool()
        .ok_or_else(|| format!("argument {name} must be a boolean"))
}

fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Cpu => "cpu",
        Backend::Hip => "hip",
    }
}

impl Connection {
    pub fn new(math: MathProfile) -> Connection {
        Self::with_budget(math, Arc::new(Pool::default()))
    }

    pub(super) fn with_budget(math: MathProfile, pool: Arc<Pool>) -> Connection {
        Connection {
            session: None,
            math,
            pool,
            plan: None,
            lease: None,
        }
    }

    fn admitted(&mut self, op: &str, args: &Value, payload: Option<&[u8]>) -> Result<Reply, String> {
        let pool = self.pool.clone();
        // Do not queue unbounded expensive work behind a busy client.
        let _work = pool
            .work
            .try_lock()
            .map_err(|_| "server busy; retry this request later")?;
        let proposed = self.preflight(op, args, payload)?;
        // Retain the old lease until successful replacement: both old and new
        // allocations can coexist during parsing/reconstruction/reproduction.
        let lease = proposed.map(|p| Lease::new(pool.clone(), p)).transpose()?;
        let reply = self.run(op, args, payload)?;
        if let Some(plan) = proposed {
            self.plan = Some(plan);
            self.lease = lease;
        }
        Ok(reply)
    }

    fn preflight(&self, op: &str, args: &Value, payload: Option<&[u8]>) -> Result<Option<Plan>, String> {
        if op == "create" && self.session.is_some() {
            return Ok(None);
        }
        if op != "create" && self.session.is_none() {
            return Ok(None);
        }
        let mut plan = self.plan.unwrap_or_default();
        match op {
            "create" => {
                let scene = budget::parse(string_arg(args, "scene")?, "scene")?;
                let network = budget::parse(string_arg(args, "network")?, "network")?;
                let model = budget::parse(string_arg(args, "model")?, "model")?;
                let options = &args["options"];
                let population = options
                    .get("population")
                    .map(|n| {
                        n.as_u64()
                            .and_then(|n| usize::try_from(n).ok())
                            .ok_or("invalid population")
                    })
                    .transpose()?;
                plan.population = budget::settings(&options["settings"], 300, population)?;
                if options.get("batchCount").and_then(Value::as_u64).is_some_and(|n| n > 8) {
                    return Err("server batchCount must be 1..8".into());
                }
                if network["inputs"].as_array().is_some_and(|a| a.len() > 64)
                    || network["outputs"].as_array().is_some_and(|a| a.len() > 64)
                    || model["vision"].as_array().is_some_and(|a| a.len() > 64)
                    || model["sensor_layout"]["sensors"]
                        .as_array()
                        .is_some_and(|a| a.len() > 64)
                {
                    return Err("server sensor/output count exceeds 64".into());
                }
                (plan.grid_items, plan.solver) = budget::scene(&scene, options["backend"] == "hip")?;
                let summary = network.get("summary").unwrap_or(&network);
                if let Some(shape) = summary.get("shape") {
                    plan.parameters = budget::shape(shape)?;
                }
            }
            "startWithShape" => {
                plan.parameters = budget::shape(arg(args, "shape")?)?;
            }
            "restoreCheckpointJson" => {
                let c = budget::parse(string_arg(args, "json")?, "checkpoint")?;
                plan.parameters = budget::shape(&c["shape"])?;
                let networks = c["networks"].as_array().ok_or("missing networks")?;
                plan.population = plan.population.max(networks.len());
            }
            "restoreCheckpointBytes" => {
                let (population, parameters) =
                    budget::binary(payload.ok_or("restoreCheckpointBytes needs a binary frame")?)?;
                plan.population = plan.population.max(population);
                plan.parameters = parameters;
            }
            "setEvolutionSettings" => {
                let settings = budget::parse(string_arg(args, "json")?, "settings")?;
                // Existing populations remain live until the next generation.
                // This operation uses serde defaults, not from_mcp's wrapper.
                let parsed: crate::training::evolution::EvolutionSettings =
                    serde_json::from_value(settings.clone()).map_err(|e| format!("invalid evolution settings: {e}"))?;
                budget::settings(&settings, parsed.population, None)?;
                plan.population = plan.population.max(parsed.population);
            }
            "replaceTrack" => {
                let scene = budget::parse(string_arg(args, "scene")?, "scene")?;
                (plan.grid_items, plan.solver) = budget::scene(
                    &scene,
                    self.session.as_ref().is_some_and(|s| s.backend() == Backend::Hip),
                )?;
            }
            "nextGeneration" => {}
            "advance" => {
                plan.ticks(u64_arg(args, "ticks")?)?;
                return Ok(None);
            }
            "advanceGeneration" => {
                let session = self.session.as_ref().unwrap();
                if session.runner.stats_phase != 0 {
                    return Err("advance_generation needs statistics phase 0".into());
                }
                plan.ticks(crate::training::remaining_generation_ticks(
                    u64_arg(args, "timeLimitTicks")?,
                    session.runner.tick,
                ))?;
                return Ok(None);
            }
            _ => return Ok(None),
        }
        plan.bytes()?;
        Ok(Some(plan))
    }

    /// Handles one request. `payload` is the binary payload of a binary
    /// request frame. Errors leave the connection usable.
    pub fn handle(&mut self, id: u64, op: &str, args: &Value, payload: Option<&[u8]>) -> Reply {
        match self.admitted(op, args, payload) {
            Ok(Reply::Text(ok)) => Reply::Text(json!({"id": id, "ok": ok, "state": self.state()})),
            Ok(Reply::Binary(ok, bytes)) => Reply::Binary(json!({"id": id, "ok": ok, "state": self.state()}), bytes),
            Err(error) => Reply::Text(json!({"id": id, "error": error})),
        }
    }

    /// The session state every successful reply carries.
    fn state(&self) -> Value {
        let Some(s) = &self.session else {
            return Value::Null;
        };
        json!({
            "started": s.started(),
            "generation": s.runner.generation,
            "tick": s.runner.tick,
            "population": s.runner.agents.len(),
            "activeCount": s.active_count(),
            "checkpointGeneration": s.boundary().map(|(g, _)| g),
            "backend": backend_name(s.backend()),
            "backendNote": s.backend_note(),
            "mathProfile": s.math_profile(),
        })
    }

    fn session(&mut self) -> Result<&mut Session, String> {
        self.session
            .as_mut()
            .ok_or_else(|| "create a session first".to_string())
    }

    fn run(&mut self, op: &str, args: &Value, payload: Option<&[u8]>) -> Result<Reply, String> {
        if payload.is_some() && op != "restoreCheckpointBytes" {
            return Err(format!("{op} takes no binary payload"));
        }
        let ok = |v: Value| Ok(Reply::Text(v));
        match op {
            "create" => {
                if self.session.is_some() {
                    return Err("the connection already has a session".into());
                }
                let mut options: SessionOptions = match args.get("options") {
                    None | Some(Value::Null) => SessionOptions::default(),
                    Some(o) => {
                        serde_json::from_value(o.clone()).map_err(|e| format!("invalid session options: {e}"))?
                    }
                };
                options.math_profile.get_or_insert(self.math);
                let session = Session::new(
                    &budget::parse(string_arg(args, "scene")?, "scene")?,
                    &budget::parse(string_arg(args, "network")?, "network")?,
                    &budget::parse(string_arg(args, "model")?, "model")?,
                    options,
                )?;
                let reply = json!({
                    "backend": backend_name(session.backend()),
                    "backendNote": session.backend_note(),
                    "mathProfile": session.math_profile(),
                });
                self.session = Some(session);
                ok(reply)
            }
            "start" => {
                self.session()?.start()?;
                ok(Value::Null)
            }
            "startWithShape" => {
                let shape = arg(args, "shape")?
                    .as_array()
                    .ok_or("argument shape must be an array")?
                    .iter()
                    .map(|v| {
                        v.as_u64()
                            .and_then(|n| u32::try_from(n).ok())
                            .map(|n| n as usize)
                            .ok_or("shape entries must be 32-bit non-negative integers")
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.session()?.start_with_shape(&shape)?;
                ok(Value::Null)
            }
            "restoreCheckpointBytes" => {
                let bytes = payload.ok_or("restoreCheckpointBytes needs a binary frame")?;
                self.session()?.restore_checkpoint_bytes(bytes)?;
                ok(Value::Null)
            }
            "restoreCheckpointJson" => {
                let checkpoint = budget::parse(string_arg(args, "json")?, "checkpoint")?;
                self.session()?.restore_checkpoint(&checkpoint)?;
                ok(Value::Null)
            }
            "advance" => {
                let (ticks, stop) = (u64_arg(args, "ticks")?, bool_arg(args, "stopWhenInactive")?);
                ok(json!(self.session()?.advance(ticks, stop)?))
            }
            "advanceGeneration" => {
                let limit = u64_arg(args, "timeLimitTicks")?;
                ok(json!(self.session()?.advance_generation(limit)?))
            }
            "nextGeneration" => {
                let (preserved, rewards) = self.session()?.next_generation()?;
                ok(json!({"preservedCount": preserved, "rewards": rewards}))
            }
            "setEvolutionSettings" => {
                let text = string_arg(args, "json")?;
                self.session()?.set_evolution_settings(text)?;
                ok(Value::Null)
            }
            "replaceTrack" => {
                let scene = budget::parse(string_arg(args, "scene")?, "scene")?;
                self.session()?.replace_track(&scene)?;
                ok(Value::Null)
            }
            "generationSummary" => {
                let session = self.session()?;
                session.sync()?;
                ok(serde_json::to_value(session.generation_summary()).unwrap())
            }
            "networkJson" => {
                let index = u64_arg(args, "index")?;
                let index = usize::try_from(index).map_err(|_| format!("car {index} is outside the population"))?;
                ok(Value::String(self.session()?.network_json(index)?))
            }
            "carStates" => {
                let session = self.session()?;
                session.sync()?;
                let states = session.car_states();
                let bytes = states.iter().flat_map(|v| v.to_le_bytes()).collect();
                Ok(Reply::Binary(json!(states.len()), bytes))
            }
            "checkpointBytes" => Ok(Reply::Binary(Value::Null, self.session()?.checkpoint_bytes()?)),
            other => Err(format!("unknown operation {other:?}")),
        }
    }
}
