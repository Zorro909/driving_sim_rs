//! Operations of `altd-sim serve` on one connection's `Session`.

use crate::training::session::{Backend, Session, SessionOptions};
use serde_json::{json, Value};

/// A reply: a text frame, or a binary frame with a payload.
#[derive(Debug)]
pub enum Reply {
    Text(Value),
    Binary(Value, Vec<u8>),
}

/// The session of one connection; `create` installs it.
#[derive(Default)]
pub struct Connection {
    session: Option<Session>,
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

fn parse(text: &str, what: &str) -> Result<Value, String> {
    serde_json::from_str(text).map_err(|e| format!("invalid {what} JSON: {e}"))
}

fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Cpu => "cpu",
        Backend::Hip => "hip",
    }
}

impl Connection {
    /// Handles one request. `payload` is the binary payload of a binary
    /// request frame. Errors leave the connection usable.
    pub fn handle(&mut self, id: u64, op: &str, args: &Value, payload: Option<&[u8]>) -> Reply {
        match self.run(op, args, payload) {
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
                let options: SessionOptions = match args.get("options") {
                    None | Some(Value::Null) => SessionOptions::default(),
                    Some(o) => {
                        serde_json::from_value(o.clone()).map_err(|e| format!("invalid session options: {e}"))?
                    }
                };
                let session = Session::new(
                    &parse(string_arg(args, "scene")?, "scene")?,
                    &parse(string_arg(args, "network")?, "network")?,
                    &parse(string_arg(args, "model")?, "model")?,
                    options,
                )?;
                let reply = json!({"backend": backend_name(session.backend()), "backendNote": session.backend_note()});
                self.session = Some(session);
                ok(reply)
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
                let checkpoint = parse(string_arg(args, "json")?, "checkpoint")?;
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
                let scene = parse(string_arg(args, "scene")?, "scene")?;
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
