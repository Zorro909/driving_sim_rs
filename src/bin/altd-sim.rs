//! CPU simulator CLI, training checkpoints, and recorded-game comparisons.
//!
//! Comparison commands retain the Python report schemas, while simulation
//! arithmetic follows the original game runtime.

use altd_sim::car::{Car, Controls, Sensor, SensorScratch, DT};
use altd_sim::evolution::{reward_values, EvolutionSettings, RewardSpec, METRIC_NAMES};
use altd_sim::network::{parameter_count, Network};
use altd_sim::pymath::{py_max, py_min, py_pow, py_remainder, py_sum};
use altd_sim::pyrandom::PyRandom;
use altd_sim::training::{Mode, ScoreTracker, SensorLayout, TrainingAgent, TrainingRunner, TrainingRandom};
use altd_sim::world::{vector, World};
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::{json, Map, Value};
use std::f64::consts::PI;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;


#[derive(Parser)]
#[command(about = "CPU game-fidelity simulator and trainer for AI Learns To Drive")]
struct Cli {
    /// Worker threads (default: all logical CPUs in the affinity mask).
    #[arg(long, global = true)]
    threads: Option<usize>,
    /// Training execution: each car runs a whole window independently, or all
    /// cars advance one tick at a time like `TrainingRunner.step`.
    #[arg(long, global = true, value_enum, default_value_t = ModeArg::Independent)]
    mode: ModeArg,
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
enum ModeArg {
    Independent,
    Lockstep,
}

impl From<ModeArg> for Mode {
    fn from(mode: ModeArg) -> Mode {
        match mode {
            ModeArg::Independent => Mode::Independent,
            ModeArg::Lockstep => Mode::Lockstep,
        }
    }
}

/// Mutation rate interpolation between `--mutation-start` and `--mutation-end`.
#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum Schedule {
    /// Constant ratio per generation (equal time per halving).
    Geometric,
    Linear,
}

#[derive(Clone, Copy, ValueEnum)]
enum ScratchReward {
    /// Distance along the track (total_score).
    Distance,
    /// Faster completed laps rank higher (best_lap_performance).
    BestLapTime,
}

impl ScratchReward {
    fn spec(self) -> RewardSpec {
        let metric = match self {
            Self::Distance => "total_score",
            Self::BestLapTime => "best_lap_performance",
        };
        RewardSpec { metric: metric.into(), weight: 100, kind: "default".into() }
    }
}

#[derive(Subcommand)]
enum Command {
    /// benchmark_population.py: timed training ticks for a population.
    Bench {
        #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/scenes_exact/rally_a07_contact_scene.json"))]
        scene: PathBuf,
        #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../docs/traces/rally_a01_live_network.json"))]
        network: PathBuf,
        #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/rally_trained_model_exact.json"))]
        model: PathBuf,
        #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../docs/traces/rally_a07_contact_reference.json"))]
        spawn_trace: PathBuf,
        #[arg(long, default_value_t = 1000)]
        population: usize,
        #[arg(long, default_value_t = 60)]
        ticks: u64,
        #[arg(long, default_value_t = 2)]
        warmup_ticks: u64,
        #[arg(long, default_value_t = 7)]
        seed: i64,
        #[arg(long, default_value_t = 0)]
        spawn_index: usize,
        #[arg(long, default_value_t = 2)]
        batch_count: usize,
        #[arg(long)]
        report: Option<PathBuf>,
        /// Write every car's final state, metrics, and the next generation's
        /// networks for parity checks against Python.
        #[arg(long)]
        dump_state: Option<PathBuf>,
    },
    /// train.py: run and evolve networks for several generations.
    Train {
        #[arg(long)]
        scene: PathBuf,
        #[arg(long)]
        network: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        spawn_trace: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        settings: Option<PathBuf>,
        #[arg(long)]
        population: Option<usize>,
        #[arg(long, default_value_t = 2)]
        generations: usize,
        #[arg(long)]
        ticks: Option<u64>,
        #[arg(long, default_value_t = 7)]
        seed: i64,
        /// Exact game RNG state JSON containing independent decisions and normals streams.
        #[arg(long)]
        game_rng_state: Option<PathBuf>,
        #[arg(long, default_value_t = 2)]
        batch_count: usize,
        #[arg(long, overrides_with = "no_eliminate_on_wall")]
        eliminate_on_wall: bool,
        #[arg(long)]
        no_eliminate_on_wall: bool,
        #[arg(long, overrides_with = "no_idle_eliminate")]
        idle_eliminate: bool,
        #[arg(long)]
        no_idle_eliminate: bool,
    },
    /// Train a network from a random Xavier initialization with a decaying
    /// mutation rate, logging lap times each generation and saving the network
    /// behind every new best lap. Defaults: generalist rally network on B06.
    TrainScratch {
        #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/scenes_exact/rally_b06_scene.json"))]
        scene: PathBuf,
        /// Trace whose first frame is the spawn pose.
        #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../docs/traces/rally_b06_reference.json"))]
        spawn_trace: PathBuf,
        /// Network export supplying the input sensor and output control names; its weights are unused.
        #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../docs/traces/rally_a01_live_network.json"))]
        network: PathBuf,
        #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/rally_trained_model_exact.json"))]
        model: PathBuf,
        /// Run directory: log, best-lap networks, and checkpoints.
        #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../training_runs/b06_scratch"))]
        out_dir: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "20,16,16,16,16,12,12,8,5")]
        shape: Vec<usize>,
        #[arg(long, default_value_t = 8192)]
        population: usize,
        #[arg(long, default_value_t = 50000)]
        generations: usize,
        /// Simulated ticks per generation (60 per second).
        #[arg(long, default_value_t = 5400)]
        ticks: u64,
        /// Mutation rate of generation 0; adaptive mutation then scales it by network shape.
        #[arg(long, default_value_t = 0.4)]
        mutation_start: f64,
        /// Mutation rate of the last generation.
        #[arg(long, default_value_t = 0.0125)]
        mutation_end: f64,
        #[arg(long, value_enum, default_value_t = Schedule::Geometric)]
        schedule: Schedule,
        /// Evolution settings JSON overriding the generalist defaults (population and mutation rate excluded).
        #[arg(long)]
        settings: Option<PathBuf>,
        /// Selection reward. Overrides settings-file rewards; otherwise defaults to distance.
        #[arg(long, value_enum)]
        reward: Option<ScratchReward>,
        #[arg(long, default_value_t = 1)]
        seed: i64,
        /// Exact game RNG state JSON containing independent decisions and normals streams.
        #[arg(long)]
        game_rng_state: Option<PathBuf>,
        /// Seed the first generation from this network export instead of a random Xavier network.
        #[arg(long)]
        init_network: Option<PathBuf>,
        /// Continue another run's population: the checkpoint.json of a run with the same shape,
        /// e.g. from another simulator. Takes its networks, generation (mutation
        /// schedule) and RNG; best laps start over.
        #[arg(long, conflicts_with = "init_network")]
        init_population: Option<PathBuf>,
        #[arg(long, default_value_t = 2)]
        batch_count: usize,
        /// Deactivate cars on wall contact. Overrides the settings file.
        #[arg(long, overrides_with = "no_eliminate_on_wall")]
        eliminate_on_wall: bool,
        #[arg(long, overrides_with = "eliminate_on_wall")]
        no_eliminate_on_wall: bool,
        /// Deactivate cars after sustained lack of forward progress. Overrides the settings file.
        #[arg(long, overrides_with = "no_idle_eliminate")]
        idle_eliminate: bool,
        #[arg(long, overrides_with = "idle_eliminate")]
        no_idle_eliminate: bool,
        /// Write a resumable checkpoint every N generations (0 disables).
        #[arg(long, default_value_t = 100)]
        checkpoint_every: usize,
        /// Continue from the checkpoint in --out-dir, allowing new settings with the same network shape.
        #[arg(long)]
        resume: bool,
        /// Simulate the generations on the GPU (gpu/sim, bit-exact with the CPU;
        /// ALTD_GPU_LIB overrides the library path).
        #[arg(long)]
        gpu: bool,
    },
    /// compare_game_trace.py: open-loop replay of recorded controls.
    CompareTrace {
        scene: PathBuf,
        trace: PathBuf,
        report: PathBuf,
        #[arg(long, default_value_t = 1)]
        start_index: usize,
        #[arg(long)]
        end_index: Option<usize>,
    },
    /// compare_one_step.py: one transition from each recorded state.
    CompareOneStep { scene: PathBuf, trace: PathBuf, report: PathBuf },
    /// compare_closed_loop.py: network and physics together against a game episode.
    CompareClosedLoop {
        scene: PathBuf,
        trace: PathBuf,
        network: PathBuf,
        model: PathBuf,
        report: PathBuf,
        #[arg(long, default_value_t = 2)]
        batch_count: usize,
        #[arg(long)]
        new_vehicle: bool,
    },
    /// compare_network.py: inference on recorded sensors.
    CompareNetwork {
        network: PathBuf,
        trace: PathBuf,
        report: PathBuf,
        #[arg(long, default_value_t = 2)]
        first_tick: usize,
    },
    /// compare_sensors.py: sensors on recorded states.
    CompareSensors {
        scene: PathBuf,
        trace: PathBuf,
        model: PathBuf,
        trajectory_report: PathBuf,
        sensor_report: PathBuf,
        #[arg(long)]
        all_frames: bool,
    },
    /// compare_score.py: path score on recorded positions.
    CompareScore { scene: PathBuf, trace: PathBuf, report: PathBuf },
}

fn load(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn write_report(path: &Path, value: &Value) {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create report directory");
        }
    }
    std::fs::write(path, serde_json::to_string_pretty(value).unwrap() + "\n")
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// Print a report without its bulky per-row key, like the Python `main()`s.
fn print_without(value: &Value, skip: &str) {
    let mut map = value.as_object().unwrap().clone();
    map.shift_remove(skip);
    println!("{}", serde_json::to_string_pretty(&Value::Object(map)).unwrap());
}

fn num(value: &Value) -> f64 {
    value.as_f64().unwrap_or_else(|| panic!("expected a number, got {value}"))
}

fn frames(trace: &Value) -> &Vec<Value> {
    trace["frames"].as_array().expect("trace frames")
}

/// `Controls(**{item["name"].lower(): item["value"]})`.
fn recorded_controls(frame: &Value) -> Controls {
    let mut controls = Controls::default();
    for item in frame["outputs"].as_array().expect("outputs") {
        controls.set(item["name"].as_str().unwrap(), num(&item["value"]));
    }
    controls
}

/// Output names in `{name.lower(): value}` dict order (first position, last value).
fn recorded_outputs(frame: &Value) -> Vec<(String, f64)> {
    let mut outputs: Vec<(String, f64)> = Vec::new();
    for item in frame["outputs"].as_array().expect("outputs") {
        let name = item["name"].as_str().unwrap().to_lowercase();
        let value = num(&item["value"]);
        match outputs.iter_mut().find(|(n, _)| *n == name) {
            Some(slot) => slot.1 = value,
            None => outputs.push((name, value)),
        }
    }
    outputs
}

/// Python `max(values)`: the first maximum wins.
fn py_max_of(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut iter = values.into_iter();
    let first = iter.next().expect("max() of an empty sequence");
    iter.fold(first, py_max)
}

fn rotation_error(a: f64, b: f64) -> f64 {
    py_remainder(a - b, 2.0 * PI).abs()
}

fn opt_tick(value: Option<usize>) -> Value {
    value.map_or(Value::Null, |v| json!(v))
}

/// `{name: {"max": max(row[name]), "rmse": sqrt(sum(row[name] ** 2) / n)}}`.
fn summary(rows: &[Value], fields: &[&str]) -> Value {
    if rows.is_empty() {
        return json!({});
    }
    let mut out = Map::new();
    for &field in fields {
        let values: Vec<f64> = rows.iter().map(|row| num(&row[field])).collect();
        let rmse = (py_sum(values.iter().map(|&v| py_pow(v, 2.0))) / values.len() as f64).sqrt();
        out.insert(field.into(), json!({"max": py_max_of(values.iter().copied()), "rmse": rmse}));
    }
    Value::Object(out)
}

/// Car state from a recorded frame plus the wheel history of the frame before.
fn car_from_frame(world: &World, state: &Value, previous: Option<&Value>) -> Car {
    altd_sim::trace_state::car_from_frame(world, state, previous, None)
}

fn compare_trace(world: &World, trace: &Value, start_index: usize, end_index: Option<usize>) -> Value {
    const LIMITS: [(&str, f64); 3] =
        [("position_error_px", 0.05), ("velocity_error_px_s", 1.0), ("rotation_error_rad", 0.001)];
    let frames = frames(trace);
    let first = &frames[start_index];
    let previous = if start_index > 0 { Some(&frames[start_index - 1]) } else { None };
    let mut car = car_from_frame(world, first, previous);
    let baseline = first["collision_count"].as_i64().expect("collision_count");
    let end_index = end_index.unwrap_or(frames.len());
    assert!(start_index < end_index && end_index <= frames.len(), "start_index and end_index must delimit trace frames");
    let mut rows = Vec::new();
    for (index, expected) in frames.iter().enumerate().take(end_index).skip(start_index + 1) {
        let contact = car.step(world, &recorded_controls(expected), DT, false);
        rows.push(json!({
            "tick": index,
            "position_error_px": (car.position - vector(&expected["position"])).length(),
            "velocity_error_px_s": (car.velocity - vector(&expected["velocity"])).length(),
            "rotation_error_rad": rotation_error(car.rotation, num(&expected["rotation"])),
            "game_contact_count": expected["collision_count"].as_i64().unwrap() - baseline,
            "python_contact": contact,
            "python_position": [car.position.x, car.position.y],
            "game_position": expected["position"],
        }));
    }
    let tick_of = |row: &Value| row["tick"].as_u64().unwrap() as usize;
    let game_increase = rows.iter().find(|r| r["game_contact_count"].as_i64().unwrap() > 0).map(tick_of);
    let first_contact = rows.iter().find(|r| r["python_contact"] == true).map(tick_of);
    let divergence =
        rows.iter().find(|r| LIMITS.iter().any(|(name, limit)| num(&r[*name]) > *limit)).map(tick_of);
    let agreement: Vec<Value> =
        rows.iter().filter(|r| divergence.is_none_or(|stop| tick_of(r) < stop)).cloned().collect();
    let fields = ["position_error_px", "velocity_error_px_s", "rotation_error_rad"];
    json!({
        "start_index": start_index,
        "end_index": end_index,
        "game_first_collision_count_increase_tick": opt_tick(game_increase),
        "python_first_contact_tick": opt_tick(first_contact),
        "first_large_divergence_tick": opt_tick(divergence),
        "agreement_limits": LIMITS.iter().map(|(n, l)| ((*n).to_string(), json!(l))).collect::<Map<_, _>>(),
        "agreement_prefix_frames": agreement.len(),
        "agreement_prefix": summary(&agreement, &fields),
        "full_trace": summary(&rows, &fields),
        "frames": rows,
    })
}

fn compare_one_step(world: &World, trace: &Value) -> Value {
    let frames = frames(trace);
    let has_contact_data = frames[0].get("physics_contacts").is_some();
    let mut rows = Vec::new();
    for index in 2..frames.len() {
        let (previous, state, expected) = (&frames[index - 2], &frames[index - 1], &frames[index]);
        let mut car = altd_sim::trace_state::car_from_frame(world, state, Some(previous), index.checked_sub(3).map(|i| &frames[i]));
        let contact = car.step(world, &recorded_controls(expected), DT, false);
        let game_contact = if has_contact_data {
            json!(match &expected["physics_contacts"] {
                Value::Array(items) => !items.is_empty(),
                Value::Null => false,
                Value::Bool(b) => *b,
                Value::Object(map) => !map.is_empty(),
                other => num(other) != 0.0,
            })
        } else {
            Value::Null
        };
        let mut bit_mismatches = Vec::new();
        let mut check = |name: &str, actual: f64, reference: &Value| {
            let expected = num(reference) as f32;
            if (actual as f32).to_bits() != expected.to_bits() {
                bit_mismatches.push(json!({"field": name,
                    "actual": actual, "expected": expected as f64,
                    "actual_bits": format!("{:08x}", (actual as f32).to_bits()),
                    "expected_bits": format!("{:08x}", expected.to_bits())}));
            }
        };
        check("position.x", car.position.x, &expected["position"][0]);
        check("position.y", car.position.y, &expected["position"][1]);
        check("velocity.x", car.velocity.x, &expected["velocity"][0]);
        check("velocity.y", car.velocity.y, &expected["velocity"][1]);
        check("rotation", car.rotation, &expected["rotation"]);
        check("angular_velocity", car.angular_velocity, &expected["angular_velocity"]);
        check("boost_energy", car.boost_energy, &expected["boost_energy"]);
        for (i, wheel) in car.wheels.iter().enumerate() {
            check(&format!("wheel_angles.{i}"), wheel.angle_deg, &expected["wheel_angles"][i]);
        }
        rows.push(json!({
            "tick": index,
            "position_error_px": (car.position - vector(&expected["position"])).length(),
            "velocity_error_px_s": (car.velocity - vector(&expected["velocity"])).length(),
            "rotation_error_rad": rotation_error(car.rotation, num(&expected["rotation"])),
            "angular_velocity_error_rad_s": (car.angular_velocity - num(&expected["angular_velocity"])).abs(),
            "python_contact": contact,
            "game_contact": game_contact,
            "bit_mismatches": bit_mismatches,
        }));
    }
    let fields = ["position_error_px", "velocity_error_px_s", "rotation_error_rad", "angular_velocity_error_rad_s"];
    let pick = |want: bool| -> Vec<Value> { rows.iter().filter(|r| r["game_contact"] == want).cloned().collect() };
    let mut worst = Map::new();
    for field in fields {
        let mut best = &rows[0];
        for row in &rows[1..] {
            if num(&row[field]) > num(&best[field]) {
                best = row;
            }
        }
        worst.insert(field.into(), best["tick"].clone());
    }
    json!({
        "frames": rows.len(),
        "bit_exact_frames": rows.iter().filter(|r| r["bit_mismatches"].as_array().unwrap().is_empty()).count(),
        "bit_mismatch_counts": ({
            let mut counts = std::collections::BTreeMap::<String, usize>::new();
            for row in &rows {
                for mismatch in row["bit_mismatches"].as_array().unwrap() {
                    *counts.entry(mismatch["field"].as_str().unwrap().to_owned()).or_default() += 1;
                }
            }
            counts
        }),
        "all": summary(&rows, &fields),
        "contacts": summary(&pick(true), &fields),
        "noncontacts": summary(&pick(false), &fields),
        "contact_mismatches": rows.iter()
            .filter(|r| !r["game_contact"].is_null() && r["game_contact"] != r["python_contact"])
            .map(|r| r["tick"].clone()).collect::<Vec<_>>(),
        "worst": worst,
        "rows": rows,
    })
}

/// Settings and runner shared by closed-loop comparisons.
fn closed_loop_runner(
    world: &Arc<World>, first: &Value, network_data: &Value, model: &Value, batch_count: usize, stats_phase: u64,
    mode: Mode,
) -> TrainingRunner {
    let settings = EvolutionSettings {
        population: 1,
        selection_size: 1,
        mutation_rate: 0.0,
        weight_decay: 0.0,
        preserve_parents: "on_custom".into(),
        preserve_parents_size: 1,
        ..EvolutionSettings::default()
    };
    let mut runner = TrainingRunner::new(
        world.clone(),
        vector(&first["position"]),
        num(&first["rotation"]),
        SensorLayout::from_exports(network_data, model),
        &output_names(network_data),
        settings,
        PyRandom::new(0),
        batch_count,
        stats_phase,
        false,
        false,
    );
    runner.mode = mode;
    runner
}

fn output_names(network_data: &Value) -> Vec<String> {
    network_data["outputs"].as_array().expect("outputs").iter().map(|v| v.as_str().unwrap().to_string()).collect()
}

fn compare_closed_loop(
    world: &Arc<World>, trace: &Value, network_data: &Value, model: &Value, batch_count: usize, reused: bool,
    mode: Mode,
) -> Value {
    let frames = frames(trace);
    let first = &frames[0];
    let score_change_ticks: Vec<usize> =
        (1..frames.len()).filter(|&i| frames[i]["score"] != frames[i - 1]["score"]).collect();
    // Counter(...).most_common(1): highest count, first-seen residue on ties.
    let mut counts: Vec<(usize, usize)> = Vec::new();
    for &tick in &score_change_ticks {
        match counts.iter_mut().find(|(r, _)| *r == tick % 6) {
            Some(entry) => entry.1 += 1,
            None => counts.push((tick % 6, 1)),
        }
    }
    let stats_phase = counts.iter().fold(None, |best: Option<(usize, usize)>, &(r, c)| match best {
        Some((_, bc)) if bc >= c => best,
        _ => Some((r, c)),
    });
    let stats_phase = stats_phase.map_or(0, |(r, _)| r) as u64;
    let valid: std::collections::HashSet<usize> = score_change_ticks.iter().skip(2).copied().collect();
    let mut runner = closed_loop_runner(world, first, network_data, model, batch_count, stats_phase, mode);
    runner.start(&Network::from_game_export(network_data));
    if reused || first.get("reset_transform").and_then(Value::as_bool) == Some(true) {
        runner.agents[0].car.reset(world, vector(&first["position"]), num(first.get("reset_rotation").unwrap_or(&first["rotation"])), true);
    }
    // RecordTrace resets the vehicle, not the running inference batch cursor.
    // Recover its phase from the first observed output update.
    let inference_phase = frames.windows(2).position(|f| f[0]["outputs"] != f[1]["outputs"]).unwrap_or(0) % batch_count;
    runner.batch_index = (8 - (inference_phase * runner.batches_per_tick()) % 8) % 8;
    runner.agents[0].controls = recorded_controls(first);
    let mut rows = Vec::new();
    for (index, expected) in frames.iter().enumerate().skip(1) {
        runner.step();
        let agent = &runner.agents[0];
        let car = &agent.car;
        let outputs = recorded_outputs(expected);
        let expected_effective = recorded_controls(expected).normalized();
        let effective = agent.controls.normalized();
        let mut row = json!({
            "tick": index,
            "position_error_px": (car.position - vector(&expected["position"])).length(),
            "velocity_error_px_s": (car.velocity - vector(&expected["velocity"])).length(),
            "rotation_error_rad": rotation_error(car.rotation, num(&expected["rotation"])),
            "output_error": py_max_of(outputs.iter().map(|(n, v)| (agent.controls.get(n) - v).abs())),
            "effective_output_error": py_max_of(outputs.iter().map(|(n, _)| (effective.get(n) - expected_effective.get(n)).abs())),
        });
        if valid.contains(&index) {
            row["score_absolute_error"] = json!((agent.stats.total_score - num(&expected["score"])).abs());
        }
        rows.push(row);
    }
    let fields = ["position_error_px", "velocity_error_px_s", "rotation_error_rad", "output_error", "effective_output_error"];
    let score_summary = |rows: &[Value]| -> Value {
        let values: Vec<f64> = rows.iter().filter_map(|r| r.get("score_absolute_error").map(num)).collect();
        if values.is_empty() {
            return json!({});
        }
        let rmse = (py_sum(values.iter().map(|v| v * v)) / values.len() as f64).sqrt();
        json!({"updates": values.len(), "max": py_max_of(values.iter().copied()), "rmse": rmse})
    };
    let first_over = |limit: f64| {
        opt_tick(rows.iter().find(|r| num(&r["position_error_px"]) > limit).map(|r| r["tick"].as_u64().unwrap() as usize))
    };
    let head = |n: usize| &rows[..rows.len().min(n)];
    json!({
        "frames": rows.len(),
        "batch_count": batch_count,
        "inference_phase": inference_phase,
        "score_update_phase": stats_phase,
        "first_position_error_over_0_05_px": first_over(0.05),
        "first_position_error_over_1_px": first_over(1.0),
        "first_position_error_over_10_px": first_over(10.0),
        "first_half_second": summary(head(30), &fields),
        "first_200_frames": summary(head(200), &fields),
        "first_500_frames": summary(head(500), &fields),
        "full": summary(&rows, &fields),
        "score_first_500_frames": score_summary(head(500)),
        "score_full": score_summary(&rows),
        "rows": rows,
    })
}

fn compare_network(network_data: &Value, trace: &Value, first_tick: usize) -> Value {
    let network = Network::from_game_export(network_data);
    let frames = frames(trace);
    let input_names: Vec<&str> = network_data["inputs"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let output_names = output_names(network_data);
    let mut per_output: Vec<Vec<f64>> = vec![Vec::new(); output_names.len()];
    let mut worst = (0.0, Value::Null);
    for frame in frames.iter().skip(first_tick) {
        let sensors = frame["sensors"].as_array().unwrap();
        let outputs = frame["outputs"].as_array().unwrap();
        assert!(sensors.iter().map(|s| s["name"].as_str().unwrap()).eq(input_names.iter().copied()), "trace sensor order differs from live network");
        assert!(outputs.iter().map(|s| s["name"].as_str().unwrap()).eq(output_names.iter().map(String::as_str)), "trace output order differs from live network");
        let inputs: Vec<f64> = sensors.iter().map(|s| num(&s["value"])).collect();
        let predicted = network.forward(&inputs);
        for (j, (value, actual)) in predicted.iter().zip(outputs).enumerate() {
            let error = (value - num(&actual["value"])).abs();
            per_output[j].push(error);
            if error > worst.0 {
                worst = (error, frame["tick"].clone());
            }
        }
    }
    // Python dict: duplicate output names share one error list.
    let mut summaries = Map::new();
    for (name, errors) in output_names.iter().zip(&per_output) {
        let rmse = (py_sum(errors.iter().map(|e| e * e)) / errors.len() as f64).sqrt();
        summaries.insert(name.clone(), json!({"max": py_max_of(errors.iter().copied()), "rmse": rmse}));
    }
    json!({
        "network_shape": network.shape,
        "compared_frames": frames.len() as i64 - first_tick as i64,
        "max_error": worst.0,
        "worst_tick": worst.1,
        "outputs": summaries,
    })
}

fn compare_sensors(world: &World, trace: &Value, model: &Value, trajectory: &Value, all_frames: bool) -> Value {
    const KINDS: [(&str, &str); 7] = [
        ("AngVel", "angularVelocity"),
        ("Boost", "boostCapacity"),
        ("Dir", "correctDirection"),
        ("Grip", "grip"),
        ("VelF", "velocityFront"),
        ("VelS", "velocitySide"),
        ("Curve", "trackCurvature"),
    ];
    let frames = frames(trace);
    let source_index = |tick: usize, parity: usize| if tick % 2 == parity { tick - 1 } else { tick - 2 };
    let actual_of = |frame: &Value, name: &str| -> f64 {
        // {item["name"]: item["value"]}: the last duplicate wins.
        frame["sensors"].as_array().unwrap().iter().rev().find(|i| i["name"] == name).map(|i| num(&i["value"]))
            .unwrap_or_else(|| panic!("missing sensor {name}"))
    };
    let first_tick = 2.max(trajectory["start_index"].as_u64().unwrap() as usize + 1);
    let stop_tick = if all_frames {
        frames.len()
    } else {
        let end = trajectory.get("end_index").map_or(frames.len(), |v| v.as_u64().unwrap() as usize);
        match trajectory["first_large_divergence_tick"].as_u64() {
            Some(d) => (d as usize).min(end),
            None => end,
        }
    };
    let residual = |parity: usize| {
        let mut error = 0.0;
        for tick in 3.max(first_tick)..stop_tick.min(first_tick + 40) {
            let sample = &frames[source_index(tick, parity)];
            let angular = py_max(-1.0, py_min(1.0, num(&sample["angular_velocity"]) / 2.0));
            error += py_pow(angular - actual_of(&frames[tick], "AngVel"), 2.0);
            error += py_pow(num(&sample["boost_energy"]) - actual_of(&frames[tick], "Boost"), 2.0);
        }
        error
    };
    let parity = if residual(1) < residual(0) { 1 } else { 0 };
    let vision: Vec<(String, f64, f64)> = model["vision"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| (item["angle"].to_string(), num(&item["angle"]), num(&item["length"])))
        .collect();
    let mut errors: Vec<(String, Vec<f64>)> = KINDS.iter().map(|(n, _)| (n.to_string(), Vec::new())).collect();
    for (label, _, _) in &vision {
        let key = format!("vision_{label}");
        if !errors.iter().any(|(n, _)| *n == key) {
            errors.push((key, Vec::new()));
        }
    }
    let push = |errors: &mut Vec<(String, Vec<f64>)>, name: &str, value: f64| {
        errors.iter_mut().find(|(n, _)| n == name).unwrap().1.push(value);
    };
    let mut scratch = SensorScratch::default();
    let mut ray_mismatches = Vec::new();
    for tick in first_tick..stop_tick {
        let index = source_index(tick, parity);
        let car = altd_sim::trace_state::car_from_frame(world, &frames[index],
            index.checked_sub(1).map(|i| &frames[i]), index.checked_sub(2).map(|i| &frames[i]));
        scratch.invalidate();
        for (name, kind) in KINDS {
            let value = car.sensor(world, Sensor::by_name(kind), &mut scratch) - actual_of(&frames[tick], name);
            push(&mut errors, name, value);
        }
        for (label, degrees, length) in &vision {
            let predicted = car.sensor(world, Sensor::Raycast { degrees: *degrees, length: *length }, &mut scratch);
            let expected = actual_of(&frames[tick], &format!("↑ {label}°"));
            let value = predicted - expected;
            if value != 0.0 {
                ray_mismatches.push(json!({"tick":tick,"source_index":index,"degrees":degrees,
                    "length":length,"actual":predicted,"expected":expected}));
            }
            push(&mut errors, &format!("vision_{label}"), value);
        }
    }
    let summarize = |errors: &[f64]| -> Value {
        if errors.is_empty() {
            return Value::Null;
        }
        let rmse = (py_sum(errors.iter().map(|e| e * e)) / errors.len() as f64).sqrt();
        json!({"samples": errors.len(), "nonzero_errors":errors.iter().filter(|e| **e != 0.0).count(), "rmse": rmse, "max": py_max_of(errors.iter().map(|e| e.abs()))})
    };
    let vision_errors: Vec<f64> =
        errors.iter().filter(|(n, _)| n.starts_with("vision_")).flat_map(|(_, v)| v.iter().copied()).collect();
    json!({
        "first_tick": first_tick,
        "stop_tick_exclusive": stop_tick,
        "sensor_update_parity": parity,
        "sample_state_rule": "tick-1 on update ticks; tick-2 on held ticks",
        "vision_all": summarize(&vision_errors),
        "ray_mismatches":ray_mismatches,
        "sensors": errors.iter().map(|(n, v)| (n.clone(), summarize(v))).collect::<Map<_, _>>(),
    })
}

fn compare_score(world: &World, trace: &Value) -> Value {
    let frames = frames(trace);
    let mut score = ScoreTracker::default();
    let mut rows = Vec::new();
    for index in 1..frames.len() {
        if frames[index]["score"].as_f64() == frames[index - 1]["score"].as_f64() {
            continue;
        }
        score.update(&world.track, vector(&frames[index - 1]["position"]));
        let game = num(&frames[index]["score"]);
        rows.push(json!({
            "tick": index,
            "python_score": score.total_score,
            "game_score": frames[index]["score"],
            "absolute_error": (score.total_score - game).abs(),
        }));
    }
    let stable = &rows[2.min(rows.len())..];
    let errors: Vec<f64> = stable.iter().map(|r| num(&r["absolute_error"])).collect();
    json!({
        "updates": rows.len(),
        "warmup_updates_excluded": rows.len() - stable.len(),
        "max_error": py_max_of(errors.iter().copied()),
        "rmse": (py_sum(errors.iter().map(|&e| py_pow(e, 2.0))) / errors.len() as f64).sqrt(),
        "rows": rows,
    })
}

fn process_cpu_seconds() -> f64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

fn peak_rss_kib() -> i64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    usage.ru_maxrss
}

fn platform_json(threads: usize) -> Value {
    let logical = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
    json!({
        "implementation": format!("rust altd_sim {}", env!("CARGO_PKG_VERSION")),
        "platform": format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        "logical_cpus": logical,
        "affinity_cpus": std::thread::available_parallelism().map(|n| n.get()).ok(),
        "threads": threads,
    })
}

fn agent_state(agent: &TrainingAgent) -> Value {
    let car = &agent.car;
    let metrics: Map<String, Value> =
        METRIC_NAMES.iter().zip(agent.stats.metrics()).map(|(n, v)| (n.to_string(), json!(v))).collect();
    json!({
        "position": [car.position.x, car.position.y],
        "velocity": [car.velocity.x, car.velocity.y],
        "rotation": car.rotation,
        "angular_velocity": car.angular_velocity,
        "boost_energy": car.boost_energy,
        "active": car.active,
        "collision_count": car.collision_count,
        "update_count": agent.stats.update_count,
        "metrics": metrics,
    })
}

#[allow(clippy::too_many_arguments)]
fn bench(
    scene: &Value, network_data: &Value, model: &Value, spawn_trace: &Value, population: usize, ticks: u64,
    warmup_ticks: u64, seed: i64, batch_count: usize, spawn_index: usize, mode: Mode, threads: usize,
    dump_state: Option<&Path>,
) -> Value {
    assert!(population >= 1 && ticks >= 1, "population and ticks must be positive");
    let spawn_frames = frames(spawn_trace);
    assert!(spawn_index < spawn_frames.len(), "spawn index is outside the trace");
    let settings = EvolutionSettings {
        population,
        selection_size: 14,
        mutation_rate: 0.05,
        adaptive_mutation: true,
        weight_decay: 0.0001,
        preserve_parents: "on_custom".into(),
        preserve_parents_size: 7,
        ..EvolutionSettings::default()
    };
    let load_started = Instant::now();
    let world = Arc::new(World::from_scene(scene));
    let world_seconds = load_started.elapsed().as_secs_f64();
    let spawn = &spawn_frames[spawn_index];
    let mut runner = TrainingRunner::new(
        world.clone(),
        vector(&spawn["position"]),
        num(&spawn["rotation"]),
        SensorLayout::from_exports(network_data, model),
        &output_names(network_data),
        settings.clone(),
        PyRandom::new(seed),
        batch_count,
        0,
        false,
        false,
    );
    runner.mode = mode;
    let seed_network = Network::from_game_export(network_data);

    let started = Instant::now();
    runner.start(&seed_network);
    assert_eq!(runner.agents.len(), population, "generation size differs from requested population");
    if spawn_index > 0 {
        let template = car_from_frame(&world, spawn, Some(&spawn_frames[spawn_index - 1]));
        let controls = recorded_controls(spawn);
        for agent in &mut runner.agents {
            agent.car.velocity = template.velocity;
            agent.car.angular_velocity = template.angular_velocity;
            agent.car.boost_energy = template.boost_energy;
            agent.controls = controls;
            for (wheel, source) in agent.car.wheels.iter_mut().zip(&template.wheels) {
                *wheel = *source;
            }
        }
        runner.tick = spawn_index as u64;
        runner.batch_index = (spawn_index * runner.batches_per_tick()) % 8;
    }
    let initialization_seconds = started.elapsed().as_secs_f64();

    let warmup_started = Instant::now();
    runner.advance(warmup_ticks, false);
    let warmup_seconds = warmup_started.elapsed().as_secs_f64();

    let cpu_started = process_cpu_seconds();
    let timed_started = Instant::now();
    runner.advance(ticks, false);
    let tick_seconds = timed_started.elapsed().as_secs_f64();
    let tick_cpu_seconds = process_cpu_seconds() - cpu_started;
    let active_after_ticks = runner.agents.iter().filter(|a| a.car.active).count();
    let contacts_after_ticks: u64 = runner.agents.iter().map(|a| a.car.collision_count).sum();
    let agents_state: Option<Vec<Value>> = dump_state.map(|_| runner.agents.iter().map(agent_state).collect());

    let turnover_started = Instant::now();
    runner.next_generation();
    let turnover_seconds = turnover_started.elapsed().as_secs_f64();
    assert_eq!(runner.agents.len(), population, "reproduction did not preserve population size");
    if let (Some(path), Some(agents)) = (dump_state, agents_state) {
        let networks: Vec<&Vec<f64>> = runner.agents.iter().map(|a| &a.network.params).collect();
        write_report(path, &json!({"agents": agents, "next_networks": networks}));
    }

    let simulated_seconds = ticks as f64 / 60.0;
    let complete_generation_seconds = (warmup_ticks + ticks) as f64 / 60.0;
    let total_wall_seconds = initialization_seconds + warmup_seconds + tick_seconds + turnover_seconds;
    let mut result = json!({
        "population": population,
        "timed_ticks": ticks,
        "warmup_ticks": warmup_ticks,
        "batch_count": batch_count,
        "seed": seed,
        "spawn_index": spawn_index,
        "settings": settings.to_json(),
        "physics_shapes": world.track.shapes.len(),
        "network_shape": seed_network.shape,
        "network_parameters": seed_network.params.len(),
        "active_after_ticks": active_after_ticks,
        "contacts_after_ticks": contacts_after_ticks,
        "world_load_seconds": world_seconds,
        "initialization_seconds": initialization_seconds,
        "warmup_seconds": warmup_seconds,
        "timed_wall_seconds": tick_seconds,
        "timed_cpu_seconds": tick_cpu_seconds,
        "turnover_seconds": turnover_seconds,
        "simulated_seconds": simulated_seconds,
        "realtime_multiplier": simulated_seconds / tick_seconds,
        "wall_seconds_per_simulated_second": tick_seconds / simulated_seconds,
        "car_ticks_per_wall_second": (population as u64 * ticks) as f64 / tick_seconds,
        "complete_generation_simulated_seconds": complete_generation_seconds,
        "complete_generation_wall_seconds": total_wall_seconds,
        "complete_generation_realtime_multiplier": complete_generation_seconds / total_wall_seconds,
        "peak_rss_kib": peak_rss_kib(),
        "mode": match mode { Mode::Independent => "independent", Mode::Lockstep => "lockstep" },
    });
    let map = result.as_object_mut().unwrap();
    for (k, v) in platform_json(threads).as_object().unwrap() {
        map.insert(k.clone(), v.clone());
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn train(
    scene: &Value, network_data: &Value, model: &Value, spawn_trace: &Value, settings_file: Option<&Value>,
    population: Option<usize>, generations: usize, ticks: Option<u64>, seed: i64, batch_count: usize,
    eliminate_on_wall: Option<bool>, idle_eliminate: Option<bool>, mode: Mode, threads: usize, game_rng_state: Option<&Value>,
) -> (Value, Value) {
    let empty = json!({});
    let settings_data = settings_file.unwrap_or(&empty);
    let settings_data = settings_data.get("settings").unwrap_or(settings_data);
    let mut settings = EvolutionSettings::from_mcp(settings_data);
    let population = population.unwrap_or(if settings_file.is_some() { settings.population } else { 8 });
    let ticks = ticks.unwrap_or_else(|| {
        settings_data.get("time_limit_s").map_or(300, |v| (num(v) * 60.0).round_ties_even() as u64)
    });
    let (eliminate_on_wall, idle_eliminate) = elimination_options(settings_data, eliminate_on_wall, idle_eliminate);
    assert!(population >= 1 && generations >= 1 && ticks >= 1, "population, generations, and ticks must be positive");
    settings.population = population;
    let spawn = &frames(spawn_trace)[0];
    let started = Instant::now();
    let world = Arc::new(World::from_scene(scene));
    let world_seconds = started.elapsed().as_secs_f64();
    let seed_network = Network::from_game_export(network_data);
    let mut runner = TrainingRunner::new(
        world,
        vector(&spawn["position"]),
        num(&spawn["rotation"]),
        SensorLayout::from_exports(network_data, model),
        &output_names(network_data),
        settings.clone(),
        game_rng_state.map_or_else(|| TrainingRandom::from(PyRandom::new(seed)), |v| altd_sim::game_random::GameRandom::from_json(v).into()),
        batch_count,
        0,
        eliminate_on_wall,
        idle_eliminate,
    );
    runner.mode = mode;
    runner.start(&seed_network);
    let mut history = Vec::new();
    let mut best_score = f64::NEG_INFINITY;
    let mut best_network = seed_network.clone();
    let mut simulated_ticks = 0u64;
    let mut simulate_seconds = 0.0;
    for generation in 0..generations {
        let tick_started = Instant::now();
        runner.advance_generation(ticks);
        simulate_seconds += tick_started.elapsed().as_secs_f64();
        simulated_ticks += runner.tick;
        let results = runner.results();
        let rewards = reward_values(&results, &settings.rewards);
        let first_max = |values: &[f64]| (1..values.len()).fold(0, |best, i| if values[i] > values[best] { i } else { best });
        let leader = first_max(&rewards);
        let scores: Vec<f64> = results.iter().map(|r| r.metrics[0].unwrap()).collect();
        let best_raw = first_max(&scores);
        if scores[best_raw] > best_score {
            best_score = scores[best_raw];
            best_network = results[best_raw].network.clone();
        }
        history.push(json!({
            "generation": generation,
            "ticks": runner.tick,
            "best_reward": rewards[leader],
            "best_score": scores[best_raw],
            "mean_score": py_sum(scores.iter().copied()) / scores.len() as f64,
            "total_collision_updates": py_sum(results.iter().map(|r| r.metrics[11].unwrap())),
        }));
        if generation + 1 < generations {
            runner.next_generation();
        }
    }
    let total_seconds = started.elapsed().as_secs_f64();
    let mut best = best_network.to_json();
    best["inputs"] = network_data["inputs"].clone();
    best["outputs"] = network_data["outputs"].clone();
    let output = json!({
        "seed": seed,
        "settings": settings.to_json(),
        "ticks_limit": ticks,
        "eliminate_on_wall": eliminate_on_wall,
        "idle_eliminate": idle_eliminate,
        "history": history,
        "best_score": best_score,
        "best_network": best,
    });
    let simulated_seconds = simulated_ticks as f64 / 60.0;
    let mut timing = json!({
        "world_load_seconds": world_seconds,
        "simulation_wall_seconds": simulate_seconds,
        "total_wall_seconds": total_seconds,
        "simulated_seconds_per_car": simulated_seconds,
        "realtime_multiplier": simulated_seconds / simulate_seconds,
        "complete_realtime_multiplier": simulated_seconds / total_seconds,
        "population": population,
        "mode": match mode { Mode::Independent => "independent", Mode::Lockstep => "lockstep" },
    });
    for (k, v) in platform_json(threads).as_object().unwrap() {
        timing[k] = v.clone();
    }
    (output, timing)
}

struct ScratchConfig {
    scene: PathBuf,
    spawn_trace: PathBuf,
    network: PathBuf,
    model: PathBuf,
    out_dir: PathBuf,
    shape: Vec<usize>,
    population: usize,
    generations: usize,
    ticks: u64,
    mutation_start: f64,
    mutation_end: f64,
    schedule: Schedule,
    settings: Option<PathBuf>,
    reward: Option<ScratchReward>,
    seed: i64,
    init_network: Option<PathBuf>,
    init_population: Option<PathBuf>,
    game_rng_state: Option<PathBuf>,
    batch_count: usize,
    checkpoint_every: usize,
    eliminate_on_wall: Option<bool>,
    idle_eliminate: Option<bool>,
    resume: bool,
    gpu: bool,
}

/// Install a checkpoint's population, RNG and mutation rate; returns its generation.
fn install_checkpoint(runner: &mut TrainingRunner, config: &ScratchConfig, meta: &mut Value, dir: &Path) -> u64 {
    let generation = meta["generation"].as_u64().unwrap();
    let file = dir.join(meta["population_file"].as_str().unwrap());
    let bytes = std::fs::read(&file).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
    // Old checkpoints kept their shape/population only in the adjacent run.json.
    // Copy those fields into the metadata before a resume can replace run.json.
    if meta.get("shape").is_none() || meta.get("population").is_none() {
        let saved = load(&dir.join("run.json"));
        if meta.get("shape").is_none() { meta["shape"] = saved["shape"].clone(); }
        if meta.get("population").is_none() { meta["population"] = saved["population"].clone(); }
    }
    let saved_shape: Vec<usize> = serde_json::from_value(meta["shape"].clone()).expect("checkpoint network shape");
    assert_eq!(saved_shape, config.shape, "checkpoint network shape differs from --shape");
    let saved_population = meta["population"].as_u64().expect("checkpoint population") as usize;
    assert!(saved_population > 0, "checkpoint population must be positive");
    let size = parameter_count(&saved_shape);
    let expected_bytes = saved_population.checked_mul(size).and_then(|n| n.checked_mul(8))
        .expect("checkpoint population size overflow");
    assert_eq!(bytes.len(), expected_bytes, "checkpoint population file size does not match its shape and population");
    let values: Vec<f64> = bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
    // Keep checkpoint order and RNG state. Expansion repeats saved networks;
    // subsequent reproduction uses the requested population and settings.
    let networks: Vec<Network> = (0..config.population).map(|i| {
        let start = (i % saved_population) * size;
        Network::from_vector(&saved_shape, values[start..start + size].to_vec())
    }).collect();
    runner.rng = altd_sim::training::TrainingRandom::from_json(&meta["rng"]);
    runner.settings.mutation_rate = scheduled_rate(config, generation as usize);
    runner.resume(&networks, generation);
    generation
}

/// Mutation rate used to create generation `generation`.
fn scheduled_rate(config: &ScratchConfig, generation: usize) -> f64 {
    let (start, end) = (config.mutation_start, config.mutation_end);
    if config.generations < 2 {
        return start;
    }
    let t = generation as f64 / (config.generations - 1) as f64;
    match config.schedule {
        Schedule::Geometric => start * (end / start).powf(t),
        Schedule::Linear => start + (end - start) * t,
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).unwrap_or_else(|e| panic!("{}: {e}", tmp.display()));
    std::fs::rename(&tmp, path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// `checkpoint_gNNNNN.bin`: every network's parameters as little-endian f64.
fn write_checkpoint(dir: &Path, runner: &TrainingRunner, state: Value) {
    let generation = runner.generation;
    let name = format!("checkpoint_g{generation:05}.bin");
    let mut bytes = Vec::with_capacity(runner.agents.len() * runner.agents[0].network.params.len() * 8);
    for agent in &runner.agents {
        for value in &agent.network.params {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    write_atomic(&dir.join(&name), &bytes);
    let mut meta = state;
    meta["generation"] = json!(generation);
    meta["population_file"] = json!(name);
    meta["shape"] = json!(runner.agents[0].network.shape);
    meta["population"] = json!(runner.agents.len());
    meta["rng"] = runner.rng.to_json();
    let previous = load_optional(&dir.join("checkpoint.json"));
    write_atomic(&dir.join("checkpoint.json"), (serde_json::to_string_pretty(&meta).unwrap() + "\n").as_bytes());
    if let Some(old) = previous.as_ref().and_then(|p| p["population_file"].as_str()) {
        if old != name {
            let _ = std::fs::remove_file(dir.join(old));
        }
    }
}

fn load_optional(path: &Path) -> Option<Value> {
    path.exists().then(|| load(path))
}

fn flag(yes: bool, no: bool) -> Option<bool> {
    if yes { Some(true) } else if no { Some(false) } else { None }
}

/// CLI switches override the game settings; omitted switches default to false.
fn elimination_options(settings: &Value, wall: Option<bool>, idle: Option<bool>) -> (bool, bool) {
    let truthy = |key: &str| settings.get(key).is_some_and(|v| v.as_bool().unwrap_or_else(|| num(v) != 0.0));
    (wall.unwrap_or_else(|| truthy("eliminate")), idle.unwrap_or_else(|| truthy("idle_eliminate")))
}

fn train_scratch(config: &ScratchConfig, mode: Mode, threads: usize) {
    let (scene, network_data, model) = (load(&config.scene), load(&config.network), load(&config.model));
    let shape = &config.shape;
    let inputs = network_data["inputs"].as_array().expect("inputs").len();
    let outputs = output_names(&network_data);
    assert!(
        shape.len() >= 2 && shape.iter().all(|&n| n > 0) && shape[0] == inputs && shape[shape.len() - 1] == outputs.len(),
        "shape {shape:?} must start with the {inputs} sensor inputs and end with the {} outputs",
        outputs.len()
    );
    assert!(config.population >= 1 && config.generations >= 1 && config.ticks >= 1, "population, generations, and ticks must be positive");
    assert!(config.mutation_start > 0.0 && config.mutation_end > 0.0, "mutation rates must be positive");

    // Generalist rally settings (docs/generalist-rally-design.md), optionally overridden.
    let mut settings = EvolutionSettings::from_mcp(&json!({
        "selection_algorithm": "tournament",
        "selection_size": 20,
        "crossover": "none",
        "adaptive_mutation": true,
        "weight_decay": 0.0,
        "preserve_parents": "on_custom",
        "preserve_parents_size": 4,
        "rewards": [{"metric": "total_score", "weight": 100, "type": "default"}],
    }));
    let settings_data = config.settings.as_deref().map(load).unwrap_or_else(|| json!({}));
    let settings_data = settings_data.get("settings").unwrap_or(&settings_data);
    let (eliminate_on_wall, idle_eliminate) =
        elimination_options(settings_data, config.eliminate_on_wall, config.idle_eliminate);
    if config.settings.is_some() {
        let data = settings_data;
        let mut merged = settings.to_json();
        for (k, v) in data.as_object().expect("settings object") {
            merged[k] = v.clone();
        }
        settings = EvolutionSettings::from_mcp(&merged);
    }
    if let Some(reward) = config.reward {
        settings.rewards = vec![reward.spec()];
    }
    settings.population = config.population;
    settings.mutation_rate = scheduled_rate(config, 0);

    let dir = &config.out_dir;
    let best_dir = dir.join("best_laps");
    std::fs::create_dir_all(&best_dir).unwrap_or_else(|e| panic!("{}: {e}", best_dir.display()));
    let mut run = json!({
        "simulator": "game-fidelity-cpu-v1",
        "eliminate_on_wall": eliminate_on_wall, "idle_eliminate": idle_eliminate,
        "scene": config.scene, "track": scene["track"]["name"], "spawn_trace": config.spawn_trace,
        "network_template": config.network, "model": config.model, "shape": shape,
        "population": config.population, "generations": config.generations, "ticks": config.ticks,
        "mutation_start": config.mutation_start, "mutation_end": config.mutation_end,
        "schedule": match config.schedule { Schedule::Geometric => "geometric", Schedule::Linear => "linear" },
        "seed": config.seed, "init_network": config.init_network, "batch_count": config.batch_count, "settings": settings.to_json(),
    });
    if let Some(path)=&config.game_rng_state {run["game_rng_state"]=load(path);}
    if let Some(path)=&config.init_population {run["init_population"]=json!(path);}
    let checkpoint = load_optional(&dir.join("checkpoint.json"));
    if config.resume {
        assert!(checkpoint.is_some(), "--resume: no checkpoint.json in {}", dir.display());
    } else {
        assert!(
            !dir.join("run.json").exists(),
            "{} already holds a run; pass --resume or choose another --out-dir",
            dir.display()
        );
    }

    let started = Instant::now();
    let world = Arc::new(World::from_scene(&scene));
    let spawn_trace = load(&config.spawn_trace);
    let spawn = &frames(&spawn_trace)[0];
    let restoring = config.resume || config.init_population.is_some();
    let mut rng: TrainingRandom = match &config.game_rng_state {
        Some(path) => altd_sim::game_random::GameRandom::from_json(&load(path)).into(),
        None => PyRandom::new(config.seed).into(),
    };
    let seed_network = if restoring { None } else {
        let network = match &config.init_network {
            Some(path) => Network::from_game_export(&load(path)),
            None => rng.xavier(shape),
        };
        assert_eq!(&network.shape, shape, "--init-network shape differs from --shape");
        Some(network)
    };
    let mut runner = TrainingRunner::new(
        world,
        vector(&spawn["position"]),
        num(&spawn["rotation"]),
        SensorLayout::from_exports(&network_data, &model),
        &outputs,
        settings,
        rng,
        config.batch_count,
        0,
        eliminate_on_wall,
        idle_eliminate,
    );
    runner.mode = mode;
    let gpu = config.gpu.then(|| altd_sim::gpu::Gpu::open(None).unwrap_or_else(|e| panic!("--gpu: {e}")));
    let gpu_world = gpu.as_ref().map(|g| altd_sim::gpu::GpuWorld::new(g, &runner.world));
    let mut gpu_sim = gpu_world.as_ref().map(|w| runner.gpu_sim(w, config.population).unwrap_or_else(|e| panic!("--gpu: {e}")));

    let log_path = dir.join("log.jsonl");
    let mut best_lap = f64::INFINITY;
    let mut best_lap_generation: Option<u64> = None;
    let first_generation = if let Some(mut meta) = checkpoint.filter(|_| config.resume) {
        let generation = install_checkpoint(&mut runner, config, &mut meta, dir);
        // Persist legacy metadata before run.json adopts the new options.
        write_atomic(&dir.join("checkpoint.json"), (serde_json::to_string_pretty(&meta).unwrap() + "\n").as_bytes());
        best_lap = meta["best_lap_s"].as_f64().unwrap_or(f64::INFINITY);
        best_lap_generation = meta["best_lap_generation"].as_u64();
        // Drop log lines of generations the checkpoint will redo.
        let kept: String = std::fs::read_to_string(&log_path)
            .unwrap_or_default()
            .lines()
            .filter(|line| serde_json::from_str::<Value>(line).is_ok_and(|v| v["generation"].as_u64() < Some(generation)))
            .map(|line| format!("{line}\n"))
            .collect();
        write_atomic(&log_path, kept.as_bytes());
        println!("resumed {} at generation {generation}", dir.display());
        generation as usize
    } else if let Some(path) = &config.init_population {
        let mut meta = load(path);
        let generation = install_checkpoint(&mut runner, config, &mut meta, path.parent().unwrap_or(Path::new(".")));
        std::fs::write(&log_path, "").expect("log file");
        println!("continuing the population of {} at generation {generation}", path.display());
        generation as usize
    } else {
        runner.start(seed_network.as_ref().unwrap());
        std::fs::write(&log_path, "").expect("log file");
        0
    };
    write_atomic(&dir.join("run.json"), (serde_json::to_string_pretty(&run).unwrap() + "\n").as_bytes());
    assert_eq!(runner.agents.len(), config.population, "generation size differs from requested population");
    let mut log = std::fs::OpenOptions::new().append(true).open(&log_path).expect("log file");
    println!(
        "{} on {}: population {}, {} generations x {:.1} s, mutation {} -> {} ({}), {}, setup {:.2} s",
        dir.display(),
        run["track"].as_str().unwrap_or("?"),
        config.population,
        config.generations,
        config.ticks as f64 / 60.0,
        config.mutation_start,
        config.mutation_end,
        run["schedule"].as_str().unwrap(),
        if config.gpu { "gpu".to_string() } else { format!("{threads} threads") },
        started.elapsed().as_secs_f64()
    );

    let fmt_lap = |lap: Option<f64>| lap.map_or("   -   ".to_string(), |t| format!("{t:7.2}"));
    for generation in first_generation..config.generations {
        let generation_started = Instant::now();
        let rate = runner.settings.mutation_rate;
        match (&mut gpu_sim, &gpu_world) {
            (Some(sim), Some(world)) => {
                runner.advance_generation_gpu(sim, world, config.ticks).unwrap_or_else(|e| panic!("--gpu: {e}"));
            }
            _ => {
                runner.advance_generation(config.ticks);
            }
        }
        let simulate_seconds = generation_started.elapsed().as_secs_f64();
        let simulated_ticks = runner.tick;
        let simulated_seconds = simulated_ticks as f64 / 60.0;
        let active_cars = runner.agents.iter().filter(|agent| agent.car.active).count();

        let laps: Vec<Option<f64>> =
            runner.agents.iter().map(|a| a.stats.best_lap_time.filter(|&t| t > 0.0)).collect();
        let scores: Vec<f64> = runner.agents.iter().map(|a| a.stats.total_score).collect();
        let lapped: Vec<f64> = laps.iter().flatten().copied().collect();
        // Fastest lap; ties go to the higher score, then the first car.
        let leader = (0..laps.len()).filter(|&i| laps[i].is_some()).fold(None, |best: Option<usize>, i| match best {
            Some(b) if (laps[b], -scores[b]) <= (laps[i], -scores[i]) => Some(b),
            _ => Some(i),
        });
        let generation_best = leader.map(|i| laps[i].unwrap());
        let mean_lap = (!lapped.is_empty()).then(|| py_sum(lapped.iter().copied()) / lapped.len() as f64);
        let mut sorted = lapped.clone();
        sorted.sort_by(f64::total_cmp);
        let median_lap = (!sorted.is_empty()).then(|| sorted[sorted.len() / 2]);
        let best_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean_score = py_sum(scores.iter().copied()) / scores.len() as f64;

        let mut saved = Value::Null;
        if let (Some(i), Some(lap)) = (leader, generation_best) {
            if lap < best_lap {
                best_lap = lap;
                best_lap_generation = Some(generation as u64);
                let mut export = runner.agents[i].network.to_json();
                export["inputs"] = network_data["inputs"].clone();
                export["outputs"] = network_data["outputs"].clone();
                export["training"] = json!({
                    "generation": generation, "car": i, "inference_batch": i / config.population.div_ceil(8),
                    "reused_vehicle": generation > 0, "best_lap_s": lap, "total_score": scores[i],
                    "mutation_rate": rate, "track": run["track"], "seed": config.seed,
                });
                let text = serde_json::to_string_pretty(&export).unwrap() + "\n";
                let name = format!("g{generation:05}_{lap:.2}s.json");
                write_atomic(&best_dir.join(&name), text.as_bytes());
                write_atomic(&dir.join("best.json"), text.as_bytes());
                saved = json!(format!("best_laps/{name}"));
            }
        }

        let turnover_started = Instant::now();
        let last = generation + 1 == config.generations;
        if !last {
            runner.settings.mutation_rate = scheduled_rate(config, generation + 1);
            runner.next_generation();
        }
        let turnover_seconds = turnover_started.elapsed().as_secs_f64();
        let wall = generation_started.elapsed().as_secs_f64();
        let record = json!({
            "generation": generation,
            "mutation_rate": rate,
            "best_lap_s": generation_best,
            "mean_best_lap_s": mean_lap,
            "median_best_lap_s": median_lap,
            "lapped_cars": lapped.len(),
            "best_score": best_score,
            "mean_score": mean_score,
            "all_time_best_lap_s": best_lap.is_finite().then_some(best_lap),
            "all_time_best_lap_generation": best_lap_generation,
            "saved": saved,
            "simulate_seconds": simulate_seconds,
            "simulated_ticks": simulated_ticks,
            "active_cars": active_cars,
            "turnover_seconds": turnover_seconds,
            "car_seconds_per_second": simulated_seconds * config.population as f64 / wall,
        });
        writeln!(log, "{record}").and_then(|_| log.flush()).expect("write log");
        println!(
            "gen {generation:5}  mut {rate:.5}  best lap {}  avg best lap {}  lapped {:5}/{}  score {:8.1}/{:7.1}  all-time {}{}  {wall:5.2} s",
            fmt_lap(generation_best),
            fmt_lap(mean_lap),
            lapped.len(),
            config.population,
            best_score,
            mean_score,
            fmt_lap(best_lap.is_finite().then_some(best_lap)),
            if saved.is_null() { "" } else { "  * saved" },
        );

        if !last && config.checkpoint_every > 0 && (generation + 1) % config.checkpoint_every == 0 {
            write_checkpoint(
                dir,
                &runner,
                json!({"best_lap_s": best_lap.is_finite().then_some(best_lap), "best_lap_generation": best_lap_generation}),
            );
        }
    }
    println!(
        "done: best lap {} (generation {}), {:.1} h",
        fmt_lap(best_lap.is_finite().then_some(best_lap)),
        best_lap_generation.map_or("-".into(), |g| g.to_string()),
        started.elapsed().as_secs_f64() / 3600.0
    );
}

fn main() {
    let cli = Cli::parse();
    let mut pool = rayon::ThreadPoolBuilder::new();
    if let Some(threads) = cli.threads {
        pool = pool.num_threads(threads);
    }
    pool.build_global().expect("thread pool");
    let threads = rayon::current_num_threads();
    let mode: Mode = cli.mode.into();
    match cli.command {
        Command::Bench {
            scene, network, model, spawn_trace, population, ticks, warmup_ticks, seed, spawn_index, batch_count,
            report, dump_state,
        } => {
            let result = bench(
                &load(&scene), &load(&network), &load(&model), &load(&spawn_trace), population, ticks, warmup_ticks,
                seed, batch_count, spawn_index, mode, threads, dump_state.as_deref(),
            );
            if let Some(report) = report {
                write_report(&report, &result);
            }
            println!("{}", serde_json::to_string_pretty(&result).unwrap());
        }
        Command::Train {
            scene, network, model, spawn_trace, output, settings, population, generations, ticks, seed,
            batch_count, eliminate_on_wall, no_eliminate_on_wall, idle_eliminate, no_idle_eliminate, game_rng_state,
        } => {
            let settings = settings.as_deref().map(load);
            let (result, timing) = train(
                &load(&scene), &load(&network), &load(&model), &load(&spawn_trace), settings.as_ref(), population,
                generations, ticks, seed, batch_count, flag(eliminate_on_wall, no_eliminate_on_wall),
                flag(idle_eliminate, no_idle_eliminate), mode, threads, game_rng_state.as_deref().map(load).as_ref(),
            );
            write_report(&output, &result);
            let history = result["history"].as_array().unwrap();
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "generations": history.len(),
                    "best_score": result["best_score"],
                    "last_generation": history.last(),
                    "timing": timing,
                }))
                .unwrap()
            );
        }
        Command::TrainScratch {
            scene, spawn_trace, network, model, out_dir, shape, population, generations, ticks, mutation_start,
            mutation_end, schedule, settings, reward, seed, init_network, init_population, game_rng_state, batch_count, checkpoint_every, resume,
            eliminate_on_wall, no_eliminate_on_wall, idle_eliminate, no_idle_eliminate, gpu,
        } => {
            let config = ScratchConfig {
                scene, spawn_trace, network, model, out_dir, shape, population, generations, ticks, mutation_start,
                mutation_end, schedule, settings, reward, seed, init_network, init_population, game_rng_state, batch_count, checkpoint_every, resume,
                eliminate_on_wall: flag(eliminate_on_wall, no_eliminate_on_wall),
                idle_eliminate: flag(idle_eliminate, no_idle_eliminate),
                gpu,
            };
            train_scratch(&config, mode, threads);
        }
        Command::CompareTrace { scene, trace, report, start_index, end_index } => {
            let world = World::from_scene(&load(&scene));
            let result = compare_trace(&world, &load(&trace), start_index, end_index);
            write_report(&report, &result);
            print_without(&result, "frames");
        }
        Command::CompareOneStep { scene, trace, report } => {
            let world = World::from_scene(&load(&scene));
            let result = compare_one_step(&world, &load(&trace));
            write_report(&report, &result);
            print_without(&result, "rows");
        }
        Command::CompareClosedLoop { scene, trace, network, model, report, batch_count, new_vehicle } => {
            let world = Arc::new(World::from_scene(&load(&scene)));
            let result =
                compare_closed_loop(&world, &load(&trace), &load(&network), &load(&model), batch_count, !new_vehicle, mode);
            write_report(&report, &result);
            print_without(&result, "rows");
        }
        Command::CompareNetwork { network, trace, report, first_tick } => {
            let result = compare_network(&load(&network), &load(&trace), first_tick);
            write_report(&report, &result);
            print_without(&result, "outputs");
        }
        Command::CompareSensors { scene, trace, model, trajectory_report, sensor_report, all_frames } => {
            let world = World::from_scene(&load(&scene));
            let result = compare_sensors(&world, &load(&trace), &load(&model), &load(&trajectory_report), all_frames);
            write_report(&sensor_report, &result);
            print_without(&result, "sensors");
        }
        Command::CompareScore { scene, trace, report } => {
            let world = World::from_scene(&load(&scene));
            let result = compare_score(&world, &load(&trace));
            write_report(&report, &result);
            print_without(&result, "rows");
        }
    }
}

#[cfg(test)]
mod scratch_reward_tests {
    use super::*;
    use altd_sim::{evolution::{reproduce, AgentResult}, training::TrainingStats};

    #[test]
    fn reward_choice_changes_the_preserved_parent() {
        // The distance leader has no completed lap; the fastest lap belongs to
        // the car with the least distance. Exercise real stats and reproduction.
        let networks: Vec<_> = (0..3).map(|i| Network::from_vector(&[1, 1], vec![i as f64, 0.0])).collect();
        let stats: Vec<_> = [(1000.0, None), (800.0, Some(60.0)), (600.0, Some(40.0))]
            .into_iter().map(|(distance, lap)| {
                let mut stats = TrainingStats::new(0.0);
                stats.total_score = distance;
                stats.best_lap_time = lap;
                stats.update_count = 100;
                stats
            }).collect();
        let agents: Vec<_> = networks.iter().zip(&stats).map(|(network, stats)| AgentResult {
            network, metrics: stats.metrics(), update_count: stats.update_count,
        }).collect();
        for (reward, expected) in [(ScratchReward::Distance, 0), (ScratchReward::BestLapTime, 2)] {
            let settings = EvolutionSettings {
                population: 1, selection_size: 1, preserve_parents: "on_custom".into(),
                preserve_parents_size: 1, rewards: vec![reward.spec()], ..Default::default()
            };
            let result = reproduce(&agents, &settings, &mut PyRandom::new(1));
            assert_eq!(result.preserved_count, 1);
            assert_eq!(result.networks[0].params, networks[expected].params);
        }
        let rewards = reward_values(&agents, &[ScratchReward::BestLapTime.spec()]);
        assert!(rewards[2] > rewards[1]);
        assert!(rewards[2] > rewards[0]);
    }
}
