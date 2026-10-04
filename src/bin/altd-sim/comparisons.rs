//! Recorded trajectory, controller, sensor, and score comparisons.

use super::inputs::{car_from_frame, frames, num, output_names, recorded_controls};
use altd_sim::math::pymath::{py_max, py_min, py_pow, py_remainder, py_sum};
use altd_sim::nn::network::Network;
use altd_sim::physics::car::{Sensor, SensorScratch, DT};
use altd_sim::track::world::{vector, World};
use altd_sim::training::evolution::EvolutionSettings;
use altd_sim::training::pyrandom::PyRandom;
use altd_sim::training::{Mode, ScoreTracker, SensorLayout, TrainingRunner};
use serde_json::{json, Map, Value};
use std::f64::consts::PI;
use std::sync::Arc;

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
        out.insert(
            field.into(),
            json!({"max": py_max_of(values.iter().copied()), "rmse": rmse}),
        );
    }
    Value::Object(out)
}

pub(super) fn compare_trace(world: &World, trace: &Value, start_index: usize, end_index: Option<usize>) -> Value {
    const LIMITS: [(&str, f64); 3] = [
        ("position_error_px", 0.05),
        ("velocity_error_px_s", 1.0),
        ("rotation_error_rad", 0.001),
    ];
    let frames = frames(trace);
    let first = &frames[start_index];
    let previous = if start_index > 0 {
        Some(&frames[start_index - 1])
    } else {
        None
    };
    let mut car = car_from_frame(world, first, previous);
    let baseline = first["collision_count"].as_i64().expect("collision_count");
    let end_index = end_index.unwrap_or(frames.len());
    assert!(
        start_index < end_index && end_index <= frames.len(),
        "start_index and end_index must delimit trace frames"
    );
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
    let game_increase = rows
        .iter()
        .find(|r| r["game_contact_count"].as_i64().unwrap() > 0)
        .map(tick_of);
    let first_contact = rows.iter().find(|r| r["python_contact"] == true).map(tick_of);
    let divergence = rows
        .iter()
        .find(|r| LIMITS.iter().any(|(name, limit)| num(&r[*name]) > *limit))
        .map(tick_of);
    let agreement: Vec<Value> = rows
        .iter()
        .filter(|r| divergence.is_none_or(|stop| tick_of(r) < stop))
        .cloned()
        .collect();
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

pub(super) fn compare_one_step(world: &World, trace: &Value) -> Value {
    let frames = frames(trace);
    let has_contact_data = frames[0].get("physics_contacts").is_some();
    let mut rows = Vec::new();
    for index in 2..frames.len() {
        let (previous, state, expected) = (&frames[index - 2], &frames[index - 1], &frames[index]);
        let mut car = altd_sim::physics::trace_state::car_from_frame(
            world,
            state,
            Some(previous),
            index.checked_sub(3).map(|i| &frames[i]),
        );
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
            check(
                &format!("wheel_angles.{i}"),
                wheel.angle_deg,
                &expected["wheel_angles"][i],
            );
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
    let fields = [
        "position_error_px",
        "velocity_error_px_s",
        "rotation_error_rad",
        "angular_velocity_error_rad_s",
    ];
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
    world: &Arc<World>,
    first: &Value,
    network_data: &Value,
    model: &Value,
    batch_count: usize,
    stats_phase: u64,
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

pub(super) fn compare_closed_loop(
    world: &Arc<World>,
    trace: &Value,
    network_data: &Value,
    model: &Value,
    batch_count: usize,
    reused: bool,
    mode: Mode,
) -> Value {
    let frames = frames(trace);
    let first = &frames[0];
    let score_change_ticks: Vec<usize> = (1..frames.len())
        .filter(|&i| frames[i]["score"] != frames[i - 1]["score"])
        .collect();
    // Counter(...).most_common(1): highest count, first-seen residue on ties.
    let mut counts: Vec<(usize, usize)> = Vec::new();
    for &tick in &score_change_ticks {
        match counts.iter_mut().find(|(r, _)| *r == tick % 6) {
            Some(entry) => entry.1 += 1,
            None => counts.push((tick % 6, 1)),
        }
    }
    let stats_phase = counts
        .iter()
        .fold(None, |best: Option<(usize, usize)>, &(r, c)| match best {
            Some((_, bc)) if bc >= c => best,
            _ => Some((r, c)),
        });
    let stats_phase = stats_phase.map_or(0, |(r, _)| r) as u64;
    let valid: std::collections::HashSet<usize> = score_change_ticks.iter().skip(2).copied().collect();
    let mut runner = closed_loop_runner(world, first, network_data, model, batch_count, stats_phase, mode);
    runner.start(&Network::from_game_export(network_data));
    if reused || first.get("reset_transform").and_then(Value::as_bool) == Some(true) {
        runner.agents[0].car.reset(
            world,
            vector(&first["position"]),
            num(first.get("reset_rotation").unwrap_or(&first["rotation"])),
            true,
        );
    }
    // RecordTrace resets the vehicle, not the running inference batch cursor.
    // Recover its phase from the first observed output update.
    let inference_phase = frames
        .windows(2)
        .position(|f| f[0]["outputs"] != f[1]["outputs"])
        .unwrap_or(0)
        % batch_count;
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
    let fields = [
        "position_error_px",
        "velocity_error_px_s",
        "rotation_error_rad",
        "output_error",
        "effective_output_error",
    ];
    let score_summary = |rows: &[Value]| -> Value {
        let values: Vec<f64> = rows
            .iter()
            .filter_map(|r| r.get("score_absolute_error").map(num))
            .collect();
        if values.is_empty() {
            return json!({});
        }
        let rmse = (py_sum(values.iter().map(|v| v * v)) / values.len() as f64).sqrt();
        json!({"updates": values.len(), "max": py_max_of(values.iter().copied()), "rmse": rmse})
    };
    let first_over = |limit: f64| {
        opt_tick(
            rows.iter()
                .find(|r| num(&r["position_error_px"]) > limit)
                .map(|r| r["tick"].as_u64().unwrap() as usize),
        )
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

pub(super) fn compare_network(network_data: &Value, trace: &Value, first_tick: usize) -> Value {
    let network = Network::from_game_export(network_data);
    let frames = frames(trace);
    let input_names: Vec<&str> = network_data["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let output_names = output_names(network_data);
    let mut per_output: Vec<Vec<f64>> = vec![Vec::new(); output_names.len()];
    let mut worst = (0.0, Value::Null);
    for frame in frames.iter().skip(first_tick) {
        let sensors = frame["sensors"].as_array().unwrap();
        let outputs = frame["outputs"].as_array().unwrap();
        assert!(
            sensors
                .iter()
                .map(|s| s["name"].as_str().unwrap())
                .eq(input_names.iter().copied()),
            "trace sensor order differs from live network"
        );
        assert!(
            outputs
                .iter()
                .map(|s| s["name"].as_str().unwrap())
                .eq(output_names.iter().map(String::as_str)),
            "trace output order differs from live network"
        );
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
        summaries.insert(
            name.clone(),
            json!({"max": py_max_of(errors.iter().copied()), "rmse": rmse}),
        );
    }
    json!({
        "network_shape": network.shape,
        "compared_frames": frames.len() as i64 - first_tick as i64,
        "max_error": worst.0,
        "worst_tick": worst.1,
        "outputs": summaries,
    })
}

pub(super) fn compare_sensors(
    world: &World,
    trace: &Value,
    model: &Value,
    trajectory: &Value,
    all_frames: bool,
) -> Value {
    const ALL_KINDS: [(&str, &str); 8] = [
        ("AngVel", "angularVelocity"),
        ("Boost", "boostCapacity"),
        ("Dir", "correctDirection"),
        ("Grip", "grip"),
        ("VelF", "velocityFront"),
        ("VelS", "velocitySide"),
        ("Curve", "trackCurvature"),
        ("Wheel", "wheelAngle"),
    ];
    let frames = frames(trace);
    // Only the sensors this trace recorded (rally vs. formula layouts differ).
    let has_sensor = |name: &str| {
        frames.iter().any(|f| {
            f["sensors"]
                .as_array()
                .is_some_and(|s| s.iter().any(|i| i["name"] == name))
        })
    };
    let kinds: Vec<(&str, &str)> = ALL_KINDS.iter().copied().filter(|(n, _)| has_sensor(n)).collect();
    let source_index = |tick: usize, parity: usize| if tick % 2 == parity { tick - 1 } else { tick - 2 };
    let actual_of = |frame: &Value, name: &str| -> f64 {
        // {item["name"]: item["value"]}: the last duplicate wins.
        frame["sensors"]
            .as_array()
            .unwrap()
            .iter()
            .rev()
            .find(|i| i["name"] == name)
            .map(|i| num(&i["value"]))
            .unwrap_or_else(|| panic!("missing sensor {name}"))
    };
    let first_tick = 2.max(trajectory["start_index"].as_u64().unwrap() as usize + 1);
    let stop_tick = if all_frames {
        frames.len()
    } else {
        let end = trajectory
            .get("end_index")
            .map_or(frames.len(), |v| v.as_u64().unwrap() as usize);
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
    let mut vision: Vec<(String, f64, f64)> = model["vision"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| (item["angle"].to_string(), num(&item["angle"]), num(&item["length"])))
        .collect();
    if vision.is_empty() {
        // model.json without vision: rays come from the trace sensor names, lengths from the game formula.
        for item in frames
            .iter()
            .find(|f| f["sensors"].as_array().is_some_and(|s| !s.is_empty()))
            .map_or(&[][..], |f| f["sensors"].as_array().unwrap())
        {
            if let Some(label) = item["name"]
                .as_str()
                .and_then(|n| n.strip_prefix("↑ "))
                .and_then(|n| n.strip_suffix('°'))
            {
                let degrees: f64 = label.parse().unwrap();
                vision.push((
                    label.to_string(),
                    degrees,
                    altd_sim::training::vision_length(degrees as f32) as f64,
                ));
            }
        }
    }
    let mut errors: Vec<(String, Vec<f64>)> = kinds.iter().map(|(n, _)| (n.to_string(), Vec::new())).collect();
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
        let car = altd_sim::physics::trace_state::car_from_frame(
            world,
            &frames[index],
            index.checked_sub(1).map(|i| &frames[i]),
            index.checked_sub(2).map(|i| &frames[i]),
        );
        scratch.invalidate();
        for &(name, kind) in &kinds {
            let value = car.sensor(world, Sensor::by_name(kind), &mut scratch) - actual_of(&frames[tick], name);
            push(&mut errors, name, value);
        }
        for (label, degrees, length) in &vision {
            let predicted = car.sensor(
                world,
                Sensor::Raycast {
                    degrees: *degrees,
                    length: *length,
                },
                &mut scratch,
            );
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
    let vision_errors: Vec<f64> = errors
        .iter()
        .filter(|(n, _)| n.starts_with("vision_"))
        .flat_map(|(_, v)| v.iter().copied())
        .collect();
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

pub(super) fn compare_score(world: &World, trace: &Value) -> Value {
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
