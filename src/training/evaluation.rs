//! CPU evaluation of one frozen model. Each track gets a new runner and car.
use crate::{
    nn::network::Network,
    track::world::{vector, World},
    training::evolution::EvolutionSettings,
    training::pyrandom::PyRandom,
    training::{sensor_type, vision_angle, SensorLayout, TrainingRunner},
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn load(path: &Path) -> Result<(Vec<u8>, Value), String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let data = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((bytes, data))
}
fn strings(value: &Value, label: &str) -> Result<Vec<String>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("missing {label}"))?
        .iter()
        .map(|v| v.as_str().map(str::to_owned).ok_or_else(|| format!("invalid {label}")))
        .collect()
}
fn suite_file(root: &Path, row: &Value, key: &str) -> Result<Value, String> {
    let relative = row[key].as_str().ok_or_else(|| format!("missing {key}"))?;
    let path = PathBuf::from(relative);
    if path.is_absolute() || path.components().any(|p| !matches!(p, std::path::Component::Normal(_))) {
        return Err(format!("{key} must be relative to the suite directory"));
    }
    let (bytes, data) = load(&root.join(path))?;
    if row[format!("{key}_sha256")].as_str() != Some(sha256(&bytes).as_str()) {
        return Err(format!("{key} hash mismatch"));
    }
    Ok(data)
}

fn finite_model_numbers(value: &Value) -> bool {
    match value {
        Value::Number(_) => value.as_f64().is_some_and(|x| x.is_finite() && (x as f32).is_finite()),
        Value::Array(values) => values.iter().all(finite_model_numbers),
        Value::Object(values) => values.values().all(finite_model_numbers),
        _ => true,
    }
}

fn valid_vector(value: &Value) -> bool {
    value.as_array().is_some_and(|v| {
        v.len() == 2
            && v.iter()
                .all(|x| x.as_f64().is_some_and(|x| x.is_finite() && x.abs() < 100_000.0))
    })
}
fn validate_scene(scene: &Value) -> Result<(), String> {
    if !finite_model_numbers(scene) {
        return Err("scene parameters must be finite".into());
    }
    let vehicle = &scene["vehicle"];
    if !vehicle.is_object() || !scene["track"].is_object() {
        return Err("scene needs vehicle and track objects".into());
    }
    for key in [
        "mass",
        "inertia_server",
        "shape_rotation",
        "grip",
        "steering_speed",
        "air_resistance",
        "max_velocity",
        "linear_damp",
        "angular_damp",
    ] {
        let number = vehicle[key]
            .as_f64()
            .filter(|x| x.is_finite())
            .ok_or_else(|| format!("invalid vehicle {key}"))?;
        if ["mass", "inertia_server"].contains(&key) && number <= 0.0 {
            return Err(format!("vehicle {key} must be positive"));
        }
    }
    if !valid_vector(&vehicle["shape_size"])
        || vehicle["shape_size"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v.as_f64().unwrap() <= 0.0)
        || !vehicle["center_steering"].is_boolean()
    {
        return Err("invalid vehicle shape or steering".into());
    }
    for key in ["shape_position", "center_of_mass"] {
        if vehicle.get(key).is_some_and(|v| !valid_vector(v)) {
            return Err(format!("invalid vehicle {key}"));
        }
    }
    if let Some(basis) = vehicle.get("shape_basis") {
        if !basis
            .as_array()
            .is_some_and(|v| v.len() == 2 && v.iter().all(valid_vector))
        {
            return Err("invalid vehicle basis".into());
        }
    }
    let surfaces = vehicle["surfaces"].as_object().ok_or("missing vehicle surfaces")?;
    for surface in surfaces.values() {
        if !surface.is_object() {
            return Err("invalid vehicle surface".into());
        }
        for key in ["grip", "power", "steering"] {
            if surface
                .get(key)
                .is_some_and(|v| !v.as_f64().is_some_and(f64::is_finite))
            {
                return Err("invalid surface parameter".into());
            }
        }
    }
    let wheels = vehicle["wheels"]
        .as_array()
        .filter(|v| !v.is_empty())
        .ok_or("missing vehicle wheels")?;
    for wheel in wheels {
        if !valid_vector(&wheel["position"]) || !wheel["steering"].is_boolean() {
            return Err("invalid wheel pose".into());
        }
        for key in ["max_angle_deg", "power", "brake_power", "handbrake_power"] {
            if !wheel[key].as_f64().is_some_and(f64::is_finite) {
                return Err(format!("invalid wheel {key}"));
            }
        }
    }
    let track = &scene["track"];
    let path = track["path"]
        .as_array()
        .filter(|v| v.len() >= 2)
        .ok_or("missing lap path")?;
    if !path.iter().all(valid_vector) {
        return Err("invalid lap path".into());
    }
    if let Some(forward) = track.get("path_forward") {
        if !forward
            .as_array()
            .is_some_and(|v| (v.is_empty() || v.len() == path.len()) && v.iter().all(valid_vector))
        {
            return Err("path forward dimensions differ".into());
        }
    }
    for key in ["tile_map_position", "tile_size"] {
        if track.get(key).is_some_and(|v| !valid_vector(v)) {
            return Err(format!("invalid track {key}"));
        }
    }
    for key in ["polygons", "physics_shapes"] {
        if let Some(shapes) = track.get(key) {
            let shapes = shapes.as_array().ok_or("invalid track shapes")?;
            for shape in shapes {
                if !shape["points"]
                    .as_array()
                    .is_some_and(|v| v.len() >= 3 && v.iter().all(valid_vector))
                {
                    return Err("invalid polygon points".into());
                }
                for key in ["origin", "tile"] {
                    if shape.get(key).is_some_and(|v| !valid_vector(v)) {
                        return Err("invalid shape origin".into());
                    }
                }
                if let Some(local) = shape.get("local_points") {
                    if !local.as_array().is_some_and(|v| {
                        v.len() == shape["points"].as_array().unwrap().len()
                            && v.iter().all(|p| {
                                valid_vector(p)
                                    || (p.is_object()
                                        && p["x"].as_f64().is_some_and(f64::is_finite)
                                        && p["y"].as_f64().is_some_and(f64::is_finite))
                            })
                    }) {
                        return Err("invalid local polygon".into());
                    }
                }
            }
        }
    }
    if let Some(walls) = track.get("walls") {
        if !walls.as_array().is_some_and(|v| {
            v.iter()
                .all(|p| p.as_array().is_some_and(|p| p.len() == 2 && p.iter().all(valid_vector)))
        }) {
            return Err("invalid track walls".into());
        }
    }
    if let Some(tiles) = track.get("tiles") {
        let tiles = tiles.as_array().ok_or("invalid track tiles")?;
        for tile in tiles {
            if !valid_vector(&tile["coords"]) || !tile["surface"].as_str().is_some_and(|n| surfaces.contains_key(n)) {
                return Err("invalid track tile".into());
            }
            if tile
                .get("connections")
                .is_some_and(|v| !v.as_array().is_some_and(|v| v.iter().all(valid_vector)))
            {
                return Err("invalid tile connections".into());
            }
        }
    }
    if track
        .get("native_cell_order")
        .is_some_and(|v| !v.as_array().is_some_and(|v| v.iter().all(valid_vector)))
    {
        return Err("invalid native cell order".into());
    }
    if let Some(curve) = track.get("curve") {
        if !curve["bake_interval"]
            .as_f64()
            .is_some_and(|x| x.is_finite() && x > 0.0)
            || !curve["points"].as_array().is_some_and(|v| {
                v.len() >= 2
                    && v.iter().all(|p| {
                        ["position", "incoming", "outgoing"].iter().all(|key| {
                            valid_vector(&p[key])
                                || (p[key].is_object()
                                    && p[key]["x"].as_f64().is_some_and(f64::is_finite)
                                    && p[key]["y"].as_f64().is_some_and(f64::is_finite))
                        })
                    })
            })
        {
            return Err("invalid track curve".into());
        }
    }
    Ok(())
}

pub fn evaluate(
    network_path: &Path,
    model_path: &Path,
    suite_path: &Path,
    report_path: &Path,
) -> Result<Value, String> {
    let started = Instant::now();
    let (candidate_bytes, candidate) = load(network_path)?;
    let (model_bytes, model) = load(model_path)?;
    if !finite_model_numbers(&model) {
        return Err("sensor model parameters must be finite".into());
    }
    let (suite_bytes, suite) = load(suite_path)?;
    if !model["vehicle_type"]
        .as_str()
        .is_some_and(|v| ["rally", "formula", "truck", "snowmobile"].contains(&v.to_lowercase().as_str()))
    {
        return Err("invalid model vehicle".into());
    }
    if suite["version"] != 1 {
        return Err("unsupported suite version".into());
    }
    let shape = candidate["shape"].as_array().ok_or("missing shape")?;
    if !(2..=10).contains(&shape.len()) || shape.iter().any(|n| n.as_u64().is_none_or(|n| n == 0 || n > 1024)) {
        return Err("invalid network dimensions".into());
    }
    let network = Network::try_from_game_export(&candidate)?;
    if network.params.iter().any(|x| !x.is_finite()) {
        return Err("parameters must be finite".into());
    }
    let inputs = strings(&candidate["inputs"], "ordered inputs")?;
    let outputs = strings(&candidate["outputs"], "ordered outputs")?;
    if inputs.len() != network.shape[0] || outputs.len() != *network.shape.last().unwrap() {
        return Err("network and ordered I/O dimensions differ".into());
    }
    if inputs
        .iter()
        .any(|n| vision_angle(n).is_none() && sensor_type(n).is_none())
    {
        return Err("unknown sensor name".into());
    }
    for name in &inputs {
        if let Some(angle) = vision_angle(name) {
            if !angle.is_finite() || angle.fract() != 0.0 || angle.abs() > 180.0 {
                return Err("invalid ray angle".into());
            }
        }
    }
    let controls: Vec<_> = outputs.iter().map(|n| n.to_lowercase()).collect();
    if controls != strings(&model["outputs"], "model outputs")?
        || controls
            .iter()
            .any(|n| !["steering", "acceleration", "brake", "handbrake", "boost"].contains(&n.as_str()))
    {
        return Err("ordered model outputs differ from candidate".into());
    }
    if let Some(exact) = model.get("sensor_layout") {
        if SensorLayout::from_ordered(exact)?.names != inputs {
            return Err("ordered sensor names differ".into());
        }
    } else {
        let sensors: Vec<_> = inputs
            .iter()
            .filter_map(|n| sensor_type(n))
            .map(str::to_owned)
            .collect();
        // Models use snake_case sensor names; aliases describe the same ordered inputs.
        let model_sensors = strings(&model["sensors"], "model sensors")?;
        let normalized: Vec<_> = model_sensors
            .iter()
            .map(|n| {
                sensor_type(n)
                    .unwrap_or_else(|| match n.as_str() {
                        "acceleration_front" => "accelerationFront",
                        "acceleration_side" => "accelerationSide",
                        "distance_from_wall" => "distanceFromWall",
                        "angular_velocity" => "angularVelocity",
                        "boost_capacity" => "boostCapacity",
                        "correct_direction" => "correctDirection",
                        "velocity_front" => "velocityFront",
                        "velocity_side" => "velocitySide",
                        "track_curvature" => "trackCurvature",
                        "wheel_angle" => "wheelAngle",
                        _ => n,
                    })
                    .to_owned()
            })
            .collect();
        if sensors != normalized {
            return Err("ordered model sensors differ from candidate".into());
        }
        let rays = model["vision"].as_array().ok_or("missing model vision")?;
        if !rays.is_empty()
            && rays.iter().map(|r| r["angle"].as_f64()).collect::<Vec<_>>()
                != inputs
                    .iter()
                    .filter_map(|n| vision_angle(n))
                    .map(Some)
                    .collect::<Vec<_>>()
        {
            return Err("ordered model rays differ from candidate".into());
        }
        for ray in rays {
            if !ray["angle"].as_f64().is_some_and(|a| a.is_finite() && a.abs() <= 180.0)
                || !ray["length"].as_f64().is_some_and(|l| l.is_finite() && l > 0.0)
            {
                return Err("invalid model vision".into());
            }
        }
    }
    let options = &suite["options"];
    let batch_count = options["batch_count"]
        .as_u64()
        .filter(|n| (1..=8).contains(n))
        .ok_or("invalid inference batch count")? as usize;
    if options["backend"] != "cpu" {
        return Err("suite backend must be cpu".into());
    }
    let wall = options["eliminate_on_wall"]
        .as_bool()
        .ok_or("missing wall elimination rule")?;
    let idle = options["idle_eliminate"]
        .as_bool()
        .ok_or("missing idle elimination rule")?;
    let tracks = suite["tracks"]
        .as_array()
        .filter(|v| !v.is_empty())
        .ok_or("suite needs tracks")?;
    let root = suite_path.parent().unwrap_or(Path::new("."));
    // Verify all files before simulation and avoid a partial successful report.
    let mut prepared = Vec::new();
    let mut ids = std::collections::HashSet::new();
    for row in tracks {
        let id = row["id"].as_str().filter(|v| !v.is_empty()).ok_or("missing track ID")?;
        if !ids.insert(id) {
            return Err("duplicate suite track ID".into());
        }
        let scene = suite_file(root, row, "scene")?;
        let spawn = suite_file(root, row, "spawn")?;
        validate_scene(&scene)?;
        if scene["vehicle"]["type"].as_str().map(str::to_lowercase)
            != model["vehicle_type"].as_str().map(str::to_lowercase)
            || model["vehicle_type"].as_str().map(str::to_lowercase) != suite["vehicle"].as_str().map(str::to_lowercase)
        {
            return Err("suite/model vehicle mismatch".into());
        }
        let frame = spawn["frames"]
            .as_array()
            .and_then(|v| v.first())
            .ok_or("missing spawn frame")?;
        if !frame["rotation"].as_f64().is_some_and(f64::is_finite)
            || !frame["position"]
                .as_array()
                .is_some_and(|v| v.len() == 2 && v.iter().all(|x| x.as_f64().is_some_and(f64::is_finite)))
        {
            return Err("invalid spawn pose".into());
        }
        let ticks = row["ticks"]
            .as_u64()
            .filter(|n| (1..=36000).contains(n))
            .ok_or("invalid track tick cap")?;
        prepared.push((row, scene, frame.clone(), ticks));
    }
    let mut results = Vec::new();
    for (row, scene, spawn, ticks) in prepared {
        let track_started = Instant::now();
        // Malformed scene exports report an error instead of aborting the CLI.
        let world = std::panic::catch_unwind(|| World::from_scene(&scene)).map_err(|_| "invalid suite scene")?;
        let one_lap_score = world.track.path_length() * (5.0 / 384.0);
        if !one_lap_score.is_finite() || one_lap_score <= 0.0 {
            return Err("track has no lap path".into());
        }
        let layout = SensorLayout::from_exports(&candidate, &model);
        let settings = EvolutionSettings {
            population: 1,
            ..Default::default()
        };
        let mut runner = TrainingRunner::new(
            Arc::new(world),
            vector(&spawn["position"]),
            spawn["rotation"].as_f64().unwrap(),
            layout,
            &outputs,
            settings,
            PyRandom::new(0),
            batch_count,
            0,
            wall,
            idle,
        );
        runner.resume(std::slice::from_ref(&network), 0);
        runner.advance_generation(ticks);
        let stats = &runner.agents[0].stats;
        results.push(json!({"id": row["id"], "score": stats.total_score, "one_lap_score": one_lap_score,
            "normalized_progress": stats.total_score / one_lap_score,
            "lap_complete": stats.best_lap_time.is_some_and(|t| t > 0.0),
            "best_lap_s": stats.best_lap_time.filter(|&t| t > 0.0), "collisions": stats.collision_count,
            "simulated_ticks": runner.tick, "tick_cap": ticks, "elapsed_seconds": track_started.elapsed().as_secs_f64()}));
    }
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let report = json!({"version": 1, "model_sha256": sha256(&model_bytes), "candidate_sha256": sha256(&candidate_bytes), "suite_sha256": sha256(&suite_bytes),
        "simulator_sha256": sha256(&fs::read(exe).map_err(|e| e.to_string())?), "options": options,
        "tracks": results, "elapsed_seconds": started.elapsed().as_secs_f64()});
    let temp = report_path.with_extension(format!("json.{}.tmp", std::process::id()));
    fs::write(&temp, serde_json::to_vec_pretty(&report).unwrap()).map_err(|e| e.to_string())?;
    fs::rename(&temp, report_path).map_err(|e| e.to_string())?;
    Ok(report)
}
