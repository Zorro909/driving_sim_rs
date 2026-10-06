//! Writes target/native-reference.json: the car states of the browser test's
//! scenario (wasm/test/scenario.json) computed natively through the same
//! `training::session::Session` the WebAssembly library exposes, so `wasm/test/run.mjs`
//! can compare the WebAssembly build bit for bit.
//! Usage: cargo run --release --example wasm_reference [SCENARIO [OUT]]
//! Regenerate all public generated scenes, spawn poses and regression references:
//! cargo run --release --example wasm_reference -- --generate-fixtures
use altd_sim::nn::network::Network;
use altd_sim::track::training_tracks::{training_scene_at, RandomTrainingTrackSettings};
use altd_sim::training::pyrandom::PyRandom;
use altd_sim::training::session::{Session, SessionOptions};
use serde_json::{json, Value};
use std::path::Path;

fn load(path: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}")))
        .unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn save(path: &str, value: &Value) {
    if let Some(parent) = Path::new(path).parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, value.to_string() + "\n").unwrap_or_else(|e| panic!("{path}: {e}"));
}

fn reference(scenario_path: &str, out_path: &str) {
    let root = env!("CARGO_MANIFEST_DIR");
    let scenario = load(scenario_path);
    let file = |key: &str| load(&format!("{root}/{}", scenario[key].as_str().expect(key)));
    let options = SessionOptions::from_json(&scenario["options"].to_string()).expect("scenario options");
    let mut session = Session::new(&file("scene"), &file("network"), &file("model"), options).expect("session");
    let shape: Vec<usize> = scenario["shape"]
        .as_array()
        .expect("shape")
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    session.start_with_shape(&shape).expect("start");
    let (mut states, mut generations, mut ticks) = (Vec::new(), Vec::new(), Vec::new());
    for step in scenario["steps"].as_array().expect("steps") {
        if let Some(n) = step.get("advance").and_then(Value::as_u64) {
            ticks.push(session.advance(n, false).unwrap());
        } else if let Some(n) = step.get("advanceGeneration").and_then(Value::as_u64) {
            ticks.push(session.advance_generation(n).unwrap());
        } else if step.get("nextGeneration").is_some() {
            let (preserved_count, rewards) = session.next_generation().unwrap();
            generations.push(json!({"preservedCount": preserved_count, "rewards": rewards}));
            ticks.push(0);
        } else {
            panic!("unknown step {step}");
        }
        states.push(Value::from(session.car_states()));
    }
    let reference = json!({
        "scenario": scenario_path.strip_prefix(&format!("{root}/")).unwrap_or(scenario_path),
        "generation": session.runner.generation,
        "tick": session.runner.tick,
        "ticks": ticks,
        "generations": generations,
        "states": states,
    });
    save(out_path, &reference);
    println!(
        "wrote {out_path}: {} steps, generation {}, tick {}",
        states.len(),
        session.runner.generation,
        session.runner.tick
    );
}

fn main() {
    let root = env!("CARGO_MANIFEST_DIR");
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|arg| arg == "--generate-fixtures") {
        for (scenario_file, reference_file) in [
            ("wasm/test/scenario.json", "wasm/test/reference.json"),
            (
                "wasm/test/scenarios/rally_asphalt.json",
                "wasm/test/scenarios/rally_asphalt-reference.json",
            ),
            (
                "wasm/test/scenarios/rally_mixed.json",
                "wasm/test/scenarios/rally_mixed-reference.json",
            ),
            (
                "wasm/test/scenarios/rally_ice.json",
                "wasm/test/scenarios/rally_ice-reference.json",
            ),
        ] {
            let path = format!("{root}/{scenario_file}");
            let mut scenario = load(&path);
            let recipe = &scenario["generated"];
            let settings: RandomTrainingTrackSettings = serde_json::from_value(recipe["settings"].clone()).unwrap();
            settings.validate().unwrap();
            let (_, scene) = training_scene_at(
                &load(&format!("{root}/{}", recipe["template"].as_str().unwrap())),
                &settings,
                None,
                recipe["seed"].as_i64().unwrap(),
                recipe["generation"].as_u64().unwrap(),
                recipe["slot"].as_u64().unwrap() as usize,
                altd_sim::math::profile::MathProfile::Proton,
            )
            .unwrap();
            let scene_file = scenario["scene"].as_str().unwrap();
            save(&format!("{root}/{scene_file}"), &scene);
            let spawn = json!({"position": scene["reset_position"], "rotation": scene["reset_rotation"]});
            save(
                &format!("{root}/{}", scene_file.replace(".scene.json", ".spawn.json")),
                &json!({"frames": [spawn]}),
            );
            scenario["options"]["spawn"] = spawn;
            save(&path, &scenario);
            reference(&path, &format!("{root}/{reference_file}"));
        }
        let mut network = Network::xavier(&[20, 16, 16, 16, 16, 12, 12, 8, 5], &mut PyRandom::new(1)).to_json();
        let names = load(&format!("{root}/assets/networks/rally.json"));
        network["inputs"] = names["inputs"].clone();
        network["outputs"] = names["outputs"].clone();
        save(&format!("{root}/wasm/test/scenarios/benchmark.network.json"), &network);
    } else {
        let scenario_path = args
            .first()
            .cloned()
            .unwrap_or_else(|| format!("{root}/wasm/test/scenario.json"));
        let out_path = args
            .get(1)
            .cloned()
            .unwrap_or_else(|| format!("{root}/target/native-reference.json"));
        reference(&scenario_path, &out_path);
    }
}
