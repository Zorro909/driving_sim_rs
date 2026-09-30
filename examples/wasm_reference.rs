//! Writes wasm/test/reference.json: the car states of the browser test's
//! scenario (wasm/test/scenario.json) computed natively through the same
//! `session::Session` the WebAssembly library exposes, so `wasm/test/run.mjs`
//! can compare the WebAssembly build bit for bit.
//! Usage: cargo run --release --example wasm_reference [SCENARIO [OUT]]
use altd_sim::session::{Session, SessionOptions};
use serde_json::{json, Value};

fn load(path: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"))).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn main() {
    let root = env!("CARGO_MANIFEST_DIR");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let scenario_path = args.first().cloned().unwrap_or_else(|| format!("{root}/wasm/test/scenario.json"));
    let out_path = args.get(1).cloned().unwrap_or_else(|| format!("{root}/wasm/test/reference.json"));
    let scenario = load(&scenario_path);
    let file = |key: &str| load(&format!("{root}/{}", scenario[key].as_str().expect(key)));
    let options = SessionOptions::from_json(&scenario["options"].to_string()).expect("scenario options");
    let mut session = Session::new(&file("scene"), &file("network"), &file("model"), options).expect("session");
    let shape: Vec<usize> = scenario["shape"].as_array().expect("shape").iter().map(|v| v.as_u64().unwrap() as usize).collect();
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
        "scenario": scenario_path.strip_prefix(&format!("{root}/")).unwrap_or(&scenario_path),
        "generation": session.runner.generation,
        "tick": session.runner.tick,
        "ticks": ticks,
        "generations": generations,
        "states": states,
    });
    std::fs::write(&out_path, reference.to_string() + "\n").unwrap_or_else(|e| panic!("{out_path}: {e}"));
    println!("wrote {out_path}: {} steps, generation {}, tick {}", states.len(), session.runner.generation, session.runner.tick);
}
