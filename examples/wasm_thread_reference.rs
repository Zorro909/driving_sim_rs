//! Exact native observations for the threaded worker test. Local fixtures only.
//! cargo run --release --example wasm_thread_reference [SCENARIO [OUT]]
use altd_sim::session::{Session, SessionOptions};
use serde_json::{json, Value};
fn bits(values: impl IntoIterator<Item = f64>) -> Vec<String> {
    values.into_iter().map(|value| format!("{:016x}", value.to_bits())).collect()
}
fn snapshot(session: &Session) -> Value {
    json!({
        "generation": session.runner.generation, "tick": session.runner.tick,
        "states": bits(session.car_states()),
        "controls": (0..session.runner.agents.len()).map(|i| bits(session.controls(i).unwrap())).collect::<Vec<_>>(),
        "sensors": (0..session.runner.agents.len()).map(|i| bits(session.sensors(i).unwrap())).collect::<Vec<_>>(),
        "metrics": (0..session.runner.agents.len()).map(|i| bits(session.metrics(i).unwrap())).collect::<Vec<_>>(),
        "summary": session.generation_summary(),
        "checkpoint": session.checkpoint_bytes().unwrap(),
    })
}
fn main() {
    let root = env!("CARGO_MANIFEST_DIR");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let scenario = args.first().map(String::as_str).unwrap_or("wasm/test/scenario.json");
    let out = args.get(1).map(String::as_str).unwrap_or("target/wasm-thread-reference.json");
    let read = |file: &str| serde_json::from_str::<Value>(&std::fs::read_to_string(format!("{root}/{file}")).unwrap()).unwrap();
    let scenario = read(scenario);
    let shape: Vec<usize> = scenario["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let mut modes = serde_json::Map::new();
    for mode in ["independent", "lockstep"] {
        let mut options = SessionOptions::from_json(&scenario["options"].to_string()).unwrap();
        options.mode = mode.into();
        let mut session = Session::new(&read(scenario["scene"].as_str().unwrap()), &read(scenario["network"].as_str().unwrap()), &read(scenario["model"].as_str().unwrap()), options).unwrap();
        session.start_with_shape(&shape).unwrap();
        let mut steps = Vec::new();
        for generation in 0..3 {
            for ticks in [1, 5, 54, 180] {
                let executed = session.advance(ticks, false).unwrap();
                steps.push(json!({ "executed": executed, "state": snapshot(&session) }));
            }
            let (preserved, rewards) = session.next_generation().unwrap();
            steps.push(json!({ "preserved": preserved, "rewards": bits(rewards), "state": snapshot(&session) }));
            assert_eq!(session.runner.generation, generation + 1);
        }
        modes.insert(mode.into(), Value::from(steps));
    }
    std::fs::write(out, Value::Object(modes).to_string()).unwrap();
    println!("wrote {out}");
}
