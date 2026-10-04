//! Generated-track session state and checkpoint regression checks.
use altd_sim::training::session::{Session, SessionOptions};
use serde_json::{json, Value};
#[path = "../examples/support/generated.rs"]
mod generated;
#[path = "support/regression.rs"]
mod regression;

fn checkpoint(mode: &str, generation: u64) -> &'static [u8] {
    match (mode, generation) {
        ("independent", 1) => include_bytes!("fixtures/generated/session-independent-g1.bin"),
        ("independent", 2) => include_bytes!("fixtures/generated/session-independent-g2.bin"),
        ("lockstep", 1) => include_bytes!("fixtures/generated/session-lockstep-g1.bin"),
        ("lockstep", 2) => include_bytes!("fixtures/generated/session-lockstep-g2.bin"),
        _ => panic!("unexpected generated fixture checkpoint"),
    }
}

#[test]
fn both_session_modes_match_pre_change_generations_and_checkpoint_bytes() {
    let scene = generated::scene("rally", 0);
    let network = generated::network("rally");
    let model = generated::model("rally");
    let golden: Value = serde_json::from_str(include_str!("fixtures/generated/sessions.json")).unwrap();
    for mode in ["independent", "lockstep"] {
        let case = &golden[mode];
        let options = case["options"].to_string();
        let mut session = Session::new(&scene, &network, &model, SessionOptions::from_json(&options).unwrap()).unwrap();
        let shape: Vec<_> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap() as usize)
            .collect();
        for (index, row) in case["rows"].as_array().unwrap().iter().enumerate() {
            match row["operation"].as_str().unwrap() {
                "start" => session.start_with_shape(&shape).unwrap(),
                "advance" => {
                    assert_eq!(
                        json!(session.advance(row["ticks"].as_u64().unwrap(), false).unwrap()),
                        row["executed"]
                    );
                }
                "next_generation" => {
                    let (preserved, rewards) = session.next_generation().unwrap();
                    assert_eq!(json!(preserved), row["preserved"]);
                    assert_eq!(json!(regression::bits(rewards)), row["rewards"]);
                    let bytes = session.checkpoint_bytes().unwrap();
                    assert_eq!(bytes, checkpoint(mode, session.runner.generation), "{mode} checkpoint");
                    let mut restored =
                        Session::new(&scene, &network, &model, SessionOptions::from_json(&options).unwrap()).unwrap();
                    restored.restore_checkpoint_bytes(&bytes).unwrap();
                    assert_eq!(
                        regression::session_snapshot(&restored),
                        row["expected"],
                        "{mode} restored checkpoint"
                    );
                    assert_eq!(restored.checkpoint_bytes().unwrap(), bytes);
                }
                operation => panic!("unexpected generated fixture operation {operation}"),
            }
            assert_eq!(
                regression::session_snapshot(&session),
                row["expected"],
                "{mode} operation {index}"
            );
        }
    }
}
