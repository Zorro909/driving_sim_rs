//! Generated-track session state and checkpoint regression checks.
use altd_sim::math::profile::MathProfile;
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

/// The options of a case on the math the reference runner and the
/// snapshots use, whatever the host's default.
fn proton(options: &str) -> SessionOptions {
    let mut options = SessionOptions::from_json(options).unwrap();
    options.math_profile = Some(MathProfile::Proton);
    options
}

#[test]
fn both_session_modes_match_pre_change_generations_and_checkpoint_bytes() {
    let scene = generated::scene("rally", 0);
    let network = generated::network("rally");
    let model = generated::model("rally");
    let golden: Value = serde_json::from_str(include_str!("fixtures/generated/sessions.json")).unwrap();
    // These snapshots and bytes were captured on Linux x86_64 GNU. Other
    // hosts can round Gaussian/transcendental operations differently, so all
    // targets also compare Session against the legacy runner on the same host.
    let captured_platform = cfg!(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"));
    for mode in ["independent", "lockstep"] {
        let case = &golden[mode];
        let options = case["options"].to_string();
        let mut session = Session::new(&scene, &network, &model, proton(&options)).unwrap();
        // Use Session only to expose the runner's state through the same views.
        // Its runner follows the public, untraced pre-Session lifecycle below.
        let mut reference = Session::new(&scene, &network, &model, proton(&options)).unwrap();
        let mut replay: Option<Session> = None;
        let shape: Vec<_> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap() as usize)
            .collect();
        for (index, row) in case["rows"].as_array().unwrap().iter().enumerate() {
            match row["operation"].as_str().unwrap() {
                "start" => {
                    session.start_with_shape(&shape).unwrap();
                    let seed = reference
                        .runner
                        .rng
                        .xavier(&shape, altd_sim::math::profile::MathProfile::Proton);
                    reference.runner.start(&seed);
                }
                "advance" => {
                    let ticks = row["ticks"].as_u64().unwrap();
                    let executed = session.advance(ticks, false).unwrap();
                    assert_eq!(executed, reference.runner.advance(ticks, false));
                    assert_eq!(json!(executed), row["executed"]);
                    if let Some(replay) = &mut replay {
                        assert_eq!(replay.advance(ticks, false).unwrap(), executed);
                    }
                }
                "next_generation" => {
                    let (preserved, rewards) = session.next_generation().unwrap();
                    let reward_bits = regression::bits(rewards);
                    let expected = reference.runner.next_generation();
                    assert_eq!(preserved, expected.preserved_count);
                    assert_eq!(reward_bits, regression::bits(expected.rewards));
                    if captured_platform {
                        assert_eq!(json!(preserved), row["preserved"]);
                        assert_eq!(json!(&reward_bits), row["rewards"]);
                    }
                    let bytes = session.checkpoint_bytes().unwrap();
                    if captured_platform {
                        assert_eq!(bytes, checkpoint(mode, session.runner.generation), "{mode} checkpoint");
                    }
                    if let Some(replay) = &mut replay {
                        let (replay_preserved, replay_rewards) = replay.next_generation().unwrap();
                        assert_eq!(replay_preserved, preserved);
                        assert_eq!(regression::bits(replay_rewards), reward_bits);
                        assert_eq!(replay.checkpoint_bytes().unwrap(), bytes, "{mode} replay checkpoint");
                    }
                    let mut restored = Session::new(&scene, &network, &model, proton(&options)).unwrap();
                    restored.restore_checkpoint_bytes(&bytes).unwrap();
                    assert_eq!(
                        regression::session_snapshot(&restored),
                        regression::session_snapshot(&reference),
                        "{mode} restored checkpoint"
                    );
                    assert_eq!(restored.checkpoint_bytes().unwrap(), bytes);
                    replay = Some(restored);
                }
                operation => panic!("unexpected generated fixture operation {operation}"),
            }
            assert_eq!(
                regression::session_snapshot(&session),
                regression::session_snapshot(&reference),
                "{mode} legacy runner operation {index}"
            );
            if captured_platform {
                assert_eq!(
                    regression::session_snapshot(&session),
                    row["expected"],
                    "{mode} captured operation {index}"
                );
            }
            if let Some(replay) = &replay {
                assert_eq!(
                    regression::session_snapshot(replay),
                    regression::session_snapshot(&reference),
                    "{mode} checkpoint replay operation {index}"
                );
            }
        }
    }
}
