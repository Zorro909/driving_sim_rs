use altd_sim::network::Network;
use serde_json::{json, Value};
use std::{fs, path::{Path, PathBuf}, process::{Command, Output}};

struct Workspace(PathBuf);
impl Workspace {
    fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!("altd-scratch-test-{}-{tag}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Workspace {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}
fn load(path: impl AsRef<Path>) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn run(root: &Path, name: &str, extra: &[&str]) -> Output {
    let defaults = [("--population", "1"), ("--generations", "2"), ("--ticks", "600"),
        ("--shape", "20,5"), ("--mutation-start", "0.000000001"),
        ("--mutation-end", "0.000000001"), ("--checkpoint-every", "1")];
    let mut command = Command::new(env!("CARGO_BIN_EXE_altd-sim"));
    command.args(["--threads", "1", "train-scratch"]);
    for (key, value) in defaults {
        if !extra.contains(&key) { command.args([key, value]); }
    }
    command
        .arg("--out-dir").arg(root.join(name))
        .arg("--init-network").arg(root.join("seed.json"))
        .arg("--settings").arg(root.join("settings.json"))
        .args(extra).output().unwrap()
}
fn success(output: Output) {
    assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
}
fn logs(root: &Path, name: &str) -> Vec<Value> {
    fs::read_to_string(root.join(name).join("log.jsonl")).unwrap().lines()
        .map(|line| serde_json::from_str(line).unwrap()).collect()
}
/// Writes a stationary 20,5 seed network and the given settings file.
fn seed_workspace(root: &Path, settings: &Value) {
    let template = load(concat!(env!("CARGO_MANIFEST_DIR"), "/../docs/traces/rally_a01_live_network.json"));
    let mut seed = Network::from_vector(&[20, 5], vec![0.0; 105]).to_json();
    seed["inputs"] = template["inputs"].clone();
    seed["outputs"] = template["outputs"].clone();
    fs::write(root.join("seed.json"), seed.to_string()).unwrap();
    fs::write(root.join("settings.json"), json!({"settings": settings}).to_string()).unwrap();
}
fn stop_reason(root: &Path, name: &str) -> Value {
    load(root.join(name).join("checkpoint.json"))["stop_reason"].clone()
}

#[test]
fn scratch_elimination_settings_overrides_and_resume() {
    let workspace = Workspace::new("settings");
    let root = &workspace.0;
    // Preserve the stationary seed so both generations exercise idle elimination.
    let settings = json!({"selection_size": 1, "preserve_parents_size": 1,
        "eliminate": true, "idle_eliminate": 1});
    seed_workspace(root, &settings);

    success(run(root, "settings", &[]));
    let config = load(root.join("settings/run.json"));
    assert_eq!(config["eliminate_on_wall"], true);
    assert_eq!(config["idle_eliminate"], true);
    assert_eq!(config["settings"]["rewards"][0]["metric"], "total_score");
    let before = logs(root, "settings");
    assert_eq!(before.len(), 2);
    for row in &before {
        assert_eq!(row["active_cars"], 0);
        assert_eq!(row["simulated_ticks"], 486);
    }
    // The final checkpoint follows generation 1, so resume continues at generation 2.
    assert_eq!(load(root.join("settings/checkpoint.json"))["generation"], 2);
    success(run(root, "settings", &["--resume", "--generations", "3"]));
    let after = logs(root, "settings");
    assert_eq!(after.len(), 3);
    assert_eq!(after[2]["generation"], 2);
    for (a, b) in before.iter().zip(&after) {
        for key in ["generation", "best_score", "mean_score", "active_cars", "simulated_ticks"] {
            assert_eq!(a[key], b[key], "resume {key}");
        }
    }
    // Resume accepts elimination, population, time-limit, and schedule changes.
    // Make this a legacy checkpoint to exercise metadata migration as well.
    let checkpoint_path = root.join("settings/checkpoint.json");
    let mut legacy = load(&checkpoint_path);
    legacy.as_object_mut().unwrap().remove("shape");
    legacy.as_object_mut().unwrap().remove("population");
    fs::write(&checkpoint_path, legacy.to_string()).unwrap();
    // No final checkpoint here, so later resumes still read the migrated one.
    success(run(root, "settings", &["--resume", "--no-idle-eliminate", "--population", "3", "--generations", "4",
        "--ticks", "120", "--mutation-start", "0.2", "--mutation-end", "0.1", "--checkpoint-every", "0"]));
    let changed = load(root.join("settings/run.json"));
    assert_eq!(changed["population"], 3);
    assert_eq!(changed["idle_eliminate"], false);
    let changed_logs = logs(root, "settings");
    assert_eq!(changed_logs.len(), 4);
    assert_eq!(changed_logs[3]["active_cars"], 3);
    assert_eq!(changed_logs[3]["simulated_ticks"], 126);
    assert_eq!(changed_logs[3]["mutation_rate"], 0.1);
    let migrated = load(&checkpoint_path);
    assert_eq!(migrated["shape"], json!([20, 5]));
    assert_eq!(migrated["population"], 1, "metadata describes the saved binary, not resized agents");
    // A second resume must still decode the original checkpoint after run.json changed.
    success(run(root, "settings", &["--resume", "--generations", "4", "--checkpoint-every", "0"]));
    assert_eq!(logs(root, "settings")[3]["active_cars"], 0);

    let before_rejection = logs(root, "settings");
    let before_config = load(root.join("settings/run.json"));
    let rejected = run(root, "settings", &["--resume", "--shape", "20,1,5"]);
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("checkpoint network shape differs"));
    assert_eq!(logs(root, "settings"), before_rejection);
    assert_eq!(load(root.join("settings/run.json")), before_config);
    // Reject an incompatible layout even when it has the same parameter count.
    let mut wrong_shape = migrated.clone();
    wrong_shape["shape"] = json!([14, 7]); // 105 parameters, like [20, 5].
    fs::write(&checkpoint_path, wrong_shape.to_string()).unwrap();
    let rejected = run(root, "settings", &["--resume"]);
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("checkpoint network shape differs"));
    fs::write(&checkpoint_path, migrated.to_string()).unwrap();
    // A truncated population is rejected before replacing the run configuration.
    let population_path = root.join("settings").join(migrated["population_file"].as_str().unwrap());
    let mut bytes = fs::read(&population_path).unwrap();
    bytes.pop();
    fs::write(&population_path, bytes).unwrap();
    let rejected = run(root, "settings", &["--resume"]);
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("population file size"));
    assert_eq!(load(root.join("settings/run.json")), before_config);

    success(run(root, "disabled", &["--no-idle-eliminate", "--no-eliminate-on-wall"]));
    let config = load(root.join("disabled/run.json"));
    assert_eq!(config["eliminate_on_wall"], false);
    assert_eq!(config["idle_eliminate"], false);
    for row in logs(root, "disabled") {
        assert_eq!(row["active_cars"], 1);
        assert_eq!(row["simulated_ticks"], 606);
    }
    // Plain settings objects work too; positive switches override disabled settings.
    let mut settings = settings;
    settings["eliminate"] = json!(false);
    settings["idle_eliminate"] = json!(false);
    fs::write(root.join("settings.json"), settings.to_string()).unwrap();
    success(run(root, "enabled", &["--idle-eliminate", "--eliminate-on-wall"]));
    let config = load(root.join("enabled/run.json"));
    assert_eq!(config["eliminate_on_wall"], true);
    assert_eq!(config["idle_eliminate"], true);
    for row in logs(root, "enabled") {
        assert_eq!(row["active_cars"], 0);
        assert_eq!(row["simulated_ticks"], 486);
    }
    // Switching rewards on resume changes the effective reproduction settings.
    success(run(root, "enabled", &["--resume", "--reward", "best-lap-time"]));
    let lap_reward = json!([{"metric": "best_lap_performance", "weight": 100, "type": "default"}]);
    assert_eq!(load(root.join("enabled/run.json"))["settings"]["rewards"], lap_reward);
    let mut settings = load(root.join("settings.json"));
    settings["rewards"] = lap_reward.clone();
    fs::write(root.join("settings.json"), settings.to_string()).unwrap();
    success(run(root, "enabled", &["--resume"]));
    assert_eq!(load(root.join("enabled/run.json"))["settings"]["rewards"], lap_reward);
    success(run(root, "enabled", &["--resume", "--reward", "distance"]));
    assert_eq!(load(root.join("enabled/run.json"))["settings"]["rewards"],
        json!([{"metric": "total_score", "weight": 100, "type": "default"}]));
}

/// Fields that differ between otherwise identical runs.
fn without_timing(mut row: Value) -> Value {
    for key in ["simulate_seconds", "turnover_seconds", "car_seconds_per_second"] {
        row.as_object_mut().unwrap().remove(key);
    }
    row
}

#[test]
fn scratch_final_checkpoint_continues_exactly() {
    let workspace = Workspace::new("continue");
    let root = &workspace.0;
    seed_workspace(root, &json!({"selection_size": 2, "preserve_parents_size": 1}));
    let common = ["--population", "4", "--ticks", "120", "--mutation-start", "0.2", "--mutation-end", "0.2"];
    success(run(root, "whole", &[&common[..], &["--generations", "3"]].concat()));
    success(run(root, "split", &[&common[..], &["--generations", "2"]].concat()));
    assert_eq!(stop_reason(root, "split"),
        json!({"condition": "generations", "generation": 1, "value": 2, "threshold": 2}));
    success(run(root, "split", &[&common[..], &["--generations", "3", "--resume"]].concat()));

    let whole: Vec<Value> = logs(root, "whole").into_iter().map(without_timing).collect();
    let split: Vec<Value> = logs(root, "split").into_iter().map(without_timing).collect();
    assert_eq!(whole.len(), 3);
    assert_eq!(whole, split);
    let (a, b) = (load(root.join("whole/checkpoint.json")), load(root.join("split/checkpoint.json")));
    assert_eq!(a["generation"], 3);
    assert_eq!(a, b);
    assert_eq!(fs::read(root.join("whole/checkpoint_g00003.bin")).unwrap(),
        fs::read(root.join("split/checkpoint_g00003.bin")).unwrap());
    assert!(!root.join("split/checkpoint_g00002.bin").exists(), "old population file removed");
}

#[test]
fn scratch_stop_conditions() {
    let workspace = Workspace::new("stop");
    let root = &workspace.0;
    // The stationary seed never laps and always scores the same.
    seed_workspace(root, &json!({"selection_size": 1, "preserve_parents_size": 1}));
    let many = ["--generations", "50"];

    success(run(root, "score", &[&many[..], &["--stop-score-above=-1000"]].concat()));
    assert_eq!(logs(root, "score").len(), 1);
    let reason = stop_reason(root, "score");
    assert_eq!(reason["condition"], "score_above");
    assert_eq!(reason["generation"], 0);
    assert_eq!(reason["threshold"], -1000.0);
    assert_eq!(load(root.join("score/checkpoint.json"))["generation"], 1);
    assert_eq!(load(root.join("score/run.json"))["stop"]["score_above"], -1000.0);

    success(run(root, "lapped", &[&many[..], &["--stop-lapped-percent", "0"]].concat()));
    assert_eq!(stop_reason(root, "lapped")["condition"], "lapped_percent");
    assert_eq!(stop_reason(root, "lapped")["value"], 0.0);

    // Unmet rules leave the generation target in charge.
    success(run(root, "unmet", &["--stop-lap-below", "1000", "--stop-score-above", "1e9", "--stop-lapped-percent", "50"]));
    assert_eq!(logs(root, "unmet").len(), 2);
    assert_eq!(stop_reason(root, "unmet")["condition"], "generations");

    // Score plateau: generation 0 improves on nothing, 1 and 2 are stale.
    success(run(root, "plateau-score", &[&many[..], &["--stop-plateau", "2", "--plateau-metric", "score"]].concat()));
    assert_eq!(logs(root, "plateau-score").len(), 3);
    assert_eq!(stop_reason(root, "plateau-score"),
        json!({"condition": "plateau", "generation": 2, "value": 2, "threshold": 2, "metric": "score"}));
    // Lap plateau: generations without a lap never improve.
    success(run(root, "plateau-lap", &[&many[..], &["--stop-plateau", "2"]].concat()));
    assert_eq!(logs(root, "plateau-lap").len(), 2);
    assert_eq!(stop_reason(root, "plateau-lap")["metric"], "lap");
    // Plateau counts restart with each invocation.
    success(run(root, "plateau-lap", &["--resume", "--generations", "50", "--stop-plateau", "2"]));
    assert_eq!(logs(root, "plateau-lap").len(), 4);
    assert_eq!(stop_reason(root, "plateau-lap")["generation"], 3);

    // Without checkpoints a stop still ends the run, just without saving.
    success(run(root, "nockpt", &[&many[..], &["--stop-score-above=-1000", "--checkpoint-every", "0"]].concat()));
    assert_eq!(logs(root, "nockpt").len(), 1);
    assert!(!root.join("nockpt/checkpoint.json").exists());
}

#[test]
fn scratch_stop_request_file() {
    let workspace = Workspace::new("request");
    let root = &workspace.0;
    seed_workspace(root, &json!({"selection_size": 1, "preserve_parents_size": 1}));
    // A request left from an earlier invocation is discarded at startup.
    fs::create_dir(root.join("stale")).unwrap();
    fs::write(root.join("stale/stop_request"), "").unwrap();
    success(run(root, "stale", &[]));
    assert_eq!(logs(root, "stale").len(), 2);
    assert!(!root.join("stale/stop_request").exists());

    let mut child = Command::new(env!("CARGO_BIN_EXE_altd-sim"))
        .args(["--threads", "1", "train-scratch", "--population", "1", "--generations", "1000000", "--ticks", "60",
            "--shape", "20,5", "--mutation-start", "0.000000001", "--mutation-end", "0.000000001",
            "--checkpoint-every", "1000"])
        .arg("--out-dir").arg(root.join("live"))
        .arg("--init-network").arg(root.join("seed.json"))
        .arg("--settings").arg(root.join("settings.json"))
        .stdout(std::process::Stdio::null())
        .spawn().unwrap();
    let log = root.join("live/log.jsonl");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    while fs::read_to_string(&log).map_or(true, |text| text.lines().count() < 2) {
        assert!(std::time::Instant::now() < deadline, "trainer produced no generations");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    fs::write(root.join("live/stop_request"), "").unwrap();
    assert!(child.wait().unwrap().success());
    let reason = stop_reason(root, "live");
    assert_eq!(reason["condition"], "stop_request");
    let generations = logs(root, "live").len();
    assert_eq!(reason["generation"], generations - 1);
    assert_eq!(load(root.join("live/checkpoint.json"))["generation"], generations);
    assert!(!root.join("live/stop_request").exists(), "honored request is removed");
}
