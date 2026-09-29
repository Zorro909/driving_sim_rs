use altd_sim::network::Network;
use serde_json::{json, Value};
use std::{fs, path::{Path, PathBuf}, process::{Command, Output}};

struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("altd-scratch-test-{}", std::process::id()));
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

#[test]
fn scratch_elimination_settings_overrides_and_resume() {
    let workspace = Workspace::new();
    let root = &workspace.0;
    let template = load(concat!(env!("CARGO_MANIFEST_DIR"), "/../docs/traces/rally_a01_live_network.json"));
    let mut seed = Network::from_vector(&[20, 5], vec![0.0; 105]).to_json();
    seed["inputs"] = template["inputs"].clone();
    seed["outputs"] = template["outputs"].clone();
    fs::write(root.join("seed.json"), seed.to_string()).unwrap();
    // Preserve the stationary seed so both generations exercise idle elimination.
    let settings = json!({"selection_size": 1, "preserve_parents_size": 1,
        "eliminate": true, "idle_eliminate": 1});
    fs::write(root.join("settings.json"), json!({"settings": settings}).to_string()).unwrap();

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
    success(run(root, "settings", &["--resume"]));
    let after = logs(root, "settings");
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
    success(run(root, "settings", &["--resume", "--no-idle-eliminate", "--population", "3",
        "--ticks", "120", "--mutation-start", "0.2", "--mutation-end", "0.1"]));
    let changed = load(root.join("settings/run.json"));
    assert_eq!(changed["population"], 3);
    assert_eq!(changed["idle_eliminate"], false);
    let changed_logs = logs(root, "settings");
    assert_eq!(changed_logs[1]["active_cars"], 3);
    assert_eq!(changed_logs[1]["simulated_ticks"], 126);
    assert_eq!(changed_logs[1]["mutation_rate"], 0.1);
    let migrated = load(&checkpoint_path);
    assert_eq!(migrated["shape"], json!([20, 5]));
    assert_eq!(migrated["population"], 1, "metadata describes the saved binary, not resized agents");
    // A second resume must still decode the original checkpoint after run.json changed.
    success(run(root, "settings", &["--resume"]));
    assert_eq!(logs(root, "settings")[1]["active_cars"], 0);

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

