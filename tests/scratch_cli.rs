use altd_sim::nn::network::Network;
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

#[path = "../examples/support/generated.rs"]
mod generated;

struct Workspace(PathBuf);
impl Workspace {
    fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!("altd-scratch-test-{}-{tag}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn load(path: impl AsRef<Path>) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn run(root: &Path, name: &str, extra: &[&str]) -> Output {
    let defaults = [
        ("--population", "1"),
        ("--generations", "2"),
        ("--ticks", "600"),
        ("--shape", "20,5"),
        ("--mutation-start", "0.000000001"),
        ("--mutation-end", "0.000000001"),
        ("--checkpoint-every", "1"),
    ];
    let mut command = Command::new(env!("CARGO_BIN_EXE_altd-sim"));
    command.args(["--threads", "1", "train-scratch"]);
    for (key, value) in defaults {
        if !extra.contains(&key) {
            command.args([key, value]);
        }
    }
    if !extra.contains(&"--spawn-trace") {
        command.arg("--spawn-trace").arg(root.join("spawn.json"));
    }
    command
        .arg("--out-dir")
        .arg(root.join(name))
        .arg("--init-network")
        .arg(root.join("seed.json"))
        .arg("--scene")
        .arg(root.join("scene.json"))
        .arg("--network")
        .arg(root.join("seed.json"))
        .arg("--model")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/models/rally.json"))
        .arg("--settings")
        .arg(root.join("settings.json"))
        .args(extra)
        .output()
        .unwrap()
}
fn success(output: Output) {
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
fn logs(root: &Path, name: &str) -> Vec<Value> {
    fs::read_to_string(root.join(name).join("log.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}
/// Writes a stationary 20,5 seed network and the given settings file.
fn seed_workspace(root: &Path, settings: &Value) {
    let template = generated::network("rally");
    let scene = generated::scene("rally", 0);
    fs::write(root.join("scene.json"), scene.to_string()).unwrap();
    fs::write(
        root.join("spawn.json"),
        json!({"frames": [generated::spawn(&scene)]}).to_string(),
    )
    .unwrap();
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
fn random_tracks_change_each_generation_and_resume_the_same_sequence() {
    let workspace = Workspace::new("random-tracks");
    let root = &workspace.0;
    seed_workspace(root, &json!({"selection_size": 1, "preserve_parents_size": 1}));
    let settings_path = root.join("tracks.json");
    fs::write(
        &settings_path,
        json!({
            "length": {"min": 6, "max": 12}, "allow_double": [false, true],
            "surfaces": {"pool": [0, 1, 2], "count": [1, 2, 3]}, "distribution": [0, 1]
        })
        .to_string(),
    )
    .unwrap();
    let options = [
        "--track-mode",
        "random",
        "--random-track-settings",
        settings_path.to_str().unwrap(),
        "--ticks",
        "12",
        "--spawn-trace",
        "/no-spawn-trace-needed",
    ];
    success(run(
        root,
        "continuous",
        &[options.as_slice(), &["--generations", "4", "--track-buffer-size", "1"]].concat(),
    ));
    success(run(
        root,
        "resumed",
        &[options.as_slice(), &["--generations", "2", "--track-buffer-size", "3"]].concat(),
    ));
    success(run(
        root,
        "resumed",
        &[
            options.as_slice(),
            &[
                "--resume",
                "--generations",
                "4",
                "--seed",
                "999",
                "--track-buffer-size",
                "2",
            ],
        ]
        .concat(),
    ));
    let continuous = logs(root, "continuous");
    let resumed = logs(root, "resumed");
    assert_eq!(continuous.len(), 4);
    let mut previous_tiles = Value::Null;
    for (a, b) in continuous.iter().zip(&resumed) {
        for key in [
            "generation",
            "track_config",
            "track_file",
            "best_score",
            "mean_score",
            "simulated_ticks",
        ] {
            assert_eq!(a[key], b[key], "resume must preserve {key}");
        }
        let name = a["track_file"].as_str().unwrap();
        let track = load(root.join("continuous").join(name));
        assert_eq!(track, load(root.join("resumed").join(name)));
        assert_ne!(track["tiles"], previous_tiles);
        previous_tiles = track["tiles"].clone();
        assert!((6..=12).contains(&track["config"]["length"].as_i64().unwrap()));
    }
    let meta = load(root.join("resumed/checkpoint.json"));
    assert_eq!(meta["random_tracks"]["seed"], 1);
    assert_eq!(meta["generation"], 4);
    assert_eq!(load(root.join("resumed/run.json"))["random_track_seed"], 1);

    let rejected = run(
        root,
        "invalid-buffer",
        &[options.as_slice(), &["--track-buffer-size", "0"]].concat(),
    );
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("track-buffer-size must be positive"));
}

#[test]
fn track_batches_resume_and_allow_stage_count_changes_without_reseeding() {
    let workspace = Workspace::new("track-batches");
    let root = &workspace.0;
    seed_workspace(root, &json!({"selection_size": 1, "preserve_parents_size": 1}));
    let options = ["--track-mode", "random", "--ticks", "12"];
    success(run(
        root,
        "continuous",
        &[
            options.as_slice(),
            &[
                "--tracks-per-generation",
                "3",
                "--generations",
                "4",
                "--track-buffer-size",
                "1",
            ],
        ]
        .concat(),
    ));
    success(run(
        root,
        "resumed",
        &[
            options.as_slice(),
            &[
                "--tracks-per-generation",
                "3",
                "--generations",
                "2",
                "--track-buffer-size",
                "4",
            ],
        ]
        .concat(),
    ));
    // Omitted count restores the checkpoint; a changed seed must not change tracks.
    success(run(
        root,
        "resumed",
        &[options.as_slice(), &["--resume", "--generations", "4", "--seed", "987"]].concat(),
    ));
    for (a, b) in logs(root, "continuous").iter().zip(logs(root, "resumed")) {
        assert_eq!(a["tracks_per_generation"], 3);
        assert_eq!(a["tracks"], b["tracks"]);
        for key in [
            "best_score",
            "mean_score",
            "best_fitness",
            "lapped_all_tracks",
            "simulated_ticks",
        ] {
            assert_eq!(a[key], b[key], "{key}");
        }
        assert_eq!(a["tracks"].as_array().unwrap().len(), 3);
        for track in a["tracks"].as_array().unwrap() {
            let name = track["track_file"].as_str().unwrap();
            assert_eq!(
                load(root.join("continuous").join(name)),
                load(root.join("resumed").join(name))
            );
        }
    }
    let a = load(root.join("continuous/checkpoint.json"));
    let b = load(root.join("resumed/checkpoint.json"));
    assert_eq!(a["generation"], 4);
    assert_eq!(a["rng"], b["rng"]);
    assert_eq!(
        fs::read(root.join("continuous").join(a["population_file"].as_str().unwrap())).unwrap(),
        fs::read(root.join("resumed").join(b["population_file"].as_str().unwrap())).unwrap()
    );
    assert!(!root.join("resumed/progress.json").exists());
    // Abandoned work beyond the checkpoint must disappear even when K changes.
    fs::write(root.join("resumed/tracks/g00004_t002.track.json"), "{}").unwrap();
    fs::write(root.join("resumed/best_laps/g00004_t002_1.00s.json"), "{}").unwrap();
    fs::write(root.join("resumed/best.json"), "{}").unwrap();
    let mut log = fs::OpenOptions::new()
        .append(true)
        .open(root.join("resumed/log.jsonl"))
        .unwrap();
    std::io::Write::write_all(&mut log, b"{\"generation\":4,\"best_score\":9999}\n").unwrap();
    success(run(
        root,
        "resumed",
        &[
            options.as_slice(),
            &["--resume", "--tracks-per-generation", "1", "--generations", "5"],
        ]
        .concat(),
    ));
    assert_eq!(logs(root, "resumed").len(), 5);
    assert!(!root.join("resumed/tracks/g00004_t002.track.json").exists());
    assert!(!root.join("resumed/best_laps/g00004_t002_1.00s.json").exists());
    assert!(!root.join("resumed/best.json").exists());
    success(run(
        root,
        "single",
        &[options.as_slice(), &["--generations", "5"]].concat(),
    ));
    let multi = load(root.join("resumed/tracks/g00004.track.json"));
    assert_eq!(multi, load(root.join("single/tracks/g00004.track.json")));
    assert_eq!(load(root.join("resumed/checkpoint.json"))["tracks_per_generation"], 1);
    for (name, extra) in [
        ("zero", vec!["--track-mode", "random", "--tracks-per-generation", "0"]),
        ("fixed-batch", vec!["--tracks-per-generation", "2"]),
    ] {
        assert!(!run(root, name, &extra).status.success());
    }
}

#[test]
fn train_random_mode_uses_generated_spawns_without_a_trace() {
    let workspace = Workspace::new("train-random");
    let root = &workspace.0;
    seed_workspace(root, &json!({"selection_size": 1, "preserve_parents_size": 1}));
    let output_path = root.join("training.json");
    let output = Command::new(env!("CARGO_BIN_EXE_altd-sim"))
        .args([
            "--threads",
            "1",
            "train",
            "--track-mode",
            "random",
            "--tracks-per-generation",
            "3",
            "--population",
            "1",
            "--generations",
            "2",
            "--ticks",
            "12",
        ])
        .arg("--scene")
        .arg(root.join("scene.json"))
        .arg("--network")
        .arg(root.join("seed.json"))
        .arg("--model")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/models/rally.json"))
        .arg("--settings")
        .arg(root.join("settings.json"))
        .arg("--output")
        .arg(&output_path)
        .output()
        .unwrap();
    success(output);
    let report = load(&output_path);
    assert_eq!(report["track_mode"], "random");
    assert_eq!(report["tracks_per_generation"], 3);
    for row in report["history"].as_array().unwrap() {
        assert_eq!(row["tracks"].as_array().unwrap().len(), 3);
        assert_eq!(row["ticks"], 54);
        assert_eq!(row["track"], row["tracks"][0]["track"]);
    }
    assert_ne!(
        report["history"][0]["track"]["tiles"],
        report["history"][1]["track"]["tiles"]
    );
    let fixed = Command::new(env!("CARGO_BIN_EXE_altd-sim"))
        .args([
            "--threads",
            "1",
            "train",
            "--scene",
            "/missing",
            "--network",
            "/missing",
            "--model",
            "/missing",
            "--output",
            "/missing",
        ])
        .output()
        .unwrap();
    assert!(!fixed.status.success());
    assert!(String::from_utf8_lossy(&fixed.stderr).contains("--spawn-trace"));
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
        for key in [
            "generation",
            "best_score",
            "mean_score",
            "active_cars",
            "simulated_ticks",
        ] {
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
    success(run(
        root,
        "settings",
        &[
            "--resume",
            "--no-idle-eliminate",
            "--population",
            "3",
            "--generations",
            "4",
            "--ticks",
            "120",
            "--mutation-start",
            "0.2",
            "--mutation-end",
            "0.1",
            "--checkpoint-every",
            "0",
        ],
    ));
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
    assert_eq!(
        migrated["population"], 1,
        "metadata describes the saved binary, not resized agents"
    );
    // A second resume must still decode the original checkpoint after run.json changed.
    success(run(
        root,
        "settings",
        &["--resume", "--generations", "4", "--checkpoint-every", "0"],
    ));
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
    let population_path = root
        .join("settings")
        .join(migrated["population_file"].as_str().unwrap());
    let mut bytes = fs::read(&population_path).unwrap();
    bytes.pop();
    fs::write(&population_path, bytes).unwrap();
    let rejected = run(root, "settings", &["--resume"]);
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("population file size"));
    assert_eq!(load(root.join("settings/run.json")), before_config);

    success(run(
        root,
        "disabled",
        &["--no-idle-eliminate", "--no-eliminate-on-wall"],
    ));
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
    assert_eq!(
        load(root.join("enabled/run.json"))["settings"]["rewards"],
        json!([{"metric": "total_score", "weight": 100, "type": "default"}])
    );
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
    let common = [
        "--population",
        "4",
        "--ticks",
        "120",
        "--mutation-start",
        "0.2",
        "--mutation-end",
        "0.2",
    ];
    success(run(root, "whole", &[&common[..], &["--generations", "3"]].concat()));
    success(run(root, "split", &[&common[..], &["--generations", "2"]].concat()));
    assert_eq!(
        stop_reason(root, "split"),
        json!({"condition": "generations", "generation": 1, "value": 2, "threshold": 2})
    );
    success(run(
        root,
        "split",
        &[&common[..], &["--generations", "3", "--resume"]].concat(),
    ));

    let whole: Vec<Value> = logs(root, "whole").into_iter().map(without_timing).collect();
    let split: Vec<Value> = logs(root, "split").into_iter().map(without_timing).collect();
    assert_eq!(whole.len(), 3);
    assert_eq!(whole, split);
    let (a, b) = (
        load(root.join("whole/checkpoint.json")),
        load(root.join("split/checkpoint.json")),
    );
    assert_eq!(a["generation"], 3);
    assert_eq!(a, b);
    assert_eq!(
        fs::read(root.join("whole/checkpoint_g00003.bin")).unwrap(),
        fs::read(root.join("split/checkpoint_g00003.bin")).unwrap()
    );
    assert!(
        !root.join("split/checkpoint_g00002.bin").exists(),
        "old population file removed"
    );
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

    success(run(
        root,
        "lapped",
        &[&many[..], &["--stop-lapped-percent", "0"]].concat(),
    ));
    assert_eq!(stop_reason(root, "lapped")["condition"], "lapped_percent");
    assert_eq!(stop_reason(root, "lapped")["value"], 0.0);

    // Unmet rules leave the generation target in charge.
    success(run(
        root,
        "unmet",
        &[
            "--stop-lap-below",
            "1000",
            "--stop-score-above",
            "1e9",
            "--stop-lapped-percent",
            "50",
        ],
    ));
    assert_eq!(logs(root, "unmet").len(), 2);
    assert_eq!(stop_reason(root, "unmet")["condition"], "generations");

    // Score plateau: generation 0 improves on nothing, 1 and 2 are stale.
    success(run(
        root,
        "plateau-score",
        &[&many[..], &["--stop-plateau", "2", "--plateau-metric", "score"]].concat(),
    ));
    assert_eq!(logs(root, "plateau-score").len(), 3);
    assert_eq!(
        stop_reason(root, "plateau-score"),
        json!({"condition": "plateau", "generation": 2, "value": 2, "threshold": 2, "metric": "score"})
    );
    // Lap plateau: generations without a lap never improve.
    success(run(
        root,
        "plateau-lap",
        &[&many[..], &["--stop-plateau", "2"]].concat(),
    ));
    assert_eq!(logs(root, "plateau-lap").len(), 2);
    assert_eq!(stop_reason(root, "plateau-lap")["metric"], "lap");
    // Plateau counts restart with each invocation.
    success(run(
        root,
        "plateau-lap",
        &["--resume", "--generations", "50", "--stop-plateau", "2"],
    ));
    assert_eq!(logs(root, "plateau-lap").len(), 4);
    assert_eq!(stop_reason(root, "plateau-lap")["generation"], 3);

    // Without checkpoints a stop still ends the run, just without saving.
    success(run(
        root,
        "nockpt",
        &[&many[..], &["--stop-score-above=-1000", "--checkpoint-every", "0"]].concat(),
    ));
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
        .args([
            "--threads",
            "1",
            "train-scratch",
            "--population",
            "1",
            "--generations",
            "1000000",
            "--ticks",
            "60",
            "--track-mode",
            "random",
            "--tracks-per-generation",
            "3",
            "--shape",
            "20,5",
            "--mutation-start",
            "0.000000001",
            "--mutation-end",
            "0.000000001",
            "--checkpoint-every",
            "1000",
        ])
        .arg("--out-dir")
        .arg(root.join("live"))
        .arg("--scene")
        .arg(root.join("scene.json"))
        .arg("--network")
        .arg(root.join("seed.json"))
        .arg("--model")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/models/rally.json"))
        .arg("--init-network")
        .arg(root.join("seed.json"))
        .arg("--settings")
        .arg(root.join("settings.json"))
        .stdout(std::process::Stdio::null())
        .spawn()
        .unwrap();
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
    assert!(logs(root, "live")
        .iter()
        .all(|row| row["tracks"].as_array().unwrap().len() == 3));
    assert_eq!(reason["generation"], generations - 1);
    assert_eq!(load(root.join("live/checkpoint.json"))["generation"], generations);
    assert!(!root.join("live/stop_request").exists(), "honored request is removed");
}

#[test]
fn final_candidate_precedes_breeding_without_changing_rng_or_offspring() {
    let workspace = Workspace::new("candidate");
    let root = &workspace.0;
    seed_workspace(
        root,
        &json!({"selection_size": 1, "preserve_parents": "off", "weight_decay": 0.0}),
    );
    let options = [
        "--population",
        "3",
        "--generations",
        "2",
        "--ticks",
        "12",
        "--mutation-start",
        "0.2",
        "--mutation-end",
        "0.2",
    ];
    success(run(root, "disabled", &options));
    success(run(
        root,
        "enabled",
        &[options.as_slice(), &["--save-final-candidate"]].concat(),
    ));
    assert!(!root.join("disabled/candidate.json").exists());
    let candidate_path = root.join("enabled/candidate.json");
    let candidate = load(&candidate_path);
    let a = load(root.join("disabled/checkpoint.json"));
    let b = load(root.join("enabled/checkpoint.json"));
    assert_eq!(a["rng"], b["rng"], "candidate export consumes no RNG draws");
    assert_eq!(
        fs::read(root.join("disabled").join(a["population_file"].as_str().unwrap())).unwrap(),
        fs::read(root.join("enabled").join(b["population_file"].as_str().unwrap())).unwrap()
    );
    assert_eq!(b["candidate"]["generation"], 1);
    assert_eq!(
        b["candidate"]["sha256"],
        altd_sim::training::evaluation::sha256(&fs::read(candidate_path).unwrap())
    );
    assert_eq!(candidate["training"]["generation"], 1);
    assert!(
        candidate["training"]["best_lap_s"].is_null(),
        "missing laps remain missing"
    );
    assert_eq!(
        candidate["training"]["fitness"],
        logs(root, "enabled")[1]["best_fitness"]
    );
    let leader = Network::from_game_export(&candidate);
    let checkpoint = fs::read(root.join("enabled").join(b["population_file"].as_str().unwrap())).unwrap();
    let bred: Vec<_> = checkpoint
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert!(
        bred.chunks_exact(leader.params.len()).all(|n| n != leader.params),
        "candidate weights precede reproduction"
    );
    for (a, b) in logs(root, "disabled").iter().zip(logs(root, "enabled")) {
        for key in [
            "generation",
            "best_fitness",
            "best_score",
            "mean_score",
            "mutation_rate",
        ] {
            assert_eq!(a[key], b[key]);
        }
    }
}

#[test]
fn frozen_evaluation_resets_tracks_validates_hashes_and_preserves_training_files() {
    let workspace = Workspace::new("evaluation");
    let root = &workspace.0;
    seed_workspace(root, &json!({"selection_size": 1, "preserve_parents_size": 1}));
    success(run(root, "trained", &["--save-final-candidate", "--ticks", "12"]));
    let sim = Path::new(env!("CARGO_MANIFEST_DIR"));
    let suite_dir = root.join("suite");
    fs::create_dir(&suite_dir).unwrap();
    fs::copy(root.join("scene.json"), suite_dir.join("scene.json")).unwrap();
    fs::copy(root.join("spawn.json"), suite_dir.join("spawn.json")).unwrap();
    let mut track = json!({"id": "first", "scene": "scene.json", "spawn": "spawn.json", "ticks": 12,
        "scene_sha256": altd_sim::training::evaluation::sha256(&fs::read(suite_dir.join("scene.json")).unwrap()),
        "spawn_sha256": altd_sim::training::evaluation::sha256(&fs::read(suite_dir.join("spawn.json")).unwrap())});
    let first = track.clone();
    track["id"] = json!("second");
    let suite = json!({"version": 1, "vehicle": "rally", "options": {"backend": "cpu", "batch_count": 1,
        "eliminate_on_wall": false, "idle_eliminate": false}, "tracks": [first, track]});
    let suite_path = suite_dir.join("manifest.json");
    fs::write(&suite_path, suite.to_string()).unwrap();
    let report_path = root.join("evaluation.json");
    let candidate_path = root.join("trained/candidate.json");
    let before = ["candidate.json", "checkpoint.json", "log.jsonl"]
        .map(|name| fs::read(root.join("trained").join(name)).unwrap());
    let evaluate = || {
        Command::new(env!("CARGO_BIN_EXE_altd-sim"))
            .args(["--threads", "1", "evaluate", "--network"])
            .arg(&candidate_path)
            .arg("--model")
            .arg(sim.join("assets/models/rally.json"))
            .arg("--suite")
            .arg(&suite_path)
            .arg("--report")
            .arg(&report_path)
            .output()
            .unwrap()
    };
    success(evaluate());
    let report = load(&report_path);
    assert_eq!(
        report["candidate_sha256"],
        altd_sim::training::evaluation::sha256(&before[0])
    );
    for key in ["score", "lap_complete", "best_lap_s", "collisions", "simulated_ticks"] {
        assert_eq!(
            report["tracks"][0][key], report["tracks"][1][key],
            "track reset preserves {key}"
        );
    }
    assert!(report["tracks"][0]["best_lap_s"].is_null());
    assert_eq!(
        before,
        ["candidate.json", "checkpoint.json", "log.jsonl"]
            .map(|name| fs::read(root.join("trained").join(name)).unwrap())
    );
    fs::remove_file(&report_path).unwrap();
    fs::write(suite_dir.join("spawn.json"), "{}").unwrap();
    let failed = evaluate();
    assert!(!failed.status.success());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("hash mismatch"));
    assert!(!report_path.exists());
}

/// Public diagnostics accept the same generated scenes and exported frame
/// schema as the captured fixtures. The stationary network gives exact
/// inference and sensor expectations without shipping trained game weights.
#[test]
fn public_diagnostics_and_benchmark_accept_generated_inputs() {
    use altd_sim::{
        physics::car::{Car, Controls, SensorScratch, DT},
        track::world::World,
        training::{ScoreTracker, SensorLayout},
    };

    let workspace = Workspace::new("diagnostics");
    let root = &workspace.0;
    seed_workspace(root, &json!({"selection_size": 1}));
    let scene = load(root.join("scene.json"));
    let network = load(root.join("seed.json"));
    let model = generated::model("rally");
    let world = World::from_scene(&scene);
    let layout = SensorLayout::from_exports(&network, &model);
    let mut car = Car::new(
        &world,
        altd_sim::track::world::vector(&scene["reset_position"]),
        scene["reset_rotation"].as_f64().unwrap(),
    );
    let mut scratch = SensorScratch::default();
    let mut inputs = Vec::new();
    // Stabilize the exported native angle before the first replayed frame.
    car.step(&world, &Controls::default(), DT, false);
    let mut frames = Vec::new();
    for tick in 0..14 {
        scratch.invalidate();
        layout.read_into(&world, &car, &mut scratch, &mut inputs);
        frames.push(json!({"tick": tick, "position": [car.position.x, car.position.y],
            "velocity": [car.velocity.x, car.velocity.y], "rotation": car.rotation,
            "angular_velocity": car.angular_velocity, "boost_energy": car.boost_energy,
            "wheel_angles": car.wheels.iter().map(|w| w.angle_deg).collect::<Vec<_>>(),
            "body_basis": [[car.body_basis.0.x, car.body_basis.0.y],
                [car.body_basis.1.x, car.body_basis.1.y]],
            "physics_contacts": [], "collision_count": car.collision_count, "score": 0.0,
            "outputs": network["outputs"].as_array().unwrap().iter()
                .map(|name| json!({"name": name, "value": 0.0})).collect::<Vec<_>>(),
            "sensors": layout.names.iter().zip(&inputs)
                .map(|(name, value)| json!({"name": name, "value": value})).collect::<Vec<_>>() }));
        car.step(&world, &Controls::default(), DT, false);
    }
    let trace_path = root.join("trace.json");
    fs::write(
        &trace_path,
        json!({"source": "generated current-code regression input",
        "frames": frames})
        .to_string(),
    )
    .unwrap();
    let scene_path = root.join("scene.json");
    let network_path = root.join("seed.json");
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models/rally.json");
    let report_path = root.join("report.json");
    let run_diagnostic = |name: &str, paths: &[&Path], options: &[&str]| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_altd-sim"));
        command.args(["--threads", "1", name]);
        for path in paths {
            command.arg(path);
        }
        success(command.args(options).output().unwrap());
        load(&report_path)
    };
    let trajectory = run_diagnostic(
        "compare-trace",
        &[&scene_path, &trace_path, &report_path],
        &["--start-index", "0", "--end-index", "10"],
    );
    assert_eq!(trajectory["end_index"], 10);
    assert_eq!(trajectory["frames"].as_array().unwrap().len(), 9);
    assert!(trajectory["first_large_divergence_tick"].is_null());
    let trajectory_path = root.join("trajectory.json");
    fs::copy(&report_path, &trajectory_path).unwrap();
    let one_step = run_diagnostic("compare-one-step", &[&scene_path, &trace_path, &report_path], &[]);
    assert_eq!(one_step["frames"], 12);
    assert_eq!(one_step["bit_exact_frames"], 12);
    assert_eq!(one_step["contact_mismatches"], json!([]));
    for options in [vec![], vec!["--new-vehicle"]] {
        let closed = run_diagnostic(
            "compare-closed-loop",
            &[&scene_path, &trace_path, &network_path, &model_path, &report_path],
            &options,
        );
        assert_eq!(closed["frames"], 13);
        assert_eq!(closed["full"]["output_error"]["max"], 0.0);
        assert!(closed["first_position_error_over_0_05_px"].is_null());
    }
    let inference = run_diagnostic("compare-network", &[&network_path, &trace_path, &report_path], &[]);
    assert_eq!(inference["compared_frames"], 12);
    assert_eq!(inference["max_error"], 0.0);
    let sensors = run_diagnostic(
        "compare-sensors",
        &[&scene_path, &trace_path, &model_path, &trajectory_path, &report_path],
        &["--all-frames"],
    );
    assert_eq!(sensors["stop_tick_exclusive"], 14);
    assert_eq!(sensors["vision_all"]["nonzero_errors"], 0);
    assert_eq!(sensors["ray_mismatches"], json!([]));
    let mut score = ScoreTracker::default();
    let mut score_frames = Vec::new();
    for i in 0..14 {
        if i > 0 {
            score.update(&world.track, world.track.path_position((i - 1) as f64 * 100.0));
        }
        let position = world.track.path_position(i as f64 * 100.0);
        score_frames.push(json!({"position": [position.x, position.y], "score": score.total_score}));
    }
    let score_path = root.join("score.json");
    fs::write(&score_path, json!({"frames": score_frames}).to_string()).unwrap();
    let score_report = run_diagnostic("compare-score", &[&scene_path, &score_path, &report_path], &[]);
    assert!(score_report["updates"].as_u64().unwrap() > 2);
    assert_eq!(score_report["max_error"], 0.0);

    let dump_path = root.join("state.json");
    success(
        Command::new(env!("CARGO_BIN_EXE_altd-sim"))
            .args([
                "--threads",
                "1",
                "bench",
                "--population",
                "8",
                "--ticks",
                "12",
                "--spawn-index",
                "1",
            ])
            .arg("--scene")
            .arg(&scene_path)
            .arg("--network")
            .arg(&network_path)
            .arg("--model")
            .arg(&model_path)
            .arg("--spawn-trace")
            .arg(&trace_path)
            .arg("--report")
            .arg(&report_path)
            .arg("--dump-state")
            .arg(&dump_path)
            .output()
            .unwrap(),
    );
    let bench = load(&report_path);
    assert_eq!(bench["population"], 8);
    assert_eq!(bench["network_shape"], json!([20, 5]));
    let dump = load(&dump_path);
    assert_eq!(dump["agents"].as_array().unwrap().len(), 8);
    assert_eq!(dump["next_networks"].as_array().unwrap().len(), 8);
}

#[test]
#[ignore = "requires an idle HIP GPU and gpu/build.sh"]
fn final_candidate_gpu_selection_matches_cpu() {
    let workspace = Workspace::new("candidate-gpu-parity");
    let root = &workspace.0;
    seed_workspace(root, &json!({"selection_size": 2, "preserve_parents": "off"}));
    let options = [
        "--save-final-candidate",
        "--population",
        "3",
        "--ticks",
        "12",
        "--mutation-start",
        "0.2",
        "--mutation-end",
        "0.2",
    ];
    success(run(root, "cpu", &options));
    success(run(root, "gpu", &[options.as_slice(), &["--gpu"]].concat()));
    assert_eq!(
        load(root.join("cpu/candidate.json")),
        load(root.join("gpu/candidate.json"))
    );
    let a = load(root.join("cpu/checkpoint.json"));
    let b = load(root.join("gpu/checkpoint.json"));
    assert_eq!(a["rng"], b["rng"]);
    assert_eq!(
        fs::read(root.join("cpu").join(a["population_file"].as_str().unwrap())).unwrap(),
        fs::read(root.join("gpu").join(b["population_file"].as_str().unwrap())).unwrap()
    );
}
