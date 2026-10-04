#[path = "support/oracle.rs"]
mod oracle;
#[path = "support/stats.rs"]
mod stats;
use serde_json::Value;

fn verify(log: &str, samples: usize, checked: u64, laps: u64, inactive: usize) {
    let capture: Value =
        serde_json::from_str(log.lines().next().unwrap().strip_prefix("FIDELITY_CAPTURE ").unwrap()).unwrap();
    let scene = oracle::scene(&capture);
    let rows: Vec<Value> = log
        .lines()
        .filter_map(|line| line.strip_prefix("FIDELITY_STATS "))
        .map(|row| serde_json::from_str(row).unwrap())
        .collect();
    assert_eq!(rows.len(), samples);
    assert_eq!(rows.last().unwrap()["stats"]["LapCount"], laps);
    assert_eq!(rows.last().unwrap()["observerLapCount"], laps);
    assert_eq!(rows.iter().filter(|row| row["active"] == false).count(), inactive);
    assert!(rows
        .iter()
        .any(|row| row["hidden"]["_continuousDriftTime"]["value"].as_f64().unwrap() > 1.0));
    let result = stats::verify(&scene, log);
    assert_eq!(result["mismatches"].as_object().unwrap().len(), 0, "{result}");
    assert_eq!(result["checked"], checked);
}

#[test]
fn generated_drive_statistics_match_original_game() {
    verify(
        include_str!("fixtures/native_generated_stats_drive.jsonl"),
        234,
        7443,
        0,
        0,
    );
}

#[test]
fn generated_lap_drift_and_inactive_statistics_match_original_game() {
    verify(
        include_str!("fixtures/native_generated_stats_laps.jsonl"),
        126,
        4111,
        2,
        3,
    );
}
