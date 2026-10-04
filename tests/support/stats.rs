use altd_sim::{
    physics::car::{Car, Controls},
    track::curve::json_vector,
    track::world::World,
    training::TrainingStats,
};
use serde_json::{json, Value};
pub(crate) fn verify(scene: &Value, log: &str) -> Value {
    let world = World::from_scene(scene);
    let mut stats = TrainingStats::new(0.0);
    let mut mismatches = std::collections::BTreeMap::<String, usize>::new();
    let mut examples = Vec::new();
    let mut count = 0;
    for row in log.lines().filter_map(|s| s.strip_prefix("FIDELITY_STATS ")) {
        let r: Value = serde_json::from_str(row).unwrap();
        let tick = r["tick"].as_u64().unwrap();
        let mut car = Car::new(&world, json_vector(&r["position"]).into(), 0.0);
        car.velocity = json_vector(&r["velocity"]).into();
        car.set_body_basis(json_vector(&r["basis"][0]), json_vector(&r["basis"][1]));
        car.active = r["active"].as_bool().unwrap_or(true);
        let mut control = Controls::default();
        for o in r["outputs"].as_array().unwrap() {
            control.set(o["name"].as_str().unwrap(), o["value"].as_f64().unwrap());
        }
        stats.update(&world, &car, &control, tick, r["collided"].as_bool().unwrap());
        let mut check = |key: &str, a: f64, e: &Value| {
            count += 1;
            let bits = if let Some(hex) = e["bits"].as_str() {
                u64::from_str_radix(hex, 16).unwrap()
            } else {
                e.as_f64().unwrap().to_bits()
            };
            if a.to_bits() != bits {
                let n = mismatches.entry(key.into()).or_default();
                *n += 1;
                if *n < 4 {
                    examples.push(json!({"tick":tick,"field":key,"actual":a,"expected":f64::from_bits(bits)}));
                }
            }
        };
        let s = &r["stats"];
        let h = &r["hidden"];
        for (key, a) in [
            ("TotalScore", stats.total_score),
            ("TotalSpeed", stats.total_speed),
            ("TotalAbsSteering", stats.total_abs_steering),
            ("TotalActualDistance", stats.total_actual_distance),
            ("TotalDriftingAmount", stats.total_drifting_amount),
            ("TotalMomentumChangeFrequency", stats.total_momentum_change_frequency),
            ("TotalThrottleChangeFrequency", stats.total_throttle_change_frequency),
            ("TotalBrakingFrequency", stats.total_braking_frequency),
            ("TotalDistanceFromWall", stats.total_distance_from_wall),
            ("TotalDistanceFromCenter", stats.total_distance_from_center),
            ("PreviousTrackOffset", stats.score.previous_offset.unwrap()),
            ("CollisionCount", stats.collision_count as f64),
            ("LapCount", stats.score.lap_count as f64),
            ("TotalRacingLineEfficiency", stats.metrics()[13].unwrap()),
        ] {
            check(key, a, &s[key]);
        }
        for (key, a) in [
            ("_idleTicks", stats.idle_ticks as f64),
            ("_updateCounter", stats.update_count as f64),
            ("_consecutiveDriftTicks", stats.drift_ticks as f64),
            ("_previousSpeed", stats.previous_speed),
            ("_previousThrottle", stats.previous_throttle),
        ] {
            check(key, a, &h[key]);
        }
        for (a, e) in stats
            .recent_score_diffs
            .iter()
            .zip(h["_recentScoreDiffs"].as_array().unwrap())
        {
            check("score_history", *a, e);
        }
        if let Some(t) = stats.best_lap_time {
            check("BestLapTime", t, &s["BestLapTime"]);
        }
        if let Some(t) = stats.all_laps_time_sum {
            check("AllLapsTimeSum", t, &s["AllLapsTimeSum"]);
        }
        check(
            "CurrentSlipAngleDegrees",
            stats.current_slip_angle_degrees,
            &s["CurrentSlipAngleDegrees"],
        );
        check(
            "_continuousDriftTime",
            stats.continuous_drift_time,
            &h["_continuousDriftTime"],
        );
        check(
            "_maxContinuousDriftTime",
            stats.max_continuous_drift_time,
            &h["_maxContinuousDriftTime"],
        );
    }
    json!({"checked":count,"mismatches":mismatches,"examples":examples})
}
