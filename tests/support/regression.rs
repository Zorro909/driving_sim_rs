//! Exact state encoding used by the generated regression fixtures.
// Each integration test imports the subset of these shared helpers it needs.
#![allow(dead_code)]
use altd_sim::{
    math::vec2::V2,
    physics::car::{Car, Controls},
    training::session::Session,
    training::TrainingStats,
};
use serde_json::{json, Value};

pub(crate) fn bit(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}
pub(crate) fn bits(values: impl IntoIterator<Item = f64>) -> Vec<String> {
    values.into_iter().map(bit).collect()
}
pub(crate) fn pair(v: V2) -> Value {
    json!(bits([v.x, v.y]))
}
pub(crate) fn controls(tick: usize) -> Controls {
    Controls {
        acceleration: if tick % 9 < 6 { 1.0 } else { -0.4 },
        steering: if tick % 7 < 3 { 0.35 } else { -0.2 },
        brake: if tick.is_multiple_of(11) { 0.5 } else { 0.0 },
        handbrake: if tick.is_multiple_of(13) { 0.25 } else { 0.0 },
        boost: if tick.is_multiple_of(5) { 1.0 } else { 0.0 },
    }
}
pub(crate) fn car_bits(c: &Car) -> Value {
    json!({
        "position": pair(c.position), "velocity": pair(c.velocity),
        "acceleration": pair(c.acceleration),
        "rotation": bit(c.rotation), "transform_angle": bit(c.transform_angle),
        "angular_velocity": bit(c.angular_velocity), "boost_energy": bit(c.boost_energy),
        "basis": [bits([c.body_basis.0.x as f64, c.body_basis.0.y as f64]), bits([c.body_basis.1.x as f64, c.body_basis.1.y as f64])],
        "active": c.active, "frozen": c.frozen, "tick": c.tick, "collision_count": c.collision_count,
        "wheels": c.wheels.iter().map(|w| json!({"angle":bit(w.angle_deg),"previous_position":w.previous_position.map(pair),"previous_surface":format!("{:?}",w.previous_surface)})).collect::<Vec<_>>(),
        "contacts": c.wall_contacts.iter().map(|contact| json!({
            "normal":pair(contact.normal), "point":pair(contact.point), "wall_point":pair(contact.wall_point),
            "local_point":pair(contact.local_point), "wall_local_point":pair(contact.wall_local_point),
            "shape_index":contact.shape_index, "wall_first":contact.wall_first, "used":contact.used,
            "values":bits([contact.depth,contact.normal_mass as f64,contact.tangent_mass as f64,contact.bias as f64,contact.bounce as f64,contact.friction as f64,contact.acc_normal as f64,contact.acc_tangent as f64,contact.acc_bias as f64,contact.acc_bias_center as f64]),
        })).collect::<Vec<_>>(),
    })
}
pub(crate) fn frame(c: &Car) -> Value {
    json!({
        "position":[c.position.x,c.position.y], "velocity":[c.velocity.x,c.velocity.y],
        "rotation":c.rotation, "angular_velocity":c.angular_velocity, "boost_energy":c.boost_energy,
        "wheel_angles":c.wheels.iter().map(|w| w.angle_deg).collect::<Vec<_>>(),
        "body_basis":[[c.body_basis.0.x,c.body_basis.0.y],[c.body_basis.1.x,c.body_basis.1.y]],
        "physics_contacts":c.wall_contacts.iter().map(|p| json!({"local_position":[p.point.x,p.point.y],"collider_position":[p.wall_point.x,p.wall_point.y],"local_normal":[p.normal.x,p.normal.y]})).collect::<Vec<_>>(),
    })
}
pub(crate) fn stats_bits(s: &TrainingStats) -> Value {
    json!({
        "totals":bits([s.network_novelty,s.total_score,s.total_speed,s.total_abs_steering,s.total_actual_distance,s.total_drifting_amount,s.total_momentum_change_frequency,s.total_throttle_change_frequency,s.total_braking_frequency,s.total_distance_from_wall,s.total_distance_from_center,s.previous_speed,s.previous_throttle,s.previous_score,s.current_slip_angle_degrees,s.continuous_drift_time,s.max_continuous_drift_time]),
        "score":{"offset":s.score.previous_offset.map(bit),"total":bit(s.score.total_score),"laps":s.score.lap_count,"backwards":s.score.went_backwards,"crossing_fraction":s.score.lap_crossing_fraction.map(bit)},
        "best_lap_time":s.best_lap_time.map(bit), "all_laps_time_sum":s.all_laps_time_sum.map(bit),
        "counts":[s.update_count,s.collision_count,s.idle_ticks,s.previous_lap_count,s.lap_start_tick,s.drift_ticks],
        "recent_score_diffs":bits(s.recent_score_diffs.iter().copied()),
        "metrics":s.metrics().iter().map(|v|v.map(bit)).collect::<Vec<_>>(),
    })
}
pub(crate) fn session_snapshot(session: &Session) -> Value {
    // These generated captures predate the optional score aggregates. Keep
    // comparing every original summary field; Session tests cover the additions.
    let mut summary = serde_json::to_value(session.generation_summary()).unwrap();
    summary.as_object_mut().unwrap().retain(|key, _| {
        matches!(
            key.as_str(),
            "bestIndex" | "bestScore" | "lapped" | "active" | "lapIndex" | "lapTime"
        )
    });
    json!({
        "generation":session.runner.generation,"tick":session.runner.tick,
        "flat_states":bits(session.car_states()),"rng":session.runner.rng.to_json(),
        "summary":summary,
        "cars":session.runner.agents.iter().map(|a|car_bits(&a.car)).collect::<Vec<_>>(),
        "stats":session.runner.agents.iter().map(|a|stats_bits(&a.stats)).collect::<Vec<_>>(),
        "network_parameters":session.runner.agents.iter().map(|a|bits(a.network.params.iter().copied())).collect::<Vec<_>>(),
        "sensors":(0..session.runner.agents.len()).map(|i|bits(session.sensors(i).unwrap())).collect::<Vec<_>>(),
        "controls":(0..session.runner.agents.len()).map(|i|bits(session.controls(i).unwrap())).collect::<Vec<_>>(),
    })
}

pub(crate) fn controls_from_json(v: &Value) -> Controls {
    let a = v.as_array().unwrap();
    Controls {
        acceleration: a[0].as_f64().unwrap(),
        steering: a[1].as_f64().unwrap(),
        brake: a[2].as_f64().unwrap(),
        handbrake: a[3].as_f64().unwrap(),
        boost: a[4].as_f64().unwrap(),
    }
}

pub(crate) fn vector(v: &Value) -> V2 {
    V2::new(v[0].as_f64().unwrap(), v[1].as_f64().unwrap())
}
