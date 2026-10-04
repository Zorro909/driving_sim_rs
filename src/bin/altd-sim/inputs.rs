//! Read the ordered values in exported traces, networks, and settings.

use altd_sim::physics::car::{Car, Controls};
use altd_sim::track::world::World;
use serde_json::Value;

pub(super) fn num(value: &Value) -> f64 {
    value
        .as_f64()
        .unwrap_or_else(|| panic!("expected a number, got {value}"))
}

pub(super) fn frames(trace: &Value) -> &Vec<Value> {
    trace["frames"].as_array().expect("trace frames")
}

/// `Controls(**{item["name"].lower(): item["value"]})`.
pub(super) fn recorded_controls(frame: &Value) -> Controls {
    let mut controls = Controls::default();
    for item in frame["outputs"].as_array().expect("outputs") {
        controls.set(item["name"].as_str().unwrap(), num(&item["value"]));
    }
    controls
}

/// Car state from a recorded frame plus the wheel history of the frame before.
pub(super) fn car_from_frame(world: &World, state: &Value, previous: Option<&Value>) -> Car {
    altd_sim::physics::trace_state::car_from_frame(world, state, previous, None)
}

pub(super) fn output_names(network_data: &Value) -> Vec<String> {
    network_data["outputs"]
        .as_array()
        .expect("outputs")
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect()
}

/// CLI switches override the game settings; omitted switches default to false.
pub(super) fn elimination_options(settings: &Value, wall: Option<bool>, idle: Option<bool>) -> (bool, bool) {
    let truthy = |key: &str| {
        settings
            .get(key)
            .is_some_and(|v| v.as_bool().unwrap_or_else(|| num(v) != 0.0))
    };
    (
        wall.unwrap_or_else(|| truthy("eliminate")),
        idle.unwrap_or_else(|| truthy("idle_eliminate")),
    )
}
