//! Fixed generated inputs shared by integration tests and public diagnostics.
// Each integration test or diagnostic imports the subset of these helpers it needs.
#![allow(dead_code)]

use altd_sim::{
    track::random_track::GeneratedTrack,
    track::training_tracks::{training_scene_at, RandomTrainingTrackSettings},
};
use serde_json::{json, Value};

pub(crate) fn load_json(path: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}")))
        .unwrap_or_else(|e| panic!("{path}: {e}"))
}

pub(crate) fn asset(path: &str) -> Value {
    load_json(&format!("{}/assets/{path}", env!("CARGO_MANIFEST_DIR")))
}

pub(crate) fn template(vehicle: &str) -> Value {
    asset(&format!("scenes/{vehicle}_template.json"))
}

pub(crate) fn settings() -> RandomTrainingTrackSettings {
    serde_json::from_value(asset("random_track_settings.json")).unwrap()
}

fn input(vehicle: &str, slot: usize) -> (GeneratedTrack, Value) {
    let settings = settings();
    settings.validate().unwrap();
    training_scene_at(
        &template(vehicle),
        &settings,
        None,
        1729,
        0,
        slot,
        altd_sim::math::profile::MathProfile::Proton,
    )
    .unwrap()
}

pub(crate) fn scene(vehicle: &str, slot: usize) -> Value {
    input(vehicle, slot).1
}

pub(crate) fn track(vehicle: &str, slot: usize) -> GeneratedTrack {
    input(vehicle, slot).0
}

pub(crate) fn spawn(scene: &Value) -> Value {
    json!({"position": scene["reset_position"], "rotation": scene["reset_rotation"]})
}

pub(crate) fn network(vehicle: &str) -> Value {
    asset(&format!("networks/{vehicle}.json"))
}

pub(crate) fn network_export(vehicle: &str) -> Value {
    let metadata = network(vehicle);
    let inputs = metadata["inputs"].as_array().unwrap().len();
    let outputs = metadata["outputs"].as_array().unwrap().len();
    let mut export = altd_sim::nn::network::Network::xavier(
        &[inputs, 8, outputs],
        &mut altd_sim::training::pyrandom::PyRandom::new(1729),
    )
    .to_json();
    for (key, value) in metadata.as_object().unwrap() {
        export[key] = value.clone();
    }
    export
}

pub(crate) fn model(vehicle: &str) -> Value {
    asset(&format!("models/{vehicle}.json"))
}
