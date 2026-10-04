//! Generated fixtures and optional caller-supplied input files.
use altd_sim::math::vec2::V2;
use altd_sim::nn::network;
use altd_sim::track::world::World;
use altd_sim::training::evolution::EvolutionSettings;
use altd_sim::training::pyrandom::PyRandom;
use altd_sim::training::{SensorLayout, TrainingRunner};
use std::sync::Arc;
#[path = "../support/generated.rs"]
mod generated;
pub(super) use generated::load_json;

pub(super) const ROOT: &str = env!("CARGO_MANIFEST_DIR");

/// Optional input overrides for a caller's own scene, spawn, network and model.
pub(super) fn input(var: &str, default: &str) -> String {
    std::env::var(var).unwrap_or_else(|_| default.to_owned())
}

pub(super) fn scene_path() -> String {
    input("ALTD_GPU_SCENE", "generated rally track, seed 1729, slot 0")
}

fn load_scene() -> serde_json::Value {
    if let Ok(path) = std::env::var("ALTD_GPU_SCENE") {
        return load_json(&path);
    }
    generated::scene("rally", 0)
}

pub(super) fn load_world() -> World {
    World::from_scene(&load_scene())
}

/// Deterministic generated-track runner, or explicit ALTD_GPU_CKPT population.
/// Larger requested populations repeat saved networks when a checkpoint is supplied.
pub(super) fn fixture_runner(networks: usize) -> TrainingRunner {
    let network_data = load_json(&input(
        "ALTD_GPU_NETWORK",
        &format!("{ROOT}/assets/networks/rally.json"),
    ));
    let model = load_json(&input("ALTD_GPU_MODEL", &format!("{ROOT}/assets/models/rally.json")));
    let scene = load_scene();
    let spawn = match std::env::var("ALTD_GPU_SPAWN") {
        Ok(path) => load_json(&path)["frames"][0].clone(),
        Err(_) => generated::spawn(&scene),
    };
    let (nets, generation) = match std::env::var("ALTD_GPU_CKPT") {
        Ok(dir) => {
            let meta = load_json(&format!("{dir}/checkpoint.json"));
            let shape: Vec<usize> = match meta["shape"].as_array() {
                Some(widths) => widths.iter().map(|w| w.as_u64().unwrap() as usize).collect(),
                None => vec![20, 16, 16, 16, 16, 12, 12, 8, 5],
            };
            let file = format!("{dir}/{}", meta["population_file"].as_str().unwrap());
            let size = network::parameter_count(&shape);
            let bytes = std::fs::read(&file).unwrap_or_else(|e| panic!("{file}: {e}"));
            assert!(
                !bytes.is_empty() && bytes.len().is_multiple_of(size * 8),
                "checkpoint population size does not match its network shape"
            );
            let nets = bytes
                .chunks_exact(size * 8)
                .cycle()
                .take(networks)
                .map(|c| {
                    network::Network::from_vector(
                        &shape,
                        c.chunks_exact(8)
                            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                            .collect(),
                    )
                })
                .collect::<Vec<_>>();
            (nets, meta["generation"].as_u64().unwrap())
        }
        Err(_) => {
            let shape = [
                network_data["inputs"].as_array().unwrap().len(),
                16,
                16,
                16,
                16,
                12,
                12,
                8,
                5,
            ];
            let mut rng = PyRandom::new(1);
            (
                (0..networks)
                    .map(|_| network::Network::xavier(&shape, &mut rng))
                    .collect(),
                0,
            )
        }
    };
    let outputs: Vec<String> = network_data["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    let settings = EvolutionSettings::from_mcp(&serde_json::json!({"population": networks}));
    let p = &spawn["position"];
    let mut runner = TrainingRunner::new(
        Arc::new(World::from_scene(&scene)),
        V2::new(p[0].as_f64().unwrap(), p[1].as_f64().unwrap()),
        spawn["rotation"].as_f64().unwrap(),
        SensorLayout::from_exports(&network_data, &model),
        &outputs,
        settings,
        PyRandom::new(1),
        2,
        0,
        false,
        false,
    );
    runner.resume(&nets, generation);
    runner
}
