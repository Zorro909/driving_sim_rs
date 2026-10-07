//! Scratch population checkpoints use ordered little-endian f64 parameters.

use super::{scheduled_rate, ScratchConfig};
use crate::io::{load, load_optional, write_atomic, write_atomic_report};
use altd_sim::nn::network::{parameter_count, Network};
use altd_sim::training::TrainingRunner;
use serde_json::{json, Value};
use std::path::Path;

/// Install a checkpoint's population, RNG and mutation rate; returns its generation.
pub(super) fn install_checkpoint(
    runner: &mut TrainingRunner,
    config: &ScratchConfig,
    meta: &mut Value,
    dir: &Path,
) -> u64 {
    let generation = meta["generation"].as_u64().unwrap();
    let file = dir.join(meta["population_file"].as_str().unwrap());
    let bytes = std::fs::read(&file).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
    // Checkpoints without shape/population fields read them from the adjacent run.json.
    // Copy those fields into the metadata before a resume can replace run.json.
    if meta.get("shape").is_none() || meta.get("population").is_none() {
        let saved = load(&dir.join("run.json"));
        if meta.get("shape").is_none() {
            meta["shape"] = saved["shape"].clone();
        }
        if meta.get("population").is_none() {
            meta["population"] = saved["population"].clone();
        }
    }
    let saved_shape: Vec<usize> = serde_json::from_value(meta["shape"].clone()).expect("checkpoint network shape");
    assert_eq!(
        saved_shape, config.shape,
        "checkpoint network shape differs from --shape"
    );
    let saved_population = meta["population"].as_u64().expect("checkpoint population") as usize;
    assert!(saved_population > 0, "checkpoint population must be positive");
    let size = parameter_count(&saved_shape);
    let expected_bytes = saved_population
        .checked_mul(size)
        .and_then(|n| n.checked_mul(8))
        .expect("checkpoint population size overflow");
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "checkpoint population file size does not match its shape and population"
    );
    let values: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    // Keep checkpoint order and RNG state. Expansion repeats saved networks;
    // subsequent reproduction uses the requested population and settings.
    let networks: Vec<Network> = (0..config.population)
        .map(|i| {
            let start = (i % saved_population) * size;
            Network::from_vector(&saved_shape, values[start..start + size].to_vec())
        })
        .collect();
    runner.rng = altd_sim::training::TrainingRandom::from_json(&meta["rng"]);
    runner.settings.mutation_rate = scheduled_rate(config, generation as usize);
    runner.resume(&networks, generation);
    generation
}

/// `checkpoint_gNNNNN.bin`: every network's parameters as little-endian f64.
pub(super) fn write_checkpoint(dir: &Path, runner: &TrainingRunner, state: Value) {
    let generation = runner.generation;
    let name = format!("checkpoint_g{generation:05}.bin");
    let mut bytes = Vec::with_capacity(runner.agents.len() * runner.agents[0].network.params.len() * 8);
    for agent in &runner.agents {
        for value in &agent.network.params {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    write_atomic(&dir.join(&name), &bytes);
    let mut meta = state;
    meta["generation"] = json!(generation);
    meta["population_file"] = json!(name);
    meta["shape"] = json!(runner.agents[0].network.shape);
    meta["population"] = json!(runner.agents.len());
    meta["rng"] = runner.rng.to_json();
    meta["math_profile"] = json!(runner.world.math);
    let previous = load_optional(&dir.join("checkpoint.json"));
    write_atomic_report(&dir.join("checkpoint.json"), &meta);
    if let Some(old) = previous.as_ref().and_then(|p| p["population_file"].as_str()) {
        if old != name {
            let _ = std::fs::remove_file(dir.join(old));
        }
    }
}
