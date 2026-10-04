//! Population timing and optional before-turnover state dumps.

use super::inputs::{car_from_frame, frames, num, output_names, recorded_controls};
use super::io::write_report;
use super::platform::{peak_rss_kib, platform_json, process_cpu_seconds};
use altd_sim::nn::network::Network;
use altd_sim::track::world::{vector, World};
use altd_sim::training::evolution::{EvolutionSettings, METRIC_NAMES};
use altd_sim::training::pyrandom::PyRandom;
use altd_sim::training::{Mode, SensorLayout, TrainingAgent, TrainingRunner};
use serde_json::{json, Map, Value};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

fn agent_state(agent: &TrainingAgent) -> Value {
    let car = &agent.car;
    let metrics: Map<String, Value> = METRIC_NAMES
        .iter()
        .zip(agent.stats.metrics())
        .map(|(n, v)| (n.to_string(), json!(v)))
        .collect();
    json!({
        "position": [car.position.x, car.position.y],
        "velocity": [car.velocity.x, car.velocity.y],
        "rotation": car.rotation,
        "angular_velocity": car.angular_velocity,
        "boost_energy": car.boost_energy,
        "active": car.active,
        "collision_count": car.collision_count,
        "update_count": agent.stats.update_count,
        "metrics": metrics,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn bench(
    scene: &Value,
    network_data: &Value,
    model: &Value,
    spawn_trace: &Value,
    population: usize,
    ticks: u64,
    warmup_ticks: u64,
    seed: i64,
    batch_count: usize,
    spawn_index: usize,
    mode: Mode,
    threads: usize,
    dump_state: Option<&Path>,
) -> Value {
    assert!(population >= 1 && ticks >= 1, "population and ticks must be positive");
    let spawn_frames = frames(spawn_trace);
    assert!(spawn_index < spawn_frames.len(), "spawn index is outside the trace");
    let settings = EvolutionSettings {
        population,
        selection_size: 14,
        mutation_rate: 0.05,
        adaptive_mutation: true,
        weight_decay: 0.0001,
        preserve_parents: "on_custom".into(),
        preserve_parents_size: 7,
        ..EvolutionSettings::default()
    };
    let load_started = Instant::now();
    let world = Arc::new(World::from_scene(scene));
    let world_seconds = load_started.elapsed().as_secs_f64();
    let spawn = &spawn_frames[spawn_index];
    let mut runner = TrainingRunner::new(
        world.clone(),
        vector(&spawn["position"]),
        num(&spawn["rotation"]),
        SensorLayout::from_exports(network_data, model),
        &output_names(network_data),
        settings.clone(),
        PyRandom::new(seed),
        batch_count,
        0,
        false,
        false,
    );
    runner.mode = mode;
    let seed_network = Network::from_game_export(network_data);

    let started = Instant::now();
    runner.start(&seed_network);
    assert_eq!(
        runner.agents.len(),
        population,
        "generation size differs from requested population"
    );
    if spawn_index > 0 {
        let template = car_from_frame(&world, spawn, Some(&spawn_frames[spawn_index - 1]));
        let controls = recorded_controls(spawn);
        for agent in &mut runner.agents {
            agent.car.velocity = template.velocity;
            agent.car.angular_velocity = template.angular_velocity;
            agent.car.boost_energy = template.boost_energy;
            agent.controls = controls;
            for (wheel, source) in agent.car.wheels.iter_mut().zip(&template.wheels) {
                *wheel = *source;
            }
        }
        runner.tick = spawn_index as u64;
        runner.batch_index = (spawn_index * runner.batches_per_tick()) % 8;
    }
    let initialization_seconds = started.elapsed().as_secs_f64();

    let warmup_started = Instant::now();
    runner.advance(warmup_ticks, false);
    let warmup_seconds = warmup_started.elapsed().as_secs_f64();

    let cpu_started = process_cpu_seconds();
    let timed_started = Instant::now();
    runner.advance(ticks, false);
    let tick_seconds = timed_started.elapsed().as_secs_f64();
    let tick_cpu_seconds = process_cpu_seconds() - cpu_started;
    let active_after_ticks = runner.agents.iter().filter(|a| a.car.active).count();
    let contacts_after_ticks: u64 = runner.agents.iter().map(|a| a.car.collision_count).sum();
    let agents_state: Option<Vec<Value>> = dump_state.map(|_| runner.agents.iter().map(agent_state).collect());

    let turnover_started = Instant::now();
    runner.next_generation();
    let turnover_seconds = turnover_started.elapsed().as_secs_f64();
    assert_eq!(
        runner.agents.len(),
        population,
        "reproduction did not preserve population size"
    );
    if let (Some(path), Some(agents)) = (dump_state, agents_state) {
        let networks: Vec<&Vec<f64>> = runner.agents.iter().map(|a| &a.network.params).collect();
        write_report(path, &json!({"agents": agents, "next_networks": networks}));
    }

    let simulated_seconds = ticks as f64 / 60.0;
    let complete_generation_seconds = (warmup_ticks + ticks) as f64 / 60.0;
    let total_wall_seconds = initialization_seconds + warmup_seconds + tick_seconds + turnover_seconds;
    let mut result = json!({
        "population": population,
        "timed_ticks": ticks,
        "warmup_ticks": warmup_ticks,
        "batch_count": batch_count,
        "seed": seed,
        "spawn_index": spawn_index,
        "settings": settings.to_json(),
        "physics_shapes": world.track.shapes.len(),
        "network_shape": seed_network.shape,
        "network_parameters": seed_network.params.len(),
        "active_after_ticks": active_after_ticks,
        "contacts_after_ticks": contacts_after_ticks,
        "world_load_seconds": world_seconds,
        "initialization_seconds": initialization_seconds,
        "warmup_seconds": warmup_seconds,
        "timed_wall_seconds": tick_seconds,
        "timed_cpu_seconds": tick_cpu_seconds,
        "turnover_seconds": turnover_seconds,
        "simulated_seconds": simulated_seconds,
        "realtime_multiplier": simulated_seconds / tick_seconds,
        "wall_seconds_per_simulated_second": tick_seconds / simulated_seconds,
        "car_ticks_per_wall_second": (population as u64 * ticks) as f64 / tick_seconds,
        "complete_generation_simulated_seconds": complete_generation_seconds,
        "complete_generation_wall_seconds": total_wall_seconds,
        "complete_generation_realtime_multiplier": complete_generation_seconds / total_wall_seconds,
        "peak_rss_kib": peak_rss_kib(),
        "mode": match mode { Mode::Independent => "independent", Mode::Lockstep => "lockstep" },
    });
    let map = result.as_object_mut().unwrap();
    for (k, v) in platform_json(threads).as_object().unwrap() {
        map.insert(k.clone(), v.clone());
    }
    result
}
