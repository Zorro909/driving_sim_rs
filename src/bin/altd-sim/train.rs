//! Training an imported network and returning the generation history.

use super::args::TrackOptions;
use super::inputs::{elimination_options, frames, num, output_names};
use super::platform::platform_json;
use altd_sim::math::profile::MathProfile;
use altd_sim::math::pymath::py_sum;
use altd_sim::nn::network::Network;
use altd_sim::track::training_tracks::TrainingTrackBuffer;
use altd_sim::track::world::{vector, World};
use altd_sim::training::batch_evaluation::BatchEvaluation;
use altd_sim::training::evolution::EvolutionSettings;
use altd_sim::training::pyrandom::PyRandom;
use altd_sim::training::{Mode, SensorLayout, TrainingRandom, TrainingRunner};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
pub(super) fn train(
    scene: &Value,
    network_data: &Value,
    model: &Value,
    spawn_trace: &Value,
    settings_file: Option<&Value>,
    population: Option<usize>,
    generations: usize,
    ticks: Option<u64>,
    seed: i64,
    batch_count: usize,
    eliminate_on_wall: Option<bool>,
    idle_eliminate: Option<bool>,
    mode: Mode,
    threads: usize,
    game_rng_state: Option<&Value>,
    tracks: &TrackOptions,
    math: MathProfile,
) -> (Value, Value) {
    let empty = json!({});
    let settings_data = settings_file.unwrap_or(&empty);
    let settings_data = settings_data.get("settings").unwrap_or(settings_data);
    let mut settings = EvolutionSettings::from_mcp(settings_data);
    let population = population.unwrap_or(if settings_file.is_some() {
        settings.population
    } else {
        8
    });
    let ticks = ticks.unwrap_or_else(|| {
        settings_data
            .get("time_limit_s")
            .map_or(300, |v| (num(v) * 60.0).round_ties_even() as u64)
    });
    let (eliminate_on_wall, idle_eliminate) = elimination_options(settings_data, eliminate_on_wall, idle_eliminate);
    assert!(
        population >= 1 && generations >= 1 && ticks >= 1,
        "population, generations, and ticks must be positive"
    );
    settings.population = population;
    settings
        .validate_algorithm()
        .unwrap_or_else(|e| panic!("settings: {e}"));
    let started = Instant::now();
    let track_settings = tracks.settings();
    let track_count = tracks.count(None);
    let track_buffer = track_settings.as_ref().map(|settings| {
        TrainingTrackBuffer::new_batched(
            scene.clone(),
            settings.clone(),
            seed,
            0,
            tracks.track_buffer_size,
            false,
            track_count,
            math,
        )
        .unwrap_or_else(|e| panic!("random tracks: {e}"))
    });
    let mut current_track = track_buffer
        .as_ref()
        .map(|b| b.next_track().unwrap_or_else(|e| panic!("random tracks: {e}")));
    let (world, position, rotation) = match &current_track {
        Some(track) => (track.world.clone(), track.position, track.rotation),
        None => {
            let spawn = &frames(spawn_trace)[0];
            (
                Arc::new(World::from_scene_with(scene, math)),
                vector(&spawn["position"]),
                num(&spawn["rotation"]),
            )
        }
    };
    let world_seconds = started.elapsed().as_secs_f64();
    let seed_network = Network::from_game_export(network_data);
    let mut runner = TrainingRunner::new(
        world,
        position,
        rotation,
        SensorLayout::from_exports(network_data, model, math),
        &output_names(network_data),
        settings.clone(),
        game_rng_state.map_or_else(
            || TrainingRandom::from(PyRandom::new(seed)),
            |v| altd_sim::training::game_random::GameRandom::from_json(v).into(),
        ),
        batch_count,
        0,
        eliminate_on_wall,
        idle_eliminate,
    );
    runner.mode = mode;
    runner.start(&seed_network);
    let mut history = Vec::new();
    let mut best_score = f64::NEG_INFINITY;
    let mut best_network = seed_network.clone();
    let mut simulated_ticks = 0u64;
    let mut simulate_seconds = 0.0;
    for generation in 0..generations {
        let mut batch = BatchEvaluation::new(population);
        let mut track_reports = Vec::new();
        let mut generation_ticks = 0;
        for track_index in 0..track_count {
            if track_index > 0 {
                let track = track_buffer
                    .as_ref()
                    .unwrap()
                    .next_track()
                    .unwrap_or_else(|e| panic!("random tracks: {e}"));
                runner.replace_track(track.world.clone(), track.position, track.rotation);
                runner.reset_evaluation();
                current_track = Some(track);
            }
            if let Some(track) = &current_track {
                assert_eq!((track.generation, track.track_index), (generation as u64, track_index));
            }
            let tick_started = Instant::now();
            runner.advance_generation(ticks);
            simulate_seconds += tick_started.elapsed().as_secs_f64();
            simulated_ticks += runner.tick;
            generation_ticks += runner.tick;
            batch.record(&runner.results(), &settings.rewards);
            track_reports.push(json!({"track_index": track_index, "ticks": runner.tick,
                "track": current_track.as_ref().map(|t| &t.track)}));
        }
        let summary = batch.finish();
        let first_max =
            |values: &[f64]| (1..values.len()).fold(0, |best, i| if values[i] > values[best] { i } else { best });
        let leader = first_max(&summary.fitness);
        let best_raw = first_max(&summary.mean_scores);
        if summary.mean_scores[best_raw] > best_score {
            best_score = summary.mean_scores[best_raw];
            best_network = runner.agents[best_raw].network.clone();
        }
        history.push(json!({
            "generation": generation, "ticks": generation_ticks, "tracks_per_generation": track_count,
            "best_reward": summary.fitness[leader], "best_score": summary.mean_scores[best_raw],
            "mean_score": py_sum(summary.mean_scores.iter().copied()) / population as f64,
            "total_collision_updates": py_sum(summary.metric_totals.iter().map(|r| r[11])),
            "lapped_all_tracks": summary.lapped_all_tracks, "best_batch_mean_lap_s": summary.best_mean_lap_s,
            "track": track_reports[0]["track"], "tracks": track_reports,
        }));
        if generation + 1 < generations {
            if let Some(buffer) = &track_buffer {
                let track = buffer.next_track().unwrap_or_else(|e| panic!("random tracks: {e}"));
                runner.replace_track(track.world.clone(), track.position, track.rotation);
                current_track = Some(track);
            }
            if track_count == 1 {
                runner.next_generation();
            } else {
                runner.next_generation_with_fitness(summary.fitness);
            }
        }
    }
    let total_seconds = started.elapsed().as_secs_f64();
    let mut best = best_network.to_json();
    best["inputs"] = network_data["inputs"].clone();
    best["outputs"] = network_data["outputs"].clone();
    let output = json!({
        "seed": seed,
        "math_profile": math,
        "settings": settings.to_json(),
        "ticks_limit": ticks, "tracks_per_generation": track_count, "track_seed_version": 2,
        "eliminate_on_wall": eliminate_on_wall,
        "idle_eliminate": idle_eliminate,
        "history": history,
        "best_score": best_score,
        "best_network": best,
        "track_mode": if track_settings.is_some() { "random" } else { "fixed" },
        "random_track_settings": track_settings,
    });
    let simulated_seconds = simulated_ticks as f64 / 60.0;
    let mut timing = json!({
        "world_load_seconds": world_seconds,
        "simulation_wall_seconds": simulate_seconds,
        "total_wall_seconds": total_seconds,
        "simulated_seconds_per_car": simulated_seconds,
        "realtime_multiplier": simulated_seconds / simulate_seconds,
        "complete_realtime_multiplier": simulated_seconds / total_seconds,
        "population": population,
        "mode": match mode { Mode::Independent => "independent", Mode::Lockstep => "lockstep" },
    });
    for (k, v) in platform_json(threads).as_object().unwrap() {
        timing[k] = v.clone();
    }
    (output, timing)
}
