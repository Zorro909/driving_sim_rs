//! Scratch generation loop, run artifacts, and CPU/GPU track activation.

use super::checkpoint::{install_checkpoint, write_checkpoint};
use super::stop::StopState;
use super::{scheduled_rate, ScratchConfig};
use crate::args::Schedule;
use crate::inputs::{elimination_options, frames, num, output_names};
use crate::io::{
    describe_input, load, load_optional, load_or, write_atomic, write_atomic_report, RALLY_MODEL, RALLY_NETWORK,
};
use altd_sim::math::pymath::py_sum;
use altd_sim::nn::network::Network;
use altd_sim::track::training_tracks::TrainingTrackBuffer;
use altd_sim::track::world::{vector, World};
use altd_sim::training::batch_evaluation::BatchEvaluation;
use altd_sim::training::evolution::EvolutionSettings;
use altd_sim::training::pyrandom::PyRandom;
use altd_sim::training::{Mode, SensorLayout, TrainingRandom, TrainingRunner};
use serde_json::{json, Value};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

fn activate_prepared_track<'a>(
    runner: &mut TrainingRunner,
    mut track: altd_sim::track::training_tracks::PreparedTrainingTrack,
    gpu: Option<&'a altd_sim::gpu::hip::Gpu>,
    sim: &mut Option<altd_sim::gpu::simulation::GpuSim<'a>>,
    world: &mut Option<altd_sim::gpu::hip::GpuWorld<'a>>,
) -> altd_sim::track::training_tracks::PreparedTrainingTrack {
    runner.replace_track(track.world.clone(), track.position, track.rotation);
    if let (Some(gpu), Some(sim)) = (gpu, sim.as_mut()) {
        let next = altd_sim::gpu::hip::GpuWorld::from_prepared(
            gpu,
            track.gpu_world.take().expect("buffered GPU track arrays"),
        );
        sim.set_world(&next);
        *world = Some(next);
    }
    track
}

fn write_progress(dir: &Path, generation: usize, track_index: usize, completed: usize, count: usize) {
    let progress = json!({"pid": std::process::id(), "generation": generation,
                          "track_index": track_index, "tracks_completed": completed,
                          "tracks_per_generation": count, "evaluating": completed < count});
    write_atomic(&dir.join("progress.json"), progress.to_string().as_bytes());
}

pub(super) fn final_candidate_leader(fitness: &[f64]) -> usize {
    (0..fitness.len()).fold(0, |best, i| if fitness[i] > fitness[best] { i } else { best })
}

pub(crate) fn train_scratch(config: &ScratchConfig, mode: Mode, threads: usize) {
    let scene = load(&config.scene);
    let network_data = load_or(config.network.as_deref(), &RALLY_NETWORK);
    let model = load_or(config.model.as_deref(), &RALLY_MODEL);
    let track_settings = config.tracks.settings();
    let shape = &config.shape;
    let inputs = network_data["inputs"].as_array().expect("inputs").len();
    let outputs = output_names(&network_data);
    assert!(
        shape.len() >= 2
            && shape.iter().all(|&n| n > 0)
            && shape[0] == inputs
            && shape[shape.len() - 1] == outputs.len(),
        "shape {shape:?} must start with the {inputs} sensor inputs and end with the {} outputs",
        outputs.len()
    );
    assert!(
        config.population >= 1 && config.generations >= 1 && config.ticks >= 1,
        "population, generations, and ticks must be positive"
    );
    assert!(
        config.mutation_start > 0.0 && config.mutation_end > 0.0,
        "mutation rates must be positive"
    );

    // Generalist Rally settings, optionally overridden.
    let mut settings = EvolutionSettings::from_mcp(&json!({
        "selection_algorithm": "tournament",
        "selection_size": 20,
        "crossover": "none",
        "adaptive_mutation": true,
        "weight_decay": 0.0,
        "preserve_parents": "on_custom",
        "preserve_parents_size": 4,
        "rewards": [{"metric": "total_score", "weight": 100, "type": "default"}],
    }));
    let settings_data = config.settings.as_deref().map(load).unwrap_or_else(|| json!({}));
    let settings_data = settings_data.get("settings").unwrap_or(&settings_data);
    let (eliminate_on_wall, idle_eliminate) =
        elimination_options(settings_data, config.eliminate_on_wall, config.idle_eliminate);
    if config.settings.is_some() {
        let data = settings_data;
        let mut merged = settings.to_json();
        for (k, v) in data.as_object().expect("settings object") {
            merged[k] = v.clone();
        }
        settings = EvolutionSettings::from_mcp(&merged);
    }
    if let Some(reward) = config.reward {
        settings.rewards = vec![reward.spec()];
    }
    settings.population = config.population;
    settings.mutation_rate = scheduled_rate(config, 0);
    settings
        .validate_algorithm()
        .unwrap_or_else(|e| panic!("settings: {e}"));

    let dir = &config.out_dir;
    let best_dir = dir.join("best_laps");
    std::fs::create_dir_all(&best_dir).unwrap_or_else(|e| panic!("{}: {e}", best_dir.display()));
    let mut run = json!({
        "simulator": "game-fidelity-cpu-v1",
        "eliminate_on_wall": eliminate_on_wall, "idle_eliminate": idle_eliminate,
        "scene": config.scene, "track": scene["track"]["name"], "spawn_trace": config.spawn_trace,
        "network_template": describe_input(config.network.as_deref(), &RALLY_NETWORK),
        "model": describe_input(config.model.as_deref(), &RALLY_MODEL), "shape": shape,
        "population": config.population, "generations": config.generations, "ticks": config.ticks,
        "mutation_start": config.mutation_start, "mutation_end": config.mutation_end,
        "schedule": match config.schedule { Schedule::Geometric => "geometric", Schedule::Linear => "linear" },
        "seed": config.seed, "init_network": config.init_network, "batch_count": config.batch_count, "settings": settings.to_json(),
        "stop": config.stop.to_json(),
        "track_mode": if track_settings.is_some() { "random" } else { "fixed" },
        "random_track_settings": track_settings,
        "track_buffer_size": config.tracks.track_buffer_size,
    });
    if track_settings.is_some() {
        run["track"] = json!("random");
        run["spawn_trace"] = Value::Null;
    }
    if let Some(path) = &config.game_rng_state {
        run["game_rng_state"] = load(path);
    }
    if let Some(path) = &config.init_population {
        run["init_population"] = json!(path);
    }
    let checkpoint = load_optional(&dir.join("checkpoint.json"));
    let initial_population_meta = config.init_population.as_deref().map(load);
    if config.resume {
        assert!(
            checkpoint.is_some(),
            "--resume: no checkpoint.json in {}",
            dir.display()
        );
    } else {
        assert!(
            !dir.join("run.json").exists(),
            "{} already holds a run; pass --resume or choose another --out-dir",
            dir.display()
        );
    }

    // A request left over from an earlier invocation must not end this one.
    let stop_request_path = dir.join("stop_request");
    let _ = std::fs::remove_file(&stop_request_path);

    let track_count = config
        .tracks
        .count(if config.resume { checkpoint.as_ref() } else { None });
    run["tracks_per_generation"] = json!(track_count);
    run["track_seed_version"] = json!(2);
    let started = Instant::now();
    let restored_meta = if config.resume {
        checkpoint.as_ref()
    } else {
        initial_population_meta.as_ref()
    };
    let starting_generation = restored_meta.and_then(|m| m["generation"].as_u64()).unwrap_or(0);
    let track_seed = if config.resume {
        checkpoint
            .as_ref()
            .and_then(|m| m["random_tracks"]["seed"].as_i64())
            .unwrap_or(config.seed)
    } else {
        config.seed
    };
    run["random_track_seed"] = track_settings.as_ref().map_or(Value::Null, |_| json!(track_seed));
    let track_buffer = track_settings.as_ref().map(|settings| {
        TrainingTrackBuffer::new_batched(
            scene.clone(),
            settings.clone(),
            track_seed,
            starting_generation,
            config.tracks.track_buffer_size,
            config.gpu,
            track_count,
        )
        .unwrap_or_else(|e| panic!("random tracks: {e}"))
    });
    let mut current_track = track_buffer
        .as_ref()
        .map(|b| b.next_track().unwrap_or_else(|e| panic!("random tracks: {e}")));
    let (world, position, rotation) = match &current_track {
        Some(track) => (track.world.clone(), track.position, track.rotation),
        None => {
            let path = config
                .spawn_trace
                .as_deref()
                .expect("--spawn-trace is required for fixed tracks");
            let spawn_trace = load(path);
            let spawn = &frames(&spawn_trace)[0];
            (
                Arc::new(World::from_scene(&scene)),
                vector(&spawn["position"]),
                num(&spawn["rotation"]),
            )
        }
    };
    let restoring = config.resume || config.init_population.is_some();
    let mut rng: TrainingRandom = match &config.game_rng_state {
        Some(path) => altd_sim::training::game_random::GameRandom::from_json(&load(path)).into(),
        None => PyRandom::new(config.seed).into(),
    };
    let seed_network = if restoring {
        None
    } else {
        let network = match &config.init_network {
            Some(path) => Network::from_game_export(&load(path)),
            None => rng.xavier(shape),
        };
        assert_eq!(&network.shape, shape, "--init-network shape differs from --shape");
        Some(network)
    };
    let mut runner = TrainingRunner::new(
        world,
        position,
        rotation,
        SensorLayout::from_exports(&network_data, &model),
        &outputs,
        settings,
        rng,
        config.batch_count,
        0,
        eliminate_on_wall,
        idle_eliminate,
    );
    runner.mode = mode;
    let gpu = config
        .gpu
        .then(|| altd_sim::gpu::hip::Gpu::open(None).unwrap_or_else(|e| panic!("--gpu: {e}")));
    let mut gpu_world = gpu.as_ref().map(|g| match current_track.as_mut() {
        Some(track) => {
            altd_sim::gpu::hip::GpuWorld::from_prepared(g, track.gpu_world.take().expect("buffered GPU track arrays"))
        }
        None => altd_sim::gpu::hip::GpuWorld::new(g, &runner.world),
    });
    let mut gpu_sim = gpu_world.as_ref().map(|w| {
        runner
            .gpu_sim(w, config.population)
            .unwrap_or_else(|e| panic!("--gpu: {e}"))
    });

    let log_path = dir.join("log.jsonl");
    let mut best_lap = f64::INFINITY;
    let mut best_lap_generation: Option<u64> = None;
    let mut best_network_file = Value::Null;
    let first_generation = if let Some(mut meta) = checkpoint.filter(|_| config.resume) {
        let generation = install_checkpoint(&mut runner, config, &mut meta, dir);
        // Persist restored shape/population fields before run.json adopts the new options.
        write_atomic_report(&dir.join("checkpoint.json"), &meta);
        best_lap = meta["best_lap_s"].as_f64().unwrap_or(f64::INFINITY);
        best_lap_generation = meta["best_lap_generation"].as_u64();
        // Drop log lines of generations the checkpoint will redo.
        let kept: String = std::fs::read_to_string(&log_path)
            .unwrap_or_default()
            .lines()
            .filter(|line| {
                serde_json::from_str::<Value>(line).is_ok_and(|v| v["generation"].as_u64() < Some(generation))
            })
            .map(|line| format!("{line}\n"))
            .collect();
        write_atomic(&log_path, kept.as_bytes());
        best_network_file = meta["best_network_file"].clone();
        if best_network_file.is_null() {
            best_network_file = kept
                .lines()
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .filter_map(|row| row["saved"].as_str().map(str::to_owned))
                .next_back()
                .map_or(Value::Null, |name| json!(name));
        }
        if let Some(name) = best_network_file.as_str() {
            if let Ok(bytes) = std::fs::read(dir.join(name)) {
                write_atomic(&dir.join("best.json"), &bytes);
            } else {
                let _ = std::fs::remove_file(dir.join("best.json"));
            }
        } else {
            let _ = std::fs::remove_file(dir.join("best.json"));
        }
        for artifact_dir in ["tracks", "best_laps"] {
            if let Ok(files) = std::fs::read_dir(dir.join(artifact_dir)) {
                for file in files.flatten() {
                    let name = file.file_name().to_string_lossy().into_owned();
                    let prefix = name.strip_prefix('g').and_then(|name| name.split(['_', '.']).next());
                    if prefix
                        .and_then(|value| value.parse::<u64>().ok())
                        .is_some_and(|value| value >= generation)
                    {
                        let _ = std::fs::remove_file(file.path());
                    }
                }
            }
        }
        println!("resumed {} at generation {generation}", dir.display());
        generation as usize
    } else if let Some(path) = &config.init_population {
        let mut meta = initial_population_meta.unwrap();
        let generation = install_checkpoint(&mut runner, config, &mut meta, path.parent().unwrap_or(Path::new(".")));
        std::fs::write(&log_path, "").expect("log file");
        println!(
            "continuing the population of {} at generation {generation}",
            path.display()
        );
        generation as usize
    } else {
        runner.start(seed_network.as_ref().unwrap());
        std::fs::write(&log_path, "").expect("log file");
        0
    };
    write_atomic_report(&dir.join("run.json"), &run);
    assert_eq!(
        runner.agents.len(),
        config.population,
        "generation size differs from requested population"
    );
    let mut log = std::fs::OpenOptions::new()
        .append(true)
        .open(&log_path)
        .expect("log file");
    println!(
        "{} on {}: population {}, {} generations x {:.1} s, mutation {} -> {} ({}), {}, setup {:.2} s",
        dir.display(),
        run["track"].as_str().unwrap_or("?"),
        config.population,
        config.generations,
        config.ticks as f64 / 60.0,
        config.mutation_start,
        config.mutation_end,
        run["schedule"].as_str().unwrap(),
        if config.gpu {
            "gpu".to_string()
        } else {
            format!("{threads} threads")
        },
        started.elapsed().as_secs_f64()
    );

    let fmt_lap = |lap: Option<f64>| lap.map_or("   -   ".to_string(), |t| format!("{t:7.2}"));
    let mut stop_state = StopState {
        best_lap: None,
        best_score: f64::NEG_INFINITY,
        stale: 0,
    };
    for generation in first_generation..config.generations {
        let generation_started = Instant::now();
        let rate = runner.settings.mutation_rate;
        let mut batch = BatchEvaluation::new(config.population);
        let mut track_reports = Vec::new();
        let mut simulated_ticks = 0u64;
        let mut active_cars = 0usize;
        let mut lap_time_sum = 0.0;
        let mut lap_count = 0usize;
        let mut median_lap = None;
        let mut saved = Value::Null;
        let mut generation_best: Option<f64> = None;
        for track_index in 0..track_count {
            if track_index > 0 {
                let track = track_buffer
                    .as_ref()
                    .unwrap()
                    .next_track()
                    .unwrap_or_else(|e| panic!("random tracks: {e}"));
                current_track = Some(activate_prepared_track(
                    &mut runner,
                    track,
                    gpu.as_ref(),
                    &mut gpu_sim,
                    &mut gpu_world,
                ));
                runner.reset_evaluation();
            }
            write_progress(dir, generation, track_index, track_index, track_count);
            let track_file = current_track.as_ref().map(|track| {
                assert_eq!((track.generation, track.track_index), (generation as u64, track_index));
                let name = if track_count == 1 {
                    format!("tracks/g{generation:05}.track.json")
                } else {
                    format!("tracks/g{generation:05}_t{track_index:03}.track.json")
                };
                std::fs::create_dir_all(dir.join("tracks")).expect("track directory");
                write_atomic(
                    &dir.join(&name),
                    (serde_json::to_string(&track.track).unwrap() + "\n").as_bytes(),
                );
                name
            });
            let track_config = current_track.as_ref().map(|track| track.track.config.clone());
            match (&mut gpu_sim, &gpu_world) {
                (Some(sim), Some(world)) => {
                    runner
                        .advance_generation_gpu(sim, world, config.ticks)
                        .unwrap_or_else(|e| panic!("--gpu: {e}"));
                }
                _ => {
                    runner.advance_generation(config.ticks);
                }
            }
            simulated_ticks += runner.tick;
            let active = runner.agents.iter().filter(|agent| agent.car.active).count();
            active_cars += active;
            let laps: Vec<Option<f64>> = runner
                .agents
                .iter()
                .map(|a| a.stats.best_lap_time.filter(|&t| t > 0.0))
                .collect();
            let scores: Vec<f64> = runner.agents.iter().map(|a| a.stats.total_score).collect();
            let lapped: Vec<f64> = laps.iter().flatten().copied().collect();
            let leader =
                (0..laps.len())
                    .filter(|&i| laps[i].is_some())
                    .fold(None, |best: Option<usize>, i| match best {
                        Some(b) if (laps[b], -scores[b]) <= (laps[i], -scores[i]) => Some(b),
                        _ => Some(i),
                    });
            let track_best = leader.map(|i| laps[i].unwrap());
            if let Some(lap) = track_best {
                generation_best = Some(generation_best.map_or(lap, |best| best.min(lap)));
            }
            batch.record(&runner.results(), &runner.settings.rewards);
            track_reports.push(json!({"track_index": track_index, "track_file": track_file, "track_config": track_config,
                "simulated_ticks": runner.tick, "active_cars": active, "lapped_cars": lapped.len(), "best_lap_s": track_best,
                "best_score": scores.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                "mean_score": py_sum(scores.iter().copied()) / scores.len() as f64}));
            lap_time_sum += py_sum(lapped.iter().copied());
            lap_count += lapped.len();
            if track_count == 1 && !lapped.is_empty() {
                let mut sorted = lapped;
                sorted.sort_by(f64::total_cmp);
                median_lap = Some(sorted[sorted.len() / 2]);
            }
            if let (Some(i), Some(lap)) = (leader, track_best) {
                if lap < best_lap {
                    best_lap = lap;
                    best_lap_generation = Some(generation as u64);
                    let mut export = runner.agents[i].network.to_json();
                    export["inputs"] = network_data["inputs"].clone();
                    export["outputs"] = network_data["outputs"].clone();
                    export["training"] = json!({
                        "generation": generation, "car": i, "inference_batch": i / config.population.div_ceil(8),
                        "reused_vehicle": generation > 0, "best_lap_s": lap, "total_score": scores[i],
                        "mutation_rate": rate, "track": run["track"], "seed": config.seed,
                        "track_file": track_file, "track_config": track_config,
                        "track_index": track_index, "tracks_per_generation": track_count,
                    });
                    let text = serde_json::to_string_pretty(&export).unwrap() + "\n";
                    let name = if track_count == 1 {
                        format!("g{generation:05}_{lap:.2}s.json")
                    } else {
                        format!("g{generation:05}_t{track_index:03}_{lap:.2}s.json")
                    };
                    write_atomic(&best_dir.join(&name), text.as_bytes());
                    write_atomic(&dir.join("best.json"), text.as_bytes());
                    saved = json!(format!("best_laps/{name}"));
                    best_network_file = saved.clone();
                }
            }
            write_progress(dir, generation, track_index, track_index + 1, track_count);
        }
        let simulate_seconds = generation_started.elapsed().as_secs_f64();
        let summary = batch.finish();
        let best_score = summary.mean_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean_score = py_sum(summary.mean_scores.iter().copied()) / config.population as f64;
        let mean_lap = (lap_count > 0).then(|| lap_time_sum / lap_count as f64);
        let batch_lap = if track_count == 1 {
            generation_best
        } else {
            summary.best_mean_lap_s
        };
        let lapped_count = summary.lapped_all_tracks;
        let stop_request = stop_request_path.exists();
        let mut stop_reason = config.stop.check(
            &mut stop_state,
            generation,
            batch_lap,
            best_score,
            lapped_count,
            config.population,
            stop_request,
        );
        if stop_request {
            let _ = std::fs::remove_file(&stop_request_path);
        }
        if stop_reason.is_none() && generation + 1 == config.generations {
            stop_reason = Some(json!({"condition": "generations", "generation": generation,
                                      "value": generation + 1, "threshold": config.generations}));
        }
        let last = stop_reason.is_some();
        let final_checkpoint = last && config.checkpoint_every > 0;
        let candidate = if last && config.save_final_candidate {
            let leader = final_candidate_leader(&summary.fitness);
            let mut export = runner.agents[leader].network.to_json();
            export["schema_version"] = json!(1);
            export["inputs"] = network_data["inputs"].clone();
            export["outputs"] = network_data["outputs"].clone();
            export["training"] = json!({"generation": generation, "car": leader,
                "selection_rule": "complete-batch-fitness-lowest-index-tie", "fitness": summary.fitness[leader],
                "rewards": runner.settings.to_json()["rewards"], "total_score": summary.mean_scores[leader],
                "metric_totals": summary.metric_totals[leader], "update_count": summary.update_counts[leader],
                "lapped_tracks": summary.lap_counts[leader], "tracks_per_generation": track_count,
                "best_lap_s": if track_count == 1 { runner.agents[leader].stats.best_lap_time } else { None },
                "mutation_rate": rate, "seed": config.seed});
            let bytes = (serde_json::to_string_pretty(&export).unwrap() + "\n").into_bytes();
            write_atomic(&dir.join("candidate.json"), &bytes);
            json!({"file": "candidate.json", "sha256": altd_sim::training::evaluation::sha256(&bytes), "generation": generation})
        } else {
            Value::Null
        };
        let turnover_started = Instant::now();
        if !last || final_checkpoint {
            runner.settings.mutation_rate = scheduled_rate(config, generation + 1);
            if let Some(buffer) = &track_buffer {
                let track = buffer.next_track().unwrap_or_else(|e| panic!("random tracks: {e}"));
                current_track = Some(activate_prepared_track(
                    &mut runner,
                    track,
                    gpu.as_ref(),
                    &mut gpu_sim,
                    &mut gpu_world,
                ));
            }
            if let Some(sim) = &mut gpu_sim {
                if track_count == 1 {
                    runner.next_generation_gpu(sim)
                } else {
                    runner.next_generation_gpu_with_fitness(sim, summary.fitness.clone())
                }
                .unwrap_or_else(|e| panic!("--gpu: {e}"));
            } else if track_count == 1 {
                runner.next_generation();
            } else {
                runner.next_generation_with_fitness(summary.fitness.clone());
            }
        }
        let turnover_seconds = turnover_started.elapsed().as_secs_f64();
        let wall = generation_started.elapsed().as_secs_f64();
        let record = json!({
            "schema_version": 2, "generation": generation, "tracks_per_generation": track_count,
            "tracks": track_reports, "track_file": track_reports[0]["track_file"], "track_config": track_reports[0]["track_config"],
            "mutation_rate": rate, "best_lap_s": generation_best, "best_batch_mean_lap_s": batch_lap,
            "mean_best_lap_s": mean_lap, "median_best_lap_s": median_lap,
            "lapped_cars": lapped_count, "lapped_all_tracks": lapped_count,
            "best_score": best_score, "best_mean_score": best_score, "mean_score": mean_score,
            "best_fitness": summary.fitness.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            "mean_fitness": py_sum(summary.fitness.iter().copied()) / config.population as f64,
            "total_metric_values": (0..14).map(|metric| py_sum(summary.metric_totals.iter().map(|row| row[metric]))).collect::<Vec<_>>(),
            "total_update_count": summary.update_counts.iter().sum::<u64>(),
            "all_time_best_lap_s": best_lap.is_finite().then_some(best_lap), "all_time_best_lap_generation": best_lap_generation,
            "saved": saved, "simulate_seconds": simulate_seconds, "simulated_ticks": simulated_ticks,
            "active_cars": active_cars / track_count, "turnover_seconds": turnover_seconds,
            "car_seconds_per_second": simulated_ticks as f64 / 60.0 * config.population as f64 / wall,
        });
        writeln!(log, "{record}").and_then(|_| log.flush()).expect("write log");
        println!("gen {generation:5}  tracks {track_count}  mut {rate:.5}  best lap {}  batch lap {}  lapped {lapped_count}/{}  score {best_score:.1}/{mean_score:.1}  {wall:.2} s",
                 fmt_lap(generation_best), fmt_lap(batch_lap), config.population);
        if final_checkpoint || (!last && config.checkpoint_every > 0 && (generation + 1) % config.checkpoint_every == 0)
        {
            write_checkpoint(
                dir,
                &runner,
                json!({
                    "best_lap_s": best_lap.is_finite().then_some(best_lap), "best_lap_generation": best_lap_generation,
                    "best_network_file": best_network_file, "tracks_per_generation": track_count, "track_seed_version": 2,
                    "random_tracks": track_settings.as_ref().map(|settings| json!({"seed": track_seed, "settings": settings})),
                    "stop_reason": if last { stop_reason.clone().unwrap() } else { Value::Null },
                    "candidate": candidate,
                }),
            );
        }
        if let Some(reason) = &stop_reason {
            println!("stopped: {reason}");
            break;
        }
    }
    let _ = std::fs::remove_file(dir.join("progress.json"));
    println!(
        "done: best lap {} (generation {}), {:.1} h",
        fmt_lap(best_lap.is_finite().then_some(best_lap)),
        best_lap_generation.map_or("-".into(), |g| g.to_string()),
        started.elapsed().as_secs_f64() / 3600.0
    );
}
