//! CPU simulator CLI, training checkpoints, and recorded-game comparisons.
//!
//! Reports compare trajectories, controller outputs, sensors, and path scores.
//! Simulation arithmetic follows the original game runtime.

mod args;
mod bench;
mod comparisons;
mod inputs;
mod io;
mod platform;
mod scratch;
mod train;

use args::{flag, Cli, Command, TrackMode};
use bench::bench;
use comparisons::{
    compare_closed_loop, compare_network, compare_one_step, compare_score, compare_sensors, compare_trace,
};
use io::{load, load_or, print_without, write_report, RALLY_MODEL};
use scratch::{train_scratch, ScratchConfig, StopRules};
use train::train;

use altd_sim::track::world::World;
use altd_sim::training::Mode;
use clap::Parser;
use serde_json::{json, Value};
use std::sync::Arc;

fn main() {
    let cli = Cli::parse();
    let libm = match cli
        .libm
        .map(Into::into)
        .map(Some)
        .map_or_else(altd_sim::math::libm_from_env, Ok)
    {
        Ok(libm) => libm.unwrap_or(altd_sim::math::Libm::Proton),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = altd_sim::math::set_libm(libm) {
        eprintln!("--libm {}: {e}", libm.name());
        std::process::exit(2);
    }
    let mut pool = rayon::ThreadPoolBuilder::new();
    if let Some(threads) = cli.threads {
        pool = pool.num_threads(threads);
    }
    pool.build_global().expect("thread pool");
    let threads = rayon::current_num_threads();
    let mode: Mode = cli.mode.into();
    match cli.command {
        Command::Bench {
            scene,
            network,
            model,
            spawn_trace,
            population,
            ticks,
            warmup_ticks,
            seed,
            spawn_index,
            batch_count,
            report,
            dump_state,
        } => {
            let result = bench(
                &load(&scene),
                &load(&network),
                &load_or(model.as_deref(), &RALLY_MODEL),
                &load(&spawn_trace),
                population,
                ticks,
                warmup_ticks,
                seed,
                batch_count,
                spawn_index,
                mode,
                threads,
                dump_state.as_deref(),
            );
            if let Some(report) = report {
                write_report(&report, &result);
            }
            println!("{}", serde_json::to_string_pretty(&result).unwrap());
        }
        Command::Train {
            tracks,
            scene,
            network,
            model,
            spawn_trace,
            output,
            settings,
            population,
            generations,
            ticks,
            seed,
            batch_count,
            eliminate_on_wall,
            no_eliminate_on_wall,
            idle_eliminate,
            no_idle_eliminate,
            game_rng_state,
        } => {
            if tracks.track_mode == TrackMode::Fixed && spawn_trace.is_none() {
                clap::Error::raw(
                    clap::error::ErrorKind::MissingRequiredArgument,
                    "--spawn-trace is required with --track-mode fixed",
                )
                .exit();
            }
            let settings = settings.as_deref().map(load);
            let spawn_data = if tracks.track_mode == TrackMode::Fixed {
                load(spawn_trace.as_deref().unwrap())
            } else {
                Value::Null
            };
            let (result, timing) = train(
                &load(&scene),
                &load(&network),
                &load(&model),
                &spawn_data,
                settings.as_ref(),
                population,
                generations,
                ticks,
                seed,
                batch_count,
                flag(eliminate_on_wall, no_eliminate_on_wall),
                flag(idle_eliminate, no_idle_eliminate),
                mode,
                threads,
                game_rng_state.as_deref().map(load).as_ref(),
                &tracks,
            );
            write_report(&output, &result);
            let history = result["history"].as_array().unwrap();
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "generations": history.len(),
                    "best_score": result["best_score"],
                    "last_generation": history.last(),
                    "timing": timing,
                }))
                .unwrap()
            );
        }
        Command::TrainScratch {
            tracks,
            scene,
            spawn_trace,
            network,
            model,
            out_dir,
            shape,
            population,
            generations,
            ticks,
            mutation_start,
            mutation_end,
            schedule,
            settings,
            reward,
            seed,
            init_network,
            init_population,
            game_rng_state,
            batch_count,
            checkpoint_every,
            save_final_candidate,
            resume,
            eliminate_on_wall,
            no_eliminate_on_wall,
            idle_eliminate,
            no_idle_eliminate,
            gpu,
            stop_score_above,
            stop_lap_below,
            stop_lapped_percent,
            stop_plateau,
            plateau_metric,
        } => {
            let stop = StopRules {
                score_above: stop_score_above,
                lap_below: stop_lap_below,
                lapped_percent: stop_lapped_percent,
                plateau: stop_plateau,
                plateau_metric,
            };
            let config = ScratchConfig {
                tracks,
                scene,
                spawn_trace,
                network,
                model,
                out_dir,
                shape,
                population,
                generations,
                ticks,
                mutation_start,
                mutation_end,
                schedule,
                settings,
                reward,
                seed,
                init_network,
                init_population,
                game_rng_state,
                batch_count,
                checkpoint_every,
                save_final_candidate,
                resume,
                stop,
                eliminate_on_wall: flag(eliminate_on_wall, no_eliminate_on_wall),
                idle_eliminate: flag(idle_eliminate, no_idle_eliminate),
                gpu,
            };
            train_scratch(&config, mode, threads);
        }
        Command::Evaluate {
            network,
            model,
            suite,
            report,
        } => match altd_sim::training::evaluation::evaluate(&network, &model, &suite, &report) {
            Ok(result) => println!("{}", serde_json::to_string_pretty(&result).unwrap()),
            Err(error) => {
                eprintln!("evaluation: {error}");
                std::process::exit(1);
            }
        },
        Command::CompareTrace {
            scene,
            trace,
            report,
            start_index,
            end_index,
        } => {
            let world = World::from_scene(&load(&scene));
            let result = compare_trace(&world, &load(&trace), start_index, end_index);
            write_report(&report, &result);
            print_without(&result, "frames");
        }
        Command::CompareOneStep { scene, trace, report } => {
            let world = World::from_scene(&load(&scene));
            let result = compare_one_step(&world, &load(&trace));
            write_report(&report, &result);
            print_without(&result, "rows");
        }
        Command::CompareClosedLoop {
            scene,
            trace,
            network,
            model,
            report,
            batch_count,
            new_vehicle,
        } => {
            let world = Arc::new(World::from_scene(&load(&scene)));
            let result = compare_closed_loop(
                &world,
                &load(&trace),
                &load(&network),
                &load(&model),
                batch_count,
                !new_vehicle,
                mode,
            );
            write_report(&report, &result);
            print_without(&result, "rows");
        }
        Command::CompareNetwork {
            network,
            trace,
            report,
            first_tick,
        } => {
            let result = compare_network(&load(&network), &load(&trace), first_tick);
            write_report(&report, &result);
            print_without(&result, "outputs");
        }
        Command::CompareSensors {
            scene,
            trace,
            model,
            trajectory_report,
            sensor_report,
            all_frames,
        } => {
            let world = World::from_scene(&load(&scene));
            let result = compare_sensors(
                &world,
                &load(&trace),
                &load(&model),
                &load(&trajectory_report),
                all_frames,
            );
            write_report(&sensor_report, &result);
            print_without(&result, "sensors");
        }
        Command::CompareScore { scene, trace, report } => {
            let world = World::from_scene(&load(&scene));
            let result = compare_score(&world, &load(&trace));
            write_report(&report, &result);
            print_without(&result, "rows");
        }
        #[cfg(feature = "server")]
        Command::Serve { port, allow_origin } => serve(port, allow_origin, threads),
        Command::GpuInfo { library } => {
            let gpu = altd_sim::gpu::hip::Gpu::open(library.as_deref()).unwrap_or_else(|error| {
                eprintln!("gpu-info: {error}");
                std::process::exit(1);
            });
            gpu.check_layout();
            let info = json!({"version": platform::VERSION, "library": gpu.path(), "layout": "ok"});
            println!("{}", serde_json::to_string_pretty(&info).unwrap());
        }
    }
}

#[cfg(feature = "server")]
fn serve(port: u16, allow_origin: Vec<String>, threads: usize) {
    use altd_sim::server::{Config, Server, DEFAULT_ORIGIN, PATH};
    let origins = if allow_origin.is_empty() {
        vec![DEFAULT_ORIGIN.to_string()]
    } else {
        allow_origin
    };
    let server = Server::bind(Config {
        port,
        origins: origins.clone(),
    })
    .unwrap_or_else(|e| {
        eprintln!("altd-sim serve: cannot listen on 127.0.0.1:{port}: {e}");
        std::process::exit(1);
    });
    let address = server.local_addr().expect("listener address");
    let hip = server.hip_status();
    let hip = match hip["available"].as_bool() {
        Some(true) => format!("available ({})", hip["device"].as_str().unwrap_or("?")),
        _ => format!("unavailable ({})", hip["reason"].as_str().unwrap_or("?")),
    };
    eprintln!("altd-sim serve: listening on ws://{address}{PATH}");
    eprintln!("  allowed origins: {}", origins.join(", "));
    eprintln!("  CPU threads: {threads}");
    eprintln!("  math: {}", altd_sim::math::libm().name());
    eprintln!("  HIP: {hip}");
    if let Err(e) = server.run() {
        eprintln!("altd-sim serve: {e}");
        std::process::exit(1);
    }
}
