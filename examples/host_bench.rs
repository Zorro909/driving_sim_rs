//! Host-side timings of the GPU training loop's CPU stages on a self-contained
//! generated Formula track: state export for upload, reproduction
//! and installation. No GPU is needed; `ALTD_BENCH_CARS` sets the population.
use altd_sim::gpu::simulation::{self as gpu_sim, GpuAgent, GpuCar, SurfaceTable};
use altd_sim::nn::network::Network;
use altd_sim::track::world::World;
use altd_sim::training::evolution::EvolutionSettings;
use altd_sim::training::pyrandom::PyRandom;
use altd_sim::training::{SensorLayout, TrainingRunner};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

#[path = "support/generated.rs"]
mod generated;

fn main() {
    let cars: usize = std::env::var("ALTD_BENCH_CARS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32768);
    let rounds: usize = std::env::var("ALTD_BENCH_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let scene = generated::scene("formula", 0);
    let template = generated::network("formula");
    let model = generated::model("formula");
    let outputs: Vec<String> = template["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    let shape = [20, 16, 16, 16, 16, 12, 12, 8, 5];
    let mut settings = EvolutionSettings::from_mcp(&serde_json::json!({
        "selection_algorithm": "tournament", "selection_size": 20, "crossover": "none",
        "adaptive_mutation": true, "weight_decay": 0.0, "preserve_parents": "on_custom", "preserve_parents_size": 4,
    }));
    settings.population = cars;
    settings.mutation_rate = 0.4;
    let world = Arc::new(World::from_scene(&scene));
    let mut rng = PyRandom::new(1);
    let seed = Network::xavier(&shape, &mut rng);
    let mut runner = TrainingRunner::new(
        world.clone(),
        altd_sim::track::world::vector(&scene["reset_position"]),
        scene["reset_rotation"].as_f64().unwrap(),
        SensorLayout::from_exports(&template, &model, altd_sim::math::profile::MathProfile::Proton),
        &outputs,
        settings,
        rng,
        2,
        0,
        true,
        true,
    );
    let t = Instant::now();
    runner.start(&seed);
    println!(
        "fixture: {cars} cars, {} params each, start {:.3} s, rayon {} threads",
        seed.params.len(),
        t.elapsed().as_secs_f64(),
        rayon::current_num_threads()
    );
    // Drive a short window so cars have contacts, pairs and statistics.
    let t = Instant::now();
    let executed = runner.advance(120, false);
    println!("cpu window: {executed} ticks in {:.3} s", t.elapsed().as_secs_f64());
    let surfaces: SurfaceTable = SurfaceTable::new(&world.vehicle);
    let vehicle = &world.vehicle;

    for round in 0..rounds {
        // State export into fresh vectors through Result collection.
        let t = Instant::now();
        let cars_out = runner
            .agents
            .par_iter()
            .map(|a| a.car.gpu_export(vehicle, &surfaces))
            .collect::<Result<Vec<GpuCar>, _>>()
            .unwrap();
        let agents_out = runner
            .agents
            .par_iter()
            .map(gpu_sim::agent_export)
            .collect::<Result<Vec<GpuAgent>, _>>()
            .unwrap();
        let export_collect = t.elapsed().as_secs_f64();
        std::hint::black_box((&cars_out, &agents_out));
        // The same export into retained buffers.
        let mut car_buffer: Vec<GpuCar> = Vec::new();
        let mut agent_buffer: Vec<GpuAgent> = Vec::new();
        let mut export_retained = Vec::new();
        for _ in 0..3 {
            let t = Instant::now();
            altd_sim::training::export_state_into(
                &runner.agents,
                vehicle,
                &surfaces,
                &mut car_buffer,
                &mut agent_buffer,
            )
            .unwrap();
            export_retained.push(t.elapsed().as_secs_f64());
        }
        assert!(car_buffer
            .iter()
            .zip(&cars_out)
            .all(|(a, b)| format!("{a:?}") == format!("{b:?}")));
        assert!(agent_buffer
            .iter()
            .zip(&agents_out)
            .all(|(a, b)| format!("{a:?}") == format!("{b:?}")));
        println!(
            "round {round}: export collect {export_collect:.3} s; retained buffers {:?} s",
            export_retained.iter().map(|s| format!("{s:.3}")).collect::<Vec<_>>()
        );

        // Reproduction and installation make up the CPU turnover.
        let t = Instant::now();
        let results: Vec<_> = runner.agents.iter().map(|a| a.result()).collect();
        let generation = runner
            .rng
            .reproduce(&results, &runner.settings, altd_sim::math::profile::MathProfile::Proton);
        let reproduce = t.elapsed().as_secs_f64();
        std::hint::black_box(&generation.networks);
        drop(generation);
        let t = Instant::now();
        runner.next_generation();
        let turnover = t.elapsed().as_secs_f64();
        println!("round {round}: reproduce {reproduce:.3} s; full cpu turnover (reproduce + install) {turnover:.3} s");
    }
}
