//! GPU generation timings with fixed fixtures and result digests.
use super::fixtures::fixture_runner;
use super::simulation::{agent_text, car_text};
use altd_sim::gpu::hip::{Gpu, GpuWorld};
use altd_sim::gpu::simulation as gpu_sim;
use rayon::prelude::*;
use std::time::Instant;

/// GPU-only generation timing (no CPU reference run), without eliminations
/// so every car drives the whole window.
pub(super) fn bench_window(gpu: &Gpu, networks: usize, ticks: u64, eliminate: bool) -> bool {
    let mut runner = fixture_runner(networks);
    runner.eliminate_on_wall = eliminate;
    runner.eliminate_when_idle = eliminate;
    let world = runner.world.clone();
    let gw = GpuWorld::new(gpu, &world);
    let mut sim = runner.gpu_sim(&gw, networks).unwrap();
    for generation in 0..2 {
        let t = Instant::now();
        let executed = runner
            .advance_generation_gpu(&mut sim, &gw, ticks)
            .unwrap_or_else(|e| panic!("{e}"));
        let seconds = t.elapsed().as_secs_f64();
        let active = runner.agents.iter().filter(|a| a.car.active).count();
        println!("  gen {generation}: {networks} cars x {executed} ticks in {seconds:.2} s ({:.0} car-ticks/s), {active} active",
            networks as f64 * executed as f64 / seconds);
        let t = Instant::now();
        runner.next_generation();
        println!("  turnover (cpu) {:.2} s", t.elapsed().as_secs_f64());
    }
    true
}

/// Repeat identical generations, including network packing, transfers and state
/// import in the timing. Resetting the fixture and hashing results are untimed.
pub(super) fn bench_fixed(gpu: &Gpu) -> bool {
    let number = |name: &str, default: usize| -> usize {
        std::env::var(name).map(|v| v.parse().expect(name)).unwrap_or(default)
    };
    let networks = number("ALTD_GPU_BENCH_CARS", 8192);
    let ticks = number("ALTD_GPU_BENCH_TICKS", 5400) as u64;
    let samples = number("ALTD_GPU_BENCH_SAMPLES", 3);
    let warmups = number("ALTD_GPU_BENCH_WARMUPS", 1);
    let eliminate = number("ALTD_GPU_BENCH_ELIMINATE", 1) != 0;
    assert!(networks > 0 && ticks > 0 && samples > 0);
    let mut runner = fixture_runner(networks);
    runner.eliminate_on_wall = eliminate;
    runner.eliminate_when_idle = eliminate;
    let world = runner.world.clone();
    let gw = GpuWorld::new(gpu, &world);
    let mut sim = runner.gpu_sim(&gw, networks).unwrap();
    let initial_cars: Vec<_> = runner
        .agents
        .iter()
        .map(|a| a.car.gpu_export(&world.vehicle, &gw.arrays.surfaces).unwrap())
        .collect();
    let initial_agents: Vec<_> = runner
        .agents
        .iter()
        .map(|a| gpu_sim::agent_export(a).unwrap())
        .collect();
    let mut expected = None;
    for sample in 0..warmups + samples {
        runner.tick = 0;
        runner.batch_index = 0;
        runner
            .agents
            .par_iter_mut()
            .zip(&initial_cars)
            .zip(&initial_agents)
            .for_each(|((a, c), g)| {
                a.car.gpu_import(c, &gw.arrays.surfaces).unwrap();
                gpu_sim::agent_import(g, a);
            });
        // A real new generation uploads networks, even when their shape is unchanged.
        sim.network_tag = None;
        let start = Instant::now();
        let executed = runner.advance_generation_gpu(&mut sim, &gw, ticks).unwrap();
        let seconds = start.elapsed().as_secs_f64();
        use std::hash::{Hash, Hasher};
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        (runner.tick, runner.batch_index).hash(&mut hash);
        for a in &runner.agents {
            car_text(&a.car.gpu_export(&world.vehicle, &gw.arrays.surfaces).unwrap()).hash(&mut hash);
            agent_text(&gpu_sim::agent_export(a).unwrap()).hash(&mut hash);
        }
        let digest = format!("{:016x}", hash.finish());
        if let Some(ref previous) = expected {
            assert_eq!(previous, &digest, "repeated generation differs");
        }
        expected = Some(digest.clone());
        println!(
            "{}",
            serde_json::json!({
                "benchmark": "fixed_generation", "cars": networks, "ticks": executed,
                "eliminate": eliminate, "sample": sample, "warmup": sample < warmups,
                "seconds": seconds, "digest": digest,
                "active": runner.agents.iter().filter(|a| a.car.active).count(),
            })
        );
    }
    true
}

/// Consecutive generations, including reproduction, reset and network upload.
pub(super) fn bench_training(gpu: &Gpu) -> bool {
    use std::hash::{Hash, Hasher};
    let number = |name: &str, default: usize| -> usize {
        std::env::var(name).map(|v| v.parse().expect(name)).unwrap_or(default)
    };
    let count = number("ALTD_GPU_BENCH_CARS", 32768);
    let generations = number("ALTD_GPU_BENCH_GENERATIONS", 3);
    let ticks = number("ALTD_GPU_BENCH_TICKS", 5400) as u64;
    let mut runner = fixture_runner(count);
    runner.eliminate_on_wall = true;
    runner.eliminate_when_idle = true;
    let world = runner.world.clone();
    let gw = GpuWorld::new(gpu, &world);
    let mut sim = runner.gpu_sim(&gw, count).unwrap();
    for generation in 0..generations {
        let start = Instant::now();
        let executed = runner.advance_generation_gpu(&mut sim, &gw, ticks).unwrap();
        let simulate = start.elapsed().as_secs_f64();
        let start = Instant::now();
        if std::env::var("ALTD_GPU_BENCH_CPU_TURNOVER").is_ok_and(|v| v == "1") {
            runner.next_generation();
        } else {
            runner.next_generation_gpu(&mut sim).unwrap();
        }
        let turnover = start.elapsed().as_secs_f64();
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        runner.rng.to_json().to_string().hash(&mut hash);
        for a in &runner.agents {
            for p in &a.network.params {
                p.to_bits().hash(&mut hash);
            }
            car_text(&a.car.gpu_export(&world.vehicle, &gw.arrays.surfaces).unwrap()).hash(&mut hash);
            agent_text(&gpu_sim::agent_export(a).unwrap()).hash(&mut hash);
        }
        println!(
            "{}",
            serde_json::json!({"benchmark":"training", "cars":count, "generation":generation,
            "ticks":executed, "simulate_seconds":simulate, "turnover_seconds":turnover,
            "total_seconds":simulate+turnover, "digest":format!("{:016x}",hash.finish())})
        );
    }
    true
}
