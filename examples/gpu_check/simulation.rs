//! Sensors, inference, physics, statistics and complete generation checks.
use super::fixtures::{fixture_runner, input, load_json, ROOT};
use super::shared::{Rng, Tally};
use altd_sim::gpu::hip::{Gpu, GpuWorld};
use altd_sim::gpu::simulation::{self as gpu_sim, GpuAgent, GpuCar, GpuSim, NearGrid, ReadMask};
use altd_sim::nn::network;
use altd_sim::physics::car::{Car, Controls, SensorScratch, DT};
use altd_sim::track::world::World;
use altd_sim::training::TrainingRunner;
use rayon::prelude::*;
use std::time::Instant;

fn grid_line(name: &str, grid: Option<&NearGrid>) {
    match grid {
        Some(g) => {
            let (cells, mean, longest) = g.stats();
            println!(
                "  {name} grid: {}x{} = {cells} cells, mean list {mean:.1}, longest {longest}",
                g.nx, g.ny
            );
        }
        None => println!("  {name} grid: none (full scans)"),
    }
}

/// Every sensor of every car, GPU against `SensorLayout::read_into`, at
/// several points of a generated-track window.
pub(super) fn check_sensors(gpu: &Gpu, networks: usize, rounds: usize, ticks: u64) -> bool {
    let mut runner = fixture_runner(networks);
    let world = runner.world.clone();
    let gw = GpuWorld::new(gpu, &world);
    grid_line("path", gw.arrays.path_grid.as_ref());
    grid_line("curve", gw.arrays.curve_grid.as_ref());
    let vehicle = gpu_sim::vehicle_desc(&world.vehicle).unwrap();
    let descs: Vec<_> = runner.layout.sensors.iter().map(gpu_sim::sensor_desc).collect();
    let count = descs.len();
    let mut sim = GpuSim::new(&gw, &vehicle, &descs, networks);
    let mut tally = Tally::default();
    let mut per_sensor = vec![0usize; count];
    let mut seconds = (0.0, 0.0);
    for round in 0..=rounds {
        if round > 0 {
            runner.advance(ticks, false);
        }
        let cars: Vec<GpuCar> = runner
            .agents
            .iter()
            .enumerate()
            .map(|(i, a)| {
                a.car
                    .gpu_export(&world.vehicle, &gw.arrays.surfaces)
                    .unwrap_or_else(|e| panic!("car {i}: {e}"))
            })
            .collect();
        sim.upload(&cars, None);
        let t = Instant::now();
        let got = sim.sensors(ReadMask::ALL);
        seconds.0 += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let want: Vec<Vec<f64>> = runner
            .agents
            .par_iter()
            .map(|a| {
                let mut scratch = SensorScratch::default();
                let mut out = Vec::new();
                runner.layout.read_into(&world, &a.car, &mut scratch, &mut out);
                out
            })
            .collect();
        seconds.1 += t.elapsed().as_secs_f64();
        let mut back = Vec::new();
        sim.download(Some(&mut back), None);
        let errors: Vec<u32> = back.iter().map(|c| c.error).collect();
        let active = runner.agents.iter().filter(|a| a.car.active).count();
        for (i, w) in want.iter().enumerate() {
            for (k, &v) in w.iter().enumerate() {
                let g = got[i * count + k];
                let same = g.to_bits() == v.to_bits();
                if !same && errors[i] == 0 {
                    per_sensor[k] += 1;
                }
                tally.record(errors[i], same, || {
                    let c = &runner.agents[i].car;
                    format!(
                        "tick {} car {i} sensor {k} ({:?}): gpu {g:e} cpu {v:e} pos ({:e}, {:e})",
                        runner.tick, runner.layout.sensors[k], c.position.x, c.position.y
                    )
                });
            }
        }
        if round % 6 == 0 {
            println!("  tick {:5}: {active} active cars", runner.tick);
        }
    }
    println!("  sensor reads: gpu {:.3} s, cpu {:.3} s (rayon)", seconds.0, seconds.1);
    if per_sensor.iter().any(|&m| m > 0) {
        println!("  mismatches per sensor: {per_sensor:?}");
    }
    tally.report("sensors")
}

/// Sensors + network forward + control assignment under the inference batch
/// schedule, against `SensorLayout::read_into` + `Network::forward_into`.
pub(super) fn check_infer(gpu: &Gpu, networks: usize, rounds: usize, ticks: u64) -> bool {
    let mut runner = fixture_runner(networks);
    let world = runner.world.clone();
    let gw = GpuWorld::new(gpu, &world);
    let vehicle = gpu_sim::vehicle_desc(&world.vehicle).unwrap();
    let descs: Vec<_> = runner.layout.sensors.iter().map(gpu_sim::sensor_desc).collect();
    let mut sim = GpuSim::new(&gw, &vehicle, &descs, networks);
    let network_data = load_json(&input(
        "ALTD_GPU_NETWORK",
        &format!("{ROOT}/assets/networks/rally.json"),
    ));
    let outputs: Vec<String> = network_data["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    let src = gpu_sim::control_sources(&outputs);
    let shape = runner.agents[0].network.shape.clone();
    let params: Vec<f64> = runner
        .agents
        .iter()
        .flat_map(|a| a.network.params.iter().copied())
        .collect();
    sim.networks(&shape, &params, src);
    let mut tally = Tally::default();
    let (mut updated, mut kept) = (0usize, 0usize);
    let mut seconds = (0.0, 0.0);
    for round in 0..=rounds {
        if round > 0 {
            runner.advance(ticks, false);
        }
        let batch_index = round % 8;
        let mask = ReadMask::training(networks, batch_index, runner.batches_per_tick());
        let cars: Vec<GpuCar> = runner
            .agents
            .iter()
            .map(|a| a.car.gpu_export(&world.vehicle, &gw.arrays.surfaces).unwrap())
            .collect();
        // Sentinel controls: cars outside the mask must keep them.
        let agents: Vec<GpuAgent> = (0..networks)
            .map(|i| GpuAgent {
                controls: [-(i as f64) - 0.5; 5],
                ..Default::default()
            })
            .collect();
        sim.upload(&cars, Some(&agents));
        let t = Instant::now();
        sim.infer(mask);
        let mut got = Vec::new();
        let mut back = Vec::new();
        sim.download(Some(&mut back), Some(&mut got));
        seconds.0 += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let batch_size = networks.div_ceil(8);
        let want: Vec<[f64; 5]> = runner
            .agents
            .par_iter()
            .enumerate()
            .map(|(i, a)| {
                let reads = (i / batch_size + 8 - batch_index) % 8 < runner.batches_per_tick() && a.car.active;
                if !reads {
                    return agents[i].controls;
                }
                let mut scratch = SensorScratch::default();
                let mut inputs = Vec::new();
                runner.layout.read_into(&world, &a.car, &mut scratch, &mut inputs);
                let mut forward = network::ForwardScratch::default();
                let out = a.network.forward_into(&inputs, &mut forward, world.math);
                let mut c = altd_sim::physics::car::Controls::default();
                for (name, &v) in outputs.iter().zip(out) {
                    c.set(name, v);
                }
                [c.acceleration, c.steering, c.brake, c.handbrake, c.boost]
            })
            .collect();
        seconds.1 += t.elapsed().as_secs_f64();
        for (i, w) in want.iter().enumerate() {
            let same = (0..5).all(|c| w[c].to_bits() == got[i].controls[c].to_bits());
            if w[0].to_bits() == agents[i].controls[0].to_bits() {
                kept += 1
            } else {
                updated += 1
            }
            tally.record(back[i].error, same, || {
                format!("tick {} car {i}: gpu {:?} cpu {w:?}", runner.tick, got[i].controls)
            });
        }
    }
    println!(
        "  {updated} control updates, {kept} unchanged (outside the batch); gpu {:.3} s incl. copies, cpu {:.3} s",
        seconds.0, seconds.1
    );
    tally.report("infer")
}

/// GPU car with the unused pair/contact slots cleared, as Debug text (f32/f64
/// Debug output round-trips, so equal text means equal bits up to NaN payloads).
pub(super) fn car_text(c: &GpuCar) -> String {
    let mut c = *c;
    for i in c.pair_count as usize..c.pairs.len() {
        c.pairs[i] = 0;
        c.axes[i] = [0.0; 2];
    }
    for i in c.contact_count as usize..c.contacts.len() {
        c.contacts[i] = Default::default();
    }
    c.error = 0;
    format!("{c:?}")
}

/// `Car::step` / `Car::passive_step` against the GPU step kernel with random
/// controls, free-running; after a mismatch the GPU is resynced from the CPU.
pub(super) fn check_step(gpu: &Gpu, networks: usize, warmup: u64, ticks: u64) -> bool {
    let mut runner = fixture_runner(networks);
    runner.advance(warmup, false);
    let world = runner.world.clone();
    let gw = GpuWorld::new(gpu, &world);
    grid_line("shape", gw.arrays.shape_grid.as_ref());
    let vehicle = gpu_sim::vehicle_desc(&world.vehicle).unwrap();
    let descs: Vec<_> = runner.layout.sensors.iter().map(gpu_sim::sensor_desc).collect();
    let mut sim = GpuSim::new(&gw, &vehicle, &descs, networks);
    let mut cars: Vec<Car> = runner.agents.iter().map(|a| a.car.clone()).collect();
    let export = |cars: &[Car]| -> Vec<GpuCar> {
        cars.iter()
            .enumerate()
            .map(|(i, c)| {
                c.gpu_export(&world.vehicle, &gw.arrays.surfaces)
                    .unwrap_or_else(|e| panic!("car {i}: {e}"))
            })
            .collect()
    };
    let mut tally = Tally::default();
    let mut rng = Rng(7);
    let mut resync = true;
    let (mut contacts, mut resyncs, mut max_pairs, mut max_contacts) = (0usize, 0usize, 0u32, 0u32);
    let mut seconds = (0.0, 0.0);
    let mut got = Vec::new();
    let mut agents_back = Vec::new();
    for t in 0..ticks {
        let drive = t % 7 != 6;
        let eliminate = (t / 50) % 2 == 1;
        let controls: Vec<Controls> = (0..networks)
            .map(|_| Controls {
                acceleration: rng.range(-1.3, 1.3),
                steering: rng.range(-1.3, 1.3),
                brake: if rng.unit() < 0.2 { rng.range(-0.2, 1.2) } else { 0.0 },
                handbrake: if rng.unit() < 0.1 { rng.range(0.0, 1.0) } else { 0.0 },
                boost: if rng.unit() < 0.3 { 1.0 } else { 0.0 },
            })
            .collect();
        let agents: Vec<GpuAgent> = controls
            .iter()
            .map(|c| GpuAgent {
                controls: [c.acceleration, c.steering, c.brake, c.handbrake, c.boost],
                ..Default::default()
            })
            .collect();
        if resync {
            sim.upload(&export(&cars), Some(&agents));
            resync = false;
        } else {
            sim.upload_agents(&agents);
        }
        let s = Instant::now();
        sim.step(drive, eliminate, t);
        sim.download(Some(&mut got), Some(&mut agents_back));
        seconds.0 += s.elapsed().as_secs_f64();
        let s = Instant::now();
        let reports: Vec<bool> = cars
            .par_iter_mut()
            .zip(&controls)
            .map(|(car, c)| {
                if drive {
                    car.step(&world, c, DT, eliminate)
                } else {
                    car.passive_step(&world, DT, eliminate)
                }
            })
            .collect();
        seconds.1 += s.elapsed().as_secs_f64();
        let want = export(&cars);
        let mut bad = false;
        for i in 0..networks {
            let pending = agents_back[i].flags & gpu_sim::AGENT_PENDING_CONTACT != 0;
            contacts += reports[i] as usize;
            max_pairs = max_pairs.max(want[i].pair_count);
            max_contacts = max_contacts.max(want[i].contact_count);
            let (a, b) = (car_text(&got[i]), car_text(&want[i]));
            let same = a == b && pending == reports[i];
            bad |= !same || got[i].error != 0;
            tally.record(got[i].error, same, || {
                let at = a.bytes().zip(b.bytes()).position(|(x, y)| x != y).unwrap_or(0).saturating_sub(40);
                format!("tick {t} car {i} (drive {drive}, eliminate {eliminate}) report gpu {pending} cpu {}: gpu ..{}.. cpu ..{}..",
                    reports[i], &a[at..(at + 160).min(a.len())], &b[at..(at + 160).min(b.len())])
            });
        }
        if bad {
            resync = true;
            resyncs += 1;
        }
        if t % 50 == 49 {
            let active = cars.iter().filter(|c| c.active).count();
            println!(
                "  tick {:4}: {active} active, {contacts} wall reports so far, {resyncs} resyncs",
                t + 1
            );
        }
    }
    // Import round trip: GPU state -> Car -> GPU state.
    let mut round = 0;
    for (i, g) in got.iter().enumerate() {
        let mut c = cars[i].clone();
        if c.gpu_import(g, &gw.arrays.surfaces).is_ok() {
            let back = c.gpu_export(&world.vehicle, &gw.arrays.surfaces).unwrap();
            round += (car_text(&back) != car_text(g)) as usize;
        }
    }
    println!("  max {max_pairs} pairs, {max_contacts} contacts per car; import round-trip mismatches {round}");
    println!(
        "  step: gpu {:.3} s incl. copies, cpu {:.3} s (rayon)",
        seconds.0, seconds.1
    );
    tally.report("step") && round == 0
}

/// The differing `name: value` fields of two Debug texts (gpu -> cpu).
pub(super) fn field_diff(gpu: &str, cpu: &str) -> String {
    gpu.split(", ")
        .zip(cpu.split(", "))
        .filter(|(a, b)| a != b)
        .map(|(a, b)| format!("{a} -> {b}"))
        .collect::<Vec<_>>()
        .join("; ")
}

/// GPU agent with the score-diff ring rotated to start at 0, unused slots cleared.
pub(super) fn agent_text(a: &GpuAgent) -> String {
    let mut a = *a;
    let ring = a.recent;
    for k in 0..gpu_sim::RECENT {
        a.recent[k] = if k < a.recent_len as usize {
            ring[(a.recent_start as usize + k) % gpu_sim::RECENT]
        } else {
            0.0
        };
    }
    a.recent_start = 0;
    format!("{a:?}")
}

/// `TrainingAgent::update_stats` against the GPU stats kernel at every
/// statistics tick of a real generated-track window (cars and agents exported from the
/// CPU each time; eliminate_when_idle alternates).
pub(super) fn check_stats(gpu: &Gpu, networks: usize, ticks: u64) -> bool {
    let mut runner = fixture_runner(networks);
    let world = runner.world.clone();
    let gw = GpuWorld::new(gpu, &world);
    let vehicle = gpu_sim::vehicle_desc(&world.vehicle).unwrap();
    let descs: Vec<_> = runner.layout.sensors.iter().map(gpu_sim::sensor_desc).collect();
    let mut sim = GpuSim::new(&gw, &vehicle, &descs, networks);
    let phase = runner.stats_phase;
    let first = if phase == 0 { 6 } else { phase };
    let mut tally = Tally::default();
    let (mut laps, mut idle_kills, mut contacts, mut calls) = (0usize, 0usize, 0usize, 0usize);
    let mut seconds = (0.0, 0.0);
    let (mut got_cars, mut got_agents) = (Vec::new(), Vec::new());
    while runner.tick < ticks {
        let tick = runner.tick + 1;
        if tick % 6 == phase {
            let idle = calls % 2 == 1;
            calls += 1;
            let cars: Vec<GpuCar> = runner
                .agents
                .iter()
                .map(|a| a.car.gpu_export(&world.vehicle, &gw.arrays.surfaces).unwrap())
                .collect();
            let agents: Vec<GpuAgent> = runner
                .agents
                .iter()
                .map(|a| gpu_sim::agent_export(a).unwrap())
                .collect();
            // A round trip through agent_import must reproduce the export.
            let mut probe = runner.agents[0].clone();
            gpu_sim::agent_import(&agents[0], &mut probe);
            assert_eq!(
                agent_text(&gpu_sim::agent_export(&probe).unwrap()),
                agent_text(&agents[0]),
                "agent import round trip"
            );
            sim.upload(&cars, Some(&agents));
            let s = Instant::now();
            sim.stats(tick, tick - first, idle);
            sim.download(Some(&mut got_cars), Some(&mut got_agents));
            seconds.0 += s.elapsed().as_secs_f64();
            let mut want = runner.agents.clone();
            let s = Instant::now();
            want.par_iter_mut()
                .for_each(|a| a.update_stats(&world, tick, phase, idle));
            seconds.1 += s.elapsed().as_secs_f64();
            for (i, w) in want.iter().enumerate() {
                let before = &runner.agents[i];
                laps += (w.stats.score.lap_count > before.stats.score.lap_count) as usize;
                idle_kills += (before.car.active && !w.car.active) as usize;
                contacts += before.pending_contact as usize;
                let wa = gpu_sim::agent_export(w).unwrap();
                let wc = w.car.gpu_export(&world.vehicle, &gw.arrays.surfaces).unwrap();
                let (ga, gc) = (agent_text(&got_agents[i]), car_text(&got_cars[i]));
                let (sa, sc) = (agent_text(&wa), car_text(&wc));
                tally.record(got_cars[i].error, ga == sa && gc == sc, || {
                    format!(
                        "tick {tick} car {i} (idle {idle}) car same {}: {}",
                        gc == sc,
                        field_diff(&ga, &sa)
                    )
                });
            }
        }
        runner.advance(1, false);
        if runner.tick.is_multiple_of(600) {
            let active = runner.agents.iter().filter(|a| a.car.active).count();
            let best = runner.agents.iter().map(|a| a.stats.score.lap_count).max().unwrap_or(0);
            println!("  tick {:5}: {active} active, max laps {best}", runner.tick);
        }
    }
    println!(
        "  {calls} stats ticks: {laps} lap completions, {idle_kills} idle eliminations, {contacts} pending contacts"
    );
    println!(
        "  stats: gpu {:.3} s incl. copies, cpu {:.3} s (rayon)",
        seconds.0, seconds.1
    );
    tally.report("stats")
}

/// Compares every agent (statistics, controls, car) of two runners.
pub(super) fn compare_runners(
    gw: &GpuWorld,
    world: &World,
    cpu: &TrainingRunner,
    gpu: &TrainingRunner,
    tally: &mut Tally,
    what: &str,
) {
    if (cpu.tick, cpu.batch_index) != (gpu.tick, gpu.batch_index) {
        eprintln!(
            "    {what}: tick/batch cpu ({}, {}) gpu ({}, {})",
            cpu.tick, cpu.batch_index, gpu.tick, gpu.batch_index
        );
        tally.record(0, false, || "window length".into());
    }
    for (i, (c, g)) in cpu.agents.iter().zip(&gpu.agents).enumerate() {
        let text = |a: &altd_sim::training::TrainingAgent| {
            let car = a.car.gpu_export(&world.vehicle, &gw.arrays.surfaces).unwrap();
            (
                agent_text(&gpu_sim::agent_export(a).unwrap()),
                car_text(&car),
                a.stats.metrics().map(|m| m.map(f64::to_bits)),
            )
        };
        let (want, got) = (text(c), text(g));
        tally.record(0, want == got, || {
            format!(
                "{what} car {i}: agent {} | car {}",
                field_diff(&got.0, &want.0),
                field_diff(&got.1, &want.1)
            )
        });
    }
}

/// Whole training generations: `advance_generation` on the CPU against
/// `advance_generation_gpu`, then `next_generation` on both and again.
pub(super) fn check_window(
    gpu: &Gpu,
    networks: usize,
    ticks: u64,
    generations: usize,
    configs: &[(bool, bool)],
) -> bool {
    let mut tally = Tally::default();
    for &(eliminate_on_wall, eliminate_when_idle) in configs {
        let mut cpu = fixture_runner(networks);
        cpu.eliminate_on_wall = eliminate_on_wall;
        cpu.eliminate_when_idle = eliminate_when_idle;
        let world = cpu.world.clone();
        let gw = GpuWorld::new(gpu, &world);
        let mut sim = cpu.gpu_sim(&gw, networks).unwrap();
        let mut on_gpu = fixture_runner(networks);
        on_gpu.eliminate_on_wall = eliminate_on_wall;
        on_gpu.eliminate_when_idle = eliminate_when_idle;
        compare_runners(&gw, &world, &cpu, &on_gpu, &mut tally, "setup");
        println!(
            "  {networks} cars, eliminate_on_wall {eliminate_on_wall}, eliminate_when_idle {eliminate_when_idle}:"
        );
        for generation in 0..generations {
            let t = Instant::now();
            let executed = cpu.advance_generation(ticks);
            let cpu_seconds = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let gpu_executed = on_gpu
                .advance_generation_gpu(&mut sim, &gw, ticks)
                .unwrap_or_else(|e| panic!("{e}"));
            let gpu_seconds = t.elapsed().as_secs_f64();
            let before = tally.mismatches;
            compare_runners(&gw, &world, &cpu, &on_gpu, &mut tally, &format!("gen {generation}"));
            let active = cpu.agents.iter().filter(|a| a.car.active).count();
            let laps = cpu.agents.iter().filter(|a| a.stats.best_lap_time.is_some()).count();
            println!("    gen {generation}: {executed}/{gpu_executed} ticks, {active} active at the end, {laps} lapped; cpu {cpu_seconds:.2} s, gpu {gpu_seconds:.2} s; {} mismatching agents",
                tally.mismatches - before);
            if generation + 1 < generations {
                let t = Instant::now();
                cpu.next_generation();
                on_gpu.next_generation_gpu(&mut sim).unwrap();
                println!("    turnover {:.2} s (x2)", t.elapsed().as_secs_f64());
                compare_runners(
                    &gw,
                    &world,
                    &cpu,
                    &on_gpu,
                    &mut tally,
                    &format!("install {}", generation + 1),
                );
            }
        }
    }
    tally.report("window")
}
