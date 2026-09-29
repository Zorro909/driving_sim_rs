//! Bitwise differential checks of the GPU simulator parts (gpu/) against the CPU.
//! Usage: gpu_check [math] ...   (build the library first with gpu/build.sh)
use altd_sim::car::{Car, Controls, SensorScratch, DT};
use altd_sim::evolution::EvolutionSettings;
use altd_sim::gpu::{Gpu, GpuWorld, MathOp, Query};
use altd_sim::gpu_sim::{self, GpuAgent, GpuCar, GpuSim, NearGrid, ReadMask};
use altd_sim::pyrandom::PyRandom;
use altd_sim::training::{SensorLayout, TrainingRunner};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;
use altd_sim::vec2::V2;
use altd_sim::world::World;
use altd_sim::{double_math, godot_math, native_math, network};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// Floats spread over every exponent, the simulator's typical ranges, and specials.
fn f32_inputs(rng: &mut Rng, n: usize, typical: f64) -> Vec<f32> {
    let mut v: Vec<f32> = [0.0f32, -0.0, 1.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::MIN_POSITIVE, 1e-40, -1e-40]
        .into_iter()
        .chain([std::f32::consts::PI, -std::f32::consts::PI, std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_4])
        .collect();
    while v.len() < n {
        v.push(match v.len() % 3 {
            0 => f32::from_bits(rng.next() as u32),
            1 => rng.range(-typical, typical) as f32,
            _ => rng.range(-1.0, 1.0) as f32,
        });
    }
    v
}

fn f64_inputs(rng: &mut Rng, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    let mut v = vec![0.0, -0.0, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1e-310, 709.78, -745.2, 800.0, -800.0];
    while v.len() < n {
        v.push(if v.len() % 3 == 0 { f64::from_bits(rng.next()) } else { rng.range(lo, hi) });
    }
    v
}

#[derive(Default)]
struct Tally {
    checked: usize,
    flagged: usize,
    mismatches: usize,
}
impl Tally {
    fn record(&mut self, err: u32, same: bool, describe: impl FnOnce() -> String) {
        if err != 0 {
            self.flagged += 1;
            return;
        }
        self.checked += 1;
        if !same {
            if self.mismatches < 8 {
                eprintln!("    mismatch: {}", describe());
            }
            self.mismatches += 1;
        }
    }
    fn report(&self, name: &str) -> bool {
        println!(
            "  {name:<16} {:>9} checked  {:>7} flagged  {:>6} mismatches  {}",
            self.checked,
            self.flagged,
            self.mismatches,
            if self.mismatches == 0 { "OK" } else { "FAIL" }
        );
        self.mismatches == 0
    }
}

fn check_math(gpu: &Gpu, n: usize, include_engine: bool) -> bool {
    let mut rng = Rng(0x5eed);
    let mut ok = true;
    println!("math primitives ({n} inputs each):");

    for (op, name, cpu) in [
        (MathOp::NativeSin, "native_sin", native_math::sin as fn(f32) -> f32),
        (MathOp::NativeCos, "native_cos", native_math::cos),
    ] {
        let input = f32_inputs(&mut rng, n, 40.0);
        let mut out = vec![0f32; n];
        let err = gpu.math(op, &input, &mut out);
        let mut t = Tally::default();
        for i in 0..n {
            // The CPU's Payne-Hanek path is not ported; the GPU flags it.
            let want = cpu(input[i]);
            t.record(err[i], want.to_bits() == out[i].to_bits(), || format!("{:e} cpu {want:e} gpu {:e}", input[i], out[i]));
        }
        ok &= t.report(name);
    }

    {
        let ys = f32_inputs(&mut rng, n, 1000.0);
        let xs = f32_inputs(&mut rng, n, 1000.0);
        let input: Vec<[f32; 2]> = ys.iter().zip(&xs).map(|(&y, &x)| [y, x]).collect();
        let mut out = vec![0f32; n];
        let err = gpu.math(MathOp::NativeAtan2, &input, &mut out);
        let mut t = Tally::default();
        for i in 0..n {
            let want = native_math::atan2(input[i][0], input[i][1]);
            t.record(err[i], want.to_bits() == out[i].to_bits(), || format!("{:?} cpu {want:e} gpu {:e}", input[i], out[i]));
        }
        ok &= t.report("native_atan2");
    }

    {
        let input = f32_inputs(&mut rng, n, 20000.0);
        let mut out = vec![[0f32; 2]; n];
        let err = gpu.math(MathOp::ManagedSinCos, &input, &mut out);
        let mut t = Tally::default();
        for i in 0..n {
            let (s, c) = godot_math::managed_sin_cos(input[i]);
            let same = s.to_bits() == out[i][0].to_bits() && c.to_bits() == out[i][1].to_bits();
            t.record(err[i], same, || format!("{:e} cpu ({s:e},{c:e}) gpu {:?}", input[i], out[i]));
        }
        ok &= t.report("managed_sin_cos");
    }

    {
        let input = f64_inputs(&mut rng, n, -760.0, 720.0);
        let mut out = vec![0f64; n];
        let err = gpu.math(MathOp::Exp, &input, &mut out);
        let mut t = Tally::default();
        for i in 0..n {
            let want = double_math::exp(input[i]);
            t.record(err[i], want.to_bits() == out[i].to_bits(), || format!("{:e} cpu {want:e} gpu {:e}", input[i], out[i]));
        }
        ok &= t.report("exp");
    }

    {
        let xs = f64_inputs(&mut rng, n, 0.0, 1.0);
        let ys = f64_inputs(&mut rng, n, 0.05, 20.0);
        let mut input: Vec<[f64; 2]> = xs.iter().zip(&ys).map(|(&x, &y)| [x, y]).collect();
        // Negative bases with integer exponents, and the godot_ease shape 1 - pow(1 - v, 1 / c).
        for (i, pair) in input.iter_mut().enumerate().skip(16).step_by(7) {
            pair[0] = -pair[0] * 3.0;
            pair[1] = (i % 9) as f64;
        }
        let mut out = vec![0f64; n];
        let err = gpu.math(MathOp::Pow, &input, &mut out);
        let mut t = Tally::default();
        for i in 0..n {
            let want = double_math::pow(input[i][0], input[i][1]);
            t.record(err[i], want.to_bits() == out[i].to_bits(), || format!("{:?} cpu {want:e} gpu {:e}", input[i], out[i]));
        }
        ok &= t.report("pow");
    }

    {
        let input = f64_inputs(&mut rng, n, -12.0, 12.0);
        let mut out = vec![0f64; n];
        let err = gpu.math(MathOp::GameTanh, &input, &mut out);
        let mut t = Tally::default();
        for i in 0..n {
            let want = network::game_tanh(input[i]);
            t.record(err[i], want.to_bits() == out[i].to_bits(), || format!("{:e} cpu {want:e} gpu {:e}", input[i], out[i]));
        }
        ok &= t.report("game_tanh");
    }

    if include_engine {
        let mut input = f32_inputs(&mut rng, n, 16.0);
        for (i, x) in input.iter_mut().enumerate() {
            if i % 3 == 0 && x.is_finite() && x.abs() > 16.0 {
                *x = (i as f32 * 1e-3) % 32.0 - 16.0;
            }
        }
        let mut out = vec![[0f32; 2]; n];
        let err = gpu.math(MathOp::EngineSinCos, &input, &mut out);
        let mut t = Tally::default();
        for i in 0..n {
            let (s, c) = native_math::engine_sin_cos(input[i]);
            let same = s.to_bits() == out[i][0].to_bits() && c.to_bits() == out[i][1].to_bits();
            t.record(err[i], same, || format!("{:e} cpu ({s:e},{c:e}) gpu {:?}", input[i], out[i]));
        }
        ok &= t.report("engine_sin_cos");
    }
    ok
}

const SCENE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/scenes_exact/rally_b06_scene.json");

/// The B06 inputs, or another track's via ALTD_GPU_SCENE, ALTD_GPU_SPAWN,
/// ALTD_GPU_NETWORK and ALTD_GPU_MODEL.
fn input(var: &str, default: &str) -> String {
    std::env::var(var).unwrap_or_else(|_| default.to_owned())
}

fn scene_path() -> String {
    input("ALTD_GPU_SCENE", SCENE)
}

fn load_world() -> World {
    let scene = scene_path();
    let text = std::fs::read_to_string(&scene).unwrap_or_else(|e| panic!("{scene}: {e}"));
    World::from_scene(&serde_json::from_str(&text).expect("scene json"))
}

/// Points near the racing line (and a few far off it) for track queries.
fn track_points(world: &World, rng: &mut Rng, n: usize, spread: f64) -> Vec<V2> {
    let path = &world.track.path;
    (0..n)
        .map(|i| {
            let p = path[(rng.next() % path.len() as u64) as usize];
            let s = if i % 50 == 0 { spread * 20.0 } else { spread };
            V2::new((p.x + rng.range(-s, s)) as f32 as f64, (p.y + rng.range(-s, s)) as f32 as f64)
        })
        .collect()
}

fn check_rays(gpu: &Gpu, world: &World, n: usize) -> bool {
    let mut rng = Rng(0xba5e);
    let gw = GpuWorld::new(gpu, world);
    let mut stamps = Default::default();
    let starts = track_points(world, &mut rng, n, 150.0);
    let mut input = Vec::with_capacity(n);
    for (i, s) in starts.iter().enumerate() {
        let angle = rng.range(-3.2, 3.2);
        let length = [200.0, 500.0, 800.0, 624.264, 1e-3, 0.0][i % 6] * rng.range(0.9, 1.1);
        input.push([s.x as f32, s.y as f32, (s.x + angle.cos() * length) as f32, (s.y + angle.sin() * length) as f32]);
    }
    // Rays ending exactly on a wall endpoint or along walls stress the ties.
    for (i, w) in world.track.walls.iter().enumerate().take(n / 10) {
        let s = starts[i];
        input[i] = [s.x as f32, s.y as f32, w.start.x as f32, w.start.y as f32];
    }
    let started = std::time::Instant::now();
    let out = gw.query(Query::Raycast, &input);
    let gpu_time = started.elapsed();
    let mut t = Tally::default();
    let mut hits = 0;
    for (q, o) in input.iter().zip(&out) {
        let want = world.track.raycast(V2::new(q[0] as f64, q[1] as f64), V2::new(q[2] as f64, q[3] as f64), &mut stamps);
        let got = (o[2] != 0.0).then(|| V2::new(o[0] as f64, o[1] as f64));
        hits += want.is_some() as usize;
        let same = match (want, got) {
            (None, None) => true,
            (Some(a), Some(b)) => a.x.to_bits() == b.x.to_bits() && a.y.to_bits() == b.y.to_bits(),
            _ => false,
        };
        t.record(0, same, || format!("ray {q:?}: cpu {want:?} gpu {got:?}"));
    }
    println!("track queries on {} ({n} each):", scene_path());
    ok_line(t.report("raycast"), &format!("{hits} hits, GPU {:.1} ms incl. copies", gpu_time.as_secs_f64() * 1e3))
        & {
            let points = track_points(world, &mut rng, n, 200.0);
            let input: Vec<[f32; 4]> = points.iter().map(|p| [p.x as f32, p.y as f32, 0.0, 0.0]).collect();
            let out = gw.query(Query::ClosestWall, &input);
            let mut t = Tally::default();
            for (p, o) in points.iter().zip(&out) {
                let want = world.track.closest_wall(*p).expect("BSP closest");
                let same = want.x.to_bits() == (o[0] as f64).to_bits() && want.y.to_bits() == (o[1] as f64).to_bits();
                t.record(0, same, || format!("point {p:?}: cpu {want:?} gpu {o:?}"));
            }
            t.report("closest_wall")
        }
}

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
const CHECKPOINT_DIR: &str = "/tmp/altd-gpu-ckpt";

fn load_json(path: &str) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"))).expect("json")
}

/// A B06 training runner (the run's settings) holding the first `networks`
/// networks of the /tmp checkpoint copy, installed at the spawn.
fn b06_runner(networks: usize) -> TrainingRunner {
    let network_data = load_json(&input("ALTD_GPU_NETWORK", &format!("{ROOT}/docs/traces/rally_a01_live_network.json")));
    let model = load_json(&input("ALTD_GPU_MODEL", concat!(env!("CARGO_MANIFEST_DIR"), "/rally_trained_model_exact.json")));
    let spawn = &load_json(&input("ALTD_GPU_SPAWN", &format!("{ROOT}/docs/traces/rally_b06_reference.json")))["frames"][0];
    // ALTD_GPU_CKPT picks another checkpoint directory (e.g. a copy of the live run's).
    let dir = std::env::var("ALTD_GPU_CKPT").unwrap_or_else(|_| CHECKPOINT_DIR.to_owned());
    let meta = load_json(&format!("{dir}/checkpoint.json"));
    let shape: Vec<usize> = match meta["shape"].as_array() {
        Some(widths) => widths.iter().map(|w| w.as_u64().unwrap() as usize).collect(),
        None => vec![20, 16, 16, 16, 16, 12, 12, 8, 5],
    };
    let file = format!("{dir}/{}", meta["population_file"].as_str().unwrap());
    let size = network::parameter_count(&shape);
    let bytes = std::fs::read(&file).unwrap_or_else(|e| panic!("{file}: {e} (copy the B06 checkpoint there)"));
    // More cars than saved networks repeat the population.
    let nets: Vec<network::Network> = bytes
        .chunks_exact(size * 8)
        .cycle()
        .take(networks)
        .map(|c| network::Network::from_vector(&shape, c.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect()))
        .collect();
    let outputs: Vec<String> = network_data["outputs"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_owned()).collect();
    let settings = EvolutionSettings::from_mcp(&serde_json::json!({"population": networks}));
    let p = &spawn["position"];
    let mut runner = TrainingRunner::new(
        Arc::new(load_world()),
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
    runner.resume(&nets, meta["generation"].as_u64().unwrap());
    runner
}

fn grid_line(name: &str, grid: Option<&NearGrid>) {
    match grid {
        Some(g) => {
            let (cells, mean, longest) = g.stats();
            println!("  {name} grid: {}x{} = {cells} cells, mean list {mean:.1}, longest {longest}", g.nx, g.ny);
        }
        None => println!("  {name} grid: none (full scans)"),
    }
}

/// Every sensor of every car, GPU against `SensorLayout::read_into`, at
/// several points of a B06 window.
fn check_sensors(gpu: &Gpu, networks: usize, rounds: usize, ticks: u64) -> bool {
    let mut runner = b06_runner(networks);
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
            .map(|(i, a)| a.car.gpu_export(&world.vehicle, &gw.arrays.surfaces).unwrap_or_else(|e| panic!("car {i}: {e}")))
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
                    format!("tick {} car {i} sensor {k} ({:?}): gpu {g:e} cpu {v:e} pos ({:e}, {:e})",
                        runner.tick, runner.layout.sensors[k], c.position.x, c.position.y)
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

/// Sensors + network forward + control assignment under the B06 batch
/// schedule, against `SensorLayout::read_into` + `Network::forward_into`.
fn check_infer(gpu: &Gpu, networks: usize, rounds: usize, ticks: u64) -> bool {
    let mut runner = b06_runner(networks);
    let world = runner.world.clone();
    let gw = GpuWorld::new(gpu, &world);
    let vehicle = gpu_sim::vehicle_desc(&world.vehicle).unwrap();
    let descs: Vec<_> = runner.layout.sensors.iter().map(gpu_sim::sensor_desc).collect();
    let mut sim = GpuSim::new(&gw, &vehicle, &descs, networks);
    let network_data = load_json(&input("ALTD_GPU_NETWORK", &format!("{ROOT}/docs/traces/rally_a01_live_network.json")));
    let outputs: Vec<String> = network_data["outputs"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_owned()).collect();
    let src = gpu_sim::control_sources(&outputs);
    let shape = runner.agents[0].network.shape.clone();
    let params: Vec<f64> = runner.agents.iter().flat_map(|a| a.network.params.iter().copied()).collect();
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
        let cars: Vec<GpuCar> = runner.agents.iter().map(|a| a.car.gpu_export(&world.vehicle, &gw.arrays.surfaces).unwrap()).collect();
        // Sentinel controls: cars outside the mask must keep them.
        let agents: Vec<GpuAgent> = (0..networks).map(|i| GpuAgent { controls: [-(i as f64) - 0.5; 5], ..Default::default() }).collect();
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
                let out = a.network.forward_into(&inputs, &mut forward);
                let mut c = altd_sim::car::Controls::default();
                for (name, &v) in outputs.iter().zip(out) {
                    c.set(name, v);
                }
                [c.acceleration, c.steering, c.brake, c.handbrake, c.boost]
            })
            .collect();
        seconds.1 += t.elapsed().as_secs_f64();
        for (i, w) in want.iter().enumerate() {
            let same = (0..5).all(|c| w[c].to_bits() == got[i].controls[c].to_bits());
            if w[0].to_bits() == agents[i].controls[0].to_bits() { kept += 1 } else { updated += 1 }
            tally.record(back[i].error, same, || format!("tick {} car {i}: gpu {:?} cpu {w:?}", runner.tick, got[i].controls));
        }
    }
    println!("  {updated} control updates, {kept} unchanged (outside the batch); gpu {:.3} s incl. copies, cpu {:.3} s", seconds.0, seconds.1);
    tally.report("infer")
}

/// GPU car with the unused pair/contact slots cleared, as Debug text (f32/f64
/// Debug output round-trips, so equal text means equal bits up to NaN payloads).
fn car_text(c: &GpuCar) -> String {
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
fn check_step(gpu: &Gpu, networks: usize, warmup: u64, ticks: u64) -> bool {
    let mut runner = b06_runner(networks);
    runner.advance(warmup, false);
    let world = runner.world.clone();
    let gw = GpuWorld::new(gpu, &world);
    grid_line("shape", gw.arrays.shape_grid.as_ref());
    let vehicle = gpu_sim::vehicle_desc(&world.vehicle).unwrap();
    let descs: Vec<_> = runner.layout.sensors.iter().map(gpu_sim::sensor_desc).collect();
    let mut sim = GpuSim::new(&gw, &vehicle, &descs, networks);
    let mut cars: Vec<Car> = runner.agents.iter().map(|a| a.car.clone()).collect();
    let export = |cars: &[Car]| -> Vec<GpuCar> {
        cars.iter().enumerate().map(|(i, c)| c.gpu_export(&world.vehicle, &gw.arrays.surfaces).unwrap_or_else(|e| panic!("car {i}: {e}"))).collect()
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
            .map(|c| GpuAgent { controls: [c.acceleration, c.steering, c.brake, c.handbrake, c.boost], ..Default::default() })
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
            .map(|(car, c)| if drive { car.step(&world, c, DT, eliminate) } else { car.passive_step(&world, DT, eliminate) })
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
            println!("  tick {:4}: {active} active, {contacts} wall reports so far, {resyncs} resyncs", t + 1);
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
    println!("  step: gpu {:.3} s incl. copies, cpu {:.3} s (rayon)", seconds.0, seconds.1);
    tally.report("step") && round == 0
}

/// The differing `name: value` fields of two Debug texts (gpu -> cpu).
fn field_diff(gpu: &str, cpu: &str) -> String {
    gpu.split(", ").zip(cpu.split(", ")).filter(|(a, b)| a != b).map(|(a, b)| format!("{a} -> {b}")).collect::<Vec<_>>().join("; ")
}

/// GPU agent with the score-diff ring rotated to start at 0, unused slots cleared.
fn agent_text(a: &GpuAgent) -> String {
    let mut a = *a;
    let ring = a.recent;
    for k in 0..gpu_sim::RECENT {
        a.recent[k] = if k < a.recent_len as usize { ring[(a.recent_start as usize + k) % gpu_sim::RECENT] } else { 0.0 };
    }
    a.recent_start = 0;
    format!("{a:?}")
}

/// `TrainingAgent::update_stats` against the GPU stats kernel at every
/// statistics tick of a real B06 window (cars and agents exported from the
/// CPU each time; eliminate_when_idle alternates).
fn check_stats(gpu: &Gpu, networks: usize, ticks: u64) -> bool {
    let mut runner = b06_runner(networks);
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
            let cars: Vec<GpuCar> = runner.agents.iter().map(|a| a.car.gpu_export(&world.vehicle, &gw.arrays.surfaces).unwrap()).collect();
            let agents: Vec<GpuAgent> = runner.agents.iter().map(|a| gpu_sim::agent_export(a).unwrap()).collect();
            // A round trip through agent_import must reproduce the export.
            let mut probe = runner.agents[0].clone();
            gpu_sim::agent_import(&agents[0], &mut probe);
            assert_eq!(agent_text(&gpu_sim::agent_export(&probe).unwrap()), agent_text(&agents[0]), "agent import round trip");
            sim.upload(&cars, Some(&agents));
            let s = Instant::now();
            sim.stats(tick, tick - first, idle);
            sim.download(Some(&mut got_cars), Some(&mut got_agents));
            seconds.0 += s.elapsed().as_secs_f64();
            let mut want = runner.agents.clone();
            let s = Instant::now();
            want.par_iter_mut().for_each(|a| a.update_stats(&world, tick, phase, idle));
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
                    format!("tick {tick} car {i} (idle {idle}) car same {}: {}", gc == sc, field_diff(&ga, &sa))
                });
            }
        }
        runner.advance(1, false);
        if runner.tick % 600 == 0 {
            let active = runner.agents.iter().filter(|a| a.car.active).count();
            let best = runner.agents.iter().map(|a| a.stats.score.lap_count).max().unwrap_or(0);
            println!("  tick {:5}: {active} active, max laps {best}", runner.tick);
        }
    }
    println!("  {calls} stats ticks: {laps} lap completions, {idle_kills} idle eliminations, {contacts} pending contacts");
    println!("  stats: gpu {:.3} s incl. copies, cpu {:.3} s (rayon)", seconds.0, seconds.1);
    tally.report("stats")
}

/// Compares every agent (statistics, controls, car) of two runners.
fn compare_runners(gw: &GpuWorld, world: &World, cpu: &TrainingRunner, gpu: &TrainingRunner, tally: &mut Tally, what: &str) {
    if (cpu.tick, cpu.batch_index) != (gpu.tick, gpu.batch_index) {
        eprintln!("    {what}: tick/batch cpu ({}, {}) gpu ({}, {})", cpu.tick, cpu.batch_index, gpu.tick, gpu.batch_index);
        tally.record(0, false, || "window length".into());
    }
    for (i, (c, g)) in cpu.agents.iter().zip(&gpu.agents).enumerate() {
        let text = |a: &altd_sim::training::TrainingAgent| {
            let car = a.car.gpu_export(&world.vehicle, &gw.arrays.surfaces).unwrap();
            (agent_text(&gpu_sim::agent_export(a).unwrap()), car_text(&car), a.stats.metrics().map(|m| m.map(f64::to_bits)))
        };
        let (want, got) = (text(c), text(g));
        tally.record(0, want == got, || {
            format!("{what} car {i}: agent {} | car {}", field_diff(&got.0, &want.0), field_diff(&got.1, &want.1))
        });
    }
}

/// Whole training generations: `advance_generation` on the CPU against
/// `advance_generation_gpu`, then `next_generation` on both and again.
fn check_window(gpu: &Gpu, networks: usize, ticks: u64, generations: usize, configs: &[(bool, bool)]) -> bool {
    let mut tally = Tally::default();
    for &(eliminate_on_wall, eliminate_when_idle) in configs {
        let mut cpu = b06_runner(networks);
        cpu.eliminate_on_wall = eliminate_on_wall;
        cpu.eliminate_when_idle = eliminate_when_idle;
        let world = cpu.world.clone();
        let gw = GpuWorld::new(gpu, &world);
        let mut sim = cpu.gpu_sim(&gw, networks).unwrap();
        let mut on_gpu = b06_runner(networks);
        on_gpu.eliminate_on_wall = eliminate_on_wall;
        on_gpu.eliminate_when_idle = eliminate_when_idle;
        compare_runners(&gw, &world, &cpu, &on_gpu, &mut tally, "setup");
        println!("  {networks} cars, eliminate_on_wall {eliminate_on_wall}, eliminate_when_idle {eliminate_when_idle}:");
        for generation in 0..generations {
            let t = Instant::now();
            let executed = cpu.advance_generation(ticks);
            let cpu_seconds = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let gpu_executed = on_gpu.advance_generation_gpu(&mut sim, &gw, ticks).unwrap_or_else(|e| panic!("{e}"));
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
                compare_runners(&gw, &world, &cpu, &on_gpu, &mut tally, &format!("install {}", generation + 1));
            }
        }
    }
    tally.report("window")
}

/// GPU-only generation timing (no CPU reference run), without eliminations
/// so every car drives the whole window.
fn bench_window(gpu: &Gpu, networks: usize, ticks: u64, eliminate: bool) -> bool {
    let mut runner = b06_runner(networks);
    runner.eliminate_on_wall = eliminate;
    runner.eliminate_when_idle = eliminate;
    let world = runner.world.clone();
    let gw = GpuWorld::new(gpu, &world);
    let mut sim = runner.gpu_sim(&gw, networks).unwrap();
    for generation in 0..2 {
        let t = Instant::now();
        let executed = runner.advance_generation_gpu(&mut sim, &gw, ticks).unwrap_or_else(|e| panic!("{e}"));
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
fn bench_fixed(gpu: &Gpu) -> bool {
    let number = |name: &str, default: usize| -> usize {
        std::env::var(name).map(|v| v.parse().expect(name)).unwrap_or(default)
    };
    let networks = number("ALTD_GPU_BENCH_CARS", 8192);
    let ticks = number("ALTD_GPU_BENCH_TICKS", 5400) as u64;
    let samples = number("ALTD_GPU_BENCH_SAMPLES", 3);
    let warmups = number("ALTD_GPU_BENCH_WARMUPS", 1);
    let eliminate = number("ALTD_GPU_BENCH_ELIMINATE", 1) != 0;
    assert!(networks > 0 && ticks > 0 && samples > 0);
    let mut runner = b06_runner(networks);
    runner.eliminate_on_wall = eliminate;
    runner.eliminate_when_idle = eliminate;
    let world = runner.world.clone();
    let gw = GpuWorld::new(gpu, &world);
    let mut sim = runner.gpu_sim(&gw, networks).unwrap();
    let initial_cars: Vec<_> = runner.agents.iter().map(|a| a.car.gpu_export(&world.vehicle, &gw.arrays.surfaces).unwrap()).collect();
    let initial_agents: Vec<_> = runner.agents.iter().map(|a| gpu_sim::agent_export(a).unwrap()).collect();
    let mut expected = None;
    for sample in 0..warmups + samples {
        runner.tick = 0;
        runner.batch_index = 0;
        runner.agents.par_iter_mut().zip(&initial_cars).zip(&initial_agents).for_each(|((a, c), g)| {
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
        if let Some(ref previous) = expected { assert_eq!(previous, &digest, "repeated generation differs"); }
        expected = Some(digest.clone());
        println!("{}", serde_json::json!({
            "benchmark": "fixed_generation", "cars": networks, "ticks": executed,
            "eliminate": eliminate, "sample": sample, "warmup": sample < warmups,
            "seconds": seconds, "digest": digest,
            "active": runner.agents.iter().filter(|a| a.car.active).count(),
        }));
    }
    true
}

/// Consecutive generations, including reproduction, reset and network upload.
fn bench_training(gpu: &Gpu) -> bool {
    use std::hash::{Hash, Hasher};
    let number = |name: &str, default: usize| -> usize {
        std::env::var(name).map(|v| v.parse().expect(name)).unwrap_or(default)
    };
    let count = number("ALTD_GPU_BENCH_CARS", 32768);
    let generations = number("ALTD_GPU_BENCH_GENERATIONS", 3);
    let ticks = number("ALTD_GPU_BENCH_TICKS", 5400) as u64;
    let mut runner = b06_runner(count);
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
            for p in &a.network.params { p.to_bits().hash(&mut hash); }
            car_text(&a.car.gpu_export(&world.vehicle, &gw.arrays.surfaces).unwrap()).hash(&mut hash);
            agent_text(&gpu_sim::agent_export(a).unwrap()).hash(&mut hash);
        }
        println!("{}", serde_json::json!({"benchmark":"training", "cars":count, "generation":generation,
            "ticks":executed, "simulate_seconds":simulate, "turnover_seconds":turnover,
            "total_seconds":simulate+turnover, "digest":format!("{:016x}",hash.finish())}));
    }
    true
}

/// Partial windows exercise batch wrapping, graph re-use, list parity, and
/// populations that do not fill a wave or divide evenly into eight batches.
fn check_schedules(gpu: &Gpu) -> bool {
    let mut tally = Tally::default();
    for networks in [1, 17, 65, 257] {
        for (batch_count, stats_phase, batch_index) in [(1, 0, 0), (2, 1, 7), (3, 5, 3), (4, 0, 6), (8, 5, 7)] {
            let mut cpu = b06_runner(networks);
            let mut on_gpu = b06_runner(networks);
            for r in [&mut cpu, &mut on_gpu] {
                r.batch_count = batch_count;
                r.stats_phase = stats_phase;
                r.batch_index = batch_index;
                r.eliminate_on_wall = true;
                r.eliminate_when_idle = true;
            }
            let world = cpu.world.clone();
            let gw = GpuWorld::new(gpu, &world);
            let mut sim = cpu.gpu_sim(&gw, networks).unwrap();
            for ticks in [1, 5, 6, 7, 23, 24, 25, 49, 600] {
                let c = cpu.advance(ticks, true);
                let g = on_gpu.advance_window_gpu(&mut sim, &gw, ticks, true, None).unwrap();
                assert_eq!(c, g);
                compare_runners(&gw, &world, &cpu, &on_gpu, &mut tally,
                    &format!("n={networks} batches={batch_count} phase={stats_phase} window={ticks}"));
            }
        }
    }
    tally.report("schedules")
}

fn check_graph_reuse(gpu: &Gpu) -> bool {
    let fixture = b06_runner(257);
    let world = fixture.world.clone();
    let gw = GpuWorld::new(gpu, &world);
    let mut sim = fixture.gpu_sim(&gw, 257).unwrap();
    let mut tally = Tally::default();
    for (case, (count, hidden)) in [(257, 16), (257, 16), (17, 16), (65, 3), (65, 8), (1, 8), (257, 16)].into_iter().enumerate() {
        let mut cpu = b06_runner(count);
        let mut on_gpu = b06_runner(count);
        for runner in [&mut cpu, &mut on_gpu] {
            for agent in &mut runner.agents {
                let shape = vec![agent.network.shape[0], hidden, *agent.network.shape.last().unwrap()];
                let params = (0..network::parameter_count(&shape))
                    .map(|i| (((i + case) % 17) as f64 - 8.0) * 0.03125).collect();
                agent.network = network::Network::from_vector(&shape, params);
            }
        }
        sim.network_tag = None;
        // The first two cases reuse the same graph with different weights and
        // window lengths, including a remainder outside the captured period.
        let ticks = if case == 1 { 145 } else { 120 };
        cpu.advance(ticks, false);
        on_gpu.advance_window_gpu(&mut sim, &gw, ticks, false, None).unwrap();
        compare_runners(&gw, &world, &cpu, &on_gpu, &mut tally, &format!("reuse case {case}"));
    }
    tally.report("graph reuse")
}

fn check_novelty(gpu: &Gpu) -> bool {
    let fixture = b06_runner(257);
    let world = fixture.world.clone();
    let gw = GpuWorld::new(gpu, &world);
    let mut sim = fixture.gpu_sim(&gw, 257).unwrap();
    let mut tally = Tally::default();
    for count in [1, 17, 65, 257] {
        for hidden in [1, 3, 16] {
            let shape = [fixture.agents[0].network.shape[0], hidden, 5];
            let networks: Vec<_> = (0..count).map(|i| {
                let params = (0..network::parameter_count(&shape)).map(|j| {
                    ((i * 13 + j * 17) % 97) as f64 * 0.125 - 6.0
                }).collect();
                network::Network::from_vector(&shape, params)
            }).collect();
            sim.upload_networks(&networks, [-1; 5]).unwrap();
            let got = sim.novelty();
            let mean: Vec<f64> = (0..networks[0].params.len()).map(|j|
                networks.iter().fold(0.0, |a, n| a + n.params[j]) / count as f64).collect();
            for (i, n) in networks.iter().enumerate() {
                let expected = n.params.iter().zip(&mean).map(|(a, b)| double_math::pow(a-b, 2.0)).fold(0.0, |a,b| a+b).sqrt();
                tally.record(0, expected.to_bits() == got[i].to_bits(), || format!("novelty n={count} width={hidden} car={i}: {expected:?} / {:?}", got[i]));
            }
        }
    }
    tally.report("novelty")
}

fn check_turnover(gpu: &Gpu) -> bool {
    let mut tally = Tally::default();
    for game_rng in [false, true] {
        for (c, crossover) in ["none", "single_point", "uniform"].into_iter().enumerate() {
            for (s, selection) in ["best", "tournament", "roulette"].into_iter().enumerate() {
                let mut cpu = b06_runner(17);
                let mut on_gpu = b06_runner(17);
                for r in [&mut cpu, &mut on_gpu] {
                    if game_rng { r.rng = altd_sim::game_random::GameRandom::new([1,2,3,4], 12345).into(); }
                    r.settings.selection_algorithm = selection.into();
                    r.settings.selection_size = 4;
                    r.settings.crossover = crossover.into();
                    r.settings.mutation_rate = [0.0, 0.05, 0.1][c];
                    r.settings.adaptive_mutation = c % 2 == 0;
                    r.settings.weight_decay = [0.0, 0.01, 1.0][s];
                    r.settings.preserve_parents = "on_custom".into();
                    r.settings.preserve_parents_size = 1;
                }
                let world = cpu.world.clone();
                let gw = GpuWorld::new(gpu, &world);
                let mut sim = cpu.gpu_sim(&gw, 65).unwrap();
                for count in [17, 1, 65, 17] {
                    cpu.advance(49, false);
                    on_gpu.advance_window_gpu(&mut sim, &gw, 49, false, None).unwrap();
                    cpu.settings.population = count;
                    on_gpu.settings.population = count;
                    let expected = cpu.next_generation();
                    let got = on_gpu.next_generation_gpu(&mut sim).unwrap();
                    assert_eq!(expected.preserved_count, got.preserved_count);
                    assert_eq!(expected.rewards, got.rewards);
                    assert_eq!(cpu.rng.to_json(), on_gpu.rng.to_json());
                    for (i, (a, b)) in expected.networks.iter().zip(&got.networks).enumerate() {
                        tally.record(0, a.shape == b.shape && a.params.iter().map(|x| x.to_bits()).eq(b.params.iter().map(|x| x.to_bits())),
                            || format!("offspring {i}: game={game_rng} crossover={crossover} selection={selection} n={count}"));
                    }
                    compare_runners(&gw, &world, &cpu, &on_gpu, &mut tally, "turnover");
                }
            }
        }
    }
    tally.report("turnover")
}

/// Diagnostic: after one generation, single-steps the cars four more times and
/// classifies the inactive ones as fixed points, 2-cycles or still changing
/// (with the fields that change).
fn settle_report(gpu: &Gpu, networks: usize) -> bool {
    let mut runner = b06_runner(networks);
    runner.eliminate_on_wall = true;
    runner.eliminate_when_idle = true;
    let world = runner.world.clone();
    let gw = GpuWorld::new(gpu, &world);
    let mut sim = runner.gpu_sim(&gw, networks).unwrap();
    let executed = runner.advance_generation_gpu(&mut sim, &gw, 6000).unwrap_or_else(|e| panic!("{e}"));
    let text = |c: &GpuCar| {
        let mut c = *c;
        c.tick = 0;
        car_text(&c)
    };
    let mut states: Vec<Vec<GpuCar>> = vec![Vec::new()];
    sim.download(Some(&mut states[0]), None);
    for k in 0..4 {
        sim.step(true, true, 100_000 + k);
        let mut cars = Vec::new();
        sim.download(Some(&mut cars), None);
        states.push(cars);
    }
    let (mut fixed, mut cycle2, mut other) = (0, 0, 0);
    let mut fields: std::collections::BTreeMap<String, usize> = Default::default();
    let mut samples = Vec::new();
    for i in 0..networks {
        if states[0][i].flags & 1 != 0 {
            continue;
        }
        let t: Vec<String> = states.iter().map(|s| text(&s[i])).collect();
        if t[0] == t[1] {
            fixed += 1;
        } else if t[1] == t[3] && t[2] == t[4] {
            cycle2 += 1;
        } else {
            other += 1;
            let diff = field_diff(&t[2], &t[1]);
            for part in diff.split("; ") {
                let name: String = part.split(':').next().unwrap_or("").chars().filter(|c| c.is_alphanumeric() || *c == '_' || *c == ' ').collect();
                let name = name.split_whitespace().last().unwrap_or("").to_string();
                *fields.entry(name).or_default() += 1;
            }
            if samples.len() < 3 {
                samples.push(format!("car {i}: {diff}"));
            }
        }
    }
    println!("  after {executed} ticks: inactive fixed {fixed}, 2-cycle {cycle2}, other {other}");
    let mut by_count: Vec<_> = fields.into_iter().collect();
    by_count.sort_by(|a, b| b.1.cmp(&a.1));
    println!("  changing fields: {:?}", &by_count[..by_count.len().min(20)]);
    for s in samples {
        println!("  {}", &s[..s.len().min(1500)]);
    }
    true
}

fn ok_line(ok: bool, note: &str) -> bool {
    println!("    {note}");
    ok
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let parts: Vec<&str> = if args.is_empty() { vec!["math"] } else { args.iter().map(String::as_str).collect() };
    let gpu = Gpu::open(None).unwrap_or_else(|e| panic!("{e}"));
    let mut ok = true;
    for part in parts {
        ok &= match part {
            "math" => check_math(&gpu, 1 << 22, true),
            "rays" => check_rays(&gpu, &load_world(), 1 << 20),
            "sensors" => check_sensors(&gpu, 2048, 36, 150),
            "infer" => check_infer(&gpu, 2048, 36, 150),
            "step" => check_step(&gpu, 2048, 120, 400),
            "stats" => check_stats(&gpu, 2048, 3600),
            "window" => check_window(&gpu, 2048, 6000, 2, &[(true, false), (false, true)]),
            "bench" => bench_window(&gpu, 32768, 6000, false),
            "benchlive" => bench_window(&gpu, 32768, 6000, true),
            "bench2k" => bench_window(&gpu, 2048, 6000, false),
            "benchfixed" => bench_fixed(&gpu),
            "benchtrain" => bench_training(&gpu),
            "schedules" => check_schedules(&gpu),
            "reuse" => check_graph_reuse(&gpu),
            "novelty" => check_novelty(&gpu),
            "turnover" => check_turnover(&gpu),
            "window3" => check_window(&gpu, 2048, 6000, 2, &[(true, true)]),
            "settle" => settle_report(&gpu, 32768),
            other => panic!("unknown part {other}"),
        };
    }
    println!("{}", if ok { "ALL OK" } else { "FAILURES" });
    std::process::exit(if ok { 0 } else { 1 });
}
