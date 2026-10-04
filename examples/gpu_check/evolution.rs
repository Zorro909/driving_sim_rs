//! Batch scheduling, graph reuse, novelty, turnover and inactive-state checks.
use super::fixtures::fixture_runner;
use super::shared::Tally;
use super::simulation::{car_text, compare_runners, field_diff};
use altd_sim::gpu::hip::{Gpu, GpuWorld};
use altd_sim::gpu::simulation::GpuCar;
use altd_sim::{math::double_math, nn::network};

/// Partial windows exercise batch wrapping, graph re-use, list parity, and
/// populations that do not fill a wave or divide evenly into eight batches.
pub(super) fn check_schedules(gpu: &Gpu) -> bool {
    let mut tally = Tally::default();
    for networks in [1, 17, 65, 257] {
        for (batch_count, stats_phase, batch_index) in [(1, 0, 0), (2, 1, 7), (3, 5, 3), (4, 0, 6), (8, 5, 7)] {
            let mut cpu = fixture_runner(networks);
            let mut on_gpu = fixture_runner(networks);
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
                compare_runners(
                    &gw,
                    &world,
                    &cpu,
                    &on_gpu,
                    &mut tally,
                    &format!("n={networks} batches={batch_count} phase={stats_phase} window={ticks}"),
                );
            }
        }
    }
    tally.report("schedules")
}

pub(super) fn check_graph_reuse(gpu: &Gpu) -> bool {
    let fixture = fixture_runner(257);
    let world = fixture.world.clone();
    let gw = GpuWorld::new(gpu, &world);
    let mut sim = fixture.gpu_sim(&gw, 257).unwrap();
    let mut tally = Tally::default();
    for (case, (count, hidden)) in [(257, 16), (257, 16), (17, 16), (65, 3), (65, 8), (1, 8), (257, 16)]
        .into_iter()
        .enumerate()
    {
        let mut cpu = fixture_runner(count);
        let mut on_gpu = fixture_runner(count);
        for runner in [&mut cpu, &mut on_gpu] {
            for agent in &mut runner.agents {
                let shape = vec![agent.network.shape[0], hidden, *agent.network.shape.last().unwrap()];
                let params = (0..network::parameter_count(&shape))
                    .map(|i| (((i + case) % 17) as f64 - 8.0) * 0.03125)
                    .collect();
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

pub(super) fn check_novelty(gpu: &Gpu) -> bool {
    let fixture = fixture_runner(257);
    let world = fixture.world.clone();
    let gw = GpuWorld::new(gpu, &world);
    let mut sim = fixture.gpu_sim(&gw, 257).unwrap();
    let mut tally = Tally::default();
    for count in [1, 17, 65, 257] {
        for hidden in [1, 3, 16] {
            let shape = [fixture.agents[0].network.shape[0], hidden, 5];
            let networks: Vec<_> = (0..count)
                .map(|i| {
                    let params = (0..network::parameter_count(&shape))
                        .map(|j| ((i * 13 + j * 17) % 97) as f64 * 0.125 - 6.0)
                        .collect();
                    network::Network::from_vector(&shape, params)
                })
                .collect();
            sim.upload_networks(&networks, [-1; 5]).unwrap();
            let got = sim.novelty();
            let mean: Vec<f64> = (0..networks[0].params.len())
                .map(|j| networks.iter().fold(0.0, |a, n| a + n.params[j]) / count as f64)
                .collect();
            for (i, n) in networks.iter().enumerate() {
                let expected = n
                    .params
                    .iter()
                    .zip(&mean)
                    .map(|(a, b)| double_math::pow(a - b, 2.0))
                    .fold(0.0, |a, b| a + b)
                    .sqrt();
                tally.record(0, expected.to_bits() == got[i].to_bits(), || {
                    format!("novelty n={count} width={hidden} car={i}: {expected:?} / {:?}", got[i])
                });
            }
        }
    }
    tally.report("novelty")
}

pub(super) fn check_turnover(gpu: &Gpu) -> bool {
    let mut tally = Tally::default();
    for game_rng in [false, true] {
        for (c, crossover) in ["none", "single_point", "uniform"].into_iter().enumerate() {
            for (s, selection) in ["best", "tournament", "roulette"].into_iter().enumerate() {
                let mut cpu = fixture_runner(17);
                let mut on_gpu = fixture_runner(17);
                for r in [&mut cpu, &mut on_gpu] {
                    if game_rng {
                        r.rng = altd_sim::training::game_random::GameRandom::new([1, 2, 3, 4], 12345).into();
                    }
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
                    // The GPU turnover's agents own the offspring it installed.
                    assert_eq!(expected.networks.len(), on_gpu.agents.len());
                    for (i, (a, b)) in expected
                        .networks
                        .iter()
                        .zip(on_gpu.agents.iter().map(|agent| &agent.network))
                        .enumerate()
                    {
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
pub(super) fn settle_report(gpu: &Gpu, networks: usize) -> bool {
    let mut runner = fixture_runner(networks);
    runner.eliminate_on_wall = true;
    runner.eliminate_when_idle = true;
    let world = runner.world.clone();
    let gw = GpuWorld::new(gpu, &world);
    let mut sim = runner.gpu_sim(&gw, networks).unwrap();
    let executed = runner
        .advance_generation_gpu(&mut sim, &gw, 6000)
        .unwrap_or_else(|e| panic!("{e}"));
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
                let name: String = part
                    .split(':')
                    .next()
                    .unwrap_or("")
                    .chars()
                    .filter(|c| c.is_alphanumeric() || *c == '_' || *c == ' ')
                    .collect();
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
