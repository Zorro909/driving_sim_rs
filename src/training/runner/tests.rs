use super::*;
use crate::training::pyrandom::PyRandom;

fn load(path: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// A fixed generated track with the Formula vehicle.
fn generated_runner(population: usize) -> TrainingRunner {
    let root = env!("CARGO_MANIFEST_DIR");
    let template_scene = load(&format!("{root}/assets/scenes/formula_template.json"));
    let track_settings = serde_json::from_value(load(&format!("{root}/assets/random_track_settings.json"))).unwrap();
    let scene = crate::track::training_tracks::training_scene_at(
        &template_scene,
        &track_settings,
        None,
        1729,
        0,
        0,
        crate::math::profile::MathProfile::Proton,
    )
    .unwrap()
    .1;
    let template = load(&format!("{root}/assets/networks/formula.json"));
    let model = load(&format!("{root}/assets/models/formula.json"));
    let spawn = serde_json::json!({"position":scene["reset_position"],"rotation":scene["reset_rotation"]});
    let outputs: Vec<String> = template["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    let settings = EvolutionSettings {
        population,
        selection_size: 3,
        preserve_parents_size: 2,
        mutation_rate: 0.3,
        weight_decay: 0.0,
        ..Default::default()
    };
    TrainingRunner::new(
        Arc::new(World::from_scene(&scene)),
        crate::track::world::vector(&spawn["position"]),
        spawn["rotation"].as_f64().unwrap(),
        SensorLayout::from_exports(&template, &model, crate::math::profile::MathProfile::Proton),
        &outputs,
        settings,
        PyRandom::new(5),
        2,
        0,
        true,
        true,
    )
}

fn param_bits(networks: impl IntoIterator<Item = impl std::borrow::Borrow<Network>>) -> Vec<Vec<u64>> {
    networks
        .into_iter()
        .map(|n| n.borrow().params.iter().map(|p| p.to_bits()).collect())
        .collect()
}

/// Traced turnover installs what untraced turnover does, and its lineage
/// rebuilds those networks and the generator state from the parents alone.
#[test]
fn traced_turnover_matches_and_its_lineage_rebuilds_it() {
    let seed = Network::xavier(&[20, 8, 5], &mut PyRandom::new(9));
    for settings in [
        EvolutionSettings {
            population: 10,
            selection_size: 3,
            preserve_parents_size: 2,
            mutation_rate: 0.3,
            weight_decay: 0.0,
            ..Default::default()
        },
        EvolutionSettings {
            population: 7,
            selection_algorithm: "tournament".into(),
            selection_size: 4,
            crossover: "uniform".into(),
            mutation_rate: 0.2,
            adaptive_mutation: true,
            preserve_parents: "off".into(),
            ..Default::default()
        },
    ] {
        let (mut a, mut b) = (generated_runner(10), generated_runner(10));
        a.settings = settings.clone();
        b.settings = settings.clone();
        a.start(&seed);
        let first = b.start_traced(&seed);
        assert_eq!(
            first.parents.len(),
            1,
            "the first generation is bred from the seed alone"
        );
        let (rebuilt, rng) = first.rebuild();
        assert_eq!(param_bits(&rebuilt), param_bits(b.agents.iter().map(|x| &x.network)));
        assert_eq!(rng.to_json(), b.rng.to_json());
        for _ in 0..2 {
            a.advance(90, false);
            b.advance(90, false);
            let expected = a.next_generation();
            let (turnover, lineage) = b.next_generation_traced();
            assert_eq!(
                (turnover.preserved_count, turnover.rewards.clone()),
                (expected.preserved_count, expected.rewards.clone())
            );
            assert_eq!(
                param_bits(b.agents.iter().map(|x| &x.network)),
                param_bits(&expected.networks)
            );
            assert_eq!(b.rng.to_json(), a.rng.to_json());
            assert_eq!((b.generation, b.stats_phase), (a.generation, a.stats_phase));
            lineage.validate().unwrap();
            assert!(lineage.parents.len() <= 1 + settings.selection_size + 2);
            // The best parent leads; with parents kept it is also the first car.
            if expected.preserved_count > 0 {
                assert_eq!(param_bits([&lineage.parents[0]]), param_bits([&expected.networks[0]]));
            }
            let (rebuilt, rng) = lineage.rebuild();
            assert_eq!(param_bits(&rebuilt), param_bits(&expected.networks));
            assert_eq!(rng.to_json(), a.rng.to_json());
        }
    }
}

/// Without preservation nothing but the lineage's first slot holds the best
/// car, and tournament draws need not pick it first.
#[test]
fn the_best_car_leads_the_parents_without_preservation() {
    let seed = Network::xavier(&[20, 8, 5], &mut PyRandom::new(9));
    let mut runner = generated_runner(10);
    runner.settings = EvolutionSettings {
        population: 10,
        selection_algorithm: "tournament".into(),
        selection_size: 3,
        mutation_rate: 0.3,
        preserve_parents: "off".into(),
        ..Default::default()
    };
    runner.start(&seed);
    let mut first_draw_differed = false;
    for _ in 0..6 {
        runner.advance(90, false);
        let before: Vec<Network> = runner.agents.iter().map(|x| x.network.clone()).collect();
        let (turnover, lineage) = runner.next_generation_traced();
        assert_eq!(turnover.preserved_count, 0);
        assert!(lineage.preserved.is_empty());
        // The first car with the highest reward, as `ranked` keeps ties in order.
        let best = (0..turnover.rewards.len()).fold(0, |best, i| {
            if turnover.rewards[i] > turnover.rewards[best] {
                i
            } else {
                best
            }
        });
        assert_eq!(param_bits([&lineage.parents[0]]), param_bits([&before[best]]));
        let first_selected = &lineage.parents[lineage.selected[0] as usize];
        first_draw_differed |= param_bits([first_selected]) != param_bits([&before[best]]);
    }
    assert!(
        first_draw_differed,
        "the test must tell the best car from the first selected parent"
    );
}

#[test]
fn lineage_validate_rejects_malformed_networks() {
    let seed = Network::xavier(&[20, 8, 5], &mut PyRandom::new(9));
    let mut runner = generated_runner(10);
    let lineage = runner.start_traced(&seed);
    lineage.validate().unwrap();
    let invalid = |edit: &dyn Fn(&mut Lineage)| {
        let mut broken = lineage.clone();
        edit(&mut broken);
        assert!(broken.validate().is_err());
    };
    invalid(&|l| {
        l.parents[0].params.pop();
    });
    invalid(&|l| l.parents[0].params.push(0.0));
    invalid(&|l| {
        l.parents[0].shape = vec![20];
    });
    invalid(&|l| {
        l.parents[0].shape = Vec::new();
    });
    invalid(&|l| {
        l.parents[0].shape = vec![20, 0, 5];
    });
    invalid(&|l| {
        l.parents[0].shape = vec![usize::MAX, usize::MAX];
    });
    invalid(&|l| {
        l.parents[0].shape = vec![4, 4];
        l.parents[0].params = vec![0.0; 3];
    });
}

fn agent_state(a: &TrainingAgent) -> String {
    let c = &a.car;
    format!(
        "{:?} {:?} {:?} {:?} {} {} {} {:?} {:?} {:?} {:?} {:?}",
        c.position,
        c.velocity,
        c.acceleration,
        c.body_basis,
        c.active,
        c.tick,
        c.collision_count,
        c.wheels,
        a.controls,
        a.pending_contact,
        a.deactivated_at,
        a.stats
    )
}

/// Installing owned networks must leave the same agents as the cloning
/// path, including reused vehicles, statistics and network ownership.
#[test]
fn owned_install_matches_cloning_install() {
    let mut a = generated_runner(24);
    let mut b = generated_runner(24);
    let seed = Network::xavier(&[20, 16, 5], &mut PyRandom::new(9));
    a.start(&seed);
    b.start(&seed);
    a.advance(90, false);
    b.advance(90, false);
    let results: Vec<AgentResult> = a.agents.iter().map(TrainingAgent::result).collect();
    let generation = a
        .rng
        .clone()
        .reproduce(&results, &a.settings, crate::math::profile::MathProfile::Proton);
    a.install(&generation.networks, true);
    b.install_with_novelty(generation.networks.clone(), true, None);
    assert_eq!(a.agents.len(), b.agents.len());
    for (x, y) in a.agents.iter().zip(&b.agents) {
        assert_eq!(x.network, y.network);
        assert_eq!(agent_state(x), agent_state(y));
    }
    // With supplied novelty values the same agents result, novelty aside.
    let novelty: Vec<f64> = (0..generation.networks.len()).map(|i| i as f64 * 0.25).collect();
    let mut c = generated_runner(24);
    c.start(&seed);
    c.advance(90, false);
    c.install_with_novelty(generation.networks.clone(), true, Some(&novelty));
    for (i, (x, y)) in a.agents.iter().zip(&c.agents).enumerate() {
        assert_eq!(x.network, y.network);
        assert_eq!(y.stats.network_novelty, novelty[i]);
        let mut stats = y.stats.clone();
        stats.network_novelty = x.stats.network_novelty;
        assert_eq!(format!("{:?}", x.stats), format!("{stats:?}"));
    }
}

/// Drives `runner` through a ray window whose hits come from the CPU ray tree.
fn advance_with_cpu_ray_window(
    runner: &mut TrainingRunner,
    ticks: u64,
    stop_when_inactive: bool,
    time_limit: Option<f64>,
) -> u64 {
    use crate::math::godot_math::F2;
    let world = runner.world.clone();
    let tree = world.track.ray_tree().expect("the fixture has a BSP ray tree");
    let mut window = runner.begin_ray_window(ticks, stop_when_inactive, time_limit).unwrap();
    let mut queries = Vec::new();
    let mut hits = Vec::new();
    while runner.prepare_ray_tick(&mut window) {
        runner.ray_queries(&window, &mut queries);
        assert_eq!(queries.len(), window.inferring.len() * runner.layout.ray_count());
        hits.clear();
        hits.extend(queries.iter().map(|q| {
            match tree.raycast(V2::new(q[0] as f64, q[1] as f64), V2::new(q[2] as f64, q[3] as f64)) {
                Some(hit) => {
                    let h = F2::from(hit);
                    [h.x, h.y, 1.0, 0.0]
                }
                None => [0.0; 4],
            }
        }));
        runner.finish_ray_tick(&window, &hits).unwrap();
    }
    runner.end_ray_window(window)
}

fn assert_same_agents(a: &TrainingRunner, b: &TrainingRunner) {
    assert_eq!(
        (a.tick, a.batch_index, a.agents.len()),
        (b.tick, b.batch_index, b.agents.len())
    );
    for (x, y) in a.agents.iter().zip(&b.agents) {
        assert_eq!(agent_state(x), agent_state(y));
    }
    let (mut sa, mut sb) = (Vec::new(), Vec::new());
    a.car_states(&mut sa);
    b.car_states(&mut sb);
    assert_eq!(sa.len(), a.agents.len() * CAR_STATE_STRIDE);
    assert!(sa.iter().zip(&sb).all(|(p, q)| p.to_bits() == q.to_bits()));
}

/// The externally served ray window reproduces `advance` in both modes,
/// including partial windows, elimination stops and the time limit.
#[test]
fn ray_window_matches_advance() {
    let seed = Network::xavier(&[20, 16, 5], &mut PyRandom::new(9));
    for mode in [Mode::Independent, Mode::Lockstep] {
        let mut a = generated_runner(24);
        a.mode = mode;
        a.start(&seed);
        let mut b = generated_runner(24);
        b.start(&seed);
        assert_eq!(
            a.advance(75, false),
            advance_with_cpu_ray_window(&mut b, 75, false, None)
        );
        assert_same_agents(&a, &b);
        assert_eq!(a.advance(0, false), advance_with_cpu_ray_window(&mut b, 0, false, None));
        assert_same_agents(&a, &b);
        // Two windows of a short generation: the time limit stops both before driving.
        let (ticks, limit) = a.generation_window(600);
        assert_eq!(
            a.advance_window(ticks, true, Some(limit)),
            advance_with_cpu_ray_window(&mut b, ticks, true, Some(limit))
        );
        assert_same_agents(&a, &b);
        // Stops at the limit, or earlier once every car is eliminated.
        assert!(
            a.tick <= ticks + 75 && (a.tick > 500 || a.agents.iter().all(|x| !x.car.active)),
            "tick {}",
            a.tick
        );
        a.next_generation();
        b.next_generation();
        assert_same_agents(&a, &b);
        assert_eq!(
            a.advance_generation(180),
            advance_with_cpu_ray_window(&mut b, 200, true, Some(3.0))
        );
        assert_same_agents(&a, &b);
    }
    // Every car eliminated on walls or idle: the inactive stop ends the window.
    let mut a = generated_runner(8);
    a.start(&seed);
    let mut b = generated_runner(8);
    b.start(&seed);
    assert_eq!(
        a.advance(5400, true),
        advance_with_cpu_ray_window(&mut b, 5400, true, None)
    );
    assert!(
        a.agents.iter().all(|x| !x.car.active) && a.tick < 5400,
        "tick {}",
        a.tick
    );
    assert_same_agents(&a, &b);
}

/// The retained export buffers hold exactly what per-agent exports produce.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn export_state_into_matches_per_agent_export() {
    let mut runner = generated_runner(24);
    runner.start(&Network::xavier(&[20, 16, 5], &mut PyRandom::new(9)));
    runner.advance(150, false);
    let surfaces = crate::gpu::simulation::SurfaceTable::new(&runner.world.vehicle);
    let (mut cars, mut agents) = (vec![crate::gpu::simulation::GpuCar::zeroed(); 3], Vec::new());
    export_state_into(&runner.agents, &runner.world.vehicle, &surfaces, &mut cars, &mut agents).unwrap();
    assert_eq!((cars.len(), agents.len()), (24, 24));
    for (i, a) in runner.agents.iter().enumerate() {
        let car = a.car.gpu_export(&runner.world.vehicle, &surfaces).unwrap();
        assert_eq!(format!("{:?}", cars[i]), format!("{car:?}"));
        assert_eq!(
            format!("{:?}", agents[i]),
            format!("{:?}", crate::gpu::simulation::agent_export(a).unwrap())
        );
    }
}
