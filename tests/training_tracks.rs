use altd_sim::track::training_tracks::{RandomTrainingTrackSettings, TrainingTrackBuffer};
use serde_json::{json, Value};

// HIP graph capture uses the process's legacy stream; GPU cases must not overlap.
static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn template() -> Value {
    serde_json::from_str::<Value>(include_str!("fixtures/native_free180.json")).unwrap()["scene"].clone()
}

#[test]
fn buffered_tracks_are_fresh_bounded_and_reproducible_on_resume() {
    let settings: RandomTrainingTrackSettings = serde_json::from_value(json!({
        "length": {"min": 6, "max": 12}, "allow_double": [false, true], "distribution": [0, 1]
    }))
    .unwrap();
    let small = TrainingTrackBuffer::new(template(), settings.clone(), 19, 0, 1, false).unwrap();
    let large = TrainingTrackBuffer::new(template(), settings.clone(), 19, 0, 4, false).unwrap();
    let resumed = TrainingTrackBuffer::new(template(), settings.clone(), 19, 2, 2, false).unwrap();
    assert!(TrainingTrackBuffer::new(template(), settings, 19, 0, 0, false).is_err());
    let mut previous = None;
    for generation in 0..4 {
        let a = small.next_track().unwrap();
        let b = large.next_track().unwrap();
        assert_eq!(a.generation, generation);
        assert_eq!(a.track, b.track);
        assert!((6..=12).contains(&a.track.config.length));
        assert!(!a.world.track.native_broadphase);
        assert!(a.world.track.curve.is_some());
        assert_ne!(previous.as_ref(), Some(&a.track.tiles));
        if generation >= 2 {
            assert_eq!(a.track, resumed.next_track().unwrap().track);
        }
        previous = Some(a.track.tiles);
    }
    // Dropping these full buffers must disconnect and join their CPU producers.
}

#[test]
fn batched_track_slots_are_queue_independent_and_keep_track_zero() {
    let settings = RandomTrainingTrackSettings::default();
    let a = TrainingTrackBuffer::new_batched(template(), settings.clone(), 7, 2, 1, false, 3).unwrap();
    let b = TrainingTrackBuffer::new_batched(template(), settings.clone(), 7, 2, 8, false, 3).unwrap();
    let single = TrainingTrackBuffer::new(template(), settings.clone(), 7, 2, 1, false).unwrap();
    for generation in 2..4 {
        for track_index in 0..3 {
            let track = a.next_track().unwrap();
            assert_eq!((track.generation, track.track_index), (generation, track_index));
            assert_eq!(track.track, b.next_track().unwrap().track);
            if track_index == 0 {
                assert_eq!(track.track, single.next_track().unwrap().track);
            }
        }
    }
}

fn batch_runner(
    world: std::sync::Arc<altd_sim::track::world::World>,
    position: altd_sim::math::vec2::V2,
    rotation: f64,
) -> altd_sim::training::TrainingRunner {
    use altd_sim::{
        nn::network::Network,
        physics::car::Sensor,
        training::evolution::EvolutionSettings,
        training::pyrandom::PyRandom,
        training::{SensorLayout, TrainingRunner},
    };
    let settings = EvolutionSettings {
        population: 17,
        selection_size: 4,
        mutation_rate: 0.01,
        ..Default::default()
    };
    let layout = SensorLayout {
        names: vec!["ray".into()],
        sensors: vec![Sensor::Raycast {
            degrees: 0.0,
            length: 800.0,
        }],
    };
    let mut runner = TrainingRunner::new(
        world,
        position,
        rotation,
        layout,
        &["Acceleration".into(), "Steering".into()],
        settings,
        PyRandom::new(9),
        2,
        0,
        false,
        true,
    );
    runner.start(&Network::from_vector(&[1, 2], vec![0.01, 0.01, 0.7, 0.2]));
    runner
}

#[test]
fn reset_keeps_networks_rng_and_novelty_and_restarts_eliminated_agents() {
    let buffer = TrainingTrackBuffer::new(template(), RandomTrainingTrackSettings::default(), 7, 0, 1, false).unwrap();
    let first = buffer.next_track().unwrap();
    let mut runner = batch_runner(first.world, first.position, first.rotation);
    runner.advance_generation(30);
    let rng = runner.rng.to_json();
    let networks: Vec<_> = runner
        .agents
        .iter()
        .map(|a| (a.network.clone(), a.network.params.as_ptr(), a.stats.network_novelty))
        .collect();
    for agent in &mut runner.agents {
        agent.car.active = false;
        agent.stats.idle_ticks = 999;
        agent.pending_contact = true;
    }
    let next = buffer.next_track().unwrap();
    runner.replace_track(next.world, next.position, next.rotation);
    runner.reset_evaluation();
    assert_eq!(runner.rng.to_json(), rng);
    assert_eq!((runner.generation, runner.tick, runner.batch_index), (0, 0, 0));
    for (agent, (network, ptr, novelty)) in runner.agents.iter().zip(networks) {
        assert_eq!(agent.network, network);
        assert_eq!(agent.network.params.as_ptr(), ptr);
        assert_eq!(agent.stats.network_novelty, novelty);
        assert!(agent.car.active);
        assert_eq!(agent.stats.update_count, 0);
        assert_eq!(agent.stats.idle_ticks, 0);
        assert_eq!(agent.stats.best_lap_time, None);
        assert!(!agent.pending_contact);
        assert_eq!(agent.controls.acceleration, 0.0);
    }
}

#[test]
fn scored_single_track_turnover_preserves_both_rng_backends() {
    let buffer = TrainingTrackBuffer::new(template(), RandomTrainingTrackSettings::default(), 7, 0, 1, false).unwrap();
    let first = buffer.next_track().unwrap();
    for game in [false, true] {
        let mut a = batch_runner(first.world.clone(), first.position, first.rotation);
        let mut b = batch_runner(first.world.clone(), first.position, first.rotation);
        if game {
            a.rng = altd_sim::training::game_random::GameRandom::new([1, 2, 3, 4], 0).into();
            b.rng = a.rng.clone();
        }
        a.advance_generation(30);
        b.advance_generation(30);
        let scores = altd_sim::training::evolution::reward_values(&b.results(), &b.settings.rewards);
        let original = a.next_generation();
        let scored = b.next_generation_with_fitness(scores);
        assert_eq!(original.networks, scored.networks);
        assert_eq!(original.rewards, scored.rewards);
        assert_eq!(a.rng.to_json(), b.rng.to_json());
    }
}

#[test]
#[ignore = "requires a HIP GPU and gpu/build.sh"]
fn gpu_track_batches_match_cpu_and_only_advance_evolution_after_all_tracks() {
    let _gpu_lock = GPU_TEST_LOCK.lock().unwrap();
    use altd_sim::{
        gpu::hip::{Gpu, GpuWorld},
        training::batch_evaluation::BatchEvaluation,
    };
    let gpu = Gpu::open(None).unwrap();
    let buffer =
        TrainingTrackBuffer::new_batched(template(), RandomTrainingTrackSettings::default(), 83, 0, 4, true, 3)
            .unwrap();
    let mut first = buffer.next_track().unwrap();
    let mut cpu = batch_runner(first.world.clone(), first.position, first.rotation);
    let mut on_gpu = batch_runner(first.world.clone(), first.position, first.rotation);
    let mut world = GpuWorld::from_prepared(&gpu, first.gpu_world.take().unwrap());
    let mut sim = on_gpu.gpu_sim(&world, 17).unwrap();
    for generation in 0..2 {
        let mut a = BatchEvaluation::new(17);
        let mut b = BatchEvaluation::new(17);
        let rng = cpu.rng.to_json();
        for slot in 0..3 {
            if slot > 0 || generation > 0 {
                let mut track = buffer.next_track().unwrap();
                for runner in [&mut cpu, &mut on_gpu] {
                    runner.replace_track(track.world.clone(), track.position, track.rotation);
                    runner.reset_evaluation();
                }
                let next = GpuWorld::from_prepared(&gpu, track.gpu_world.take().unwrap());
                sim.set_world(&next);
                world = next;
            }
            cpu.advance_generation(180);
            on_gpu.advance_generation_gpu(&mut sim, &world, 180).unwrap();
            assert_eq!(cpu.rng.to_json(), rng);
            assert_eq!(sim.network_tag, Some(generation));
            for (left, right) in cpu.agents.iter().zip(&on_gpu.agents) {
                assert_eq!(left.stats.metrics(), right.stats.metrics());
                assert_eq!(left.network, right.network);
            }
            a.record(&cpu.results(), &cpu.settings.rewards);
            b.record(&on_gpu.results(), &on_gpu.settings.rewards);
        }
        let (a, b) = (a.finish(), b.finish());
        assert_eq!(a.fitness, b.fitness);
        cpu.next_generation_with_fitness(a.fitness);
        on_gpu.next_generation_gpu_with_fitness(&mut sim, b.fitness).unwrap();
        assert_eq!(cpu.generation, generation + 1);
        assert_eq!(cpu.rng.to_json(), on_gpu.rng.to_json());
        for (a, b) in cpu.agents.iter().zip(&on_gpu.agents) {
            assert_eq!(a.network, b.network);
        }
    }
}

#[test]
#[ignore = "requires a HIP GPU and gpu/build.sh"]
fn gpu_track_switches_preserve_cpu_results_and_reuse_population_buffers() {
    let _gpu_lock = GPU_TEST_LOCK.lock().unwrap();
    use altd_sim::{
        gpu::hip::{Gpu, GpuWorld},
        math::vec2::V2,
        nn::network::Network,
        physics::car::Sensor,
        training::evolution::EvolutionSettings,
        training::pyrandom::PyRandom,
        training::{SensorLayout, TrainingRunner},
    };
    let gpu = Gpu::open(None).unwrap();
    let buffer = TrainingTrackBuffer::new(template(), RandomTrainingTrackSettings::default(), 83, 0, 4, true).unwrap();
    let mut first = buffer.next_track().unwrap();
    let make_runner = || {
        let settings = EvolutionSettings {
            population: 17,
            selection_size: 4,
            mutation_rate: 0.01,
            ..Default::default()
        };
        let layout = SensorLayout {
            names: vec!["ray".into(), "grip".into(), "dir".into(), "curve".into()],
            sensors: vec![
                Sensor::Raycast {
                    degrees: 0.0,
                    length: 800.0,
                },
                Sensor::Grip { offset: V2::ZERO },
                Sensor::CorrectDirection,
                Sensor::TrackCurvature {
                    min_lookahead: 0.0,
                    max_lookahead: 800.0,
                },
            ],
        };
        let mut runner = TrainingRunner::new(
            first.world.clone(),
            first.position,
            first.rotation,
            layout,
            &["Acceleration".into(), "Steering".into()],
            settings,
            PyRandom::new(9),
            2,
            0,
            false,
            false,
        );
        let mut params = vec![0.01; 10];
        // Network parameter layout is weights then biases for each layer.
        params[8] = 0.7;
        params[9] = 0.2;
        runner.start(&Network::from_vector(&[4, 2], params));
        runner
    };
    let mut cpu = make_runner();
    let mut on_gpu = make_runner();
    let mut gpu_world = GpuWorld::from_prepared(&gpu, first.gpu_world.take().unwrap());
    let mut sim = on_gpu.gpu_sim(&gpu_world, 17).unwrap();
    for generation in 0..3 {
        if generation > 0 {
            let mut track = buffer.next_track().unwrap();
            for runner in [&mut cpu, &mut on_gpu] {
                runner.replace_track(track.world.clone(), track.position, track.rotation);
            }
            let next_world = GpuWorld::from_prepared(&gpu, track.gpu_world.take().unwrap());
            sim.set_world(&next_world);
            gpu_world = next_world;
            cpu.next_generation();
            on_gpu.next_generation_gpu(&mut sim).unwrap();
        }
        // Enough ticks to capture and replay a HIP graph before replacing it.
        cpu.advance_generation(300);
        on_gpu.advance_generation_gpu(&mut sim, &gpu_world, 300).unwrap();
        assert_eq!((cpu.tick, cpu.batch_index), (on_gpu.tick, on_gpu.batch_index));
        assert_eq!(cpu.rng.to_json(), on_gpu.rng.to_json());
        for (a, b) in cpu.agents.iter().zip(&on_gpu.agents) {
            assert_eq!(a.network.params, b.network.params);
            assert_eq!(a.stats.metrics(), b.stats.metrics());
            let export = |agent: &altd_sim::training::TrainingAgent| {
                format!(
                    "{:?}",
                    agent
                        .car
                        .gpu_export(&cpu.world.vehicle, &gpu_world.arrays.surfaces)
                        .unwrap()
                )
            };
            assert_eq!(export(a), export(b), "generation {generation}");
        }
    }
}
