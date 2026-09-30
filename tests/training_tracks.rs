use altd_sim::training_tracks::{RandomTrainingTrackSettings, TrainingTrackBuffer};
use serde_json::{json, Value};

fn template() -> Value {
    serde_json::from_str::<Value>(include_str!("fixtures/native_free180.json")).unwrap()["scene"]
        .clone()
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
#[ignore = "requires a HIP GPU and gpu/build.sh"]
fn gpu_track_switches_preserve_cpu_results_and_reuse_population_buffers() {
    use altd_sim::{
        car::Sensor,
        evolution::EvolutionSettings,
        gpu::{Gpu, GpuWorld},
        network::Network,
        pyrandom::PyRandom,
        training::{SensorLayout, TrainingRunner},
        vec2::V2,
    };
    let gpu = Gpu::open(None).unwrap();
    let buffer = TrainingTrackBuffer::new(
        template(),
        RandomTrainingTrackSettings::default(),
        83,
        0,
        4,
        true,
    )
    .unwrap();
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
        on_gpu
            .advance_generation_gpu(&mut sim, &gpu_world, 300)
            .unwrap();
        assert_eq!(
            (cpu.tick, cpu.batch_index),
            (on_gpu.tick, on_gpu.batch_index)
        );
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
