#[path = "support/rng.rs"]
mod rng;
#[path = "support/stats.rs"]
mod stats;
use altd_sim::{training::game_random::GameRandom, training::TrainingRandom};
use serde_json::{json, Value};
#[path = "../examples/support/generated.rs"]
mod generated;
#[path = "support/regression.rs"]
mod regression;
#[test]
fn random_streams_and_reproduction_match_original_assemblies() {
    let data: Value = serde_json::from_str(include_str!("fixtures/windows_random.json")).unwrap();
    let result = rng::verify(&data);
    assert_eq!(result["checked"], 1990);
    assert_eq!(result["errors"].as_array().unwrap().len(), 0, "{result}");
}
#[test]
fn generated_statistics_and_laps_match_pre_change_bits() {
    let scene = generated::scene("rally", 0);
    let world = altd_sim::track::world::World::from_scene(&scene);
    let golden: Value = serde_json::from_str(include_str!("fixtures/generated/stats.json")).unwrap();
    assert_eq!(regression::bit(world.track.path_length()), golden["path_length"]);
    let mut stats = altd_sim::training::TrainingStats::new(golden["network_novelty"].as_f64().unwrap());
    let mut log = String::new();
    for row in golden["rows"].as_array().unwrap() {
        let mut car = altd_sim::physics::car::Car::new(&world, regression::vector(&row["position"]), 0.0);
        car.velocity = regression::vector(&row["velocity"]);
        let controls = regression::controls_from_json(&row["controls"]);
        stats.update(
            &world,
            &car,
            &controls,
            row["tick"].as_u64().unwrap(),
            row["pending_contact"].as_bool().unwrap(),
        );
        assert_eq!(regression::stats_bits(&stats), row["expected"], "tick {}", row["tick"]);
        // Also exercise the captured-game statistics verifier with the generated golden.
        let expected = &row["expected"];
        let value = |bits: &Value| json!({"bits":bits});
        let names = [
            "TotalScore",
            "TotalSpeed",
            "TotalAbsSteering",
            "TotalActualDistance",
            "TotalDriftingAmount",
            "TotalMomentumChangeFrequency",
            "TotalThrottleChangeFrequency",
            "TotalBrakingFrequency",
            "TotalDistanceFromWall",
            "TotalDistanceFromCenter",
        ];
        let mut exported = serde_json::Map::new();
        for (i, name) in names.into_iter().enumerate() {
            exported.insert(name.into(), value(&expected["totals"][i + 1]));
        }
        exported.insert("PreviousTrackOffset".into(), value(&expected["score"]["offset"]));
        exported.insert("CollisionCount".into(), expected["counts"][1].clone());
        exported.insert("LapCount".into(), expected["score"]["laps"].clone());
        exported.insert("TotalRacingLineEfficiency".into(), value(&expected["metrics"][13]));
        exported.insert("BestLapTime".into(), value(&expected["best_lap_time"]));
        exported.insert("AllLapsTimeSum".into(), value(&expected["all_laps_time_sum"]));
        exported.insert("CurrentSlipAngleDegrees".into(), value(&expected["totals"][14]));
        let outputs: Vec<_> = ["Acceleration", "Steering", "Brake", "Handbrake", "Boost"]
            .into_iter()
            .zip(row["controls"].as_array().unwrap())
            .map(|(name, value)| json!({"name":name,"value":value}))
            .collect();
        let sample = json!({"tick":row["tick"],"position":row["position"],"velocity":row["velocity"],"basis":[[1.0,0.0],[0.0,1.0]],"active":true,"outputs":outputs,"collided":row["pending_contact"],"stats":exported,"hidden":{
         "_idleTicks":expected["counts"][2],"_updateCounter":expected["counts"][0],"_consecutiveDriftTicks":expected["counts"][5],"_previousSpeed":value(&expected["totals"][11]),"_previousThrottle":value(&expected["totals"][12]),
         "_recentScoreDiffs":expected["recent_score_diffs"].as_array().unwrap().iter().map(value).collect::<Vec<_>>(),"_continuousDriftTime":value(&expected["totals"][15]),"_maxContinuousDriftTime":value(&expected["totals"][16])
        }});
        log.push_str(&format!("FIDELITY_STATS {sample}\n"));
    }
    // The verifier's constructor starts novelty at zero; novelty does not affect these exported fields.
    let result = stats::verify(&scene, &log);
    assert!(result["checked"].as_u64().unwrap() > 4000);
    assert_eq!(result["mismatches"].as_object().unwrap().len(), 0, "{result}");
    assert!(stats.score.lap_count >= 2);
}
#[test]
fn checkpoint_roundtrip_preserves_both_game_streams() {
    let mut original = GameRandom::new([1, 2, 3, 4], 12345);
    original.normal_array(3, 0.1);
    original.randrange(37);
    let mut restored = GameRandom::from_json(&original.to_json());
    assert_eq!(original.normal_array(17, 0.2), restored.normal_array(17, 0.2));
    assert_eq!(original.random().to_bits(), restored.random().to_bits());
    for rng in [
        TrainingRandom::Game(original),
        TrainingRandom::Python(altd_sim::training::pyrandom::PyRandom::new(9)),
    ] {
        let json = rng.to_json();
        assert_eq!(TrainingRandom::from_json(&json).to_json(), json);
    }
}

fn tiny_runner() -> altd_sim::training::TrainingRunner {
    use altd_sim::{
        math::vec2::V2,
        nn::network::Network,
        physics::car::Sensor,
        track::world::World,
        training::evolution::EvolutionSettings,
        training::{SensorLayout, TrainingRunner},
    };
    let fixture: Value = serde_json::from_str(include_str!("fixtures/native_free180.json")).unwrap();
    let world = std::sync::Arc::new(World::from_scene(&fixture["scene"]));
    let settings = EvolutionSettings {
        population: 1,
        selection_size: 1,
        preserve_parents_size: 1,
        mutation_rate: 0.0,
        weight_decay: 0.0,
        ..Default::default()
    };
    let mut runner = TrainingRunner::new(
        world,
        V2::new(3840.0, 1664.0),
        std::f32::consts::FRAC_PI_2 as f64,
        SensorLayout {
            names: vec!["Boost".into()],
            sensors: vec![Sensor::BoostCapacity],
        },
        &["Acceleration".into()],
        settings,
        GameRandom::new([1, 2, 3, 4], 1),
        4,
        0,
        false,
        false,
    );
    runner.start(&Network::from_vector(&[1, 1], vec![0.0, 0.0]));
    runner
}

#[test]
fn idle_elimination_stops_scoring_but_retains_passive_cars() {
    for mode in [
        altd_sim::training::Mode::Independent,
        altd_sim::training::Mode::Lockstep,
    ] {
        let mut runner = tiny_runner();
        runner.mode = mode;
        runner.eliminate_when_idle = true;
        runner.advance(480, false);
        assert!(runner.agents[0].car.active);
        assert_eq!(runner.agents[0].stats.idle_ticks, 80);
        runner.advance(6, false);
        assert!(!runner.agents[0].car.active);
        let stats = format!("{:?}", runner.agents[0].stats);
        let physical_tick = runner.agents[0].car.tick;
        runner.advance(120, false);
        assert_eq!(runner.agents.len(), 1);
        assert_eq!(runner.agents[0].car.tick, physical_tick + 120);
        assert_eq!(format!("{:?}", runner.agents[0].stats), stats);
    }
}
#[test]
fn generation_stop_matches_original_scheduler_callbacks() {
    let oracle: Value = serde_json::from_str(include_str!("fixtures/windows_schedule.json")).unwrap();
    for name in ["time_limit", "inactive"] {
        let events = oracle[name].as_array().unwrap();
        let start = events.iter().find(|e| e["kind"] == "start").unwrap()["frame"]
            .as_u64()
            .unwrap();
        let end = events.iter().find(|e| e["kind"] == "transition").unwrap()["frame"]
            .as_u64()
            .unwrap();
        let mut runner = tiny_runner();
        if name == "inactive" {
            runner.step();
            runner.agents[0].car.deactivate();
            runner.agents[0].pending_contact = true;
        }
        runner.advance_generation(60);
        assert_eq!(runner.tick, end - start + 1);
        assert_eq!(
            runner.agents[0].car.tick,
            end - start,
            "transition callback must not drive"
        );
        assert!(!runner.agents[0].pending_contact);
        assert_eq!(
            runner.agents[0].stats.collision_count,
            if name == "inactive" { 1 } else { 0 }
        );
        if name == "time_limit" {
            assert_eq!(runner.agents[0].stats.update_count, 11);
        }
    }
}

#[test]
fn installed_generation_reaches_native_waiting_state_before_drive() {
    let oracle: Value = serde_json::from_str(include_str!("fixtures/native_warmup_unwrapped.json")).unwrap();
    let first = &oracle["states"][0];
    let mut runner = tiny_runner();
    runner.rotation = oracle["reset_rotation"].as_f64().unwrap();
    let network = runner.agents[0].network.clone();
    runner.resume(&[network], 1);
    let car = &runner.agents[0].car;
    let expected = altd_sim::track::curve::json_vector(&first["body_basis"][0]);
    assert_eq!(
        (car.body_basis.0.x.to_bits(), car.body_basis.0.y.to_bits()),
        (expected.x.to_bits(), expected.y.to_bits())
    );
    assert_eq!(runner.tick, 0);
    assert_eq!(car.tick, 0);
    for wheel in &car.wheels {
        assert_eq!(wheel.previous_position, Some(car.position));
    }
}

#[test]
fn every_original_sensor_name_loads_with_its_original_defaults() {
    use altd_sim::{physics::car::Sensor, training::SensorLayout};
    let original: Value = serde_json::from_str(include_str!("fixtures/native_sensor_layout.json")).unwrap();
    let layout = SensorLayout::from_exports(
        &serde_json::json!({"inputs":original["names"]}),
        &serde_json::json!({"vision":[]}),
    );
    for (actual, original) in layout.sensors.iter().zip(original["sensors"].as_array().unwrap()) {
        assert_eq!(*actual, Sensor::by_name(original["$type"].as_str().unwrap()));
        if let Sensor::AccelerationSide { max_acceleration } = actual {
            assert_eq!(*max_acceleration, original["MaxAcceleration"].as_f64().unwrap());
        }
    }
}

#[test]
fn random_tracks_and_rng_consumption_match_original_game() {
    use altd_sim::track::random_track::{self, RandomTrackConfig};
    let cases: Value = serde_json::from_str(include_str!("fixtures/windows_tracks.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let mut config: RandomTrackConfig = serde_json::from_value(case["config"].clone()).unwrap();
        let mut state: [u64; 4] = serde_json::from_value(case["state"].clone()).unwrap();
        let result = random_track::generate(&mut config, &mut state);
        if let Some(error) = case["error"].as_str() {
            assert_eq!(result.unwrap_err(), error, "case {}", case["i"]);
        } else {
            assert_eq!(
                serde_json::to_value(result.unwrap()).unwrap(),
                case["result"],
                "case {} generation {}",
                case["i"],
                case["generation"]
            );
        }
        assert_eq!(
            serde_json::to_value(state).unwrap(),
            case["after"],
            "RNG case {}",
            case["i"]
        );
        assert_eq!(
            serde_json::to_value(config).unwrap(),
            case["input_after"],
            "input config case {}",
            case["i"]
        );
    }
}

#[test]
fn generated_curve_controls_and_baking_match_original_engine() {
    use altd_sim::{track::curve::Curve, track::random_track::GeneratedTrack};
    let cases: Value = serde_json::from_str(include_str!("fixtures/windows_tracks.json")).unwrap();
    let curves: Value = serde_json::from_str(include_str!("fixtures/native_generated_curves.json")).unwrap();
    let vector_bits = |v: &Value| [v[0].as_f64().unwrap() as f32, v[1].as_f64().unwrap() as f32].map(f32::to_bits);
    for row in curves.as_array().unwrap() {
        let case = cases
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["i"] == row["i"] && c["generation"] == row["generation"])
            .unwrap();
        let track: GeneratedTrack = serde_json::from_value(case["result"].clone()).unwrap();
        let actual = track.curve_controls();
        let expected = &row["curve"];
        assert_eq!(
            actual["points"].as_array().unwrap().len(),
            expected["points"].as_array().unwrap().len()
        );
        for (i, (a, b)) in actual["points"]
            .as_array()
            .unwrap()
            .iter()
            .zip(expected["points"].as_array().unwrap())
            .enumerate()
        {
            for key in ["position", "incoming", "outgoing"] {
                assert_eq!(
                    vector_bits(&a[key]),
                    vector_bits(&b[key]),
                    "case {} point {i} {key}",
                    row["i"]
                );
            }
        }
        let curve = Curve::from_json(&actual);
        let baked = row["baked"].as_array().unwrap();
        assert_eq!(curve.points.len(), baked.len());
        for (i, (a, b)) in curve.points.iter().zip(baked).enumerate() {
            assert_eq!(
                [a.x.to_bits(), a.y.to_bits()],
                vector_bits(b),
                "case {} baked {i}",
                row["i"]
            );
        }
        assert_eq!(
            curve.length().to_bits(),
            (row["length"].as_f64().unwrap() as f32).to_bits()
        );
        let direction = curve.direction(0.0);
        let rotation = altd_sim::math::native_math::atan2(direction.y, direction.x) + std::f32::consts::PI / 2.0;
        assert_eq!(
            rotation.to_bits(),
            (row["rotation"].as_f64().unwrap() as f32).to_bits(),
            "spawn case {}",
            row["i"]
        );
    }
}

#[test]
fn random_training_transitions_install_generated_worlds_and_preserve_config() {
    use altd_sim::{nn::network::Network, track::random_track::RandomTrackConfig};
    let cases: Value = serde_json::from_str(include_str!("fixtures/windows_tracks.json")).unwrap();
    let template: Value = serde_json::from_str(include_str!("fixtures/native_free180.json")).unwrap();
    let mut runner = tiny_runner();
    let seed = Network::from_vector(&[1, 1], vec![0.0, 0.0]);
    let mut config: RandomTrackConfig = serde_json::from_value(cases[2]["config"].clone()).unwrap();
    let mut state = serde_json::from_value(cases[2]["state"].clone()).unwrap();
    let (generation, track) = runner
        .start_random_track(&seed, &template["scene"], &mut config, &mut state)
        .unwrap();
    assert_eq!(generation.networks.len(), 1);
    assert_eq!(config, track.config);
    assert_eq!(runner.world.track.shapes.len(), 86);
    assert_eq!(runner.agents[0].car.position, runner.position);
    assert_eq!(runner.agents[0].car.velocity, altd_sim::math::vec2::V2::ZERO);
    config = serde_json::from_value(cases[11]["config"].clone()).unwrap();
    state = serde_json::from_value(cases[11]["state"].clone()).unwrap();
    let (_, track) = runner
        .next_generation_random_track(&template["scene"], &mut config, &mut state)
        .unwrap();
    assert_eq!(serde_json::to_value(&track).unwrap(), cases[11]["result"]);
    assert_eq!(config.length, 68);
    assert_eq!(config, track.config);
    assert_eq!(runner.generation, 1);
    assert_eq!(runner.agents[0].car.position, runner.position);
    assert_eq!(runner.agents[0].car.velocity, altd_sim::math::vec2::V2::ZERO);
    // A failing track generation still follows reproduction, but installs no cars.
    runner.settings.population = 2;
    runner.settings.mutation_rate = 0.1;
    let old_world = runner.world.clone();
    let old_rng = runner.rng.to_json();
    config = serde_json::from_value(cases[8]["config"].clone()).unwrap();
    state = serde_json::from_value(cases[8]["state"].clone()).unwrap();
    assert!(runner
        .next_generation_random_track(&template["scene"], &mut config, &mut state)
        .is_err());
    assert!(std::sync::Arc::ptr_eq(&runner.world, &old_world));
    assert_eq!(runner.generation, 1);
    assert!(
        runner.rng.to_json() != old_rng,
        "reproduction must consume its RNG before track generation fails"
    );
    assert_eq!(serde_json::to_value(state).unwrap(), cases[8]["after"]);
}

#[path = "support/double_math.rs"]
mod double_math;
#[test]
fn double_math_matches_shipped_windows_runtime() {
    let data: Value = serde_json::from_str(include_str!("fixtures/windows_double_math.json")).unwrap();
    let result = double_math::verify(&data);
    assert!(result["checked"]["exp"].as_u64().unwrap() > 12000);
    assert_eq!(result["errors"].as_object().unwrap().len(), 0, "{result}");
}

#[path = "support/track_hash.rs"]
mod track_hash;
#[test]
fn process_hash_seed_matches_original_track_catalog_and_generation() {
    for fixture in [
        include_str!("fixtures/windows_tracks_hash0.json"),
        include_str!("fixtures/windows_tracks_hashmax.json"),
    ] {
        let data: Value = serde_json::from_str(fixture).unwrap();
        track_hash::verify(&data);
    }
}
#[test]
fn exact_sensor_export_retains_order_and_parameters() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/native_generated_ordered_sensors32.json")).unwrap();
    let model = serde_json::json!({"sensor_layout":fixture["sensor_layout"]});
    let network = serde_json::json!({"inputs":fixture["sensor_layout"]["names"]});
    let layout = altd_sim::training::SensorLayout::from_exports(&network, &model);
    assert_eq!(layout.names.len(), 10);
    assert_eq!(layout.names[0], layout.names[1]);
    match (layout.sensors[0], layout.sensors[1]) {
        (
            altd_sim::physics::car::Sensor::Raycast { length: a, .. },
            altd_sim::physics::car::Sensor::Raycast { length: b, .. },
        ) => {
            assert_eq!(a, 800.1234741210938);
            assert_eq!(b, 1200.654296875);
        }
        _ => panic!("ordered rays"),
    }
    let mut incomplete = fixture["sensor_layout"].clone();
    incomplete["sensors"][6]["PositionOffset"] = serde_json::json!({});
    assert!(altd_sim::training::SensorLayout::from_ordered(&incomplete).is_err());
}

#[test]
fn pause_preserves_original_scheduler_phase_and_batch_cursor() {
    let frames: Value = serde_json::from_str(include_str!("fixtures/windows_pause_schedule.json")).unwrap();
    let mut runner = tiny_runner();
    for frame in frames.as_array().unwrap() {
        if let Some(paused) = frame["change"].as_bool() {
            runner.set_paused(paused);
        }
        assert_eq!(runner.advance(1, false), 1);
        assert_eq!(runner.is_paused(), frame["paused"].as_bool().unwrap());
        assert_eq!(
            runner.tick,
            frame["logical"].as_u64().unwrap(),
            "frame {}",
            frame["frame"]
        );
        assert_eq!(
            runner.agents[0].stats.update_count,
            frame["stats"].as_u64().unwrap(),
            "frame {}",
            frame["frame"]
        );
        assert_eq!(runner.batch_index, frame["batch_index"].as_u64().unwrap() as usize);
    }
}

#[test]
fn paused_runner_matches_original_native_freeze_and_restore() {
    use altd_sim::{
        math::vec2::V2,
        nn::network::Network,
        physics::car::Sensor,
        track::world::World,
        training::evolution::EvolutionSettings,
        training::{Mode, SensorLayout, TrainingRunner},
    };
    let fixture: Value = serde_json::from_str(include_str!("fixtures/native_user_pause32.json")).unwrap();
    let states = fixture["states"].as_array().unwrap();
    let bits = |x: f64| (x as f32).to_bits();
    for mode in [Mode::Independent, Mode::Lockstep] {
        let world = std::sync::Arc::new(World::from_scene(&fixture["scene"]));
        let mut runner = TrainingRunner::new(
            world,
            V2::new(3840.0, 1664.0),
            std::f32::consts::FRAC_PI_2 as f64,
            SensorLayout {
                names: vec!["Boost".into()],
                sensors: vec![Sensor::BoostCapacity],
            },
            &["Acceleration", "Steering", "Brake", "Handbrake", "Boost"].map(str::to_owned),
            EvolutionSettings {
                population: 1,
                ..Default::default()
            },
            GameRandom::new([1, 2, 3, 4], 0),
            1,
            0,
            false,
            false,
        );
        let params = fixture["network"]["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        runner.resume(&[Network::from_vector(&[1, 5], params)], 0);
        runner.mode = mode;
        // This capture starts immediately after Reset, before the next native basis
        // reconstruction. Runner installation includes the waiting callbacks, so
        // restore the captured transform before testing the pause scheduler.
        runner.agents[0].car.set_body_basis(
            altd_sim::track::curve::json_vector(&states[0]["body_basis"][0]),
            altd_sim::track::curve::json_vector(&states[0]["body_basis"][1]),
        );
        for tick in 0..states.len() - 1 {
            for event in fixture["events"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|e| e["tick"].as_u64() == Some(tick as u64))
            {
                assert_eq!(event["kind"], "freeze");
                runner.set_paused(event["value"].as_bool().unwrap());
            }
            runner.step();
            let car = &runner.agents[0].car;
            let want = &states[tick + 1];
            for (key, v) in [
                ("position", car.position),
                ("velocity", car.velocity),
                ("acceleration", car.acceleration),
            ] {
                assert_eq!(
                    [bits(v.x), bits(v.y)],
                    [
                        bits(want[key][0].as_f64().unwrap()),
                        bits(want[key][1].as_f64().unwrap())
                    ],
                    "{key} tick {tick}"
                );
            }
            assert_eq!(
                bits(car.angular_velocity),
                bits(want["angular_velocity"].as_f64().unwrap()),
                "angular tick {tick}"
            );
            assert_eq!(
                bits(car.boost_energy),
                bits(want["boost_energy"].as_f64().unwrap()),
                "boost tick {tick}"
            );
            assert_eq!(car.frozen, want["frozen"].as_bool().unwrap());
        }
    }
}

/// Independent mode runs each car through a window alone; Lockstep runs the
/// original tick-major loop. Both must stop at the same tick with identical
/// cars, including when cars go inactive at different ticks.
#[test]
fn car_major_windows_match_tick_major_scheduler() {
    use altd_sim::{
        nn::network::Network,
        track::world::World,
        training::evolution::EvolutionSettings,
        training::{Mode, SensorLayout, TrainingRunner},
    };
    let scene = generated::scene("rally", 0);
    let model = generated::model("rally");
    let network = generated::network_export("rally");
    let spawn = generated::spawn(&scene);
    let outputs: Vec<String> = network["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    let world = std::sync::Arc::new(World::from_scene(&scene));
    let run = |mode: Mode, eliminate: bool, window: &dyn Fn(&mut TrainingRunner) -> u64| {
        let settings = EvolutionSettings {
            population: 48,
            selection_size: 1,
            preserve_parents_size: 1,
            mutation_rate: 0.5,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut runner = TrainingRunner::new(
            world.clone(),
            altd_sim::track::world::vector(&spawn["position"]),
            spawn["rotation"].as_f64().unwrap(),
            SensorLayout::from_exports(&network, &model),
            &outputs,
            settings,
            GameRandom::new([1, 2, 3, 4], 3),
            2,
            0,
            eliminate,
            eliminate,
        );
        runner.mode = mode;
        runner.start(&Network::from_game_export(&network));
        let executed = window(&mut runner);
        let cars: Vec<String> = runner
            .agents
            .iter()
            .map(|a| {
                let c = &a.car;
                format!(
                    "{:?} {:?} {:?} {} {} {} {} {} {:?} {:?} {:?}",
                    [
                        c.position.x.to_bits(),
                        c.position.y.to_bits(),
                        c.velocity.x.to_bits(),
                        c.velocity.y.to_bits()
                    ],
                    [
                        c.angular_velocity.to_bits(),
                        c.rotation.to_bits(),
                        c.boost_energy.to_bits()
                    ],
                    [c.acceleration.x.to_bits(), c.acceleration.y.to_bits()],
                    c.active,
                    c.frozen,
                    c.tick,
                    c.collision_count,
                    a.pending_contact,
                    a.deactivated_at,
                    a.stats,
                    a.result().metrics
                )
            })
            .collect();
        (executed, runner.tick, runner.batch_index, cars)
    };
    type Window<'a> = (&'a str, bool, &'a dyn Fn(&mut TrainingRunner) -> u64);
    let windows: [Window<'_>; 4] = [
        ("all inactive", true, &|r| r.advance_generation(5400)),
        ("time limit", false, &|r| r.advance_generation(301)),
        ("some inactive", true, &|r| r.advance(250, true)),
        ("fixed window", true, &|r| r.advance(250, false)),
    ];
    for (name, eliminate, window) in windows {
        let (car_major, tick_major) = (
            run(Mode::Independent, eliminate, window),
            run(Mode::Lockstep, eliminate, window),
        );
        assert_eq!(car_major.0, tick_major.0, "{name}: executed");
        assert_eq!(car_major.1, tick_major.1, "{name}: tick");
        assert_eq!(car_major.2, tick_major.2, "{name}: batch index");
        for (i, (a, b)) in car_major.3.iter().zip(&tick_major.3).enumerate() {
            assert_eq!(a, b, "{name}: car {i}");
        }
        let inactive = car_major.3.iter().filter(|c| c.contains(" false false ")).count();
        eprintln!("{name}: {} ticks, {inactive}/48 inactive", car_major.0);
    }
}
#[test]
fn vision_lengths_match_the_game_editor_for_every_whole_angle() {
    use altd_sim::{
        physics::car::Sensor,
        training::{vision_length, SensorLayout},
    };
    let data: Value = serde_json::from_str(include_str!("fixtures/native_vision_lengths.json")).unwrap();
    let rows = data["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 361);
    let mismatches: Vec<_> = rows
        .iter()
        .filter(|r| vision_length(r[0].as_f64().unwrap() as f32).to_bits() as u64 != r[1].as_u64().unwrap())
        .collect();
    assert!(mismatches.is_empty(), "{mismatches:?}");
    // Unlisted angles fall back to the editor length; listed ones keep the model's.
    let layout = SensorLayout::from_exports(
        &serde_json::json!({"inputs":["↑ 45°","↑ 0°"]}),
        &serde_json::json!({"vision":[{"angle":0,"length":123.0}]}),
    );
    let lengths: Vec<_> = layout
        .sensors
        .iter()
        .map(|s| match s {
            Sensor::Raycast { length, .. } => *length,
            _ => unreachable!(),
        })
        .collect();
    assert_eq!(lengths, [vision_length(45.0) as f64, 123.0]);
}
