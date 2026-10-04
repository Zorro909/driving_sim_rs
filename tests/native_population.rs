use altd_sim::{
    math::godot_math::F2,
    physics::car::{Car, Controls},
    physics::simulation::NativePopulation,
    track::curve::json_vector,
    track::world::World,
};
use serde_json::Value;
use std::sync::Arc;

fn check_game_population(fixture: &str, expected: u64) {
    let data: Value = serde_json::from_str(fixture).unwrap();
    let result = game_population::verify(
        &data["scene"],
        data["states"].as_array().unwrap(),
        data["controls"].as_array().unwrap(),
    );
    assert_eq!(result["exact"], expected, "{result}");
    assert!(result["errors"].as_array().unwrap().is_empty(), "{result}");
}

#[test]
fn coalesced_track_initialization_preserves_native_cell_order() {
    check_game_population(include_str!("fixtures/native_tilemap_coalesced600.json"), 1797);
}

#[test]
fn tilemap_redraw_preserves_native_wall_lifecycle() {
    check_game_population(include_str!("fixtures/native_tilemap_redraw600.json"), 1797);
}

#[test]
fn track_redraw_and_population_resize_preserve_deferred_order() {
    check_game_population(include_str!("fixtures/native_tilemap_redraw_resize600.json"), 2097);
}

fn bits(v: F2) -> [u32; 2] {
    [v.x.to_bits(), v.y.to_bits()]
}
fn check_car_state(car: &Car, state: &Value, id: usize, tick: &Value) {
    for (key, actual, expected) in [
        ("position", F2::from(car.position), json_vector(&state["position"])),
        ("velocity", F2::from(car.velocity), json_vector(&state["velocity"])),
        ("basis_x", car.body_basis.0, json_vector(&state["basis"][0])),
        ("basis_y", car.body_basis.1, json_vector(&state["basis"][1])),
    ] {
        assert_eq!(bits(actual), bits(expected), "{key}, car {id}, tick {}", tick);
    }
    assert_eq!(
        (car.angular_velocity as f32).to_bits(),
        (state["angular"].as_f64().unwrap() as f32).to_bits(),
        "angular, car {id}, tick {}",
        tick
    );
    let contacts = state["contacts"].as_array().unwrap();
    assert_eq!(car.wall_contacts.len(), contacts.len());
    for (a, e) in car.wall_contacts.iter().zip(contacts) {
        assert_eq!(bits(F2::from(a.normal)), bits(json_vector(&e["normal"])));
        assert_eq!(bits(F2::from(a.point)), bits(json_vector(&e["point"])));
    }
}
fn check(data: &str) {
    let data: Value = serde_json::from_str(data).unwrap();
    let world = Arc::new(World::from_scene(&data["scene"]));
    let frames = data["frames"].as_array().unwrap();
    let first = frames[0]["states"].as_array().unwrap();
    let cars = first
        .iter()
        .map(|s| Car::new(&world, json_vector(&s["position"]).into(), 0.0))
        .collect();
    let mut sim = NativePopulation::new(world.clone(), cars);
    let mut controls = vec![Controls::default(); first.len()];
    for _ in 0..8 {
        sim.step(&controls, false);
    }
    for (car, state) in sim.cars.iter_mut().zip(first) {
        car.velocity = json_vector(&state["velocity"]).into();
    }
    for frame in frames.iter().skip(1) {
        sim.step(&controls, false);
        let expected = frame["states"].as_array().unwrap();
        if expected.len() < sim.cars.len() {
            sim.truncate(expected.len());
        } else if expected.len() > sim.cars.len() {
            let extra = expected[sim.cars.len()..]
                .iter()
                .map(|state| {
                    let mut car = Car::new(&world, json_vector(&state["position"]).into(), 0.0);
                    car.velocity = json_vector(&state["velocity"]).into();
                    car
                })
                .collect::<Vec<_>>();
            sim.extend_cars(extra);
        }
        controls.resize(sim.cars.len(), Controls::default());
        for (id, (car, state)) in sim.cars.iter().zip(frame["states"].as_array().unwrap()).enumerate() {
            check_car_state(car, state, id, &frame["tick"]);
        }
    }
}
#[test]
fn population_uses_one_optimizer_and_native_constraint_order() {
    check(include_str!("fixtures/native_population80.json"));
}
#[test]
fn population_crosses_native_dynamic_leaf_capacity() {
    check(include_str!("fixtures/native_population_large80.json"));
}
#[test]
fn population_resize_retains_native_allocation_and_optimizer_history() {
    check(include_str!("fixtures/native_population_resize80.json"));
}

#[path = "support/population.rs"]
mod game_population;
#[test]
fn packed_game_population_preserves_reset_callbacks_and_contacts() {
    check_game_population(include_str!("fixtures/native_game_population180.json"), 537);
}
#[test]
fn original_tilemap_population_matches_native_drives_and_resets() {
    for (fixture, expected) in [
        (include_str!("fixtures/native_tilemap_a01_drive.json"), 537),
        (include_str!("fixtures/native_tilemap_b06_drive.json"), 537),
        (include_str!("fixtures/native_tilemap_a01_contact600.json"), 1797),
        (include_str!("fixtures/native_tilemap_b06_contact600.json"), 1797),
    ] {
        check_game_population(fixture, expected);
    }
}
#[test]
fn packed_vehicle_resize_preserves_mixed_native_pair_orientation() {
    for (fixture, expected) in [
        (include_str!("fixtures/native_tilemap_a01_resize600.json"), 2097),
        (include_str!("fixtures/native_tilemap_b06_resize600.json"), 2097),
    ] {
        check_game_population(fixture, expected);
    }
}
#[test]
fn original_vehicle_types_preserve_native_tilemap_contact_bits() {
    for (fixture, expected) in [
        (include_str!("fixtures/native_tilemap_truck600.json"), 1797),
        (include_str!("fixtures/native_tilemap_snowmobile600.json"), 1797),
    ] {
        check_game_population(fixture, expected);
    }
}

#[test]
fn training_runner_preserves_shared_native_physics_order() {
    use altd_sim::{
        math::vec2::V2,
        nn::network::Network,
        physics::car::Sensor,
        training::evolution::EvolutionSettings,
        training::game_random::GameRandom,
        training::{SensorLayout, TrainingRunner},
    };
    let data: Value = serde_json::from_str(include_str!("fixtures/native_population_coincident80.json")).unwrap();
    let world = Arc::new(World::from_scene(&data["scene"]));
    let settings = EvolutionSettings {
        population: 3,
        selection_size: 1,
        preserve_parents_size: 1,
        mutation_rate: 0.0,
        weight_decay: 0.0,
        ..Default::default()
    };
    let mut runner = TrainingRunner::new(
        world,
        V2::new(63.0, 79.5),
        0.0,
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
    runner.step();
    runner.step();
    let frames = data["frames"].as_array().unwrap();
    for (agent, state) in runner.agents.iter_mut().zip(frames[0]["states"].as_array().unwrap()) {
        agent.car.velocity = json_vector(&state["velocity"]).into();
    }
    for frame in frames.iter().skip(1) {
        runner.step();
        for (id, (agent, state)) in runner
            .agents
            .iter()
            .zip(frame["states"].as_array().unwrap())
            .enumerate()
        {
            let car = &agent.car;
            check_car_state(car, state, id, &frame["tick"]);
        }
    }
}

#[test]
fn generated_tracks_preserve_native_redraw_and_resize_physics() {
    check_game_population(include_str!("fixtures/native_generated_redraw_resize600.json"), 2097);
}

#[path = "../examples/support/generated.rs"]
mod generated;
#[path = "support/regression.rs"]
mod regression;

#[test]
fn generated_population_contacts_resize_and_redraw_match_pre_change_bits() {
    use altd_sim::math::vec2::V2;
    let mut scene = generated::scene("rally", 0);
    let mut next_scene = generated::scene("rally", 1);
    scene["track"]["native_broadphase"] = serde_json::json!(true);
    next_scene["track"]["native_broadphase"] = serde_json::json!(true);
    let golden: Value = serde_json::from_str(include_str!("fixtures/generated/population.json")).unwrap();
    let world = Arc::new(World::from_scene(&scene));
    let start = regression::vector(&golden["start_position"]);
    let cars = (0..3)
        .map(|i| {
            let mut car = Car::new(&world, start + V2::new(i as f64 * 8.0, 0.0), 0.2);
            car.velocity = V2::new(0.0, 160.0);
            car
        })
        .collect();
    let mut population = NativePopulation::new(world.clone(), cars);
    assert_eq!(
        serde_json::json!(population.cars.iter().map(regression::car_bits).collect::<Vec<_>>()),
        golden["initial"]
    );
    for row in golden["rows"].as_array().unwrap() {
        let event = &row["event"];
        if let Some(count) = event["truncate"].as_u64() {
            population.truncate(count as usize);
        }
        if event["extend"].is_object() {
            population.extend_cars([Car::new(
                &world,
                regression::vector(&event["extend"]["position"]),
                event["extend"]["rotation"].as_f64().unwrap(),
            )]);
        }
        if event["redraw"].is_string() {
            population.redraw_tilemap(Arc::new(World::from_scene(&next_scene)));
        }
        let control = regression::controls_from_json(&row["controls"]);
        population.step(&vec![control; population.cars.len()], false);
        assert_eq!(
            serde_json::json!(population.cars.iter().map(regression::car_bits).collect::<Vec<_>>()),
            row["expected"],
            "tick {}",
            row["tick"]
        );
    }
}
