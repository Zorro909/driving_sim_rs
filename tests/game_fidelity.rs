//! Tiny regression cases from the decompiled game and recorded game states.
//! No population simulation, training, or benchmark workloads.
use altd_sim::{
    car::{Car, Controls, DT},
    training::TrainingStats,
    vec2::V2,
    world::World,
};
use serde_json::{json, Value};

fn scene() -> Value {
    serde_json::from_str(include_str!(
        "../../docs/traces/rally_a07_contact_scene.json"
    ))
    .unwrap()
}
fn folded_world() -> World {
    let mut data = scene();
    data["track"] = json!({"path": [[0,0], [1000,0], [1000,10], [0,10], [0,0]]});
    World::from_scene(&data)
}
#[test]
fn lap_times_interpolate_the_crossing_within_the_stats_interval() {
    let world = folded_world();
    let mut times = Vec::new();
    for (before, after) in [(2.0, 8.0), (8.0, 2.0)] {
        let mut stats = TrainingStats::new(0.0);
        stats.score.previous_offset = Some(world.track.path_length() - before);
        let car = Car::new(&world, V2::new(after, 0.0), 0.0);
        stats.update(&world, &car, &Controls::default(), 600, false);
        times.push(stats.best_lap_time.unwrap());
    }
    // EvolutionStatsLapsObserver.OnLapCompleted: Round(time-start-.1+fraction*.1, 3).
    assert_eq!(times, vec![9.920, 9.980]);
}
#[test]
fn center_distance_follows_the_score_track_section() {
    let world = folded_world();
    let mut stats = TrainingStats::new(0.0);
    stats.score.previous_offset = Some(500.0);
    let car = Car::new(&world, V2::new(500.0, 9.0), 0.0);
    stats.update(&world, &car, &Controls::default(), 6, false);
    assert_eq!(stats.score.previous_offset, Some(500.0));
    assert_eq!(stats.total_distance_from_center, 9.0);
}
#[test]
fn spawn_score_does_not_count_as_movement_for_idle_detection() {
    let world = folded_world();
    let mut stats = TrainingStats::new(0.0);
    let car = Car::new(&world, V2::new(500.0, 0.0), 0.0);
    stats.update(&world, &car, &Controls::default(), 6, false);
    assert!(stats.total_score > 0.0);
    assert_eq!(stats.recent_score_diffs.front(), Some(&0.0));
    assert_eq!(stats.idle_ticks, 1);
}
fn float(v: &Value) -> f64 {
    v.as_f64().unwrap() as f32 as f64
}
fn vector(v: &Value) -> V2 {
    V2::new(float(&v[0]), float(&v[1]))
}
fn replay_transition(world: &World, trace: &Value) -> Car {
    let frames = &trace["frames"];
    let previous = &frames[0];
    let state = &frames[1];
    let expected = &frames[2];
    let mut car = altd_sim::trace_state::car_from_frame(world, state, Some(previous), None);
    let mut controls = Controls::default();
    for output in expected["outputs"].as_array().unwrap() {
        controls.set(
            output["name"].as_str().unwrap(),
            output["value"].as_f64().unwrap(),
        );
    }
    car.step(world, &controls, DT, false);
    car
}
#[test]
fn a07_tick_four_reproduces_recorded_free_motion() {
    let world = World::from_scene(&scene());
    let trace: Value = serde_json::from_str(include_str!("fixtures/a07_tick4.json")).unwrap();
    let car = replay_transition(&world, &trace);
    let expected = &trace["frames"][2];
    assert!(car.wall_contacts.is_empty());
    assert_eq!(car.position, vector(&expected["position"]));
    assert_eq!(car.velocity, vector(&expected["velocity"]));
    assert_eq!(car.rotation, float(&expected["rotation"]));
    assert_eq!(car.angular_velocity, float(&expected["angular_velocity"]));
}

#[test]
fn a07_tick_two_reproduces_velocity_bits() {
    let world = World::from_scene(&scene());
    let trace: Value = serde_json::from_str(include_str!("fixtures/a07_tick2.json")).unwrap();
    let car = replay_transition(&world, &trace);
    assert_eq!(car.velocity, vector(&trace["frames"][2]["velocity"]));
}

#[test]
fn b06_tick_four_reproduces_rotation_bits() {
    let scene: Value = serde_json::from_str(include_str!("../../docs/traces/rally_b06_scene.json")).unwrap();
    let trace: Value = serde_json::from_str(include_str!("fixtures/b06_tick4.json")).unwrap();
    let car = replay_transition(&World::from_scene(&scene), &trace);
    assert_eq!(car.rotation, float(&trace["frames"][2]["rotation"]));
}

#[test]
fn a07_contact_at_tick_195_reproduces_velocity_bits_with_native_pair_order() {
    let mut world = World::from_scene(&scene());
    // Confirmed independently by the shipped native engine, wall-first probe.
    world.track.shapes[90].wall_first = true;
    let trace: Value = serde_json::from_str(include_str!("fixtures/a07_contact195.json")).unwrap();
    let frames = &trace["frames"];
    let mut car = altd_sim::trace_state::car_from_frame(&world, &frames[2], Some(&frames[1]), Some(&frames[0]));
    let mut controls = Controls::default();
    for output in frames[3]["outputs"].as_array().unwrap() {
        controls.set(output["name"].as_str().unwrap(), output["value"].as_f64().unwrap());
    }
    car.step(&world, &controls, DT, false);
    assert_eq!(car.velocity, vector(&frames[3]["velocity"]));
    assert_eq!(car.angular_velocity, float(&frames[3]["angular_velocity"]));
}

#[test]
fn managed_transform_trig_matches_windows_game_runtime() {
    // Captured from the installed GodotSharp under Windows .NET 8.0.2.
    for (angle, sin, cos) in [
        (0x00000000, 0x00000000, 0x3f800000),
        (0xbe7a35dd, 0xbe77ba60, 0x3f78654c),
        (0x3fc90fdb, 0x3f7fffff, 0xb3bbbd2e),
        (0x3f800000, 0x3f576aa4, 0x3f0a513f),
        (0xbf800000, 0xbf576aa4, 0x3f0a513f),
        (0x40490fdb, 0xb3bbbd2e, 0xbf7fffff),
    ] {
        let actual = altd_sim::godot_math::managed_sin_cos(f32::from_bits(angle));
        assert_eq!((actual.0.to_bits(), actual.1.to_bits()), (sin, cos));
    }
}

#[test]
fn bsp_reconstructs_the_games_exported_split_segments() {
    let world = World::from_scene(&scene());
    let mut segments = Vec::new();
    world.track.bsp.as_ref().unwrap().collect(&mut segments);
    let expected: Vec<_> = world.track.walls.iter().map(|w| altd_sim::world::Wall {
        start: V2::new(w.start.x as f32 as f64, w.start.y as f32 as f64),
        end: V2::new(w.end.x as f32 as f64, w.end.y as f32 as f64),
    }).collect();
    assert_eq!(segments, expected);
}
fn check_contact(world: World, trace: Value) {
    let car = replay_transition(&world, &trace);
    let expected = &trace["frames"][2];
    let position_error = (car.position - vector(&expected["position"])).length();
    let velocity_error = (car.velocity - vector(&expected["velocity"])).length();
    assert_eq!(car.wall_contacts.len(), 1);
    assert_eq!(
        car.wall_contacts[0].normal,
        vector(&expected["physics_contacts"][0]["local_normal"])
    );
    assert!(
        position_error < 0.0001 && velocity_error < 0.001,
        "position={position_error}, velocity={velocity_error}, contacts={:?}, game={}",
        car.wall_contacts,
        expected["physics_contacts"]
    );
}
#[test]
fn a07_contact_at_tick_663_matches_game() {
    check_contact(
        World::from_scene(&scene()),
        serde_json::from_str(include_str!("fixtures/a07_contact663.json")).unwrap(),
    );
}
#[test]
fn b06_first_contact_at_tick_1125_matches_game() {
    let scene: Value =
        serde_json::from_str(include_str!("../../docs/traces/rally_b06_scene.json")).unwrap();
    check_contact(
        World::from_scene(&scene),
        serde_json::from_str(include_str!("fixtures/b06_contact1125.json")).unwrap(),
    );
}

#[test]
fn a07_recorded_velocity_inputs_use_native_float_remap() {
    use altd_sim::car::{Sensor, SensorScratch};
    let world = World::from_scene(&scene());
    let trace: Value = serde_json::from_str(include_str!("fixtures/a07_tick4.json")).unwrap();
    // Game refreshes inputs on tick 3 from the body state at tick 2.
    let car = altd_sim::trace_state::car_from_frame(&world, &trace["frames"][0], None, None);
    let actual = trace["frames"][1]["sensors"].as_array().unwrap();
    for (name, sensor) in [
        ("VelF", Sensor::VelocityFront),
        ("VelS", Sensor::VelocitySide),
        ("AngVel", Sensor::AngularVelocity),
    ] {
        let expected = actual.iter().find(|s| s["name"] == name).unwrap()["value"]
            .as_f64()
            .unwrap();
        assert_eq!(
            car.sensor(&world, sensor, &mut SensorScratch::default()),
            expected,
            "{name}"
        );
    }
}

#[test]
fn native_transforms_match_shipped_engine_bits() {
    let rows: Vec<[u32; 4]> = serde_json::from_str(include_str!("fixtures/native_engine_math_bits.json")).unwrap();
    for [a, s, c, r] in rows {
        let angle = f32::from_bits(a);
        let (x, _) = altd_sim::godot_math::basis(angle as f64);
        assert_eq!((x.y.to_bits(), x.x.to_bits()), (s,c), "angle bits {a}");
        assert_eq!(altd_sim::native_math::atan2(x.y,x.x).to_bits(), r);
    }
}
#[test]
fn scalar_math_full_float_range_matches_original_runtime() {
 let rows:Vec<[u32;5]>=serde_json::from_str(include_str!("fixtures/windows_full_math_bits.json")).unwrap();
 for [input,s,c,_,_] in rows {
  let x=f32::from_bits(input);
  assert_eq!(altd_sim::native_math::sin(x).to_bits(),s,"sin {input:08x}");
  assert_eq!(altd_sim::native_math::cos(x).to_bits(),c,"cos {input:08x}");
 }
}

#[test]
fn packed_math_full_float_range_matches_original_runtime() {
 let rows:Vec<[u32;5]>=serde_json::from_str(include_str!("fixtures/windows_full_math_bits.json")).unwrap();
 for [input,_,_,s,c] in rows {
  let (actual_s,actual_c)=altd_sim::godot_math::managed_sin_cos(f32::from_bits(input));
  assert_eq!(actual_s.to_bits(),s,"packed sin {input:08x}");
  assert_eq!(actual_c.to_bits(),c,"packed cos {input:08x}");
 }
}

#[test]
fn engine_math_full_float_range_matches_original_runtime() {
 let rows:Vec<[u32;4]>=serde_json::from_str(include_str!("fixtures/native_full_math_bits.json")).unwrap();
 for [input,s,c,rotation] in rows {
  let x=f32::from_bits(input);
  assert_eq!(altd_sim::native_math::engine_sin(x).to_bits(),s,"engine sin {input:08x}");
  assert_eq!(altd_sim::native_math::engine_cos(x).to_bits(),c,"engine cos {input:08x}");
  assert_eq!(altd_sim::native_math::atan2(f32::from_bits(s),f32::from_bits(c)).to_bits(),rotation,"engine rotation {input:08x}");
 }
}
