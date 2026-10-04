//! Continuous finite drives through the unmodified game vehicle and native engine.
use altd_sim::{
    math::godot_math::F2,
    math::native_math,
    math::vec2::V2,
    physics::car::{Car, Controls, Sensor, SensorScratch, DT},
    track::world::World,
};
use serde_json::Value;
fn f(v: &Value) -> f64 {
    v.as_f64().unwrap() as f32 as f64
}
fn vec(v: &Value) -> V2 {
    V2::new(f(&v[0]), f(&v[1]))
}
fn vector_bits(v: V2) -> [u32; 2] {
    [(v.x as f32).to_bits(), (v.y as f32).to_bits()]
}
fn scalar_bits(v: f64) -> u32 {
    (v as f32).to_bits()
}
fn check_sensor_sample(data: &Value, world: &World, car: &Car, tick: usize, phase: &str) {
    let Some(samples) = data["sensor_samples"].as_array() else {
        return;
    };
    let sample = samples
        .iter()
        .find(|s| s["tick"].as_u64() == Some(tick as u64) && s["phase"].as_str() == Some(phase))
        .unwrap();
    let mut kinds = vec![
        Sensor::by_name("accelerationFront"),
        Sensor::by_name("accelerationSide"),
        Sensor::AccelerationFront {
            max_acceleration: 10.1f32 as f64,
        },
        Sensor::AngularVelocity,
        Sensor::WheelAngle,
        Sensor::Grip { offset: V2::ZERO },
        Sensor::Grip {
            offset: V2::new(13.25, -9.75),
        },
        Sensor::VelocityFront,
        Sensor::VelocitySide,
        Sensor::Speed,
    ];
    if sample["values"].as_array().unwrap().len() > 10 {
        kinds.extend([
            Sensor::DistanceFromWall { max_distance: 150.0 },
            Sensor::DistanceFromWall {
                max_distance: 150.1f32 as f64,
            },
            Sensor::CorrectDirection,
            Sensor::TrackCurvature {
                min_lookahead: 256.0,
                max_lookahead: 512.0,
            },
            Sensor::TrackCurvature {
                min_lookahead: -20.0,
                max_lookahead: 30.0,
            },
            Sensor::TrackCurvature {
                min_lookahead: 300.0,
                max_lookahead: 100.0,
            },
        ]);
    }
    if !data["sensor_layout"].is_null() {
        kinds = altd_sim::training::SensorLayout::from_ordered(&data["sensor_layout"])
            .unwrap()
            .sensors;
    }
    let mut scratch = SensorScratch::default();
    for (i, kind) in kinds.into_iter().enumerate() {
        assert_eq!(
            scalar_bits(car.sensor(world, kind, &mut scratch)),
            scalar_bits(f(&sample["values"][i])),
            "sensor {kind:?} tick {tick} {phase}"
        );
    }
}
fn replay(data: &str, check_contacts: bool) {
    let data: Value = serde_json::from_str(data).unwrap();
    let mut scene = data["scene"].clone();
    let mut w = World::from_scene(&scene);
    let eliminate = data["eliminate"].as_bool().unwrap_or(false);
    let states = data["states"].as_array().unwrap();
    let first = &states[0];
    let x = F2::from(vec(&first["body_basis"][0]));
    let y = F2::from(vec(&first["body_basis"][1]));
    let mut car = Car::new(&w, vec(&first["position"]), native_math::atan2(x.y, x.x) as f64);
    car.set_body_basis(x, y);
    car.velocity = vec(&first["velocity"]);
    car.angular_velocity = f(&first["angular_velocity"]);
    for (i, wheel) in car.wheels.iter_mut().enumerate() {
        wheel.angle_deg = f(&first["wheel_angles"][i]);
        wheel.previous_position = Some(vec(&first["wheel_positions"][i]));
    }
    for _ in 0..data["warmup_steps"].as_u64().unwrap_or(0) {
        car.passive_step(&w, DT, eliminate);
    }
    for (i, state) in states.iter().enumerate().skip(1) {
        let t = i - 1;
        check_sensor_sample(&data, &w, &car, t, "before");
        if let Some(events) = data["events"].as_array() {
            for event in events.iter().filter(|e| e["tick"].as_u64() == Some(t as u64)) {
                match event["kind"].as_str().unwrap() {
                    "freeze" => car.set_frozen(event["value"].as_bool().unwrap()),
                    "active" => car.set_active(event["value"].as_bool().unwrap()),
                    "reset" => car.queue_reset(vec(&event["position"]), event["rotation"].as_f64().unwrap()),
                    "vehicle" | "wheel" | "surface" => {
                        let target = match event["kind"].as_str().unwrap() {
                            "vehicle" => &mut scene["vehicle"],
                            "wheel" => &mut scene["vehicle"]["wheels"][event["index"].as_u64().unwrap() as usize],
                            "surface" => &mut scene["vehicle"]["surfaces"][event["name"].as_str().unwrap()],
                            _ => unreachable!(),
                        };
                        for (key, value) in event["values"].as_object().unwrap() {
                            target[key] = value.clone();
                        }
                        w.vehicle = altd_sim::track::world::load_vehicle(&scene);
                    }
                    _ => panic!("unknown fixture event {event}"),
                }
            }
        }
        check_sensor_sample(&data, &w, &car, t, "requested");
        if let Some(requests) = data["requests"].as_array() {
            let request = &requests[t];
            assert_eq!(car.active, request["active"].as_bool().unwrap(), "requested active {t}");
            assert_eq!(car.frozen, request["frozen"].as_bool().unwrap(), "requested frozen {t}");
            assert_eq!(
                car.has_pending_request(),
                request["pending"].as_bool().unwrap(),
                "pending request {t}"
            );
            for (name, actual) in [
                ("position", car.actual_position()),
                ("velocity", car.actual_velocity()),
                ("acceleration", car.actual_acceleration()),
                ("display_velocity", car.display_velocity()),
            ] {
                assert_eq!(
                    vector_bits(actual),
                    vector_bits(vec(&request[name])),
                    "requested {name} {t}"
                );
            }
            assert_eq!(
                scalar_bits(car.actual_angular_velocity()),
                scalar_bits(f(&request["angular"])),
                "requested angular {t}"
            );
        }
        if data["deactivate_tick"].as_u64() == Some(t as u64) {
            car.deactivate();
        }
        let control = if let Some(c) = states[t]["controls"].as_array() {
            Controls {
                acceleration: c[0].as_f64().unwrap(),
                steering: c[1].as_f64().unwrap(),
                brake: c[2].as_f64().unwrap(),
                handbrake: c[3].as_f64().unwrap(),
                boost: c[4].as_f64().unwrap(),
            }
        } else {
            Controls {
                acceleration: if t < 120 { 1.0 } else { 0.0 },
                steering: if t < 30 {
                    0.0
                } else if t < 90 {
                    0.4
                } else {
                    -0.3
                },
                ..Controls::default()
            }
        };
        let reset_tick = data["reset"]["tick"].as_u64();
        if reset_tick == Some(t as u64) {
            car.queue_reset(
                vec(&data["reset"]["position"]),
                data["reset"]["rotation"].as_f64().unwrap(),
            );
        }
        if reset_tick
            .is_some_and(|start| t as u64 >= start && (t as u64) < data["reset"]["resume_tick"].as_u64().unwrap())
        {
            car.passive_step(&w, DT, eliminate);
        } else {
            car.step(&w, &control, DT, eliminate);
        }
        if let Some(frozen) = state["frozen"].as_bool() {
            assert_eq!(car.frozen, frozen, "frozen tick {i}");
        }
        if let Some(active) = state["active"].as_bool() {
            assert_eq!(car.active, active, "active tick {i}");
        }
        assert_eq!(
            vector_bits(car.position),
            vector_bits(vec(&state["position"])),
            "position tick {i}"
        );
        assert_eq!(
            vector_bits(car.velocity),
            vector_bits(vec(&state["velocity"])),
            "velocity tick {i}"
        );
        assert_eq!(
            scalar_bits(car.angular_velocity),
            scalar_bits(f(&state["angular_velocity"])),
            "angular tick {i}"
        );
        assert_eq!(
            vector_bits(car.body_basis.0.into()),
            vector_bits(vec(&state["body_basis"][0])),
            "basis X tick {i}"
        );
        assert_eq!(
            vector_bits(car.body_basis.1.into()),
            vector_bits(vec(&state["body_basis"][1])),
            "basis Y tick {i}"
        );
        assert_eq!(
            scalar_bits(car.boost_energy),
            scalar_bits(f(&state["boost_energy"])),
            "boost tick {i}"
        );
        if state["acceleration"].is_array() {
            assert_eq!(
                vector_bits(car.acceleration),
                vector_bits(vec(&state["acceleration"])),
                "acceleration tick {i}"
            );
        }
        for (j, wheel) in car.wheels.iter().enumerate() {
            assert_eq!(
                scalar_bits(wheel.angle_deg),
                scalar_bits(f(&state["wheel_angles"][j])),
                "wheel angle {j} tick {i}"
            );
            assert_eq!(
                wheel.previous_position.map(vector_bits),
                Some(vector_bits(vec(&state["wheel_positions"][j]))),
                "wheel history {j} tick {i}"
            );
        }
        if check_contacts {
            let expected = state["contacts"].as_array().unwrap();
            assert_eq!(car.wall_contacts.len(), expected.len(), "contact count tick {i}");
            for (c, e) in car.wall_contacts.iter().zip(expected) {
                assert_eq!(
                    vector_bits(c.normal),
                    vector_bits(vec(&e["normal"])),
                    "contact normal tick {i}"
                );
                assert_eq!(
                    vector_bits(c.point),
                    vector_bits(vec(&e["point"])),
                    "contact point tick {i}"
                );
            }
        }
    }
}
#[test]
fn standalone_car_matches_native_tilemap_contacts_and_reset() {
    replay(include_str!("fixtures/native_tilemap_single600.json"), true);
}
#[test]
fn free_drive_matches_every_captured_bit() {
    replay(include_str!("fixtures/native_free180.json"), false);
}
#[test]
fn persistent_wall_contacts_match_every_captured_bit() {
    replay(include_str!("fixtures/native_wall180.json"), true);
}
#[test]
fn eliminated_body_keeps_native_passive_physics() {
    replay(include_str!("fixtures/native_wall_eliminate180.json"), true);
}
#[test]
fn manual_deactivation_resets_velocity_on_native_callback() {
    replay(include_str!("fixtures/native_deactivate180.json"), false);
}
#[test]
fn reset_during_contact_matches_native_callback_order() {
    replay(include_str!("fixtures/native_wall_reset180.json"), true);
}
#[test]
fn boost_brakes_reverse_and_clamped_controls_match_native_vehicle() {
    replay(include_str!("fixtures/native_controls320.json"), false);
}
#[test]
fn velocity_limit_matches_native_impulse_order() {
    replay(include_str!("fixtures/native_highspeed180.json"), false);
}
#[test]
fn every_vehicle_type_matches_original_mixed_controls() {
    for fixture in [
        include_str!("fixtures/native_formula320.json"),
        include_str!("fixtures/native_snowmobile320.json"),
        include_str!("fixtures/native_truck320.json"),
    ] {
        replay(fixture, false);
    }
}
#[test]
fn offset_collision_shapes_and_centers_of_mass_match_native_contacts() {
    for fixture in [
        include_str!("fixtures/native_snowmobile_wall180.json"),
        include_str!("fixtures/native_truck_wall180.json"),
    ] {
        replay(fixture, true);
    }
}
#[test]
fn crossing_onto_dirt_matches_original_vehicle() {
    replay(include_str!("fixtures/native_dirt180.json"), false);
}
#[test]
fn ice_drive_matches_original_vehicle() {
    replay(include_str!("fixtures/native_ice180.json"), false);
}
#[test]
fn simultaneous_corner_contacts_match_original_engine() {
    replay(include_str!("fixtures/native_corner180.json"), true);
}
#[test]
fn waiting_preserves_wheels_while_native_basis_settles() {
    let data: Value = serde_json::from_str(include_str!("fixtures/native_warmup_a01.json")).unwrap();
    let world = World::from_scene(&data["scene"]);
    let position = vec(&data["states"][0]["position"]);
    let mut car = Car::new(&world, position, data["reset_rotation"].as_f64().unwrap());
    car.reset(&world, position, data["reset_rotation"].as_f64().unwrap(), true);
    // The first capture precedes application of the queued managed reset.
    for (i, s) in data["warmup"].as_array().unwrap().iter().enumerate().skip(1) {
        assert_eq!(car.position, vec(&s["position"]), "waiting position {i}");
        assert_eq!(
            V2::from(car.body_basis.0),
            vec(&s["body_basis"][0]),
            "waiting basis {i}"
        );
        assert_eq!(car.velocity, vec(&s["velocity"]), "waiting velocity {i}");
        for wheel in &car.wheels {
            assert_eq!(wheel.previous_position, Some(position));
        }
        car.passive_step(&world, DT, false);
    }
    assert_eq!(car.tick, 0);
    replay(include_str!("fixtures/native_warmup_a01.json"), false);
}

#[test]
fn broadphase_pair_creation_precedes_touching_and_preserves_order() {
    raw_replay(&[
        include_str!("fixtures/native_broadphase80.json"),
        include_str!("fixtures/native_broadphase_tie80.json"),
        include_str!("fixtures/native_broadphase_large80.json"),
    ]);
}
#[test]
fn native_contact_report_capacity_preserves_solver_and_slot_order() {
    raw_replay(&[
        include_str!("fixtures/native_contacts8.json"),
        include_str!("fixtures/native_contacts32.json"),
        include_str!("fixtures/native_contacts0.json"),
    ]);
}
#[test]
fn translated_contact_coordinates_preserve_native_rounding() {
    raw_replay(&[include_str!("fixtures/native_contacts_shift.json")]);
}
#[test]
fn distant_body_origins_preserve_native_report_coordinates() {
    raw_replay(&[include_str!("fixtures/native_contacts_origin.json")]);
}
#[test]
fn shallow_edge_contact_epsilon_matches_native_precision() {
    raw_replay(&[
        include_str!("fixtures/native_shallow_one.json"),
        include_str!("fixtures/native_shallow_two.json"),
    ]);
}
#[test]
fn restitution_uses_native_velocity_before_damping() {
    raw_replay(&[
        include_str!("fixtures/native_bounce_undamped.json"),
        include_str!("fixtures/native_bounce_damped.json"),
    ]);
}
#[test]
fn gravity_preserves_native_mass_and_inverse_mass_rounding() {
    raw_replay(&[include_str!("fixtures/native_gravity4.json")]);
}
#[test]
fn wall_material_friction_matches_native_smooth_and_rough_combination() {
    raw_replay(&[
        include_str!("fixtures/native_friction_smooth.json"),
        include_str!("fixtures/native_friction_rough.json"),
    ]);
}
#[test]
fn slanted_wall_with_rotating_offset_shape_matches_native_contacts() {
    raw_replay(&[
        include_str!("fixtures/native_slanted24.json"),
        include_str!("fixtures/native_slanted_corner24.json"),
    ]);
}
#[test]
fn near_parallel_native_contacts_match_all_captured_bits() {
    let cases: Vec<Value> = serde_json::from_str(include_str!("fixtures/native_sat32.json")).unwrap();
    for (i, case) in cases.iter().enumerate() {
        eprintln!("SAT case {i}");
        raw_replay(&[&serde_json::to_string(case).unwrap()]);
    }
}
fn raw_replay(fixtures: &[&str]) {
    for fixture in fixtures {
        let data: Value = serde_json::from_str(fixture).unwrap();
        let world = World::from_scene(&data["scene"]);
        let states = data["states"].as_array().unwrap();
        let mut car = Car::new(&world, vec(&states[0]["position"]), 0.0);
        car.set_body_basis(
            F2::from(vec(&states[0]["basis"][0])),
            F2::from(vec(&states[0]["basis"][1])),
        );
        car.angular_velocity = f(&states[0]["angular"]);
        for _ in 0..data["warmup"].as_u64().unwrap_or(8) {
            car.passive_step(&world, DT, false);
        }
        car.velocity = vec(&states[0]["velocity"]);
        for (i, s) in states.iter().enumerate().skip(1) {
            car.step(&world, &Controls::default(), DT, false);
            assert_eq!(
                vector_bits(car.position),
                vector_bits(vec(&s["position"])),
                "position {i}"
            );
            assert_eq!(
                vector_bits(car.velocity),
                vector_bits(vec(&s["velocity"])),
                "velocity {i}"
            );
            assert_eq!(
                (car.angular_velocity as f32).to_bits(),
                (f(&s["angular"]) as f32).to_bits(),
                "angular {i}"
            );
            assert_eq!(
                vector_bits(car.body_basis.0.into()),
                vector_bits(vec(&s["basis"][0])),
                "basis X {i}"
            );
            assert_eq!(
                vector_bits(car.body_basis.1.into()),
                vector_bits(vec(&s["basis"][1])),
                "basis Y {i}"
            );
            let contacts = s["contacts"].as_array().unwrap();
            assert_eq!(car.wall_contacts.len(), contacts.len(), "contacts {i}");
            for (c, e) in car.wall_contacts.iter().zip(contacts) {
                assert_eq!(vector_bits(c.normal), vector_bits(vec(&e["normal"])), "normal {i}");
                assert_eq!(vector_bits(c.point), vector_bits(vec(&e["point"])), "point {i}");
                if e["other"].is_array() {
                    assert_eq!(
                        vector_bits(c.wall_point),
                        vector_bits(vec(&e["other"])),
                        "wall point {i}"
                    );
                }
            }
        }
    }
}
#[test]
fn network_uses_game_order_and_windows_activation() {
    let n = altd_sim::nn::network::Network::from_vector(&[3, 1], vec![1e16, 1.0, -1e16, 0.0]);
    assert_eq!(n.forward(&[1.0, 1.0, 1.0]), vec![0.0]);
    let pairs: Vec<[f64; 2]> = serde_json::from_str(include_str!("fixtures/windows_tanh.json")).unwrap();
    for [x, y] in pairs {
        assert_eq!(altd_sim::nn::network::game_tanh(x).to_bits(), y.to_bits(), "tanh({x})");
    }
}

#[test]
fn native_curve_baking_and_samples_match_shipped_engine() {
    let data: Value = serde_json::from_str(include_str!("fixtures/native_curve_a07.json")).unwrap();
    let curve = altd_sim::track::curve::Curve::from_json(&data["curve"]);
    let baked = data["curve"]["baked"].as_array().unwrap();
    assert_eq!(curve.points.len(), baked.len());
    assert_eq!(curve.length() as f64, data["curve"]["length"].as_f64().unwrap());
    for (a, e) in curve.points.iter().zip(baked) {
        let e = altd_sim::track::curve::json_vector(e);
        assert_eq!((a.x.to_bits(), a.y.to_bits()), (e.x.to_bits(), e.y.to_bits()));
    }
    for s in data["samples"].as_array().unwrap() {
        let offset = curve.closest_offset(altd_sim::track::curve::json_vector(&s["query"]));
        assert_eq!(offset as f64, s["offset"].as_f64().unwrap());
        for (key, offset) in [
            ("forward", offset),
            ("ahead", s["aheadOffset"].as_f64().unwrap() as f32),
        ] {
            let a = curve.direction(offset).normalized();
            let e = altd_sim::track::curve::json_vector(&s[key]);
            assert_eq!((a.x.to_bits(), a.y.to_bits()), (e.x.to_bits(), e.y.to_bits()));
        }
    }
}

#[test]
fn tiny_handbrake_power_uses_game_zero_tolerance() {
    for data in [
        include_str!("fixtures/native_handbrake_below16.json"),
        include_str!("fixtures/native_handbrake_threshold16.json"),
    ] {
        replay(data, false);
    }
}

#[test]
fn freezing_preserves_native_physics_and_restores_callback_velocity() {
    for data in [
        include_str!("fixtures/native_freeze32.json"),
        include_str!("fixtures/native_freeze_gravity32.json"),
        include_str!("fixtures/native_freeze_builtin32.json"),
        include_str!("fixtures/native_freeze_edges32.json"),
    ] {
        replay(data, false);
    }
}

#[test]
fn runtime_properties_preserve_wheel_initialization_snapshots() {
    replay(include_str!("fixtures/native_mutation32.json"), false);
}

#[test]
fn native_sensor_bits_cover_pending_requests_and_property_changes() {
    for fixture in [
        include_str!("fixtures/native_freeze_sensors32.json"),
        include_str!("fixtures/native_mutation_sensors32.json"),
    ] {
        replay(fixture, false);
    }
}

#[test]
fn track_sensor_bits_follow_deferred_resets() {
    replay(include_str!("fixtures/native_path_sensors32.json"), true);
}

#[test]
fn distance_sensor_preserves_empty_raycasters_and_float_boundaries() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/native_free180.json")).unwrap();
    let cases: Value = serde_json::from_str(include_str!("fixtures/native_distance_sensors.json")).unwrap();
    for (i, case) in cases.as_array().unwrap().iter().enumerate() {
        let mut scene = fixture["scene"].clone();
        scene["track"] = serde_json::json!({"tile_size":[768,768],"tiles":[],"polygons":[{"points":case["points"]}],"physics_shapes":[],"raycaster_present":case["present"]});
        let world = World::from_scene(&scene);
        let mut car = Car::new(&world, V2::ZERO, 0.0);
        car.queue_reset(vec(&case["position"]), 0.0);
        let value = car.sensor(
            &world,
            Sensor::DistanceFromWall {
                max_distance: case["maximum"].as_f64().unwrap(),
            },
            &mut SensorScratch::default(),
        );
        assert_eq!(
            scalar_bits(value),
            case["value_bits"].as_u64().unwrap() as u32,
            "distance sensor case {i}: {case}"
        );
    }
}

#[test]
fn generated_world_matches_original_tilemap_drive_and_reset() {
    replay(include_str!("fixtures/native_generated_case2_180.json"), true);
}

#[test]
fn ordered_sensor_parameters_and_duplicates_match_game() {
    replay(include_str!("fixtures/native_ordered_sensors32.json"), true);
}
