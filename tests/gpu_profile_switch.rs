//! Ray endpoints must follow the current world's math profile after reuse.
use altd_sim::{
    gpu::{
        hip::{Gpu, GpuWorld},
        simulation::{sensor_desc, vehicle_desc, GpuSim, ReadMask},
    },
    math::{profile::MathProfile, vec2::V2},
    physics::car::{Car, Sensor, SensorScratch},
    track::world::World,
};

#[test]
#[ignore = "requires a HIP GPU and gpu/build.sh"]
fn reused_ray_sensors_follow_the_replacement_world_profile() {
    let gpu = Gpu::open(None).unwrap();
    let mut scene: serde_json::Value =
        serde_json::from_str(include_str!("../assets/scenes/rally_template.json")).unwrap();
    scene["track"] = serde_json::json!({
        "polygons": [{"points": [[1.0, 0.0], [2.0, 0.0], [2.0, 900.0], [1.0, 900.0]]}],
        "path": [[0.0, 0.0], [0.0, 100.0]]
    });
    // At this angle, Windows' and Proton's cosine differ by one float32 ULP.
    let sensors = [
        Sensor::Raycast {
            degrees: f32::from_bits(0x4333_dab6) as f64,
            length: 1000.0,
        },
        Sensor::Speed,
        Sensor::Grip {
            offset: V2::new(3.0, 7.0),
        },
    ];
    let descriptions: Vec<_> = sensors.iter().map(sensor_desc).collect();
    for initial_profile in MathProfile::ALL {
        let initial_world = World::from_scene_with(&scene, initial_profile);
        let initial_gpu_world = GpuWorld::new(&gpu, &initial_world);
        let vehicle = vehicle_desc(&initial_world.vehicle).unwrap();
        let mut reused = GpuSim::new(&initial_gpu_world, &vehicle, &descriptions, 1);
        for next_profile in MathProfile::ALL {
            let next_world = World::from_scene_with(&scene, next_profile);
            let next_gpu_world = GpuWorld::new(&gpu, &next_world);
            let mut car = Car::new(&next_world, V2::ZERO, 0.0);
            car.velocity = V2::new(40.0, 10.0);
            let state = car
                .gpu_export(&next_world.vehicle, &next_gpu_world.arrays.surfaces)
                .unwrap();
            reused.set_world(&next_gpu_world);
            reused.upload(&[state], None);
            let values = reused.sensors(ReadMask::ALL);
            let mut scratch = SensorScratch::default();
            for (sensor, value) in sensors.iter().zip(values) {
                assert_eq!(
                    value.to_bits(),
                    car.sensor(&next_world, *sensor, &mut scratch).to_bits(),
                    "{initial_profile} -> {next_profile}: {sensor:?}"
                );
            }
            // Restore a live handle before next_gpu_world is released.
            reused.set_world(&initial_gpu_world);
        }
    }
}
