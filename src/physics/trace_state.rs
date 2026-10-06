//! Restore exported float32 state and, when justified, its integration angle.
use crate::{
    math::godot_math::{self, F2},
    math::vec2::V2,
    physics::car::{Car, DT},
    track::world::World,
};
use serde_json::Value;
fn f(v: &Value) -> f64 {
    v.as_f64().expect("trace number") as f32 as f64
}
fn v(v: &Value) -> V2 {
    V2::new(f(&v[0]), f(&v[1]))
}

pub(crate) fn transform_angle(world: &World, state: &Value, previous: Option<&Value>) -> f64 {
    let recorded = f(&state["rotation"]);
    let Some(previous) = previous else {
        return recorded;
    };
    let Some(contacts) = state.get("physics_contacts").and_then(Value::as_array) else {
        return recorded;
    };
    // Without penetration bias, integrate_velocities used the preceding angle
    // and this frame's angular velocity. Contact points are world-space floats;
    // allow two coordinate ULPs when deciding whether depth is below the limit.
    let no_bias = contacts.iter().all(|c| {
        let a = v(&c["local_position"]);
        let b = v(&c["collider_position"]);
        let scale = a.x.abs().max(a.y.abs()).max(b.x.abs()).max(b.y.abs()) as f32;
        let ulp = f32::from_bits(scale.to_bits() + 1) - scale;
        (a - b).length() + 2.0 * (ulp as f64) < world.vehicle.allowed_penetration
    });
    if !no_bias {
        return recorded;
    }
    let angle = f(&previous["rotation"]) as f32 + f(&state["angular_velocity"]) as f32 * DT as f32;
    let extracted = world.math.atan2(
        crate::math::native_math::engine_sin(angle),
        crate::math::native_math::engine_cos(angle),
    );
    // Reject resets/discontinuities. Two float32 ULPs account for angle
    // extraction by the native engine versus the managed trace exporter.
    let ulp = f32::from_bits((recorded as f32).abs().to_bits() + 1) - (recorded as f32).abs();
    let error = (extracted as f64 - recorded)
        .sin()
        .atan2((extracted as f64 - recorded).cos())
        .abs();
    if error <= 2.0 * ulp as f64 {
        angle as f64
    } else {
        recorded
    }
}

pub(crate) fn body_basis(world: &World, state: &Value, previous: Option<&Value>) -> (F2, F2) {
    if let Some(basis) = state.get("body_basis") {
        return (v(&basis[0]).into(), v(&basis[1]).into());
    }
    if state.get("reset_transform").and_then(Value::as_bool) == Some(true) {
        return godot_math::reset_basis(f(state.get("reset_rotation").unwrap_or(&state["rotation"])) as f32);
    }
    godot_math::basis(transform_angle(world, state, previous))
}

pub fn car_from_frame(world: &World, state: &Value, previous: Option<&Value>, older: Option<&Value>) -> Car {
    let mut car = Car::new(world, v(&state["position"]), f(&state["rotation"]));
    car.set_transform_angle(transform_angle(world, state, previous));
    let (x, y) = body_basis(world, state, previous);
    car.body_basis = (x, y);
    car.velocity = v(&state["velocity"]);
    car.angular_velocity = f(&state["angular_velocity"]);
    car.boost_energy = f(&state["boost_energy"]);
    for (i, (spec, wheel)) in world.vehicle.wheels.iter().zip(&mut car.wheels).enumerate() {
        wheel.angle_deg = f(&state["wheel_angles"][i]);
        if let Some(previous) = previous {
            let position = godot_math::transform_point(
                v(&previous["position"]),
                body_basis(world, previous, older),
                spec.position,
            );
            wheel.previous_position = Some(position);
            wheel.previous_surface = world.track.surface(position);
        }
    }
    car
}
