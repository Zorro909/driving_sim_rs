use altd_sim::{math::godot_math::F2, math::native_math, math::vec2::V2, physics::car::Car, track::world::World};
use serde_json::{json, Value};
fn vec(v: &Value) -> V2 {
    if let Some(bits) = v["bits"].as_array() {
        return V2::new(
            f32::from_bits(bits[0].as_u64().unwrap() as u32) as f64,
            f32::from_bits(bits[1].as_u64().unwrap() as u32) as f64,
        );
    }
    V2::new(v["x"].as_f64().unwrap(), v["y"].as_f64().unwrap())
}
pub(crate) fn restore(w: &World, s: &Value) -> Car {
    let t = &s["physical"];
    let x = F2::from(vec(&t["x"]));
    let y = F2::from(vec(&t["y"]));
    let mut c = Car::new(w, vec(&t["origin"]), native_math::atan2(x.y, x.x) as f64);
    c.active = s["active"].as_bool().unwrap_or(true);
    c.set_body_basis(x, y);
    c.velocity = vec(&s["velocity"]);
    c.angular_velocity = s["angular"].as_f64().unwrap();
    c.boost_energy = s["boost"].as_f64().unwrap();
    for (wheel, sw) in c.wheels.iter_mut().zip(s["wheels"].as_array().unwrap()) {
        wheel.angle_deg = sw["_currentRotationDegrees"].as_f64().unwrap() as f32 as f64;
        wheel.previous_position = Some(vec(&sw["_lastPosition"]));
    }
    c
}
pub(crate) fn differences(c: &Car, s: &Value) -> Value {
    let t = &s["physical"];
    let mut bad = serde_json::Map::new();
    if let Some(active) = s["active"].as_bool() {
        if c.active != active {
            bad.insert("active".into(), json!([c.active, active]));
        }
    }
    for (name, a, e) in [
        ("position.x", c.position.x, vec(&t["origin"]).x),
        ("position.y", c.position.y, vec(&t["origin"]).y),
        ("velocity.x", c.velocity.x, vec(&s["velocity"]).x),
        ("velocity.y", c.velocity.y, vec(&s["velocity"]).y),
        ("angular", c.angular_velocity, s["angular"].as_f64().unwrap()),
        ("basis.x", c.body_basis.0.x as f64, vec(&t["x"]).x),
        ("basis.y", c.body_basis.0.y as f64, vec(&t["x"]).y),
    ] {
        if (a as f32).to_bits() != (e as f32).to_bits() {
            bad.insert(name.into(), json!([a, e]));
        }
    }
    for (name, a, e) in [
        ("basis_y.x", c.body_basis.1.x as f64, vec(&t["y"]).x),
        ("basis_y.y", c.body_basis.1.y as f64, vec(&t["y"]).y),
        ("boost", c.boost_energy, s["boost"].as_f64().unwrap()),
        ("acceleration.x", c.acceleration.x, vec(&s["acceleration"]).x),
        ("acceleration.y", c.acceleration.y, vec(&s["acceleration"]).y),
    ] {
        if (a as f32).to_bits() != (e as f32).to_bits() {
            bad.insert(name.into(), json!([a, e]));
        }
    }
    for (i, (wheel, sw)) in c.wheels.iter().zip(s["wheels"].as_array().unwrap()).enumerate() {
        let a = wheel.angle_deg as f32;
        let e = sw["_currentRotationDegrees"].as_f64().unwrap() as f32;
        if a.to_bits() != e.to_bits() {
            bad.insert(format!("wheel{i}"), json!([a as f64, e as f64]));
        }
        let e = vec(&sw["_lastPosition"]);
        let a = wheel.previous_position.unwrap_or_default();
        if (a.x as f32).to_bits() != (e.x as f32).to_bits() || (a.y as f32).to_bits() != (e.y as f32).to_bits() {
            bad.insert(format!("wheel{i}.history"), json!([[a.x, a.y], [e.x, e.y]]));
        }
    }
    if let Some(contacts) = s["contacts"].as_array() {
        if c.wall_contacts.len() != contacts.len() {
            bad.insert("contact_count".into(), json!([c.wall_contacts.len(), contacts.len()]));
        }
        for (i, (a, e)) in c.wall_contacts.iter().zip(contacts).enumerate() {
            for (key, a, e) in [
                ("normal", a.normal, vec(&e["normal"])),
                ("point", a.point, vec(&e["point"])),
            ] {
                if (a.x as f32).to_bits() != (e.x as f32).to_bits() || (a.y as f32).to_bits() != (e.y as f32).to_bits()
                {
                    bad.insert(format!("contact{i}.{key}"), json!([[a.x, a.y], [e.x, e.y]]));
                }
            }
        }
    }
    Value::Object(bad)
}
