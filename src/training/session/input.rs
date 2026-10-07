//! Validate embedding-host scene JSON before legacy, trusted-data constructors.
use serde_json::Value;

fn number(v: &Value, label: &str) -> Result<f64, String> {
    v.as_f64()
        .filter(|n| n.is_finite() && (*n as f32).is_finite())
        .ok_or_else(|| format!("invalid number: {label}"))
}
fn vector(v: &Value, label: &str) -> Result<(), String> {
    let pair = v
        .as_array()
        .filter(|a| a.len() == 2)
        .ok_or_else(|| format!("invalid vector: {label}"))?;
    for n in pair {
        number(n, label)?;
    }
    Ok(())
}
fn curve_vector(v: &Value, label: &str) -> Result<(), String> {
    if v.is_array() {
        vector(v, label)
    } else {
        number(&v["x"], label)?;
        number(&v["y"], label)?;
        Ok(())
    }
}
fn optional_number(v: &Value, key: &str) -> Result<(), String> {
    if let Some(n) = v.get(key) {
        number(n, key)?;
    }
    Ok(())
}
fn optional_vector(v: &Value, key: &str) -> Result<(), String> {
    if let Some(p) = v.get(key) {
        vector(p, key)?;
    }
    Ok(())
}
fn array<'a>(v: &'a Value, key: &str) -> Result<&'a [Value], String> {
    match v.get(key) {
        None => Ok(&[]),
        Some(a) => a
            .as_array()
            .map(Vec::as_slice)
            .ok_or_else(|| format!("invalid array: {key}")),
    }
}

pub(super) fn scene(v: &Value) -> Result<(), String> {
    let car = v.get("vehicle").filter(|c| c.is_object()).ok_or("missing vehicle")?;
    let track = v.get("track").filter(|c| c.is_object()).ok_or("missing track")?;
    vector(&car["shape_size"], "shape_size")?;
    if car["shape_size"]
        .as_array()
        .unwrap()
        .iter()
        .any(|x| x.as_f64().unwrap() <= 0.0)
    {
        return Err("shape_size must be positive".into());
    }
    for key in [
        "shape_rotation",
        "mass",
        "inertia_server",
        "grip",
        "steering_speed",
        "air_resistance",
        "max_velocity",
        "linear_damp",
        "angular_damp",
    ] {
        number(&car[key], key)?;
    }
    if number(&car["mass"], "mass")? <= 0.0 || number(&car["inertia_server"], "inertia_server")? <= 0.0 {
        return Err("mass and inertia must be positive".into());
    }
    if car["center_steering"].as_bool().is_none() {
        return Err("invalid center_steering".into());
    }
    for key in ["shape_position", "center_of_mass"] {
        optional_vector(car, key)?;
    }
    if let Some(basis) = car.get("shape_basis") {
        let rows = basis.as_array().filter(|a| a.len() == 2).ok_or("invalid shape_basis")?;
        for row in rows {
            vector(row, "shape_basis")?;
        }
    }
    let wheels = car["wheels"]
        .as_array()
        .filter(|a| !a.is_empty())
        .ok_or("missing wheels")?;
    for wheel in wheels {
        vector(&wheel["position"], "wheel position")?;
        if wheel["steering"].as_bool().is_none() {
            return Err("invalid wheel steering".into());
        }
        for key in ["max_angle_deg", "power", "brake_power", "handbrake_power"] {
            number(&wheel[key], key)?;
        }
    }
    let surfaces = car["surfaces"].as_object().ok_or("missing vehicle surfaces")?;
    for surface in surfaces.values() {
        for key in ["grip", "power", "steering"] {
            optional_number(surface, key)?;
        }
    }
    // These fields have numeric/default semantics in the trusted constructor.
    for key in ["friction", "max_contacts_reported", "gravity_scale", "bounce"] {
        if let Some(n) = car.get(key).filter(|n| !n.is_null()) {
            number(n, key)?;
        }
    }
    if let Some(physics) = v.get("physics") {
        for key in [
            "solver_iterations",
            "contact_default_bias",
            "contact_recycle_radius",
            "contact_max_separation",
            "contact_max_allowed_penetration",
        ] {
            optional_number(physics, key)?;
        }
        optional_vector(physics, "gravity")?;
    }
    optional_vector(v, "reset_position")?;
    optional_number(v, "reset_rotation")?;
    for key in ["tile_map_position", "tile_size"] {
        optional_vector(track, key)?;
    }
    if let Some(size) = track.get("tile_size") {
        if size.as_array().unwrap().iter().any(|x| x.as_f64().unwrap() <= 0.0) {
            return Err("tile_size must be positive".into());
        }
    }
    for wall in array(track, "walls")? {
        let points = wall.as_array().filter(|a| a.len() == 2).ok_or("invalid wall")?;
        for point in points {
            vector(point, "wall")?;
        }
    }
    for shape in array(track, "physics_shapes")? {
        for key in ["tile", "origin"] {
            optional_vector(shape, key)?;
        }
        let points = shape["points"]
            .as_array()
            .filter(|a| a.len() >= 3)
            .ok_or("invalid shape points")?;
        for p in points {
            vector(p, "shape points")?;
        }
        if let Some(local) = shape.get("local_points") {
            let rows = local
                .as_array()
                .filter(|a| a.len() == points.len())
                .ok_or("invalid local_points")?;
            for p in rows {
                curve_vector(p, "local_points")?;
            }
        }
    }
    for polygon in array(track, "polygons")? {
        let points = polygon["points"].as_array().ok_or("missing polygon points")?;
        for p in points {
            vector(p, "polygon point")?;
        }
    }
    for key in ["path", "path_forward", "native_cell_order"] {
        for p in array(track, key)? {
            vector(p, key)?;
        }
    }
    if array(track, "path")?.len() < 2 {
        return Err("track path needs at least two points".into());
    }
    let forward = array(track, "path_forward")?;
    if !forward.is_empty() && forward.len() != array(track, "path")?.len() {
        return Err("path_forward and path lengths differ".into());
    }
    for tile in array(track, "tiles")? {
        vector(&tile["coords"], "tile coords")?;
        if tile["surface"].as_str().is_none() {
            return Err("invalid tile surface".into());
        }
        for p in array(tile, "connections")? {
            vector(p, "tile connection")?;
        }
    }
    if let Some(curve) = track.get("curve") {
        let interval = number(&curve["bake_interval"], "bake_interval")?;
        if interval <= 0.0 || !(interval as f32).is_finite() {
            return Err("invalid bake_interval".into());
        }
        let controls = curve["points"]
            .as_array()
            .filter(|a| a.len() >= 2)
            .ok_or("curve needs at least two control points")?;
        for control in controls {
            for key in ["position", "incoming", "outgoing"] {
                curve_vector(&control[key], key)?;
            }
        }
    }
    Ok(())
}
