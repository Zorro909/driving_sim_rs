//! Validate embedding-host scene JSON before legacy, trusted-data constructors.
use crate::math::vec2::V2;
use crate::track::world::Surface;
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

fn cell_coords(v: &Value, label: &str) -> Result<[i64; 2], String> {
    vector(v, label)?;
    let read = |v: &Value| {
        let n = number(v, label)?;
        // Match the constructor's truncation without allowing saturating casts.
        if n <= i64::MIN as f64 || n >= i64::MAX as f64 {
            return Err(format!("{label} exceeds signed integer range"));
        }
        Ok(n as i64)
    };
    Ok([read(&v[0])?, read(&v[1])?])
}

fn tile_capacity(lo: [i64; 2], hi: [i64; 2]) -> Result<(), String> {
    let span = |i: usize| hi[i].checked_sub(lo[i]).and_then(|n| n.checked_add(1));
    let cells = span(0)
        .and_then(|nx| span(1).and_then(|ny| nx.checked_mul(ny)))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or("tile table dimensions overflow")?;
    // Each actual dense table must fit Vec's signed byte-offset capacity.
    // This checks numerical allocation capacity, not a work quota.
    if [
        std::alloc::Layout::array::<Option<Surface>>(cells),
        std::alloc::Layout::array::<Option<String>>(cells),
        std::alloc::Layout::array::<Option<Option<(V2, V2)>>>(cells),
    ]
    .iter()
    .any(Result::is_err)
    {
        return Err("tile table exceeds allocation capacity".into());
    }
    Ok(())
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
        optional_vector(shape, "origin")?;
        if let Some(tile) = shape.get("tile") {
            cell_coords(tile, "shape tile")?;
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
    for key in ["path", "path_forward"] {
        for p in array(track, key)? {
            vector(p, key)?;
        }
    }
    for cell in array(track, "native_cell_order")? {
        cell_coords(cell, "native_cell_order")?;
    }
    if array(track, "path")?.len() < 2 {
        return Err("track path needs at least two points".into());
    }
    let forward = array(track, "path_forward")?;
    if !forward.is_empty() && forward.len() != array(track, "path")?.len() {
        return Err("path_forward and path lengths differ".into());
    }
    let tiles = array(track, "tiles")?;
    let mut lo = [i64::MAX; 2];
    let mut hi = [i64::MIN; 2];
    for tile in tiles {
        let coords = cell_coords(&tile["coords"], "tile coords")?;
        for i in 0..2 {
            lo[i] = lo[i].min(coords[i]);
            hi[i] = hi[i].max(coords[i]);
        }
        if tile["surface"].as_str().is_none() {
            return Err("invalid tile surface".into());
        }
        for p in array(tile, "connections")? {
            vector(p, "tile connection")?;
        }
    }
    if !tiles.is_empty() {
        tile_capacity(lo, hi)?;
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
