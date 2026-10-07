//! Server admission policy. CLI/operator-selected workloads remain unrestricted.
use serde_json::Value;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

pub(super) const MAX_JSON: usize = 16 * 1024 * 1024;
pub(super) const MAX_CHECKPOINT: usize = 32 * 1024 * 1024;
const MAX_POPULATION: usize = 32_768;
const MAX_PARAMETERS: usize = 16_384;
const MAX_TOTAL_PARAMETERS: usize = 1_048_576;
const MAX_SESSION: usize = 512 * 1024 * 1024;
const MAX_RESERVED: usize = 1024 * 1024 * 1024;
const MAX_TICKS: u64 = 3600;
const MAX_WORK: u64 = 500_000_000;
const MAX_GRID_ITEMS: usize = 16_777_216;

#[derive(Default)]
pub(super) struct Pool {
    reserved: AtomicUsize,
    pub work: Mutex<()>,
}

pub(super) struct Lease {
    pool: Arc<Pool>,
    bytes: usize,
}
impl Lease {
    pub fn new(pool: Arc<Pool>, plan: Plan) -> Result<Self, String> {
        let bytes = plan.bytes()?;
        pool.reserved
            .fetch_update(Ordering::AcqRel, Ordering::Relaxed, |n| {
                n.checked_add(bytes).filter(|&total| total <= MAX_RESERVED)
            })
            .map_err(|_| "server session resource budget exhausted")?;
        Ok(Self { pool, bytes })
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        self.pool.reserved.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

#[derive(Clone, Copy, Default)]
pub(super) struct Plan {
    pub population: usize,
    pub parameters: usize,
    pub grid_items: usize,
    pub solver: usize,
}
impl Plan {
    pub fn bytes(self) -> Result<usize, String> {
        if self.population == 0 || self.population > MAX_POPULATION {
            return Err("population exceeds server limit 32768".into());
        }
        let params = self
            .population
            .checked_mul(self.parameters)
            .filter(|&n| n <= MAX_TOTAL_PARAMETERS)
            .ok_or("population/network parameter budget exceeded")?;
        // Conservative admission units for population snapshots, reproduction,
        // scratch state and spatial tables. This is not an allocator/RSS limit.
        let bytes = 32 * 1024 * 1024 + params * 64 + self.population * 8192 + self.grid_items * 8;
        if bytes > MAX_SESSION {
            return Err("session admission budget exceeds 512 MiB".into());
        }
        Ok(bytes)
    }
    pub fn ticks(self, ticks: u64) -> Result<(), String> {
        let work = (self.population as u64)
            .checked_mul(ticks)
            .and_then(|n| n.checked_mul((self.parameters + 32 * self.solver + 1) as u64));
        if ticks > MAX_TICKS || work.is_none_or(|n| n > MAX_WORK) {
            return Err("simulation window exceeds server work budget; use shorter advance windows".into());
        }
        Ok(())
    }
}

pub(super) fn parse(text: &str, what: &str) -> Result<Value, String> {
    if text.len() > MAX_JSON {
        return Err(format!("{what} JSON exceeds 16 MiB server limit"));
    }
    serde_json::from_str(text).map_err(|e| format!("invalid {what} JSON: {e}"))
}

pub(super) fn shape(v: &Value) -> Result<usize, String> {
    let rows = v.as_array().ok_or("missing shape")?;
    let numbers = rows
        .iter()
        .map(|n| {
            n.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .map(|n| n as usize)
                .ok_or("shape entries must be 32-bit non-negative integers")
        })
        .collect::<Result<Vec<_>, _>>()?;
    shape_numbers(&numbers)
}
pub(super) fn shape_numbers(shape: &[usize]) -> Result<usize, String> {
    if !(2..=20).contains(&shape.len()) || shape.iter().any(|&n| n == 0 || n > 64) {
        return Err("server networks need 2..20 layers of 1..64 nodes".into());
    }
    let count = shape
        .windows(2)
        .try_fold(0usize, |n, w| n.checked_add((w[0] + 1) * w[1]))
        .filter(|&n| n <= MAX_PARAMETERS)
        .ok_or("network exceeds server parameter limit 16384")?;
    Ok(count)
}

pub(super) fn settings(v: &Value, default_population: usize) -> Result<usize, String> {
    let v = v.get("settings").unwrap_or(v);
    let integer = |key: &str, default: usize, max: usize| -> Result<usize, String> {
        match v.get(key) {
            None => Ok(default),
            Some(n) => n
                .as_f64()
                .filter(|n| n.is_finite() && *n >= 0.0 && n.fract() == 0.0 && *n <= max as f64)
                .map(|n| n as usize)
                .ok_or_else(|| format!("{key} exceeds server limit {max}")),
        }
    };
    for key in ["selection_size", "preserve_parents_size"] {
        integer(key, 3, MAX_POPULATION)?;
    }
    if v.get("rewards").and_then(Value::as_array).is_some_and(|r| r.len() > 13) {
        return Err("too many server reward terms".into());
    }
    let population = integer("population", default_population, MAX_POPULATION)?;
    if population == 0 {
        return Err("population must be positive".into());
    }
    Ok(population)
}

fn array<'a>(v: &'a Value, key: &str, max: usize) -> Result<&'a [Value], String> {
    match v.get(key) {
        None => Ok(&[]),
        Some(v) => v
            .as_array()
            .filter(|a| a.len() <= max)
            .map(Vec::as_slice)
            .ok_or_else(|| format!("{key} exceeds server array limit {max}")),
    }
}
fn vector(v: &Value) -> Result<[f64; 2], String> {
    let read = |n: &Value| {
        n.as_f64()
            .filter(|n| n.is_finite() && n.abs() <= 1_000_000.0)
            .ok_or("geometry coordinate exceeds server bounds")
    };
    if v.is_array() {
        Ok([read(&v[0])?, read(&v[1])?])
    } else {
        Ok([read(&v["x"])?, read(&v["y"])?])
    }
}
fn box_items(a: [f64; 2], b: [f64; 2], cell: f64) -> usize {
    // Extra cells cover alignment and edge rounding. All coordinates have
    // already passed the finite magnitude check, so these casts cannot saturate.
    ((a[0] - b[0]).abs() / cell + 3.0).ceil() as usize * ((a[1] - b[1]).abs() / cell + 3.0).ceil() as usize
}
pub(super) fn scene(v: &Value) -> Result<(usize, usize), String> {
    let t = &v["track"];
    let path = array(t, "path", 16384)?;
    let walls = array(t, "walls", 2048)?;
    for key in ["path_forward", "native_cell_order"] {
        for p in array(t, key, 16384)? {
            vector(p)?;
        }
    }
    let shapes = array(t, "physics_shapes", 512)?;
    let tiles = array(t, "tiles", 4096)?;
    let polygons = array(t, "polygons", 512)?;
    let mut polygon_points = 0;
    for polygon in polygons {
        polygon_points += array(polygon, "points", 512)?.len();
    }
    if polygon_points > 2048 {
        return Err("BSP polygons exceed server segment budget 2048".into());
    }
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    let mut include = |p: [f64; 2]| {
        for i in 0..2 {
            lo[i] = lo[i].min(p[i]);
            hi[i] = hi[i].max(p[i]);
        }
    };
    let path = path.iter().map(vector).collect::<Result<Vec<_>, _>>()?;
    for &p in &path {
        include(p);
    }
    let mut items = 0usize;
    for pair in path.windows(2) {
        items = items.saturating_add(box_items(pair[0], pair[1], 64.0));
    }
    for wall in walls {
        let (a, b) = (vector(&wall[0])?, vector(&wall[1])?);
        include(a);
        include(b);
        items = items.saturating_add(box_items(a, b, 128.0));
    }
    let origin = t.get("tile_map_position").map(vector).transpose()?.unwrap_or([0.0; 2]);
    for s in shapes {
        let points = array(s, "points", 64)?;
        for p in array(s, "local_points", 64)? {
            vector(p)?;
        }
        let mut a = [f64::INFINITY; 2];
        let mut b = [f64::NEG_INFINITY; 2];
        for p in points {
            let mut p = vector(p)?;
            for i in 0..2 {
                p[i] += origin[i];
                a[i] = a[i].min(p[i]);
                b[i] = b[i].max(p[i]);
            }
            include(p);
        }
        items = items.saturating_add(box_items(a, b, 128.0));
    }
    for p in polygons {
        for p in array(p, "points", 512)? {
            vector(p)?;
        }
    }
    if let Some(curve) = t.get("curve") {
        let controls = array(curve, "points", 64)?;
        for pair in controls.windows(2) {
            let a = vector(&pair[0]["position"])?;
            let d = vector(&pair[1]["position"])?;
            let out = vector(&pair[0]["outgoing"])?;
            let inc = vector(&pair[1]["incoming"])?;
            let controls = [a, [a[0] + out[0], a[1] + out[1]], [d[0] + inc[0], d[1] + inc[1]], d];
            let mut min = a;
            let mut max = a;
            for p in controls {
                for i in 0..2 {
                    min[i] = min[i].min(p[i]);
                    max[i] = max[i].max(p[i]);
                }
            }
            // Convex hull bounds all baked points; subdivision is capped at 10.
            items = items.saturating_add(box_items(min, max, 64.0).saturating_mul(1025));
        }
    }
    if !tiles.is_empty() {
        let mut min = [f64::INFINITY; 2];
        let mut max = [f64::NEG_INFINITY; 2];
        for tile in tiles {
            let p = vector(&tile["coords"])?;
            for i in 0..2 {
                min[i] = min[i].min(p[i]);
                max[i] = max[i].max(p[i]);
            }
        }
        if box_items(min, max, 1.0) > 65536 {
            return Err("tile table exceeds server cell budget".into());
        }
    }
    let cells = if lo[0].is_finite() {
        box_items([lo[0] - 768.0, lo[1] - 768.0], [hi[0] + 768.0, hi[1] + 768.0], 32.0)
    } else {
        0
    };
    if cells > 262144 {
        return Err("geometry spans too many server grid cells".into());
    }
    if polygons.is_empty() && t.get("raycaster_present").is_none() {
        items = items.saturating_add(cells.saturating_mul(walls.len()));
    }
    if items > MAX_GRID_ITEMS {
        return Err("geometry exceeds server grid-entry budget".into());
    }
    let solver = v["physics"]
        .get("solver_iterations")
        .and_then(Value::as_f64)
        .unwrap_or(16.0);
    if !solver.is_finite() || !(1.0..=64.0).contains(&solver) || solver.fract() != 0.0 {
        return Err("server solver_iterations must be 1..64".into());
    }
    if v["vehicle"]
        .get("max_contacts_reported")
        .and_then(Value::as_f64)
        .is_some_and(|n| !n.is_finite() || !(0.0..=64.0).contains(&n) || n.fract() != 0.0)
    {
        return Err("too many reported contacts".into());
    }
    if v["vehicle"]["wheels"].as_array().is_some_and(|a| a.len() > 16) {
        return Err("too many vehicle wheels".into());
    }
    Ok((items.saturating_add(cells), solver as usize))
}

pub(super) fn binary(bytes: &[u8]) -> Result<(usize, usize), String> {
    if bytes.len() > MAX_CHECKPOINT {
        return Err("checkpoint exceeds server limit 32 MiB".into());
    }
    let word = |i: usize| {
        bytes
            .get(8 + i * 4..12 + i * 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
            .ok_or("invalid checkpoint header")
    };
    let population = word(2)?;
    let layers = word(3)?;
    if !(2..=20).contains(&layers) {
        return Err("checkpoint layer count exceeds server limit".into());
    }
    let parent = bytes.get(..8) == Some(b"ALTDCKP2".as_slice());
    if !parent && bytes.get(..8) != Some(b"ALTDCKP1".as_slice()) {
        return Err("invalid checkpoint magic".into());
    }
    if parent {
        for i in [5, 6, 7] {
            if word(i)? > MAX_POPULATION {
                return Err("checkpoint lineage count exceeds server limit".into());
            }
        }
    }
    let first = if parent { 10 } else { 5 };
    let shape = (0..layers).map(|i| word(first + i)).collect::<Result<Vec<_>, _>>()?;
    Ok((population, shape_numbers(&shape)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn parameter_population_and_work_products_are_bounded() {
        assert!(shape(&json!([20, 8, 5])).is_ok());
        assert!(shape(&json!([20, u64::MAX, 5])).is_err());
        assert!(shape(&json!([20, 0, 5])).is_err());
        assert!(shape_numbers(&[64; 20]).is_err());
        let p = Plan {
            population: 800,
            parameters: 213,
            solver: 16,
            ..Plan::default()
        };
        assert!(p.bytes().is_ok());
        assert!(p.ticks(180).is_ok());
        assert!(p.ticks(u64::MAX).is_err());
        assert!(Plan {
            population: 32768,
            parameters: 16384,
            ..p
        }
        .bytes()
        .is_err());
        assert!(settings(&json!({"population": u64::MAX}), 300).is_err());
    }
    #[test]
    fn admission_is_shared_and_releases_after_failure_or_disconnect() {
        let pool = Arc::new(Pool::default());
        let p = Plan {
            population: 32768,
            parameters: 2,
            ..Plan::default()
        };
        let a = Lease::new(pool.clone(), p).unwrap();
        let b = Lease::new(pool.clone(), p).unwrap();
        let c = Lease::new(pool.clone(), p).unwrap();
        let before = pool.reserved.load(Ordering::Relaxed);
        assert!(Lease::new(pool.clone(), p).is_err());
        assert_eq!(pool.reserved.load(Ordering::Relaxed), before);
        drop(a);
        let d = Lease::new(pool.clone(), p).unwrap();
        drop((b, c, d));
        assert_eq!(pool.reserved.load(Ordering::Relaxed), 0);
    }
    #[test]
    fn sparse_coordinates_and_large_geometry_are_rejected_before_building() {
        let mut scene = json!({"track":{"path":[[0,0],[1,1]],"tiles":[
            {"coords":[0,0]},{"coords":[1000000,1000000]}
        ]},"vehicle":{},"physics":{}});
        assert!(super::scene(&scene).is_err());
        scene["track"]["tiles"] = json!([]);
        scene["track"]["path"] = json!([[0, 0], [1000000, 1000000]]);
        assert!(super::scene(&scene).is_err());
        scene["track"]["path"] = json!([[0, 0], [1, 1]]);
        scene["physics"]["solver_iterations"] = json!(u64::MAX);
        assert!(super::scene(&scene).is_err());
    }
}

#[cfg(test)]
mod contention_tests {
    use super::*;
    #[test]
    fn busy_work_returns_an_error_instead_of_queueing() {
        let pool = Arc::new(Pool::default());
        let mut connection =
            crate::server::dispatch::Connection::with_budget(crate::math::profile::MathProfile::Proton, pool.clone());
        let lock = pool.work.lock().unwrap();
        let busy = connection.handle(1, "unknown", &Value::Null, None);
        assert!(format!("{busy:?}").contains("server busy"));
        drop(lock);
        let retry = connection.handle(1, "unknown", &Value::Null, None);
        assert!(format!("{retry:?}").contains("unknown operation"));
    }
}
