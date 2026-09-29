//! Scene loading (`compare_game_trace.load_scene`) and the static track.
//!
//! A `World` is built once per map and shared by every simulated car. All
//! spatial acceleration structures are conservative: a grid only narrows the
//! candidate set, and the final choice is made with the same arithmetic and
//! tie-breaking as the Python linear scans. Whenever a grid cannot prove its
//! answer, the query falls back to the exact linear scan.

use crate::pymath::{clamp, f32r, floor_i64};
use crate::vec2::{closest_point, V2};
use crate::godot_math::{ray_intersection, F2};
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surface {
    pub grip: f64,
    pub power: f64,
    pub steering: f64,
}

pub const ASPHALT: Surface = Surface { grip: 1.0, power: 1.0, steering: 1.0 };

#[derive(Clone, Debug)]
pub struct WheelConfig {
    pub position: V2,
    pub steering: bool,
    pub max_angle_deg: f64,
    pub power: f64,
    pub brake_power: f64,
    pub handbrake_power: f64,
}

#[derive(Clone, Debug)]
pub struct VehicleConfig {
    pub wheels: Vec<WheelConfig>,
    pub mass: f64,
    pub inertia: f64,
    pub half_width: f64,
    pub half_length: f64,
    pub shape_size: V2,
    pub shape_rotation: f64,
    pub shape_basis: (F2, F2),
    pub shape_position: V2,
    pub center_of_mass: V2,
    pub friction: f64,
    pub solver_iterations: usize,
    pub max_contacts_reported: usize,
    pub contact_bias: f64,
    pub allowed_penetration: f64,
    pub contact_recycle_radius: f64,
    pub contact_max_separation: f64,
    pub grip: f64,
    pub steering_speed: f64,
    pub center_steering: bool,
    pub air_resistance: f64,
    pub max_velocity: f64,
    pub custom_integrator: bool,
    pub gravity: V2,
    pub linear_damp: f64,
    pub angular_damp: f64,
    pub bounce: f64,
    pub wall_bounce: f64,
    pub surfaces: HashMap<String, Surface>,
}

impl VehicleConfig {
    /// `config.surfaces.get(name, ASPHALT)`.
    pub fn surface(&self, name: &str) -> Surface {
        self.surfaces.get(name).copied().unwrap_or(ASPHALT)
    }
}

/// A static convex tile collider with everything that does not depend on the car.
#[derive(Clone, Debug)]
pub struct PhysicsShape {
    pub tile:Option<(i64,i64)>,
    pub friction: f64,
    pub bounce: f64,
    /// Explicit pair ordering for a scene with known BVH allocation order.
    /// Legacy exports omit this hidden native state; retain car-first behavior.
    pub wall_first: bool,
    pub points: Vec<V2>,
    /// `_f32(origin)`.
    pub origin_f32: V2,
    /// `_f32(p - origin_f32)` for each point.
    pub local: Vec<V2>,
    /// `_normalize_f32((edge.y, -edge.x))` of each local edge.
    pub local_normals: Vec<V2>,
    pub min: V2,
    pub max: V2,
}

#[derive(Clone, Copy, Debug)]
pub struct PathSegment {
    pub start: V2,
    pub delta: V2,
    pub length_sq: f64,
    pub tangent: V2,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wall {
    pub start: V2,
    pub end: V2,
}

/// Uniform grid over which a list of item indices is stored per cell.
#[derive(Clone, Debug)]
struct CellLists {
    origin: V2,
    cell: f64,
    nx: usize,
    ny: usize,
    starts: Vec<u32>,
    items: Vec<u32>,
}

impl CellLists {
    fn from_lists(origin: V2, cell: f64, nx: usize, ny: usize, lists: Vec<Vec<u32>>) -> CellLists {
        let mut starts = Vec::with_capacity(lists.len() + 1);
        let mut items = Vec::new();
        starts.push(0);
        for list in lists {
            items.extend_from_slice(&list);
            starts.push(items.len() as u32);
        }
        CellLists { origin, cell, nx, ny, starts, items }
    }

    #[inline(always)]
    fn cell_index(&self, point: V2) -> Option<(usize, usize)> {
        let fx = ((point.x - self.origin.x) / self.cell).floor();
        let fy = ((point.y - self.origin.y) / self.cell).floor();
        if fx >= 0.0 && fy >= 0.0 && fx < self.nx as f64 && fy < self.ny as f64 {
            Some((fx as usize, fy as usize))
        } else {
            None
        }
    }

    #[inline(always)]
    fn list(&self, ix: usize, iy: usize) -> &[u32] {
        let cell = iy * self.nx + ix;
        &self.items[self.starts[cell] as usize..self.starts[cell + 1] as usize]
    }

    /// Cell range covering an axis-aligned box, or `None` if it leaves the grid.
    #[inline(always)]
    fn range(&self, min: V2, max: V2) -> Option<(usize, usize, usize, usize)> {
        let x0 = ((min.x - self.origin.x) / self.cell).floor();
        let y0 = ((min.y - self.origin.y) / self.cell).floor();
        let x1 = ((max.x - self.origin.x) / self.cell).floor();
        let y1 = ((max.y - self.origin.y) / self.cell).floor();
        if x0 < 0.0 || y0 < 0.0 || x1 >= self.nx as f64 || y1 >= self.ny as f64 {
            return None;
        }
        Some((x0 as usize, y0 as usize, x1 as usize, y1 as usize))
    }
}

/// Candidate lists for nearest-segment queries.
///
/// For cell `C`, `bound = min_s max_{p in C} dist(p, s)` (attained at a
/// corner because distance to a segment is convex). The list holds every
/// segment whose distance to `C` is at most `bound` plus a tolerance, in index
/// order. A best candidate within `bound` is therefore the global minimum, and
/// segments outside the list are strictly farther.
#[derive(Clone, Debug)]
struct NearestGrid {
    lists: CellLists,
    bound_sq: Vec<f64>,
}

fn point_segment_distance(p: V2, a: V2, b: V2) -> f64 {
    let d = b - a;
    let len_sq = d.x * d.x + d.y * d.y;
    let t = if len_sq > 0.0 { (((p.x - a.x) * d.x + (p.y - a.y) * d.y) / len_sq).clamp(0.0, 1.0) } else { 0.0 };
    let (qx, qy) = (a.x + d.x * t - p.x, a.y + d.y * t - p.y);
    (qx * qx + qy * qy).sqrt()
}

fn point_box_distance(p: V2, min: V2, max: V2) -> f64 {
    let dx = (min.x - p.x).max(0.0).max(p.x - max.x);
    let dy = (min.y - p.y).max(0.0).max(p.y - max.y);
    (dx * dx + dy * dy).sqrt()
}

/// Liang–Barsky: does the segment touch the closed box?
fn segment_touches_box(a: V2, b: V2, min: V2, max: V2) -> bool {
    let d = b - a;
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for (p, q) in [(-d.x, a.x - min.x), (d.x, max.x - a.x), (-d.y, a.y - min.y), (d.y, max.y - a.y)] {
        if p == 0.0 {
            if q < 0.0 {
                return false;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
            if t0 > t1 {
                return false;
            }
        }
    }
    true
}

fn box_segment_distance(min: V2, max: V2, a: V2, b: V2) -> f64 {
    if segment_touches_box(a, b, min, max) {
        return 0.0;
    }
    let corners = [min, V2::new(max.x, min.y), max, V2::new(min.x, max.y)];
    let mut best = point_box_distance(a, min, max).min(point_box_distance(b, min, max));
    for c in corners {
        best = best.min(point_segment_distance(c, a, b));
    }
    best
}

impl NearestGrid {
    fn build(segments: &[(V2, V2)], min: V2, max: V2, cell: f64) -> NearestGrid {
        let nx = (((max.x - min.x) / cell).ceil() as usize).max(1);
        let ny = (((max.y - min.y) / cell).ceil() as usize).max(1);
        let half_diag = cell * std::f64::consts::SQRT_2 * 0.5;
        let cells: Vec<(Vec<u32>, f64)> = (0..nx * ny)
            .into_par_iter()
            .map(|index| {
                let (ix, iy) = (index % nx, index / nx);
                let lo = V2::new(min.x + ix as f64 * cell, min.y + iy as f64 * cell);
                let hi = V2::new(lo.x + cell, lo.y + cell);
                let center = V2::new(lo.x + cell * 0.5, lo.y + cell * 0.5);
                let center_dist: Vec<f64> =
                    segments.iter().map(|&(a, b)| point_segment_distance(center, a, b)).collect();
                let nearest_center = center_dist.iter().copied().fold(f64::INFINITY, f64::min);
                // max over corners >= distance from the center, so only these can set the bound.
                let corners = [lo, V2::new(hi.x, lo.y), hi, V2::new(lo.x, hi.y)];
                let mut bound = f64::INFINITY;
                for (i, &(a, b)) in segments.iter().enumerate() {
                    if center_dist[i] <= nearest_center + half_diag + 1e-9 {
                        let far = corners
                            .iter()
                            .map(|&c| point_segment_distance(c, a, b))
                            .fold(0.0, f64::max);
                        bound = bound.min(far);
                    }
                }
                let tolerance = 1e-6 * (1.0 + bound) + 1e-3;
                let mut list = Vec::new();
                for (i, &(a, b)) in segments.iter().enumerate() {
                    // box distance >= center distance - half diagonal.
                    if center_dist[i] - half_diag <= bound + tolerance
                        && box_segment_distance(lo, hi, a, b) <= bound + tolerance
                    {
                        list.push(i as u32);
                    }
                }
                (list, bound * bound)
            })
            .collect();
        let (lists, bound_sq): (Vec<Vec<u32>>, Vec<f64>) = cells.into_iter().unzip();
        NearestGrid { lists: CellLists::from_lists(min, cell, nx, ny, lists), bound_sq }
    }

    #[inline(always)]
    fn query(&self, point: V2) -> Option<(&[u32], f64)> {
        let (ix, iy) = self.lists.cell_index(point)?;
        Some((self.lists.list(ix, iy), self.bound_sq[iy * self.lists.nx + ix]))
    }
}

/// Items registered in every cell that their (padded) bounding box touches.
#[derive(Clone, Debug)]
struct BoxGrid {
    lists: CellLists,
}

impl BoxGrid {
    fn build(boxes: &[(V2, V2)], min: V2, max: V2, cell: f64, pad: f64) -> BoxGrid {
        let nx = (((max.x - min.x) / cell).ceil() as usize).max(1);
        let ny = (((max.y - min.y) / cell).ceil() as usize).max(1);
        let mut lists = vec![Vec::new(); nx * ny];
        for (i, &(lo, hi)) in boxes.iter().enumerate() {
            let x0 = (((lo.x - pad - min.x) / cell).floor().max(0.0) as usize).min(nx - 1);
            let y0 = (((lo.y - pad - min.y) / cell).floor().max(0.0) as usize).min(ny - 1);
            let x1 = (((hi.x + pad - min.x) / cell).floor().max(0.0) as usize).min(nx - 1);
            let y1 = (((hi.y + pad - min.y) / cell).floor().max(0.0) as usize).min(ny - 1);
            for iy in y0..=y1 {
                for ix in x0..=x1 {
                    lists[iy * nx + ix].push(i as u32);
                }
            }
        }
        BoxGrid { lists: CellLists::from_lists(min, cell, nx, ny, lists) }
    }
}

/// Dense lookup keyed by `(floor(x / 768), floor(y / 768))`.
#[derive(Clone, Debug)]
struct TileTable<T: Clone> {
    x0: i64,
    y0: i64,
    nx: i64,
    ny: i64,
    cells: Vec<Option<T>>,
}

impl<T: Clone> TileTable<T> {
    fn new(entries: &[((i64, i64), T)]) -> TileTable<T> {
        if entries.is_empty() {
            return TileTable { x0: 0, y0: 0, nx: 0, ny: 0, cells: Vec::new() };
        }
        let x0 = entries.iter().map(|e| e.0 .0).min().unwrap();
        let y0 = entries.iter().map(|e| e.0 .1).min().unwrap();
        let nx = entries.iter().map(|e| e.0 .0).max().unwrap() - x0 + 1;
        let ny = entries.iter().map(|e| e.0 .1).max().unwrap() - y0 + 1;
        let mut cells = vec![None; (nx * ny) as usize];
        for ((x, y), value) in entries {
            // Python dict construction: later tiles with the same key win.
            cells[((y - y0) * nx + (x - x0)) as usize] = Some(value.clone());
        }
        TileTable { x0, y0, nx, ny, cells }
    }

    #[inline(always)]
    fn get(&self, point: V2) -> Option<&T> {
        let x = floor_i64(point.x / 768.0) - self.x0;
        let y = floor_i64(point.y / 768.0) - self.y0;
        if x < 0 || y < 0 || x >= self.nx || y >= self.ny {
            return None;
        }
        self.cells[(y * self.nx + x) as usize].as_ref()
    }
}

pub struct Track {
    /// Fresh native tree with an explicit shape construction order.
    /// Omitted by legacy geometry exports, which lack native construction state.
    pub native_broadphase: bool,
    pub native_body_order_reverse: bool,
    pub native_tilemap: bool,
    pub native_cell_order:Vec<(i64,i64)>,
    pub native_tile_order:Vec<(i64,i64)>,
    pub bsp: Option<Box<crate::bsp::Node>>,
    /// `bsp` arranged for raycasts.
    bsp_rays: Option<crate::bsp::RayTree>,
    // Legacy exports omit whether an empty raycaster was constructed.
    raycaster_present: Option<bool>,
    pub walls: Vec<Wall>,
    pub shapes: Vec<PhysicsShape>,
    pub path: Vec<V2>,
    pub path_forward: Vec<V2>,
    pub curve: Option<crate::curve::Curve>,
    pub path_offsets: Vec<f64>,
    path_grid: Option<crate::segment_grid::SegmentGrid>,
    path_segments: crate::path_segments::PathSegments,
    pub segments: Vec<PathSegment>,
    /// `atan2(a.cross(b), a.dot(b))` between consecutive exported tangents.
    forward_angles: Vec<f64>,
    surfaces: TileTable<Surface>,
    surface_names: TileTable<String>,
    default_surface: Surface,
    connections: TileTable<Option<(V2, V2)>>,
    wall_grid: Option<NearestGrid>,
    ray_grid: Option<BoxGrid>,
    shape_grid: Option<BoxGrid>,
}

/// Everything a simulation needs that is shared and immutable.
pub struct World {
    pub vehicle: VehicleConfig,
    pub track: Track,
}

pub const PATH_WINDOW: f64 = 450.0;
const RAY_CELL: f64 = 64.0;
const RAY_PAD: f64 = 1e-3;

#[derive(Clone, Copy, Debug)]
pub struct PathHit {
    pub offset: f64,
    pub near: V2,
    pub tangent: V2,
}

impl Track {
    pub fn path_length(&self) -> f64 {
        *self.path_offsets.last().unwrap_or(&0.0)
    }

    /// The raycast arrangement of `bsp`, if any (used by the GPU export).
    pub fn ray_tree(&self) -> Option<&crate::bsp::RayTree> {
        self.bsp_rays.as_ref()
    }

    /// Surface names and path-sensor connections over the union of both tile
    /// tables: `(x0, y0, nx, ny, cells)` with cells row-major, each queried
    /// with `get` at the tile centre.
    #[allow(clippy::type_complexity)]
    pub fn gpu_tiles(&self) -> (i64, i64, i64, i64, Vec<(Option<String>, Option<(V2, V2)>)>) {
        let tables = [(self.surface_names.x0, self.surface_names.y0, self.surface_names.nx, self.surface_names.ny),
            (self.connections.x0, self.connections.y0, self.connections.nx, self.connections.ny)];
        let used: Vec<_> = tables.iter().filter(|t| t.2 > 0 && t.3 > 0).collect();
        if used.is_empty() {
            return (0, 0, 0, 0, Vec::new());
        }
        let x0 = used.iter().map(|t| t.0).min().unwrap();
        let y0 = used.iter().map(|t| t.1).min().unwrap();
        let x1 = used.iter().map(|t| t.0 + t.2).max().unwrap();
        let y1 = used.iter().map(|t| t.1 + t.3).max().unwrap();
        let mut cells = Vec::with_capacity(((x1 - x0) * (y1 - y0)) as usize);
        for y in y0..y1 {
            for x in x0..x1 {
                let centre = V2::new((x as f64 + 0.5) * 768.0, (y as f64 + 0.5) * 768.0);
                cells.push((self.surface_names.get(centre).cloned(), self.connections.get(centre).and_then(|c| *c)));
            }
        }
        (x0, y0, x1 - x0, y1 - y0, cells)
    }

    /// Baked path segment rows `[x, y, dx, dy, len2]` (GPU export).
    pub fn gpu_path_rows(&self) -> Vec<[f32; 5]> {
        self.path_segments.rows()
    }

    /// Whether the scene constructed a raycaster (`None` for legacy exports).
    pub fn raycaster_present(&self) -> Option<bool> {
        self.raycaster_present
    }

    /// `config.surfaces.get(track.surface_at(point), ASPHALT)`.
    #[inline(always)]
    pub fn surface(&self, point: V2) -> Surface {
        self.surfaces.get(point).copied().unwrap_or(self.default_surface)
    }

    /// Vehicle surface properties are read live; only terrain identity belongs to the track.
    pub fn vehicle_surface(&self, point: V2, vehicle: &VehicleConfig) -> Surface {
        vehicle.surface(self.surface_names.get(point).map_or("asphalt", String::as_str))
    }

    /// Original TrackManager float projection and wrapped segment-index window.
    pub fn closest_path(&self, point: V2, previous: Option<f64>) -> PathHit {
        use crate::godot_math::F2;
        assert!(self.path.len() >= 2, "track path needs at least two baked points");
        let count = self.path.len() - 1;
        let length = self.path_length() as f32;
        let posmod = |x: f32| { let r = x % length; if r < 0.0 { r + length } else { r } };
        let index = |offset: f32| self.path_offsets.binary_search_by(|v| v.partial_cmp(&(offset as f64)).unwrap()).unwrap_or_else(|i| i.saturating_sub(1)) as isize;
        let (mut from, mut to) = (0isize, count as isize);
        if let Some(previous) = previous.filter(|_| length > 2.0 * PATH_WINDOW as f32) {
            let previous = previous as f32;
            let center = index(posmod(previous));
            from = index(posmod(previous - PATH_WINDOW as f32));
            to = index(posmod(previous + PATH_WINDOW as f32)) + 1;
            if from > center { from -= count as isize; }
            if to < center { to += count as isize; }
        }
        let point = F2::from(point);
        let segments = &self.path_segments;
        let mut distance = f32::MAX;
        let mut nearest = None;
        // A full ascending scan can visit only the exact candidate subset.
        let full = from == 0 && to == count as isize;
        let visit = |i: usize| {
            let d2 = segments.project(point, i).d2;
            if d2 < distance {
                distance = d2;
                nearest = Some(i);
            }
        };
        if !(full && self.path_grid.as_ref().is_some_and(|grid| grid.scan(point, |i| segments.project(point, i).d2, visit))) {
            // The wrapped window, as runs of ascending segment indices.
            let mut i = from;
            while i < to {
                let start = i.rem_euclid(count as isize) as usize;
                let run = ((to - i) as usize).min(count - start);
                segments.scan(point, start..start + run, &mut distance, &mut nearest);
                i += run as isize;
            }
        }
        let Some(i) = nearest else {
            return PathHit { offset: 0.0, near: V2::ZERO, tangent: V2::ZERO };
        };
        let hit = segments.project(point, i);
        let begin = self.path_offsets[i] as f32;
        let end = self.path_offsets[i + 1] as f32;
        PathHit {
            offset: if length > 0.0 { posmod(begin + (end - begin) * hit.fraction) as f64 } else { 0.0 },
            near: hit.near.into(),
            tangent: hit.delta.normalized().into(),
        }
    }

    /// `Track.closest_wall(point)`.
    pub fn closest_wall(&self, point: V2) -> Option<V2> {
        if self.raycaster_present == Some(false) { return None; }
        if let Some(bsp) = &self.bsp { return Some(bsp.closest(point)); }
        if self.raycaster_present == Some(true) { return Some(V2::ZERO); }
        let candidate = |index: usize| {
            let wall = &self.walls[index];
            let p = closest_point(point, wall.start, wall.end);
            ((p - point).dot(p - point), p)
        };
        if let Some((list, bound_sq)) = self.wall_grid.as_ref().and_then(|g| g.query(point)) {
            let mut best: Option<(f64, V2)> = None;
            for &index in list {
                let c = candidate(index as usize);
                if best.map_or(true, |b| c.0 < b.0) {
                    best = Some(c);
                }
            }
            if let Some((d2, p)) = best {
                if d2 <= bound_sq {
                    return Some(p);
                }
            }
        }
        let mut best: Option<(f64, V2)> = None;
        for index in 0..self.walls.len() {
            let c = candidate(index);
            if best.map_or(true, |b| c.0 < b.0) {
                best = Some(c);
            }
        }
        best.map(|b| b.1)
    }

    fn raycast_linear(&self, start: V2, end: V2) -> Option<V2> {
        let mut best: Option<(f64, V2)> = None;
        for wall in &self.walls {
            if let Some(hit) = ray_intersection(start, end, wall.start, wall.end) {
                let delta = F2::from(hit) - F2::from(start);
                let d2 = delta.dot(delta) as f64;
                if best.map_or(true, |b| d2 < b.0) {
                    best = Some((d2, hit));
                }
            }
        }
        best.map(|b| b.1)
    }

    /// `Track.raycast(start, end)`: nearest wall crossing, first wall on ties.
    pub fn raycast(&self, start: V2, end: V2, stamps: &mut RayStamps) -> Option<V2> {
        if self.raycaster_present == Some(false) { return None; }
        if let Some(tree) = &self.bsp_rays { return tree.raycast(start, end); }
        let Some(grid) = self.ray_grid.as_ref() else {
            return self.raycast_linear(start, end);
        };
        let lo = V2::new(start.x.min(end.x) - RAY_PAD, start.y.min(end.y) - RAY_PAD);
        let hi = V2::new(start.x.max(end.x) + RAY_PAD, start.y.max(end.y) + RAY_PAD);
        if grid.lists.range(lo, hi).is_none() {
            return self.raycast_linear(start, end);
        }
        stamps.begin(self.walls.len());
        let ray = end - start;
        let length = (ray.x * ray.x + ray.y * ray.y).sqrt();
        let chunks = ((length / (RAY_CELL * 0.5)).ceil() as usize).max(1);
        // (d2, wall index, hit)
        let mut best: Option<(f64, usize, V2)> = None;
        for chunk in 0..chunks {
            let t0 = chunk as f64 / chunks as f64;
            let t1 = (chunk + 1) as f64 / chunks as f64;
            let a = V2::new(start.x + ray.x * t0, start.y + ray.y * t0);
            let b = V2::new(start.x + ray.x * t1, start.y + ray.y * t1);
            let clo = V2::new(a.x.min(b.x) - RAY_PAD, a.y.min(b.y) - RAY_PAD);
            let chi = V2::new(a.x.max(b.x) + RAY_PAD, a.y.max(b.y) + RAY_PAD);
            let (x0, y0, x1, y1) = grid.lists.range(clo, chi).expect("chunk inside ray box");
            for iy in y0..=y1 {
                for ix in x0..=x1 {
                    for &index in grid.lists.list(ix, iy) {
                        let index = index as usize;
                        if !stamps.mark(index) {
                            continue;
                        }
                        let wall = &self.walls[index];
                        if let Some(hit) = ray_intersection(start, end, wall.start, wall.end) {
                            let delta = F2::from(hit) - F2::from(start);
                            let d2 = delta.dot(delta) as f64;
                            if best.map_or(true, |b| d2 < b.0 || (d2 == b.0 && index < b.1)) {
                                best = Some((d2, index, hit));
                            }
                        }
                    }
                }
            }
            if let Some((d2, _, _)) = best {
                let reach = t1 * length;
                if chunk + 1 < chunks && d2 < (reach - RAY_PAD).max(0.0).powi(2) {
                    break;
                }
            }
        }
        best.map(|b| b.2)
    }

    /// `Track.path_sensor_point(point)`.
    pub fn path_sensor_point(&self, point: V2) -> V2 {
        match self.connections.get(point) {
            Some(Some((first, last))) => {
                let p = crate::godot_math::F2::from(point);
                let a = crate::godot_math::F2::from(*first);
                let v = crate::godot_math::F2::from(*last) - a;
                let n = v.dot(v);
                let closest = if n == 0.0 { a } else { a + v * ((p.dot(v) - a.dot(v)).clamp(0.0,n)/n) };
                (closest + (p - closest) * 0.5).into()
            },
            _ => point,
        }
    }

    /// Linear baked-path sample, used by AgentStats for center distance at
    /// the same continuity-constrained offset that was selected for score.
    pub fn path_position(&self, offset: f64) -> V2 {
        if let Some(curve) = &self.curve { return curve.position(offset as f32).into(); }
        if self.path.len() < 2 { return self.path.first().copied().unwrap_or(V2::ZERO); }
        let offset = clamp(offset, 0.0, self.path_length());
        let index = (self.path.len() - 2).min(self.path_offsets.partition_point(|&x| x <= offset).saturating_sub(1));
        let (start, end) = (self.path_offsets[index], self.path_offsets[index + 1]);
        let fraction = if end > start { (offset - start) / (end - start) } else { 0.0 };
        self.path[index] + (self.path[index + 1] - self.path[index]) * fraction
    }

    /// `Track.path_direction(offset)`.
    pub fn path_direction(&self, offset: f64) -> V2 {
        let length = self.path_length();
        if length <= 0.0 {
            return V2::ZERO;
        }
        let offset = clamp(offset, 0.0, length);
        let index = (self.path.len() - 2).min(self.path_offsets.partition_point(|&x| x <= offset) - 1);
        let (start, end) = (self.path_offsets[index], self.path_offsets[index + 1]);
        let fraction = if end > start { (offset - start) / (end - start) } else { 0.5 };
        if !self.path_forward.is_empty() {
            self.path_forward[index].rotated(self.forward_angles[index] * fraction).normalized()
        } else {
            (self.path[index + 1] - self.path[index]).normalized()
        }
    }

    /// Physics shapes whose bounding box may overlap the given box, sorted by index.
    pub fn shape_candidates(&self, lo: V2, hi: V2, out: &mut Vec<u32>) {
        out.clear();
        match self.shape_grid.as_ref().and_then(|g| g.lists.range(lo, hi).map(|r| (g, r))) {
            Some((grid, (x0, y0, x1, y1))) => {
                for iy in y0..=y1 {
                    for ix in x0..=x1 {
                        out.extend_from_slice(grid.lists.list(ix, iy));
                    }
                }
                out.sort_unstable();
                out.dedup();
            }
            None => out.extend(0..self.shapes.len() as u32),
        }
    }
}

/// Per-thread scratch marks for deduplicating walls during one raycast.
#[derive(Default, Clone)]
pub struct RayStamps {
    marks: Vec<u32>,
    generation: u32,
}

impl RayStamps {
    fn begin(&mut self, count: usize) {
        if self.marks.len() < count {
            self.marks.resize(count, 0);
        }
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.marks.iter_mut().for_each(|m| *m = 0);
            self.generation = 1;
        }
    }

    /// Returns true the first time an index is seen in this raycast.
    #[inline(always)]
    fn mark(&mut self, index: usize) -> bool {
        if self.marks[index] == self.generation {
            false
        } else {
            self.marks[index] = self.generation;
            true
        }
    }
}

// ---------------------------------------------------------------------------
// Scene JSON
// ---------------------------------------------------------------------------

fn num(value: &Value) -> f64 {
    value.as_f64().unwrap_or_else(|| panic!("expected a number, got {value}"))
}

pub fn vector(value: &Value) -> V2 {
    V2::new(num(&value[0]), num(&value[1]))
}

/// Python truthiness fallback: `value or default`.
fn num_or(value: Option<&Value>, default: f64) -> f64 {
    match value.and_then(Value::as_f64) {
        Some(v) => v,
        _ => default,
    }
}

fn get_or(object: &Value, key: &str, default: f64) -> f64 {
    object.get(key).map(num).unwrap_or(default)
}

pub fn load_vehicle(scene: &Value) -> VehicleConfig {
    let car = &scene["vehicle"];
    let size = vector(&car["shape_size"]);
    let angle = num(&car["shape_rotation"]);
    // An exported matrix preserves managed/native constructor rounding and scale.
    let shape_basis = car.get("shape_basis").map(|b| (F2::from(vector(&b[0])), F2::from(vector(&b[1]))))
        .unwrap_or_else(|| {
            let c=crate::native_math::engine_cos(angle as f32);
            let s=crate::native_math::engine_sin(angle as f32);
            (F2{x:c,y:s},F2{x:-s,y:c})
        });
    let physics = scene.get("physics").cloned().unwrap_or(Value::Null);
    let physics_get = |key: &str, default: f64| physics.get(key).map(num).unwrap_or(default);
    let surfaces = car["surfaces"]
        .as_object()
        .expect("vehicle surfaces")
        .iter()
        .map(|(name, values)| {
            (
                name.clone(),
                Surface {
                    grip: get_or(values, "grip", 1.0),
                    power: get_or(values, "power", 1.0),
                    steering: get_or(values, "steering", 1.0),
                },
            )
        })
        .collect();
    let config = VehicleConfig {
        wheels: car["wheels"]
            .as_array()
            .expect("wheels")
            .iter()
            .map(|w| WheelConfig {
                position: vector(&w["position"]),
                steering: w["steering"].as_bool().expect("wheel steering flag"),
                max_angle_deg: num(&w["max_angle_deg"]),
                power: num(&w["power"]),
                brake_power: num(&w["brake_power"]),
                handbrake_power: num(&w["handbrake_power"]),
            })
            .collect(),
        mass: num(&car["mass"]),
        inertia: num(&car["inertia_server"]),
        half_width: (angle.cos().abs() * size.x + angle.sin().abs() * size.y) / 2.0,
        half_length: (angle.sin().abs() * size.x + angle.cos().abs() * size.y) / 2.0,
        shape_size: size,
        shape_rotation: angle,
        shape_basis,
        shape_position: car.get("shape_position").map(vector).unwrap_or(V2::ZERO),
        center_of_mass: car.get("center_of_mass").or_else(|| car.get("shape_position")).map(vector).unwrap_or(V2::ZERO),
        friction: num_or(car.get("friction"), 1.0),
        solver_iterations: physics_get("solver_iterations", 16.0) as usize,
        max_contacts_reported: num_or(car.get("max_contacts_reported"), 8.0) as usize,
        contact_bias: physics_get("contact_default_bias", 0.8),
        contact_recycle_radius: physics_get("contact_recycle_radius", 1.0),
        contact_max_separation: physics_get("contact_max_separation", 1.5),
        allowed_penetration: physics_get("contact_max_allowed_penetration", 0.3),
        grip: num(&car["grip"]),
        steering_speed: num(&car["steering_speed"]),
        center_steering: car["center_steering"].as_bool().expect("center_steering"),
        air_resistance: num(&car["air_resistance"]),
        max_velocity: num(&car["max_velocity"]),
        custom_integrator: car["custom_integrator"].as_bool().unwrap_or(false),
        gravity: (F2::from(physics.get("gravity").map(vector).unwrap_or(V2::ZERO))
            * num_or(car.get("gravity_scale"),1.0) as f32).into(),
        linear_damp: num(&car["linear_damp"]),
        angular_damp: num(&car["angular_damp"]),
        bounce: num_or(car.get("bounce"), 0.0),
        wall_bounce: get_or(&scene["track"], "wall_bounce", 0.0),
        surfaces,
    };
    assert!(
        config.mass > 0.0 && config.inertia > 0.0 && config.half_width > 0.0 && config.half_length > 0.0,
        "mass, inertia and car half extents must be positive"
    );
    assert!(!config.wheels.is_empty(), "at least one wheel is required");
    config
}

fn normalize_f32(v: V2) -> V2 {
    let dot = f32r(f32r(v.x * v.x) + f32r(v.y * v.y));
    let length = f32r(dot.sqrt());
    if length != 0.0 {
        V2::new(f32r(v.x / length), f32r(v.y / length))
    } else {
        V2::ZERO
    }
}

impl PhysicsShape {
    fn new(points: Vec<V2>, origin: V2, native_local:Option<Vec<V2>>) -> PhysicsShape {
        let origin_f32 = V2::new(f32r(origin.x), f32r(origin.y));
        // Native exports preserve polygon support order and local float bits.
        // Reconstructing them from translated world coordinates can lose both.
        let local: Vec<V2> = native_local.unwrap_or_else(||
            points.iter().map(|p| V2::new(f32r(p.x - origin_f32.x), f32r(p.y - origin_f32.y))).collect());
        let local_normals = (0..local.len())
            .map(|i| {
                let (point, next) = (local[i], local[(i + 1) % local.len()]);
                let edge = V2::new(f32r(next.x - point.x), f32r(next.y - point.y));
                normalize_f32(V2::new(edge.y, -edge.x))
            })
            .collect();
        let min = V2::new(
            points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min),
            points.iter().map(|p| p.y).fold(f64::INFINITY, f64::min),
        );
        let max = V2::new(
            points.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max),
            points.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max),
        );
        PhysicsShape { points, origin_f32, local, local_normals, min, max, wall_first: false,friction:1.0,bounce:0.0,tile:None }
    }
}

impl World {
    /// `compare_game_trace.load_scene(scene)` plus the static acceleration grids.
    pub fn from_scene(scene: &Value) -> World {
        let vehicle = load_vehicle(scene);
        let track = &scene["track"];
        let empty = Vec::new();
        let array = |key: &str| track.get(key).and_then(Value::as_array).unwrap_or(&empty);
        let tile_origin = track.get("tile_map_position").map(vector).unwrap_or(V2::ZERO);
        let tile_size = track.get("tile_size").map(vector).unwrap_or(V2::new(768.0, 768.0));

        let walls: Vec<Wall> =
            array("walls").iter().map(|w| Wall { start: vector(&w[0]), end: vector(&w[1]) }).collect();
        let shapes: Vec<PhysicsShape> = array("physics_shapes")
            .iter()
            .map(|shape| {
                let tile = shape.get("tile").map(vector).unwrap_or(V2::ZERO);
                let origin = V2::new(
                    (tile.x + 0.5) * tile_size.x + tile_origin.x,
                    (tile.y + 0.5) * tile_size.y + tile_origin.y,
                );
                let origin = shape.get("origin").map(vector).unwrap_or(origin);
                let points = shape["points"].as_array().expect("shape points").iter().map(|p| vector(p) + tile_origin).collect();
                let local=shape["local_points"].as_array().map(|points|points.iter().map(|v|crate::curve::json_vector(v).into()).collect());
                let mut collider = PhysicsShape::new(points, origin, local);
                collider.tile=shape.get("tile").map(|t|(num(&t[0]) as i64,num(&t[1]) as i64));
                collider.wall_first = shape.get("wall_first").and_then(Value::as_bool).unwrap_or(false);
                collider.friction=num_or(shape.get("friction"),num_or(track.get("wall_friction"),1.0));
                collider.bounce=num_or(shape.get("bounce"),num_or(track.get("wall_bounce"),0.0));
                collider
            })
            .collect();
        let path: Vec<V2> = array("path").iter().map(|v| crate::godot_math::F2::from(vector(v)).into()).collect();
        let path_forward: Vec<V2> = array("path_forward").iter().map(vector).collect();
        assert!(
            path_forward.is_empty() || path_forward.len() == path.len(),
            "path_forward and path must have the same number of points"
        );

        let tiles = array("tiles");
        let key = |tile: &Value| (num(&tile["coords"][0]) as i64, num(&tile["coords"][1]) as i64);
        let surface_entries: Vec<((i64, i64), Surface)> = tiles
            .iter()
            .map(|tile| (key(tile), vehicle.surface(tile["surface"].as_str().expect("tile surface"))))
            .collect();
        let connection_entries: Vec<((i64, i64), Option<(V2, V2)>)> = tiles
            .iter()
            .filter_map(|tile| {
                let points = tile.get("connections")?.as_array()?;
                let pair = if points.is_empty() {
                    None
                } else {
                    Some((vector(&points[0]), vector(points.last().unwrap())))
                };
                Some((key(tile), pair))
            })
            .collect();

        let mut path_offsets = vec![0.0];
        for pair in path.windows(2) {
            path_offsets.push((*path_offsets.last().unwrap() as f32 + (crate::godot_math::F2::from(pair[1]) - crate::godot_math::F2::from(pair[0])).length()) as f64);
        }
        let segments: Vec<PathSegment> = path
            .windows(2)
            .map(|pair| {
                let delta = pair[1] - pair[0];
                PathSegment { start: pair[0], delta, length_sq: delta.dot(delta), tangent: delta.normalized() }
            })
            .collect();
        let forward_angles = path_forward.windows(2).map(|p| p[0].cross(p[1]).atan2(p[0].dot(p[1]))).collect();

        // Grid extents: everything static on the map plus a margin for cars
        // that leave the walls. Queries outside use the linear scans.
        let mut lo = V2::new(f64::INFINITY, f64::INFINITY);
        let mut hi = V2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
        let mut include = |p: V2| {
            lo = V2::new(lo.x.min(p.x), lo.y.min(p.y));
            hi = V2::new(hi.x.max(p.x), hi.y.max(p.y));
        };
        path.iter().for_each(|&p| include(p));
        walls.iter().for_each(|w| {
            include(w.start);
            include(w.end)
        });
        shapes.iter().for_each(|s| s.points.iter().for_each(|&p| include(p)));
        let margin = 768.0;
        let (lo, hi) = (V2::new(lo.x - margin, lo.y - margin), V2::new(hi.x + margin, hi.y + margin));
        let finite = lo.x.is_finite() && hi.x.is_finite();

        let wall_segments: Vec<(V2, V2)> = walls.iter().map(|w| (w.start, w.end)).collect();
        let wall_grid = (finite && !walls.is_empty()).then(|| NearestGrid::build(&wall_segments, lo, hi, 32.0));
        let wall_boxes: Vec<(V2, V2)> = walls
            .iter()
            .map(|w| {
                (V2::new(w.start.x.min(w.end.x), w.start.y.min(w.end.y)), V2::new(w.start.x.max(w.end.x), w.start.y.max(w.end.y)))
            })
            .collect();
        let ray_grid = (finite && !walls.is_empty()).then(|| BoxGrid::build(&wall_boxes, lo, hi, RAY_CELL, RAY_PAD * 2.0));
        let shape_boxes: Vec<(V2, V2)> = shapes.iter().map(|s| (s.min, s.max)).collect();
        let shape_grid = (finite && !shapes.is_empty()).then(|| BoxGrid::build(&shape_boxes, lo, hi, 128.0, 1.0));

        let bsp = crate::bsp::Node::from_polygons(array("polygons"));
        let path_segments = crate::path_segments::PathSegments::new(&path);
        let path_grid = crate::segment_grid::SegmentGrid::new(&path.iter().map(|&p| F2::from(p)).collect::<Vec<_>>());
        let default_surface = vehicle.surface("asphalt");
        World {
            track: Track {
                native_broadphase: track.get("native_broadphase").and_then(Value::as_bool).unwrap_or(false),
                native_body_order_reverse: track.get("native_body_order").and_then(Value::as_str)==Some("reverse"),
                native_tilemap: track.get("native_creation").and_then(Value::as_str)==Some("tilemap"),
                native_tile_order:tiles.iter().map(key).collect(),
                native_cell_order:track["native_cell_order"].as_array().map(|cells|cells.iter().map(|t|(num(&t[0]) as i64,num(&t[1]) as i64)).collect()).unwrap_or_else(||tiles.iter().map(key).collect()),
                bsp_rays: bsp.as_deref().map(crate::bsp::RayTree::new),
                bsp,
                raycaster_present: track["raycaster_present"].as_bool(),
                walls,
                shapes,
                path,
                path_forward,
                curve: track.get("curve").map(crate::curve::Curve::from_json),
                path_grid,
                path_segments,
                path_offsets,
                segments,
                forward_angles,
                surfaces: TileTable::new(&surface_entries),
                surface_names: TileTable::new(&tiles.iter().map(|tile| (key(tile), tile["surface"].as_str().unwrap().to_owned())).collect::<Vec<_>>()),
                default_surface,
                connections: TileTable::new(&connection_entries),
                wall_grid,
                ray_grid,
                shape_grid,
            },
            vehicle,
        }
    }

    /// Drop the acceleration grids to force the Python linear scans (for testing).
    pub fn without_grids(mut self) -> World {
        self.track.wall_grid = None;
        self.track.ray_grid = None;
        self.track.shape_grid = None;
        self
    }
}
