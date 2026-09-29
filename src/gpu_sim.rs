//! Host side of the GPU simulator: #[repr(C)] mirrors of gpu/sim/state.h and
//! gpu/sim/world.h, the exporters that fill them from the CPU structures, and
//! the handle that runs the simulation kernels (libaltd_gpu.so).
use crate::car::{Sensor, DT};
use crate::godot_math::F2;
use crate::gpu::{Gpu, GpuWorld};
use crate::vec2::V2;
use crate::world::{Surface, VehicleConfig, World, ASPHALT};
use rayon::prelude::*;
use std::ffi::c_void;

pub type Vec2 = [f32; 2];

pub const MAX_WHEELS: usize = 4;
pub const MAX_PAIRS: usize = 16;
pub const MAX_CONTACTS: usize = 32;
pub const MAX_SHAPE_POINTS: usize = 16;
pub const MAX_SENSORS: usize = 32;
pub const RECENT: usize = 10;

/// GPU per-car error bits beyond the math bits of `crate::gpu`.
pub const ERR_PAIRS: u32 = 8;
pub const ERR_CONTACTS: u32 = 16;
pub const ERR_PATH_NAN: u32 = 32;

pub const CONTACT_WALL_FIRST: u32 = 1;
pub const CONTACT_USED: u32 = 2;

pub const CAR_ACTIVE: u32 = 1;
pub const CAR_RESET_RIGHT: u32 = 2;
pub const CAR_PENDING_VELOCITY: u32 = 4;
pub const CAR_HAS_SHAPE: u32 = 8;
pub const CAR_HAS_LEAF: u32 = 16;
pub const CAR_PAIR_CHECK: u32 = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuContact {
    pub normal: Vec2,
    pub point: Vec2,
    pub wall_point: Vec2,
    pub local_point: Vec2,
    pub wall_local_point: Vec2,
    pub depth: f32,
    pub shape: u32,
    pub flags: u32,
    pub normal_mass: f32,
    pub tangent_mass: f32,
    pub bias: f32,
    pub bounce: f32,
    pub friction: f32,
    pub acc_normal: f32,
    pub acc_tangent: f32,
    pub acc_bias: f32,
    pub acc_bias_center: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuCar {
    pub position: Vec2,
    pub velocity: Vec2,
    pub acceleration: Vec2,
    pub reset_acceleration: Vec2,
    pub basis_x: Vec2,
    pub basis_y: Vec2,
    pub rotation: f32,
    pub transform_angle: f32,
    pub angular_velocity: f32,
    pub boost_energy: f32,
    pub flags: u32,
    pub collision_count: u32,
    pub tick: u64,
    pub wheel_angle: [f32; MAX_WHEELS],
    pub wheel_previous: [Vec2; MAX_WHEELS],
    pub wheel_surface: [u32; MAX_WHEELS],
    pub wheel_has_previous: u32,
    pub pair_count: u32,
    pub contact_count: u32,
    pub error: u32,
    pub shape_position: Vec2,
    pub shape_size: Vec2,
    pub leaf_min: Vec2,
    pub leaf_max: Vec2,
    pub pairs: [u32; MAX_PAIRS],
    pub axes: [Vec2; MAX_PAIRS],
    pub contacts: [GpuContact; MAX_CONTACTS],
}

impl GpuCar {
    pub fn zeroed() -> GpuCar {
        // SAFETY: plain old data; all-zero bits are a valid value.
        unsafe { std::mem::zeroed() }
    }
}

pub const AGENT_HAS_PREVIOUS_OFFSET: u32 = 1;
pub const AGENT_WENT_BACKWARDS: u32 = 2;
pub const AGENT_HAS_LAP_CROSSING: u32 = 4;
pub const AGENT_PENDING_CONTACT: u32 = 8;
pub const AGENT_HAS_BEST_LAP: u32 = 16;
pub const AGENT_HAS_ALL_LAPS: u32 = 32;
pub const AGENT_HAS_DEACTIVATED: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuAgent {
    pub controls: [f64; 5],
    pub previous_offset: f64,
    pub tracker_score: f64,
    pub lap_crossing_fraction: f64,
    pub lap_count: u64,
    pub network_novelty: f64,
    pub update_count: u64,
    pub collision_count: u64,
    pub total_score: f64,
    pub total_speed: f64,
    pub total_abs_steering: f64,
    pub total_actual_distance: f64,
    pub total_drifting_amount: f64,
    pub total_momentum_change_frequency: f64,
    pub total_throttle_change_frequency: f64,
    pub total_braking_frequency: f64,
    pub total_distance_from_wall: f64,
    pub total_distance_from_center: f64,
    pub best_lap_time: f64,
    pub idle_ticks: u64,
    pub previous_speed: f64,
    pub previous_throttle: f64,
    pub previous_score: f64,
    pub previous_lap_count: u64,
    pub lap_start_tick: u64,
    pub drift_ticks: u64,
    pub current_slip_angle_degrees: f64,
    pub continuous_drift_time: f64,
    pub max_continuous_drift_time: f64,
    pub all_laps_time_sum: f64,
    pub recent: [f64; RECENT],
    pub recent_start: u32,
    pub recent_len: u32,
    pub deactivated_at: u64,
    pub flags: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct WheelDesc {
    pub position: Vec2,
    pub steering: u32,
    pub handbrake_off: u32,
    pub power: f32,
    pub brake_power: f32,
    pub handbrake_power: f32,
    pub brake_max: f32,
    pub max_angle_deg: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VehicleDesc {
    pub wheels: [WheelDesc; MAX_WHEELS],
    pub wheel_count: u32,
    pub custom_integrator: u32,
    pub center_steering: u32,
    pub solver_iterations: u32,
    pub max_contacts_reported: u32,
    pub pad0: u32,
    pub steering_speed: f64,
    pub max_velocity: f64,
    pub dt: f64,
    pub contact_max_separation: f64,
    pub grip: f32,
    pub air_resistance: f32,
    pub mass: f32,
    pub inertia: f32,
    pub linear_damp: f32,
    pub angular_damp: f32,
    pub dt_f: f32,
    pub max_velocity_f: f32,
    pub gravity: Vec2,
    pub shape_basis_x: Vec2,
    pub shape_basis_y: Vec2,
    pub shape_size: Vec2,
    pub shape_half: Vec2,
    pub shape_position: Vec2,
    pub center_of_mass: Vec2,
    pub friction: f32,
    pub bounce: f32,
    pub contact_bias: f32,
    pub allowed_penetration: f32,
    pub recycle_radius: f32,
    pub max_separation_f: f32,
    pub pad1: f32,
    pub pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SensorDesc {
    pub kind: u32,
    pub a: f32,
    pub b: f32,
    pub offset: Vec2,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TileCell {
    pub surface: u32,
    pub connected: u32,
    pub first: Vec2,
    pub last: Vec2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ShapeDev {
    pub origin: Vec2,
    pub friction: f32,
    pub bounce: f32,
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub first: u32,
    pub count: u32,
    pub wall_first: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NearGridDesc {
    pub x0: f64,
    pub y0: f64,
    pub cell: f64,
    pub nx: u32,
    pub ny: u32,
    pub start: *const u32,
    pub items: *const u32,
}

/// `World` of gpu/sim/world.h with host pointers.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WorldDesc {
    pub ray_node_count: u32,
    pub ray_wall_count: u32,
    pub ray_depth: u32,
    pub ray_magnitude: f32,
    pub ray_nodes: *const crate::gpu::RayNode,
    pub ray_walls: *const [f32; 4],
    pub surface_count: u32,
    pub default_surface: u32,
    pub surfaces: *const [f32; 4],
    pub tile_x0: i64,
    pub tile_y0: i64,
    pub tile_nx: i64,
    pub tile_ny: i64,
    pub tiles: *const TileCell,
    pub path_points: u32,
    pub pad0: u32,
    pub path_segments: *const [f32; 8],
    pub path_offsets: *const f64,
    pub path_grid: NearGridDesc,
    pub curve_points: u32,
    pub curve_length: f32,
    pub curve_position: *const Vec2,
    pub curve_offsets: *const f32,
    pub curve_forwards: *const Vec2,
    pub curve_grid: NearGridDesc,
    pub shape_count: u32,
    pub shape_point_count: u32,
    pub shapes: *const ShapeDev,
    pub shape_local: *const Vec2,
    pub shape_normals: *const Vec2,
    pub shape_grid: NearGridDesc,
    pub shape_margin: f64,
    pub path_xy: *const f64,
}

/// Checks the Rust mirrors against the library's struct layout.
pub fn check_layout(gpu: &Gpu) {
    use std::mem::{offset_of, size_of};
    let f: unsafe extern "C" fn(*mut u64, i32) -> i32 = gpu.symbol("altd_gpu_layout");
    let mut got = [0u64; 32];
    let count = unsafe { f(got.as_mut_ptr(), got.len() as i32) } as usize;
    let want = [
        size_of::<WorldDesc>(), size_of::<GpuCar>(), size_of::<GpuContact>(), size_of::<GpuAgent>(),
        size_of::<VehicleDesc>(), size_of::<SensorDesc>(), size_of::<WheelDesc>(), size_of::<ShapeDev>(),
        size_of::<TileCell>(), size_of::<[f32; 8]>(), size_of::<NearGridDesc>(),
        offset_of!(WorldDesc, tiles), offset_of!(WorldDesc, curve_grid), offset_of!(WorldDesc, shape_grid),
        offset_of!(GpuCar, tick), offset_of!(GpuCar, pairs), offset_of!(GpuCar, contacts), offset_of!(GpuAgent, recent),
        offset_of!(VehicleDesc, steering_speed), offset_of!(VehicleDesc, gravity),
    ];
    assert_eq!(count, want.len(), "libaltd_gpu.so layout table differs; rebuild it with gpu/build.sh");
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w as u64, "GPU struct layout entry {i} differs (library {g}, Rust {w})");
    }
}

// ---- exactness helpers ----

pub(crate) fn exact(v: f64, what: &str) -> Result<f32, String> {
    let f = v as f32;
    if (f as f64).to_bits() == v.to_bits() || v.is_nan() {
        Ok(f)
    } else {
        Err(format!("{what} = {v:e} is not float32-exact"))
    }
}

pub(crate) fn exact2(v: V2, what: &str) -> Result<Vec2, String> {
    Ok([exact(v.x, what)?, exact(v.y, what)?])
}

pub(crate) fn f2(v: F2) -> Vec2 {
    [v.x, v.y]
}

pub(crate) fn v2(v: Vec2) -> V2 {
    V2::new(v[0] as f64, v[1] as f64)
}

// ---- surfaces ----

/// Vehicle surfaces by GPU index: 0 is `ASPHALT` (the missing-name default),
/// then the configured surfaces in name order.
pub struct SurfaceTable {
    pub names: Vec<String>,
    pub values: Vec<Surface>,
}

impl SurfaceTable {
    pub fn new(vehicle: &VehicleConfig) -> SurfaceTable {
        let mut names: Vec<String> = vehicle.surfaces.keys().cloned().collect();
        names.sort();
        let mut values = vec![ASPHALT];
        values.extend(names.iter().map(|n| vehicle.surfaces[n]));
        SurfaceTable { names, values }
    }

    /// Index of `vehicle.surface(name)`: the first one holding its value, as
    /// `by_value` (the CPU keeps surfaces by value, e.g. a configured
    /// "asphalt" equal to the default).
    pub fn by_name(&self, name: &str) -> u32 {
        let named = self.names.iter().position(|n| n == name).map_or(0, |i| i + 1);
        self.values.iter().position(|&s| s == self.values[named]).unwrap_or(named) as u32
    }

    /// First index holding `surface`.
    pub fn by_value(&self, surface: Surface) -> Result<u32, String> {
        self.values.iter().position(|&s| s == surface).map(|i| i as u32).ok_or_else(|| format!("unknown surface {surface:?}"))
    }
}

// ---- nearest-segment candidate grid ----

/// Cell lists for exact nearest-segment searches on the GPU. For every
/// point of a cell (grown by one unit against index rounding), the list
/// holds, in ascending order, every segment whose distance can come within
/// the float32 slack of the nearest one; see `near_grid`.
pub struct NearGrid {
    pub x0: f64,
    pub y0: f64,
    pub cell: f64,
    pub nx: u32,
    pub ny: u32,
    pub start: Vec<u32>,
    pub items: Vec<u32>,
}

fn segment_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let l2 = d[0] * d[0] + d[1] * d[1];
    let t = if l2 > 0.0 { (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / l2).clamp(0.0, 1.0) } else { 0.0 };
    let q = [a[0] + d[0] * t - p[0], a[1] + d[1] * t - p[1]];
    (q[0] * q[0] + q[1] * q[1]).sqrt()
}

/// Builds the grid over the segments' bounding box grown by `margin`.
/// For a cell box C, `bound = min_s max_{corner c} |c - s|` bounds the
/// nearest distance of every point of C (distance to a segment is convex);
/// segment s is listed if the distance from C to its bounding box is at most
/// `bound` plus the slack. The GPU's float32 distances differ from the exact
/// ones by far less than the slack `1 + 1e-4 * magnitude` (as in
/// segment_grid), so no segment that can tie or beat the nearest is dropped.
pub fn near_grid(segments: &[([f64; 2], [f64; 2])], cell: f64, margin: f64) -> Option<NearGrid> {
    if segments.is_empty() || segments.iter().any(|(a, b)| !(a[0].is_finite() && a[1].is_finite() && b[0].is_finite() && b[1].is_finite())) {
        return None;
    }
    let boxes: Vec<[f64; 4]> =
        segments.iter().map(|(a, b)| [a[0].min(b[0]), a[1].min(b[1]), a[0].max(b[0]), a[1].max(b[1])]).collect();
    let lo = boxes.iter().fold([f64::INFINITY; 2], |l, b| [l[0].min(b[0]), l[1].min(b[1])]);
    let hi = boxes.iter().fold([f64::NEG_INFINITY; 2], |h, b| [h[0].max(b[2]), h[1].max(b[3])]);
    let (x0, y0) = (lo[0] - margin, lo[1] - margin);
    let nx = ((hi[0] + margin - x0) / cell).ceil() as u32;
    let ny = ((hi[1] + margin - y0) / cell).ceil() as u32;
    let magnitude = [x0, y0, x0 + nx as f64 * cell, y0 + ny as f64 * cell].iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let slack = 1.0 + magnitude * 1e-4;
    let lists: Vec<Vec<u32>> = (0..nx as usize * ny as usize)
        .into_par_iter()
        .map(|c| {
            let (cx, cy) = ((c % nx as usize) as f64, (c / nx as usize) as f64);
            let bx = [x0 + cx * cell - 1.0, y0 + cy * cell - 1.0, x0 + (cx + 1.0) * cell + 1.0, y0 + (cy + 1.0) * cell + 1.0];
            let corners = [[bx[0], bx[1]], [bx[2], bx[1]], [bx[0], bx[3]], [bx[2], bx[3]]];
            let bound = segments
                .iter()
                .map(|&(a, b)| corners.iter().map(|&p| segment_distance(p, a, b)).fold(0.0f64, f64::max))
                .fold(f64::INFINITY, f64::min);
            let limit = bound * (1.0 + 1e-4) + slack;
            (0..segments.len() as u32)
                .filter(|&i| {
                    let b = &boxes[i as usize];
                    let dx = (b[0] - bx[2]).max(bx[0] - b[2]).max(0.0);
                    let dy = (b[1] - bx[3]).max(bx[1] - b[3]).max(0.0);
                    (dx * dx + dy * dy).sqrt() <= limit
                })
                .collect()
        })
        .collect();
    let mut start = Vec::with_capacity(lists.len() + 1);
    let mut items = Vec::new();
    start.push(0u32);
    for list in &lists {
        items.extend_from_slice(list);
        start.push(items.len() as u32);
    }
    Some(NearGrid { x0, y0, cell, nx, ny, start, items })
}

pub const SHAPE_CELL: f64 = 256.0;
pub const SHAPE_MARGIN: f64 = 128.0;

/// Physics shape lists for the broad-pair update (gpu/sim/step.h
/// update_pairs): shape i is listed, ascending, in every cell that its box
/// grown by one unit meets after growing the cell by `margin`. A query box
/// starting in a cell and ending before its far edges plus `margin` finds
/// every overlapping shape there; other queries scan all shapes.
pub fn shape_grid(boxes: &[([f64; 2], [f64; 2])], cell: f64, margin: f64) -> Option<NearGrid> {
    if boxes.is_empty() || boxes.iter().any(|(a, b)| !(a[0].is_finite() && a[1].is_finite() && b[0].is_finite() && b[1].is_finite())) {
        return None;
    }
    let lo = boxes.iter().fold([f64::INFINITY; 2], |l, (a, _)| [l[0].min(a[0]), l[1].min(a[1])]);
    let hi = boxes.iter().fold([f64::NEG_INFINITY; 2], |h, (_, b)| [h[0].max(b[0]), h[1].max(b[1])]);
    let (x0, y0) = (lo[0] - margin, lo[1] - margin);
    let nx = ((hi[0] + margin - x0) / cell).ceil().max(1.0) as u32;
    let ny = ((hi[1] + margin - y0) / cell).ceil().max(1.0) as u32;
    let mut start = vec![0u32];
    let mut items = Vec::new();
    for cy in 0..ny {
        for cx in 0..nx {
            let c = [x0 + cx as f64 * cell - margin, y0 + cy as f64 * cell - margin,
                     x0 + (cx + 1) as f64 * cell + margin, y0 + (cy + 1) as f64 * cell + margin];
            items.extend((0..boxes.len() as u32).filter(|&i| {
                let (a, b) = boxes[i as usize];
                a[0] - 1.0 <= c[2] && b[0] + 1.0 >= c[0] && a[1] - 1.0 <= c[3] && b[1] + 1.0 >= c[1]
            }));
            start.push(items.len() as u32);
        }
    }
    Some(NearGrid { x0, y0, cell, nx, ny, start, items })
}

impl NearGrid {
    fn desc(grid: Option<&NearGrid>) -> NearGridDesc {
        match grid {
            Some(g) => NearGridDesc { x0: g.x0, y0: g.y0, cell: g.cell, nx: g.nx, ny: g.ny, start: g.start.as_ptr(), items: g.items.as_ptr() },
            None => NearGridDesc { x0: 0.0, y0: 0.0, cell: 1.0, nx: 0, ny: 0, start: std::ptr::null(), items: std::ptr::null() },
        }
    }

    /// (cells, mean list length, longest list)
    pub fn stats(&self) -> (usize, f64, usize) {
        let cells = self.start.len() - 1;
        let longest = self.start.windows(2).map(|w| (w[1] - w[0]) as usize).max().unwrap_or(0);
        (cells, self.items.len() as f64 / cells.max(1) as f64, longest)
    }
}

pub const GRID_CELL: f64 = 64.0;
pub const GRID_MARGIN: f64 = 1024.0;

// ---- track export ----

/// Owned host arrays behind a `WorldDesc`.
pub struct TrackArrays {
    pub surfaces: SurfaceTable,
    surface_rows: Vec<[f32; 4]>,
    default_surface: u32,
    tiles: (i64, i64, i64, i64, Vec<TileCell>),
    path_rows: Vec<[f32; 8]>,
    path_offsets: Vec<f64>,
    path_xy: Vec<f64>,
    pub path_grid: Option<NearGrid>,
    curve_position: Vec<Vec2>,
    curve_offsets: Vec<f32>,
    curve_forwards: Vec<Vec2>,
    curve_length: f32,
    pub curve_grid: Option<NearGrid>,
    shapes: Vec<ShapeDev>,
    shape_local: Vec<Vec2>,
    shape_normals: Vec<Vec2>,
    pub shape_grid: Option<NearGrid>,
}

impl TrackArrays {
    pub fn new(world: &World) -> Result<TrackArrays, String> {
        let track = &world.track;
        let surfaces = SurfaceTable::new(&world.vehicle);
        let surface_rows = surfaces.values.iter().map(|s| [s.grip as f32, s.power as f32, s.steering as f32, 0.0]).collect();
        let default_surface = surfaces.by_name("asphalt");
        let (x0, y0, nx, ny, cells) = track.gpu_tiles();
        let tiles = cells
            .into_iter()
            .map(|(name, connection)| {
                let surface = name.map_or(default_surface, |n| surfaces.by_name(&n));
                let (connected, first, last) = match connection {
                    Some((a, b)) => (1, f2(F2::from(a)), f2(F2::from(b))),
                    None => (0, [0.0; 2], [0.0; 2]),
                };
                TileCell { surface, connected, first, last }
            })
            .collect();
        let path_rows: Vec<[f32; 8]> = track.gpu_path_rows().into_iter().map(|r| [r[0], r[1], r[2], r[3], r[4], 0.0, 0.0, 0.0]).collect();
        let path_offsets = track.path_offsets.clone();
        if track.path.len() != path_offsets.len() {
            return Err("path points and offsets differ in length".into());
        }
        let path_xy = track.path.iter().flat_map(|p| [p.x, p.y]).collect();
        if path_offsets.windows(2).any(|w| !(w[1] > w[0])) {
            return Err("GPU closest_path needs strictly increasing path offsets".into());
        }
        let path_grid = near_grid(
            &path_rows.iter().map(|r| ([r[0] as f64, r[1] as f64], [(r[0] + r[2]) as f64, (r[1] + r[3]) as f64])).collect::<Vec<_>>(),
            GRID_CELL,
            GRID_MARGIN,
        );
        let (curve_position, curve_offsets, curve_forwards, curve_length, curve_grid) = match &track.curve {
            Some(c) => {
                let points: Vec<Vec2> = c.points.iter().map(|&p| f2(p)).collect();
                let segments: Vec<_> =
                    points.windows(2).map(|w| ([w[0][0] as f64, w[0][1] as f64], [w[1][0] as f64, w[1][1] as f64])).collect();
                (points, c.offsets.clone(), c.forwards.iter().map(|&p| f2(p)).collect(), c.length(), near_grid(&segments, GRID_CELL, GRID_MARGIN))
            }
            None => (Vec::new(), Vec::new(), Vec::new(), 0.0, None),
        };
        let mut shape_local = Vec::new();
        let mut shape_normals = Vec::new();
        let mut shapes = Vec::new();
        for s in &track.shapes {
            if s.local.len() > MAX_SHAPE_POINTS {
                return Err(format!("shape with {} points exceeds MAX_SHAPE_POINTS", s.local.len()));
            }
            let first = shape_local.len() as u32;
            for (p, n) in s.local.iter().zip(&s.local_normals) {
                shape_local.push(exact2(*p, "shape local point")?);
                shape_normals.push(exact2(*n, "shape normal")?);
            }
            shapes.push(ShapeDev {
                origin: exact2(s.origin_f32, "shape origin")?,
                friction: s.friction as f32,
                bounce: s.bounce as f32,
                min_x: s.min.x,
                min_y: s.min.y,
                max_x: s.max.x,
                max_y: s.max.y,
                first,
                count: s.local.len() as u32,
                wall_first: s.wall_first as u32,
                pad: 0,
            });
        }
        let shape_boxes: Vec<_> = track.shapes.iter().map(|s| ([s.min.x, s.min.y], [s.max.x, s.max.y])).collect();
        let shape_grid = shape_grid(&shape_boxes, SHAPE_CELL, SHAPE_MARGIN);
        Ok(TrackArrays {
            surfaces,
            surface_rows,
            default_surface,
            tiles: (x0, y0, nx, ny, tiles),
            path_rows,
            path_offsets,
            path_xy,
            path_grid,
            curve_position,
            curve_offsets,
            curve_forwards,
            curve_length,
            curve_grid,
            shapes,
            shape_local,
            shape_normals,
            shape_grid,
        })
    }

    /// The descriptor; `rays` are the BSP arrays of `bsp::RayTree::gpu_arrays`.
    pub(crate) fn desc(&self, rays: &(Vec<crate::gpu::RayNode>, Vec<[f32; 4]>, f32, usize)) -> WorldDesc {
        let (x0, y0, nx, ny, tiles) = &self.tiles;
        WorldDesc {
            ray_node_count: rays.0.len() as u32,
            ray_wall_count: rays.1.len() as u32,
            ray_depth: rays.3 as u32,
            ray_magnitude: rays.2,
            ray_nodes: rays.0.as_ptr(),
            ray_walls: rays.1.as_ptr(),
            surface_count: self.surface_rows.len() as u32,
            default_surface: self.default_surface,
            surfaces: self.surface_rows.as_ptr(),
            tile_x0: *x0,
            tile_y0: *y0,
            tile_nx: *nx,
            tile_ny: *ny,
            tiles: tiles.as_ptr(),
            path_points: self.path_offsets.len() as u32,
            pad0: 0,
            path_segments: self.path_rows.as_ptr(),
            path_offsets: self.path_offsets.as_ptr(),
            path_grid: NearGrid::desc(self.path_grid.as_ref()),
            curve_points: self.curve_position.len() as u32,
            curve_length: self.curve_length,
            curve_position: self.curve_position.as_ptr(),
            curve_offsets: self.curve_offsets.as_ptr(),
            curve_forwards: self.curve_forwards.as_ptr(),
            curve_grid: NearGrid::desc(self.curve_grid.as_ref()),
            shape_count: self.shapes.len() as u32,
            shape_point_count: self.shape_local.len() as u32,
            shapes: self.shapes.as_ptr(),
            shape_local: self.shape_local.as_ptr(),
            shape_normals: self.shape_normals.as_ptr(),
            shape_grid: NearGrid::desc(self.shape_grid.as_ref()),
            shape_margin: SHAPE_MARGIN,
            path_xy: self.path_xy.as_ptr(),
        }
    }
}

// ---- vehicle and sensors ----

pub fn vehicle_desc(cfg: &VehicleConfig) -> Result<VehicleDesc, String> {
    if cfg.wheels.len() > MAX_WHEELS {
        return Err(format!("{} wheels exceed MAX_WHEELS", cfg.wheels.len()));
    }
    let mut d = VehicleDesc::default();
    for (w, spec) in d.wheels.iter_mut().zip(&cfg.wheels) {
        *w = WheelDesc {
            position: f2(F2::from(spec.position)),
            steering: spec.steering as u32,
            handbrake_off: (spec.handbrake_power.abs() < 0.00001) as u32,
            power: spec.power as f32,
            brake_power: spec.brake_power as f32,
            handbrake_power: spec.handbrake_power as f32,
            brake_max: spec.brake_power.max(spec.handbrake_power) as f32,
            max_angle_deg: spec.max_angle_deg,
        };
    }
    d.wheel_count = cfg.wheels.len() as u32;
    d.custom_integrator = cfg.custom_integrator as u32;
    d.center_steering = cfg.center_steering as u32;
    d.solver_iterations = cfg.solver_iterations as u32;
    d.max_contacts_reported = cfg.max_contacts_reported as u32;
    d.steering_speed = cfg.steering_speed;
    d.max_velocity = cfg.max_velocity;
    d.dt = DT;
    d.contact_max_separation = cfg.contact_max_separation;
    d.grip = cfg.grip as f32;
    d.air_resistance = cfg.air_resistance as f32;
    d.mass = cfg.mass as f32;
    d.inertia = cfg.inertia as f32;
    d.linear_damp = cfg.linear_damp as f32;
    d.angular_damp = cfg.angular_damp as f32;
    d.dt_f = DT as f32;
    d.max_velocity_f = cfg.max_velocity as f32;
    d.gravity = f2(F2::from(cfg.gravity));
    d.shape_basis_x = f2(cfg.shape_basis.0);
    d.shape_basis_y = f2(cfg.shape_basis.1);
    d.shape_size = f2(F2::from(cfg.shape_size));
    d.shape_half = [(cfg.shape_size.x / 2.0) as f32, (cfg.shape_size.y / 2.0) as f32];
    d.shape_position = f2(F2::from(cfg.shape_position));
    d.center_of_mass = f2(F2::from(cfg.center_of_mass));
    d.friction = cfg.friction as f32;
    d.bounce = cfg.bounce as f32;
    d.contact_bias = cfg.contact_bias as f32;
    d.allowed_penetration = cfg.allowed_penetration as f32;
    d.recycle_radius = cfg.contact_recycle_radius as f32;
    d.max_separation_f = cfg.contact_max_separation as f32;
    Ok(d)
}

pub fn sensor_desc(sensor: &Sensor) -> SensorDesc {
    let d = |kind: u32, a: f64, b: f64| SensorDesc { kind, a: a as f32, b: b as f32, ..Default::default() };
    match *sensor {
        Sensor::Raycast { degrees, length } => {
            assert!(length > 0.0, "ray length must be positive");
            d(0, degrees, length)
        }
        Sensor::DistanceFromWall { max_distance } => d(1, max_distance, 0.0),
        Sensor::Speed => d(2, 0.0, 0.0),
        Sensor::VelocityFront => d(3, 0.0, 0.0),
        Sensor::VelocitySide => d(4, 0.0, 0.0),
        Sensor::AccelerationFront { max_acceleration } => d(5, max_acceleration, 0.0),
        Sensor::AccelerationSide { max_acceleration } => d(6, max_acceleration, 0.0),
        Sensor::AngularVelocity => d(7, 0.0, 0.0),
        Sensor::WheelAngle => d(8, 0.0, 0.0),
        Sensor::BoostCapacity => d(9, 0.0, 0.0),
        Sensor::Grip { offset } => SensorDesc { kind: 10, offset: f2(F2::from(offset)), ..Default::default() },
        Sensor::CorrectDirection => d(11, 0.0, 0.0),
        Sensor::TrackCurvature { min_lookahead, max_lookahead } => d(12, min_lookahead, max_lookahead),
    }
}

// ---- simulation handle ----

/// `ReadMask` of gpu/sim/sensors.h: the cars whose controls update this tick.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReadMask {
    pub batch_size: u32,
    pub batch_index: u32,
    pub batches_per_tick: u32,
    pub require_active: u32,
}

impl ReadMask {
    /// Every car, active or not.
    pub const ALL: ReadMask = ReadMask { batch_size: u32::MAX, batch_index: 0, batches_per_tick: 8, require_active: 0 };
}

/// GPU control order of `GpuAgent::controls`.
pub const CONTROL_NAMES: [&str; 5] = ["acceleration", "steering", "brake", "handbrake", "boost"];

/// For each control, the last output neuron with its name (-1: none, 0.0),
/// as `Controls(**dict(zip(names, outputs)))` assigns them.
pub fn control_sources(outputs: &[String]) -> [i32; 5] {
    CONTROL_NAMES.map(|c| outputs.iter().rposition(|o| o.to_lowercase() == c).map_or(-1, |j| j as i32))
}

impl ReadMask {
    /// `Schedule::update_controls` for `cars` cars at `batch_index`.
    pub fn training(cars: usize, batch_index: usize, batches_per_tick: usize) -> ReadMask {
        ReadMask {
            batch_size: cars.div_ceil(8).max(1) as u32,
            batch_index: batch_index as u32,
            batches_per_tick: batches_per_tick as u32,
            require_active: 1,
        }
    }
}

pub struct GpuSim<'a> {
    gpu: &'a Gpu,
    handle: *mut c_void,
    pub sensor_count: usize,
    pub cars: usize,
    /// Caller-chosen tag of the uploaded networks (e.g. the generation), so
    /// unchanged networks are not uploaded again.
    pub network_tag: Option<u64>,
    network_count: usize,
    parameter_buffer: Vec<f64>,
}

/// Arguments of `altd_gpu_sim_window` (gpu/sim/altd_gpu.hip).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct WindowArgs {
    pub start_tick: u64,
    pub ticks: u64,
    pub start_batch: u32,
    pub batches_per_tick: u32,
    pub stats_phase: u32,
    pub stop_when_inactive: u32,
    pub eliminate_on_wall: u32,
    pub eliminate_when_idle: u32,
    pub has_time_limit: u32,
    pub pad: u32,
    pub time_limit: f64,
}

impl Drop for GpuSim<'_> {
    fn drop(&mut self) {
        let free: unsafe extern "C" fn(*mut c_void) = self.gpu.symbol("altd_gpu_sim_free");
        unsafe { free(self.handle) };
    }
}

fn check(status: i32, what: &str) {
    assert_eq!(status, 0, "GPU {what} failed with status {status}");
}

impl<'a> GpuSim<'a> {
    pub fn new(world: &GpuWorld<'a>, vehicle: &VehicleDesc, sensors: &[SensorDesc], capacity: usize) -> GpuSim<'a> {
        let gpu = world.gpu();
        let create: unsafe extern "C" fn(*const c_void, *const VehicleDesc, *const SensorDesc, u32, u32) -> *mut c_void =
            gpu.symbol("altd_gpu_sim_create");
        assert!(sensors.len() <= MAX_SENSORS);
        let handle = unsafe { create(world.handle(), vehicle, sensors.as_ptr(), sensors.len() as u32, capacity as u32) };
        assert!(!handle.is_null(), "altd_gpu_sim_create failed");
        GpuSim { gpu, handle, sensor_count: sensors.len(), cars: 0, network_tag: None, network_count: 0, parameter_buffer: Vec::new() }
    }

    pub fn upload(&mut self, cars: &[GpuCar], agents: Option<&[GpuAgent]>) {
        let f: unsafe extern "C" fn(*mut c_void, u32, *const GpuCar, *const GpuAgent) -> i32 = self.gpu.symbol("altd_gpu_sim_upload");
        if let Some(a) = agents {
            assert_eq!(a.len(), cars.len());
        }
        check(unsafe { f(self.handle, cars.len() as u32, cars.as_ptr(), agents.map_or(std::ptr::null(), |a| a.as_ptr())) }, "upload");
        self.cars = cars.len();
    }

    /// A training window over the uploaded cars (`TrainingRunner::advance_window`
    /// order); returns (executed ticks, transition_without_drive).
    pub fn window(&self, args: &WindowArgs) -> (u64, bool) {
        let f: unsafe extern "C" fn(*mut c_void, *const WindowArgs, *mut u64, *mut u32) -> i32 = self.gpu.symbol("altd_gpu_sim_window");
        let (mut executed, mut transition) = (0u64, 0u32);
        check(unsafe { f(self.handle, args, &mut executed, &mut transition) }, "window");
        (executed, transition != 0)
    }

    /// Replaces the agents of the uploaded cars.
    pub fn upload_agents(&mut self, agents: &[GpuAgent]) {
        let f: unsafe extern "C" fn(*mut c_void, u32, *const GpuAgent) -> i32 = self.gpu.symbol("altd_gpu_sim_upload_agents");
        check(unsafe { f(self.handle, agents.len() as u32, agents.as_ptr()) }, "upload_agents");
    }

    pub fn download(&self, cars: Option<&mut Vec<GpuCar>>, agents: Option<&mut Vec<GpuAgent>>) {
        let f: unsafe extern "C" fn(*mut c_void, *mut GpuCar, *mut GpuAgent) -> i32 = self.gpu.symbol("altd_gpu_sim_download");
        let cars = cars.map(|c| {
            c.resize(self.cars, GpuCar::zeroed());
            c.as_mut_ptr()
        });
        let agents = agents.map(|a| {
            a.resize(self.cars, GpuAgent::default());
            a.as_mut_ptr()
        });
        check(unsafe { f(self.handle, cars.unwrap_or(std::ptr::null_mut()), agents.unwrap_or(std::ptr::null_mut())) }, "download");
    }

    /// One network per car: `params` holds `networks` flat parameter vectors of `shape`.
    pub fn networks(&mut self, shape: &[usize], params: &[f64], control_src: [i32; 5]) {
        let f: unsafe extern "C" fn(*mut c_void, u32, *const u32, u32, *const f64, *const i32) -> i32 =
            self.gpu.symbol("altd_gpu_sim_networks");
        let widths: Vec<u32> = shape.iter().map(|&w| w as u32).collect();
        let size = crate::network::parameter_count(shape);
        assert_eq!(params.len() % size, 0);
        let status = unsafe { f(self.handle, widths.len() as u32, widths.as_ptr(), (params.len() / size) as u32, params.as_ptr(), control_src.as_ptr()) };
        check(status, "networks");
        self.network_count = params.len() / size;
    }

    pub fn network_count(&self) -> usize { self.network_count }

    pub(crate) fn take_parameter_buffer(&mut self) -> Vec<f64> { std::mem::take(&mut self.parameter_buffer) }
    pub(crate) fn return_parameter_buffer(&mut self, buffer: Vec<f64>) { self.parameter_buffer = buffer; }

    /// Reuse a flat upload buffer, filling disjoint networks in parallel.
    pub fn upload_networks(&mut self, networks: &[crate::network::Network], control_src: [i32; 5]) -> Result<(), String> {
        self.upload_network_refs(&networks.iter().collect::<Vec<_>>(), control_src)
    }

    pub fn upload_agent_networks(&mut self, agents: &[crate::training::TrainingAgent], control_src: [i32; 5]) -> Result<(), String> {
        self.upload_network_refs(&agents.iter().map(|a| &a.network).collect::<Vec<_>>(), control_src)
    }

    fn upload_network_refs(&mut self, networks: &[&crate::network::Network], control_src: [i32; 5]) -> Result<(), String> {
        use rayon::prelude::*;
        let first = networks.first().ok_or("no networks")?;
        let size = crate::network::parameter_count(&first.shape);
        if networks.iter().any(|n| n.shape != first.shape || n.params.len() != size) {
            return Err("networks of different shapes or invalid parameter counts".into());
        }
        let mut profile = crate::training_profile::Profile::new("network_upload");
        let mut params = std::mem::take(&mut self.parameter_buffer);
        params.resize(networks.len().checked_mul(size).ok_or("network parameter count overflow")?, 0.0);
        params.par_chunks_mut(size).zip(networks.par_iter()).for_each(|(out, network)| out.copy_from_slice(&network.params));
        profile.mark("pack");
        self.networks(&first.shape, &params, control_src);
        profile.mark("upload");
        self.parameter_buffer = params;
        Ok(())
    }

    /// Exact population novelty for the currently uploaded networks.
    pub fn novelty(&self) -> Vec<f64> {
        let f: unsafe extern "C" fn(*mut c_void, u32, *mut f64) -> i32 = self.gpu.symbol("altd_gpu_sim_novelty");
        let mut out = vec![0.0; self.network_count];
        check(unsafe { f(self.handle, self.network_count as u32, out.as_mut_ptr()) }, "population novelty");
        out
    }

    /// Sensors and network forward for the cars in `mask` (sets their controls).
    pub fn infer(&self, mask: ReadMask) {
        let f: unsafe extern "C" fn(*mut c_void, ReadMask) -> i32 = self.gpu.symbol("altd_gpu_sim_infer");
        check(unsafe { f(self.handle, mask) }, "infer");
    }

    /// One physics tick for every car with its agent's controls (`Car::step`
    /// if `drive`, else `Car::passive_step`); a reported wall contact sets
    /// AGENT_PENDING_CONTACT.
    /// A car the step deactivates records `tick` in `deactivated_at`.
    pub fn step(&self, drive: bool, eliminate_on_wall: bool, tick: u64) {
        let f: unsafe extern "C" fn(*mut c_void, u32, u32, u64) -> i32 = self.gpu.symbol("altd_gpu_sim_step");
        check(unsafe { f(self.handle, drive as u32, eliminate_on_wall as u32, tick) }, "step");
    }

    /// `Schedule::update_stats` for every car at statistics tick `tick`;
    /// `stats_tick` is `tick - first_stats_tick`.
    pub fn stats(&self, tick: u64, stats_tick: u64, eliminate_when_idle: bool) {
        let f: unsafe extern "C" fn(*mut c_void, u64, u64, u32) -> i32 = self.gpu.symbol("altd_gpu_sim_stats");
        check(unsafe { f(self.handle, tick, stats_tick, eliminate_when_idle as u32) }, "stats");
    }

    /// Reads the sensors of the cars in `mask`; returns all cars' inputs, car-major.
    pub fn sensors(&self, mask: ReadMask) -> Vec<f64> {
        let f: unsafe extern "C" fn(*mut c_void, ReadMask, *mut f64) -> i32 = self.gpu.symbol("altd_gpu_sim_sensors");
        let mut out = vec![0.0; self.cars * self.sensor_count];
        check(unsafe { f(self.handle, mask, out.as_mut_ptr()) }, "sensors");
        out
    }
}

// ---- agents ----

/// The GPU agent state of a training agent (controls, statistics, window
/// bookkeeping); errors on states the GPU port does not model.
pub fn agent_export(a: &crate::training::TrainingAgent) -> Result<GpuAgent, String> {
    let s = &a.stats;
    let t = &s.score;
    if s.recent_score_diffs.len() > RECENT {
        return Err("more than RECENT score diffs".into());
    }
    let c = &a.controls;
    let mut g = GpuAgent {
        controls: [c.acceleration, c.steering, c.brake, c.handbrake, c.boost],
        previous_offset: t.previous_offset.unwrap_or(0.0),
        tracker_score: t.total_score,
        lap_crossing_fraction: t.lap_crossing_fraction.unwrap_or(0.0),
        lap_count: t.lap_count,
        network_novelty: s.network_novelty,
        update_count: s.update_count,
        collision_count: s.collision_count,
        total_score: s.total_score,
        total_speed: s.total_speed,
        total_abs_steering: s.total_abs_steering,
        total_actual_distance: s.total_actual_distance,
        total_drifting_amount: s.total_drifting_amount,
        total_momentum_change_frequency: s.total_momentum_change_frequency,
        total_throttle_change_frequency: s.total_throttle_change_frequency,
        total_braking_frequency: s.total_braking_frequency,
        total_distance_from_wall: s.total_distance_from_wall,
        total_distance_from_center: s.total_distance_from_center,
        best_lap_time: s.best_lap_time.unwrap_or(0.0),
        idle_ticks: s.idle_ticks,
        previous_speed: s.previous_speed,
        previous_throttle: s.previous_throttle,
        previous_score: s.previous_score,
        previous_lap_count: s.previous_lap_count,
        lap_start_tick: s.lap_start_tick,
        drift_ticks: s.drift_ticks,
        current_slip_angle_degrees: s.current_slip_angle_degrees,
        continuous_drift_time: s.continuous_drift_time,
        max_continuous_drift_time: s.max_continuous_drift_time,
        all_laps_time_sum: s.all_laps_time_sum.unwrap_or(0.0),
        recent: [0.0; RECENT],
        recent_start: 0,
        recent_len: s.recent_score_diffs.len() as u32,
        deactivated_at: a.deactivated_at.unwrap_or(0),
        flags: 0,
        pad: 0,
    };
    for (slot, &d) in g.recent.iter_mut().zip(&s.recent_score_diffs) {
        *slot = d;
    }
    for (on, bit) in [
        (t.previous_offset.is_some(), AGENT_HAS_PREVIOUS_OFFSET),
        (t.went_backwards, AGENT_WENT_BACKWARDS),
        (t.lap_crossing_fraction.is_some(), AGENT_HAS_LAP_CROSSING),
        (a.pending_contact, AGENT_PENDING_CONTACT),
        (s.best_lap_time.is_some(), AGENT_HAS_BEST_LAP),
        (s.all_laps_time_sum.is_some(), AGENT_HAS_ALL_LAPS),
        (a.deactivated_at.is_some(), AGENT_HAS_DEACTIVATED),
    ] {
        g.flags |= if on { bit } else { 0 };
    }
    Ok(g)
}

/// The inverse of `agent_export` (the network and car are left alone).
pub fn agent_import(g: &GpuAgent, a: &mut crate::training::TrainingAgent) {
    let on = |bit: u32| g.flags & bit != 0;
    let [acceleration, steering, brake, handbrake, boost] = g.controls;
    a.controls = crate::car::Controls { acceleration, steering, brake, handbrake, boost };
    a.pending_contact = on(AGENT_PENDING_CONTACT);
    a.deactivated_at = on(AGENT_HAS_DEACTIVATED).then_some(g.deactivated_at);
    let s = &mut a.stats;
    s.score = crate::training::ScoreTracker {
        previous_offset: on(AGENT_HAS_PREVIOUS_OFFSET).then_some(g.previous_offset),
        total_score: g.tracker_score,
        lap_count: g.lap_count,
        went_backwards: on(AGENT_WENT_BACKWARDS),
        lap_crossing_fraction: on(AGENT_HAS_LAP_CROSSING).then_some(g.lap_crossing_fraction),
    };
    s.network_novelty = g.network_novelty;
    s.update_count = g.update_count;
    s.collision_count = g.collision_count;
    s.total_score = g.total_score;
    s.total_speed = g.total_speed;
    s.total_abs_steering = g.total_abs_steering;
    s.total_actual_distance = g.total_actual_distance;
    s.total_drifting_amount = g.total_drifting_amount;
    s.total_momentum_change_frequency = g.total_momentum_change_frequency;
    s.total_throttle_change_frequency = g.total_throttle_change_frequency;
    s.total_braking_frequency = g.total_braking_frequency;
    s.total_distance_from_wall = g.total_distance_from_wall;
    s.total_distance_from_center = g.total_distance_from_center;
    s.best_lap_time = on(AGENT_HAS_BEST_LAP).then_some(g.best_lap_time);
    s.idle_ticks = g.idle_ticks;
    s.previous_speed = g.previous_speed;
    s.previous_throttle = g.previous_throttle;
    s.previous_score = g.previous_score;
    s.previous_lap_count = g.previous_lap_count;
    s.lap_start_tick = g.lap_start_tick;
    s.drift_ticks = g.drift_ticks;
    s.current_slip_angle_degrees = g.current_slip_angle_degrees;
    s.continuous_drift_time = g.continuous_drift_time;
    s.max_continuous_drift_time = g.max_continuous_drift_time;
    s.all_laps_time_sum = on(AGENT_HAS_ALL_LAPS).then_some(g.all_laps_time_sum);
    s.recent_score_diffs.clear();
    s.recent_score_diffs.extend((0..g.recent_len as usize).map(|k| g.recent[(g.recent_start as usize + k) % RECENT]));
}
