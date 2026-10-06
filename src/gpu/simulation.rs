//! Host side of the GPU simulator: #[repr(C)] mirrors of gpu/sim/state.h and
//! gpu/sim/world.h, the exporters that fill them from the CPU structures, and
//! the handle that runs the simulation kernels (libaltd_gpu.so).
#[cfg(not(target_arch = "wasm32"))]
use crate::gpu::hip::{Gpu, GpuWorld};
use crate::math::godot_math::F2;
use crate::math::vec2::V2;
use crate::physics::car::{Sensor, DT};
use crate::track::world::{Surface, VehicleConfig, World, ASPHALT};
use rayon::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use std::ffi::c_void;

/// `RayNode` of gpu/sim/world.h.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct RayNode {
    pub(crate) links: [u32; 4],
    pub(crate) own: [f32; 4],
    pub(crate) subtree: [f32; 4],
}

pub(crate) type Vec2 = [f32; 2];

pub(crate) const MAX_WHEELS: usize = 4;
pub(crate) const MAX_PAIRS: usize = 16;
pub(crate) const MAX_CONTACTS: usize = 32;
pub(crate) const MAX_SHAPE_POINTS: usize = 16;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const MAX_SENSORS: usize = 32;
#[cfg(not(target_arch = "wasm32"))]
const MAX_LAYERS: usize = 12;
#[cfg(not(target_arch = "wasm32"))]
const MAX_WIDTH: usize = 32;
#[cfg(not(target_arch = "wasm32"))]
const LANES: usize = 16;
pub const RECENT: usize = 10;

pub(crate) const CONTACT_WALL_FIRST: u32 = 1;
pub(crate) const CONTACT_USED: u32 = 2;

pub(crate) const CAR_ACTIVE: u32 = 1;
pub(crate) const CAR_RESET_RIGHT: u32 = 2;
pub(crate) const CAR_PENDING_VELOCITY: u32 = 4;
pub(crate) const CAR_HAS_SHAPE: u32 = 8;
pub(crate) const CAR_HAS_LEAF: u32 = 16;
pub(crate) const CAR_PAIR_CHECK: u32 = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuContact {
    pub(crate) normal: Vec2,
    pub(crate) point: Vec2,
    pub(crate) wall_point: Vec2,
    pub(crate) local_point: Vec2,
    pub(crate) wall_local_point: Vec2,
    pub(crate) depth: f32,
    pub(crate) shape: u32,
    pub(crate) flags: u32,
    pub(crate) normal_mass: f32,
    pub(crate) tangent_mass: f32,
    pub(crate) bias: f32,
    pub(crate) bounce: f32,
    pub(crate) friction: f32,
    pub(crate) acc_normal: f32,
    pub(crate) acc_tangent: f32,
    pub(crate) acc_bias: f32,
    pub(crate) acc_bias_center: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuCar {
    pub(crate) position: Vec2,
    pub(crate) velocity: Vec2,
    pub(crate) acceleration: Vec2,
    pub(crate) reset_acceleration: Vec2,
    pub(crate) basis_x: Vec2,
    pub(crate) basis_y: Vec2,
    pub(crate) rotation: f32,
    pub(crate) transform_angle: f32,
    pub(crate) angular_velocity: f32,
    pub(crate) boost_energy: f32,
    pub flags: u32,
    pub(crate) collision_count: u32,
    pub tick: u64,
    pub(crate) wheel_angle: [f32; MAX_WHEELS],
    pub(crate) wheel_previous: [Vec2; MAX_WHEELS],
    pub(crate) wheel_surface: [u32; MAX_WHEELS],
    pub(crate) wheel_has_previous: u32,
    pub pair_count: u32,
    pub contact_count: u32,
    pub error: u32,
    pub(crate) shape_position: Vec2,
    pub(crate) shape_size: Vec2,
    pub(crate) leaf_min: Vec2,
    pub(crate) leaf_max: Vec2,
    pub pairs: [u32; MAX_PAIRS],
    pub axes: [Vec2; MAX_PAIRS],
    pub contacts: [GpuContact; MAX_CONTACTS],
}

impl GpuCar {
    pub(crate) fn zeroed() -> GpuCar {
        // SAFETY: plain old data; all-zero bits are a valid value.
        unsafe { std::mem::zeroed() }
    }
}

pub(crate) const AGENT_HAS_PREVIOUS_OFFSET: u32 = 1;
pub(crate) const AGENT_WENT_BACKWARDS: u32 = 2;
pub(crate) const AGENT_HAS_LAP_CROSSING: u32 = 4;
pub const AGENT_PENDING_CONTACT: u32 = 8;
pub(crate) const AGENT_HAS_BEST_LAP: u32 = 16;
pub(crate) const AGENT_HAS_ALL_LAPS: u32 = 32;
pub(crate) const AGENT_HAS_DEACTIVATED: u32 = 64;

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
pub(crate) struct WheelDesc {
    pub(crate) position: Vec2,
    pub(crate) steering: u32,
    pub(crate) handbrake_off: u32,
    pub(crate) power: f32,
    pub(crate) brake_power: f32,
    pub(crate) handbrake_power: f32,
    pub(crate) brake_max: f32,
    pub(crate) max_angle_deg: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VehicleDesc {
    pub(crate) wheels: [WheelDesc; MAX_WHEELS],
    pub(crate) wheel_count: u32,
    pub(crate) custom_integrator: u32,
    pub(crate) center_steering: u32,
    pub(crate) solver_iterations: u32,
    pub(crate) max_contacts_reported: u32,
    pub(crate) pad0: u32,
    pub(crate) steering_speed: f64,
    pub(crate) max_velocity: f64,
    pub(crate) dt: f64,
    pub(crate) contact_max_separation: f64,
    pub(crate) grip: f32,
    pub(crate) air_resistance: f32,
    pub(crate) mass: f32,
    pub(crate) inertia: f32,
    pub(crate) linear_damp: f32,
    pub(crate) angular_damp: f32,
    pub(crate) dt_f: f32,
    pub(crate) max_velocity_f: f32,
    pub(crate) gravity: Vec2,
    pub(crate) shape_basis_x: Vec2,
    pub(crate) shape_basis_y: Vec2,
    pub(crate) shape_size: Vec2,
    pub(crate) shape_half: Vec2,
    pub(crate) shape_position: Vec2,
    pub(crate) center_of_mass: Vec2,
    pub(crate) friction: f32,
    pub(crate) bounce: f32,
    pub(crate) contact_bias: f32,
    pub(crate) allowed_penetration: f32,
    pub(crate) recycle_radius: f32,
    pub(crate) max_separation_f: f32,
    pub(crate) pad1: f32,
    pub(crate) pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SensorDesc {
    pub(crate) kind: u32,
    pub(crate) a: f32,
    pub(crate) b: f32,
    pub(crate) offset: Vec2,
    pub(crate) pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TileCell {
    pub(crate) surface: u32,
    pub(crate) connected: u32,
    pub(crate) first: Vec2,
    pub(crate) last: Vec2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ShapeDev {
    pub(crate) origin: Vec2,
    pub(crate) friction: f32,
    pub(crate) bounce: f32,
    pub(crate) min_x: f64,
    pub(crate) min_y: f64,
    pub(crate) max_x: f64,
    pub(crate) max_y: f64,
    pub(crate) first: u32,
    pub(crate) count: u32,
    pub(crate) wall_first: u32,
    pub(crate) pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NearGridDesc {
    pub(crate) x0: f64,
    pub(crate) y0: f64,
    pub(crate) cell: f64,
    pub(crate) nx: u32,
    pub(crate) ny: u32,
    pub(crate) start: *const u32,
    pub(crate) items: *const u32,
}

/// `World` of gpu/sim/world.h with host pointers.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct WorldDesc {
    pub(crate) ray_node_count: u32,
    pub(crate) ray_wall_count: u32,
    pub(crate) ray_depth: u32,
    pub(crate) ray_magnitude: f32,
    pub(crate) ray_nodes: *const RayNode,
    pub(crate) ray_walls: *const [f32; 4],
    pub(crate) surface_count: u32,
    pub(crate) default_surface: u32,
    pub(crate) surfaces: *const [f32; 4],
    pub(crate) tile_x0: i64,
    pub(crate) tile_y0: i64,
    pub(crate) tile_nx: i64,
    pub(crate) tile_ny: i64,
    pub(crate) tiles: *const TileCell,
    pub(crate) path_points: u32,
    pub(crate) pad0: u32,
    pub(crate) path_segments: *const [f32; 8],
    pub(crate) path_offsets: *const f64,
    pub(crate) path_grid: NearGridDesc,
    pub(crate) curve_points: u32,
    pub(crate) curve_length: f32,
    pub(crate) curve_position: *const Vec2,
    pub(crate) curve_offsets: *const f32,
    pub(crate) curve_forwards: *const Vec2,
    pub(crate) curve_grid: NearGridDesc,
    pub(crate) shape_count: u32,
    pub(crate) shape_point_count: u32,
    pub(crate) shapes: *const ShapeDev,
    pub(crate) shape_local: *const Vec2,
    pub(crate) shape_normals: *const Vec2,
    pub(crate) shape_grid: NearGridDesc,
    pub(crate) shape_margin: f64,
    pub(crate) path_xy: *const f64,
}

/// The little-endian words of `values`. The GPU mirrors hold only numbers
/// and explicit, initialized padding, so every byte is defined.
pub(crate) fn words<T: Copy>(values: &[T]) -> Vec<u32> {
    assert_eq!(std::mem::size_of::<T>() % 4, 0);
    let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) };
    bytes
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}

/// Checks the Rust mirrors against the library's struct layout.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn check_layout(gpu: &Gpu) {
    if let Err(e) = layout_matches(gpu) {
        panic!("{e}");
    }
}

/// `check_layout` as an error instead of a panic.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn layout_matches(gpu: &Gpu) -> Result<(), String> {
    use std::mem::{offset_of, size_of};
    let f: unsafe extern "C" fn(*mut u64, i32) -> i32 = gpu
        .try_symbol("altd_gpu_layout")
        .ok_or("libaltd_gpu.so lacks altd_gpu_layout; rebuild it with gpu/build.sh or gpu/build-cuda.*")?;
    let mut got = [0u64; 32];
    let count = unsafe { f(got.as_mut_ptr(), got.len() as i32) } as usize;
    let want = [
        size_of::<WorldDesc>(),
        size_of::<GpuCar>(),
        size_of::<GpuContact>(),
        size_of::<GpuAgent>(),
        size_of::<VehicleDesc>(),
        size_of::<SensorDesc>(),
        size_of::<WheelDesc>(),
        size_of::<ShapeDev>(),
        size_of::<TileCell>(),
        size_of::<[f32; 8]>(),
        size_of::<NearGridDesc>(),
        offset_of!(WorldDesc, tiles),
        offset_of!(WorldDesc, curve_grid),
        offset_of!(WorldDesc, shape_grid),
        offset_of!(GpuCar, tick),
        offset_of!(GpuCar, pairs),
        offset_of!(GpuCar, contacts),
        offset_of!(GpuAgent, recent),
        offset_of!(VehicleDesc, steering_speed),
        offset_of!(VehicleDesc, gravity),
    ];
    if count != want.len() {
        return Err("libaltd_gpu.so layout table differs; rebuild it with gpu/build.sh or gpu/build-cuda.*".into());
    }
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        if g != w as u64 {
            return Err(format!("GPU struct layout entry {i} differs (library {g}, Rust {w})"));
        }
    }
    Ok(())
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
    pub(crate) names: Vec<String>,
    pub(crate) values: Vec<Surface>,
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
    pub(crate) fn by_name(&self, name: &str) -> u32 {
        let named = self.names.iter().position(|n| n == name).map_or(0, |i| i + 1);
        self.values
            .iter()
            .position(|&s| s == self.values[named])
            .unwrap_or(named) as u32
    }

    /// First index holding `surface`.
    pub(crate) fn by_value(&self, surface: Surface) -> Result<u32, String> {
        self.values
            .iter()
            .position(|&s| s == surface)
            .map(|i| i as u32)
            .ok_or_else(|| format!("unknown surface {surface:?}"))
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
    pub(crate) cell: f64,
    pub nx: u32,
    pub ny: u32,
    pub start: Vec<u32>,
    pub items: Vec<u32>,
}

fn segment_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let l2 = d[0] * d[0] + d[1] * d[1];
    let t = if l2 > 0.0 {
        (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let q = [a[0] + d[0] * t - p[0], a[1] + d[1] * t - p[1]];
    (q[0] * q[0] + q[1] * q[1]).sqrt()
}

fn box_distance(a: [f64; 4], b: [f64; 4]) -> f64 {
    let dx = (a[0] - b[2]).max(b[0] - a[2]).max(0.0);
    let dy = (a[1] - b[3]).max(b[1] - a[3]).max(0.0);
    (dx * dx + dy * dy).sqrt()
}

/// Index for preparing GPU grids. Bounds only prune candidates; leaves
/// retain the original distance calculation and segment indices.
struct SegmentBoxTree {
    bounds: [f64; 4],
    indices: Vec<usize>,
    children: Option<[Box<SegmentBoxTree>; 2]>,
}

impl SegmentBoxTree {
    fn new(boxes: &[[f64; 4]], mut indices: Vec<usize>) -> Self {
        let bounds = indices.iter().fold(
            [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
            |b, &i| {
                let a = boxes[i];
                [b[0].min(a[0]), b[1].min(a[1]), b[2].max(a[2]), b[3].max(a[3])]
            },
        );
        let children = if indices.len() > 16 {
            let axis = usize::from(bounds[3] - bounds[1] > bounds[2] - bounds[0]);
            let middle = indices.len() / 2;
            indices.select_nth_unstable_by(middle, |&a, &b| {
                (boxes[a][axis] + boxes[a][axis + 2]).total_cmp(&(boxes[b][axis] + boxes[b][axis + 2]))
            });
            let right = indices.split_off(middle);
            let left = std::mem::take(&mut indices);
            Some([Box::new(Self::new(boxes, left)), Box::new(Self::new(boxes, right))])
        } else {
            None
        };
        Self {
            bounds,
            indices,
            children,
        }
    }

    fn visit(&self, query: [f64; 4], limit: &mut f64, visitor: &mut impl FnMut(usize, &mut f64)) {
        if box_distance(self.bounds, query) > *limit + 1e-12 * (1.0 + *limit) {
            return;
        }
        if let Some(children) = &self.children {
            let first = usize::from(box_distance(children[1].bounds, query) < box_distance(children[0].bounds, query));
            children[first].visit(query, limit, visitor);
            children[first ^ 1].visit(query, limit, visitor);
        } else {
            for &i in &self.indices {
                visitor(i, limit);
            }
        }
    }
}

/// Builds the grid over the segments' bounding box grown by `margin`.
/// For a cell box C, `bound = min_s max_{corner c} |c - s|` bounds the
/// nearest distance of every point of C (distance to a segment is convex);
/// segment s is listed if the distance from C to its bounding box is at most
/// `bound` plus the slack. The GPU's float32 distances differ from the exact
/// ones by far less than the slack `1 + 1e-4 * magnitude` (as in
/// segment_grid), so no segment that can tie or beat the nearest is dropped.
pub fn near_grid(segments: &[([f64; 2], [f64; 2])], cell: f64, margin: f64) -> Option<NearGrid> {
    if segments.is_empty()
        || segments
            .iter()
            .any(|(a, b)| !(a[0].is_finite() && a[1].is_finite() && b[0].is_finite() && b[1].is_finite()))
    {
        return None;
    }
    let boxes: Vec<[f64; 4]> = segments
        .iter()
        .map(|(a, b)| [a[0].min(b[0]), a[1].min(b[1]), a[0].max(b[0]), a[1].max(b[1])])
        .collect();
    let index = SegmentBoxTree::new(&boxes, (0..segments.len()).collect());
    let lo = boxes
        .iter()
        .fold([f64::INFINITY; 2], |l, b| [l[0].min(b[0]), l[1].min(b[1])]);
    let hi = boxes
        .iter()
        .fold([f64::NEG_INFINITY; 2], |h, b| [h[0].max(b[2]), h[1].max(b[3])]);
    let (x0, y0) = (lo[0] - margin, lo[1] - margin);
    let nx = ((hi[0] + margin - x0) / cell).ceil() as u32;
    let ny = ((hi[1] + margin - y0) / cell).ceil() as u32;
    let magnitude = [x0, y0, x0 + nx as f64 * cell, y0 + ny as f64 * cell]
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()));
    let slack = 1.0 + magnitude * 1e-4;
    let lists: Vec<Vec<u32>> = (0..nx as usize * ny as usize)
        .into_par_iter()
        .map(|c| {
            let (cx, cy) = ((c % nx as usize) as f64, (c / nx as usize) as f64);
            let bx = [
                x0 + cx * cell - 1.0,
                y0 + cy * cell - 1.0,
                x0 + (cx + 1.0) * cell + 1.0,
                y0 + (cy + 1.0) * cell + 1.0,
            ];
            let corners = [[bx[0], bx[1]], [bx[2], bx[1]], [bx[0], bx[3]], [bx[2], bx[3]]];
            // Distance to a segment's box from the cell center is a lower
            // bound on its greatest corner distance. Visit nearby boxes
            // first, then prune boxes that cannot improve the bound.
            let center = [(bx[0] + bx[2]) * 0.5, (bx[1] + bx[3]) * 0.5];
            let mut bound = f64::INFINITY;
            index.visit(
                [center[0], center[1], center[0], center[1]],
                &mut bound,
                &mut |i, bound| {
                    let (a, b) = segments[i];
                    let farthest = corners
                        .iter()
                        .map(|&p| segment_distance(p, a, b))
                        .fold(0.0f64, f64::max);
                    *bound = bound.min(farthest);
                },
            );
            let mut limit = bound * (1.0 + 1e-4) + slack;
            let mut list = Vec::new();
            index.visit(bx, &mut limit, &mut |i, limit| {
                if box_distance(boxes[i], bx) <= *limit {
                    list.push(i as u32);
                }
            });
            list.sort_unstable();
            list
        })
        .collect();
    let mut start = Vec::with_capacity(lists.len() + 1);
    let mut items = Vec::new();
    start.push(0u32);
    for list in &lists {
        items.extend_from_slice(list);
        start.push(items.len() as u32);
    }
    Some(NearGrid {
        x0,
        y0,
        cell,
        nx,
        ny,
        start,
        items,
    })
}

pub(crate) const SHAPE_CELL: f64 = 256.0;
pub(crate) const SHAPE_MARGIN: f64 = 128.0;

/// Physics shape lists for the broad-pair update (gpu/sim/step.h
/// update_pairs): shape i is listed, ascending, in every cell that its box
/// grown by one unit meets after growing the cell by `margin`. A query box
/// starting in a cell and ending before its far edges plus `margin` finds
/// every overlapping shape there; other queries scan all shapes.
pub fn shape_grid(boxes: &[([f64; 2], [f64; 2])], cell: f64, margin: f64) -> Option<NearGrid> {
    if boxes.is_empty()
        || boxes
            .iter()
            .any(|(a, b)| !(a[0].is_finite() && a[1].is_finite() && b[0].is_finite() && b[1].is_finite()))
    {
        return None;
    }
    let lo = boxes
        .iter()
        .fold([f64::INFINITY; 2], |l, (a, _)| [l[0].min(a[0]), l[1].min(a[1])]);
    let hi = boxes
        .iter()
        .fold([f64::NEG_INFINITY; 2], |h, (_, b)| [h[0].max(b[0]), h[1].max(b[1])]);
    let (x0, y0) = (lo[0] - margin, lo[1] - margin);
    let nx = ((hi[0] + margin - x0) / cell).ceil().max(1.0) as u32;
    let ny = ((hi[1] + margin - y0) / cell).ceil().max(1.0) as u32;
    let padded: Vec<_> = boxes
        .iter()
        .map(|(a, b)| [a[0] - 1.0, a[1] - 1.0, b[0] + 1.0, b[1] + 1.0])
        .collect();
    let index = SegmentBoxTree::new(&padded, (0..boxes.len()).collect());
    let mut start = vec![0u32];
    let mut items = Vec::new();
    for cy in 0..ny {
        for cx in 0..nx {
            let c = [
                x0 + cx as f64 * cell - margin,
                y0 + cy as f64 * cell - margin,
                x0 + (cx + 1) as f64 * cell + margin,
                y0 + (cy + 1) as f64 * cell + margin,
            ];
            let row_start = items.len();
            index.visit(c, &mut 0.0, &mut |i, _| {
                let (a, b) = boxes[i];
                if a[0] - 1.0 <= c[2] && b[0] + 1.0 >= c[0] && a[1] - 1.0 <= c[3] && b[1] + 1.0 >= c[1] {
                    items.push(i as u32);
                }
            });
            items[row_start..].sort_unstable();
            start.push(items.len() as u32);
        }
    }
    Some(NearGrid {
        x0,
        y0,
        cell,
        nx,
        ny,
        start,
        items,
    })
}

impl NearGrid {
    fn desc(grid: Option<&NearGrid>) -> NearGridDesc {
        match grid {
            Some(g) => NearGridDesc {
                x0: g.x0,
                y0: g.y0,
                cell: g.cell,
                nx: g.nx,
                ny: g.ny,
                start: g.start.as_ptr(),
                items: g.items.as_ptr(),
            },
            None => NearGridDesc {
                x0: 0.0,
                y0: 0.0,
                cell: 1.0,
                nx: 0,
                ny: 0,
                start: std::ptr::null(),
                items: std::ptr::null(),
            },
        }
    }

    /// (cells, mean list length, longest list)
    pub fn stats(&self) -> (usize, f64, usize) {
        let cells = self.start.len() - 1;
        let longest = self.start.windows(2).map(|w| (w[1] - w[0]) as usize).max().unwrap_or(0);
        (cells, self.items.len() as f64 / cells.max(1) as f64, longest)
    }
}

pub(crate) const GRID_CELL: f64 = 64.0;
pub(crate) const GRID_MARGIN: f64 = 1024.0;

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
    pub(crate) fn new(world: &World) -> Result<TrackArrays, String> {
        let track = &world.track;
        let surfaces = SurfaceTable::new(&world.vehicle);
        let surface_rows = surfaces
            .values
            .iter()
            .map(|s| [s.grip as f32, s.power as f32, s.steering as f32, 0.0])
            .collect();
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
                TileCell {
                    surface,
                    connected,
                    first,
                    last,
                }
            })
            .collect();
        let path_rows: Vec<[f32; 8]> = track
            .gpu_path_rows()
            .into_iter()
            .map(|r| [r[0], r[1], r[2], r[3], r[4], 0.0, 0.0, 0.0])
            .collect();
        let path_offsets = track.path_offsets.clone();
        if track.path.len() != path_offsets.len() {
            return Err("path points and offsets differ in length".into());
        }
        let path_xy = track.path.iter().flat_map(|p| [p.x, p.y]).collect();
        if path_offsets
            .windows(2)
            .any(|w| w[1].partial_cmp(&w[0]) != Some(std::cmp::Ordering::Greater))
        {
            return Err("GPU closest_path needs strictly increasing path offsets".into());
        }
        let path_grid = near_grid(
            &path_rows
                .iter()
                .map(|r| ([r[0] as f64, r[1] as f64], [(r[0] + r[2]) as f64, (r[1] + r[3]) as f64]))
                .collect::<Vec<_>>(),
            GRID_CELL,
            GRID_MARGIN,
        );
        let (curve_position, curve_offsets, curve_forwards, curve_length, curve_grid) = match &track.curve {
            Some(c) => {
                let points: Vec<Vec2> = c.points.iter().map(|&p| f2(p)).collect();
                let segments: Vec<_> = points
                    .windows(2)
                    .map(|w| ([w[0][0] as f64, w[0][1] as f64], [w[1][0] as f64, w[1][1] as f64]))
                    .collect();
                (
                    points,
                    c.offsets.clone(),
                    c.forwards.iter().map(|&p| f2(p)).collect(),
                    c.length(),
                    near_grid(&segments, GRID_CELL, GRID_MARGIN),
                )
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
        let shape_boxes: Vec<_> = track
            .shapes
            .iter()
            .map(|s| ([s.min.x, s.min.y], [s.max.x, s.max.y]))
            .collect();
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

    /// Checks every count represented by a u32 in the native C descriptor.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn validate_counts(&self, rays: &(Vec<RayNode>, Vec<[f32; 4]>, f32, usize)) -> Result<(), String> {
        for (what, count) in [
            ("BSP nodes", rays.0.len()),
            ("BSP walls", rays.1.len()),
            ("BSP depth", rays.3),
            ("surfaces", self.surface_rows.len()),
            ("path points", self.path_offsets.len()),
            ("curve points", self.curve_position.len()),
            ("physics shapes", self.shapes.len()),
            ("physics shape points", self.shape_local.len()),
        ] {
            u32::try_from(count).map_err(|_| format!("too many {what} for the HIP simulator"))?;
        }
        // gpu/sim/world.h fixes the BSP traversal stack at 48 entries.
        if rays.3 > 48 {
            return Err("BSP depth exceeds the HIP simulator's maximum of 48".into());
        }
        for grid in [&self.path_grid, &self.curve_grid, &self.shape_grid]
            .into_iter()
            .flatten()
        {
            u32::try_from(grid.items.len()).map_err(|_| "too many grid entries for the HIP simulator")?;
            let cells = (grid.nx as usize)
                .checked_mul(grid.ny as usize)
                .and_then(|n| n.checked_add(1))
                .ok_or("HIP grid cell count overflow")?;
            if grid.start.len() != cells {
                return Err("HIP grid dimensions do not match its offsets".into());
            }
        }
        Ok(())
    }

    /// The descriptor; `rays` are the BSP arrays of `bsp::RayTree::gpu_arrays`.
    pub(crate) fn desc(&self, rays: &(Vec<RayNode>, Vec<[f32; 4]>, f32, usize)) -> WorldDesc {
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
    let d = |kind: u32, a: f64, b: f64| SensorDesc {
        kind,
        a: a as f32,
        b: b as f32,
        ..Default::default()
    };
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
        Sensor::Grip { offset } => SensorDesc {
            kind: 10,
            offset: f2(F2::from(offset)),
            ..Default::default()
        },
        Sensor::CorrectDirection => d(11, 0.0, 0.0),
        Sensor::TrackCurvature {
            min_lookahead,
            max_lookahead,
        } => d(12, min_lookahead, max_lookahead),
    }
}

// ---- simulation handle ----

/// `ReadMask` of gpu/sim/sensors.h: the cars whose controls update this tick.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReadMask {
    pub(crate) batch_size: u32,
    pub(crate) batch_index: u32,
    pub(crate) batches_per_tick: u32,
    pub(crate) require_active: u32,
}

impl ReadMask {
    /// Every car, active or not.
    pub const ALL: ReadMask = ReadMask {
        batch_size: u32::MAX,
        batch_index: 0,
        batches_per_tick: 8,
        require_active: 0,
    };
}

/// GPU control order of `GpuAgent::controls`.
pub(crate) const CONTROL_NAMES: [&str; 5] = ["acceleration", "steering", "brake", "handbrake", "boost"];

/// For each control, the last output neuron with its name (-1: none, 0.0),
/// as `Controls(**dict(zip(names, outputs)))` assigns them.
pub fn control_sources(outputs: &[String]) -> [i32; 5] {
    CONTROL_NAMES.map(|c| {
        outputs
            .iter()
            .rposition(|o| o.to_lowercase() == c)
            .map_or(-1, |j| j as i32)
    })
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

#[cfg(not(target_arch = "wasm32"))]
pub struct GpuSim<'a> {
    gpu: &'a Gpu,
    handle: *mut c_void,
    free: unsafe extern "C" fn(*mut c_void),
    capacity: usize,
    pub(crate) sensor_count: usize,
    pub(crate) cars: usize,
    /// Caller-chosen tag of the uploaded networks (e.g. the generation), so
    /// unchanged networks are not uploaded again.
    pub network_tag: Option<u64>,
    network_count: usize,
    parameter_buffer: Vec<f64>,
    /// Host staging buffers for car and agent state, retained between windows
    /// so their pages stay mapped and no per-window allocation is needed.
    car_buffer: Vec<GpuCar>,
    agent_buffer: Vec<GpuAgent>,
}

/// Arguments of `altd_gpu_sim_window` (gpu/sim/altd_gpu.hip).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct WindowArgs {
    pub(crate) start_tick: u64,
    pub(crate) ticks: u64,
    pub(crate) start_batch: u32,
    pub(crate) batches_per_tick: u32,
    pub(crate) stats_phase: u32,
    pub(crate) stop_when_inactive: u32,
    pub(crate) eliminate_on_wall: u32,
    pub(crate) eliminate_when_idle: u32,
    pub(crate) has_time_limit: u32,
    pub(crate) pad: u32,
    pub(crate) time_limit: f64,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for GpuSim<'_> {
    fn drop(&mut self) {
        unsafe { (self.free)(self.handle) };
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn check(status: i32, what: &str) {
    assert_eq!(
        status,
        0,
        "GPU {what} failed with {}",
        crate::gpu::hip::status_message(status)
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn try_check(status: i32, what: &str) -> Result<(), String> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!(
            "GPU {what} failed with {}",
            crate::gpu::hip::status_message(status)
        ))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn network_size(shape: &[usize], sensor_count: usize, control_src: [i32; 5]) -> Result<usize, String> {
    if !(2..=MAX_LAYERS).contains(&shape.len()) {
        return Err(format!("the HIP simulator needs 2..{MAX_LAYERS} network layers"));
    }
    if shape[0] != sensor_count {
        return Err(format!(
            "the HIP network has {} inputs but the simulator has {sensor_count} sensors",
            shape[0]
        ));
    }
    for (layer, &width) in shape.iter().enumerate() {
        let limit = if layer == 0 { MAX_WIDTH } else { LANES };
        if width == 0 || width > limit {
            return Err(format!(
                "HIP network layer {layer} has width {width}; supported widths are 1..{limit}"
            ));
        }
    }
    let outputs = *shape.last().unwrap();
    if control_src.iter().any(|&index| index < -1 || index >= outputs as i32) {
        return Err("HIP control mapping is outside the network output layer".into());
    }
    shape.windows(2).try_fold(0usize, |size, layer| {
        layer[0]
            .checked_add(1)
            .and_then(|rows| rows.checked_mul(layer[1]))
            .and_then(|count| size.checked_add(count))
            .ok_or_else(|| "HIP network parameter count overflow".to_string())
    })
}

#[cfg(not(target_arch = "wasm32"))]
impl<'a> GpuSim<'a> {
    pub fn new(world: &GpuWorld<'a>, vehicle: &VehicleDesc, sensors: &[SensorDesc], capacity: usize) -> GpuSim<'a> {
        Self::try_new(world, vehicle, sensors, capacity).unwrap_or_else(|e| panic!("{e}"))
    }

    /// `new`, reporting invalid capacities, unsupported sensors and allocation failures.
    pub fn try_new(
        world: &GpuWorld<'a>,
        vehicle: &VehicleDesc,
        sensors: &[SensorDesc],
        capacity: usize,
    ) -> Result<GpuSim<'a>, String> {
        let gpu = world.gpu();
        if sensors.len() > MAX_SENSORS {
            return Err(format!(
                "{} sensors exceed the HIP maximum of {MAX_SENSORS}",
                sensors.len()
            ));
        }
        if vehicle.wheel_count as usize > MAX_WHEELS {
            return Err(format!(
                "{} wheels exceed the HIP maximum of {MAX_WHEELS}",
                vehicle.wheel_count
            ));
        }
        let count = u32::try_from(capacity).map_err(|_| "population exceeds the HIP maximum of 4294967295")?;
        if count == 0 || count > u32::MAX - 7 {
            return Err("HIP population capacity must be positive and leave room for eight inference batches".into());
        }
        for size in [
            std::mem::size_of::<GpuCar>(),
            std::mem::size_of::<GpuAgent>(),
            MAX_SENSORS * std::mem::size_of::<f64>(),
        ] {
            if capacity
                .checked_mul(size)
                .is_none_or(|bytes| bytes > isize::MAX as usize)
            {
                return Err("HIP population capacity exceeds the native address space".into());
            }
        }
        let create: unsafe extern "C" fn(
            *const c_void,
            *const VehicleDesc,
            *const SensorDesc,
            u32,
            u32,
        ) -> *mut c_void = gpu.required_symbol("altd_gpu_sim_create")?;
        let free = gpu.required_symbol("altd_gpu_sim_free")?;
        let handle = unsafe { create(world.handle(), vehicle, sensors.as_ptr(), sensors.len() as u32, count) };
        if handle.is_null() {
            return Err("altd_gpu_sim_create failed to allocate the HIP simulator".into());
        }
        Ok(GpuSim {
            gpu,
            handle,
            free,
            capacity,
            sensor_count: sensors.len(),
            cars: 0,
            network_tag: None,
            network_count: 0,
            parameter_buffer: Vec::new(),
            car_buffer: Vec::new(),
            agent_buffer: Vec::new(),
        })
    }

    /// Switch uploaded tracks while retaining population and network buffers.
    /// Keep `world` alive until the next switch or until this simulator is dropped.
    pub fn set_world(&mut self, world: &GpuWorld<'a>) {
        self.try_set_world(world).unwrap_or_else(|e| panic!("{e}"));
    }

    /// `set_world`, reporting incompatible handles and HIP failures.
    pub fn try_set_world(&mut self, world: &GpuWorld<'a>) -> Result<(), String> {
        if !std::ptr::eq(self.gpu, world.gpu()) {
            return Err("GPU world belongs to another device handle".into());
        }
        let f: unsafe extern "C" fn(*mut c_void, *const c_void) -> i32 =
            self.gpu.required_symbol("altd_gpu_sim_set_world")?;
        try_check(unsafe { f(self.handle, world.handle()) }, "set_world")
    }

    /// The retained host state buffers (empty on first use); return them with
    /// `return_state_buffers` so the next window reuses the allocations.
    pub(crate) fn take_state_buffers(&mut self) -> (Vec<GpuCar>, Vec<GpuAgent>) {
        (
            std::mem::take(&mut self.car_buffer),
            std::mem::take(&mut self.agent_buffer),
        )
    }
    pub(crate) fn return_state_buffers(&mut self, cars: Vec<GpuCar>, agents: Vec<GpuAgent>) {
        self.car_buffer = cars;
        self.agent_buffer = agents;
    }

    pub fn upload(&mut self, cars: &[GpuCar], agents: Option<&[GpuAgent]>) {
        self.try_upload(cars, agents).unwrap_or_else(|e| panic!("{e}"));
    }

    /// `upload`, reporting invalid state counts and HIP failures.
    pub fn try_upload(&mut self, cars: &[GpuCar], agents: Option<&[GpuAgent]>) -> Result<(), String> {
        if cars.len() > self.capacity {
            return Err("car population exceeds the HIP simulator capacity".into());
        }
        let count = u32::try_from(cars.len()).map_err(|_| "too many cars for the HIP simulator")?;
        let f: unsafe extern "C" fn(*mut c_void, u32, *const GpuCar, *const GpuAgent) -> i32 =
            self.gpu.required_symbol("altd_gpu_sim_upload")?;
        if let Some(a) = agents {
            if a.len() != cars.len() {
                return Err("HIP car and agent counts differ".into());
            }
        }
        try_check(
            unsafe {
                f(
                    self.handle,
                    count,
                    cars.as_ptr(),
                    agents.map_or(std::ptr::null(), |a| a.as_ptr()),
                )
            },
            "upload",
        )?;
        self.cars = cars.len();
        Ok(())
    }

    /// A training window over the uploaded cars (`TrainingRunner::advance_window`
    /// order); returns (executed ticks, transition_without_drive).
    pub(crate) fn try_window(&self, args: &WindowArgs) -> Result<(u64, bool), String> {
        let f: unsafe extern "C" fn(*mut c_void, *const WindowArgs, *mut u64, *mut u32) -> i32 =
            self.gpu.required_symbol("altd_gpu_sim_window")?;
        let (mut executed, mut transition) = (0u64, 0u32);
        try_check(
            unsafe { f(self.handle, args, &mut executed, &mut transition) },
            "window",
        )?;
        if executed > args.ticks || transition > 1 || executed < transition as u64 {
            return Err("HIP window returned invalid tick counters".into());
        }
        Ok((executed, transition != 0))
    }

    /// Replaces the agents of the uploaded cars.
    pub fn upload_agents(&mut self, agents: &[GpuAgent]) {
        let f: unsafe extern "C" fn(*mut c_void, u32, *const GpuAgent) -> i32 =
            self.gpu.symbol("altd_gpu_sim_upload_agents");
        check(
            unsafe { f(self.handle, agents.len() as u32, agents.as_ptr()) },
            "upload_agents",
        );
    }

    /// The number of active uploaded cars, counted on the device; `None` when
    /// the library predates `altd_gpu_sim_active_count`.
    pub(crate) fn try_active_count(&self) -> Result<Option<usize>, String> {
        let Some(f) = self
            .gpu
            .try_symbol::<unsafe extern "C" fn(*mut c_void, *mut u32) -> i32>("altd_gpu_sim_active_count")
        else {
            return Ok(None);
        };
        let mut count = 0u32;
        try_check(unsafe { f(self.handle, &mut count) }, "active_count")?;
        if count as usize > self.cars {
            return Err("HIP active count exceeds the uploaded cars".into());
        }
        Ok(Some(count as usize))
    }

    pub fn download(&self, cars: Option<&mut Vec<GpuCar>>, agents: Option<&mut Vec<GpuAgent>>) {
        self.try_download(cars, agents).unwrap_or_else(|e| panic!("{e}"));
    }

    /// `download`, reporting allocation, missing-symbol and HIP failures.
    pub fn try_download(
        &self,
        cars: Option<&mut Vec<GpuCar>>,
        agents: Option<&mut Vec<GpuAgent>>,
    ) -> Result<(), String> {
        let f: unsafe extern "C" fn(*mut c_void, *mut GpuCar, *mut GpuAgent) -> i32 =
            self.gpu.required_symbol("altd_gpu_sim_download")?;
        let cars = cars
            .map(|c| {
                c.try_reserve(self.cars.saturating_sub(c.len()))
                    .map_err(|e| format!("HIP car readback allocation failed: {e}"))?;
                c.resize(self.cars, GpuCar::zeroed());
                Ok::<_, String>(c.as_mut_ptr())
            })
            .transpose()?;
        let agents = agents
            .map(|a| {
                a.try_reserve(self.cars.saturating_sub(a.len()))
                    .map_err(|e| format!("HIP agent readback allocation failed: {e}"))?;
                a.resize(self.cars, GpuAgent::default());
                Ok::<_, String>(a.as_mut_ptr())
            })
            .transpose()?;
        try_check(
            unsafe {
                f(
                    self.handle,
                    cars.unwrap_or(std::ptr::null_mut()),
                    agents.unwrap_or(std::ptr::null_mut()),
                )
            },
            "download",
        )
    }

    /// One network per car: `params` holds `networks` flat parameter vectors of `shape`.
    pub fn networks(&mut self, shape: &[usize], params: &[f64], control_src: [i32; 5]) {
        self.try_networks(shape, params, control_src)
            .unwrap_or_else(|e| panic!("{e}"));
    }

    /// `networks`, validating the HIP kernel limits before upload and allocation.
    pub fn try_networks(&mut self, shape: &[usize], params: &[f64], control_src: [i32; 5]) -> Result<(), String> {
        let size = network_size(shape, self.sensor_count, control_src)?;
        if params.is_empty() || !params.len().is_multiple_of(size) {
            return Err("HIP network parameters do not contain complete networks".into());
        }
        let count = params.len() / size;
        if count > self.capacity {
            return Err("network population exceeds the HIP simulator capacity".into());
        }
        let count = u32::try_from(count).map_err(|_| "too many networks for the HIP simulator")?;
        let f: unsafe extern "C" fn(*mut c_void, u32, *const u32, u32, *const f64, *const i32) -> i32 =
            self.gpu.required_symbol("altd_gpu_sim_networks")?;
        let widths: Vec<u32> = shape.iter().map(|&w| w as u32).collect();
        let status = unsafe {
            f(
                self.handle,
                widths.len() as u32,
                widths.as_ptr(),
                count,
                params.as_ptr(),
                control_src.as_ptr(),
            )
        };
        try_check(status, "networks")?;
        self.network_count = params.len() / size;
        Ok(())
    }

    pub(crate) fn network_count(&self) -> usize {
        self.network_count
    }

    pub(crate) fn take_parameter_buffer(&mut self) -> Vec<f64> {
        std::mem::take(&mut self.parameter_buffer)
    }
    pub(crate) fn return_parameter_buffer(&mut self, buffer: Vec<f64>) {
        self.parameter_buffer = buffer;
    }

    /// Reuse a flat upload buffer, filling disjoint networks in parallel.
    pub fn upload_networks(
        &mut self,
        networks: &[crate::nn::network::Network],
        control_src: [i32; 5],
    ) -> Result<(), String> {
        self.upload_network_refs(&networks.iter().collect::<Vec<_>>(), control_src)
    }

    pub(crate) fn upload_agent_networks(
        &mut self,
        agents: &[crate::training::TrainingAgent],
        control_src: [i32; 5],
    ) -> Result<(), String> {
        self.upload_network_refs(&agents.iter().map(|a| &a.network).collect::<Vec<_>>(), control_src)
    }

    fn upload_network_refs(
        &mut self,
        networks: &[&crate::nn::network::Network],
        control_src: [i32; 5],
    ) -> Result<(), String> {
        use rayon::prelude::*;
        let first = networks.first().ok_or("no networks")?;
        let size = network_size(&first.shape, self.sensor_count, control_src)?;
        if networks.len() > self.capacity {
            return Err("network population exceeds the HIP simulator capacity".into());
        }
        if networks
            .iter()
            .any(|n| n.shape != first.shape || n.params.len() != size)
        {
            return Err("networks of different shapes or invalid parameter counts".into());
        }
        let mut profile = crate::training::training_profile::Profile::new("network_upload");
        let mut params = std::mem::take(&mut self.parameter_buffer);
        let length = networks
            .len()
            .checked_mul(size)
            .ok_or("network parameter count overflow")?;
        params
            .try_reserve(length.saturating_sub(params.len()))
            .map_err(|e| format!("HIP network upload allocation failed: {e}"))?;
        params.resize(length, 0.0);
        params
            .par_chunks_mut(size)
            .zip(networks.par_iter())
            .for_each(|(out, network)| out.copy_from_slice(&network.params));
        profile.mark("pack");
        let result = self.try_networks(&first.shape, &params, control_src);
        profile.mark("upload");
        self.parameter_buffer = params;
        result
    }

    /// Exact population novelty for the currently uploaded networks.
    pub fn novelty(&self) -> Vec<f64> {
        self.try_novelty().unwrap_or_else(|e| panic!("{e}"))
    }

    /// `novelty`, reporting allocation, missing-symbol and HIP failures.
    pub fn try_novelty(&self) -> Result<Vec<f64>, String> {
        let f: unsafe extern "C" fn(*mut c_void, u32, *mut f64) -> i32 =
            self.gpu.required_symbol("altd_gpu_sim_novelty")?;
        let mut out = Vec::new();
        out.try_reserve(self.network_count)
            .map_err(|e| format!("HIP novelty allocation failed: {e}"))?;
        out.resize(self.network_count, 0.0);
        try_check(
            unsafe { f(self.handle, self.network_count as u32, out.as_mut_ptr()) },
            "population novelty",
        )?;
        Ok(out)
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
        check(
            unsafe { f(self.handle, drive as u32, eliminate_on_wall as u32, tick) },
            "step",
        );
    }

    /// `Schedule::update_stats` for every car at statistics tick `tick`;
    /// `stats_tick` is `tick - first_stats_tick`.
    pub fn stats(&self, tick: u64, stats_tick: u64, eliminate_when_idle: bool) {
        let f: unsafe extern "C" fn(*mut c_void, u64, u64, u32) -> i32 = self.gpu.symbol("altd_gpu_sim_stats");
        check(
            unsafe { f(self.handle, tick, stats_tick, eliminate_when_idle as u32) },
            "stats",
        );
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
    a.controls = crate::physics::car::Controls {
        acceleration,
        steering,
        brake,
        handbrake,
        boost,
    };
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
    s.recent_score_diffs
        .extend((0..g.recent_len as usize).map(|k| g.recent[(g.recent_start as usize + k) % RECENT]));
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod native_validation_tests {
    use super::*;

    #[test]
    fn network_limits_match_the_hip_forward_kernel() {
        let controls = [0, 1, 2, 3, 4];
        assert_eq!(network_size(&[20, 16, 5], 20, controls).unwrap(), 421);
        assert!(network_size(&[32, 16, 5], 32, controls).is_ok());
        assert!(network_size(&[20, 32, 5], 20, controls)
            .unwrap_err()
            .contains("width 32"));
        assert!(network_size(&[20, 5, 17], 20, controls).unwrap_err().contains("1..16"));
        assert!(network_size(&[33, 5], 33, controls).unwrap_err().contains("1..32"));
        assert!(network_size(&[20, 0, 5], 20, controls).is_err());
        assert!(network_size(&[20], 20, controls).is_err());
        assert!(network_size(&[20; MAX_LAYERS + 1], 20, controls)
            .unwrap_err()
            .contains("network layers"));
        assert!(network_size(&[20, 5], 19, controls).unwrap_err().contains("sensors"));
        assert!(network_size(&[20, 5], 20, [0, 1, 2, 3, 5])
            .unwrap_err()
            .contains("control mapping"));
        assert!(network_size(&[20, 5], 20, [-2; 5]).is_err());
        assert!(network_size(&[20, 5], 20, [-1; 5]).is_ok());
    }

    #[test]
    fn a_hip_window_error_returns_without_panicking() {
        let gpu = match crate::training::session::hip_device() {
            Ok((gpu, _)) => gpu,
            Err(reason) => {
                eprintln!("skipping HIP window error check: {reason}");
                return;
            }
        };
        let template = serde_json::from_str(include_str!("../../assets/scenes/formula_template.json")).unwrap();
        let settings = serde_json::from_str(include_str!("../../assets/random_track_settings.json")).unwrap();
        let (_, scene) =
            crate::track::training_tracks::training_scene_at(&template, &settings, None, 1729, 0, 0).unwrap();
        let world = World::from_scene(&scene);
        let prepared = crate::gpu::hip::PreparedGpuWorld::new(&world).unwrap();
        let uploaded = GpuWorld::try_from_prepared(gpu, prepared).unwrap();
        let mut sim = GpuSim::try_new(
            &uploaded,
            &vehicle_desc(&world.vehicle).unwrap(),
            &[sensor_desc(&Sensor::Speed)],
            1,
        )
        .unwrap();
        // An empty simulator has no uploaded networks, which the C API rejects.
        let error = sim
            .try_window(&WindowArgs {
                ticks: 1,
                ..Default::default()
            })
            .unwrap_err();
        assert!(error.contains("window") && error.contains("-2"), "{error}");
        let error = sim.try_networks(&[1, 32, 5], &[], [0, 1, 2, 3, 4]).unwrap_err();
        assert!(error.contains("width 32"), "{error}");
    }
}
