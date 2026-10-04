//! Car rectangle against static convex tile shapes using Godot 4.3 float32
//! SAT and the sequential impulse solver.

use crate::math::godot_math::F2;
use crate::math::pymath::{f32r, py_max, py_min};
use crate::math::vec2::V2;
use crate::physics::broadphase::{Aabb, BroadPhase};
use crate::track::world::{PhysicsShape, Track, VehicleConfig};

const SUPPORT_EDGE_DOT: f64 = 0.99998;
const CMP_EPSILON: f64 = 1e-5;
// This captured float literal is part of the contact solver contract.
#[allow(clippy::approx_constant)]
const MAX_BIAS_ROTATION: f64 = 0.39269908169872414;

#[inline(always)]
fn dot_f32(a: V2, b: V2) -> f64 {
    f32r(f32r(a.x * b.x) + f32r(a.y * b.y))
}

#[inline(always)]
fn normalize_f32(v: V2) -> V2 {
    let length = f32r(dot_f32(v, v).sqrt());
    if length != 0.0 {
        V2::new(f32r(v.x / length), f32r(v.y / length))
    } else {
        V2::ZERO
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Contact {
    /// Points from the static shape toward the car.
    pub normal: V2,
    /// Point on the car.
    pub point: V2,
    pub depth: f64,
    pub shape_index: usize,
    pub wall_first: bool,
    pub wall_point: V2,
    pub local_point: V2,
    pub wall_local_point: V2,
    pub used: bool,
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

/// Up to two support points in their contact order.
#[derive(Clone, Copy)]
struct Support {
    points: [V2; 2],
    len: usize,
}

impl Support {
    fn one(a: V2) -> Support {
        Support {
            points: [a, V2::ZERO],
            len: 1,
        }
    }
    fn two(a: V2, b: V2) -> Support {
        Support { points: [a, b], len: 2 }
    }
    fn slice(&self) -> &[V2] {
        &self.points[..self.len]
    }
}

fn project_on_line(point: V2, edge: &[V2]) -> V2 {
    let direction = V2::new(f32r(edge[1].x - edge[0].x), f32r(edge[1].y - edge[0].y));
    let offset = V2::new(f32r(point.x - edge[0].x), f32r(point.y - edge[0].y));
    let length_squared = dot_f32(direction, direction);
    if length_squared < 1e-20f32 as f64 {
        return edge[0];
    }
    let fraction = f32r(dot_f32(direction, offset) / length_squared);
    V2::new(
        f32r(edge[0].x + f32r(direction.x * fraction)),
        f32r(edge[0].y + f32r(direction.y * fraction)),
    )
}

fn rectangle_supports(car: &[V2; 4], direction: V2, basis_x: V2, basis_y: V2) -> Support {
    let local = normalize_f32(V2::new(dot_f32(basis_x, direction), dot_f32(basis_y, direction)));
    if local.x.abs() > SUPPORT_EDGE_DOT {
        return if local.x > 0.0 {
            Support::two(car[2], car[1])
        } else {
            Support::two(car[3], car[0])
        };
    }
    if local.y.abs() > SUPPORT_EDGE_DOT {
        return if local.y > 0.0 {
            Support::two(car[2], car[3])
        } else {
            Support::two(car[1], car[0])
        };
    }
    if local.x < 0.0 {
        return Support::one(if local.y < 0.0 { car[0] } else { car[3] });
    }
    Support::one(if local.y < 0.0 { car[1] } else { car[2] })
}

fn polygon_supports(wall: &[V2], shape: &PhysicsShape, direction: V2) -> Support {
    let direction = normalize_f32(direction);
    let mut maximum = f64::NEG_INFINITY;
    let mut best = 0;
    let n = shape.local.len();
    for i in 0..n {
        let projection = dot_f32(shape.local[i], direction);
        if projection > maximum {
            maximum = projection;
            best = i;
        }
        if dot_f32(shape.local_normals[i], direction) > SUPPORT_EDGE_DOT {
            return Support::two(wall[i], wall[(i + 1) % wall.len()]);
        }
    }
    Support::one(wall[best])
}

fn contact_pairs(a: &[V2], b: &[V2], push: V2, out: &mut [(V2, V2); 2]) -> usize {
    if a.len() == 1 && b.len() == 1 {
        out[0] = (a[0], b[0]);
        return 1;
    }
    if a.len() == 1 {
        out[0] = (a[0], project_on_line(a[0], b));
        return 1;
    }
    if b.len() == 1 {
        out[0] = (project_on_line(b[0], a), b[0]);
        return 1;
    }
    // Godot Vector2::orthogonal is (y, -x). This also fixes the order of
    // the two edge contacts, which is observable in solver roundoff.
    let tangent = V2::new(push.y, -push.x);
    let distance_a = dot_f32(push, a[0]);
    let distance_b = dot_f32(push, b[0]);
    // Stable sort of 4 entries by projection, car points first.
    let mut ordered = [
        (dot_f32(a[0], tangent), true, a[0]),
        (dot_f32(a[1], tangent), true, a[1]),
        (dot_f32(b[0], tangent), false, b[0]),
        (dot_f32(b[1], tangent), false, b[1]),
    ];
    for i in 1..4 {
        let mut j = i;
        while j > 0 && ordered[j].0 < ordered[j - 1].0 {
            ordered.swap(j, j - 1);
            j -= 1;
        }
    }
    let mut count = 0;
    for &(_, from_car, point) in &ordered[1..3] {
        let pair = if from_car {
            let distance = f32r(dot_f32(push, point) - distance_b);
            let other = V2::new(
                f32r(point.x - f32r(push.x * distance)),
                f32r(point.y - f32r(push.y * distance)),
            );
            (point, other)
        } else {
            let distance = f32r(dot_f32(push, point) - distance_a);
            let other = V2::new(
                f32r(point.x - f32r(push.x * distance)),
                f32r(point.y - f32r(push.y * distance)),
            );
            (other, point)
        };
        if dot_f32(push, pair.0) <= dot_f32(push, pair.1) - CMP_EPSILON {
            out[count] = pair;
            count += 1;
        }
    }
    count
}

/// Scratch buffers reused across ticks by one car.
#[derive(Default, Clone)]
pub(crate) struct CollisionScratch {
    candidates: Vec<u32>,
    wall: Vec<V2>,
    axes: Vec<V2>,
    active_indices: Vec<usize>,
    collided_shapes: Vec<usize>,
    broad_pairs: Vec<usize>,
    shape_aabb: Option<FloatRect>,
    leaf_aabb: Option<Aabb>,
    broadphase: Option<BroadPhase>,
    pair_check_pending: bool,
    external_pairs: bool,
    pair_orientation: Vec<bool>,
    pair_epochs: Vec<u64>,
    separating_axes: std::collections::HashMap<usize, V2>,
}

#[derive(Clone, Copy)]
struct FloatRect {
    position: F2,
    size: F2,
}
impl FloatRect {
    fn end(self) -> F2 {
        self.position + self.size
    }
    fn grow(mut self, margin: f32) -> Self {
        self.position = self.position - F2 { x: margin, y: margin };
        self.size = self.size
            + F2 {
                x: margin * 2.0,
                y: margin * 2.0,
            };
        self
    }
    fn expand(&mut self, p: F2) {
        let end = self.end();
        let begin = F2 {
            x: self.position.x.min(p.x),
            y: self.position.y.min(p.y),
        };
        let end = F2 {
            x: end.x.max(p.x),
            y: end.y.max(p.y),
        };
        self.position = begin;
        self.size = end - begin;
    }
}
impl CollisionScratch {
    pub(crate) fn has_shape(&self) -> bool {
        self.shape_aabb.is_some()
    }
    pub(crate) fn shape_bounds(&self) -> Aabb {
        let r = self.shape_aabb.expect("native shape initialized");
        Aabb::from_rect(r.position, r.size)
    }
    pub(crate) fn set_external_pairs(
        &mut self,
        pairs: &[usize],
        wall_first: &[bool],
        epochs: &[u64],
        previous: &mut Vec<Contact>,
    ) {
        let retained = |wall: usize| {
            pairs.iter().position(|&p| p == wall).is_some_and(|i| {
                self.broad_pairs
                    .iter()
                    .position(|&p| p == wall)
                    .is_some_and(|j| epochs[i] == self.pair_epochs[j])
            })
        };
        previous.retain(|c| retained(c.shape_index));
        self.separating_axes.retain(|&wall, _| retained(wall));
        self.external_pairs = true;
        self.broad_pairs.clear();
        self.broad_pairs.extend_from_slice(pairs);
        self.pair_orientation.clear();
        self.pair_orientation.extend_from_slice(wall_first);
        self.pair_epochs.clear();
        self.pair_epochs.extend_from_slice(epochs);
    }
    /// Native _update_shapes runs when a transform changes, including resets;
    /// pair processing happens later in the broad-phase update.
    pub(crate) fn update_shape(&mut self, cfg: &VehicleConfig, position: V2, body: (F2, F2)) {
        let (sx, sy) = cfg.shape_basis;
        let x = body.0 * sx.x + body.1 * sx.y;
        let y = body.0 * sy.x + body.1 * sy.y;
        let size = F2::from(cfg.shape_size);
        let shape_origin =
            body.0 * cfg.shape_position.x as f32 + body.1 * cfg.shape_position.y as f32 + F2::from(position);
        let pos = x * (-size.x * 0.5) + y * (-size.y * 0.5) + shape_origin;
        let dx = x * size.x;
        let dy = y * size.y;
        let mut rect = FloatRect {
            position: pos,
            size: F2::default(),
        };
        rect.expand(pos + dx);
        rect.expand(pos + dy);
        rect.expand((pos + dx) + dy);
        let margin = self
            .shape_aabb
            .map_or(0.0, |old| ((old.size.x + old.size.y) as f64 * 0.5 * 0.05) as f32);
        rect = rect.grow(margin);
        self.shape_aabb = Some(rect);
        let Some(old) = self.leaf_aabb else {
            // item_add inserts the unexpanded box; the following move is a no-op.
            self.leaf_aabb = Some(Aabb::from_rect(rect.position, rect.size));
            self.pair_check_pending = true;
            return;
        };
        let threshold = (0.1f32 as f64 * 2.0 * 2.0 * 1.1f32 as f64) as f32;
        let density = (self.broad_pairs.len() as f64 * (1.0 / 9.0)) as f32;
        let raw = Aabb::from_rect(rect.position, rect.size);
        let expanded = raw.grow(0.1f32 * (1.0 - density.min(1.0)));
        let within_node = self.broadphase.as_ref().is_none_or(|b| b.body_node.encloses(expanded));
        let old_size = old.size();
        if within_node
            && old.roundtrip_rect().encloses(raw)
            && (old_size.x + old_size.y) - (rect.size.x + rect.size.y) < threshold
        {
            return;
        }
        if !within_node {
            self.broadphase.as_mut().unwrap().body_node = expanded.grow(0.5);
        }
        self.leaf_aabb = Some(expanded);
        self.pair_check_pending = true;
    }
}

fn find_contacts_with_basis(
    cfg: &VehicleConfig,
    track: &Track,
    position: V2,
    _rotation: f64,
    body_basis: (F2, F2),
    scratch: &mut CollisionScratch,
    contacts: &mut Vec<Contact>,
) {
    contacts.clear();
    let (sx, sy) = cfg.shape_basis;
    let basis_x = V2::from(body_basis.0 * sx.x + body_basis.1 * sx.y);
    let basis_y = V2::from(body_basis.0 * sy.x + body_basis.1 * sy.y);
    let half_x = f32r(cfg.shape_size.x / 2.0);
    let half_y = f32r(cfg.shape_size.y / 2.0);

    if scratch.shape_aabb.is_none() {
        scratch.update_shape(cfg, position, body_basis);
    }
    let leaf = scratch.leaf_aabb.unwrap();
    if !scratch.external_pairs && track.native_broadphase && scratch.broadphase.is_none() {
        let walls: Vec<Aabb> = track
            .shapes
            .iter()
            .map(|s| Aabb {
                min: F2::from(s.min),
                max: F2::from(s.max),
            })
            .collect();
        let broadphase = BroadPhase::with_layout(&walls, leaf, track.native_tilemap);
        scratch.broad_pairs = broadphase.initial_pairs.iter().map(|&id| id as usize).collect();
        scratch.broadphase = Some(broadphase);
    }
    if let Some(b) = scratch.broadphase.as_mut() {
        b.update(leaf);
    }
    let query = leaf.roundtrip_rect();
    let (left, right, top, bottom) = (
        query.min.x as f64,
        query.max.x as f64,
        query.min.y as f64,
        query.max.y as f64,
    );
    let car_origin = V2::new(f32r(position.x), f32r(position.y));

    let mut car_local = [V2::ZERO; 4];
    for (i, (sx, sy)) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        .into_iter()
        .enumerate()
    {
        let (x, y) = (f32r(sx * half_x), f32r(sy * half_y));
        car_local[i] = V2::new(
            f32r(f32r(basis_x.x * x) + f32r(basis_y.x * y)),
            f32r(f32r(basis_x.y * x) + f32r(basis_y.y * y)),
        );
    }
    let car_axes = [normalize_f32(basis_x), normalize_f32(basis_y)];

    if scratch.pair_check_pending && !scratch.external_pairs {
        if let Some(b) = scratch.broadphase.as_ref() {
            b.cull(query, &mut scratch.candidates);
        } else {
            track.shape_candidates(V2::new(left, top), V2::new(right, bottom), &mut scratch.candidates);
        }
        // Godot's body constraint list preserves broadphase pair creation order.
        // A pair can exist for many frames before its convex shapes actually touch.
        let overlaps = |index: usize| {
            if let Some(b) = scratch.broadphase.as_ref() {
                return b.overlaps(index, query);
            }
            let shape = &track.shapes[index];
            !(shape.max.x < left || shape.min.x > right || shape.max.y < top || shape.min.y > bottom)
        };
        scratch.broad_pairs.retain(|&index| overlaps(index));
        scratch
            .separating_axes
            .retain(|index, _| scratch.broad_pairs.contains(index));
        for &index in &scratch.candidates {
            let index = index as usize;
            if overlaps(index) && !scratch.broad_pairs.contains(&index) {
                scratch.broad_pairs.push(index);
            }
        }
        scratch.pair_check_pending = false;
    }
    for ci in 0..scratch.broad_pairs.len() {
        let shape_index = scratch.broad_pairs[ci];
        let shape = &track.shapes[shape_index];
        let wall_first = if scratch.external_pairs {
            scratch.pair_orientation[ci]
        } else {
            shape.wall_first
        };
        let wall_origin = shape.origin_f32;
        // BodyPair uses the origin of the body with the lower BVH handle.
        let pair_origin = if wall_first { wall_origin } else { car_origin };
        let car_offset = F2::from(car_origin) - F2::from(pair_origin);
        let wall_offset = F2::from(wall_origin) - F2::from(pair_origin);
        let shape_origin =
            body_basis.0 * cfg.shape_position.x as f32 + body_basis.1 * cfg.shape_position.y as f32 + car_offset;
        let car = car_local.map(|p| V2::from(F2::from(p) + shape_origin));
        let wall = &mut scratch.wall;
        wall.clear();
        wall.extend(shape.local.iter().map(|p| V2::from(F2::from(*p) + wall_offset)));
        let axes = &mut scratch.axes;
        axes.clear();
        let separating_axis = scratch.separating_axes.entry(shape_index).or_insert(V2::ZERO);
        if *separating_axis != V2::ZERO {
            axes.push(*separating_axis);
        }
        axes.extend_from_slice(&car_axes);
        let n = wall.len();
        for i in 0..n {
            let edge = V2::new(
                f32r(wall[(i + 1) % n].x - wall[i].x),
                f32r(wall[(i + 1) % n].y - wall[i].y),
            );
            let edge_normalized = normalize_f32(edge);
            let axis = V2::new(edge_normalized.y, -edge_normalized.x);
            axes.push(axis);
        }
        let mut best_depth = 1e15f32 as f64;
        let mut push = V2::ZERO;
        let mut separated = false;
        for &axis in axes.iter() {
            let axis = if axis.x.abs() < CMP_EPSILON && axis.y.abs() < CMP_EPSILON {
                V2::new(0.0, 1.0)
            } else {
                axis
            };
            let (mut min_a, mut max_a) = (f64::INFINITY, f64::NEG_INFINITY);
            for &p in &car {
                let d = dot_f32(p, axis);
                min_a = py_min(min_a, d);
                max_a = py_max(max_a, d);
            }
            let (mut min_b, mut max_b) = (f64::INFINITY, f64::NEG_INFINITY);
            for &p in wall.iter() {
                let d = dot_f32(p, axis);
                min_b = py_min(min_b, d);
                max_b = py_max(max_b, d);
            }
            let width_a = f32r(f32r(max_a - min_a) * 0.5);
            let center_a = f32r(f32r(min_a + max_a) * 0.5);
            let dmin = f32r(f32r(min_b - width_a) - center_a);
            let dmax = f32r(f32r(max_b + width_a) - center_a);
            if dmin > 0.0 || dmax < 0.0 {
                *separating_axis = axis;
                separated = true;
                break;
            }
            let (depth, candidate) = if dmax < dmin.abs() {
                (dmax, axis)
            } else {
                (dmin.abs(), -axis)
            };
            if depth < best_depth {
                best_depth = depth;
                push = candidate;
            }
        }
        if separated || push == V2::ZERO {
            continue;
        }
        let support_car = rectangle_supports(&car, -push, basis_x, basis_y);
        let support_wall = polygon_supports(wall, shape, push);
        let mut pairs = [(V2::ZERO, V2::ZERO); 2];
        let count = contact_pairs(support_car.slice(), support_wall.slice(), push, &mut pairs);
        *separating_axis = V2::ZERO;
        for &(car_point, wall_point) in &pairs[..count] {
            // The pair stores normalize(A - B), then reports its negation to A.
            // Reversing subtraction algebraically loses native negative zeros.
            let (a, b) = if wall_first {
                (wall_point, car_point)
            } else {
                (car_point, wall_point)
            };
            let delta = V2::new(f32r(a.x - b.x), f32r(a.y - b.y));
            let depth = f32r(dot_f32(delta, delta).sqrt());
            if depth > 0.0 {
                let mut normal = V2::new(f32r(delta.x / depth), f32r(delta.y / depth));
                if !wall_first {
                    normal = -normal;
                }
                // BodyPair stores local points, then transforms them again in
                // pre_solve. The two float32 round trips affect the lever arm.
                let (x, y) = body_basis;
                let relative = F2::from(car_point) - car_offset;
                let local = F2 {
                    x: x.dot(relative),
                    y: y.dot(relative),
                };
                let global_a = (x * local.x + y * local.y) + car_offset;
                let global_b = (F2::from(wall_point) - wall_offset) + wall_offset;
                let depth = (global_b - global_a).dot(normal.into()) as f64;
                let point = V2::from(global_a + F2::from(pair_origin));
                let other = V2::from(global_b + F2::from(pair_origin));
                contacts.push(Contact {
                    normal,
                    point,
                    depth,
                    shape_index,
                    wall_first,
                    wall_point: other,
                    local_point: local.into(),
                    wall_local_point: (F2::from(wall_point) - wall_offset).into(),
                    used: true,
                    ..Contact::default()
                });
            }
        }
    }
}

fn refresh_contact(c: &mut Contact, track: &Track, position: V2, (x, y): (F2, F2)) {
    let shape = &track.shapes[c.shape_index];
    let origin = if c.wall_first { shape.origin_f32 } else { position };
    let car_offset = F2::from(position) - F2::from(origin);
    let wall_offset = F2::from(shape.origin_f32) - F2::from(origin);
    let car = (x * c.local_point.x as f32 + y * c.local_point.y as f32) + car_offset;
    let wall = F2::from(c.wall_local_point) + wall_offset;
    c.depth = (wall - car).dot(c.normal.into()) as f64;
    c.point = V2::from(car + F2::from(origin));
    c.wall_point = V2::from(wall + F2::from(origin));
}

pub(crate) struct SolveResult {
    pub(crate) velocity: V2,
    pub(crate) angular_velocity: f64,
    pub(crate) bias_velocity: V2,
    pub(crate) bias_angular: f64,
    pub(crate) collided: bool,
}

/// Resolve every persistent pair contact, then return the native capped reports.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_wall_contacts(
    cfg: &VehicleConfig,
    track: &Track,
    position: V2,
    rotation: f64,
    body_basis: (F2, F2),
    velocity: V2,
    angular_velocity: f64,
    initial_velocity: F2,
    initial_angular: f32,
    dt: f64,
    scratch: &mut CollisionScratch,
    contacts: &mut Vec<Contact>,
    previous: &mut Vec<Contact>,
) -> SolveResult {
    // Godot retains two local contacts per pair, including points that are
    // temporarily inactive. Validate before adding this tick's SAT contacts.
    let mut i = 0;
    while i < previous.len() {
        refresh_contact(&mut previous[i], track, position, body_basis);
        let c = &previous[i];
        let pair_origin = if c.wall_first {
            track.shapes[c.shape_index].origin_f32
        } else {
            position
        };
        let car_offset = F2::from(position) - F2::from(pair_origin);
        let car_point = body_basis.0 * c.local_point.x as f32 + body_basis.1 * c.local_point.y as f32 + car_offset;
        let wall_offset = F2::from(track.shapes[c.shape_index].origin_f32) - F2::from(pair_origin);
        let wall_point = F2::from(c.wall_local_point) + wall_offset;
        let separation = wall_point - F2::from(c.normal) * c.depth as f32 - car_point;
        if !c.used
            || c.depth < -cfg.contact_max_separation
            || separation.dot(separation) > (cfg.contact_max_separation as f32).powi(2)
        {
            let shape = c.shape_index;
            let last = previous.iter().rposition(|p| p.shape_index == shape).unwrap();
            previous[i] = previous[last];
            previous.remove(last);
        } else {
            previous[i].used = false;
            i += 1;
        }
    }
    find_contacts_with_basis(cfg, track, position, rotation, body_basis, scratch, contacts);
    previous.retain(|c| scratch.broad_pairs.contains(&c.shape_index));
    scratch.collided_shapes.clear();
    for new in contacts.iter() {
        if !scratch.collided_shapes.contains(&new.shape_index) {
            scratch.collided_shapes.push(new.shape_index);
        }
        let radius2 = (cfg.contact_recycle_radius as f32).powi(2);
        if let Some(old) = previous.iter_mut().find(|old| {
            let a = F2::from(old.local_point) - F2::from(new.local_point);
            let b = F2::from(old.wall_local_point) - F2::from(new.wall_local_point);
            old.shape_index == new.shape_index && a.dot(a) < radius2 && b.dot(b) < radius2
        }) {
            let mut replacement = *new;
            replacement.acc_normal = old.acc_normal;
            replacement.acc_tangent = old.acc_tangent;
            replacement.acc_bias = old.acc_bias;
            replacement.acc_bias_center = old.acc_bias_center;
            *old = replacement;
        } else if previous.iter().filter(|c| c.shape_index == new.shape_index).count() < 2 {
            previous.push(*new);
        } else {
            let mut least = None;
            let mut depth = new.depth;
            for (i, c) in previous
                .iter()
                .enumerate()
                .filter(|(_, c)| c.shape_index == new.shape_index)
            {
                if c.depth < depth {
                    depth = c.depth;
                    least = Some(i);
                }
            }
            if let Some(i) = least {
                previous[i] = *new;
            }
        }
    }
    contacts.clear();
    scratch.active_indices.clear();
    for &shape in &scratch.broad_pairs {
        for (i, c) in previous.iter().enumerate().filter(|(_, c)| c.shape_index == shape) {
            if c.depth > 0.0 && scratch.collided_shapes.contains(&c.shape_index) {
                contacts.push(*c);
                scratch.active_indices.push(i);
            }
        }
    }
    if contacts.is_empty() {
        return SolveResult {
            velocity,
            angular_velocity,
            bias_velocity: V2::ZERO,
            bias_angular: 0.0,
            collided: false,
        };
    }
    let inv_mass = 1.0f32 / cfg.mass as f32;
    let inv_inertia = 1.0f32 / cfg.inertia as f32;
    let inv_dt = 1.0f32 / dt as f32;
    let mut v = F2::from(velocity);
    let mut w = angular_velocity as f32;
    let center = body_basis.0 * cfg.center_of_mass.x as f32 + body_basis.1 * cfg.center_of_mass.y as f32;
    let contact_arm = |contact: &Contact| {
        let local = body_basis.0 * contact.local_point.x as f32 + body_basis.1 * contact.local_point.y as f32;
        let offset = if contact.wall_first {
            F2::from(position) - F2::from(track.shapes[contact.shape_index].origin_f32)
        } else {
            F2::default()
        };
        ((local + offset) - center) - offset
    };
    for contact in contacts.iter_mut() {
        let arm = contact_arm(contact);
        let n = F2::from(contact.normal);
        let t = F2 { x: -n.y, y: n.x };
        // GodotBodyPair2D::pre_solve uses squared length minus squared dot,
        // not the algebraically equivalent squared cross product.
        let rn = arm.dot(n);
        let rt = arm.dot(t);
        contact.normal_mass = 1.0 / (inv_mass + inv_inertia * (arm.dot(arm) - rn * rn));
        contact.tangent_mass = 1.0 / (inv_mass + inv_inertia * (arm.dot(arm) - rt * rt));
        contact.bias =
            -(cfg.contact_bias as f32) * inv_dt * (-(contact.depth as f32) + cfg.allowed_penetration as f32).min(0.0);
        let rotational = F2 {
            x: -arm.y * initial_angular,
            y: arm.x * initial_angular,
        };
        contact.bounce = (cfg.bounce as f32 + track.shapes[contact.shape_index].bounce as f32).clamp(0.0, 1.0);
        if contact.bounce != 0.0 {
            contact.bounce *= (initial_velocity + rotational).dot(n);
        }
        contact.friction = (cfg.friction as f32)
            .min(track.shapes[contact.shape_index].friction as f32)
            .abs();
        let impulse = n * contact.acc_normal + t * contact.acc_tangent;
        v = v + impulse * inv_mass;
        w += ((arm + center) - center).cross(impulse) * inv_inertia;
    }
    let mut bv = F2::default();
    let mut bw = 0.0f32;
    let bias_limit = (MAX_BIAS_ROTATION / dt as f32 as f64) as f32;
    for _ in 0..cfg.solver_iterations {
        for contact in contacts.iter_mut() {
            let arm = contact_arm(contact);
            let impulse_arm = (arm + center) - center;
            let n = F2::from(contact.normal);
            let t = F2 { x: -n.y, y: n.x };
            let rotational = F2 { x: -arm.y, y: arm.x };
            let velocity_at_contact = v + rotational * w;
            let normal_speed = velocity_at_contact.dot(n);
            let tangent_speed = velocity_at_contact.dot(t);
            let biased_normal_speed = (bv + rotational * bw).dot(n);
            let bias_impulse = (contact.bias - biased_normal_speed) * contact.normal_mass;
            let previous_bias = contact.acc_bias;
            contact.acc_bias = (previous_bias + bias_impulse).max(0.0);
            let impulse = n * (contact.acc_bias - previous_bias);
            bv = bv + impulse * inv_mass;
            bw = (bw + impulse_arm.cross(impulse) * inv_inertia).clamp(-bias_limit, bias_limit);
            let biased_normal_speed = (bv + rotational * bw).dot(n);
            if (contact.bias - biased_normal_speed).abs() > 0.001 {
                let center_impulse = (contact.bias - biased_normal_speed) / inv_mass;
                let previous_center = contact.acc_bias_center;
                contact.acc_bias_center = (previous_center + center_impulse).max(0.0);
                bv = bv + (n * (contact.acc_bias_center - previous_center)) * inv_mass;
            }
            let normal_impulse = -(normal_speed + contact.bounce) * contact.normal_mass;
            let previous_normal = contact.acc_normal;
            contact.acc_normal = (previous_normal + normal_impulse).max(0.0);
            let friction_limit = contact.friction * contact.acc_normal;
            let tangent_impulse = -tangent_speed * contact.tangent_mass;
            let previous_tangent = contact.acc_tangent;
            contact.acc_tangent = (previous_tangent + tangent_impulse).clamp(-friction_limit, friction_limit);
            let impulse = n * (contact.acc_normal - previous_normal) + t * (contact.acc_tangent - previous_tangent);
            v = v + impulse * inv_mass;
            w += impulse_arm.cross(impulse) * inv_inertia;
        }
    }
    for (c, &i) in contacts.iter().zip(&scratch.active_indices) {
        previous[i] = *c;
    }
    // Reporting is independent of solving and of the two-contact pair cache.
    // Godot replaces the first shallowest slot only for a strictly deeper hit.
    let limit = cfg.max_contacts_reported;
    if limit == 0 {
        contacts.clear();
    } else if contacts.len() > limit {
        for incoming in limit..contacts.len() {
            let mut shallowest = 0;
            for slot in 1..limit {
                if contacts[slot].depth < contacts[shallowest].depth {
                    shallowest = slot;
                }
            }
            if contacts[shallowest].depth < contacts[incoming].depth {
                contacts[shallowest] = contacts[incoming];
            }
        }
        contacts.truncate(limit);
    }
    SolveResult {
        velocity: v.into(),
        angular_velocity: w as f64,
        bias_velocity: bv.into(),
        bias_angular: bw as f64,
        collided: !contacts.is_empty(),
    }
}

impl CollisionScratch {
    /// Fills the scratch fields of a GPU car (gpu/sim/state.h) and its
    /// retained contacts; errors on states the GPU port does not model.
    pub(crate) fn gpu_export(
        &self,
        c: &mut crate::gpu::simulation::GpuCar,
        previous: &[Contact],
    ) -> Result<(), String> {
        use crate::gpu::simulation::*;
        if self.broadphase.is_some() || self.external_pairs {
            return Err("native broadphase pairs are not supported".into());
        }
        if self.broad_pairs.len() > MAX_PAIRS || previous.len() > MAX_CONTACTS {
            return Err(format!(
                "{} pairs / {} contacts exceed the GPU limits",
                self.broad_pairs.len(),
                previous.len()
            ));
        }
        if self.separating_axes.keys().any(|k| !self.broad_pairs.contains(k)) {
            return Err("separating axis without a broad pair".into());
        }
        c.pair_count = self.broad_pairs.len() as u32;
        for (i, &index) in self.broad_pairs.iter().enumerate() {
            c.pairs[i] = index as u32;
            c.axes[i] = exact2(
                self.separating_axes.get(&index).copied().unwrap_or(V2::ZERO),
                "separating axis",
            )?;
        }
        if let Some(r) = self.shape_aabb {
            c.flags |= CAR_HAS_SHAPE;
            c.shape_position = f2(r.position);
            c.shape_size = f2(r.size);
        }
        if let Some(a) = self.leaf_aabb {
            c.flags |= CAR_HAS_LEAF;
            c.leaf_min = f2(a.min);
            c.leaf_max = f2(a.max);
        }
        if self.pair_check_pending {
            c.flags |= CAR_PAIR_CHECK;
        }
        c.contact_count = previous.len() as u32;
        for (g, p) in c.contacts.iter_mut().zip(previous) {
            *g = GpuContact {
                normal: exact2(p.normal, "contact normal")?,
                point: exact2(p.point, "contact point")?,
                wall_point: exact2(p.wall_point, "contact wall point")?,
                local_point: exact2(p.local_point, "contact local point")?,
                wall_local_point: exact2(p.wall_local_point, "contact wall local point")?,
                depth: exact(p.depth, "contact depth")?,
                shape: p.shape_index as u32,
                flags: if p.wall_first { CONTACT_WALL_FIRST } else { 0 } | if p.used { CONTACT_USED } else { 0 },
                normal_mass: p.normal_mass,
                tangent_mass: p.tangent_mass,
                bias: p.bias,
                bounce: p.bounce,
                friction: p.friction,
                acc_normal: p.acc_normal,
                acc_tangent: p.acc_tangent,
                acc_bias: p.acc_bias,
                acc_bias_center: p.acc_bias_center,
            };
        }
        Ok(())
    }

    /// The inverse of `gpu_export`.
    pub(crate) fn gpu_import(&mut self, c: &crate::gpu::simulation::GpuCar, previous: &mut Vec<Contact>) {
        use crate::gpu::simulation::*;
        let n = c.pair_count as usize;
        self.broad_pairs = c.pairs[..n].iter().map(|&i| i as usize).collect();
        self.separating_axes = c.pairs[..n]
            .iter()
            .zip(&c.axes)
            .map(|(&i, &a)| (i as usize, v2(a)))
            .collect();
        let fv = |v: Vec2| F2 { x: v[0], y: v[1] };
        self.shape_aabb = (c.flags & CAR_HAS_SHAPE != 0).then(|| FloatRect {
            position: fv(c.shape_position),
            size: fv(c.shape_size),
        });
        self.leaf_aabb = (c.flags & CAR_HAS_LEAF != 0).then(|| Aabb {
            min: fv(c.leaf_min),
            max: fv(c.leaf_max),
        });
        self.pair_check_pending = c.flags & CAR_PAIR_CHECK != 0;
        previous.clear();
        previous.extend(c.contacts[..c.contact_count as usize].iter().map(|g| Contact {
            normal: v2(g.normal),
            point: v2(g.point),
            depth: g.depth as f64,
            shape_index: g.shape as usize,
            wall_first: g.flags & CONTACT_WALL_FIRST != 0,
            wall_point: v2(g.wall_point),
            local_point: v2(g.local_point),
            wall_local_point: v2(g.wall_local_point),
            used: g.flags & CONTACT_USED != 0,
            normal_mass: g.normal_mass,
            tangent_mass: g.tangent_mass,
            bias: g.bias,
            bounce: g.bounce,
            friction: g.friction,
            acc_normal: g.acc_normal,
            acc_tangent: g.acc_tangent,
            acc_bias: g.acc_bias,
            acc_bias_center: g.acc_bias_center,
        }));
    }
}
