//! Game BspTreeRaycaster construction and traversal, using managed float order.
use crate::{godot_math::{ray_intersection, ray_wall_hit, F2}, vec2::V2, world::Wall};
use serde_json::Value;

pub struct Node {
    partition: Wall,
    segments: Vec<Wall>,
    front: Option<Box<Node>>,
    back: Option<Box<Node>>,
}
fn classify(p: V2, wall: Wall) -> i8 {
    let a = F2::from(wall.end) - F2::from(wall.start);
    let b = F2::from(p) - F2::from(wall.start);
    let cross = a.cross(b);
    if cross > 0.0001 { 1 } else if cross < -0.0001 { -1 } else { 0 }
}
fn sides(wall: Wall, partition: Wall) -> (i8, i8) {
    (classify(wall.start, partition), classify(wall.end, partition))
}
impl Node {
    pub fn from_polygons(polygons: &[Value]) -> Option<Box<Node>> {
        let mut segments = Vec::new();
        for polygon in polygons {
            let points: Vec<V2> = polygon["points"].as_array().unwrap().iter().map(|p|
                V2::new(p[0].as_f64().unwrap() as f32 as f64, p[1].as_f64().unwrap() as f32 as f64)).collect();
            for i in 0..points.len() {
                let wall = Wall { start: points[i], end: points[(i + 1) % points.len()] };
                if wall.start != wall.end { segments.push(wall); }
            }
        }
        Self::build(segments)
    }
    fn build(walls: Vec<Wall>) -> Option<Box<Node>> {
        if walls.is_empty() { return None; }
        let mut best = i32::MAX;
        let mut partition = walls[0];
        for &candidate in &walls {
            let (mut front, mut back, mut split) = (0i32, 0i32, 0i32);
            for &wall in &walls {
                if wall == candidate { continue; }
                let (a, b) = sides(wall, candidate);
                if a == 0 && b == 0 { continue; }
                if a >= 0 && b >= 0 { front += 1; }
                else if a <= 0 && b <= 0 { back += 1; }
                else { split += 1; }
            }
            let score = (front - back).abs() + split * 10;
            if score < best { best = score; partition = candidate; }
        }
        let (mut front, mut back, mut segments) = (Vec::new(), Vec::new(), Vec::new());
        for wall in walls {
            if wall == partition { continue; }
            let (a, b) = sides(wall, partition);
            if a == 0 && b == 0 { segments.push(wall); }
            else if a >= 0 && b >= 0 { front.push(wall); }
            else if a <= 0 && b <= 0 { back.push(wall); }
            else {
                let start = F2::from(partition.start);
                let delta = F2::from(partition.end) - start;
                let other = F2::from(wall.end) - F2::from(wall.start);
                let denominator = delta.x * other.y - other.x * delta.y;
                let d = start - F2::from(wall.start);
                let intersection: V2 = if denominator.abs() < 1e-6 { V2::ZERO } else {
                    (start + delta * ((other.x * d.y - other.y * d.x) / denominator)).into()
                };
                if a == 1 {
                    front.push(Wall { start: wall.start, end: intersection });
                    back.push(Wall { start: intersection, end: wall.end });
                } else {
                    front.push(Wall { start: wall.end, end: intersection });
                    back.push(Wall { start: intersection, end: wall.start });
                }
            }
        }
        Some(Box::new(Node { partition, segments, front: Self::build(front), back: Self::build(back) }))
    }
    pub fn collect(&self, out: &mut Vec<Wall>) {
        out.push(self.partition); out.extend(&self.segments);
        if let Some(node) = &self.front { node.collect(out); }
        if let Some(node) = &self.back { node.collect(out); }
    }
    pub fn closest(&self, point: V2) -> V2 {
        let mut distance = f64::MAX;
        let mut closest = F2::default();
        self.closest_recursive(point, &mut distance, &mut closest);
        closest.into()
    }
    fn closest_recursive(&self, point: V2, distance: &mut f64, closest: &mut F2) {
        let p = F2::from(point);
        for wall in std::iter::once(&self.partition).chain(&self.segments) {
            let a = F2::from(wall.start);
            let delta = F2::from(wall.end)-a;
            let len2 = delta.dot(delta);
            let candidate = if len2==0.0 { a } else { a+delta*((p.dot(delta)-a.dot(delta)).clamp(0.0,len2)/len2) };
            let d = p-candidate;
            if (d.dot(d) as f64) < *distance { *distance = d.dot(d) as f64; *closest = candidate; }
        }
        let (near,far) = if classify(point,self.partition)>=0 { (&self.front,&self.back) } else { (&self.back,&self.front) };
        if let Some(n)=near { n.closest_recursive(point,distance,closest); }
        let a=F2::from(self.partition.start);
        let delta=F2::from(self.partition.end)-a;
        let t=(p-a).dot(delta)/delta.dot(delta);
        let d=p-(a+delta*t);
        if (d.dot(d) as f64) < *distance { if let Some(n)=far { n.closest_recursive(point,distance,closest); } }
    }
    pub fn raycast(&self, start: V2, end: V2) -> Option<V2> {
        let a = classify(start, self.partition);
        let b = classify(end, self.partition);
        let (near, far) = if a == 1 || a == 0 && b == 1 { (&self.front, &self.back) } else { (&self.back, &self.front) };
        if let Some(hit) = near.as_ref().and_then(|n| n.raycast(start, end)) { return Some(hit); }
        let mut nearest = None;
        let mut distance = f32::MAX;
        for wall in std::iter::once(&self.partition).chain(&self.segments) {
            if let Some(p) = ray_intersection(start, end, wall.start, wall.end) {
                let d = F2::from(p) - F2::from(start);
                let d2 = d.dot(d);
                if d2 < distance { distance = d2; nearest = Some(p); }
            }
        }
        nearest.or_else(|| if a != b { far.as_ref().and_then(|n| n.raycast(start, end)) } else { None })
    }
}

const NONE: u32 = u32::MAX;

/// `Node` for raycasts: one arena, float32 wall geometry, and box tests that
/// skip exactly the walls `ray_intersection` rejects for being well beside the ray.
pub struct RayTree {
    nodes: Vec<RayNode>,
    /// Wall start and `end - start`; each node's partition, then its segments.
    walls: Vec<(F2, F2)>,
    /// Bound on `|x| + |y|` over wall endpoints.
    magnitude: f32,
}
struct RayNode {
    first: u32,
    last: u32,
    front: u32,
    back: u32,
    /// The node's own walls, and those of its whole subtree.
    own: Bounds,
    subtree: Bounds,
}
/// Box center and half extents, covering its wall endpoints.
#[derive(Clone, Copy)]
struct Bounds {
    cx: f32,
    cy: f32,
    hx: f32,
    hy: f32,
}
struct Ray {
    start: F2,
    end: F2,
    b: F2,
    margin: f32,
}

impl Bounds {
    fn new(lo: (f64, f64), hi: (f64, f64)) -> Bounds {
        // Round the center, then widen the half extent to still cover the walls.
        let half = |lo: f64, hi: f64| {
            let center = ((lo + hi) * 0.5) as f32;
            let half = (hi - center as f64).max(center as f64 - lo);
            let mut rounded = half as f32;
            if (rounded as f64) < half { rounded = f32::from_bits(rounded.to_bits() + 1); }
            (center, rounded)
        };
        let ((cx, hx), (cy, hy)) = (half(lo.0, hi.0), half(lo.1, hi.1));
        Bounds { cx, cy, hx, hy }
    }

    /// True when the box lies beyond `margin` from the ray's infinite line,
    /// entirely on one side. `|cross(b, p - start)|` is `|b|` times the
    /// distance of `p`; over the box it varies by `|b.x| hy + |b.y| hx`, and
    /// `|b.x| + |b.y| >= |b|`.
    #[inline]
    fn beside(&self, ray: &Ray) -> bool {
        let b = ray.b;
        let cross = b.x * (self.cy - ray.start.y) - b.y * (self.cx - ray.start.x);
        cross.abs() > b.x.abs() * (self.hy + ray.margin) + b.y.abs() * (self.hx + ray.margin)
    }
}

impl RayTree {
    pub fn new(root: &Node) -> RayTree {
        let mut tree = RayTree { nodes: Vec::new(), walls: Vec::new(), magnitude: 0.0 };
        tree.push(root);
        tree
    }

    /// Appends `node` and its subtree in preorder; returns its index and the
    /// subtree's wall extent.
    fn push(&mut self, node: &Node) -> (u32, (f64, f64), (f64, f64)) {
        let index = self.nodes.len() as u32;
        let first = self.walls.len() as u32;
        let (mut lo, mut hi) = ((f64::INFINITY, f64::INFINITY), (f64::NEG_INFINITY, f64::NEG_INFINITY));
        for wall in std::iter::once(&node.partition).chain(&node.segments) {
            let (start, end) = (F2::from(wall.start), F2::from(wall.end));
            self.walls.push((start, end - start));
            for p in [start, end] {
                let (x, y) = (p.x as f64, p.y as f64);
                lo = (lo.0.min(x), lo.1.min(y));
                hi = (hi.0.max(x), hi.1.max(y));
                self.magnitude = self.magnitude.max((x.abs() + y.abs()) as f32 * (1.0 + f32::EPSILON));
            }
        }
        let own = Bounds::new(lo, hi);
        self.nodes.push(RayNode { first, last: self.walls.len() as u32, front: NONE, back: NONE, own, subtree: own });
        for (child, front) in [(&node.front, true), (&node.back, false)] {
            let Some(child) = child else { continue };
            let (child, child_lo, child_hi) = self.push(child);
            lo = (lo.0.min(child_lo.0), lo.1.min(child_lo.1));
            hi = (hi.0.max(child_hi.0), hi.1.max(child_hi.1));
            let node = &mut self.nodes[index as usize];
            if front { node.front = child } else { node.back = child }
        }
        self.nodes[index as usize].subtree = Bounds::new(lo, hi);
        (index, lo, hi)
    }

    /// Flat arrays for the GPU port (gpu/sim/rays.h): per node `[first, last,
    /// front, back]` and the own and subtree boxes as `[cx, cy, hx, hy]`; per
    /// wall `[start.x, start.y, delta.x, delta.y]`; the coordinate bound; and
    /// the tree depth (nodes on the longest root-to-leaf path).
    pub fn gpu_arrays(&self) -> (Vec<([u32; 4], [f32; 4], [f32; 4])>, Vec<[f32; 4]>, f32, usize) {
        let nodes: Vec<_> = self.nodes.iter().map(|n| {
            let b = |b: &Bounds| [b.cx, b.cy, b.hx, b.hy];
            ([n.first, n.last, n.front, n.back], b(&n.own), b(&n.subtree))
        }).collect();
        let walls = self.walls.iter().map(|(s, d)| [s.x, s.y, d.x, d.y]).collect();
        fn depth(nodes: &[RayNode], i: u32) -> usize {
            if i == NONE { return 0; }
            let n = &nodes[i as usize];
            1 + depth(nodes, n.front).max(depth(nodes, n.back))
        }
        let d = if self.nodes.is_empty() { 0 } else { depth(&self.nodes, 0) };
        (nodes, walls, self.magnitude, d)
    }

    /// `Node::raycast`.
    ///
    /// Skipping is exact: a wall is hit only if the float32 wall fraction
    /// `N / D` lies in [0, 1], where `N = cross(b, ws - start)` and
    /// `D = cross(a, b)`. `N` and `N - D` are `|b|` times the signed distances
    /// of the wall's endpoints from the ray's line. With both of one sign and
    /// at least `|b| M` in size, either `D` has the opposite sign (fraction
    /// below 0) or `|N| - |D| >= |b| M` (fraction above 1). With unit
    /// roundoff `u = 2^-24` and coordinates bounded by `C` (walls) and `S`
    /// (ray start), float32 errs by under `12u |b| (C + S)` in `N` and
    /// `8u |a| |b|` in `D`, and the quotient rounds to 1 only within `2u |D|`
    /// of it; `M > 32u (C + S)` therefore rejects every such wall. `beside`
    /// errs by under `10u |b| (C + S + margin)`, so `margin` keeps `M` about
    /// tenfold above that bound.
    pub fn raycast(&self, start: V2, end: V2) -> Option<V2> {
        let (start, end) = (F2::from(start), F2::from(end));
        let b = end - start;
        let scale = self.magnitude + start.x.abs() + start.y.abs();
        // Tiny rays would underflow the bound; they are never skipped.
        let margin = if b.x.abs() + b.y.abs() > 1e-10 { 1e-3 + 2e-5 * scale } else { f32::INFINITY };
        if self.nodes.is_empty() { return None; }
        self.cast(0, &Ray { start, end, b, margin }).map(V2::from)
    }

    fn cast(&self, index: u32, ray: &Ray) -> Option<F2> {
        let node = &self.nodes[index as usize];
        if node.subtree.beside(ray) { return None; }
        let (origin, direction) = self.walls[node.first as usize];
        // `classify`
        let side = |p: F2| {
            let cross = direction.cross(p - origin);
            if cross > 0.0001 { 1 } else if cross < -0.0001 { -1 } else { 0 }
        };
        let (a, b) = (side(ray.start), side(ray.end));
        let (near, far) = if a == 1 || a == 0 && b == 1 { (node.front, node.back) } else { (node.back, node.front) };
        if near != NONE {
            if let Some(hit) = self.cast(near, ray) { return Some(hit); }
        }
        let mut nearest = None;
        if !node.own.beside(ray) {
            let mut distance = f32::MAX;
            for &(ws, wa) in &self.walls[node.first as usize..node.last as usize] {
                if let Some(p) = ray_wall_hit(ray.start, ray.b, ws, wa) {
                    let d = p - ray.start;
                    let d2 = d.dot(d);
                    if d2 < distance { distance = d2; nearest = Some(p); }
                }
            }
        }
        nearest.or_else(|| if a != b && far != NONE { self.cast(far, ray) } else { None })
    }
}
