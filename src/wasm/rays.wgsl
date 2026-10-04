// BSP wall queries of src/track/bsp.rs for WebGPU, matching gpu/sim/rays.h with
// the same descent and pop order, so a query visits the walls the CPU visits
// and keeps the first of equal hits. The arithmetic is written operation by
// operation as the CPU evaluates it. float.wgsl supplies rounding boundaries
// and integer binary32 division. The host prepends those helpers to this module.

struct RayNode {
    // first wall, one past the last wall, front child, back child (NONE = 0xffffffff)
    links: vec4<u32>,
    // (cx, cy, hx, hy): the node's own walls, and its whole subtree
    own: vec4<f32>,
    subtree: vec4<f32>,
}

struct Params {
    node_count: u32,
    count: u32,
    // 0: raycast (start, end) -> (hit, hit != 0, 0); 1: closest wall (point) -> (point, 1, 0)
    op: u32,
    pad: u32,
    magnitude: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodes: array<RayNode>;
// (start.x, start.y, delta.x, delta.y): each node's split, then its segments
@group(0) @binding(2) var<storage, read> walls: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> results: array<vec4<f32>>;

const NONE: u32 = 0xffffffffu;
const MAX_DEPTH: u32 = 48u;
const F32_MAX: f32 = 3.4028234663852886e38;

// `Bounds::beside`: the box lies beyond `margin` on one side of the ray's line.
fn beside(bounds: vec4<f32>, start: vec2<f32>, b: vec2<f32>, margin: f32) -> bool {
    let c = fp_sub(fp_mul(b.x, fp_sub(bounds.y, start.y)), fp_mul(b.y, fp_sub(bounds.x, start.x)));
    return abs(c) > fp_add(fp_mul(abs(b.x), fp_add(bounds.w, margin)), fp_mul(abs(b.y), fp_add(bounds.z, margin)));
}

// `classify`: the side of p relative to the wall (origin, direction).
fn classify(wall: vec4<f32>, p: vec2<f32>) -> i32 {
    let c = fp_sub(fp_mul(wall.z, fp_sub(p.y, wall.y)), fp_mul(wall.w, fp_sub(p.x, wall.x)));
    if (c > 0.0001) { return 1; }
    if (c < -0.0001) { return -1; }
    return 0;
}

fn front_first(split: vec4<f32>, start: vec2<f32>, end: vec2<f32>) -> bool {
    let a = classify(split, start);
    return a == 1 || (a == 0 && classify(split, end) == 1);
}

// `RayTree::raycast`: the near subtree, then the node's own walls, then the far
// subtree when the ray crosses the split. Tiny rays never skip a box
// (the CPU uses an infinite margin).
fn raycast(start: vec2<f32>, end: vec2<f32>) -> vec4<f32> {
    let b = vec2<f32>(fp_sub(end.x, start.x), fp_sub(end.y, start.y));
    let scale = fp_add(fp_add(params.magnitude, abs(start.x)), abs(start.y));
    let skip = fp_add(abs(b.x), abs(b.y)) > 1e-10;
    let margin = fp_add(1e-3, fp_mul(2e-5, scale));
    var stack: array<u32, MAX_DEPTH>;
    var depth: u32 = 0u;
    var current: u32 = select(NONE, 0u, params.node_count != 0u);
    var hit = vec2<f32>(0.0, 0.0);
    var pops: u32 = 0u;
    loop {
        loop {
            if (current == NONE || depth >= MAX_DEPTH) { break; }
            let node = nodes[current];
            if (skip && beside(node.subtree, start, b, margin)) { break; }
            let split = walls[node.links.x];
            let a = classify(split, start);
            let e = classify(split, end);
            let near_front = a == 1 || (a == 0 && e == 1);
            // Low bit: the ray crosses the split, so the far side follows.
            stack[depth] = (current << 1u) | select(0u, 1u, a != e);
            depth = depth + 1u;
            current = select(node.links.w, node.links.z, near_front);
        }
        if (depth == 0u || pops > params.node_count) { break; }
        pops = pops + 1u;
        depth = depth - 1u;
        let entry = stack[depth];
        let node = nodes[entry >> 1u];
        if (!(skip && beside(node.own, start, b, margin))) {
            var best_d2 = F32_MAX;
            var found = false;
            for (var i = node.links.x; i < node.links.y; i = i + 1u) {
                let wall = walls[i];
                // `godot_math::ray_wall_hit`
                let denominator = fp_sub(fp_mul(wall.z, b.y), fp_mul(b.x, wall.w));
                let dx = fp_sub(wall.x, start.x);
                let dy = fp_sub(wall.y, start.y);
                let ray_fraction = fp_div(fp_add(fp_mul(fp_sub(0.0, wall.w), dx), fp_mul(wall.z, dy)), denominator);
                let wall_fraction = fp_div(fp_sub(fp_mul(b.x, dy), fp_mul(b.y, dx)), denominator);
                let p = vec2<f32>(fp_add(wall.x, fp_mul(wall.z, wall_fraction)), fp_add(wall.y, fp_mul(wall.w, wall_fraction)));
                let ok = !(abs(denominator) < 1e-6) && ray_fraction >= 0.0 && ray_fraction <= 1.0
                    && wall_fraction >= 0.0 && wall_fraction <= 1.0;
                let ddx = fp_sub(p.x, start.x);
                let ddy = fp_sub(p.y, start.y);
                let d2 = fp_add(fp_mul(ddx, ddx), fp_mul(ddy, ddy));
                let better = ok && d2 < best_d2;
                best_d2 = select(best_d2, d2, better);
                hit = select(hit, p, better);
                found = found || better;
            }
            if (found) { return vec4<f32>(hit.x, hit.y, 1.0, 0.0); }
        }
        let split = walls[node.links.x];
        let far = select(node.links.z, node.links.w, front_first(split, start, end));
        current = select(NONE, far, (entry & 1u) == 1u);
    }
    // Keep the terminal return outside the loop for Naga's control-flow validation.
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

// `Node::closest`: own walls, then the near subtree, then the far subtree
// when the split line is closer than the best wall so far.
fn closest_wall(p: vec2<f32>) -> vec4<f32> {
    var best_d2 = F32_MAX;
    var found = false;
    var closest = vec2<f32>(0.0, 0.0);
    var stack: array<u32, MAX_DEPTH>;
    var depth: u32 = 0u;
    var current: u32 = select(NONE, 0u, params.node_count != 0u);
    var pops: u32 = 0u;
    loop {
        loop {
            if (current == NONE || depth >= MAX_DEPTH) { break; }
            let node = nodes[current];
            for (var i = node.links.x; i < node.links.y; i = i + 1u) {
                let wall = walls[i];
                let len2 = fp_add(fp_mul(wall.z, wall.z), fp_mul(wall.w, wall.w));
                var t = fp_sub(fp_add(fp_mul(p.x, wall.z), fp_mul(p.y, wall.w)), fp_add(fp_mul(wall.x, wall.z), fp_mul(wall.y, wall.w)));
                t = select(t, 0.0, t < 0.0);
                t = select(t, len2, t > len2);
                t = fp_div(t, len2);
                var candidate = vec2<f32>(fp_add(wall.x, fp_mul(wall.z, t)), fp_add(wall.y, fp_mul(wall.w, t)));
                candidate = select(candidate, vec2<f32>(wall.x, wall.y), len2 == 0.0);
                let dx = fp_sub(p.x, candidate.x);
                let dy = fp_sub(p.y, candidate.y);
                let d2 = fp_add(fp_mul(dx, dx), fp_mul(dy, dy));
                let better = !found || d2 < best_d2;
                best_d2 = select(best_d2, d2, better);
                closest = select(closest, candidate, better);
                found = true;
            }
            stack[depth] = current;
            depth = depth + 1u;
            current = select(node.links.w, node.links.z, classify(walls[node.links.x], p) >= 0);
        }
        if (depth == 0u || pops > params.node_count) { break; }
        pops = pops + 1u;
        depth = depth - 1u;
        let node = nodes[stack[depth]];
        let split = walls[node.links.x];
        let t = fp_div(fp_add(fp_mul(fp_sub(p.x, split.x), split.z), fp_mul(fp_sub(p.y, split.y), split.w)),
            fp_add(fp_mul(split.z, split.z), fp_mul(split.w, split.w)));
        let dx = fp_sub(p.x, fp_add(split.x, fp_mul(split.z, t)));
        let dy = fp_sub(p.y, fp_add(split.y, fp_mul(split.w, t)));
        let far_side = fp_add(fp_mul(dx, dx), fp_mul(dy, dy)) < best_d2;
        let far = select(node.links.z, node.links.w, classify(split, p) >= 0);
        current = select(NONE, far, far_side);
    }
    return vec4<f32>(closest.x, closest.y, 1.0, 0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) { return; }
    let q = queries[i];
    if (params.op == 0u) {
        results[i] = raycast(q.xy, q.zw);
    } else {
        results[i] = closest_wall(q.xy);
    }
}
