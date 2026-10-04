// BSP queries of src/track/bsp.rs with explicit stacks instead of recursion:
// RayTree::raycast and Node::closest, which share the RayTree arena.
#pragma once
#include "world.h"
#include "math.h"

namespace altd {

// `Bounds::beside`: box (cx, cy, hx, hy) lies beyond `margin` on one side of the ray's line.
__device__ inline bool beside(float4 box, float2 start, float2 b, float margin) {
    float cross = b.x * (box.y - start.y) - b.y * (box.x - start.x);
    return fabsf(cross) > fabsf(b.x) * (box.w + margin) + fabsf(b.y) * (box.z + margin);
}

// `classify`: side of p relative to the wall (origin, direction).
__device__ inline int classify(float4 wall, float2 p) {
    float cross = wall.z * (p.y - wall.y) - wall.w * (p.x - wall.x);
    return cross > 0.0001f ? 1 : (cross < -0.0001f ? -1 : 0);
}

// `godot_math::ray_wall_hit`
__device__ inline bool ray_wall_hit(float2 start, float2 b, float4 wall, float2& hit) {
    float denominator = wall.z * b.y - b.x * wall.w;
    float dx = wall.x - start.x, dy = wall.y - start.y;
    float ray_fraction = ((0.0f - wall.w) * dx + wall.z * dy) / denominator;
    float wall_fraction = (b.x * dy - b.y * dx) / denominator;
    hit = make_float2(wall.x + wall.z * wall_fraction, wall.y + wall.w * wall_fraction);
    return !(fabsf(denominator) < 1e-6f) && ray_fraction >= 0.0f && ray_fraction <= 1.0f
        && wall_fraction >= 0.0f && wall_fraction <= 1.0f;
}

// `RayTree::raycast`. The recursion "near subtree, else own walls, else the
// far subtree if the ray crosses the partition" becomes a descent loop that
// pushes each entered node, and a pop that tests the node's own walls and
// then continues into its far child.
__device__ inline bool raycast(const World& w, float2 start, float2 end, float2& hit) {
    float2 b = make_float2(end.x - start.x, end.y - start.y);
    float scale = w.ray_magnitude + fabsf(start.x) + fabsf(start.y);
    float margin = fabsf(b.x) + fabsf(b.y) > 1e-10f ? 1e-3f + 2e-5f * scale : INFINITY;
    uint32_t stack[MAX_RAY_DEPTH];
    int depth = 0;
    uint32_t current = w.ray_node_count ? 0 : NONE;
    for (;;) {
        while (current != NONE) {
            const RayNode& node = w.ray_nodes[current];
            if (beside(node.subtree, start, b, margin)) break;
            float4 partition = w.ray_walls[node.first];
            int a = classify(partition, start), e = classify(partition, end);
            bool front_first = a == 1 || (a == 0 && e == 1);
            // Low bit: the ray crosses the partition, so the far side follows.
            stack[depth++] = current << 1 | (a != e);
            current = front_first ? node.front : node.back;
        }
        if (depth == 0) return false;
        uint32_t entry = stack[--depth];
        const RayNode& node = w.ray_nodes[entry >> 1];
        if (!beside(node.own, start, b, margin)) {
            float distance = 3.40282347e38f;
            bool found = false;
            for (uint32_t i = node.first; i < node.last; i++) {
                float2 p;
                bool ok = ray_wall_hit(start, b, w.ray_walls[i], p);
                float dx = p.x - start.x, dy = p.y - start.y;
                float d2 = dx * dx + dy * dy;
                bool better = ok && d2 < distance;
                distance = better ? d2 : distance;
                hit = better ? p : hit;
                found |= better;
            }
            if (found) return true;
        }
        float4 partition = w.ray_walls[node.first];
        bool front_first = classify(partition, start) == 1
            || (classify(partition, start) == 0 && classify(partition, end) == 1);
        current = (entry & 1) ? (front_first ? node.back : node.front) : NONE;
    }
}

// `Node::closest`: own walls, then the near subtree, then the far subtree
// when the partition line is closer than the best wall so far.
__device__ inline float2 closest_wall_point(const World& w, float2 p) {
    float distance = INFINITY;
    float2 closest = make_float2(0.0f, 0.0f);
    uint32_t stack[MAX_RAY_DEPTH];
    int depth = 0;
    uint32_t current = w.ray_node_count ? 0 : NONE;
    for (;;) {
        while (current != NONE) {
            const RayNode& node = w.ray_nodes[current];
            for (uint32_t i = node.first; i < node.last; i++) {
                float4 wall = w.ray_walls[i];
                float len2 = wall.z * wall.z + wall.w * wall.w;
                float t = (p.x * wall.z + p.y * wall.w) - (wall.x * wall.z + wall.y * wall.w);
                t = t < 0.0f ? 0.0f : (t > len2 ? len2 : t);  // f32::clamp
                t = t / len2;
                float2 candidate = len2 == 0.0f ? make_float2(wall.x, wall.y)
                                                : make_float2(wall.x + wall.z * t, wall.y + wall.w * t);
                float dx = p.x - candidate.x, dy = p.y - candidate.y;
                float d2 = dx * dx + dy * dy;
                bool better = d2 < distance;
                distance = better ? d2 : distance;
                closest = better ? candidate : closest;
            }
            stack[depth++] = current;
            current = classify(w.ray_walls[node.first], p) >= 0 ? node.front : node.back;
        }
        if (depth == 0) return closest;
        const RayNode& node = w.ray_nodes[stack[--depth]];
        float4 partition = w.ray_walls[node.first];
        float t = ((p.x - partition.x) * partition.z + (p.y - partition.y) * partition.w)
            / (partition.z * partition.z + partition.w * partition.w);
        float dx = p.x - (partition.x + partition.z * t), dy = p.y - (partition.y + partition.w * t);
        bool far = dx * dx + dy * dy < distance;
        current = far ? (classify(partition, p) >= 0 ? node.back : node.front) : NONE;
    }
}

}  // namespace altd
