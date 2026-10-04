// Track queries of src/track/world.rs and src/track/curve.rs: tile surfaces, the path
// sensor point, Curve2D sampling and closest offsets, and closest_path.
#pragma once
#include <cfloat>
#include "vec.h"
#include "world.h"

namespace altd {

// `TileTable::get`; null outside the table.
__device__ inline const TileCell* tile_at(const World& w, Vec2 p) {
    int64_t x = sat_i64(floor((double)p.x / 768.0)) - w.tile_x0;
    int64_t y = sat_i64(floor((double)p.y / 768.0)) - w.tile_y0;
    if (x < 0 || y < 0 || x >= w.tile_nx || y >= w.tile_ny) return nullptr;
    return &w.tiles[y * w.tile_nx + x];
}

// `Track::vehicle_surface`, as an index into w.surfaces.
__device__ inline uint32_t vehicle_surface(const World& w, Vec2 p) {
    const TileCell* t = tile_at(w, p);
    return t ? t->surface : w.default_surface;
}

// `Track::path_sensor_point`
__device__ inline Vec2 path_sensor_point(const World& w, Vec2 p) {
    const TileCell* t = tile_at(w, p);
    if (!t || !t->connected) return p;
    Vec2 a = t->first;
    Vec2 v = t->last - a;
    float n = dot(v, v);
    Vec2 closest = n == 0.0f ? a : a + v * (rclamp(dot(p, v) - dot(a, v), 0.0f, n) / n);
    return closest + (p - closest) * 0.5f;
}

// Cell of `g` holding q, or NONE (outside or not finite).
__device__ inline uint32_t grid_cell(const NearGrid& g, Vec2 q) {
    double fx = floor(((double)q.x - g.x0) / g.cell), fy = floor(((double)q.y - g.y0) / g.cell);
    if (!(fx >= 0.0 && fy >= 0.0 && fx < (double)g.nx && fy < (double)g.ny)) return NONE;
    return (uint32_t)fy * g.nx + (uint32_t)fx;
}

// ---- Curve2D ----

// `Curve::interval`
__device__ inline uint32_t curve_interval(const World& w, float offset, float& t, uint32_t& err) {
    if (isnan(offset)) err |= ERR_PATH_NAN;
    offset = rclamp(offset, 0.0f, w.curve_length);
    uint32_t start = 0, end = w.curve_points, index = (start + end) / 2;
    while (start < index) {
        if (offset <= w.curve_offsets[index]) end = index; else start = index;
        index = (start + end) / 2;
    }
    index = min(index, w.curve_points - 2);  // only a NaN offset (flagged) gets here
    float span = w.curve_offsets[index + 1] - w.curve_offsets[index];
    t = span < FLT_EPSILON ? 0.5f : (offset - w.curve_offsets[index]) / span;
    return index;
}

// `Curve::direction`
__device__ inline Vec2 curve_direction(const World& w, float offset, uint32_t& err) {
    float t;
    uint32_t i = curve_interval(w, offset, t, err);
    Vec2 a = w.curve_forwards[i], b = w.curve_forwards[i + 1];
    float a2 = dot(a, a), b2 = dot(b, b);
    if (a2 == 0.0f || b2 == 0.0f) return normalized(a + (b - a) * t);
    float len = sqrtf(a2);
    float result_length = len + (sqrtf(b2) - len) * t;
    float angle = native_atan2(cross(a, b), dot(a, b)) * t;
    float s, c;
    engine_sin_cos(angle, s, c, err);
    return normalized(Vec2{a.x * c - a.y * s, a.x * s + a.y * c} * (result_length / len));
}

// `Curve::position`
__device__ inline Vec2 curve_position(const World& w, float offset, uint32_t& err) {
    float t;
    uint32_t i = curve_interval(w, offset, t, err);
    Vec2 a = w.curve_position[i];
    return a + (w.curve_position[i + 1] - a) * t;
}

// `Curve::project`: (offset, squared distance) of q on segment i.
__device__ inline float curve_project(const World& w, Vec2 q, uint32_t i, float& d2) {
    float span = w.curve_offsets[i + 1] - w.curve_offsets[i];
    Vec2 origin = w.curve_position[i];
    Vec2 e = w.curve_position[i + 1] - origin;
    Vec2 direction = Vec2{e.x / span, e.y / span};
    float d = rclamp(dot(q - origin, direction), 0.0f, span);
    Vec2 delta = (origin + direction * d) - q;
    d2 = dot(delta, delta);
    return w.curve_offsets[i] + d;
}

// `Curve::closest_offset`: segment 0 first (a NaN there sticks), then strict
// improvements in ascending order, over a candidate superset of the nearest.
__device__ inline float curve_closest_offset(const World& w, Vec2 q) {
    float nearest = 0.0f, distance = -1.0f;
    uint32_t count = w.curve_points - 1;
    if (w.curve_points < 2) return nearest;
    auto visit = [&](uint32_t i) {
        float d2;
        float offset = curve_project(w, q, i, d2);
        if (distance < 0.0f || d2 < distance) { nearest = offset; distance = d2; }
    };
    visit(0);
    uint32_t cell = grid_cell(w.curve_grid, q);
    if (cell != NONE) {
        for (uint32_t k = w.curve_grid.start[cell], e = w.curve_grid.start[cell + 1]; k < e; k++) {
            uint32_t i = w.curve_grid.items[k];
            if (i > 0) visit(i);
        }
    } else {
        for (uint32_t i = 1; i < count; i++) visit(i);
    }
    return nearest;
}

// ---- baked path ----

struct PathHit { double offset; Vec2 near, tangent; };

// `PathSegments::project`
__device__ inline float path_project(const PathSeg& s, Vec2 p, float& fraction, Vec2& near) {
    Vec2 start = Vec2{s.x, s.y}, delta = Vec2{s.dx, s.dy};
    fraction = s.len2 > 0.0f ? rclamp(dot(p - start, delta) / s.len2, 0.0f, 1.0f) : 0.0f;
    near = start + delta * fraction;
    Vec2 diff = p - near;
    return dot(diff, diff);
}

__device__ inline float path_posmod(float x, float length) {
    float r = fmodf(x, length);
    return r < 0.0f ? r + length : r;
}

// `binary_search_by(partial_cmp).unwrap_or_else(|i| i.saturating_sub(1))`
// over strictly increasing path offsets.
__device__ inline int64_t path_index(const World& w, float offset, uint32_t& err) {
    double v = (double)offset;
    if (isnan(v)) { err |= ERR_PATH_NAN; return 0; }
    uint32_t lo = 0, hi = w.path_points;  // first index with offsets[i] >= v
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (w.path_offsets[mid] < v) lo = mid + 1; else hi = mid;
    }
    if (lo < w.path_points && w.path_offsets[lo] == v) return lo;
    return lo > 0 ? lo - 1 : 0;
}

// `Track::closest_path(point, previous)`; `has_previous` selects the window.
__device__ inline PathHit closest_path(const World& w, Vec2 point, bool has_previous, double previous, uint32_t& err) {
    int64_t count = w.path_points - 1;
    float length = (float)w.path_offsets[count];
    int64_t from = 0, to = count;
    if (has_previous && length > 900.0f) {
        float prev = (float)previous;
        int64_t center = path_index(w, path_posmod(prev, length), err);
        from = path_index(w, path_posmod(prev - 450.0f, length), err);
        to = path_index(w, path_posmod(prev + 450.0f, length), err) + 1;
        if (from > center) from -= count;
        if (to < center) to += count;
    }
    float distance = FLT_MAX;
    int64_t nearest = -1;
    auto visit = [&](int64_t i) {
        float fraction;
        Vec2 near;
        float d2 = path_project(w.path_segments[i], point, fraction, near);
        if (d2 < distance) { distance = d2; nearest = i; }
    };
    uint32_t cell = (from == 0 && to == count) ? grid_cell(w.path_grid, point) : NONE;
    if (cell != NONE) {
        for (uint32_t k = w.path_grid.start[cell], e = w.path_grid.start[cell + 1]; k < e; k++) visit(w.path_grid.items[k]);
    } else {
        for (int64_t i = from; i < to; i++) visit(((i % count) + count) % count);
    }
    if (nearest < 0) return PathHit{0.0, Vec2{0.0f, 0.0f}, Vec2{0.0f, 0.0f}};
    const PathSeg& s = w.path_segments[nearest];
    float fraction;
    Vec2 near;
    path_project(s, point, fraction, near);
    float begin = (float)w.path_offsets[nearest], end = (float)w.path_offsets[nearest + 1];
    double offset = length > 0.0f ? (double)path_posmod(begin + (end - begin) * fraction, length) : 0.0;
    return PathHit{offset, near, normalized(Vec2{s.dx, s.dy})};
}

}  // namespace altd
