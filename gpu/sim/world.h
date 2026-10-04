// Static track data on the device. `World` is filled by Rust (src/gpu/simulation.rs
// mirrors it) with host pointers and copied field by field to the device by
// altd_gpu_world_create.
#pragma once
#include <cstdint>
#include "state.h"

namespace altd {

constexpr uint32_t NONE = 0xffffffffu;
constexpr int MAX_RAY_DEPTH = 48;

struct RayNode { uint32_t first, last, front, back; float4 own, subtree; };

// world::Surface, with the `as f32` every user applies.
struct SurfaceDev { float grip, power, steering, pad; };

// One 768-pixel tile: surface index (vehicle_surface) and the path-sensor
// connection line (first, last) if `connected`.
struct TileCell { uint32_t surface, connected; Vec2 first, last; };

// path_segments::PathSegments, one row per segment.
struct PathSeg { float x, y, dx, dy, len2, pad0, pad1, pad2; };

// world::PhysicsShape.
struct ShapeDev {
    Vec2 origin;          // origin_f32
    float friction, bounce;
    double min_x, min_y, max_x, max_y;
    uint32_t first, count, wall_first, pad;  // points first .. first + count of shape_local/shape_normals
};

// Cell lists of the segments that can be nearest to any point of the cell
// (gpu::simulation::near_grid). Points outside the grid use the full scan.
struct NearGrid {
    double x0, y0, cell;
    uint32_t nx, ny;
    const uint32_t* start;  // nx * ny + 1 offsets into items
    const uint32_t* items;  // ascending segment indices per cell
};

struct World {
    // BSP raycaster (bsp::RayTree)
    uint32_t ray_node_count, ray_wall_count, ray_depth;
    float ray_magnitude;
    const RayNode* ray_nodes;
    const float4* ray_walls;
    // vehicle surfaces by tile
    uint32_t surface_count, default_surface;  // default: vehicle.surface("asphalt")
    const SurfaceDev* surfaces;
    int64_t tile_x0, tile_y0, tile_nx, tile_ny;
    const TileCell* tiles;
    // baked path (Track::closest_path)
    uint32_t path_points, pad0;
    const PathSeg* path_segments;
    const double* path_offsets;
    NearGrid path_grid;
    // Curve2D (curve::Curve)
    uint32_t curve_points;
    float curve_length;
    const Vec2* curve_position;
    const float* curve_offsets;
    const Vec2* curve_forwards;
    NearGrid curve_grid;
    // physics shapes
    uint32_t shape_count, shape_point_count;
    const ShapeDev* shapes;
    const Vec2* shape_local;
    const Vec2* shape_normals;
    // Shapes per cell whose box (grown by 1) meets the cell grown by
    // shape_margin, ascending (gpu::simulation::shape_grid).
    NearGrid shape_grid;
    double shape_margin;
    // Track::path as (x, y) pairs, path_points of them (Track::path_position).
    const double* path_xy;
};

}  // namespace altd
