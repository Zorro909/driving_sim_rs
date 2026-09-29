// godot_math::F2 and the scalar helpers with the CPU's exact semantics.
// Build with -ffp-contract=off: every operation rounds on its own.
#pragma once
#include <hip/hip_runtime.h>
#include "math.h"
#include "state.h"

namespace altd {

__device__ inline Vec2 v2(float x, float y) { return Vec2{x, y}; }
__device__ inline Vec2 operator+(Vec2 a, Vec2 b) { return Vec2{a.x + b.x, a.y + b.y}; }
__device__ inline Vec2 operator-(Vec2 a, Vec2 b) { return Vec2{a.x - b.x, a.y - b.y}; }
__device__ inline Vec2 operator*(Vec2 a, float s) { return Vec2{a.x * s, a.y * s}; }
__device__ inline Vec2 operator-(Vec2 a) { return Vec2{-a.x, -a.y}; }
__device__ inline float dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
__device__ inline float cross(Vec2 a, Vec2 b) { return a.x * b.y - a.y * b.x; }
// sqrtf is correctly rounded only with -fhip-fp32-correctly-rounded-divide-sqrt
// (gpu/build.sh); __fsqrt_rn is NOT (ROCm lowers it to the approximate
// instruction: ~15% of results differ by an ulp).
__device__ inline float length(Vec2 a) { return sqrtf(dot(a, a)); }
__device__ inline bool same_bits(Vec2 a, Vec2 b) { return fbits(a.x) == fbits(b.x) && fbits(a.y) == fbits(b.y); }

__device__ inline Vec2 normalized(Vec2 a) {
    float n = length(a);
    return n == 0.0f ? Vec2{0.0f, 0.0f} : Vec2{a.x / n, a.y / n};
}

// Godot C# Vector2.LimitLength divides before multiplying.
__device__ inline Vec2 limit(Vec2 a, float max) {
    float n = length(a);
    return (n > max && n > 0.0f) ? Vec2{a.x / n, a.y / n} * max : a;
}

__device__ inline Vec2 rotated(Vec2 a, float angle, uint32_t& err) {
    float s, c;
    managed_sin_cos(angle, s, c, err);
    return Vec2{a.x * c - a.y * s, a.x * s + a.y * c};
}

// Rust f32::max / f32::min as lowered on x86 (a NaN operand yields the other).
__device__ inline float rmax(float a, float b) { return isnan(a) ? b : (b > a ? b : a); }
__device__ inline float rmin(float a, float b) { return isnan(a) ? b : (b < a ? b : a); }
// Rust f32::clamp: NaN passes through.
__device__ inline float rclamp(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
// pymath::py_max / py_min / clamp (f64).
__device__ inline double py_max(double a, double b) { return b > a ? b : a; }
__device__ inline double py_min(double a, double b) { return b < a ? b : a; }
__device__ inline double py_clamp(double v, double lo, double hi) { return py_min(hi, py_max(lo, v)); }

// Rust `as i64` from f64 saturates and maps NaN to 0.
__device__ inline int64_t sat_i64(double x) {
    if (isnan(x)) return 0;
    if (x >= 9223372036854775807.0) return INT64_MAX;
    if (x <= -9223372036854775808.0) return INT64_MIN;
    return (int64_t)x;
}

// `godot_math::transform_point`: products before the origin.
__device__ inline Vec2 transform_point(Vec2 position, Vec2 x, Vec2 y, Vec2 p) {
    return (x * p.x + y * p.y) + position;
}

// `godot_math::signed_sensor`: Mathf.Remap(value, -max, max, -1, 1), then Clamp.
__device__ inline double signed_sensor(float value, float maximum) {
    float from = 0.0f - maximum;
    float weight = (value - from) / (maximum - from);
    return (double)rclamp(-1.0f + 2.0f * weight, -1.0f, 1.0f);
}

}  // namespace altd
