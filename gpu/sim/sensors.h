// `Car::sensor` (src/physics/car.rs) for every sensor kind, reading the GPU car state.
#pragma once
#include "rays.h"
#include "track.h"

namespace altd {

constexpr float PI_F = 3.14159265358979323846f;

struct Sensors {
    uint32_t count, needs_path;  // needs_path: some sensor reads the path offset
    SensorDesc s[MAX_SENSORS];
};

// Cars whose controls update this tick (Schedule::update_controls).
struct ReadMask {
    uint32_t batch_size, batch_index, batches_per_tick, require_active;
};

__device__ inline bool reads(const ReadMask& m, const Car& car, uint32_t index) {
    uint32_t batch = index / m.batch_size;
    return (batch + 8 - m.batch_index) % 8 < m.batches_per_tick && (!m.require_active || (car.flags & CAR_ACTIVE));
}

__device__ inline Vec2 actual_velocity(const Car& c) { return (c.flags & CAR_PENDING_VELOCITY) ? Vec2{0.0f, 0.0f} : c.velocity; }
__device__ inline float actual_angular(const Car& c) { return (c.flags & CAR_PENDING_VELOCITY) ? 0.0f : c.angular_velocity; }
__device__ inline Vec2 actual_acceleration(const Car& c) { return (c.flags & CAR_PENDING_VELOCITY) ? c.reset_acceleration : c.acceleration; }
__device__ inline Vec2 cached_right(const Car& c) { return (c.flags & CAR_RESET_RIGHT) ? Vec2{0.0f, 0.0f} : normalized(c.basis_x); }
// `Car::cached_front`: cached_right rotated by -PI/2, whose managed sine and
// cosine are constant (godot_math::QUARTER_TURN_BACK).
__device__ inline Vec2 cached_front(const Car& c) {
    Vec2 r = cached_right(c);
    float s = ffrom(0xbf7fffffu), co = ffrom(0xb3bbbd2eu);
    return Vec2{r.x * co - r.y * s, r.x * s + r.y * co};
}

// `Car::sensor_path_offset` (actual position equals position: no transform reset is pending).
__device__ inline float sensor_path_offset(const World& w, const Car& car) {
    return curve_closest_offset(w, path_sensor_point(w, car.position));
}

// `Car::sensor`. `here` is sensor_path_offset for the path sensors.
__device__ inline double sensor_value(const World& w, const VehicleDesc& v, const Car& car, const SensorDesc& s,
                                      float here, uint32_t& err) {
    switch (s.kind) {
        case S_RAYCAST: {
            Vec2 local = s.offset;  // `Car::ray_local`, precomputed on the host (with_ray_offset)
            Vec2 end = transform_point(car.position, car.basis_x, car.basis_y, local);
            float2 hit;
            if (!raycast(w, make_float2(car.position.x, car.position.y), make_float2(end.x, end.y), hit)) return 0.0;
            if (hit.x == end.x && hit.y == end.y) return 0.0;
            return (double)(1.0f - length(Vec2{hit.x, hit.y} - car.position) / s.b);
        }
        case S_DISTANCE_FROM_WALL: {
            float2 wall = closest_wall_point(w, make_float2(car.position.x, car.position.y));
            return (double)rclamp(length(car.position - Vec2{wall.x, wall.y}) / s.a, 0.0f, 1.0f);
        }
        case S_SPEED: return (double)rclamp(length(car.velocity) / v.max_velocity_f, 0.0f, 1.0f);
        case S_VELOCITY_FRONT:
            return signed_sensor(dot(actual_velocity(car), cached_front(car)), v.max_velocity_f);
        case S_VELOCITY_SIDE:
            return signed_sensor(dot(actual_velocity(car), cached_right(car)), v.max_velocity_f / 2.0f);
        case S_ACCELERATION_FRONT:
            return signed_sensor(dot(actual_acceleration(car), cached_front(car)), s.a);
        case S_ACCELERATION_SIDE:
            return signed_sensor(dot(actual_acceleration(car), cached_right(car)), s.a);
        case S_ANGULAR_VELOCITY: return signed_sensor(actual_angular(car), 2.0f);
        case S_WHEEL_ANGLE: {
            uint32_t i = 0;
            while (i + 1 < v.wheel_count && !v.wheels[i].steering) i++;
            return signed_sensor(car.wheel_angle[i], 45.0f);
        }
        case S_BOOST_CAPACITY: return py_clamp((double)car.boost_energy, 0.0, 1.0);
        case S_GRIP: {
            Vec2 p = car.position + rotated(s.offset, car.rotation, err);
            return (double)w.surfaces[vehicle_surface(w, p)].grip;
        }
        case S_CORRECT_DIRECTION: {
            if (w.path_points < 2) return 0.0;
            Vec2 tangent = normalized(curve_direction(w, here, err));
            Vec2 forward = normalized(cached_front(car));
            return (double)rclamp(dot(forward, tangent), -1.0f, 1.0f);
        }
        case S_TRACK_CURVATURE: {
            if (w.path_points < 2) return 0.0;
            Vec2 tangent = normalized(curve_direction(w, here, err));
            if (w.curve_length <= 0.0f) return 0.0;
            float speed = rclamp(length(actual_velocity(car)) / v.max_velocity_f, 0.0f, 1.0f);
            float lo = rmax(s.a, 0.0f), hi = rmax(s.b, lo);
            float offset = fmodf(here + (lo + (hi - lo) * speed), w.curve_length);
            Vec2 ahead = normalized(curve_direction(w, offset, err));
            return (double)rclamp(fabsf(profile_atan2(w.math_profile, cross(tangent, ahead), dot(tangent, ahead))) / PI_F, 0.0f, 1.0f);
        }
    }
    return 0.0;
}

}  // namespace altd
