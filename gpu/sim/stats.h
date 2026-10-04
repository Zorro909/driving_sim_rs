// training::ScoreTracker and TrainingStats::update (src/training/stats.rs) on the
// GPU agent state, and Schedule::update_stats.
#pragma once
#include "rays.h"
#include "track.h"

namespace altd {

__device__ inline double path_length(const World& w) { return w.path_points ? w.path_offsets[w.path_points - 1] : 0.0; }

// `Track::path_position` (path_points >= 2).
__device__ inline Vec2 path_position(const World& w, double offset, uint32_t& err) {
    if (w.curve_points) return curve_position(w, (float)offset, err);
    offset = py_clamp(offset, 0.0, path_length(w));
    uint32_t lo = 0, hi = w.path_points;  // partition_point(x <= offset)
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (w.path_offsets[mid] <= offset) lo = mid + 1; else hi = mid;
    }
    uint32_t index = min(w.path_points - 2, lo ? lo - 1 : 0);
    double start = w.path_offsets[index], end = w.path_offsets[index + 1];
    double fraction = end > start ? (offset - start) / (end - start) : 0.0;
    const double* a = w.path_xy + 2 * index;
    return Vec2{(float)(a[0] + (a[2] - a[0]) * fraction), (float)(a[1] + (a[3] - a[1]) * fraction)};
}

// `ScoreTracker::update`
__device__ inline void score_update(const World& w, Agent& a, Vec2 position, uint32_t& err) {
    a.flags &= ~AGENT_HAS_LAP_CROSSING;
    a.lap_crossing_fraction = 0.0;  // None exports as 0.0
    double length = path_length(w);
    if (length <= 0.0) return;
    if (!(a.flags & AGENT_HAS_PREVIOUS_OFFSET)) {
        double offset = closest_path(w, position, false, 0.0, err).offset;
        a.tracker_score += (offset > length / 2.0 ? offset - length : offset) * (5.0 / 384.0);
        a.flags = offset > length / 2.0 ? a.flags | AGENT_WENT_BACKWARDS : a.flags & ~AGENT_WENT_BACKWARDS;
        a.previous_offset = offset;
        a.flags |= AGENT_HAS_PREVIOUS_OFFSET;
        return;
    }
    double previous = a.previous_offset;
    double offset = closest_path(w, position, true, previous, err).offset;
    double change = offset - previous;
    if (change > length / 2.0) change -= length;
    else if (change < -length / 2.0) change += length;
    a.tracker_score += change * (5.0 / 384.0);
    double crossing = previous + change;
    if (crossing < 0.0) {
        a.flags |= AGENT_WENT_BACKWARDS;
    } else if (crossing >= length) {
        if (!(a.flags & AGENT_WENT_BACKWARDS)) {
            a.lap_count++;
            a.lap_crossing_fraction = (length - previous) / (length - previous + offset);
            a.flags |= AGENT_HAS_LAP_CROSSING;
        }
        a.flags &= ~AGENT_WENT_BACKWARDS;
    }
    a.previous_offset = offset;
}

// `TrainingStats::update`
__device__ inline void stats_update(const World& w, const Car& car, Agent& a, uint64_t tick, uint32_t& err) {
    if (a.flags & AGENT_PENDING_CONTACT) a.collision_count++;
    if (!(car.flags & CAR_ACTIVE)) return;
    a.update_count++;
    double speed = (double)length(car.velocity);
    bool initializing = !(a.flags & AGENT_HAS_PREVIOUS_OFFSET);
    score_update(w, a, car.position, err);
    double score = a.tracker_score;
    double diff = initializing ? 0.0 : score - a.previous_score;
    if (a.recent_len == RECENT) {
        a.recent[a.recent_start] = diff;
        a.recent_start = (a.recent_start + 1) % RECENT;
    } else {
        a.recent[(a.recent_start + a.recent_len) % RECENT] = diff;
        a.recent_len++;
    }
    double sum = 0.0;
    for (uint32_t k = 0; k < a.recent_len; k++) sum += a.recent[(a.recent_start + k) % RECENT];
    a.idle_ticks = sum / (double)a.recent_len < 0.1 ? a.idle_ticks + 1 : 0;
    a.total_score = score;
    a.previous_score = score;
    a.total_speed += speed;
    a.total_abs_steering += fabs(py_clamp(a.controls[1], -1.0, 1.0));
    a.total_actual_distance += speed * 0.1;
    a.total_momentum_change_frequency += fabs(speed - a.previous_speed);
    double throttle = py_clamp(a.controls[0], -1.0, 1.0);
    a.total_throttle_change_frequency += fabs(throttle - a.previous_throttle);
    if (a.controls[2] > 0.0001 || a.controls[3] > 0.0001) a.total_braking_frequency += 0.1;
    a.previous_speed = speed;
    a.previous_throttle = throttle;
    float2 wall = closest_wall_point(w, make_float2(car.position.x, car.position.y));
    a.total_distance_from_wall += (double)length(Vec2{wall.x, wall.y} - car.position);
    if (w.path_points >= 2) {
        Vec2 center = path_position(w, (a.flags & AGENT_HAS_PREVIOUS_OFFSET) ? a.previous_offset : 0.0, err);
        a.total_distance_from_center += (double)length(center - car.position);
    }
    Vec2 right = normalized(car.basis_x);
    float front = dot(car.velocity, rotated(right, -1.57079632679489661923f, err));
    float side = dot(car.velocity, right);
    a.current_slip_angle_degrees = (double)(native_atan2(side, front) * 57.29578f);
    double slip = fabs(a.current_slip_angle_degrees);
    a.drift_ticks = (speed > 50.0 && slip >= 15.0 && slip <= 60.0) ? a.drift_ticks + 1 : 0;
    if (a.drift_ticks >= 3) {
        a.total_drifting_amount += 0.1;
        a.continuous_drift_time += 0.1;
        a.max_continuous_drift_time = fmax(a.max_continuous_drift_time, a.continuous_drift_time);
    }
    if (a.drift_ticks == 0) a.continuous_drift_time = 0.0;
    if (a.lap_count > a.previous_lap_count) {
        double fraction = (a.flags & AGENT_HAS_LAP_CROSSING) ? a.lap_crossing_fraction : 1.0;
        fraction = fraction < 0.0 ? 0.0 : (fraction > 1.0 ? 1.0 : fraction);  // f64::clamp
        double elapsed = (double)(tick / 6) * 0.1 - (double)(a.lap_start_tick / 6) * 0.1;
        double lap_time = rint((elapsed - 0.1 + fraction * 0.1) * 1000.0) / 1000.0;
        double current = (a.flags & AGENT_HAS_BEST_LAP) ? a.best_lap_time : lap_time;
        a.best_lap_time = py_min(current, lap_time);
        a.all_laps_time_sum = ((a.flags & AGENT_HAS_ALL_LAPS) ? a.all_laps_time_sum : 0.0) + lap_time;
        a.flags |= AGENT_HAS_BEST_LAP | AGENT_HAS_ALL_LAPS;
        a.lap_start_tick = tick;
        a.previous_lap_count = a.lap_count;
    }
}

// `Car::deactivate`: velocity reset to zero at the next callback.
__device__ inline void car_deactivate(Car& car) {
    car.flags = (car.flags & ~CAR_ACTIVE) | CAR_PENDING_VELOCITY;
    car.reset_acceleration = car.acceleration;
}

// `Schedule::update_stats` on a statistics tick (tick % 6 == stats_phase);
// `stats_tick` is tick - first_stats_tick.
__device__ inline void update_stats(const World& w, Car& car, Agent& a, uint64_t tick, uint64_t stats_tick,
                                    bool eliminate_when_idle, uint32_t& err) {
    bool was_active = car.flags & CAR_ACTIVE;
    stats_update(w, car, a, stats_tick, err);
    a.flags &= ~AGENT_PENDING_CONTACT;
    if ((car.flags & CAR_ACTIVE) && eliminate_when_idle && a.idle_ticks > 80) car_deactivate(car);
    if (was_active && !(car.flags & CAR_ACTIVE)) {
        a.deactivated_at = tick;
        a.flags |= AGENT_HAS_DEACTIVATED;
    }
}

}  // namespace altd
