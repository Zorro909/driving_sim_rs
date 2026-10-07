//! Signed path scores, lap timing, and accumulated driving metrics.

use crate::math::pymath::{clamp, py_min};
use crate::math::vec2::V2;
use crate::physics::car::{Car, Controls};
use crate::track::world::{Track, World};
use crate::training::evolution::Metrics;
use std::collections::VecDeque;

/// `ScoreTracker`: signed path score and forward-lap crossing on baked points.
#[derive(Clone, Debug, Default)]
pub struct ScoreTracker {
    pub previous_offset: Option<f64>,
    pub total_score: f64,
    pub lap_count: u64,
    pub went_backwards: bool,
    /// Fraction of the last stats interval at which a forward lap completed.
    pub lap_crossing_fraction: Option<f64>,
}

impl ScoreTracker {
    pub fn update(&mut self, track: &Track, position: V2) {
        self.lap_crossing_fraction = None;
        let length = track.path_length();
        if length <= 0.0 {
            return;
        }
        let Some(previous) = self.previous_offset else {
            let offset = track.closest_path(position, None).offset;
            self.total_score += (if offset > length / 2.0 { offset - length } else { offset }) * (5.0 / 384.0);
            self.went_backwards = offset > length / 2.0;
            self.previous_offset = Some(offset);
            return;
        };
        let offset = track.closest_path(position, Some(previous)).offset;
        let mut change = offset - previous;
        if change > length / 2.0 {
            change -= length;
        } else if change < -length / 2.0 {
            change += length;
        }
        self.total_score += change * (5.0 / 384.0);
        let crossing = previous + change;
        if crossing < 0.0 {
            self.went_backwards = true;
        } else if crossing >= length {
            if !self.went_backwards {
                self.lap_count += 1;
                self.lap_crossing_fraction = Some((length - previous) / (length - previous + offset));
            }
            self.went_backwards = false;
        }
        self.previous_offset = Some(offset);
    }
}

#[derive(Clone, Debug)]
pub struct TrainingStats {
    pub score: ScoreTracker,
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
    pub best_lap_time: Option<f64>,
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
    pub all_laps_time_sum: Option<f64>,
    pub recent_score_diffs: VecDeque<f64>,
}

impl TrainingStats {
    pub fn new(network_novelty: f64) -> TrainingStats {
        TrainingStats {
            score: ScoreTracker::default(),
            network_novelty,
            update_count: 0,
            collision_count: 0,
            total_score: 0.0,
            total_speed: 0.0,
            total_abs_steering: 0.0,
            total_actual_distance: 0.0,
            total_drifting_amount: 0.0,
            total_momentum_change_frequency: 0.0,
            total_throttle_change_frequency: 0.0,
            total_braking_frequency: 0.0,
            total_distance_from_wall: 0.0,
            total_distance_from_center: 0.0,
            best_lap_time: None,
            idle_ticks: 0,
            previous_speed: 0.0,
            previous_throttle: 0.0,
            previous_score: 0.0,
            previous_lap_count: 0,
            lap_start_tick: 0,
            drift_ticks: 0,
            current_slip_angle_degrees: 0.0,
            continuous_drift_time: 0.0,
            max_continuous_drift_time: 0.0,
            all_laps_time_sum: None,
            recent_score_diffs: VecDeque::with_capacity(10),
        }
    }

    pub fn update(&mut self, world: &World, car: &Car, controls: &Controls, tick: u64, pending_contact: bool) {
        let track = &world.track;
        if pending_contact {
            self.collision_count += 1;
        }
        if !car.active {
            return;
        }
        self.update_count += 1;
        let speed = crate::math::godot_math::F2::from(car.velocity).length() as f64;
        let initializing_score = self.score.previous_offset.is_none();
        self.score.update(track, car.position);
        let score = self.score.total_score;
        // AgentStats.GetClosestOffset seeds TotalScore before UpdateScore
        // takes its movement-difference baseline.
        let diff = if initializing_score {
            0.0
        } else {
            score - self.previous_score
        };
        if self.recent_score_diffs.len() == 10 {
            self.recent_score_diffs.pop_front();
        }
        self.recent_score_diffs.push_back(diff);
        let mean =
            self.recent_score_diffs.iter().copied().fold(0.0, |a, b| a + b) / self.recent_score_diffs.len() as f64;
        self.idle_ticks = if mean < 0.1 { self.idle_ticks + 1 } else { 0 };
        self.total_score = score;
        self.previous_score = score;
        self.total_speed += speed;
        self.total_abs_steering += clamp(controls.steering, -1.0, 1.0).abs();
        self.total_actual_distance += speed * 0.1;
        self.total_momentum_change_frequency += (speed - self.previous_speed).abs();
        self.total_throttle_change_frequency +=
            (clamp(controls.acceleration, -1.0, 1.0) - self.previous_throttle).abs();
        if controls.brake > 0.0001 || controls.handbrake > 0.0001 {
            self.total_braking_frequency += 0.1;
        }
        self.previous_speed = speed;
        self.previous_throttle = clamp(controls.acceleration, -1.0, 1.0);
        if let Some(wall) = track.closest_wall(car.position) {
            self.total_distance_from_wall += (crate::math::godot_math::F2::from(wall)
                - crate::math::godot_math::F2::from(car.position))
            .length() as f64;
        }
        if track.path.len() >= 2 {
            let center = track.path_position(self.score.previous_offset.unwrap_or(0.0));
            self.total_distance_from_center += (crate::math::godot_math::F2::from(center)
                - crate::math::godot_math::F2::from(car.position))
            .length() as f64;
        }
        let right = car.body_basis.0.normalized();
        let velocity = crate::math::godot_math::F2::from(car.velocity);
        let front = velocity.dot(right.rotated(-std::f32::consts::FRAC_PI_2));
        let side = velocity.dot(right);
        self.current_slip_angle_degrees = (car.math.atan2(side, front) * 57.29578f32) as f64;
        let slip_degrees = self.current_slip_angle_degrees.abs();
        self.drift_ticks = if speed > 50.0 && (15.0..=60.0).contains(&slip_degrees) {
            self.drift_ticks + 1
        } else {
            0
        };
        if self.drift_ticks >= 3 {
            self.total_drifting_amount += 0.1;
            self.continuous_drift_time += 0.1;
            self.max_continuous_drift_time = self.max_continuous_drift_time.max(self.continuous_drift_time);
        }
        if self.drift_ticks == 0 {
            self.continuous_drift_time = 0.0;
        }
        if self.score.lap_count > self.previous_lap_count {
            // EvolutionStatsLapsObserver rounds to milliseconds, with .NET's
            // default midpoint-to-even rule. It stores the sample time as the
            // next lap start, not the interpolated crossing time.
            let fraction = self.score.lap_crossing_fraction.unwrap_or(1.0).clamp(0.0, 1.0);
            let elapsed = (tick / 6) as f64 * 0.1 - (self.lap_start_tick / 6) as f64 * 0.1;
            let lap_time = ((elapsed - 0.1 + fraction * 0.1) * 1000.0).round_ties_even() / 1000.0;
            let current = match self.best_lap_time {
                Some(best) => best,
                _ => lap_time,
            };
            self.best_lap_time = Some(py_min(current, lap_time));
            self.all_laps_time_sum = Some(self.all_laps_time_sum.unwrap_or(0.0) + lap_time);
            self.lap_start_tick = tick;
            self.previous_lap_count = self.score.lap_count;
        }
    }

    /// `TrainingStats.metrics()` in `METRIC_NAMES` order.
    pub fn metrics(&self) -> Metrics {
        let best_lap_performance = match self.best_lap_time {
            Some(t) if t != 0.0 => Some(1.0 / t),
            _ => None,
        };
        let efficiency = if self.total_actual_distance != 0.0 {
            self.total_score / (self.total_actual_distance * (5.0 / 384.0))
        } else {
            0.0
        };
        [
            Some(self.total_score),
            Some(self.total_speed),
            Some(self.total_abs_steering),
            Some(self.total_actual_distance),
            Some(self.total_drifting_amount),
            Some(self.total_momentum_change_frequency),
            Some(self.total_throttle_change_frequency),
            Some(self.total_braking_frequency),
            Some(self.total_distance_from_wall),
            Some(self.total_distance_from_center),
            Some(self.network_novelty),
            Some(self.collision_count as f64),
            best_lap_performance,
            Some(efficiency),
        ]
    }
}
