//! Game training scheduler, statistics, and population lifecycle.
//!
//! Independent mode runs each car through a window, then catches inactive cars
//! up to the final tick. Native shared broadphase scenes use a tick-major loop
//! to preserve contact ordering. Deactivated cars retain passive physics, while
//! their driving metrics stop updating.

use crate::car::{Car, Controls, Sensor, SensorScratch, DT};
use crate::evolution::{reproduce, AgentResult, EvolutionSettings, Generation, Metrics};
use crate::network::{ForwardScratch, Network};
use crate::pymath::{clamp, py_min};
use crate::pyrandom::PyRandom;
use crate::vec2::V2;
use crate::world::{Track, World};
use rayon::prelude::*;
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::Arc;

/// Untagged checkpoints retain the historical Python backend. Game replay
/// checkpoints carry both independent .NET streams under an explicit tag.
#[derive(Clone)]
pub enum TrainingRandom { Python(PyRandom), Game(crate::game_random::GameRandom) }
impl From<PyRandom> for TrainingRandom {fn from(r:PyRandom)->Self{Self::Python(r)}}
impl From<crate::game_random::GameRandom> for TrainingRandom {fn from(r:crate::game_random::GameRandom)->Self{Self::Game(r)}}
impl TrainingRandom {
    pub fn to_json(&self)->Value {match self{Self::Python(r)=>r.to_json(),Self::Game(r)=>r.to_json()}}
    pub fn from_json(v:&Value)->Self {
        if v["backend"].as_str()==Some("game") {Self::Game(crate::game_random::GameRandom::from_json(v))}else{Self::Python(PyRandom::from_json(v))}
    }
    pub fn reproduce(&mut self,agents:&[AgentResult],settings:&EvolutionSettings)->Generation {
        match self{Self::Python(r)=>reproduce(agents,settings,r),Self::Game(r)=>crate::evolution::reproduce_game(agents,settings,r)}
    }
    fn reproduce_with_scratch(&mut self, agents: &[AgentResult], settings: &EvolutionSettings, scratch: &mut Vec<f64>) -> Generation {
        match self {
            Self::Python(r) => crate::evolution::reproduce_with_scratch(agents, settings, r, scratch),
            Self::Game(r) => crate::evolution::reproduce_game(agents, settings, r),
        }
    }
    pub fn xavier(&mut self,shape:&[usize])->Network {
        match self{Self::Python(r)=>Network::xavier(shape,r),Self::Game(r)=>Network::xavier_game(shape,r)}
    }
}

/// `SENSOR_TYPES` in training.py.
pub fn sensor_type(name: &str) -> Option<&'static str> {
    Some(match name {
        "AccF" => "accelerationFront",
        "AccS" => "accelerationSide",
        "Wall" => "distanceFromWall",
        "AngVel" => "angularVelocity",
        "Boost" => "boostCapacity",
        "Dir" => "correctDirection",
        "Grip" => "grip",
        "VelF" => "velocityFront",
        "VelS" => "velocitySide",
        "Curve" => "trackCurvature",
        "Speed" | "VelAbs" => "speed",
        "Wheel" => "wheelAngle",
        _ => return None,
    })
}

/// Parse `"↑ -35°"` into its angle.
pub fn vision_angle(name: &str) -> Option<f64> {
    name.strip_prefix("↑ ")?.strip_suffix('°')?.trim().parse().ok()
}

/// `SensorsEditor.CalculateSensorLength`: the editor gives every vision ray
/// `200 + 600 * |cos(degrees)|` pixels, in float32 with the game's UCRT cosf.
pub fn vision_length(degrees: f32) -> f32 {
    200.0 + 600.0 * crate::native_math::cos(degrees * (std::f32::consts::PI / 180.0)).abs()
}

#[derive(Clone, Debug)]
pub struct SensorLayout {
    pub names: Vec<String>,
    pub sensors: Vec<Sensor>,
}

impl SensorLayout {
    /// Exact ordered descriptors captured from instantiated game sensors.
    /// Names may repeat; each descriptor retains its own float parameters.
    pub fn from_ordered(value: &Value) -> Result<SensorLayout, String> {
        let rows = value["sensors"].as_array().ok_or("missing ordered sensors")?;
        let names: Vec<String> = value["names"].as_array().ok_or("missing ordered names")?
            .iter().map(|v| v.as_str().map(str::to_owned).ok_or_else(|| "invalid sensor name".to_owned()))
            .collect::<Result<_, _>>()?;
        if names.len() != rows.len() { return Err("sensor/name count differs".into()); }
        let mut sensors = Vec::with_capacity(rows.len());
        for row in rows {
            let number = |key: &str| row[key].as_f64().map(|x| x as f32 as f64)
                .ok_or_else(|| format!("missing sensor parameter {key}"));
            let kind = row["$type"].as_str().ok_or("missing sensor type")?;
            sensors.push(match kind {
                "raycast" => Sensor::Raycast { degrees: number("Degrees")?, length: number("Length")? },
                "accelerationFront" => Sensor::AccelerationFront { max_acceleration: number("MaxAcceleration")? },
                "accelerationSide" => Sensor::AccelerationSide { max_acceleration: number("MaxAcceleration")? },
                "distanceFromWall" => Sensor::DistanceFromWall { max_distance: number("MaxDistance")? },
                "trackCurvature" => {
                    let min = number("MinLookaheadDistance")?.max(0.0);
                    Sensor::TrackCurvature { min_lookahead: min, max_lookahead: number("MaxLookaheadDistance")?.max(min) }
                },
                "grip" => {
                    let offset = &row["PositionOffset"];
                    let (x,y) = if offset.is_array() { (&offset[0],&offset[1]) } else { (&offset["X"],&offset["Y"]) };
                    let component = |v: &Value| v.as_f64().map(|x|x as f32 as f64).ok_or("missing grip offset component");
                    Sensor::Grip { offset: V2::new(component(x)?,component(y)?) }
                },
                "speed" | "velocityFront" | "velocitySide" | "angularVelocity" | "wheelAngle" |
                "boostCapacity" | "correctDirection" => Sensor::by_name(kind),
                _ => return Err(format!("unknown sensor type {kind}")),
            });
        }
        Ok(SensorLayout { names, sensors })
    }

    /// `SensorLayout.from_exports(live_network, model)`.
    pub fn from_exports(live_network: &Value, model: &Value) -> SensorLayout {
        if let Some(exact) = model.get("sensor_layout") {
            let layout = Self::from_ordered(exact).expect("exact sensor descriptors");
            let inputs: Vec<_> = live_network["inputs"].as_array().expect("network inputs")
                .iter().map(|v|v.as_str().expect("input name")).collect();
            assert!(layout.names.iter().map(String::as_str).eq(inputs), "ordered sensor inputs differ from network");
            return layout;
        }
        let vision: Vec<(f64, f64)> = model["vision"]
            .as_array()
            .expect("model vision")
            .iter()
            .map(|item| (item["angle"].as_f64().unwrap(), item["length"].as_f64().unwrap()))
            .collect();
        let names: Vec<String> = live_network["inputs"]
            .as_array()
            .expect("network inputs")
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let sensors = names
            .iter()
            .map(|name| match vision_angle(name) {
                Some(angle) => {
                    // Python dict: the last vision entry with this angle wins.
                    // Angles the model does not list get the editor's length.
                    let length = vision
                        .iter()
                        .rev()
                        .find(|(a, _)| *a == angle)
                        .map_or_else(|| vision_length(angle as f32) as f64, |&(_, length)| length);
                    Sensor::Raycast { degrees: angle, length }
                }
                None => Sensor::by_name(sensor_type(name).unwrap_or_else(|| panic!("unknown sensor: {name}"))),
            })
            .collect();
        SensorLayout { names, sensors }
    }

    pub fn read_into(&self, world: &World, car: &Car, scratch: &mut SensorScratch, out: &mut Vec<f64>) {
        scratch.invalidate();
        out.clear();
        out.extend(self.sensors.iter().map(|&sensor| car.sensor(world, sensor, scratch)));
    }
}

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
        let speed = crate::godot_math::F2::from(car.velocity).length() as f64;
        let initializing_score = self.score.previous_offset.is_none();
        self.score.update(track, car.position);
        let score = self.score.total_score;
        // AgentStats.GetClosestOffset seeds TotalScore before UpdateScore
        // takes its movement-difference baseline.
        let diff = if initializing_score { 0.0 } else { score - self.previous_score };
        if self.recent_score_diffs.len() == 10 {
            self.recent_score_diffs.pop_front();
        }
        self.recent_score_diffs.push_back(diff);
        let mean = self.recent_score_diffs.iter().copied().fold(0.0, |a,b| a+b) / self.recent_score_diffs.len() as f64;
        self.idle_ticks = if mean < 0.1 { self.idle_ticks + 1 } else { 0 };
        self.total_score = score;
        self.previous_score = score;
        self.total_speed += speed;
        self.total_abs_steering += clamp(controls.steering, -1.0, 1.0).abs();
        self.total_actual_distance += speed * 0.1;
        self.total_momentum_change_frequency += (speed - self.previous_speed).abs();
        self.total_throttle_change_frequency += (clamp(controls.acceleration, -1.0, 1.0) - self.previous_throttle).abs();
        if controls.brake > 0.0001 || controls.handbrake > 0.0001 {
            self.total_braking_frequency += 0.1;
        }
        self.previous_speed = speed;
        self.previous_throttle = clamp(controls.acceleration, -1.0, 1.0);
        if let Some(wall) = track.closest_wall(car.position) {
            self.total_distance_from_wall += (crate::godot_math::F2::from(wall) - crate::godot_math::F2::from(car.position)).length() as f64;
        }
        if track.path.len() >= 2 {
            let center = track.path_position(self.score.previous_offset.unwrap_or(0.0));
            self.total_distance_from_center += (crate::godot_math::F2::from(center) - crate::godot_math::F2::from(car.position)).length() as f64;
        }
        let right = car.body_basis.0.normalized();
        let velocity = crate::godot_math::F2::from(car.velocity);
        let front = velocity.dot(right.rotated(-std::f32::consts::FRAC_PI_2));
        let side = velocity.dot(right);
        self.current_slip_angle_degrees = (crate::native_math::atan2(side,front) * 57.29578f32) as f64;
        let slip_degrees = self.current_slip_angle_degrees.abs();
        self.drift_ticks = if speed > 50.0 && (15.0..=60.0).contains(&slip_degrees) { self.drift_ticks + 1 } else { 0 };
        if self.drift_ticks >= 3 {
            self.total_drifting_amount += 0.1;
            self.continuous_drift_time += 0.1;
            self.max_continuous_drift_time = self.max_continuous_drift_time.max(self.continuous_drift_time);
        }
        if self.drift_ticks == 0 { self.continuous_drift_time = 0.0; }
        if self.score.lap_count > self.previous_lap_count {
            // EvolutionStatsLapsObserver rounds to milliseconds, with .NET's
            // default midpoint-to-even rule. It stores the sample time as the
            // next lap start, not the interpolated crossing time.
            let fraction = self.score.lap_crossing_fraction.unwrap_or(1.0).clamp(0.0, 1.0);
            let elapsed = (tick/6) as f64 * 0.1 - (self.lap_start_tick/6) as f64 * 0.1;
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

/// Per-agent scratch buffers, kept with the agent so threads never share them.
#[derive(Default, Clone)]
struct AgentScratch {
    sensors: SensorScratch,
    inputs: Vec<f64>,
    forward: ForwardScratch,
}

#[derive(Clone)]
pub struct TrainingAgent {
    pub network: Network,
    pub car: Car,
    pub stats: TrainingStats,
    pub controls: Controls,
    pub pending_contact: bool,
    /// Tick at which the car became inactive in the current window.
    pub deactivated_at: Option<u64>,
    scratch: AgentScratch,
}

impl TrainingAgent {
    /// `Schedule::update_stats`: statistics on ticks where
    /// `tick % 6 == stats_phase`, then the idle elimination.
    pub fn update_stats(&mut self, world: &World, tick: u64, stats_phase: u64, eliminate_when_idle: bool) {
        let was_active = self.car.active;
        if tick % 6 == stats_phase {
            let first_stats_tick = if stats_phase == 0 { 6 } else { stats_phase };
            self.stats.update(world, &self.car, &self.controls, tick - first_stats_tick, self.pending_contact);
            self.pending_contact = false;
            if self.car.active && eliminate_when_idle && self.stats.idle_ticks > 80 {
                self.car.deactivate();
            }
        }
        if was_active && !self.car.active { self.deactivated_at = Some(tick); }
    }
    pub fn result(&self) -> AgentResult<'_> {
        AgentResult { network: &self.network, metrics: self.stats.metrics(), update_count: self.stats.update_count }
    }
}

/// Fills `cars` and `out` with the GPU state of `agents` (`Car::gpu_export`
/// and `agent_export`) in place, resizing them to the population. Retaining
/// the vectors between windows keeps their pages mapped, and the indexed
/// parallel fill avoids the linked-list concatenation of a `Result` collect.
pub fn export_state_into(agents: &[TrainingAgent], vehicle: &crate::world::VehicleConfig, surfaces: &crate::gpu_sim::SurfaceTable,
                         cars: &mut Vec<crate::gpu_sim::GpuCar>, out: &mut Vec<crate::gpu_sim::GpuAgent>) -> Result<(), String> {
    cars.resize(agents.len(), crate::gpu_sim::GpuCar::zeroed());
    out.resize(agents.len(), crate::gpu_sim::GpuAgent::default());
    agents.par_iter().zip(cars.par_iter_mut()).zip(out.par_iter_mut()).try_for_each(|((a, c), g)| {
        *c = a.car.gpu_export(vehicle, surfaces)?;
        *g = crate::gpu_sim::agent_export(a)?;
        Ok::<(), String>(())
    })
}

/// Which output slot of `Controls` each network output feeds.
#[derive(Clone, Copy, Debug)]
enum ControlSlot {
    Acceleration,
    Steering,
    Brake,
    Handbrake,
    Boost,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Each car runs the whole window on its own (fast path).
    Independent,
    /// All cars advance one tick at a time, as `TrainingRunner.step`.
    Lockstep,
}

/// Immutable per-window schedule shared by all agents.
struct Schedule<'a> {
    world: &'a World,
    layout: &'a SensorLayout,
    outputs: &'a [ControlSlot],
    stats_phase: u64,
    batch_size: usize,
    batches_per_tick: usize,
    eliminate_on_wall: bool,
    eliminate_when_idle: bool,
}

impl Schedule<'_> {
    fn update_stats(&self, agent: &mut TrainingAgent, tick: u64) {
        agent.update_stats(self.world, tick, self.stats_phase, self.eliminate_when_idle);
    }
    fn drive_agent(&self, index: usize, agent: &mut TrainingAgent, tick: u64, batch_index: usize) {
        let was_active = agent.car.active;
        self.update_controls(index, agent, batch_index);
        let contact = agent.car.step(self.world, &agent.controls, DT, self.eliminate_on_wall);
        agent.pending_contact |= contact;
        if was_active && !agent.car.active {
            agent.deactivated_at = Some(tick);
        }
    }
    fn update_controls(&self, index: usize, agent: &mut TrainingAgent, batch_index: usize) {
        let batch = index / self.batch_size;
        if (batch + 8 - batch_index) % 8 < self.batches_per_tick {
            if agent.car.active {
                let scratch = &mut agent.scratch;
                self.layout.read_into(self.world, &agent.car, &mut scratch.sensors, &mut scratch.inputs);
                let outputs = agent.network.forward_into(&scratch.inputs, &mut scratch.forward);
                let mut controls = Controls::default();
                for (slot, &value) in self.outputs.iter().zip(outputs) {
                    match slot {
                        ControlSlot::Acceleration => controls.acceleration = value,
                        ControlSlot::Steering => controls.steering = value,
                        ControlSlot::Brake => controls.brake = value,
                        ControlSlot::Handbrake => controls.handbrake = value,
                        ControlSlot::Boost => controls.boost = value,
                    }
                }
                agent.controls = controls;
            }
        }
    }
}

pub struct TrainingRunner {
    pub world: Arc<World>,
    pub position: V2,
    pub rotation: f64,
    pub layout: SensorLayout,
    outputs: Vec<ControlSlot>,
    pub settings: EvolutionSettings,
    pub rng: TrainingRandom,
    pub batch_count: usize,
    pub stats_phase: u64,
    pub eliminate_on_wall: bool,
    pub eliminate_when_idle: bool,
    pub agents: Vec<TrainingAgent>,
    pub generation: u64,
    pub tick: u64,
    pub batch_index: usize,
    pub mode: Mode,
    user_paused: bool,
    physics: Option<crate::simulation::SharedPhysics>,
}

impl TrainingRunner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        world: Arc<World>,
        position: V2,
        rotation: f64,
        layout: SensorLayout,
        output_names: &[String],
        settings: EvolutionSettings,
        rng: impl Into<TrainingRandom>,
        batch_count: usize,
        stats_phase: u64,
        eliminate_on_wall: bool,
        eliminate_when_idle: bool,
    ) -> TrainingRunner {
        assert!(batch_count > 0, "batch_count must be positive");
        // Controls(**dict(zip(names, outputs))): later duplicates win, which
        // sequential assignment reproduces.
        let outputs = output_names
            .iter()
            .map(|name| match name.to_lowercase().as_str() {
                "acceleration" => ControlSlot::Acceleration,
                "steering" => ControlSlot::Steering,
                "brake" => ControlSlot::Brake,
                "handbrake" => ControlSlot::Handbrake,
                "boost" => ControlSlot::Boost,
                other => panic!("unknown control output: {other}"),
            })
            .collect();
        TrainingRunner {
            world,
            position,
            rotation,
            layout,
            outputs,
            settings,
            rng: rng.into(),
            batch_count,
            stats_phase: stats_phase % 6,
            eliminate_on_wall,
            eliminate_when_idle,
            agents: Vec::new(),
            generation: 0,
            tick: 0,
            batch_index: 0,
            mode: Mode::Independent,
            user_paused: false,
            physics: None,
        }
    }

    /// GeneralConstants divides two integers before RoundToInt/Clamp.
    pub fn batches_per_tick(&self) -> usize {
        (8 / self.batch_count).clamp(1, 8)
    }

    pub fn start(&mut self, seed: &Network) -> Generation {
        let seed_result = AgentResult { network: seed, metrics: [None; 14], update_count: 0 };
        let initial = self.rng.reproduce(&[seed_result], &self.settings);
        self.install(&initial.networks, false);
        initial
    }

    /// Reproduce the initial population, generate its track, then install cars.
    /// Supply a fresh, independent track RNG state for each game generation.
    pub fn start_random_track(
        &mut self, seed: &Network, template: &Value,
        config: &mut crate::random_track::RandomTrackConfig, track_state: &mut [u64; 4],
    ) -> Result<(Generation, crate::random_track::GeneratedTrack), &'static str> {
        let seed_result = AgentResult { network: seed, metrics: [None; 14], update_count: 0 };
        let initial = self.rng.reproduce(&[seed_result], &self.settings);
        let track = self.generate_track(template, config, track_state)?;
        self.install(&initial.networks, false);
        Ok((initial, track))
    }

    /// Restore the state `next_generation` left behind: `networks` installed on
    /// reused vehicles as generation `generation` (restore `rng` separately).
    pub fn resume(&mut self, networks: &[Network], generation: u64) {
        self.install(networks, generation > 0);
        self.generation = generation;
    }

    fn install(&mut self, networks: &[Network], reused: bool) {
        self.install_with_novelty(networks.par_iter().cloned().collect(), reused, None);
    }

    /// Installs `networks`, which the new agents take over without copying.
    fn install_with_novelty(&mut self, networks: Vec<Network>, reused: bool, novelty: Option<&[f64]>) {
        let mut profile = crate::training_profile::Profile::new("install");
        assert!(!networks.is_empty(), "a training generation needs at least one network");
        if let Some(values) = novelty { assert_eq!(values.len(), networks.len()); }
        let count = networks.len() as f64;
        let size = networks[0].params.len();
        let mean: Vec<f64> = if novelty.is_none() {
            (0..size).into_par_iter().map(|i| networks.iter().map(|n| n.params[i]).fold(0.0, |a,b| a+b) / count).collect()
        } else { Vec::new() };
        profile.mark("mean");
        let world = &*self.world;
        let (position, rotation) = (self.position, self.rotation);
        let retained=if reused {self.agents.len().min(networks.len())}else{0};
        // VehicleManager keeps a FIFO of physical vehicles, independently of
        // network selection. Native contact and broad-phase caches survive.
        let mut old = std::mem::take(&mut self.agents).into_iter();
        let cars: Vec<Car> = networks.iter().map(|_| {
            if reused { old.next().map(|agent| agent.car).unwrap_or_else(|| Car::new(world, position, rotation)) }
            else { Car::new(world, position, rotation) }
        }).collect();
        profile.mark("vehicles");
        self.agents = cars.into_par_iter().zip(networks.into_par_iter()).enumerate()
            .map(|(i, (mut car, network))| {
                // VehicleManager resets both new and reused instances.
                car.queue_reset(position, rotation);
                let novelty = novelty.map_or_else(||
                    network.params.iter().zip(&mean).map(|(&a, &b)| crate::double_math::pow(a - b, 2.0)).fold(0.0,|a,b|a+b).sqrt(),
                    |values| values[i]);
                TrainingAgent {
                    network,
                    car,
                    stats: TrainingStats::new(novelty),
                    controls: Controls::default(),
                    pending_contact: false,
                    deactivated_at: None,
                    scratch: AgentScratch::default(),
                }
            })
            .collect();
        profile.mark("agents_and_novelty");
        if world.track.native_broadphase && self.physics.is_none() {
            self.physics=Some(crate::simulation::SharedPhysics::new(world,&mut self.agents.iter_mut().map(|a|&mut a.car).collect::<Vec<_>>()));
        }
        else if let Some(physics)=self.physics.as_mut() {
            physics.resize(world,&mut self.agents.iter_mut().map(|a|&mut a.car).collect::<Vec<_>>(),retained);
        }
        Self::advance_physics(world,&mut self.agents,&mut self.physics,self.eliminate_on_wall,false,0);
        // Reset above represents the first _IntegrateForces callback applying
        // the pending spawn request. StartNextGeneration runs on the sixth
        // eligible stable callback, before that callback's drive update.
        let mut stable_callbacks = 0;
        loop {
            if self.agents.iter().all(|a| crate::godot_math::F2::from(a.car.velocity).length() < 0.001 && (a.car.angular_velocity as f32) < 0.001) {
                stable_callbacks += 1;
            }
            if stable_callbacks == 6 { break; }
            Self::advance_physics(world,&mut self.agents,&mut self.physics,self.eliminate_on_wall,false,0);
        }
        self.tick = 0;
        self.batch_index = 0;
        profile.mark("passive_reset");
    }

    fn advance_physics(world:&World,agents:&mut[TrainingAgent],physics:&mut Option<crate::simulation::SharedPhysics>,eliminate:bool,drive:bool,tick:u64) {
        if let Some(physics)=physics {
            let controls:Vec<_>=agents.iter().map(|a|a.controls).collect();
            let was_active:Vec<_>=agents.iter().map(|a|a.car.active).collect();
            let contacts=physics.advance(world,&mut agents.iter_mut().map(|a|&mut a.car).collect::<Vec<_>>(),&controls,eliminate,drive);
            for ((agent,contact),was_active) in agents.iter_mut().zip(contacts).zip(was_active) {
                agent.pending_contact|=contact;
                if was_active && !agent.car.active{agent.deactivated_at=Some(tick);}
            }
        } else {
            agents.par_iter_mut().for_each(|a|{
                let was_active=a.car.active;
                a.pending_contact|=if drive{a.car.step(world,&a.controls,DT,eliminate)}else{a.car.passive_step(world,DT,eliminate)};
                if was_active && !a.car.active{a.deactivated_at=Some(tick);}
            });
        }
    }

    fn schedule(&self) -> Schedule<'_> {
        Schedule {
            world: &self.world,
            layout: &self.layout,
            outputs: &self.outputs,
            stats_phase: self.stats_phase,
            batch_size: self.agents.len().div_ceil(8),
            batches_per_tick: self.batches_per_tick(),
            eliminate_on_wall: self.eliminate_on_wall,
            eliminate_when_idle: self.eliminate_when_idle,
        }
    }

    /// GameManager.OnPause during an installed running generation. Freeze
    /// requests are deferred to native callbacks; the scheduler is not reset.
    pub fn set_paused(&mut self, paused: bool) {
        self.user_paused = paused;
        for agent in &mut self.agents { agent.car.set_frozen(paused); }
    }
    pub fn is_paused(&self) -> bool { self.user_paused }

    /// `TrainingRunner.step`: advance all cars one physics tick.
    pub fn step(&mut self) {
        self.advance(1, false);
    }

    /// Advance up to `ticks` ticks. With `stop_when_inactive`, stop after the
    /// statistics callback at which every car is inactive. Returns the
    /// number of native frames advanced. Paused frames preserve `tick` and
    /// the inference/statistics cursors while native physics continues.
    pub fn advance(&mut self, ticks: u64, stop_when_inactive: bool) -> u64 {
        self.advance_window(ticks, stop_when_inactive, None)
    }

    /// TrainGameManager checks elapsed time after stats and before driving.
    /// `time_limit_ticks` specifies the configured time limit in 1/60 seconds.
    pub fn advance_generation(&mut self, time_limit_ticks: u64) -> u64 {
        assert_eq!(self.stats_phase, 0, "fresh game generations use reset statistics counters");
        let bound = (time_limit_ticks / 6 + 2) * 6;
        self.advance_window(bound.saturating_sub(self.tick), true, Some(time_limit_ticks as f64 / 60.0))
    }

    fn advance_window(&mut self, ticks: u64, stop_when_inactive: bool, time_limit: Option<f64>) -> u64 {
        if ticks == 0 {
            return 0;
        }
        if self.user_paused {
            for _ in 0..ticks {
                Self::advance_physics(&self.world, &mut self.agents, &mut self.physics,
                    self.eliminate_on_wall, false, self.tick);
            }
            return ticks;
        }
        let start_tick = self.tick;
        let start_batch = self.batch_index;
        let bpt = self.batches_per_tick();
        let mut agents = std::mem::take(&mut self.agents);
        let mut physics = self.physics.take();
        for agent in &mut agents {
            agent.deactivated_at = None;
        }
        let batch_at = |k: u64| (start_batch + ((k - 1) as usize % 8) * bpt) % 8;
        let executed;
        let mut transition_without_drive = false;
        {
            let schedule = self.schedule_for(agents.len());
            match self.mode {
                Mode::Independent if physics.is_none() && !agents.is_empty() => {
                    // Car-major: each car runs the whole window alone, which keeps its
                    // network and state in cache and avoids per-tick barriers. A car stops
                    // before driving at the first statistics tick where the time
                    // limit is reached or, with `stop_when_inactive`, it is inactive.
                    // Cars never reactivate within a window, so the tick-major loop
                    // below would stop at the latest of these ticks, if every car
                    // stopped; cars that stopped earlier then catch up to it.
                    let stats_phase = self.stats_phase;
                    let limit_reached = |k: u64| {
                        let first = if stats_phase == 0 { 6 } else { stats_phase };
                        time_limit.is_some_and(|limit| ((start_tick + k - first) / 6) as f64 * 0.1 >= limit)
                    };
                    // Runs ticks `from..=to`; tick `from`'s statistics already ran
                    // when `resume` is set. Returns the tick at which `stop` held.
                    let run = |index: usize, agent: &mut TrainingAgent, from: u64, to: u64, resume: bool,
                               stop: &dyn Fn(u64, &TrainingAgent) -> bool| {
                        for k in from..=to {
                            let tick = start_tick + k;
                            if !(resume && k == from) {
                                schedule.update_stats(agent, tick);
                                if tick % 6 == stats_phase && stop(k, agent) {
                                    return Some(k);
                                }
                            }
                            schedule.drive_agent(index, agent, tick, batch_at(k));
                        }
                        None
                    };
                    let stops: Vec<Option<u64>> = agents.par_iter_mut().with_max_len(1).enumerate()
                        .map(|(index, agent)| run(index, agent, 1, ticks, false,
                            &|k, agent| limit_reached(k) || (stop_when_inactive && !agent.car.active)))
                        .collect();
                    let end = stops.iter().try_fold(0, |end, stop| stop.map(|k| end.max(k)));
                    agents.par_iter_mut().with_max_len(1).enumerate().zip(&stops).for_each(|((index, agent), stop)| {
                        let Some(stopped) = *stop else { return };
                        match end {
                            Some(end) if stopped < end => {
                                run(index, agent, stopped, end, true, &|k, agent| {
                                    assert!(!agent.car.active, "a car reactivated within a window");
                                    k == end
                                });
                            }
                            Some(_) => {}
                            None => { run(index, agent, stopped, ticks, true, &|_, _| false); }
                        }
                    });
                    transition_without_drive = end.is_some();
                    executed = end.unwrap_or(ticks);
                }
                _ => {
                    let mut k = 0;
                    while k < ticks {
                        k += 1;
                        let tick = start_tick + k;
                        let batch_index = batch_at(k);
                        agents.par_iter_mut().for_each(|agent| schedule.update_stats(agent, tick));
                        if tick % 6 == self.stats_phase {
                            let first = if self.stats_phase == 0 { 6 } else { self.stats_phase };
                            let elapsed = ((tick - first) / 6) as f64 * 0.1;
                            if (stop_when_inactive && agents.iter().all(|a| !a.car.active))
                                || time_limit.is_some_and(|limit| elapsed >= limit) {
                                transition_without_drive = true;
                                break;
                            }
                        }
                        if physics.is_some() {
                            agents.par_iter_mut().enumerate().for_each(|(index,agent)|schedule.update_controls(index,agent,batch_index));
                            Self::advance_physics(schedule.world,&mut agents,&mut physics,self.eliminate_on_wall,true,tick);
                        } else {
                            agents.par_iter_mut().enumerate().for_each(|(index, agent)| schedule.drive_agent(index, agent, tick, batch_index));
                        }
                    }
                    executed = k;
                }

            }
        }
        self.agents = agents;
        self.physics = physics;
        self.tick = start_tick + executed;
        self.batch_index = (start_batch + ((executed as usize - transition_without_drive as usize) % 8) * bpt) % 8;
        executed
    }

    /// A GPU simulator for this runner's world, vehicle and sensors, with
    /// room for `capacity` agents.
    pub fn gpu_sim<'a>(&self, world: &crate::gpu::GpuWorld<'a>, capacity: usize) -> Result<crate::gpu_sim::GpuSim<'a>, String> {
        let vehicle = crate::gpu_sim::vehicle_desc(&self.world.vehicle)?;
        // The GPU path sensors sample the Curve2D only; without track.curve the
        // CPU falls back to the baked path, which the GPU does not implement.
        let path_sensor = self.layout.sensors.iter().any(|s| matches!(s, Sensor::CorrectDirection | Sensor::TrackCurvature { .. }));
        if path_sensor && self.world.track.curve.is_none() && self.world.track.path.len() >= 2 {
            return Err("the scene has no track.curve (Curve2D control points), which the correct_direction and \
                        track_curvature sensors need on the GPU; add the curve to the scene or train without --gpu".into());
        }
        let sensors: Vec<_> = self.layout.sensors.iter().map(crate::gpu_sim::sensor_desc).collect();
        Ok(crate::gpu_sim::GpuSim::new(world, &vehicle, &sensors, capacity))
    }

    /// `advance_generation` on the GPU.
    pub fn advance_generation_gpu(&mut self, sim: &mut crate::gpu_sim::GpuSim, world: &crate::gpu::GpuWorld, time_limit_ticks: u64) -> Result<u64, String> {
        assert_eq!(self.stats_phase, 0, "fresh game generations use reset statistics counters");
        let bound = (time_limit_ticks / 6 + 2) * 6;
        self.advance_window_gpu(sim, world, bound.saturating_sub(self.tick), true, Some(time_limit_ticks as f64 / 60.0))
    }

    /// `advance_window` on the GPU: agents and cars are uploaded, advanced in
    /// the tick-major order and read back. Networks are uploaded when
    /// `sim.network_tag` differs from the generation.
    pub fn advance_window_gpu(&mut self, sim: &mut crate::gpu_sim::GpuSim, world: &crate::gpu::GpuWorld, ticks: u64,
                              stop_when_inactive: bool, time_limit: Option<f64>) -> Result<u64, String> {
        use crate::gpu_sim::*;
        let mut profile = crate::training_profile::Profile::new("gpu_window");
        if ticks == 0 {
            return Ok(0);
        }
        if self.user_paused || self.physics.is_some() {
            return Err("the GPU runner does not model paused or native-broadphase windows".into());
        }
        if self.agents.is_empty() {
            return Err("no agents".into());
        }
        let surfaces = &world.arrays.surfaces;
        let vehicle = &self.world.vehicle;
        for agent in &mut self.agents {
            agent.deactivated_at = None;
        }
        if sim.network_tag != Some(self.generation) || sim.network_count() != self.agents.len() {
            let src = std::array::from_fn(|c| {
                self.outputs.iter().rposition(|&slot| slot as usize == c).map_or(-1, |j| j as i32)
            });
            // Agents own the networks; the GPU uploader reuses its staging
            // allocation while copying them without cloning parameter vectors.
            sim.upload_agent_networks(&self.agents, src)?;
            profile.mark("networks");
            sim.network_tag = Some(self.generation);
        }
        // Retained staging buffers: filling them in place avoids allocating,
        // page-faulting and concatenating population-sized vectors per window.
        let (mut cars, mut agents) = sim.take_state_buffers();
        export_state_into(&self.agents, vehicle, surfaces, &mut cars, &mut agents)?;
        profile.mark("export_state");
        sim.upload(&cars, Some(&agents));
        profile.mark("upload_state");
        let args = WindowArgs {
            start_tick: self.tick,
            ticks,
            start_batch: self.batch_index as u32,
            batches_per_tick: self.batches_per_tick() as u32,
            stats_phase: self.stats_phase as u32,
            stop_when_inactive: stop_when_inactive as u32,
            eliminate_on_wall: self.eliminate_on_wall as u32,
            eliminate_when_idle: self.eliminate_when_idle as u32,
            has_time_limit: time_limit.is_some() as u32,
            pad: 0,
            time_limit: time_limit.unwrap_or(0.0),
        };
        let (executed, transition_without_drive) = sim.window(&args);
        profile.mark("window");
        // The upload has finished. Read back into the same allocations.
        sim.download(Some(&mut cars), Some(&mut agents));
        profile.mark("download_state");
        self.import_state(&cars, &agents, surfaces)?;
        profile.mark("import_state");
        sim.return_state_buffers(cars, agents);
        let bpt = self.batches_per_tick();
        self.tick += executed;
        self.batch_index = (self.batch_index + ((executed as usize - transition_without_drive as usize) % 8) * bpt) % 8;
        Ok(executed)
    }

    /// The rest of `gpu_import` for the whole population: `cars` and `agents`
    /// hold what `advance_window_gpu` read back.
    fn import_state(&mut self, cars: &[crate::gpu_sim::GpuCar], agents: &[crate::gpu_sim::GpuAgent],
                    surfaces: &crate::gpu_sim::SurfaceTable) -> Result<(), String> {
        self.agents.par_iter_mut().zip(cars).zip(agents).try_for_each(|((a, c), g)| {
            a.car.gpu_import(c, surfaces)?;
            crate::gpu_sim::agent_import(g, a);
            Ok::<(), String>(())
        })
    }

    fn schedule_for(&self, count: usize) -> Schedule<'_> {
        let mut schedule = self.schedule();
        schedule.batch_size = count.div_ceil(8);
        schedule
    }

    /// `TrainingRunner.run_generation(ticks)`.
    pub fn run_generation(&mut self, ticks: u64) -> Vec<AgentResult<'_>> {
        self.advance_generation(ticks);
        self.results()
    }

    pub fn results(&self) -> Vec<AgentResult<'_>> {
        self.agents.iter().map(TrainingAgent::result).collect()
    }

    /// Reproduce, redraw the native TileMap, then install the next population.
    pub fn next_generation_on_track(&mut self,world:Arc<World>,position:V2,rotation:f64)->Generation {
        let generation = self.reproduce_next();
        self.replace_track(world, position, rotation);
        self.install_next(&generation);
        generation
    }

    /// The original reproduction -> random track -> TileMap redraw -> car reset order.
    /// On success, `config` becomes the returned track's configuration, including
    /// a pinned start and any fallback shortening, for the following generation.
    pub fn next_generation_random_track(
        &mut self, template: &Value, config: &mut crate::random_track::RandomTrackConfig,
        track_state: &mut [u64; 4],
    ) -> Result<(Generation, crate::random_track::GeneratedTrack), &'static str> {
        let generation = self.reproduce_next();
        let track = self.generate_track(template, config, track_state)?;
        self.install_next(&generation);
        Ok((generation, track))
    }

    fn generate_track(
        &mut self, template: &Value, config: &mut crate::random_track::RandomTrackConfig,
        track_state: &mut [u64; 4],
    ) -> Result<crate::random_track::GeneratedTrack, &'static str> {
        let track = if let Some(seed) = template["runtime"]["hashcode_seed"].as_u64() {
            let seed = u32::try_from(seed).map_err(|_| "HashCode seed exceeds u32")?;
            crate::random_track::TrackGenerator::new(seed).generate(config, track_state)?
        } else { crate::random_track::generate(config, track_state)? };
        let scene = track.to_scene(template);
        let position = crate::curve::json_vector(&scene["reset_position"]).into();
        let rotation = scene["reset_rotation"].as_f64().unwrap();
        self.replace_track(Arc::new(World::from_scene(&scene)), position, rotation);
        *config = track.config.clone();
        Ok(track)
    }

    fn replace_track(&mut self, world: Arc<World>, position: V2, rotation: f64) {
        if let Some(physics) = self.physics.as_mut() { physics.queue_tilemap_redraw(self.world.clone()); }
        self.world = world;
        self.position = position;
        self.rotation = rotation;
    }

    fn reproduce_next(&mut self) -> Generation {
        let results: Vec<AgentResult> = self.agents.iter().map(TrainingAgent::result).collect();
        self.rng.reproduce(&results, &self.settings)
    }

    fn install_next(&mut self, generation: &Generation) {
        self.install(&generation.networks, true);
        self.stats_phase = 0;
        self.generation += 1;
    }

    /// Reproduce and reset cars on the current track.
    pub fn next_generation(&mut self) -> Generation {
        let generation = self.reproduce_next();
        self.install_next(&generation);
        generation
    }

    /// Reproduce with the same RNG stream, then compute population novelty on
    /// the GPU. Uploaded offspring stay in place for the next driving window,
    /// and the new agents take over the offspring networks without a copy
    /// (read them from `agents`; only the selection summary is returned).
    pub fn next_generation_gpu(&mut self, sim: &mut crate::gpu_sim::GpuSim) -> Result<Turnover, String> {
        let mut profile = crate::training_profile::Profile::new("gpu_turnover");
        // The preceding upload is complete, so its host buffer can hold noise
        // until reproduction finishes. Refilling it then uploads the offspring.
        let mut scratch = sim.take_parameter_buffer();
        let results: Vec<AgentResult> = self.agents.iter().map(TrainingAgent::result).collect();
        let Generation { networks, preserved_count, rewards } = self.rng.reproduce_with_scratch(&results, &self.settings, &mut scratch);
        sim.return_parameter_buffer(scratch);
        profile.mark("reproduce");
        let src = std::array::from_fn(|c| self.outputs.iter().rposition(|&slot| slot as usize == c).map_or(-1, |j| j as i32));
        sim.upload_networks(&networks, src)?;
        profile.mark("networks");
        let novelty = sim.novelty();
        profile.mark("novelty");
        self.install_with_novelty(networks, true, Some(&novelty));
        self.stats_phase = 0;
        self.generation += 1;
        sim.network_tag = Some(self.generation);
        profile.mark("install");
        Ok(Turnover { preserved_count, rewards })
    }
}

/// The selection summary of a GPU turnover (`Generation` without the
/// offspring, which the installed agents own).
pub struct Turnover {
    pub preserved_count: usize,
    pub rewards: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(path: &str) -> Value {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    /// A self-contained fixture: Autumn 04 with the formula vehicle.
    fn autumn_runner(population: usize) -> TrainingRunner {
        let root = env!("CARGO_MANIFEST_DIR");
        let scene = load(&format!("{root}/scenes_exact/autumn_04_formula_scene.json"));
        let template = load(&format!("{root}/formula_network_template.json"));
        let model = load(&format!("{root}/formula_trained_model_exact.json"));
        let spawn = &load(&format!("{root}/traces/autumn_04_spawn.json"))["frames"][0];
        let outputs: Vec<String> = template["outputs"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_owned()).collect();
        let settings = EvolutionSettings { population, selection_size: 3, preserve_parents_size: 2, mutation_rate: 0.3, weight_decay: 0.0, ..Default::default() };
        TrainingRunner::new(
            Arc::new(World::from_scene(&scene)), crate::world::vector(&spawn["position"]), spawn["rotation"].as_f64().unwrap(),
            SensorLayout::from_exports(&template, &model), &outputs, settings, PyRandom::new(5), 2, 0, true, true,
        )
    }

    fn agent_state(a: &TrainingAgent) -> String {
        let c = &a.car;
        format!("{:?} {:?} {:?} {:?} {} {} {} {:?} {:?} {:?} {:?} {:?}",
            c.position, c.velocity, c.acceleration, c.body_basis, c.active, c.tick, c.collision_count, c.wheels,
            a.controls, a.pending_contact, a.deactivated_at, a.stats)
    }

    /// Installing owned networks must leave the same agents as the cloning
    /// path, including reused vehicles, statistics and network ownership.
    #[test]
    fn owned_install_matches_cloning_install() {
        let mut a = autumn_runner(24);
        let mut b = autumn_runner(24);
        let seed = Network::xavier(&[20, 16, 5], &mut PyRandom::new(9));
        a.start(&seed);
        b.start(&seed);
        a.advance(90, false);
        b.advance(90, false);
        let results: Vec<AgentResult> = a.agents.iter().map(TrainingAgent::result).collect();
        let generation = a.rng.clone().reproduce(&results, &a.settings);
        a.install(&generation.networks, true);
        b.install_with_novelty(generation.networks.clone(), true, None);
        assert_eq!(a.agents.len(), b.agents.len());
        for (x, y) in a.agents.iter().zip(&b.agents) {
            assert_eq!(x.network, y.network);
            assert_eq!(agent_state(x), agent_state(y));
        }
        // With supplied novelty values the same agents result, novelty aside.
        let novelty: Vec<f64> = (0..generation.networks.len()).map(|i| i as f64 * 0.25).collect();
        let mut c = autumn_runner(24);
        c.start(&seed);
        c.advance(90, false);
        c.install_with_novelty(generation.networks.clone(), true, Some(&novelty));
        for (i, (x, y)) in a.agents.iter().zip(&c.agents).enumerate() {
            assert_eq!(x.network, y.network);
            assert_eq!(y.stats.network_novelty, novelty[i]);
            let mut stats = y.stats.clone();
            stats.network_novelty = x.stats.network_novelty;
            assert_eq!(format!("{:?}", x.stats), format!("{stats:?}"));
        }
    }

    /// The retained export buffers hold exactly what per-agent exports produce.
    #[test]
    fn export_state_into_matches_per_agent_export() {
        let mut runner = autumn_runner(24);
        runner.start(&Network::xavier(&[20, 16, 5], &mut PyRandom::new(9)));
        runner.advance(150, false);
        let surfaces = crate::gpu_sim::SurfaceTable::new(&runner.world.vehicle);
        let (mut cars, mut agents) = (vec![crate::gpu_sim::GpuCar::zeroed(); 3], Vec::new());
        export_state_into(&runner.agents, &runner.world.vehicle, &surfaces, &mut cars, &mut agents).unwrap();
        assert_eq!((cars.len(), agents.len()), (24, 24));
        for (i, a) in runner.agents.iter().enumerate() {
            let car = a.car.gpu_export(&runner.world.vehicle, &surfaces).unwrap();
            assert_eq!(format!("{:?}", cars[i]), format!("{car:?}"));
            assert_eq!(format!("{:?}", agents[i]), format!("{:?}", crate::gpu_sim::agent_export(a).unwrap()));
        }
    }
}
