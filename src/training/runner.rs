//! Population installation, vehicle reuse, and generation turnover.

use super::agent::AgentScratch;
use super::optimizer::{self, Optimizer, Produced, Record};
use super::{SensorLayout, TrainingAgent, TrainingRandom, TrainingStats};
use crate::math::vec2::V2;
use crate::nn::network::Network;
use crate::physics::car::{Car, Controls, SensorScratch, DT};
use crate::track::world::World;
use crate::training::evolution::{AgentResult, EvolutionSettings, Generation};
use rayon::prelude::*;
use serde_json::Value;
use std::sync::Arc;

mod gpu;
mod rays;
mod schedule;

pub use gpu::export_state_into;

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

/// A window advanced tick by tick with the ray sensors served by an external
/// raycaster (the WebGPU raycaster of the WebAssembly library). After
/// `TrainingRunner::begin_ray_window`, each tick is `prepare_ray_tick`,
/// `ray_queries`, and `finish_ray_tick` with the hits, until `prepare_ray_tick`
/// returns false; `end_ray_window` then commits the window. With hits equal
/// to the track's raycasts, the agents equal those of `advance`.
#[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
pub(crate) struct RayWindow {
    start_tick: u64,
    start_batch: usize,
    ticks: u64,
    stop_when_inactive: bool,
    time_limit: Option<f64>,
    executed: u64,
    finished: bool,
    transition_without_drive: bool,
    batch_index: usize,
    /// The agents whose controls update in the prepared tick.
    pub(crate) inferring: Vec<usize>,
}

#[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
impl RayWindow {
    /// The tick the prepared step drives (`TrainingRunner::tick` plus the
    /// executed count).
    pub(crate) fn tick(&self) -> u64 {
        self.start_tick + self.executed
    }
}

/// Values per car in `TrainingRunner::car_states`, in `CAR_STATE_FIELDS` order.
pub(crate) const CAR_STATE_STRIDE: usize = 14;
/// `car_states` layout: position, rotation, velocity, angular velocity,
/// active flag, distance score, lap count, best lap time (-1 without a lap),
/// wall contacts, boost energy, the steering wheel angle in degrees, and the
/// statistics update count.
#[cfg(any(feature = "server", all(target_arch = "wasm32", feature = "wasm")))]
pub(crate) const CAR_STATE_FIELDS: [&str; CAR_STATE_STRIDE] = [
    "x",
    "y",
    "rotation",
    "velocity_x",
    "velocity_y",
    "angular_velocity",
    "active",
    "score",
    "lap_count",
    "best_lap_time",
    "collision_count",
    "boost",
    "wheel_angle",
    "update_count",
];

pub struct TrainingRunner {
    pub world: Arc<World>,
    pub position: V2,
    pub rotation: f64,
    pub layout: SensorLayout,
    outputs: Vec<ControlSlot>,
    pub settings: EvolutionSettings,
    pub rng: TrainingRandom,
    /// Builds each new population; replaced when `settings.algorithm` changes.
    pub(crate) optimizer: Box<dyn Optimizer>,
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
    physics: Option<crate::physics::simulation::SharedPhysics>,
}

impl TrainingRunner {
    // Preserve the public constructor's independent replay settings.
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
            optimizer: optimizer::create(&settings.algorithm),
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

    /// Replaces the optimizer when `settings.algorithm` no longer names it.
    /// A fresh optimizer starts from the cars it is next given.
    fn sync_optimizer(&mut self) {
        if self.optimizer.name() != self.settings.algorithm {
            self.optimizer = optimizer::create(&self.settings.algorithm);
        }
    }

    /// Forgets the optimizer's state, which belonged to cars no longer installed.
    pub(crate) fn reset_optimizer(&mut self) {
        self.optimizer = optimizer::create(&self.settings.algorithm);
    }

    fn start_produced(&mut self, seed: &Network, trace: bool) -> Produced {
        self.sync_optimizer();
        self.optimizer.start(seed, &self.settings, &mut self.rng, trace)
    }

    /// The population after the current agents, by the configured algorithm.
    fn turnover(&mut self, scores: Option<Vec<f64>>, scratch: &mut Vec<f64>, trace: bool) -> Produced {
        self.sync_optimizer();
        let results: Vec<AgentResult> = self.agents.iter().map(TrainingAgent::result).collect();
        self.optimizer
            .next(&results, scores, &self.settings, &mut self.rng, scratch, trace)
    }

    pub fn start(&mut self, seed: &Network) -> Generation {
        let initial = self.start_produced(seed, false).generation;
        self.install(&initial.networks, false);
        initial
    }

    /// `start`, returning the record a checkpoint rebuilds the first population from.
    pub(crate) fn start_traced(&mut self, seed: &Network) -> Record {
        let Produced { generation, record } = self.start_produced(seed, true);
        self.install_with_novelty(generation.networks, false, None);
        record.expect("a traced start returns its record")
    }

    /// `resume` with networks the new agents take over without a copy.
    pub(crate) fn resume_owned(&mut self, networks: Vec<Network>, generation: u64) {
        self.reset_optimizer();
        self.install_with_novelty(networks, generation > 0, None);
        self.generation = generation;
    }

    /// `next_generation`, installing the offspring without a copy and
    /// returning the record a checkpoint rebuilds them from.
    pub(crate) fn next_generation_traced(&mut self) -> (Turnover, Record) {
        let Produced {
            generation:
                Generation {
                    networks,
                    preserved_count,
                    rewards,
                },
            record,
        } = self.turnover(None, &mut Vec::new(), true);
        self.install_with_novelty(networks, true, None);
        self.stats_phase = 0;
        self.generation += 1;
        (
            Turnover {
                preserved_count,
                rewards,
            },
            record.expect("a traced turnover returns its record"),
        )
    }

    /// Reproduce the initial population, generate its track, then install cars.
    /// Supply a fresh, independent track RNG state for each game generation.
    pub fn start_random_track(
        &mut self,
        seed: &Network,
        template: &Value,
        config: &mut crate::track::random_track::RandomTrackConfig,
        track_state: &mut [u64; 4],
    ) -> Result<(Generation, crate::track::random_track::GeneratedTrack), &'static str> {
        let initial = self.start_produced(seed, false).generation;
        let track = self.generate_track(template, config, track_state)?;
        self.install(&initial.networks, false);
        Ok((initial, track))
    }

    /// Restore the state `next_generation` left behind: `networks` installed on
    /// reused vehicles as generation `generation` (restore `rng` separately).
    pub fn resume(&mut self, networks: &[Network], generation: u64) {
        self.reset_optimizer();
        self.install(networks, generation > 0);
        self.generation = generation;
    }

    fn install(&mut self, networks: &[Network], reused: bool) {
        self.install_with_novelty(networks.par_iter().cloned().collect(), reused, None);
    }

    /// Installs `networks`, which the new agents take over without copying.
    fn install_with_novelty(&mut self, networks: Vec<Network>, reused: bool, novelty: Option<&[f64]>) {
        let mut profile = crate::training::training_profile::Profile::new("install");
        assert!(!networks.is_empty(), "a training generation needs at least one network");
        if let Some(values) = novelty {
            assert_eq!(values.len(), networks.len());
        }
        let count = networks.len() as f64;
        let size = networks[0].params.len();
        let mean: Vec<f64> = if novelty.is_none() {
            (0..size)
                .into_par_iter()
                .map(|i| networks.iter().map(|n| n.params[i]).fold(0.0, |a, b| a + b) / count)
                .collect()
        } else {
            Vec::new()
        };
        profile.mark("mean");
        let world = &*self.world;
        let (position, rotation) = (self.position, self.rotation);
        let retained = if reused {
            self.agents.len().min(networks.len())
        } else {
            0
        };
        // VehicleManager keeps a FIFO of physical vehicles, independently of
        // network selection. Native contact and broad-phase caches survive.
        let mut old = std::mem::take(&mut self.agents).into_iter();
        let cars: Vec<Car> = networks
            .iter()
            .map(|_| {
                if reused {
                    old.next()
                        .map(|agent| agent.car)
                        .unwrap_or_else(|| Car::new(world, position, rotation))
                } else {
                    Car::new(world, position, rotation)
                }
            })
            .collect();
        profile.mark("vehicles");
        self.agents = cars
            .into_par_iter()
            .zip(networks.into_par_iter())
            .enumerate()
            .map(|(i, (mut car, network))| {
                // VehicleManager resets both new and reused instances.
                car.queue_reset(position, rotation);
                let novelty = novelty.map_or_else(
                    || {
                        network
                            .params
                            .iter()
                            .zip(&mean)
                            .map(|(&a, &b)| crate::math::double_math::pow(a - b, 2.0))
                            .fold(0.0, |a, b| a + b)
                            .sqrt()
                    },
                    |values| values[i],
                );
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
            self.physics = Some(crate::physics::simulation::SharedPhysics::new(
                world,
                &mut self.agents.iter_mut().map(|a| &mut a.car).collect::<Vec<_>>(),
            ));
        } else if let Some(physics) = self.physics.as_mut() {
            physics.resize(
                world,
                &mut self.agents.iter_mut().map(|a| &mut a.car).collect::<Vec<_>>(),
                retained,
            );
        }
        self.settle_reset();
        profile.mark("passive_reset");
    }

    /// Reset an evaluation without replacing networks or advancing evolution.
    /// Random training uses independent cars; native shared redraw replay is separate.
    pub fn reset_evaluation(&mut self) {
        assert!(
            !self.world.track.native_broadphase,
            "track batches require independent cars"
        );
        let (position, rotation) = (self.position, self.rotation);
        self.agents.par_iter_mut().for_each(|agent| {
            let novelty = agent.stats.network_novelty;
            agent.car.queue_reset(position, rotation);
            agent.stats = TrainingStats::new(novelty);
            agent.controls = Controls::default();
            agent.pending_contact = false;
            agent.deactivated_at = None;
            agent.scratch.sensors = SensorScratch::default();
        });
        self.stats_phase = 0;
        self.settle_reset();
    }

    fn settle_reset(&mut self) {
        let world = &*self.world;
        Self::advance_physics(
            world,
            &mut self.agents,
            &mut self.physics,
            self.eliminate_on_wall,
            false,
            0,
        );
        // Reset above represents the first _IntegrateForces callback applying
        // the pending spawn request. StartNextGeneration runs on the sixth
        // eligible stable callback, before that callback's drive update.
        let mut stable_callbacks = 0;
        loop {
            if self.agents.iter().all(|a| {
                crate::math::godot_math::F2::from(a.car.velocity).length() < 0.001
                    && (a.car.angular_velocity as f32) < 0.001
            }) {
                stable_callbacks += 1;
            }
            if stable_callbacks == 6 {
                break;
            }
            Self::advance_physics(
                world,
                &mut self.agents,
                &mut self.physics,
                self.eliminate_on_wall,
                false,
                0,
            );
        }
        self.tick = 0;
        self.batch_index = 0;
    }

    fn advance_physics(
        world: &World,
        agents: &mut [TrainingAgent],
        physics: &mut Option<crate::physics::simulation::SharedPhysics>,
        eliminate: bool,
        drive: bool,
        tick: u64,
    ) {
        if let Some(physics) = physics {
            let controls: Vec<_> = agents.iter().map(|a| a.controls).collect();
            let was_active: Vec<_> = agents.iter().map(|a| a.car.active).collect();
            let contacts = physics.advance(
                world,
                &mut agents.iter_mut().map(|a| &mut a.car).collect::<Vec<_>>(),
                &controls,
                eliminate,
                drive,
            );
            for ((agent, contact), was_active) in agents.iter_mut().zip(contacts).zip(was_active) {
                agent.pending_contact |= contact;
                if was_active && !agent.car.active {
                    agent.deactivated_at = Some(tick);
                }
            }
        } else {
            agents.par_iter_mut().for_each(|a| {
                let was_active = a.car.active;
                a.pending_contact |= if drive {
                    a.car.step(world, &a.controls, DT, eliminate)
                } else {
                    a.car.passive_step(world, DT, eliminate)
                };
                if was_active && !a.car.active {
                    a.deactivated_at = Some(tick);
                }
            });
        }
    }

    pub fn results(&self) -> Vec<AgentResult<'_>> {
        self.agents.iter().map(TrainingAgent::result).collect()
    }

    /// The original reproduction -> random track -> TileMap redraw -> car reset order.
    /// On success, `config` becomes the returned track's configuration, including
    /// a pinned start and any fallback shortening, for the following generation.
    pub fn next_generation_random_track(
        &mut self,
        template: &Value,
        config: &mut crate::track::random_track::RandomTrackConfig,
        track_state: &mut [u64; 4],
    ) -> Result<(Generation, crate::track::random_track::GeneratedTrack), &'static str> {
        let generation = self.reproduce_next();
        let track = self.generate_track(template, config, track_state)?;
        self.install_next(&generation);
        Ok((generation, track))
    }

    fn generate_track(
        &mut self,
        template: &Value,
        config: &mut crate::track::random_track::RandomTrackConfig,
        track_state: &mut [u64; 4],
    ) -> Result<crate::track::random_track::GeneratedTrack, &'static str> {
        let track = if let Some(seed) = template["runtime"]["hashcode_seed"].as_u64() {
            let seed = u32::try_from(seed).map_err(|_| "HashCode seed exceeds u32")?;
            crate::track::random_track::TrackGenerator::new(seed).generate(config, track_state)?
        } else {
            crate::track::random_track::generate(config, track_state)?
        };
        let scene = track.to_scene(template);
        let position = crate::track::curve::json_vector(&scene["reset_position"]).into();
        let rotation = scene["reset_rotation"].as_f64().unwrap();
        self.replace_track(Arc::new(World::from_scene(&scene)), position, rotation);
        *config = track.config.clone();
        Ok(track)
    }

    /// Select a prepared track before starting or installing a generation.
    pub fn replace_track(&mut self, world: Arc<World>, position: V2, rotation: f64) {
        if let Some(physics) = self.physics.as_mut() {
            physics.queue_tilemap_redraw(self.world.clone());
        }
        if self.physics.is_none() {
            for agent in &mut self.agents {
                agent.car.clear_track_contacts();
            }
        }
        self.world = world;
        self.position = position;
        self.rotation = rotation;
    }

    fn reproduce_next(&mut self) -> Generation {
        self.turnover(None, &mut Vec::new(), false).generation
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

    pub fn next_generation_with_fitness(&mut self, scores: Vec<f64>) -> Generation {
        let generation = self.turnover(Some(scores), &mut Vec::new(), false).generation;
        self.install_next(&generation);
        generation
    }
}

/// The selection summary of a GPU turnover (`Generation` without the
/// offspring, which the installed agents own).
pub struct Turnover {
    pub preserved_count: usize,
    pub rewards: Vec<f64>,
}

#[cfg(test)]
mod tests;
