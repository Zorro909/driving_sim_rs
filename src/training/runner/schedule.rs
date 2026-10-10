//! Preserve car-major independent windows and tick-major shared physics windows.

use super::{ControlSlot, Mode, TrainingRunner};
use crate::physics::car::{Controls, DT};
use crate::track::world::World;
use crate::training::{SensorLayout, TrainingAgent};
use rayon::prelude::*;

/// Immutable per-window schedule shared by all agents.
pub(super) struct Schedule<'a> {
    pub(super) world: &'a World,
    pub(super) layout: &'a SensorLayout,
    outputs: &'a [ControlSlot],
    stats_phase: u64,
    batch_size: usize,
    batches_per_tick: usize,
    eliminate_on_wall: bool,
    eliminate_when_idle: bool,
}

impl Schedule<'_> {
    pub(super) fn update_stats(&self, agent: &mut TrainingAgent, tick: u64) {
        agent.update_stats(self.world, tick, self.stats_phase, self.eliminate_when_idle);
    }
    pub(super) fn drive_agent(&self, index: usize, agent: &mut TrainingAgent, tick: u64, batch_index: usize) {
        self.update_controls(index, agent, batch_index);
        self.step_agent(agent, tick);
    }
    /// The physics step of `drive_agent`, with the agent's current controls.
    pub(super) fn step_agent(&self, agent: &mut TrainingAgent, tick: u64) {
        let was_active = agent.car.active;
        let contact = agent.car.step(self.world, &agent.controls, DT, self.eliminate_on_wall);
        agent.pending_contact |= contact;
        if was_active && !agent.car.active {
            agent.deactivated_at = Some(tick);
        }
    }
    /// Whether agent `index` belongs to an inference batch due at `batch_index`.
    pub(super) fn infers(&self, index: usize, batch_index: usize) -> bool {
        let batch = index / self.batch_size;
        (batch + 8 - batch_index) % 8 < self.batches_per_tick
    }
    pub(super) fn update_controls(&self, index: usize, agent: &mut TrainingAgent, batch_index: usize) {
        if self.infers(index, batch_index) && agent.car.active {
            let scratch = &mut agent.scratch;
            self.layout
                .read_into(self.world, &agent.car, &mut scratch.sensors, &mut scratch.inputs);
            self.set_controls(agent);
        }
    }
    /// The network forward of the read inputs into the agent's controls.
    pub(super) fn set_controls(&self, agent: &mut TrainingAgent) {
        let scratch = &mut agent.scratch;
        let outputs = agent
            .network
            .forward_into(&scratch.inputs, &mut scratch.forward, self.world.math);
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

impl TrainingRunner {
    pub(super) fn schedule(&self) -> Schedule<'_> {
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
        for agent in &mut self.agents {
            agent.car.set_frozen(paused);
        }
    }
    pub fn is_paused(&self) -> bool {
        self.user_paused
    }

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
        let (ticks, time_limit) = self.generation_window(time_limit_ticks);
        self.advance_window(ticks, true, Some(time_limit))
    }

    /// The remaining window of a generation with `time_limit_ticks`: the tick
    /// bound and the time limit in seconds that `advance_generation` applies.
    pub(crate) fn generation_window(&self, time_limit_ticks: u64) -> (u64, f64) {
        assert_eq!(
            self.stats_phase, 0,
            "fresh game generations use reset statistics counters"
        );
        (
            super::remaining_generation_ticks(time_limit_ticks, self.tick),
            time_limit_ticks as f64 / 60.0,
        )
    }

    pub(super) fn advance_window(&mut self, ticks: u64, stop_when_inactive: bool, time_limit: Option<f64>) -> u64 {
        if ticks == 0 {
            return 0;
        }
        if self.user_paused {
            for _ in 0..ticks {
                Self::advance_physics(
                    &self.world,
                    &mut self.agents,
                    &mut self.physics,
                    self.eliminate_on_wall,
                    false,
                    self.tick,
                );
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
                    let run = |index: usize,
                               agent: &mut TrainingAgent,
                               from: u64,
                               to: u64,
                               resume: bool,
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
                    let stops: Vec<Option<u64>> = agents
                        .par_iter_mut()
                        .with_max_len(1)
                        .enumerate()
                        .map(|(index, agent)| {
                            run(index, agent, 1, ticks, false, &|k, agent| {
                                limit_reached(k) || (stop_when_inactive && !agent.car.active)
                            })
                        })
                        .collect();
                    let end = stops.iter().try_fold(0, |end, stop| stop.map(|k| end.max(k)));
                    agents
                        .par_iter_mut()
                        .with_max_len(1)
                        .enumerate()
                        .zip(&stops)
                        .for_each(|((index, agent), stop)| {
                            let Some(stopped) = *stop else { return };
                            match end {
                                Some(end) if stopped < end => {
                                    run(index, agent, stopped, end, true, &|k, agent| {
                                        assert!(!agent.car.active, "a car reactivated within a window");
                                        k == end
                                    });
                                }
                                Some(_) => {}
                                None => {
                                    run(index, agent, stopped, ticks, true, &|_, _| false);
                                }
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
                        agents
                            .par_iter_mut()
                            .for_each(|agent| schedule.update_stats(agent, tick));
                        if tick % 6 == self.stats_phase {
                            let first = if self.stats_phase == 0 { 6 } else { self.stats_phase };
                            let elapsed = ((tick - first) / 6) as f64 * 0.1;
                            if (stop_when_inactive && agents.iter().all(|a| !a.car.active))
                                || time_limit.is_some_and(|limit| elapsed >= limit)
                            {
                                transition_without_drive = true;
                                break;
                            }
                        }
                        if physics.is_some() {
                            agents
                                .par_iter_mut()
                                .enumerate()
                                .for_each(|(index, agent)| schedule.update_controls(index, agent, batch_index));
                            Self::advance_physics(
                                schedule.world,
                                &mut agents,
                                &mut physics,
                                self.eliminate_on_wall,
                                true,
                                tick,
                            );
                        } else {
                            agents
                                .par_iter_mut()
                                .enumerate()
                                .for_each(|(index, agent)| schedule.drive_agent(index, agent, tick, batch_index));
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

    pub(super) fn schedule_for(&self, count: usize) -> Schedule<'_> {
        let mut schedule = self.schedule();
        schedule.batch_size = count.div_ceil(8);
        schedule
    }
}
