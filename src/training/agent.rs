//! Training agents own their controls, statistics, and inference scratch.

use super::TrainingStats;
use crate::nn::network::{ForwardScratch, Network};
use crate::physics::car::{Car, Controls, SensorScratch};
use crate::track::world::World;
use crate::training::evolution::AgentResult;

/// Per-agent scratch buffers, kept with the agent so threads never share them.
#[derive(Default, Clone)]
pub(super) struct AgentScratch {
    pub(super) sensors: SensorScratch,
    pub(super) inputs: Vec<f64>,
    pub(super) forward: ForwardScratch,
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
    pub(super) scratch: AgentScratch,
}

impl TrainingAgent {
    /// `Schedule::update_stats`: statistics on ticks where
    /// `tick % 6 == stats_phase`, then the idle elimination.
    pub fn update_stats(&mut self, world: &World, tick: u64, stats_phase: u64, eliminate_when_idle: bool) {
        let was_active = self.car.active;
        if tick % 6 == stats_phase {
            let first_stats_tick = if stats_phase == 0 { 6 } else { stats_phase };
            self.stats.update(
                world,
                &self.car,
                &self.controls,
                tick - first_stats_tick,
                self.pending_contact,
            );
            self.pending_contact = false;
            if self.car.active && eliminate_when_idle && self.stats.idle_ticks > 80 {
                self.car.deactivate();
            }
        }
        if was_active && !self.car.active {
            self.deactivated_at = Some(tick);
        }
    }
    pub fn result(&self) -> AgentResult<'_> {
        AgentResult {
            network: &self.network,
            metrics: self.stats.metrics(),
            update_count: self.stats.update_count,
        }
    }
}
