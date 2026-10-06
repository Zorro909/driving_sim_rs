//! Scratch training configuration, checkpoints, and stopping rules.

mod checkpoint;
mod runner;
mod stop;

pub(super) use runner::train_scratch;
pub(super) use stop::StopRules;

use super::args::{Schedule, ScratchReward, TrackOptions};
use altd_sim::math::profile::MathProfile;
use std::path::PathBuf;

#[cfg(test)]
mod tests;

pub(super) struct ScratchConfig {
    pub(super) tracks: TrackOptions,
    pub(super) scene: PathBuf,
    pub(super) spawn_trace: Option<PathBuf>,
    pub(super) network: Option<PathBuf>,
    pub(super) model: Option<PathBuf>,
    pub(super) out_dir: PathBuf,
    pub(super) shape: Vec<usize>,
    pub(super) population: usize,
    pub(super) generations: usize,
    pub(super) ticks: u64,
    pub(super) mutation_start: f64,
    pub(super) mutation_end: f64,
    pub(super) schedule: Schedule,
    pub(super) settings: Option<PathBuf>,
    pub(super) reward: Option<ScratchReward>,
    pub(super) seed: i64,
    pub(super) init_network: Option<PathBuf>,
    pub(super) init_population: Option<PathBuf>,
    pub(super) game_rng_state: Option<PathBuf>,
    pub(super) batch_count: usize,
    pub(super) checkpoint_every: usize,
    pub(super) save_final_candidate: bool,
    pub(super) stop: StopRules,
    pub(super) eliminate_on_wall: Option<bool>,
    pub(super) idle_eliminate: Option<bool>,
    pub(super) resume: bool,
    pub(super) gpu: bool,
    /// `--math-profile`; a resumed run keeps its own.
    pub(super) math_profile: Option<MathProfile>,
}

/// Mutation rate used to create generation `generation`. The final checkpoint's
/// population, bred after the last generation, keeps the end rate.
fn scheduled_rate(config: &ScratchConfig, generation: usize) -> f64 {
    let (start, end) = (config.mutation_start, config.mutation_end);
    if config.generations < 2 {
        return start;
    }
    let t = (generation as f64 / (config.generations - 1) as f64).min(1.0);
    match config.schedule {
        Schedule::Geometric => start * (end / start).powf(t),
        Schedule::Linear => start + (end - start) * t,
    }
}
