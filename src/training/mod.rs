//! Game training scheduler, statistics, and population lifecycle.
//!
//! Independent mode runs each car through a window, then catches inactive cars
//! up to the final tick. Native shared broadphase scenes use a tick-major loop
//! to preserve contact ordering. Deactivated cars retain passive physics, while
//! their driving metrics stop updating.

pub mod batch_evaluation;
#[cfg(not(target_arch = "wasm32"))]
pub mod evaluation;
pub mod evolution;
pub mod game_random;
pub mod pyrandom;
pub mod session;
pub(crate) mod training_profile;

mod agent;
mod lineage;
mod runner;
mod sensors;
mod stats;

pub use agent::TrainingAgent;
pub(crate) use lineage::Lineage;
pub use lineage::TrainingRandom;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub(crate) use runner::CAR_STATE_FIELDS;
pub(crate) use runner::CAR_STATE_STRIDE;
pub use runner::{export_state_into, Mode, TrainingRunner};
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use sensors::{sensor_type, vision_angle};
pub use sensors::{vision_length, SensorLayout};
pub use stats::{ScoreTracker, TrainingStats};
