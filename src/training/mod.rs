//! Game training scheduler, statistics, and population lifecycle.
//!
//! Each car runs through a window on its own, then inactive cars catch up to
//! the final tick. Deactivated cars retain passive physics, while their
//! driving metrics stop updating.

pub mod batch_evaluation;
#[cfg(not(target_arch = "wasm32"))]
pub mod evaluation;
pub mod evolution;
pub mod game_random;
pub mod generation_stats;
pub mod pyrandom;
pub mod session;
pub(crate) mod training_profile;

mod agent;
mod ars;
mod lineage;
mod optimizer;
mod runner;
#[cfg(all(feature = "server", not(target_arch = "wasm32")))]
pub(crate) use runner::remaining_generation_ticks;
mod sensors;
mod stats;

pub use agent::TrainingAgent;
pub(crate) use lineage::Lineage;
pub use lineage::TrainingRandom;
#[cfg(any(feature = "server", all(target_arch = "wasm32", feature = "wasm")))]
pub(crate) use runner::CAR_STATE_FIELDS;
pub(crate) use runner::CAR_STATE_STRIDE;
pub use runner::{export_state_into, TrainingRunner};
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use sensors::{sensor_type, vision_angle};
pub use sensors::{vision_length, SensorLayout};
pub use stats::{ScoreTracker, TrainingStats};
