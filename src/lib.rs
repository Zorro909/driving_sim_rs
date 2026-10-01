//! CPU driving simulator matching the original game runtime.

pub mod car;
pub mod collision;
mod broadphase;
pub mod evolution;
pub mod batch_evaluation;
pub mod network;
#[cfg(target_arch = "x86_64")]
mod network_simd;
pub mod pymath;
pub mod pyrandom;
pub mod training;
mod training_profile;
pub mod vec2;
pub mod world;

pub mod godot_math;
pub mod native_math;
mod float_reduction;
mod managed_trig;
pub mod bsp;

pub mod trace_state;

pub mod curve;
mod segment_grid;
mod path_segments;
pub mod game_random;
pub mod simulation;

pub mod random_track;
// Random training tracks. The buffered producer uses native threads and
// prepares HIP worlds, so the WebAssembly target keeps only track selection.
pub mod training_tracks;

pub mod double_math;
mod double_math_tables;

/// Embedding sessions over `training::TrainingRunner` (the WebAssembly
/// library and examples/wasm_reference.rs).
pub mod session;

/// GPU simulator (HIP, gpu/sim): bit-exact with the CPU. The loader uses
/// dlopen, so the WebAssembly target leaves it out.
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu;
pub mod gpu_sim;

/// The WebAssembly library (`--target wasm32-unknown-unknown --features wasm`):
/// JavaScript bindings and the WebGPU raycaster.
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub mod wasm;

#[cfg(not(target_arch = "wasm32"))]
pub mod evaluation;
