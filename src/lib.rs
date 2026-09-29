//! CPU driving simulator matching the original game runtime.

pub mod car;
pub mod collision;
mod broadphase;
pub mod evolution;
pub mod network;
#[cfg(target_arch = "x86_64")]
mod network_simd;
pub mod pymath;
pub mod pyrandom;
pub mod training;
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

pub mod double_math;
mod double_math_tables;

/// GPU simulator (HIP, gpu/sim): bit-exact with the CPU.
pub mod gpu;
pub mod gpu_sim;

