//! Bit-exact driving physics, generated tracks, and neural-network training.

/// GPU state layouts and the native HIP loader.
pub mod gpu;
/// Runtime-compatible arithmetic and vector types.
pub mod math;
/// Neural-network parameters and inference.
pub mod nn;
/// Vehicle dynamics, contacts, and recorded-state reconstruction.
pub mod physics;
/// Scene geometry, spatial queries, and generated tracks.
pub mod track;
/// Population execution, evolution, evaluation, and embedding sessions.
pub mod training;
/// JavaScript bindings and WebGPU execution.
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub mod wasm;
