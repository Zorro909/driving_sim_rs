//! GPU state layouts, exports, and native HIP execution.

/// Runtime-loaded HIP library and device geometry.
#[cfg(not(target_arch = "wasm32"))]
pub mod hip;
/// Shared GPU layouts and simulation windows.
pub mod simulation;
