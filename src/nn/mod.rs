//! Flat network parameters and scalar or AVX2 inference.

pub mod network;
#[cfg(target_arch = "x86_64")]
mod network_simd;
