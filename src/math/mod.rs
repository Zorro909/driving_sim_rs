//! Runtime-compatible scalar arithmetic, tables, and vectors.

pub mod double_math;
mod double_math_tables;
mod float_reduction;
pub mod godot_math;
mod managed_sine_tables;
mod managed_trig;
pub mod native_math;
pub mod pymath;
pub mod ucrt_math;
mod ucrt_tables;
pub mod vec2;

use std::sync::atomic::{AtomicBool, Ordering};

/// Which C runtime's `sinf`, `cosf`, `atan2f`, `exp`, `pow`, `tanh` and `log` the game under
/// simulation uses. Results differ in the last bits, so a run, its checkpoints and the GPU
/// library must agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Libm {
    /// The Proton/Wine UCRT, which incorporates musl (`native_math`, `double_math`). The default.
    Proton,
    /// The Windows UCRT, FMA3 paths (`ucrt_math`).
    Windows,
}

impl Libm {
    /// `"proton"` or `"windows"`.
    pub fn parse(name: &str) -> Option<Libm> {
        match name {
            "proton" => Some(Libm::Proton),
            "windows" => Some(Libm::Windows),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Libm::Proton => "proton",
            Libm::Windows => "windows",
        }
    }
}

/// The flavour named by the ALTD_LIBM environment variable, if it is set.
pub fn libm_from_env() -> Result<Option<Libm>, String> {
    match std::env::var("ALTD_LIBM") {
        Ok(name) => Libm::parse(&name)
            .map(Some)
            .ok_or_else(|| format!("ALTD_LIBM must be proton or windows, not {name:?}")),
        Err(_) => Ok(None),
    }
}

static WINDOWS_LIBM: AtomicBool = AtomicBool::new(false);

/// The process-wide math flavour. Every simulator in the process uses it.
pub fn libm() -> Libm {
    if windows_libm() {
        Libm::Windows
    } else {
        Libm::Proton
    }
}

#[inline(always)]
pub(crate) fn windows_libm() -> bool {
    WINDOWS_LIBM.load(Ordering::Relaxed)
}

/// Selects the math flavour for the whole process, before any simulation starts. The Windows
/// flavour needs a Windows MSVC build, whose `f64::ln` is the UCRT's `log`; the port has none.
pub fn set_libm(libm: Libm) -> Result<(), String> {
    if libm == Libm::Windows && !cfg!(all(windows, target_env = "msvc")) {
        return Err("the Windows math flavour needs a Windows MSVC build: its log is the system's".into());
    }
    WINDOWS_LIBM.store(libm == Libm::Windows, Ordering::Relaxed);
    Ok(())
}
