//! The C runtime math the game sees, which depends on where it runs.
//!
//! Under Proton the game's ucrtbase is Wine's, which forwards to musl
//! (native_math.rs, double_math.rs). On Windows it is Microsoft's, whose FMA3
//! code paths changed in Windows 11 24H2 (build 26100). The Windows functions
//! are generated from math/kernels/ucrt.h (ucrt.rs); sinf and cosf are musl's
//! with a table of the inputs where Microsoft's result differs.
use crate::math::{double_math, native_math, ucrt};

/// Which C runtime math a session reproduces. The value is part of a run: a
/// network trained under one profile can drive differently under another.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MathProfile {
    /// Linux and macOS through Proton: Wine's ucrtbase, which is musl.
    #[default]
    Proton,
    /// Windows 10, and Windows 11 up to 23H2 (build 22631), on FMA3 hardware.
    Win10Fma3,
    /// Windows 11 24H2 (build 26100) and later on FMA3 hardware.
    Win11Fma3,
}

impl MathProfile {
    pub const ALL: [MathProfile; 3] = [MathProfile::Proton, MathProfile::Win10Fma3, MathProfile::Win11Fma3];

    /// The name of the profile in options, checkpoints and on the command line.
    pub fn name(self) -> &'static str {
        match self {
            MathProfile::Proton => "proton",
            MathProfile::Win10Fma3 => "win10-fma3",
            MathProfile::Win11Fma3 => "win11-fma3",
        }
    }

    pub fn parse(name: &str) -> Option<MathProfile> {
        MathProfile::ALL.into_iter().find(|p| p.name() == name)
    }

    /// The number the GPU simulators read (World::math_profile, the WGSL shader).
    pub fn index(self) -> u32 {
        self as u32
    }

    /// The profile of a Windows build number.
    pub fn for_windows_build(build: u32) -> MathProfile {
        if build >= 26100 {
            MathProfile::Win11Fma3
        } else {
            MathProfile::Win10Fma3
        }
    }

    /// The profile the game would use on this machine: Windows by its build,
    /// everything else Proton. In a browser, the platform's
    /// (wasm::math_profile).
    pub fn detect() -> MathProfile {
        #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
        return crate::wasm::math_profile::browser_math_profile();
        #[cfg(windows)]
        if let Some(build) = windows_build() {
            return MathProfile::for_windows_build(build);
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
        MathProfile::Proton
    }

    pub fn sin(self, x: f32) -> f32 {
        let musl = native_math::sin(x);
        match self {
            MathProfile::Proton => musl,
            MathProfile::Win10Fma3 | MathProfile::Win11Fma3 => ucrt::win_sinf_fix_all(x, musl),
        }
    }

    pub fn cos(self, x: f32) -> f32 {
        let musl = native_math::cos(x);
        match self {
            MathProfile::Proton => musl,
            MathProfile::Win10Fma3 | MathProfile::Win11Fma3 => ucrt::win_cosf_fix_all(x, musl),
        }
    }

    pub fn atan2(self, y: f32, x: f32) -> f32 {
        match self {
            MathProfile::Proton => native_math::atan2(y, x),
            MathProfile::Win10Fma3 | MathProfile::Win11Fma3 => ucrt::win_atan2f(y, x),
        }
    }

    pub fn exp(self, x: f64) -> f64 {
        match self {
            MathProfile::Proton => double_math::exp(x),
            MathProfile::Win10Fma3 => ucrt::win10_exp(x),
            MathProfile::Win11Fma3 => ucrt::win11_exp(x),
        }
    }

    pub fn pow(self, x: f64, y: f64) -> f64 {
        match self {
            MathProfile::Proton => double_math::pow(x, y),
            MathProfile::Win10Fma3 => ucrt::win10_pow(x, y),
            MathProfile::Win11Fma3 => ucrt::win11_pow(x, y),
        }
    }

    pub fn log(self, x: f64) -> f64 {
        match self {
            MathProfile::Proton => double_math::log(x),
            MathProfile::Win10Fma3 | MathProfile::Win11Fma3 => ucrt::win_log(x),
        }
    }

    /// .NET Math.Tanh, which calls the C runtime's tanh.
    pub fn tanh(self, x: f64) -> f64 {
        match self {
            MathProfile::Proton => crate::nn::network::game_tanh(x),
            MathProfile::Win10Fma3 => ucrt::win10_tanh(x),
            MathProfile::Win11Fma3 => ucrt::win11_tanh(x),
        }
    }
}

impl std::fmt::Display for MathProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for MathProfile {
    type Err = String;

    fn from_str(name: &str) -> Result<MathProfile, String> {
        MathProfile::parse(name).ok_or_else(|| {
            let names: Vec<_> = MathProfile::ALL.iter().map(|p| p.name()).collect();
            format!("unknown math profile {name:?} (expected {})", names.join(", "))
        })
    }
}

/// The real build number: GetVersionEx reports 9200 to unmanifested programs.
#[cfg(windows)]
fn windows_build() -> Option<u32> {
    use windows_sys::Wdk::System::SystemServices::RtlGetVersion;
    use windows_sys::Win32::System::SystemInformation::OSVERSIONINFOW;
    let mut info = OSVERSIONINFOW {
        dwOSVersionInfoSize: std::mem::size_of::<OSVERSIONINFOW>() as u32,
        ..Default::default()
    };
    // SAFETY: info is a writable OSVERSIONINFOW whose size field is set.
    let status = unsafe { RtlGetVersion(&mut info) };
    (status == 0).then_some(info.dwBuildNumber)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// math/fixtures/<name>.bin rows: input words, then proton, win10-fma3 and
    /// win11-fma3 result bits (tools/ucrt/fixtures.py).
    fn rows<const W: usize>(name: &str, arity: usize) -> Vec<Vec<u64>> {
        let path = format!("{}/math/fixtures/{name}.bin", env!("CARGO_MANIFEST_DIR"));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let words: Vec<u64> = bytes
            .chunks_exact(W)
            .map(|c| c.iter().rev().fold(0u64, |a, &b| (a << 8) | b as u64))
            .collect();
        let rows: Vec<Vec<u64>> = words.chunks_exact(arity + 3).map(<[u64]>::to_vec).collect();
        assert!(rows.len() > 100, "{path}");
        rows
    }

    fn check(name: &str, arity: usize, width: usize, f: impl Fn(MathProfile, &[u64]) -> u64) {
        let rows = if width == 4 {
            rows::<4>(name, arity)
        } else {
            rows::<8>(name, arity)
        };
        for (column, profile) in MathProfile::ALL.into_iter().enumerate() {
            let bad: Vec<_> = rows
                .iter()
                .filter(|row| f(profile, &row[..arity]) != row[arity + column])
                .map(|row| {
                    format!(
                        "{:x?} = {:#x}, want {:#x}",
                        &row[..arity],
                        f(profile, &row[..arity]),
                        row[arity + column]
                    )
                })
                .collect();
            assert!(
                bad.is_empty(),
                "{name} {profile}: {} of {} rows differ, e.g. {:?}",
                bad.len(),
                rows.len(),
                &bad[..bad.len().min(4)]
            );
        }
    }

    fn f32_of(bits: u64) -> f32 {
        f32::from_bits(bits as u32)
    }

    #[test]
    fn sin_matches_fixtures() {
        check("sinf", 1, 4, |p, x| p.sin(f32_of(x[0])).to_bits() as u64);
    }

    #[test]
    fn cos_matches_fixtures() {
        check("cosf", 1, 4, |p, x| p.cos(f32_of(x[0])).to_bits() as u64);
    }

    #[test]
    fn atan2_matches_fixtures() {
        check("atan2f", 2, 4, |p, x| {
            p.atan2(f32_of(x[0]), f32_of(x[1])).to_bits() as u64
        });
    }

    #[test]
    fn exp_matches_fixtures() {
        check("exp", 1, 8, |p, x| p.exp(f64::from_bits(x[0])).to_bits());
    }

    #[test]
    fn log_matches_fixtures() {
        check("log", 1, 8, |p, x| p.log(f64::from_bits(x[0])).to_bits());
    }

    #[test]
    fn tanh_matches_fixtures() {
        check("tanh", 1, 8, |p, x| p.tanh(f64::from_bits(x[0])).to_bits());
    }

    #[test]
    fn pow_matches_fixtures() {
        check("pow", 2, 8, |p, x| {
            p.pow(f64::from_bits(x[0]), f64::from_bits(x[1])).to_bits()
        });
    }

    #[test]
    fn names_round_trip() {
        for p in MathProfile::ALL {
            assert_eq!(p.name().parse::<MathProfile>(), Ok(p));
            assert_eq!(serde_json::to_value(p).unwrap(), serde_json::json!(p.name()));
        }
        assert_eq!(MathProfile::for_windows_build(22631), MathProfile::Win10Fma3);
        assert_eq!(MathProfile::for_windows_build(26100), MathProfile::Win11Fma3);
    }

    /// Every float through sinf and cosf against the scans' tables is too slow
    /// for every run: cargo test --release -- --ignored exhaustive
    #[test]
    #[ignore]
    fn exhaustive_sin_cos_tables_are_sorted_and_signed() {
        let mut fixed = 0u64;
        for bits in 0..=u32::MAX {
            let x = f32::from_bits(bits);
            for p in [MathProfile::Win10Fma3, MathProfile::Win11Fma3] {
                fixed += (p.sin(x).to_bits() != native_math::sin(x).to_bits()) as u64;
                fixed += (p.cos(x).to_bits() != native_math::cos(x).to_bits()) as u64;
            }
        }
        // Both profiles, both signs: the four tables' rows.
        let keys = 2
            * 2
            * (ucrt::UCRT_SINF_COUNT + ucrt::UCRT_SINF_WIDE_COUNT + ucrt::UCRT_COSF_COUNT + ucrt::UCRT_COSF_WIDE_COUNT);
        assert_eq!(fixed, keys as u64);
    }
}
