//! Float math used by the game's Proton UCRT, which incorporates musl.
//!
//! atan2f/atanf/sinf/cosf adapted from Wine proton_11.0 libs/musl/src/math.
//! Conversion to float by Ian Lance Taylor, Cygnus Support.
//! Trigonometric kernels optimized by Bruce D. Evans.
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//! Developed at SunPro, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this software is freely
//! granted, provided that this notice is preserved.

// Preserve the musl polynomial and reduction literals exactly as transcribed.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

fn sin_kernel(x: f64) -> f32 {
    let z = x * x;
    let s1 = -0.16666666666666666;
    if x > -7.8175831586122513e-3 && x < 7.8175831586122513e-3 {
        return (x * (1.0 + s1 * z)) as f32;
    }
    let w = z * z;
    let r = -0.0001984126984126984 + z * 2.7557319223985893e-6;
    let s = z * x;
    ((x + s * (s1 + z * 0.008333333333333333)) + s * w * r) as f32
}
fn cos_kernel(x: f64) -> f32 {
    let z = x * x;
    if x > -7.816314697265625e-3 && x < 7.816314697265625e-3 {
        return (1.0 - 0.5 * z) as f32;
    }
    (1.0 + z
        * (-0.5
            + z * (0.041666666666666664
                + z * (-0.001388888888888889 + z * (2.48015873015873e-5 + z * -2.755731922398589e-7))))) as f32
}
fn reduce(x: f32) -> (i32, f64) {
    if (x.to_bits() & 0x7fffffff) >= 0x4dc90fdb {
        return crate::math::float_reduction::reduce(x);
    }
    let toint = 1.5 / f64::EPSILON;
    let mut n = (x as f64 * 6.36619772367581382433e-1 + toint) - toint;
    let rem = |n| x as f64 - n * 1.57079631090164184570 - n * 1.58932547735281966916e-8;
    let mut y = rem(n);
    if y < -(std::f32::consts::FRAC_PI_4 as f64) {
        n -= 1.0;
        y = rem(n);
    } else if y > std::f32::consts::FRAC_PI_4 as f64 {
        n += 1.0;
        y = rem(n);
    }
    (n as i32, y)
}
// x - x preserves the runtime NaN result for nonfinite inputs.
#[allow(clippy::eq_op)]
pub fn sin(x: f32) -> f32 {
    let bits = x.to_bits() & 0x7fffffff;
    let negative = x.is_sign_negative();
    let a = x as f64;
    let p = std::f64::consts::FRAC_PI_2;
    if bits <= 0x3f490fda {
        return if bits < 0x39800000 { x } else { sin_kernel(a) };
    }
    if bits <= 0x407b53d1 {
        if bits <= 0x4016cbe3 {
            return if negative {
                -cos_kernel(a + p)
            } else {
                cos_kernel(a - p)
            };
        }
        return sin_kernel(if negative { -(a + 2.0 * p) } else { -(a - 2.0 * p) });
    }
    if bits <= 0x40e231d5 {
        if bits <= 0x40afeddf {
            return if negative {
                cos_kernel(a + 3.0 * p)
            } else {
                -cos_kernel(a - 3.0 * p)
            };
        }
        return sin_kernel(if negative { a + 4.0 * p } else { a - 4.0 * p });
    }
    if !x.is_finite() {
        return x - x;
    }
    let (n, y) = reduce(x);
    match n & 3 {
        0 => sin_kernel(y),
        1 => cos_kernel(y),
        2 => sin_kernel(-y),
        _ => -cos_kernel(y),
    }
}
// x - x preserves the runtime NaN result for nonfinite inputs.
#[allow(clippy::eq_op)]
pub fn cos(x: f32) -> f32 {
    let bits = x.to_bits() & 0x7fffffff;
    let negative = x.is_sign_negative();
    let a = x as f64;
    let p = std::f64::consts::FRAC_PI_2;
    if bits <= 0x3f490fda {
        return if bits < 0x39800000 { 1.0 } else { cos_kernel(a) };
    }
    if bits <= 0x407b53d1 {
        if bits > 0x4016cbe3 {
            return -cos_kernel(if negative { a + 2.0 * p } else { a - 2.0 * p });
        }
        return sin_kernel(if negative { a + p } else { p - a });
    }
    if bits <= 0x40e231d5 {
        if bits > 0x40afeddf {
            return cos_kernel(if negative { a + 4.0 * p } else { a - 4.0 * p });
        }
        return sin_kernel(if negative { -a - 3.0 * p } else { a - 3.0 * p });
    }
    if !x.is_finite() {
        return x - x;
    }
    let (n, y) = reduce(x);
    match n & 3 {
        0 => cos_kernel(y),
        1 => sin_kernel(-y),
        2 => -cos_kernel(y),
        _ => sin_kernel(y),
    }
}

fn atan(mut x: f32) -> f32 {
    let negative = x.is_sign_negative();
    let bits = x.to_bits() & 0x7fff_ffff;
    let high = [0x3eed6338, 0x3f490fda, 0x3f7b985e, 0x3fc90fda].map(f32::from_bits);
    let low = [0x31ac3769, 0x33222168, 0x33140fb4, 0x33a22168].map(f32::from_bits);
    if bits >= 0x4c800000 {
        return if x.is_nan() { x } else { high[3].copysign(x) };
    }
    let id;
    if bits < 0x3ee00000 {
        if bits < 0x39800000 {
            return x;
        }
        id = None;
    } else {
        x = x.abs();
        if bits < 0x3f980000 {
            if bits < 0x3f300000 {
                id = Some(0);
                x = (2.0 * x - 1.0) / (2.0 + x);
            } else {
                id = Some(1);
                x = (x - 1.0) / (x + 1.0);
            }
        } else if bits < 0x401c0000 {
            id = Some(2);
            x = (x - 1.5) / (1.0 + 1.5 * x);
        } else {
            id = Some(3);
            x = -1.0 / x;
        }
    }
    let a = [
        3.3333328366e-01f32,
        -1.9999158382e-01,
        1.4253635705e-01,
        -1.0648017377e-01,
        6.1687607318e-02,
    ];
    let z = x * x;
    let w = z * z;
    let s1 = z * (a[0] + w * (a[2] + w * a[4]));
    let s2 = w * (a[1] + w * a[3]);
    match id {
        None => x - x * (s1 + s2),
        Some(i) => {
            let z = high[i] - ((x * (s1 + s2) - low[i]) - x);
            if negative {
                -z
            } else {
                z
            }
        }
    }
}

pub fn atan2(y: f32, x: f32) -> f32 {
    let pi = std::f32::consts::PI;
    let pi_lo = f32::from_bits(0xb3bbbd2e);
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    let (mut ix, mut iy) = (x.to_bits(), y.to_bits());
    if ix == 0x3f800000 {
        return atan(y);
    }
    let m = ((iy >> 31) & 1) | ((ix >> 30) & 2);
    ix &= 0x7fffffff;
    iy &= 0x7fffffff;
    if iy == 0 {
        return match m {
            0 | 1 => y,
            2 => pi,
            _ => -pi,
        };
    }
    if ix == 0 {
        return (pi / 2.0).copysign(y);
    }
    if ix == 0x7f800000 {
        return if iy == 0x7f800000 {
            match m {
                0 => pi / 4.0,
                1 => -pi / 4.0,
                2 => 3.0 * pi / 4.0,
                _ => -3.0 * pi / 4.0,
            }
        } else {
            match m {
                0 => 0.0,
                1 => -0.0,
                2 => pi,
                _ => -pi,
            }
        };
    }
    if ix + (26 << 23) < iy || iy == 0x7f800000 {
        return (pi / 2.0).copysign(y);
    }
    let z = if m & 2 != 0 && iy + (26 << 23) < ix {
        0.0
    } else {
        atan((y / x).abs())
    };
    match m {
        0 => z,
        1 => -z,
        2 => pi - (z - pi_lo),
        _ => (z - pi_lo) - pi,
    }
}

/// The shipped x86 Godot compiler combines adjacent sin/cos calls into an x87
/// helper. Its internal pi approximation is observable for large arguments.
/// Managed MathF scalar and packed routines above use different algorithms.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn engine_sin_cos(x: f32) -> (f32, f32) {
    let mut sine = 0.0f32;
    let mut cosine = 0.0f32;
    // The x87 stack is balanced on both paths. FSINCOS leaves its operand on
    // range failure; FPREM1 reduces against the same extended FLDPI constant.
    // FSCALE doubles pi without rounding it to the caller's x87 precision.
    unsafe {
        core::arch::asm!(
            "fld dword ptr [{input}]",
            "fsincos",
            "fnstsw ax",
            "test ax, 0x400",
            "jz 3f",
            "fldpi",
            "fld1",
            "fxch st(1)",
            "fscale",
            "fstp st(1)",
            "fxch st(1)",
            "2:",
            "fprem1",
            "fnstsw ax",
            "test ax, 0x400",
            "jnz 2b",
            "fstp st(1)",
            "fsincos",
            "3:",
            "fstp dword ptr [{cosine}]",
            "fstp dword ptr [{sine}]",
            input=in(reg) &x,sine=in(reg) &mut sine,cosine=in(reg) &mut cosine,
            out("ax") _,out("st(0)") _,out("st(1)") _,out("st(2)") _,options(nostack),
        );
    }
    (sine, cosine)
}
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub fn engine_sin_cos(x: f32) -> (f32, f32) {
    engine_sin_cos_portable(x)
}

/// Inputs up to this magnitude have been compared exhaustively with x87 `FSINCOS`.
#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
const ENGINE_PORTABLE_DOMAIN: f32 = 16.0;

/// `engine_sin_cos` without the x87 instruction set (the WebAssembly library
/// and other non-x86 targets): the fdlibm kernels in f64 after a Cody-Waite
/// reduction, as gpu/sim/math.h `engine_sin_cos_raw`. The float results
/// equal x87 `FSINCOS` for every float with |x| <= `ENGINE_PORTABLE_DOMAIN`
/// (examples/gpu_engine_scan.rs found no exception, and
/// `portable_engine_sin_cos_matches_x87` in this module rechecks the domain
/// natively). Larger finite arguments use the same reduction, whose products
/// stay exact for |x| below 2^20; those results have not been compared.
/// FSINCOS leaves NaNs quieted and turns infinities into the x87 indefinite.
// The literals are the fdlibm constants as written (2/pi is the rounded value).
#[allow(clippy::approx_constant, clippy::excessive_precision)]
#[cfg(any(test, not(any(target_arch = "x86", target_arch = "x86_64"))))]
pub(crate) fn engine_sin_cos_portable(xf: f32) -> (f32, f32) {
    if xf.is_nan() {
        let quiet = f32::from_bits(xf.to_bits() | 0x0040_0000);
        return (quiet, quiet);
    }
    if xf.is_infinite() {
        let indefinite = f32::from_bits(0xffc0_0000);
        return (indefinite, indefinite);
    }
    let x = xf as f64;
    let n = (x * 6.36619772367581382433e-01).round_ties_even();
    // Keeps -0 (and every tiny argument) unchanged when no reduction happens.
    let y = if n == 0.0 {
        x
    } else {
        (x - n * 1.57079632673412561417e+00) - n * 6.07710050650619224932e-11
    };
    let z = y * y;
    let w = z * z;
    let r = 8.33333333332248946124e-03
        + z * (-1.98412698298579493134e-04 + z * 2.75573137070700676789e-06)
        + z * w * (-2.50507602534068634195e-08 + z * 1.58969099521155010221e-10);
    // fdlibm returns tiny arguments unchanged, which also keeps the sign of -0.
    let sy = if y.abs() < f64::from_bits(0x3e40_0000_0000_0000) {
        y
    } else {
        y + z * y * (-1.66666666666666324348e-01 + z * r)
    };
    let rc = z * (4.16666666666666019037e-02 + z * (-1.38888888888741095749e-03 + z * 2.48015872894767294178e-05))
        + w * w * (-2.75573143513906633035e-07 + z * (2.08757232129817482790e-09 + z * -1.13596475577881948265e-11));
    let hz = 0.5 * z;
    let one_minus = 1.0 - hz;
    let cy = one_minus + (((1.0 - one_minus) - hz) + z * rc);
    let q = (n as i64) & 3;
    let (mut sv, mut cv) = if q & 1 != 0 { (cy, sy) } else { (sy, cy) };
    if q & 2 != 0 {
        sv = -sv;
    }
    if (q + 1) & 2 != 0 {
        cv = -cv;
    }
    (sv as f32, cv as f32)
}

pub fn engine_sin(x: f32) -> f32 {
    engine_sin_cos(x).0
}
pub fn engine_cos(x: f32) -> f32 {
    engine_sin_cos(x).1
}

/// Comparisons with x87 `FSINCOS`, which only x86 provides.
#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod engine_tests {
    use super::*;

    #[test]
    fn large_engine_arguments_match_the_capture_at_every_x87_precision() {
        fn control_word() -> u16 {
            let mut word = 0u16;
            unsafe {
                core::arch::asm!("fnstcw word ptr [{word}]", word = in(reg) &mut word, options(nostack));
            }
            word
        }
        struct RestoreControlWord(u16);
        impl Drop for RestoreControlWord {
            fn drop(&mut self) {
                unsafe {
                    core::arch::asm!("fldcw word ptr [{word}]", word = in(reg) &self.0, options(nostack));
                }
            }
        }

        let original = control_word();
        let _restore = RestoreControlWord(original);
        // Captured results at the FSINCOS range boundary and the largest float.
        let rows = [
            (0x5eff_ffff, 0x3e54_4316, 0xbf7a_7091),
            (0x5f00_0000, 0x3f7c_c47f, 0x3e22_3675),
            (0x7f7f_ffff, 0x3f7a_128f, 0xbe5b_1472),
        ];
        for precision in [0x0000, 0x0200, 0x0300] {
            let word = (original & !0x0300) | precision;
            unsafe {
                core::arch::asm!("fldcw word ptr [{word}]", word = in(reg) &word, options(nostack));
            }
            for (input, sine, cosine) in rows {
                let actual = engine_sin_cos(f32::from_bits(input));
                assert_eq!(
                    (actual.0.to_bits(), actual.1.to_bits()),
                    (sine, cosine),
                    "input {input:08x}, x87 control word {word:04x}"
                );
                assert_eq!(
                    control_word(),
                    word,
                    "engine math changed the caller's x87 control word"
                );
            }
        }
    }

    fn check(bits: u32) -> Option<(u32, u32, u32, u32, u32)> {
        let x = f32::from_bits(bits);
        let (s, c) = engine_sin_cos(x);
        let (ps, pc) = engine_sin_cos_portable(x);
        (s.to_bits() != ps.to_bits() || c.to_bits() != pc.to_bits()).then_some((
            bits,
            s.to_bits(),
            c.to_bits(),
            ps.to_bits(),
            pc.to_bits(),
        ))
    }

    /// Every 61st float with |x| <= 16 plus the domain edges and specials,
    /// against the x87 instruction (`cargo test -- --ignored` runs the whole domain).
    #[test]
    fn portable_engine_sin_cos_matches_x87() {
        let limit = ENGINE_PORTABLE_DOMAIN.to_bits();
        let mut mismatches = Vec::new();
        for sign in [0u32, 0x8000_0000] {
            for bits in (0..=limit).step_by(61).chain([
                limit,
                limit - 1,
                0x3fc9_0fdb,
                0x4049_0fdb,
                0x40c9_0fdb,
                0x3f49_0fdb,
                1,
                0,
            ]) {
                mismatches.extend(check(sign | bits));
                if mismatches.len() > 8 {
                    break;
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "portable engine trig differs from x87: {mismatches:x?}"
        );
        for bits in [0x7fc0_0000u32, 0xffc0_0000, 0x7f80_0000, 0xff80_0000, 0x7f80_0001] {
            let (s, c) = engine_sin_cos(f32::from_bits(bits));
            let (ps, pc) = engine_sin_cos_portable(f32::from_bits(bits));
            assert!(s.is_nan() && c.is_nan() && ps.is_nan() && pc.is_nan());
        }
    }

    #[test]
    #[ignore = "exhaustive: about two billion evaluations"]
    fn portable_engine_sin_cos_matches_x87_exhaustively() {
        let limit = ENGINE_PORTABLE_DOMAIN.to_bits();
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let per = (limit as usize + 1).div_ceil(threads) as u32;
        let mismatches: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads as u32)
                .map(|t| {
                    scope.spawn(move || {
                        let mut found = Vec::new();
                        for sign in [0u32, 0x8000_0000] {
                            for bits in (t * per)..((t + 1) * per).min(limit + 1) {
                                found.extend(check(sign | bits));
                            }
                        }
                        found
                    })
                })
                .collect();
            handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        assert!(
            mismatches.is_empty(),
            "{} mismatches, first {:x?}",
            mismatches.len(),
            &mismatches[..mismatches.len().min(8)]
        );
    }
}
