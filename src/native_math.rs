//! Float math used by the game's Proton UCRT, which incorporates musl.
//!
//! atan2f/atanf/sinf/cosf adapted from Wine proton_11.0 libs/musl/src/math.
//! Conversion to float by Ian Lance Taylor, Cygnus Support.
//! Trigonometric kernels optimized by Bruce D. Evans.
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//! Developed at SunPro, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this software is freely
//! granted, provided that this notice is preserved.

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
    (1.0 + z * (-0.5 + z * (0.041666666666666664 + z * (-0.001388888888888889
        + z * (2.48015873015873e-5 + z * -2.755731922398589e-7))))) as f32
}
fn reduce(x: f32) -> (i32, f64) {
    if (x.to_bits() & 0x7fffffff)>=0x4dc90fdb {return crate::float_reduction::reduce(x);}
    let toint = 1.5 / f64::EPSILON;
    let mut n = (x as f64 * 6.36619772367581382433e-1 + toint) - toint;
    let rem = |n| x as f64 - n * 1.57079631090164184570 - n * 1.58932547735281966916e-8;
    let mut y = rem(n);
    if y < -(std::f32::consts::FRAC_PI_4 as f64) { n -= 1.0; y = rem(n); }
    else if y > std::f32::consts::FRAC_PI_4 as f64 { n += 1.0; y = rem(n); }
    (n as i32, y)
}
pub fn sin(x: f32) -> f32 {
    let bits = x.to_bits() & 0x7fffffff;
    let negative = x.is_sign_negative();
    let a = x as f64;
    let p = std::f64::consts::FRAC_PI_2;
    if bits <= 0x3f490fda { return if bits < 0x39800000 { x } else { sin_kernel(a) }; }
    if bits <= 0x407b53d1 {
        if bits <= 0x4016cbe3 { return if negative { -cos_kernel(a + p) } else { cos_kernel(a - p) }; }
        return sin_kernel(if negative { -(a + 2.0 * p) } else { -(a - 2.0 * p) });
    }
    if bits <= 0x40e231d5 {
        if bits <= 0x40afeddf { return if negative { cos_kernel(a + 3.0 * p) } else { -cos_kernel(a - 3.0 * p) }; }
        return sin_kernel(if negative { a + 4.0 * p } else { a - 4.0 * p });
    }
    if !x.is_finite() { return x - x; }
    let (n, y) = reduce(x);
    match n & 3 { 0 => sin_kernel(y), 1 => cos_kernel(y), 2 => sin_kernel(-y), _ => -cos_kernel(y) }
}
pub fn cos(x: f32) -> f32 {
    let bits = x.to_bits() & 0x7fffffff;
    let negative = x.is_sign_negative();
    let a = x as f64;
    let p = std::f64::consts::FRAC_PI_2;
    if bits <= 0x3f490fda { return if bits < 0x39800000 { 1.0 } else { cos_kernel(a) }; }
    if bits <= 0x407b53d1 {
        if bits > 0x4016cbe3 { return -cos_kernel(if negative { a + 2.0 * p } else { a - 2.0 * p }); }
        return sin_kernel(if negative { a + p } else { p - a });
    }
    if bits <= 0x40e231d5 {
        if bits > 0x40afeddf { return cos_kernel(if negative { a + 4.0 * p } else { a - 4.0 * p }); }
        return sin_kernel(if negative { -a - 3.0 * p } else { a - 3.0 * p });
    }
    if !x.is_finite() { return x - x; }
    let (n, y) = reduce(x);
    match n & 3 { 0 => cos_kernel(y), 1 => sin_kernel(-y), 2 => -cos_kernel(y), _ => sin_kernel(y) }
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
        if bits < 0x39800000 { return x; }
        id = None;
    } else {
        x = x.abs();
        if bits < 0x3f980000 {
            if bits < 0x3f300000 {
                id = Some(0); x = (2.0 * x - 1.0) / (2.0 + x);
            } else {
                id = Some(1); x = (x - 1.0) / (x + 1.0);
            }
        } else if bits < 0x401c0000 {
            id = Some(2); x = (x - 1.5) / (1.0 + 1.5 * x);
        } else {
            id = Some(3); x = -1.0 / x;
        }
    }
    let a = [3.3333328366e-01f32, -1.9999158382e-01, 1.4253635705e-01,
        -1.0648017377e-01, 6.1687607318e-02];
    let z = x * x;
    let w = z * z;
    let s1 = z * (a[0] + w * (a[2] + w * a[4]));
    let s2 = w * (a[1] + w * a[3]);
    match id {
        None => x - x * (s1 + s2),
        Some(i) => {
            let z = high[i] - ((x * (s1 + s2) - low[i]) - x);
            if negative { -z } else { z }
        }
    }
}

pub fn atan2(y: f32, x: f32) -> f32 {
    let pi = std::f32::consts::PI;
    let pi_lo = f32::from_bits(0xb3bbbd2e);
    if x.is_nan() || y.is_nan() { return x + y; }
    let (mut ix, mut iy) = (x.to_bits(), y.to_bits());
    if ix == 0x3f800000 { return atan(y); }
    let m = ((iy >> 31) & 1) | ((ix >> 30) & 2);
    ix &= 0x7fffffff; iy &= 0x7fffffff;
    if iy == 0 {
        return match m { 0 | 1 => y, 2 => pi, _ => -pi };
    }
    if ix == 0 { return (pi / 2.0).copysign(y); }
    if ix == 0x7f800000 {
        return if iy == 0x7f800000 {
            match m { 0 => pi / 4.0, 1 => -pi / 4.0, 2 => 3.0 * pi / 4.0, _ => -3.0 * pi / 4.0 }
        } else { match m { 0 => 0.0, 1 => -0.0, 2 => pi, _ => -pi } };
    }
    if ix + (26 << 23) < iy || iy == 0x7f800000 { return (pi / 2.0).copysign(y); }
    let z = if m & 2 != 0 && iy + (26 << 23) < ix { 0.0 } else { atan((y / x).abs()) };
    match m { 0 => z, 1 => -z, 2 => pi - (z - pi_lo), _ => (z - pi_lo) - pi }
}

/// The shipped x86 Godot compiler combines adjacent sin/cos calls into an x87
/// helper. Its internal pi approximation is observable for large arguments.
/// Managed MathF scalar and packed routines above use different algorithms.
#[cfg(any(target_arch="x86",target_arch="x86_64"))]
pub fn engine_sin_cos(x:f32)->(f32,f32) {
    let mut sine=0.0f32;let mut cosine=0.0f32;
    // The x87 stack is balanced on both paths. FSINCOS leaves its operand on
    // range failure; FPREM1 reduces against the same extended FLDPI constant.
    unsafe {core::arch::asm!(
        "fld dword ptr [{input}]",
        "fsincos",
        "fnstsw ax",
        "test ax, 0x400",
        "jz 3f",
        "fldpi",
        "fadd st(0), st(0)",
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
        out("ax") _,out("st(0)") _,out("st(1)") _,options(nostack),
    );}
    (sine,cosine)
}
#[cfg(not(any(target_arch="x86",target_arch="x86_64")))]
pub fn engine_sin_cos(_x:f32)->(f32,f32) {
    panic!("bit-exact shipped Godot trigonometry requires the x86 x87 instruction set")
}
pub fn engine_sin(x:f32)->f32 {engine_sin_cos(x).0}
pub fn engine_cos(x:f32)->f32 {engine_sin_cos(x).1}
