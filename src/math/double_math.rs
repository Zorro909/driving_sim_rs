//! Double math from the shipped Proton 11 UCRT (musl/Arm, non-FMA path).
//! Copyright (c) 2018, Arm Limited. SPDX-License-Identifier: MIT
//! Arithmetic order, C round (ties away), and unsigned wrap are intentional.
use crate::math::double_math_tables::*;
const SIGN: u64 = 1 << 63;
const INF: u64 = 0x7ff0000000000000;
const NEG_NAN: f64 = f64::from_bits(0xfff8000000000000);
// Keep the compensated sum grouping visible; reassociation changes rounding.
#[allow(clippy::assign_op_pattern)]
fn special(tmp: f64, mut bits: u64, k: u64) -> f64 {
    if k & 0x80000000 == 0 {
        bits = bits.wrapping_sub(1009u64 << 52);
        let scale = f64::from_bits(bits);
        return f64::from_bits(2032u64 << 52) * (scale + scale * tmp);
    }
    bits = bits.wrapping_add(1022u64 << 52);
    let scale = f64::from_bits(bits);
    let mut y = scale + scale * tmp;
    if y.abs() < 1.0 {
        let one = if y < 0.0 { -1.0 } else { 1.0 };
        let mut lo = scale - y + scale * tmp;
        let hi = one + y;
        lo = one - hi + y + lo;
        y = (hi + lo) - one;
        if y == 0.0 {
            y = f64::from_bits(bits & SIGN);
        }
    }
    f64::MIN_POSITIVE * y
}
fn exp_inner(x: f64, xtail: f64, negative: bool) -> f64 {
    let exponent = (x.to_bits() >> 52) & 0x7ff;
    if exponent < 969 {
        return if negative { -(1.0 + x) } else { 1.0 + x };
    }
    if exponent >= 1033 {
        let y = if x.is_sign_negative() { 0.0 } else { f64::INFINITY };
        return if negative { -y } else { y };
    }
    let kd = (INV_LN2_N * x).round();
    let ki = kd as i64 as u64;
    let r = (x + kd * NEG_LN2_HI_N + kd * NEG_LN2_LO_N) + xtail;
    let idx = 2 * (ki as usize % 128);
    let bias = if negative { 0x800u64 << 7 } else { 0 };
    let top = ki.wrapping_add(bias) << 45;
    let tail = f64::from_bits(EXP_TAB[idx]);
    let bits = EXP_TAB[idx + 1].wrapping_add(top);
    let r2 = r * r;
    let c = EXP_POLY;
    let tmp = tail + r + r2 * (c[0] + r * c[1]) + r2 * r2 * (c[2] + r * c[3]);
    if exponent >= 1032 {
        return special(tmp, bits, ki);
    }
    let scale = f64::from_bits(bits);
    scale + scale * tmp
}
pub fn exp(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    exp_inner(x, 0.0, false)
}
fn pow_log(ix: u64) -> (f64, f64) {
    let tmp = ix.wrapping_sub(0x3fe6955500000000);
    let i = ((tmp >> 45) % 128) as usize;
    let kd = ((tmp as i64) >> 52) as f64;
    let iz = ix.wrapping_sub(tmp & (0xfffu64 << 52));
    let z = f64::from_bits(iz);
    let [invc, logc, logctail] = POW_TAB[i];
    let zhi = f64::from_bits(iz.wrapping_add(1 << 31) & (u64::MAX << 32));
    let zlo = z - zhi;
    let rhi = zhi * invc - 1.0;
    let rlo = zlo * invc;
    let r = rhi + rlo;
    let t1 = kd * LN2_HI + logc;
    let t2 = t1 + r;
    let lo1 = kd * LN2_LO + logctail;
    let lo2 = t1 - t2 + r;
    let a = POW_POLY;
    let ar = a[0] * r;
    let ar2 = r * ar;
    let ar3 = r * ar2;
    let arhi = a[0] * rhi;
    let arhi2 = rhi * arhi;
    let hi = t2 + arhi2;
    let lo3 = rlo * (ar + arhi);
    let lo4 = t2 - hi + arhi2;
    let p = ar3 * (a[1] + r * a[2] + ar2 * (a[3] + r * a[4] + ar2 * (a[5] + r * a[6])));
    let lo = lo1 + lo2 + lo3 + lo4 + p;
    let y = hi + lo;
    (y, hi - y + lo)
}
fn integer_kind(bits: u64) -> u32 {
    let e = (bits >> 52) & 0x7ff;
    if e < 1023 {
        return 0;
    }
    if e > 1075 {
        return 2;
    }
    let bit = 1u64 << (1075 - e);
    if bits & (bit - 1) != 0 {
        0
    } else if bits & bit != 0 {
        1
    } else {
        2
    }
}
pub fn pow(x: f64, y: f64) -> f64 {
    let mut ix = x.to_bits();
    let iy = y.to_bits();
    let mut topx = (ix >> 52) as u32;
    let topy = (iy >> 52) as u32;
    let mut negative = false;
    if topx.wrapping_sub(1) >= 0x7fe || (topy & 0x7ff).wrapping_sub(0x3be) >= 0x80 {
        if y == 0.0 || !y.is_finite() {
            if y == 0.0 {
                return 1.0;
            }
            if x == 1.0 {
                return 1.0;
            }
            if x.is_nan() || y.is_nan() {
                return x + y;
            }
            if x.abs() == 1.0 {
                return 1.0;
            }
            if (x.abs() < 1.0) != y.is_sign_negative() {
                return 0.0;
            }
            return y * y;
        }
        if x == 0.0 || !x.is_finite() {
            let mut x2 = x * x;
            if x.is_sign_negative() && integer_kind(iy) == 1 {
                x2 = -x2;
            }
            return if y.is_sign_negative() { 1.0 / x2 } else { x2 };
        }
        if x.is_sign_negative() {
            let kind = integer_kind(iy);
            if kind == 0 {
                return NEG_NAN;
            }
            negative = kind == 1;
            ix &= !SIGN;
            topx &= 0x7ff;
        }
        if (topy & 0x7ff).wrapping_sub(0x3be) >= 0x80 {
            if ix == 1.0f64.to_bits() {
                return 1.0;
            }
            if topy & 0x7ff < 0x3be {
                return if ix > 1.0f64.to_bits() { 1.0 + y } else { 1.0 - y };
            }
            return if (ix > 1.0f64.to_bits()) == (topy < 0x800) {
                f64::INFINITY
            } else {
                0.0
            };
        }
        if topx == 0 {
            ix = ((x * 4503599627370496.0).to_bits() & !SIGN).wrapping_sub(52u64 << 52);
        }
    }
    let (hi, lo) = pow_log(ix);
    let yhi = f64::from_bits(iy & (u64::MAX << 27));
    let ylo = y - yhi;
    let lhi = f64::from_bits(hi.to_bits() & (u64::MAX << 27));
    let llo = hi - lhi + lo;
    let ehi = yhi * lhi;
    let elo = ylo * lhi + y * llo;
    exp_inner(ehi, elo, negative)
}
pub fn log(x: f64) -> f64 {
    let mut ix = x.to_bits();
    if ix.wrapping_sub(0.9375f64.to_bits()) < (1.064697265625f64.to_bits() - 0.9375f64.to_bits()) {
        if x == 1.0 {
            return 0.0;
        }
        let r = x - 1.0;
        let r2 = r * r;
        let r3 = r * r2;
        let b = LOG_NEAR_POLY;
        let mut y = r3
            * (b[1]
                + r * b[2]
                + r2 * b[3]
                + r3 * (b[4] + r * b[5] + r2 * b[6] + r3 * (b[7] + r * b[8] + r2 * b[9] + r3 * b[10])));
        let w = r * 134217728.0;
        let rhi = r + w - w;
        let rlo = r - rhi;
        let w = rhi * rhi * b[0];
        let hi = r + w;
        let mut lo = r - hi + w;
        lo += b[0] * rlo * (rhi + r);
        y += lo;
        y += hi;
        return y;
    }
    let top = (ix >> 48) as u32;
    if top.wrapping_sub(0x10) >= 0x7fe0 {
        if x == 0.0 {
            return f64::NEG_INFINITY;
        }
        if ix == INF || x.is_nan() {
            return x;
        }
        if x.is_sign_negative() {
            return NEG_NAN;
        }
        ix = (x * 4503599627370496.0).to_bits().wrapping_sub(52u64 << 52);
    }
    let tmp = ix.wrapping_sub(0x3fe6000000000000);
    let i = ((tmp >> 45) % 128) as usize;
    let kd = ((tmp as i64) >> 52) as f64;
    let iz = ix.wrapping_sub(tmp & (0xfffu64 << 52));
    let [invc, logc] = LOG_TAB[i];
    let z = f64::from_bits(iz);
    let r = (z - LOG_TAB2[i][0] - LOG_TAB2[i][1]) * invc;
    let w = kd * LN2_HI + logc;
    let hi = w + r;
    let lo = w - hi + r + kd * LN2_LO;
    let r2 = r * r;
    let a = LOG_POLY;
    lo + r2 * a[0] + r * r2 * (a[1] + r * a[2] + r2 * (a[3] + r * a[4])) + hi
}
