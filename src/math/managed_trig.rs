//! Scalar transcription of the shipped .NET SSE4.1 large packed-sine lane.
//! Reference coreclr.dll RVAs 16b82b..16bd86; tables at 4de300 and 4def00.
use super::managed_sine_tables as tables;
use std::num::Wrapping as W;
fn f(bits: u32) -> f32 {
    f32::from_bits(bits)
}

pub(crate) fn sine_large(value: f32) -> f32 {
    if !value.is_finite() {
        return super::native_math::nonfinite_trig_result(value);
    }
    let bits = value.to_bits();
    let sign = bits & 0x80000000;
    let exponent = ((bits >> 23) & 255) as usize;
    let m = (bits & 0x007fffff) + 0x00800000;
    let [t0, t1, t2] = tables::EXPONENTS[exponent];
    let (l, h) = (W(m & 65535), W(m >> 16));
    let (a, b, c, d, e, g) = (
        W(t2 & 65535),
        W(t2 >> 16),
        W(t1 & 65535),
        W(t1 >> 16),
        W(t0 & 65535),
        W(t0 >> 16),
    );
    let mask = W(65535);
    let w = ((l * c) & mask) + h * b + ((l * b) >> 16) + ((h * a) >> 16);
    let v = (w >> 16) + ((l * d) & mask) + h * c + ((l * c) >> 16);
    let u = (v >> 16) + ((l * e) & mask) + h * d + ((l * d) >> 16);
    let q = (u >> 16) + ((l * g) & mask) + h * e + ((l * e) >> 16);
    let hi = ((q << 16) + (u & mask)).0;
    let lo = ((v << 16) + (w & mask)).0;
    let a0 = f((hi >> 9) | (0x3f800000 ^ sign));
    let rounded = a0 + f(0x47400000);
    let coarse = rounded - f(0x47400000);
    let r0 = a0 - coarse;
    let mut r1 = f(((hi & 511) << 14) | (lo >> 18) | (0x34000000 ^ sign)) - f(0x34000000 ^ sign);
    let mut r2 = f(((lo & 0x3ffff) << 5) | (0x28800000 ^ sign)) - f(0x28800000 ^ sign);
    let r = r0 + r1;
    let rh0 = f(r.to_bits() & 0xfffff000);
    let rl0 = r - rh0;
    r1 += r0 - r;
    r2 += r1;
    let main = rh0 * f(0x40c91000);
    let mut tail = rl0 * f(0x40c91000);
    let mut small = r2 * f(0x40c90fdb);
    tail += rh0 * f(0xb795777a);
    small += rl0 * f(0xb795777a);
    tail += small;
    let rh = tail + main;
    let rl = tail + (main - rh);
    let [cd, sh, sl, ch] = tables::TRIG[(rounded.to_bits() & 255) as usize].map(f);
    let z = rh * rh;
    let cp = f(0x3d2aaa7c) * z + f(0xbf000000);
    let mut sp = f(0x3c08885c) * z + f(0xbe2aaaab);
    sp *= z;
    sp *= rh;
    let term = rh * ch;
    let s0 = sh + term;
    let err0 = (sh - s0) + term;
    let term = rh * cd;
    let s1 = s0 + term;
    let err1 = (s0 - s1) + term;
    let mut err = err0 + err1;
    let adjusted_cos = (cd + ch) - rh * sh;
    err += sp * adjusted_cos;
    let mut tail = sl + rl * adjusted_cos;
    tail += sh * (z * cp);
    tail += err;
    s1 + tail
}
