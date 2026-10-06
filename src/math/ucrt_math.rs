//! Float math of the Windows UCRT (ucrtbase.dll), FMA3 code paths: sinf, cosf, atan2f, exp, tanh
//! and pow, for the game running natively on Windows. The Proton build of the game uses the musl
//! versions in `native_math` and `double_math`; [`crate::math::libm`] selects between them.
//!
//! Every function mirrors the DLL operation for operation. Fused multiply-adds are explicit
//! (`mul_add`) and everything else rounds separately, which Rust never contracts. `mul_add`
//! is exact on CPUs without FMA too, only slower.
//!
//! The ported domains are sinf |x| < 16779436 and cosf |x| < 3373259520. Outside them the DLL takes slow paths that are not ported:
//! a Windows build calls the system function, other hosts return NaN.

// The DLL's literals are kept exactly as extracted.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

use crate::math::ucrt_tables::*;

#[inline(always)]
fn f(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}
#[inline(always)]
fn d(bits: u64) -> f64 {
    f64::from_bits(bits)
}
#[inline(always)]
fn b(x: f64) -> u64 {
    x.to_bits()
}
#[inline(always)]
fn tab(table: &[u64], i: i32) -> f64 {
    f64::from_bits(table[i as usize])
}

const QNAN32: u32 = 0x7FC00000;
const INDEFINITE32: u32 = 0xFFC00000;
const INDEFINITE64: u64 = 0xFFF8000000000000;
const QNAN64: u64 = 0x7FF8000000000000;
const INF64: u64 = 0x7FF0000000000000;
const SIGN64: u64 = 0x8000000000000000;
const ABS64: u64 = 0x7FFFFFFFFFFFFFFF;

const S1: f64 = -0.16666666666666666;
const S2: f64 = 0.008333333333333333;
const S3: f64 = -0.0001984126984126984;
const S4: f64 = 2.7557319223985893E-06;
const C1: f64 = -0.5;
const C2: f64 = 0.041666666666666664;
const C3: f64 = -0.0013888888888888887;
const C4: f64 = 2.4801587301587298E-05;
const C5: f64 = -2.755731922398589E-07;

/// sinf/cosf of an infinity or NaN: the indefinite NaN, or the input NaN made quiet.
fn non_finite_trig(bits: u32) -> u32 {
    if bits & 0x007FFFFF != 0 {
        bits | 0x00400000
    } else {
        INDEFINITE32
    }
}

fn sin_kernel(r: f64) -> f64 {
    let x2 = r * r;
    let mut p = f(x2, S4, S3);
    p = f(x2, p, S2);
    p = f(x2, p, S1);
    f(p, r * x2, r)
}

fn cos_kernel_sinf(r: f64) -> f64 {
    let x2 = r * r;
    let a = f(x2, C1, 1.0);
    let mut p = f(x2, C5, C4);
    p = f(x2, p, C3);
    p = f(x2, p, C2);
    f(p, x2 * x2, a)
}

fn cos_kernel_cosf(r: f64) -> f64 {
    let x2 = r * r;
    let a = 1.0 - x2 * 0.5;
    let mut p = f(x2, C5, C4);
    p = f(x2, p, C3);
    p = f(x2, p, C2);
    f(p, x2 * x2, a)
}

/// Whether `sinf` is ported for `x` (finite or not; non-finite inputs are handled).
pub fn sinf_in_domain(x: f32) -> bool {
    b(x as f64) & ABS64 < 0x4170008AC0000000
}

/// Whether `cosf` is ported for `x`.
pub fn cosf_in_domain(x: f32) -> bool {
    b(x as f64) & ABS64 < 0x41E921FB60000000
}

fn sinf_ported(xf: f32) -> f32 {
    let ib = xf.to_bits();
    if ib & 0x7F800000 == 0x7F800000 {
        return f32::from_bits(non_finite_trig(ib));
    }
    let x = xf as f64;
    let ab = b(x) & ABS64;
    if ab <= 0x3FE921FB54442D18 {
        if ab >= 0x3F80000000000000 {
            return sin_kernel(x) as f32;
        }
        if ab >= 0x3F20000000000000 {
            let t = x * x * x;
            return f(-t, 0.16666666666666666, x) as f32;
        }
        return xf;
    }
    if ab >= 0x4170008AC0000000 {
        return f32::from_bits(QNAN32);
    }
    let ax = d(ab);
    let t0 = f(ax, 0.6366197723675814, 0.5);
    let n = t0 as i32;
    let q = n & 3;
    let nd = n as f64;
    let r1 = f(-nd, 1.5707963267341256, ax);
    let r = r1 - nd * 6.077100506506192E-11;
    let mut res = if q & 1 == 0 { sin_kernel(r) } else { cos_kernel_sinf(r) };
    let flip = (q == 0 || q == 1) ^ (ib >> 31 == 0);
    if flip {
        res = -res;
    }
    res as f32
}

fn cosf_ported(xf: f32) -> f32 {
    let ib = xf.to_bits();
    if ib & 0x7F800000 == 0x7F800000 {
        return f32::from_bits(non_finite_trig(ib));
    }
    let x = xf as f64;
    let ab = b(x) & ABS64;
    if ab <= 0x3FE921FB54442D18 {
        if ab >= 0x3F80000000000000 {
            return cos_kernel_cosf(x) as f32;
        }
        if ab >= 0x3F20000000000000 {
            return f(-x, x * 0.5, 1.0) as f32;
        }
        return 1.0;
    }
    if ab >= 0x41E921FB60000000 {
        return f32::from_bits(QNAN32);
    }
    let ax = d(ab);
    let t0 = f(ax, 0.6366197723675814, 0.5);
    let n = t0 as i32;
    let q = n & 3;
    let nd = n as f64;
    let r1 = f(-nd, 1.5707963267341256, ax);
    let r = r1 - nd * 6.077100506506192E-11;
    let mut res = if q & 1 != 0 { sin_kernel(r) } else { cos_kernel_cosf(r) };
    if ((q + 1) >> 1) & 1 != 0 {
        res = -res;
    }
    res as f32
}

pub fn sinf(x: f32) -> f32 {
    if sinf_in_domain(x) || !x.is_finite() {
        return sinf_ported(x);
    }
    system_sinf(x)
}

pub fn cosf(x: f32) -> f32 {
    if cosf_in_domain(x) || !x.is_finite() {
        return cosf_ported(x);
    }
    system_cosf(x)
}

pub fn atan2f(yf: f32, xf: f32) -> f32 {
    let x7 = xf as f64;
    let y6 = yf as f64;
    let (xb, yb) = (b(x7), b(y6));
    let exp_x = ((xb >> 52) & 0x7FF) as i32;
    let exp_y = ((yb >> 52) & 0x7FF) as i32;
    let esi = exp_y - exp_x;
    let (ax, ay) = (xb & ABS64, yb & ABS64);
    let xneg = (xb as i64) < 0;
    let yneg = (yb as i64) < 0;
    if ax > INF64 || ay > INF64 {
        let nan = if ax > INF64 { xf } else { yf };
        return f32::from_bits(nan.to_bits() | 0x00400000);
    }
    if ay == 0 {
        if xneg {
            return if yneg { -3.1415927 } else { 3.1415927 };
        }
        return y6 as f32;
    }
    if ax == 0 && yneg {
        return -1.5707964;
    }
    if esi > 0x1A {
        return if yneg { -1.5707964 } else { 1.5707964 };
    }
    if esi < -13 && !xneg {
        if esi < -150 {
            return if yneg { -0.0 } else { 0.0 };
        }
        if esi < -126 {
            let v = d(0x4630000000000000) * y6 / x7;
            let vb = b(v);
            let sign = vb & SIGN64;
            let mag = vb & ABS64;
            let e = (mag >> 52) as u32;
            let mut rbx: u64;
            if e > 0x64 {
                rbx = (((e - 0x64) as u64) << 52) | (mag & 0x800FFFFFFFFFFFFF);
            } else {
                let mut m = (mag & 0x801FFFFFFFFFFFFF) | 0x10000000000000;
                let mut cnt = 0x65 - e as i32;
                if cnt > 0x36 {
                    rbx = 0;
                } else {
                    cnt -= 1;
                    m >>= cnt;
                    let carry = m & 1;
                    rbx = (m >> 1) + carry;
                }
            }
            rbx |= sign;
            return d(rbx) as f32;
        }
        return (y6 / x7) as f32;
    }
    if esi < -26 && xneg {
        return if yneg { -3.1415927 } else { 3.1415927 };
    }
    if ay == INF64 && ax == INF64 {
        if xneg {
            return if yneg { -2.3561945 } else { 2.3561945 };
        }
        return if yneg { -0.7853982 } else { 0.7853982 };
    }
    let sx = if xneg { -x7 } else { x7 };
    let sy = if yneg { -y6 } else { y6 };
    let swap = sy > sx;
    let small = if swap { sx } else { sy };
    let large = if swap { sy } else { sx };
    let t = small / large;
    let mut res;
    if t > 0.0625 {
        let s256 = small * 256.0;
        let tt = t * 256.0 + 0.5;
        let n = tt as i32;
        let nd = n as f64;
        let a = nd * small;
        let bb = nd * large;
        let l256 = large * 256.0;
        let num = s256 - bb;
        let den = a + l256;
        let u = num / den;
        let hi = u + tab(&ATAN2F_ATAN256, n - 16);
        let mut u3 = u * u;
        u3 *= u;
        u3 *= 0.33333333333224097;
        res = hi - u3;
    } else if 0.0001 > t {
        res = t;
    } else {
        let t2 = t * t;
        let mut p = 0.19999999999393223 - t2 * 0.1428571356180717;
        p *= t2;
        let t3 = t2 * t;
        let mut q = 0.3333333333333317 - p;
        q *= t3;
        res = t - q;
    }
    if swap {
        res = 1.5707963267948966 - res;
    }
    if xneg {
        res = 3.141592653589793 - res;
    }
    if yneg {
        res = -res;
    }
    res as f32
}

pub fn exp(x: f64) -> f64 {
    let ab = b(x) & ABS64;
    if ab >= INF64 {
        if b(x) == INF64 {
            return x;
        }
        if b(x) == 0xFFF0000000000000 {
            return 0.0;
        }
        return d(b(x) | 0x0008000000000000);
    }
    if x > 709.782712893384 {
        return d(INF64);
    }
    if x < -744.0346068132731 {
        return 0.0;
    }
    if ab <= 0x3E50000000000000 {
        return x + 1.0;
    }
    let x1 = x * 92.33248261689366;
    let nd = x1.trunc();
    let nn = x1 as i32;
    let r1 = f(nd, -0.010830424696223417, x);
    let r2 = nd * -2.5728046223276688E-14;
    let r = r2 + r1;
    let j = nn & 0x3F;
    let k = nn >> 6;
    let mut p = f(0.001388888888888889, r, 0.008333333333333333);
    p = f(p, r, 0.041666666666666664);
    p = f(p, r, 0.16666666666666666);
    p = f(p, r, 0.5);
    let rr = r * r;
    let e = f(rr, p, r);
    let mut t = e * tab(&EXP_MUL, j);
    t += tab(&EXP_LO, j);
    let res = t + tab(&EXP_HI, j);
    if k > -1022 || (k == -1022 && res >= 1.0) {
        return d(b(res).wrapping_add(((k as i64) as u64) << 52));
    }
    let cl = (k + 0x432) & 63;
    res * d(1u64 << cl)
}

pub fn tanh(x: f64) -> f64 {
    let xb = b(x);
    let ab = xb & ABS64;
    if ab < 0x3E30000000000000 {
        return x;
    }
    if ab > INF64 {
        return d(xb | 0x0008000000000000);
    }
    let neg = ab != xb;
    let a = if neg { -x } else { x };
    if a > 20.0 {
        return if neg { -1.0 } else { 1.0 };
    }
    let large = 1.0 < a;
    let (num, den);
    if large {
        let a2 = a + a;
        let mut c = a2 * 46.16624130844683;
        c = if c > 0.0 { c + 0.5 } else { c - 0.5 };
        let nn = c as i32;
        let j = nn & 0x1F;
        let nd = nn as f64;
        let hi = tab(&TANH_HI, j);
        let lo = tab(&TANH_LO, j);
        let e = nn - j;
        let k = (e + ((e >> 31) & 0x1F)) >> 5;
        let r6 = (a2 - nd * 0.021660849335603416) * 1.0;
        let r1 = ((-nn) as f64) * 5.689487495325456E-11 * 1.0;
        let r = r1 + r6;
        let mut p = r * 0.001388894908637772 + 0.008333367984342196;
        p = p * r + 0.04166666666622608;
        p = p * r + 0.16666666666526087;
        p *= r;
        let rr = r * r;
        p += 0.5;
        p *= rr;
        let hl = lo + hi;
        p += r1;
        p += r6;
        p *= hl;
        p += lo;
        let k1 = (k + if k < 0 { 1 } else { 0 }) >> 1;
        let k2 = k - k1;
        let s1 = d((((k1 + 0x3FF) as i64) as u64) << 52);
        let s2 = d((((k2 + 0x3FF) as i64) as u64) << 52);
        let mut t = (hi + p) * s1;
        t *= s2;
        t += 1.0;
        num = 2.0;
        den = t;
    } else {
        let x2 = a * a;
        let x3 = x2 * a;
        let up = 0.9 <= a;
        let sel = |lo: f64, hi: f64| if up { hi } else { lo };
        let k0 = sel(-0.0002000476210719095, -0.00016559704390354995);
        let k1 = sel(1.4207792637883471E-08, 1.154758789961434E-08);
        let k2 = sel(0.00020911402625291644, 0.00017307605012622596);
        let k3 = sel(0.020156216602693764, 0.016735877546189656);
        let k4 = sel(0.017601634900304468, 0.014617304728873168);
        let k5 = sel(0.3816414142883289, 0.3172045589772944);
        let k6 = sel(0.27403042465617977, 0.2277938706590883);
        let k7 = sel(0.8220912739685393, 0.6833816119772959);
        let mut p = k0 - x2 * k1;
        let mut q = x2 * k2;
        p *= x2;
        q += k3;
        p -= k4;
        q *= x2;
        p *= x2;
        q += k5;
        p -= k6;
        q *= x2;
        q += k7;
        p *= x3;
        num = p;
        den = q;
    }
    let dd = num / den;
    let res = if large { 1.0 - dd } else { dd + a };
    if neg {
        -res
    } else {
        res
    }
}

/// `pow` for finite x > 0 and finite y != 0; the DLL's code for |y| up to 2^63.
fn pow_core(x: f64, y: f64) -> f64 {
    let (xb, yb) = (b(x), b(y));
    let nan = d(QNAN64);
    if yb & ABS64 == 0 {
        return 1.0;
    }
    if yb == 0x3FF0000000000000 {
        return x;
    }
    if xb & SIGN64 != 0 {
        return nan;
    }
    if xb == 0x3FF0000000000000 {
        return 1.0;
    }
    if xb == 0 {
        return if yb & SIGN64 != 0 { d(INF64) } else { 0.0 };
    }
    if xb & 0x7FF0000000000000 == 0x7FF0000000000000 {
        return nan;
    }
    let yab = (yb & 0x7FF0000000000000) as i64;
    if yab > 0x43E0000000000000 {
        return nan;
    }
    if yab < 0x3C00000000000000 {
        return y + 1.0;
    }

    let mut x8 = xb;
    let mut e_int = (xb >> 52) as i32 - 0x3FF;
    let mut mant = xb & 0x000FFFFFFFFFFFFF;
    if e_int == -1023 {
        let m2 = d(mant | 0x3FF0000000000000) - 1.0;
        let mb = b(m2);
        mant = mb & 0x000FFFFFFFFFFFFF;
        e_int = (mb >> 52) as i32 - 0x7FD;
        x8 = mb;
    }
    let e6 = e_int as f64;
    let mut r8 = x8 & 0x000FF00000000000;
    let r9 = (x8 & 0x0000080000000000) << 1;
    r8 += r9;
    let c = d(r8 | 0x3FE0000000000000);
    let j = (r8 >> 44) as i32;
    let m = d(mant | 0x3FE0000000000000);
    let d4 = c - m;
    let hiv = d4 * tab(&POW_T1, j);
    let lov = d4 * tab(&POW_T2, j);
    let z = hiv + lov;
    let mut err = hiv - z;
    let zz = z * z;
    err += lov;
    let mut pp = f(0.1428571428571429, z, 0.16666666666666666);
    pp = f(pp, z, 0.2);
    pp = f(pp, z, 0.25);
    pp = f(pp, z, 0.3333333333333333);
    pp = f(pp, z, 0.5);
    pp = f(pp, zz, err);
    let s5 = f(e6, 5.7699990475432854E-08, -pp);
    let t0 = tab(&POW_T3, j);
    let mut s3 = s5 + tab(&POW_T4, j);
    let mut s1 = s3;
    s3 -= z;
    let mut lg = f(e6, 0.6931471228599548, t0);
    let mut s7 = lg;
    lg += s3;
    let lg_full = lg;
    let lg_hi = d(b(lg) & 0xFFFFFFFFF8000000);
    let zs = z + s3;
    s7 -= lg_full;
    s1 -= zs;
    s7 += s3;
    let s5b = lg_full - lg_hi;
    s7 += s1;
    s7 += s5b;
    let y_hi = d(yb & 0xFFFFFFFFF8000000);
    let y_lo = y - y_hi;
    let mut a3 = y_lo * s7;
    let a4 = y_lo * lg_hi;
    let a5 = s7 * y_hi;
    let a6 = lg_hi * y_hi;
    a3 += a4;
    a3 += a5;
    let prod_hi = a6 + a3;
    let mut prod_lo = a6 - prod_hi;
    prod_lo += a3;
    let w = prod_hi * 92.33248261689366;
    if w > 65536.0 {
        return d(INF64);
    }
    if w < -68800.0 {
        return 0.0;
    }
    let nn = w.round_ties_even() as i32;
    let nd = nn as f64;
    let r0 = f(-nd, 0.010830424260348082, prod_hi);
    let jj = nn & 0x3F;
    let k = (nn - jj) >> 6;
    let lo2 = nd * -4.359010638708991E-10;
    let mut rr2 = r0 + lo2;
    rr2 += prod_lo;
    let mut q = f(0.001388888888888889, rr2, 0.008333333333333333);
    q = f(q, rr2, 0.041666666666666664);
    q = f(q, rr2, 0.16666666666666666);
    q = f(q, rr2, 0.5);
    q = f(q, rr2, 1.0);
    q *= rr2;
    let mut u5 = q * tab(&EXP_LO, jj);
    let mut u1 = q * tab(&EXP_HI, jj);
    u5 += tab(&EXP_LO, jj);
    u1 += u5;
    u1 += tab(&EXP_HI, jj);
    let scale_bits = (((k + 0x3FF) as i64) as u64) << 52;
    if k <= -1022 {
        if k == -1022 && u1 >= 1.0 {
            return u1 * d(scale_bits);
        }
        if (b(prod_hi) as i64) > 0xC0874046DFEFD9D0u64 as i64 {
            return d(1);
        }
        let sh = (k + 0x432).max(0);
        return u1 * d(1u64 << (sh & 63));
    }
    if k + 0x3FF == 0x7FF {
        if u1 >= 1.0 {
            return d(INF64);
        }
        return d(b(u1) | 0x7FE0000000000000);
    }
    u1 * d(scale_bits)
}

/// The NaN `pow` returns for NaN operand(s): the only NaN, or for two the first unless its sign is
/// set and the second's is not.
fn nan_operand(xb: u64, yb: u64) -> u64 {
    let is_nan = |bits: u64| bits & ABS64 > INF64;
    if is_nan(xb) && (!is_nan(yb) || xb & SIGN64 == 0 || yb & SIGN64 != 0) {
        xb
    } else {
        yb
    }
}

/// 0 not an integer, 1 an odd integer, 2 an even integer.
fn integer_kind(bits: u64) -> u32 {
    let e = (bits >> 52) & 0x7FF;
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
    let (xb, yb) = (b(x), b(y));
    // A signalling NaN is invalid even where C99 gives 1.
    let quiet = 0x0008000000000000;
    let signalling = |v: f64| v.is_nan() && b(v) & quiet == 0;
    if signalling(x) || signalling(y) {
        return d(nan_operand(xb, yb) | quiet);
    }
    if yb & ABS64 == 0 || xb == 0x3FF0000000000000 {
        return 1.0;
    }
    if x.is_nan() || y.is_nan() {
        return d(nan_operand(xb, yb) | quiet);
    }
    let xneg = xb & SIGN64 != 0;
    let ax = x.abs();
    if ax == 1.0 {
        // pow(-1, y) is 1 for infinite y and for every integer y beyond the odd/even range.
        if y.is_infinite() || integer_kind(yb) == 2 {
            return 1.0;
        }
    }
    if y.is_infinite() {
        return if (ax < 1.0) == (y > 0.0) { 0.0 } else { d(INF64) };
    }
    let kind = integer_kind(yb);
    let negate = xneg && kind == 1;
    let signed = |v: f64| if negate { -v } else { v };
    if ax == 0.0 {
        return signed(if y < 0.0 { d(INF64) } else { 0.0 });
    }
    if ax.is_infinite() {
        return signed(if y < 0.0 { 0.0 } else { d(INF64) });
    }
    if xneg && kind == 0 {
        return d(INDEFINITE64);
    }
    if (yb & 0x7FF0000000000000) as i64 > 0x43E0000000000000 {
        // |y| beyond 2^63 is an even integer.
        return if (ax < 1.0) == (y > 0.0) { 0.0 } else { d(INF64) };
    }
    signed(pow_core(ax, y))
}

#[cfg(all(windows, target_env = "msvc"))]
mod system {
    pub fn sinf(x: f32) -> f32 {
        x.sin()
    }
    pub fn cosf(x: f32) -> f32 {
        x.cos()
    }
}

// Other hosts have no ucrt: the unported domains return NaN.
#[cfg(not(all(windows, target_env = "msvc")))]
mod system {
    pub fn sinf(_: f32) -> f32 {
        f32::NAN
    }
    pub fn cosf(_: f32) -> f32 {
        f32::NAN
    }
}

use system::{cosf as system_cosf, sinf as system_sinf};

/// The UCRT's `log`. Not ported: this is the system's, so only a Windows MSVC build reproduces it
/// ([`crate::math::set_libm`] refuses the Windows flavour elsewhere).
pub fn log(x: f64) -> f64 {
    x.ln()
}
