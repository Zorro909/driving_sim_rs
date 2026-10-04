//! Arithmetic that reproduces CPython's float semantics bit for bit.
//!
//! The reference arithmetic follows CPython, so each helper here
//! mirrors a specific CPython behaviour: `math.hypot` has its own correctly
//! rounded algorithm, `x ** 2` calls libm `pow` (not `x * x`), `sum()` uses
//! Neumaier compensation since 3.12, `%` is floored, and `min`/`max` keep the
//! first argument on ties.

use std::hint::black_box;

extern "C" {
    fn pow(x: f64, y: f64) -> f64;
    fn remainder(x: f64, y: f64) -> f64;
}

/// `math.remainder(x, y)`; the IEEE remainder is exact, so libm matches CPython.
#[inline]
pub fn py_remainder(x: f64, y: f64) -> f64 {
    unsafe { remainder(x, y) }
}

/// Round to float32 and back, like `ctypes.c_float(value).value`.
#[inline(always)]
pub(crate) fn f32r(x: f64) -> f64 {
    x as f32 as f64
}

/// Python `x ** y` for floats. `black_box` stops LLVM from turning `pow(x, 2)`
/// into `x * x`, which differs from glibc `pow` in the last bit.
#[inline]
pub fn py_pow(x: f64, y: f64) -> f64 {
    unsafe { pow(x, black_box(y)) }
}

/// Python `max(a, b)`: the first argument wins unless `b > a`.
#[inline(always)]
pub fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python `min(a, b)`: the first argument wins unless `b < a`.
#[inline(always)]
pub fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// Clamp as `min(high, max(low, value))`, preserving comparison order.
#[inline(always)]
pub(crate) fn clamp(value: f64, low: f64, high: f64) -> f64 {
    py_min(high, py_max(low, value))
}

/// Python float `a % b` (CPython `float_rem`).
#[inline]
pub(crate) fn py_mod(a: f64, b: f64) -> f64 {
    let m = a % b;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m + b
        } else {
            m
        }
    } else {
        0.0f64.copysign(b)
    }
}

/// Python `sum(iterable)` over floats: Neumaier compensated summation.
#[inline]
pub fn py_sum<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let mut sum = NeumaierSum::default();
    for value in values {
        sum.add(value);
    }
    sum.value()
}

#[derive(Clone, Copy, Default)]
pub(crate) struct NeumaierSum {
    total: f64,
    compensation: f64,
}

impl NeumaierSum {
    #[inline(always)]
    pub(crate) fn add(&mut self, x: f64) {
        let t = self.total + x;
        if self.total.abs() >= x.abs() {
            self.compensation += (self.total - t) + x;
        } else {
            self.compensation += (x - t) + self.total;
        }
        self.total = t;
    }

    #[inline(always)]
    pub(crate) fn value(self) -> f64 {
        if self.compensation != 0.0 && self.compensation.is_finite() {
            self.total + self.compensation
        } else {
            self.total
        }
    }
}

#[inline(always)]
fn dl_mul(x: f64, y: f64) -> (f64, f64) {
    let z = x * y;
    (z, x.mul_add(y, -z))
}

#[inline(always)]
fn dl_fast_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    (x, (a - x) + b)
}

/// Exponent returned by C `frexp`.
fn frexp_exponent(x: f64) -> i32 {
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if biased == 0 {
        frexp_exponent(x * f64::from_bits(0x4350_0000_0000_0000)) - 54 // 2**54
    } else {
        biased - 1022
    }
}

/// C `ldexp(1.0, exponent)` for the range `vector_norm` needs.
fn power_of_two(exponent: i32) -> f64 {
    if exponent >= -1022 {
        f64::from_bits(((exponent + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (exponent + 1074))
    }
}

fn vector_norm2(a: f64, b: f64, max: f64, found_nan: bool) -> f64 {
    if max.is_infinite() {
        return max;
    }
    if found_nan {
        return f64::NAN;
    }
    if max == 0.0 {
        return max;
    }
    let max_e = frexp_exponent(max);
    if max_e < -1023 {
        let m = f64::MIN_POSITIVE;
        return m * vector_norm2(a / m, b / m, max / m, found_nan);
    }
    let scale = power_of_two(-max_e);
    let (mut csum, mut frac1, mut frac2) = (1.0f64, 0.0f64, 0.0f64);
    for x in [a, b] {
        let x = x * scale;
        let (hi, lo) = dl_mul(x, x);
        let (sum_hi, sum_lo) = dl_fast_sum(csum, hi);
        csum = sum_hi;
        frac1 += lo;
        frac2 += sum_lo;
    }
    let mut h = (csum - 1.0 + (frac1 + frac2)).sqrt();
    let (hi, lo) = dl_mul(-h, h);
    let (sum_hi, sum_lo) = dl_fast_sum(csum, hi);
    csum = sum_hi;
    frac1 += lo;
    frac2 += sum_lo;
    let x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    h / scale
}

/// CPython `math.hypot(x, y)` (`vector_norm` in Modules/mathmodule.c).
#[inline]
pub(crate) fn hypot(x: f64, y: f64) -> f64 {
    let (ax, ay) = (x.abs(), y.abs());
    let found_nan = ax.is_nan() || ay.is_nan();
    let mut max = 0.0;
    if ax > max {
        max = ax;
    }
    if ay > max {
        max = ay;
    }
    vector_norm2(ax, ay, max, found_nan)
}

/// Python `math.floor(x)` for track tile coordinates.
#[inline(always)]
pub(crate) fn floor_i64(x: f64) -> i64 {
    x.floor() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floored_modulo_matches_python() {
        assert_eq!(py_mod(-1.0, 5.0), 4.0);
        assert_eq!(py_mod(1.0, -5.0), -4.0);
        assert_eq!(py_mod(7.5, 5.0), 2.5);
        assert!(py_mod(0.0, -5.0).is_sign_negative());
    }

    #[test]
    fn hypot_basic_values() {
        assert_eq!(hypot(3.0, 4.0), 5.0);
        assert_eq!(hypot(0.0, 0.0), 0.0);
        assert_eq!(hypot(-3.0, 0.0), 3.0);
        assert_eq!(hypot(f64::INFINITY, f64::NAN), f64::INFINITY);
        assert!(hypot(1e-310, 1e-310) > 0.0);
    }

    #[test]
    fn compensated_sum_recovers_cancelled_terms() {
        assert_eq!(py_sum([1e16, 1.0, -1e16]), 1.0);
        assert_eq!(py_sum([0.1; 10]), 1.0);
    }
}
