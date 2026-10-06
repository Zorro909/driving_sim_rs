//! AVX2 `Network::forward_into`, bit-identical to the scalar loop.
//!
//! Each register holds four output neurons. Every lane adds its products in
//! input order, starting from +0.0, with separate multiplies and adds (no
//! fused multiply-add), as the scalar loop does. `tanh4` evaluates every
//! `game_tanh`/`game_expm1` branch lane-wise with the scalar operations and
//! keeps the result of the branch the scalar code takes. It is musl's tanh,
//! so the Windows math profiles apply their scalar tanh to the sums instead.
// SIMD coefficients must retain the scalar runtime approximation bits.
#![allow(clippy::approx_constant, clippy::excessive_precision)]

use crate::math::profile::MathProfile;
use std::arch::x86_64::*;

/// Runs all layers. `values` holds the inputs on entry and the outputs on return.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn forward(
    shape: &[usize],
    params: &[f64],
    values: &mut Vec<f64>,
    next: &mut Vec<f64>,
    math: MathProfile,
) {
    let mut position = 0;
    for w in shape.windows(2) {
        let (n_in, n_out) = (w[0], w[1]);
        next.clear();
        next.resize(n_out, 0.0);
        let matrix = &params[position..position + n_in * n_out];
        let bias = &params[position + n_in * n_out..position + (n_in + 1) * n_out];
        let mut j = 0;
        while j + 4 <= n_out {
            // Up to four registers per pass over the inputs.
            j += match (n_out - j) / 4 {
                1 => block::<1>(values, matrix, bias, n_out, j, next, math),
                2 => block::<2>(values, matrix, bias, n_out, j, next, math),
                3 => block::<3>(values, matrix, bias, n_out, j, next, math),
                _ => block::<4>(values, matrix, bias, n_out, j, next, math),
            };
        }
        for j in j..n_out {
            let mut sum = 0.0;
            for (i, &value) in values.iter().enumerate() {
                sum += value * matrix[i * n_out + j];
            }
            next[j] = math.tanh(sum + bias[j]);
        }
        std::mem::swap(values, next);
        position += (n_in + 1) * n_out;
    }
}

/// Outputs `j..j + 4 * N` of one layer; returns the number written.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn block<const N: usize>(
    values: &[f64],
    matrix: &[f64],
    bias: &[f64],
    n_out: usize,
    j: usize,
    next: &mut [f64],
    math: MathProfile,
) -> usize {
    assert!(j + 4 * N <= n_out && matrix.len() == values.len() * n_out && bias.len() == n_out && next.len() == n_out);
    let mut acc = [_mm256_setzero_pd(); N];
    for (i, &value) in values.iter().enumerate() {
        let v = _mm256_set1_pd(value);
        let row = matrix.as_ptr().add(i * n_out + j);
        for (b, a) in acc.iter_mut().enumerate() {
            *a = _mm256_add_pd(*a, _mm256_mul_pd(v, _mm256_loadu_pd(row.add(4 * b))));
        }
    }
    for (b, a) in acc.iter().enumerate() {
        let sum = _mm256_add_pd(*a, _mm256_loadu_pd(bias.as_ptr().add(j + 4 * b)));
        let out = next.as_mut_ptr().add(j + 4 * b);
        if math == MathProfile::Proton {
            _mm256_storeu_pd(out, tanh4(sum));
        } else {
            _mm256_storeu_pd(out, sum);
            for lane in &mut next[j + 4 * b..j + 4 * b + 4] {
                *lane = math.tanh(*lane);
            }
        }
    }
    4 * N
}

#[inline(always)]
unsafe fn splat(v: f64) -> __m256d {
    _mm256_set1_pd(v)
}
/// Upper 32 bits of each lane, as non-negative 64-bit integers.
#[inline(always)]
unsafe fn high(x: __m256d) -> __m256i {
    _mm256_srli_epi64(_mm256_castpd_si256(x), 32)
}
/// Lane mask: `high > c`.
#[inline(always)]
unsafe fn above(h: __m256i, c: i64) -> __m256d {
    _mm256_castsi256_pd(_mm256_cmpgt_epi64(h, _mm256_set1_epi64x(c)))
}
/// Lane mask: `high < c`.
#[inline(always)]
unsafe fn below(h: __m256i, c: i64) -> __m256d {
    _mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_set1_epi64x(c), h))
}
/// `mask ? a : b` per lane.
#[inline(always)]
unsafe fn select(mask: __m256d, a: __m256d, b: __m256d) -> __m256d {
    _mm256_blendv_pd(b, a, mask)
}
#[inline(always)]
unsafe fn equal(a: __m256d, v: f64) -> __m256d {
    _mm256_cmp_pd::<_CMP_EQ_OQ>(a, splat(v))
}
/// `2^e` from integer-valued `e` (exponent-field arithmetic, as the scalar code).
#[inline(always)]
unsafe fn power_of_two(e: __m256d) -> __m256d {
    let e = _mm256_cvtepi32_epi64(_mm256_cvttpd_epi32(e));
    _mm256_castsi256_pd(_mm256_slli_epi64::<52>(_mm256_add_epi64(e, _mm256_set1_epi64x(0x3ff))))
}

/// `game_tanh` on four lanes.
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn tanh4(value: __m256d) -> __m256d {
    let sign = splat(-0.0);
    let x = _mm256_andnot_pd(sign, value);
    let h = high(x);
    let big = above(h, 0x3fe193ea);
    let huge = above(h, 0x40340000);
    let mid = above(h, 0x3fd058ae);
    let normal = above(h, 0x000fffff);
    let t = expm1_tanh(select(mid, _mm256_mul_pd(splat(2.0), x), _mm256_mul_pd(splat(-2.0), x)));
    let d = _mm256_add_pd(t, splat(2.0));
    // t / (t + 2); the small branch's (-t) / (t + 2) is its exact negation.
    let q = _mm256_div_pd(t, d);
    // 1 - 2 / (t + 2), or 1 - 0 / x above 0x40340000: one division for both.
    let q2 = _mm256_div_pd(select(huge, _mm256_setzero_pd(), splat(2.0)), select(huge, x, d));
    let r = _mm256_sub_pd(splat(1.0), q2);
    let result = select(big, r, select(mid, q, select(normal, _mm256_xor_pd(q, sign), x)));
    _mm256_xor_pd(result, _mm256_and_pd(value, sign))
}

/// `game_expm1` for the arguments `game_tanh` passes on its selected branches:
/// finite, `|x| < 0x4043687a` high word or positive below 709.78, so the
/// overflow block never returns early. Other lanes produce unused values.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn expm1_tanh(x: __m256d) -> __m256d {
    let h = high(_mm256_andnot_pd(splat(-0.0), x));
    let negative = _mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_setzero_si256(), _mm256_castpd_si256(x)));
    let reduce = above(h, 0x3fd62e42);
    let tiny = below(h, 0x3c900000);
    let near = below(h, 0x3ff0a2b2);
    // k = ±1 near ln 2, else trunc(x / ln 2 ± 0.5); 0 without reduction.
    let rounded = _mm256_add_pd(
        _mm256_mul_pd(splat(1.44269504088896338700), x),
        select(negative, splat(-0.5), splat(0.5)),
    );
    let k = select(
        near,
        select(negative, splat(-1.0), splat(1.0)),
        _mm256_cvtepi32_pd(_mm256_cvttpd_epi32(rounded)),
    );
    let k = _mm256_and_pd(reduce, k);
    // For k = ±1, x - k*ln2_hi and k*ln2_lo equal the scalar x ∓ ln2_hi, ±ln2_lo exactly.
    let hi = _mm256_sub_pd(x, _mm256_mul_pd(k, splat(6.93147180369123816490e-1)));
    let lo = _mm256_mul_pd(k, splat(1.90821492927058770002e-10));
    let xr = select(reduce, _mm256_sub_pd(hi, lo), x);
    let c = _mm256_and_pd(reduce, _mm256_sub_pd(_mm256_sub_pd(hi, xr), lo));

    let hfx = _mm256_mul_pd(splat(0.5), xr);
    let hxs = _mm256_mul_pd(xr, hfx);
    let mut p = _mm256_mul_pd(hxs, splat(-2.01099218183624371326e-7));
    p = _mm256_mul_pd(hxs, _mm256_add_pd(splat(4.00821782732936239552e-6), p));
    p = _mm256_mul_pd(hxs, _mm256_add_pd(splat(-7.93650757867487942473e-5), p));
    p = _mm256_mul_pd(hxs, _mm256_add_pd(splat(1.58730158725481460165e-3), p));
    p = _mm256_mul_pd(hxs, _mm256_add_pd(splat(-3.33333333333331316428e-2), p));
    let r1 = _mm256_add_pd(splat(1.0), p);
    let t = _mm256_sub_pd(splat(3.0), _mm256_mul_pd(r1, hfx));
    let e = _mm256_mul_pd(
        hxs,
        _mm256_div_pd(_mm256_sub_pd(r1, t), _mm256_sub_pd(splat(6.0), _mm256_mul_pd(xr, t))),
    );
    let k0 = _mm256_sub_pd(xr, _mm256_sub_pd(_mm256_mul_pd(xr, e), hxs));

    let e = _mm256_sub_pd(_mm256_sub_pd(_mm256_mul_pd(xr, _mm256_sub_pd(e, c)), c), hxs);
    let x_minus_e = _mm256_sub_pd(xr, e);
    let k_minus1 = _mm256_sub_pd(_mm256_mul_pd(splat(0.5), x_minus_e), splat(0.5));
    let k1 = select(
        _mm256_cmp_pd::<_CMP_LT_OQ>(xr, splat(-0.25)),
        _mm256_mul_pd(splat(-2.0), _mm256_sub_pd(e, _mm256_add_pd(xr, splat(0.5)))),
        _mm256_add_pd(splat(1.0), _mm256_mul_pd(splat(2.0), x_minus_e)),
    );
    let twopk = power_of_two(k);
    let inverse = power_of_two(_mm256_sub_pd(_mm256_setzero_pd(), k));
    // k < 0 || k > 56 (k == 1024 needs x > 709, outside the domain).
    let far = _mm256_or_pd(
        _mm256_cmp_pd::<_CMP_LT_OQ>(k, splat(0.0)),
        _mm256_cmp_pd::<_CMP_GT_OQ>(k, splat(56.0)),
    );
    let far_value = _mm256_sub_pd(_mm256_mul_pd(_mm256_add_pd(x_minus_e, splat(1.0)), twopk), splat(1.0));
    let below20 = _mm256_mul_pd(_mm256_add_pd(x_minus_e, _mm256_sub_pd(splat(1.0), inverse)), twopk);
    let from20 = _mm256_mul_pd(
        _mm256_add_pd(_mm256_sub_pd(xr, _mm256_add_pd(e, inverse)), splat(1.0)),
        twopk,
    );
    let general = select(
        far,
        far_value,
        select(_mm256_cmp_pd::<_CMP_LT_OQ>(k, splat(20.0)), below20, from20),
    );
    let result = select(
        equal(k, 0.0),
        k0,
        select(equal(k, -1.0), k_minus1, select(equal(k, 1.0), k1, general)),
    );
    select(tiny, x, result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::network::game_tanh;

    /// xorshift64*: deterministic cases without an RNG dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn check_tanh(inputs: &[f64]) {
        for chunk in inputs.chunks(4) {
            let mut lanes = [0.0; 4];
            lanes[..chunk.len()].copy_from_slice(chunk);
            let mut out = [0.0; 4];
            unsafe { _mm256_storeu_pd(out.as_mut_ptr(), tanh4(_mm256_loadu_pd(lanes.as_ptr()))) };
            for (&v, &o) in lanes.iter().zip(&out) {
                assert_eq!(
                    o.to_bits(),
                    game_tanh(v).to_bits(),
                    "tanh({v:e}) bits {:#x}",
                    v.to_bits()
                );
            }
        }
    }

    #[test]
    fn simd_forward_matches_scalar_loop() {
        use crate::nn::network::{ForwardScratch, Network};
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = Rng(0x2545F4914F6CDD1D);
        // The rally shape, plus widths that exercise 1-3 register blocks and scalar tails.
        let shapes: [&[usize]; 4] = [
            &[20, 16, 16, 16, 16, 12, 12, 8, 5],
            &[7, 13, 5, 9, 3, 20, 1],
            &[4, 4],
            &[1, 1],
        ];
        let (mut a, mut b) = (ForwardScratch::default(), ForwardScratch::default());
        for shape in shapes {
            for scale in [0.3, 1.0, 12.0] {
                let count = crate::nn::network::parameter_count(shape);
                let network =
                    Network::from_vector(shape, (0..count).map(|_| (rng.unit() * 2.0 - 1.0) * scale).collect());
                for step in 0..20_000 {
                    let math = MathProfile::ALL[step % 3];
                    let inputs: Vec<f64> = (0..shape[0])
                        .map(|_| {
                            let u = rng.unit();
                            if u < 0.05 {
                                0.0
                            } else if u < 0.08 {
                                1.0
                            } else if u < 0.1 {
                                -0.0
                            } else {
                                (rng.unit() * 2.0 - 1.0) * 4.0
                            }
                        })
                        .collect();
                    let simd = network.forward_into(&inputs, &mut a, math);
                    let scalar = network.forward_into_scalar(&inputs, &mut b, math);
                    assert_eq!(
                        simd.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "{math} {shape:?} {inputs:?}"
                    );
                }
            }
        }
    }

    /// The scalar `game_expm1` branches often agree to the last bit, and tanh
    /// rounding hides more, so check the vector expm1 on its own.
    #[test]
    fn expm1_matches_game_expm1_on_the_tanh_domain() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = Rng(0xD1B54A32D192ED03);
        let mut values = vec![0.0, -0.0, 5e-324, -5e-324, f64::MIN_POSITIVE, 709.0, -38.0];
        // Whole high words at each threshold (random low bits), both signs.
        for high in [0x3c900000u64, 0x3fd62e42, 0x3ff0a2b2, 0x4043687a] {
            for h in high - 2..=high + 2 {
                for _ in 0..20_000 {
                    values.push(f64::from_bits(h << 32 | (rng.next() & 0xffff_ffff)));
                }
            }
        }
        // The k = 1 branch switches formulas where the reduced x crosses -0.25.
        let split = (std::f64::consts::LN_2 - 0.25).to_bits();
        values.extend((split - 20_000..split + 20_000).map(f64::from_bits));
        // Around k = ±1, 20, 56 and 57.
        for k in [
            -1.5f64, -1.0, -0.5, 0.5, 1.0, 1.5, 19.5, 20.0, 20.5, 55.5, 56.0, 56.5, 57.0, 57.5,
        ] {
            let x = k * std::f64::consts::LN_2;
            for _ in 0..20_000 {
                values.push(x + (rng.unit() - 0.5) * 1e-6 * x.abs());
            }
        }
        for _ in 0..1_000_000 {
            values.push(rng.unit() * 747.0 - 38.0);
            values.push((rng.unit() - 0.5) * 3.0);
            values.push(if rng.unit() < 0.5 { -1.0 } else { 1.0 } * 10f64.powf(rng.unit() * 20.0 - 19.0));
        }
        values.retain(|v| (-38.8..=709.7).contains(v));
        for chunk in values.chunks(4) {
            let mut lanes = [0.0; 4];
            lanes[..chunk.len()].copy_from_slice(chunk);
            let mut out = [0.0; 4];
            unsafe { _mm256_storeu_pd(out.as_mut_ptr(), expm1_tanh(_mm256_loadu_pd(lanes.as_ptr()))) };
            for (&v, &o) in lanes.iter().zip(&out) {
                let want = crate::nn::network::game_expm1(v);
                assert_eq!(
                    o.to_bits(),
                    want.to_bits(),
                    "expm1({v:e}) bits {:#x}: {o:e} vs {want:e}",
                    v.to_bits()
                );
            }
        }
    }

    #[test]
    fn tanh4_matches_every_game_tanh_branch() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let mut values = vec![
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -f64::NAN,
            f64::MIN_POSITIVE,
            5e-324,
            f64::MAX,
        ];
        values.push(f64::from_bits(0x7ff0_0000_0000_0001)); // signalling NaN payload
                                                            // A few ulps around every tanh and expm1 threshold, for x and for 2x.
        for high in [
            0x3fe193eau64,
            0x40340000,
            0x3fd058ae,
            0x00100000,
            0x3c900000,
            0x3fd62e42,
            0x3ff0a2b2,
            0x4043687a,
        ] {
            for base in [high << 32, (high << 32) - (1 << 52)] {
                for delta in -3i64..=3 {
                    values.push(f64::from_bits((base as i64).wrapping_add(delta) as u64));
                }
            }
        }
        // Whole high words at each threshold (random low bits), for x and 2x.
        for high in [
            0x3fe193eau64,
            0x40340000,
            0x3fd058ae,
            0x00100000,
            0x3c900000,
            0x3fd62e42,
            0x3ff0a2b2,
            0x4043687a,
        ] {
            for h in (high - 2..=high + 2).flat_map(|h| [h, h.wrapping_sub(0x100000)]) {
                for _ in 0..10_000 {
                    values.push(f64::from_bits(h << 32 | (rng.next() & 0xffff_ffff)));
                }
            }
        }
        // k boundaries (20, 56) of the scaled result.
        for k in [19.5f64, 20.0, 20.5, 55.5, 56.0, 56.5, 57.0, 57.5, 58.0] {
            let x = k * std::f64::consts::LN_2 / 2.0;
            for delta in -40i64..=40 {
                values.push(f64::from_bits((x.to_bits() as i64 + delta) as u64));
            }
        }
        for _ in 0..1_000_000 {
            values.push(10f64.powf(rng.unit() * 26.0 - 24.0));
            values.push(rng.unit() * 24.0);
            values.push(f64::from_bits(rng.next()));
        }
        let values: Vec<f64> = values.into_iter().flat_map(|v| [v, -v]).collect();
        check_tanh(&values);
    }
}
