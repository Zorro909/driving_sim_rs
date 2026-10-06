// The float math of the Windows UCRT (ucrtbase.dll, FMA3 code paths), bit for bit.
//
// win10_*: ucrtbase 10.0.22621 (Windows 10, and Windows 11 up to 23H2).
// win11_*: ucrtbase 10.0.26100 (Windows 11 24H2 and later).
// win_*:   the same in both.
//
// One source for every backend: tools/ucrt/gen-math.py translates it to Rust for the CPU
// (x86, ARM and wasm), gpu/sim/math.h includes it for HIP and CUDA, and wasm/gen-sim.py
// translates it to WGSL. It is therefore restricted C:
// - The operands of an operator have one type, except shift counts and literals; every other
//   conversion is a cast.
// - A fused multiply-add is an explicit fma(). Nothing else may be contracted, so C and HIP/CUDA
//   builds use -ffp-contract=off.
// - No pointers, references, recursion, macros, or side effects inside expressions.
// - NaN results are built from bits, because GPU arithmetic does not return x86's NaNs.
// - Doubles are only converted to integers when the value is in range.
//
// The includer defines ALTD_MATH_FN (function specifiers), ALTD_MATH_TABLE (table specifiers)
// and ALTD_MATH_CONST (scalar constant specifiers). It also provides bool, the <stdint.h> types,
// fma, and the bit casts fbits/ffrom (float) and dbits/dfrom (double).
#pragma once
#include "ucrt_tables.h"

ALTD_MATH_FN bool win_isnan(double x) { return (dbits(x) & 0x7fffffffffffffffull) > 0x7ff0000000000000ull; }

// x86's NaN result of an operation on NaN operands: the first NaN, quieted.
ALTD_MATH_FN double win_quiet(double x) { return dfrom(dbits(x) | 0x0008000000000000ull); }
ALTD_MATH_FN double win_nan_operand(double a, double b) { return win_isnan(a) ? win_quiet(a) : win_quiet(b); }

// ---- atan2f ---------------------------------------------------------------------------------

ALTD_MATH_FN float win_atan2f(float yf, float xf) {
    // The x operand's NaN first.
    if ((fbits(xf) & 0x7fffffffu) > 0x7f800000u) return ffrom(fbits(xf) | 0x00400000u);
    if ((fbits(yf) & 0x7fffffffu) > 0x7f800000u) return ffrom(fbits(yf) | 0x00400000u);
    double x = (double)xf;
    double y = (double)yf;
    uint64_t xb = dbits(x);
    uint64_t yb = dbits(y);
    int32_t de = (int32_t)((yb >> 52) & 0x7ffull) - (int32_t)((xb >> 52) & 0x7ffull);
    uint64_t ax = xb & 0x7fffffffffffffffull;
    uint64_t ay = yb & 0x7fffffffffffffffull;
    bool xneg = (xb >> 63) != 0ull;
    bool yneg = (yb >> 63) != 0ull;
    float pi = ffrom(0x40490fdbu);
    float half_pi = ffrom(0x3fc90fdbu);
    if (ay == 0ull) {
        if (xneg) return yneg ? -pi : pi;
        return yf;
    }
    if (ax == 0ull && yneg) return -half_pi;
    if (de > 26) return yneg ? -half_pi : half_pi;
    if (de < -13 && !xneg) {
        // The DLL scales quotients below 2^-126 by 2^100 and back, which rounds the same.
        if (de < -150) return yneg ? ffrom(0x80000000u) : ffrom(0u);
        return (float)(y / x);
    }
    if (de < -26 && xneg) return yneg ? -pi : pi;
    if (ay == 0x7ff0000000000000ull && ax == 0x7ff0000000000000ull) {
        if (xneg) return yneg ? ffrom(0xc016cbe4u) : ffrom(0x4016cbe4u);
        return yneg ? ffrom(0xbf490fdbu) : ffrom(0x3f490fdbu);
    }
    double sx = xneg ? -x : x;
    double sy = yneg ? -y : y;
    bool swap = sy > sx;
    double num = swap ? sx : sy;
    double den = swap ? sy : sx;
    double t = num / den;
    double r;
    if (t > 0.0625) {
        int32_t n = (int32_t)(t * 256.0 + 0.5);
        double nd = (double)n;
        double u = (num * 256.0 - nd * den) / (nd * num + den * 256.0);
        double hi = u + dfrom(UCRT_ATAN2F_ATAN256[n - 16]);
        double u3 = u * u;
        u3 = u3 * u;
        u3 = u3 * 0.33333333333224097;
        r = hi - u3;
    } else if (t < 0.0001) {
        r = t;
    } else {
        double t2 = t * t;
        double p = 0.19999999999393223 - t2 * 0.1428571356180717;
        p = p * t2;
        double t3 = t2 * t;
        double q = 0.3333333333333317 - p;
        q = q * t3;
        r = t - q;
    }
    if (swap) r = 1.5707963267948966 - r;
    if (xneg) r = 3.141592653589793 - r;
    if (yneg) r = -r;
    return (float)r;
}

// ---- exp ------------------------------------------------------------------------------------

// exp of finite x in [-744.0346068132731, 709.782712893384].
ALTD_MATH_FN double win_exp_finite(double x) {
    if ((dbits(x) & 0x7fffffffffffffffull) <= 0x3e50000000000000ull) return x + 1.0;
    int32_t n = (int32_t)(x * 92.33248261689366);
    double nd = (double)n;
    double r = nd * -2.5728046223276688e-14 + fma(nd, -0.010830424696223417, x);
    int32_t j = n & 63;
    int32_t k = n >> 6;
    double p = fma(0.001388888888888889, r, 0.008333333333333333);
    p = fma(p, r, 0.041666666666666664);
    p = fma(p, r, 0.16666666666666666);
    p = fma(p, r, 0.5);
    double e = fma(r * r, p, r);
    double t = e * dfrom(UCRT_EXP_MUL[j]) + dfrom(UCRT_EXP_LO[j]);
    double res = t + dfrom(UCRT_EXP_HI[j]);
    if (k > -1022 || (k == -1022 && res >= 1.0)) return dfrom(dbits(res) + ((uint64_t)(int64_t)k << 52));
    return res * dfrom(1ull << (uint32_t)((k + 1074) & 63));
}

ALTD_MATH_FN double win10_exp(double x) {
    uint64_t xb = dbits(x);
    if ((xb & 0x7fffffffffffffffull) >= 0x7ff0000000000000ull) {
        if (xb == 0x7ff0000000000000ull) return x;
        if (xb == 0xfff0000000000000ull) return 0.0;
        return win_quiet(x);
    }
    if (x > 709.782712893384) return dfrom(0x7ff0000000000000ull);
    if (x < -744.0346068132731) return 0.0;
    return win_exp_finite(x);
}

// As win10_exp, but the smallest subnormal below -744.0346068132731 down to -745.1332191019411.
ALTD_MATH_FN double win11_exp(double x) {
    uint64_t xb = dbits(x);
    if ((xb & 0x7fffffffffffffffull) >= 0x7ff0000000000000ull) {
        if (xb == 0x7ff0000000000000ull) return x;
        if (xb == 0xfff0000000000000ull) return 0.0;
        return win_quiet(x);
    }
    if (x > 709.782712893384) return dfrom(0x7ff0000000000000ull);
    if (x < -744.0346068132731) return x >= -745.1332191019411 ? dfrom(1ull) : 0.0;
    return win_exp_finite(x);
}

// ---- tanh -----------------------------------------------------------------------------------

// The polynomial coefficients of |x| <= 1, for |x| below and from 0.9.
ALTD_MATH_FN double win_tanh_k(int32_t i, bool up) {
    double v = 0.0;
    if (i == 0) v = up ? -0.00016559704390354995 : -0.0002000476210719095;
    if (i == 1) v = up ? 1.154758789961434e-08 : 1.4207792637883471e-08;
    if (i == 2) v = up ? 0.00017307605012622596 : 0.00020911402625291644;
    if (i == 3) v = up ? 0.016735877546189656 : 0.020156216602693764;
    if (i == 4) v = up ? 0.014617304728873168 : 0.017601634900304468;
    if (i == 5) v = up ? 0.3172045589772944 : 0.3816414142883289;
    if (i == 6) v = up ? 0.2277938706590883 : 0.27403042465617977;
    if (i == 7) v = up ? 0.6833816119772959 : 0.8220912739685393;
    return v;
}

// 2^k as the product of two normal powers of two, for 1 + 2^k * m.
ALTD_MATH_FN double win_tanh_scale1(int32_t k) { return dfrom((uint64_t)(int64_t)(((k + (k < 0 ? 1 : 0)) >> 1) + 1023) << 52); }
ALTD_MATH_FN double win_tanh_scale2(int32_t k) { return dfrom((uint64_t)(int64_t)(k - ((k + (k < 0 ? 1 : 0)) >> 1) + 1023) << 52); }

// win10 tanh rounds every operation.
ALTD_MATH_FN double win10_tanh(double x) {
    uint64_t xb = dbits(x);
    uint64_t ab = xb & 0x7fffffffffffffffull;
    if (ab < 0x3e30000000000000ull) return x;
    if (ab > 0x7ff0000000000000ull) return win_quiet(x);
    bool neg = ab != xb;
    double a = dfrom(ab);
    if (a > 20.0) return neg ? -1.0 : 1.0;
    double res;
    if (a > 1.0) {
        double a2 = a + a;
        int32_t n = (int32_t)(a2 * 46.16624130844683 + 0.5);
        int32_t j = n & 31;
        double nd = (double)n;
        double hi = dfrom(UCRT_TANH_HI[j]);
        double lo = dfrom(UCRT_TANH_LO[j]);
        int32_t k = (n - j) >> 5;
        double r6 = a2 - nd * 0.021660849335603416;
        double r1 = (double)(-n) * 5.689487495325456e-11;
        double r = r1 + r6;
        double p = r * 0.001388894908637772 + 0.008333367984342196;
        p = p * r + 0.04166666666622608;
        p = p * r + 0.16666666666526087;
        p = p * r;
        p = p + 0.5;
        p = p * (r * r);
        p = p + r1;
        p = p + r6;
        p = p * (lo + hi);
        p = p + lo;
        double t = (hi + p) * win_tanh_scale1(k);
        t = t * win_tanh_scale2(k);
        t = t + 1.0;
        res = 1.0 - 2.0 / t;
    } else {
        bool up = a >= 0.9;
        double x2 = a * a;
        double x3 = x2 * a;
        double p = win_tanh_k(0, up) - x2 * win_tanh_k(1, up);
        p = p * x2;
        p = p - win_tanh_k(4, up);
        p = p * x2;
        p = p - win_tanh_k(6, up);
        p = p * x3;
        double q = x2 * win_tanh_k(2, up) + win_tanh_k(3, up);
        q = q * x2;
        q = q + win_tanh_k(5, up);
        q = q * x2;
        q = q + win_tanh_k(7, up);
        res = p / q + a;
    }
    return neg ? -res : res;
}

// win11 tanh fuses the polynomials.
ALTD_MATH_FN double win11_tanh(double x) {
    uint64_t xb = dbits(x);
    uint64_t ab = xb & 0x7fffffffffffffffull;
    if (ab < 0x3e30000000000000ull) return x;
    if (ab > 0x7ff0000000000000ull) return win_quiet(x);
    bool neg = ab != xb;
    double a = dfrom(ab);
    if (a > 20.0) return neg ? -1.0 : 1.0;
    double res;
    if (a > 1.0) {
        double a2 = a + a;
        int32_t n = (int32_t)(a2 * 46.16624130844683 + 0.5);
        int32_t j = n & 31;
        double nd = (double)n;
        double hi = dfrom(UCRT_TANH_HI[j]);
        double lo = dfrom(UCRT_TANH_LO[j]);
        int32_t k = (n - j) >> 5;
        double r6 = fma(-nd, 0.021660849335603416, a2);
        double r1 = (double)(-n) * 5.689487495325456e-11;
        double r = r1 + r6;
        double p = fma(0.001388894908637772, r, 0.008333367984342196);
        p = fma(p, r, 0.04166666666622608);
        p = fma(p, r, 0.16666666666526087);
        p = fma(p, r, 0.5);
        double m = fma(r * r, p, r1) + r6;
        m = fma(hi + lo, m, lo);
        double t = (hi + m) * win_tanh_scale1(k);
        t = fma(t, win_tanh_scale2(k), 1.0);
        res = 1.0 - 2.0 / t;
    } else {
        bool up = a >= 0.9;
        double x2 = a * a;
        double x3 = x2 * a;
        double p = fma(-win_tanh_k(1, up), x2, win_tanh_k(0, up));
        p = fma(p, x2, -win_tanh_k(4, up));
        p = fma(p, x2, -win_tanh_k(6, up));
        p = p * x3;
        double q = fma(win_tanh_k(2, up), x2, win_tanh_k(3, up));
        q = fma(q, x2, win_tanh_k(5, up));
        q = fma(q, x2, win_tanh_k(7, up));
        res = p / q + a;
    }
    return neg ? -res : res;
}

// ---- pow ------------------------------------------------------------------------------------

// 0: y is not an integer, 1: an odd integer, 2: an even integer.
ALTD_MATH_FN int32_t win_integer_kind(uint64_t yb) {
    uint64_t e = (yb >> 52) & 0x7ffull;
    if (e < 1023ull) return 0;
    if (e > 1075ull) return 2;
    uint64_t bit = 1ull << (uint32_t)(1075ull - e);
    if ((yb & (bit - 1ull)) != 0ull) return 0;
    return (yb & bit) != 0ull ? 1 : 2;
}

// win10 pow of finite x > 0 and finite y with 2^-63 <= |y| <= 2^63.
ALTD_MATH_FN double win10_pow_core(double x, double y) {
    uint64_t xb = dbits(x);
    uint64_t yb = dbits(y);
    if (yb == 0x3ff0000000000000ull) return x;
    if (xb == 0x3ff0000000000000ull) return 1.0;
    uint64_t mb = xb;
    int32_t ex = (int32_t)(xb >> 52) - 1023;
    uint64_t mant = xb & 0x000fffffffffffffull;
    if (ex == -1023) {
        mb = dbits(dfrom(mant | 0x3ff0000000000000ull) - 1.0);
        mant = mb & 0x000fffffffffffffull;
        ex = (int32_t)(mb >> 52) - 2045;
    }
    double e = (double)ex;
    uint64_t f1 = (mb & 0x000ff00000000000ull) + ((mb & 0x0000080000000000ull) << 1);
    int32_t j = (int32_t)(f1 >> 44);
    double d = dfrom(f1 | 0x3fe0000000000000ull) - dfrom(mant | 0x3fe0000000000000ull);
    double hiv = d * dfrom(UCRT_POW_T1[j]);
    double lov = d * dfrom(UCRT_POW_T2[j]);
    double z = hiv + lov;
    double err = hiv - z;
    double zz = z * z;
    err = err + lov;
    double p = fma(0.1428571428571429, z, 0.16666666666666666);
    p = fma(p, z, 0.2);
    p = fma(p, z, 0.25);
    p = fma(p, z, 0.3333333333333333);
    p = fma(p, z, 0.5);
    p = fma(p, zz, err);
    double s5 = fma(e, 5.7699990475432854e-08, -p);
    double s3 = s5 + dfrom(UCRT_POW_T4[j]);
    double s1 = s3;
    s3 = s3 - z;
    double lg = fma(e, 0.6931471228599548, dfrom(UCRT_POW_T3[j]));
    double s7 = lg;
    lg = lg + s3;
    double lg_hi = dfrom(dbits(lg) & 0xfffffffff8000000ull);
    double zs = z + s3;
    s7 = s7 - lg;
    s1 = s1 - zs;
    s7 = s7 + s3;
    s7 = s7 + s1;
    s7 = s7 + (lg - lg_hi);
    double y_hi = dfrom(yb & 0xfffffffff8000000ull);
    double y_lo = y - y_hi;
    double a3 = y_lo * s7;
    a3 = a3 + y_lo * lg_hi;
    a3 = a3 + s7 * y_hi;
    double a6 = lg_hi * y_hi;
    double prod_hi = a6 + a3;
    double prod_lo = a6 - prod_hi;
    prod_lo = prod_lo + a3;
    double w = prod_hi * 92.33248261689366;
    if (w > 65536.0) return dfrom(0x7ff0000000000000ull);
    if (w < -68800.0) return 0.0;
    // Round to nearest even: |w| <= 68800 is far below 2^51.
    int32_t n = (int32_t)(w + 6755399441055744.0 - 6755399441055744.0);
    double nd = (double)n;
    double r0 = fma(-nd, 0.010830424260348082, prod_hi);
    int32_t jj = n & 63;
    int32_t k = (n - jj) >> 6;
    double r = r0 + nd * -4.359010638708991e-10;
    r = r + prod_lo;
    double q = fma(0.001388888888888889, r, 0.008333333333333333);
    q = fma(q, r, 0.041666666666666664);
    q = fma(q, r, 0.16666666666666666);
    q = fma(q, r, 0.5);
    q = fma(q, r, 1.0);
    q = q * r;
    double lo = q * dfrom(UCRT_EXP_LO[jj]);
    double u = q * dfrom(UCRT_EXP_HI[jj]);
    lo = lo + dfrom(UCRT_EXP_LO[jj]);
    u = u + lo;
    u = u + dfrom(UCRT_EXP_HI[jj]);
    uint64_t scale = (uint64_t)(int64_t)(k + 1023) << 52;
    if (k <= -1022) {
        if (k == -1022 && u >= 1.0) return u * dfrom(scale);
        if ((int64_t)dbits(prod_hi) > (int64_t)0xc0874046dfefd9d0ull) return dfrom(1ull);
        int32_t sh = k + 1074 > 0 ? k + 1074 : 0;
        return u * dfrom(1ull << (uint32_t)(sh & 63));
    }
    if (k + 1023 == 2047) {
        if (u >= 1.0) return dfrom(0x7ff0000000000000ull);
        return dfrom(dbits(u) | 0x7fe0000000000000ull);
    }
    return u * dfrom(scale);
}

ALTD_MATH_FN double win10_pow(double x, double y) {
    uint64_t xb = dbits(x);
    uint64_t yb = dbits(y);
    bool xnan = win_isnan(x);
    bool ynan = win_isnan(y);
    // The NaN operand, or of two the first unless only it has the sign bit set.
    double qnan = (xnan && (!ynan || (xb >> 63) == 0ull || (yb >> 63) != 0ull)) ? win_quiet(x) : win_quiet(y);
    // A signalling NaN is invalid even where C99 gives 1.
    if ((xnan && (xb & 0x0008000000000000ull) == 0ull) || (ynan && (yb & 0x0008000000000000ull) == 0ull)) return qnan;
    if ((yb & 0x7fffffffffffffffull) == 0ull || xb == 0x3ff0000000000000ull) return 1.0;
    if (xnan || ynan) return qnan;
    bool xneg = (xb >> 63) != 0ull;
    uint64_t axb = xb & 0x7fffffffffffffffull;
    bool yinf = (yb & 0x7fffffffffffffffull) == 0x7ff0000000000000ull;
    int32_t kind = win_integer_kind(yb);
    // pow(-1, y) is 1 for infinite y and for every even y.
    if (axb == 0x3ff0000000000000ull && (yinf || kind == 2)) return 1.0;
    bool ypos = (yb >> 63) == 0ull;
    if (yinf) return (axb < 0x3ff0000000000000ull) == ypos ? 0.0 : dfrom(0x7ff0000000000000ull);
    uint64_t sign = xneg && kind == 1 ? 0x8000000000000000ull : 0ull;
    if (axb == 0ull) return dfrom((ypos ? 0ull : 0x7ff0000000000000ull) | sign);
    if (axb == 0x7ff0000000000000ull) return dfrom((ypos ? 0x7ff0000000000000ull : 0ull) | sign);
    if (xneg && kind == 0) return dfrom(0xfff8000000000000ull);
    uint64_t ye = yb & 0x7ff0000000000000ull;
    // |y| beyond 2^63 is an even integer.
    if (ye > 0x43e0000000000000ull) return (axb < 0x3ff0000000000000ull) == ypos ? 0.0 : dfrom(0x7ff0000000000000ull);
    if (ye < 0x3c00000000000000ull) return dfrom(dbits(y + 1.0) | sign);
    return dfrom(dbits(win10_pow_core(dfrom(axb), y)) | sign);
}

// win11 pow is Arm's optimized-routines pow (N = 128 log, N = 256 exp) with FMA, plus two
// shortcuts: pow(x, 1) = x and pow(+-1, y) = +-1.
ALTD_MATH_FN double win11_pow(double x, double y) {
    uint64_t ix = dbits(x);
    uint64_t iy = dbits(y);
    uint32_t topx = (uint32_t)(ix >> 52);
    uint32_t topy = (uint32_t)(iy >> 52);
    uint64_t sign_bias = 0ull;
    if (topx - 1u >= 0x7feu || (topy & 0x7ffu) - 0x3beu >= 0x80u) {
        // Special y: zero, infinite or NaN.
        if (iy * 2ull - 1ull >= 0xffdfffffffffffffull) {
            if (iy * 2ull == 0ull) return (ix ^ 0x0008000000000000ull) * 2ull > 0xfff0000000000000ull ? win_nan_operand(x, y) : 1.0;
            if (ix == 0x3ff0000000000000ull) return (iy ^ 0x0008000000000000ull) * 2ull > 0xfff0000000000000ull ? win_nan_operand(x, y) : 1.0;
            if (ix * 2ull > 0xffe0000000000000ull || iy * 2ull > 0xffe0000000000000ull) return win_nan_operand(x, y);
            if (ix * 2ull == 0x7fe0000000000000ull) return 1.0;
            if ((ix * 2ull < 0x7fe0000000000000ull) == ((iy >> 63) == 0ull)) return 0.0;
            return y * y;
        }
        // Special x: zero, infinite or NaN.
        if (ix * 2ull - 1ull >= 0xffdfffffffffffffull) {
            uint64_t x2 = win_isnan(x) ? dbits(win_quiet(x)) : dbits(x * x);
            uint64_t neg = 0ull;
            if ((ix >> 63) != 0ull && win_integer_kind(iy) == 1) neg = 0x8000000000000000ull;
            x2 = x2 ^ neg;
            if ((iy >> 63) == 0ull || win_isnan(x)) return dfrom(x2);
            if (ix * 2ull == 0ull) return dfrom(0x7ff0000000000000ull | neg);
            return 1.0 / dfrom(x2);
        }
        // Finite x < 0: y must be an integer.
        if ((ix >> 63) != 0ull) {
            int32_t kind = win_integer_kind(iy);
            if (kind == 0) return dfrom(0xfff8000000000000ull);
            if (kind == 1) sign_bias = 0x80000ull;
            ix = ix & 0x7fffffffffffffffull;
            topx = topx & 0x7ffu;
        }
        // |y| outside [2^-65, 2^63].
        if ((topy & 0x7ffu) - 0x3beu >= 0x80u) {
            if (ix == 0x3ff0000000000000ull) return 1.0;
            if ((topy & 0x7ffu) < 0x3beu) return ix > 0x3ff0000000000000ull ? y + 1.0 : 1.0 - y;
            return (ix > 0x3ff0000000000000ull) == (topy < 0x800u) ? dfrom(0x7ff0000000000000ull) : 0.0;
        }
        // Subnormal x.
        if (topx == 0u) ix = (dbits(x * 4503599627370496.0) & 0x7fffffffffffffffull) - 0x0340000000000000ull;
    }
    if (iy == 0x3ff0000000000000ull) return x;
    if (ix == 0x3ff0000000000000ull) return sign_bias != 0ull ? -1.0 : 1.0;

    // log(x) as hi + lo.
    uint64_t tmp = ix - 0x3fe6955500000000ull;
    int32_t i = (int32_t)((tmp >> 45) & 127ull);
    double kd = (double)(int32_t)((int64_t)tmp >> 52);
    double z = dfrom(ix - (tmp & 0xfff0000000000000ull));
    double r = fma(dfrom(UCRT_POW11_INVC[i]), z, -1.0);
    double t1 = kd * 0.6931471805598903 + dfrom(UCRT_POW11_LOGC[i]);
    double ar = r * -0.5;
    double ar2 = ar * r;
    double lo3 = fma(ar, r, -ar2);
    double lo1 = kd * 5.497923018708371e-14 + dfrom(UCRT_POW11_LOGCTAIL[i]);
    double t2 = t1 + r;
    double lo2 = t1 - t2 + r;
    double lo = lo3 + (lo1 + lo2);
    double hi = ar2 + t2;
    double lo4 = t2 - hi + ar2;
    lo = lo + lo4;
    double p = (0.7999999995323976 - r * 0.6666666663487739) + (r * 1.0000415263675542 - 1.142909628459501) * ar2;
    p = p * ar2 + (r * 0.5000000000000007 - 0.6666666666666679);
    p = p * (ar2 * r);
    lo = lo + p;
    double lhi = lo + hi;
    double ltail = hi - lhi + lo;

    // exp(y * log(x)) with the tail ehi + elo.
    double ehi = lhi * y;
    double elo = fma(y, lhi, -ehi) + ltail * y;
    uint32_t abstop = (uint32_t)(dbits(ehi) >> 52) & 0x7ffu;
    if (abstop - 0x3c9u >= 0x3fu) {
        if (abstop - 0x3c9u >= 0x80000000u) {
            double one = ehi + 1.0;
            return sign_bias != 0ull ? -one : one;
        }
        if (abstop >= 0x409u) {
            uint64_t s = sign_bias != 0ull ? 0x8000000000000000ull : 0ull;
            return dfrom(((dbits(ehi) >> 63) != 0ull ? 0ull : 0x7ff0000000000000ull) | s);
        }
        abstop = 0u;
    }
    double kd2 = ehi * 369.3299304675746 + 103079215104.5;
    uint64_t ki = dbits(kd2) >> 16;
    double kn = (double)(int32_t)(uint32_t)ki;
    double er = ehi - kn * 0.0027076061741126978 + kn * 5.0411407304988844e-14 + elo;
    int32_t idx = 2 * (int32_t)(ki & 255ull);
    uint64_t sbits = UCRT_POW11_EXP[idx + 1] + ((ki + sign_bias) << 44);
    double er2 = er * er;
    double e = er2 * (er * 0.16666666666665886 + 0.49999999999996786) + (er + dfrom(UCRT_POW11_EXP[idx]));
    e = e + er2 * er2 * (er * 0.008333335853047663 + 0.04166668084093659);
    if (abstop != 0u) return e * dfrom(sbits) + dfrom(sbits);
    // The result may be subnormal or overflow.
    if ((ki & 0x80000000ull) == 0ull) {
        double scale = dfrom(sbits - 0x3f10000000000000ull);
        return (e * scale + scale) * 5.486124068793689e+303;
    }
    uint64_t sb = sbits + 0x3fe0000000000000ull;
    double scale = dfrom(sb);
    double se = scale * e;
    double v = scale + se;
    if ((dbits(v) & 0x7fffffffffffffffull) < 0x3ff0000000000000ull) {
        double one = v < 0.0 ? -1.0 : 1.0;
        double h = one + v;
        double l = (scale - v + se) + (one - h + v);
        v = l + h - one;
        if (v == 0.0) v = dfrom(sb & 0x8000000000000000ull);
    }
    return v * 2.2250738585072014e-308;
}

#ifndef ALTD_MATH_WGSL
// ---- log (CPU only) -------------------------------------------------------------------------

// AMD's libm log, the same in both builds.
ALTD_MATH_FN double win_log(double x) {
    uint64_t ux = dbits(x);
    if ((ux & 0x7ff0000000000000ull) == 0x7ff0000000000000ull) {
        if (ux == 0x7ff0000000000000ull) return x;
        if (ux == 0xfff0000000000000ull) return dfrom(0xfff8000000000000ull);
        return win_quiet(x);
    }
    if (!(x > 0.0)) return x == 0.0 ? dfrom(0xfff0000000000000ull) : dfrom(0xfff8000000000000ull);
    double xm1 = x - 1.0;
    if ((dbits(xm1) & 0x7fffffffffffffffull) < 0x3fb0000000000000ull) {
        double u = xm1 / (2.0 + xm1);
        double ru = xm1 * u;
        double u2 = u + u;
        double v = u2 * u2;
        double a = fma(0.012500000003771751, v, 0.08333333333333179);
        double b = fma(0.0004348877777076146, v, 0.0022321399879194482);
        double w = v * u2;
        a = a * w;
        double w3 = w * w * u2;
        a = fma(b, w3, a) - ru;
        return xm1 + a;
    }
    int32_t ex = (int32_t)(ux >> 52) - 1023;
    uint64_t mb = ux;
    if (ex == -1023) {
        mb = dbits(dfrom((ux & 0x000fffffffffffffull) | 0x3ff0000000000000ull) - 1.0);
        ex = (int32_t)(mb >> 52) - 2045;
    }
    double e = (double)ex;
    uint64_t f1 = (mb & 0x000ff00000000000ull) + ((mb & 0x0000080000000000ull) << 1);
    int32_t j = (int32_t)(f1 >> 44);
    double u = (dfrom(f1 | 0x3fe0000000000000ull) - dfrom((mb & 0x000fffffffffffffull) | 0x3fe0000000000000ull)) * dfrom(UCRT_LOG_INV[j]);
    double u2 = u * u;
    double p3 = fma(0.16666666666666666, u, 0.2);
    p3 = fma(p3, u, 0.25);
    double s = fma(fma(0.3333333333333333, u, 0.5), u2, u);
    s = fma(p3, u2 * u2, s);
    double x5 = fma(5.7699990475432854e-08, e, -s);
    return fma(e, 0.6931471228599548, dfrom(UCRT_LOG_LEAD[j])) + (dfrom(UCRT_LOG_TAIL[j]) + x5);
}

// ---- sinf, cosf (CPU, HIP and CUDA) ---------------------------------------------------------

// The genuine functions are musl's (src/math/native_math.rs, gpu/sim/math.h) except for the
// inputs in the exception tables. `musl` is musl's result for x.
ALTD_MATH_CONST uint32_t WIN_TRIG_LIMIT = 0x4dc90fdbu;

// For |x| < WIN_TRIG_LIMIT.
ALTD_MATH_FN float win_sinf_fix(float x, float musl) {
    uint32_t key = fbits(x) & 0x7fffffffu;
    int32_t lo = 0;
    for (int32_t step = UCRT_SINF_STEP; step > 0; step = step >> 1) {
        int32_t probe = lo + step;
        if (probe < UCRT_SINF_COUNT && UCRT_SINF_KEYS[probe] <= key) lo = probe;
    }
    if (UCRT_SINF_KEYS[lo] != key) return musl;
    return ffrom(UCRT_SINF_VALUES[lo] ^ (fbits(x) & 0x80000000u));
}
ALTD_MATH_FN float win_cosf_fix(float x, float musl) {
    uint32_t key = fbits(x) & 0x7fffffffu;
    int32_t lo = 0;
    for (int32_t step = UCRT_COSF_STEP; step > 0; step = step >> 1) {
        int32_t probe = lo + step;
        if (probe < UCRT_COSF_COUNT && UCRT_COSF_KEYS[probe] <= key) lo = probe;
    }
    if (UCRT_COSF_KEYS[lo] != key) return musl;
    return ffrom(UCRT_COSF_VALUES[lo]);
}

// Every x.
ALTD_MATH_FN float win_sinf_fix_all(float x, float musl) {
    uint32_t key = fbits(x) & 0x7fffffffu;
    if (key < WIN_TRIG_LIMIT) return win_sinf_fix(x, musl);
    int32_t lo = 0;
    for (int32_t step = UCRT_SINF_WIDE_STEP; step > 0; step = step >> 1) {
        int32_t probe = lo + step;
        if (probe < UCRT_SINF_WIDE_COUNT && UCRT_SINF_WIDE_KEYS[probe] <= key) lo = probe;
    }
    if (UCRT_SINF_WIDE_KEYS[lo] != key) return musl;
    return ffrom(UCRT_SINF_WIDE_VALUES[lo] ^ (fbits(x) & 0x80000000u));
}
ALTD_MATH_FN float win_cosf_fix_all(float x, float musl) {
    uint32_t key = fbits(x) & 0x7fffffffu;
    if (key < WIN_TRIG_LIMIT) return win_cosf_fix(x, musl);
    int32_t lo = 0;
    for (int32_t step = UCRT_COSF_WIDE_STEP; step > 0; step = step >> 1) {
        int32_t probe = lo + step;
        if (probe < UCRT_COSF_WIDE_COUNT && UCRT_COSF_WIDE_KEYS[probe] <= key) lo = probe;
    }
    if (UCRT_COSF_WIDE_KEYS[lo] != key) return musl;
    return ffrom(UCRT_COSF_WIDE_VALUES[lo]);
}
#endif
