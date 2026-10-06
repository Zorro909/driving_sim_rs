// Bit-exact device ports of the CPU math in src/math/native_math.rs,
// src/math/godot_math.rs (managed_sin_cos), src/math/double_math.rs and src/nn/network.rs.
// Build with -ffp-contract=off: a fused multiply-add rounds once and breaks
// bit equality. Arguments outside the ported domains set an `err` bit instead
// of taking the rare large-argument paths of the CPU code.
#pragma once
#include "compat.h"
#include <cstdint>
#include "double_tables.h"
#include "engine_exceptions.h"

namespace altd {

enum : uint32_t {
    ERR_MANAGED_LARGE = 1u << 0,  // managed_sin_cos |x| > 10000 (sine_large not ported)
    ERR_NATIVE_LARGE = 1u << 1,   // musl sinf/cosf |x| >= 0x4dc90fdb (Payne-Hanek not ported)
    ERR_ENGINE_DOMAIN = 1u << 2,  // engine_sin_cos |x| > ENGINE_DOMAIN (not validated against x87)
};

__device__ __host__ inline uint32_t fbits(float x) { return __builtin_bit_cast(uint32_t, x); }
__device__ __host__ inline float ffrom(uint32_t b) { return __builtin_bit_cast(float, b); }
__device__ __host__ inline uint64_t dbits(double x) { return __builtin_bit_cast(uint64_t, x); }
__device__ __host__ inline double dfrom(uint64_t b) { return __builtin_bit_cast(double, b); }

// ---- musl sinf/cosf (native_math.rs) ------------------------------------

__device__ inline float sin_kernel(double x) {
    double z = x * x;
    const double s1 = -0.16666666666666666;
    if (x > -7.8175831586122513e-3 && x < 7.8175831586122513e-3) return (float)(x * (1.0 + s1 * z));
    double w = z * z;
    double r = -0.0001984126984126984 + z * 2.7557319223985893e-6;
    double s = z * x;
    return (float)((x + s * (s1 + z * 0.008333333333333333)) + s * w * r);
}
__device__ inline float cos_kernel(double x) {
    double z = x * x;
    if (x > -7.816314697265625e-3 && x < 7.816314697265625e-3) return (float)(1.0 - 0.5 * z);
    return (float)(1.0 + z * (-0.5 + z * (0.041666666666666664 + z * (-0.001388888888888889
        + z * (2.48015873015873e-5 + z * -2.755731922398589e-7)))));
}
// Medium reduction; the large path is flagged.
__device__ inline int reduce_medium(float x, double& y, uint32_t& err) {
    if ((fbits(x) & 0x7fffffffu) >= 0x4dc90fdbu) err |= ERR_NATIVE_LARGE;
    const double toint = 1.5 / 2.220446049250313e-16;
    double n = ((double)x * 6.36619772367581382433e-1 + toint) - toint;
    double xd = (double)x;
    y = xd - n * 1.57079631090164184570 - n * 1.58932547735281966916e-8;
    const double quarter = (double)0.7853981852531433f;  // f32 FRAC_PI_4
    if (y < -quarter) { n -= 1.0; y = xd - n * 1.57079631090164184570 - n * 1.58932547735281966916e-8; }
    else if (y > quarter) { n += 1.0; y = xd - n * 1.57079631090164184570 - n * 1.58932547735281966916e-8; }
    return (int)n;
}
__device__ inline float native_sin(float x, uint32_t& err) {
    uint32_t bits = fbits(x) & 0x7fffffffu;
    bool negative = signbit(x);
    double a = (double)x;
    const double p = 1.5707963267948966;
    if (bits <= 0x3f490fdau) return bits < 0x39800000u ? x : sin_kernel(a);
    if (bits <= 0x407b53d1u) {
        if (bits <= 0x4016cbe3u) return negative ? -cos_kernel(a + p) : cos_kernel(a - p);
        return sin_kernel(negative ? -(a + 2.0 * p) : -(a - 2.0 * p));
    }
    if (bits <= 0x40e231d5u) {
        if (bits <= 0x40afeddfu) return negative ? cos_kernel(a + 3.0 * p) : -cos_kernel(a - 3.0 * p);
        return sin_kernel(negative ? a + 4.0 * p : a - 4.0 * p);
    }
    if (!isfinite(x)) return altd_invalid(x);
    double y;
    int n = reduce_medium(x, y, err);
    switch (n & 3) { case 0: return sin_kernel(y); case 1: return cos_kernel(y); case 2: return sin_kernel(-y); default: return -cos_kernel(y); }
}
__device__ inline float native_cos(float x, uint32_t& err) {
    uint32_t bits = fbits(x) & 0x7fffffffu;
    bool negative = signbit(x);
    double a = (double)x;
    const double p = 1.5707963267948966;
    if (bits <= 0x3f490fdau) return bits < 0x39800000u ? 1.0f : cos_kernel(a);
    if (bits <= 0x407b53d1u) {
        if (bits > 0x4016cbe3u) return -cos_kernel(negative ? a + 2.0 * p : a - 2.0 * p);
        return sin_kernel(negative ? a + p : p - a);
    }
    if (bits <= 0x40e231d5u) {
        if (bits > 0x40afeddfu) return cos_kernel(negative ? a + 4.0 * p : a - 4.0 * p);
        return sin_kernel(negative ? -a - 3.0 * p : a - 3.0 * p);
    }
    if (!isfinite(x)) return altd_invalid(x);
    double y;
    int n = reduce_medium(x, y, err);
    switch (n & 3) { case 0: return cos_kernel(y); case 1: return sin_kernel(-y); case 2: return -cos_kernel(y); default: return sin_kernel(y); }
}

__device__ inline float native_atan(float x) {
    bool negative = signbit(x);
    uint32_t bits = fbits(x) & 0x7fffffffu;
    const float high[4] = {ffrom(0x3eed6338u), ffrom(0x3f490fdau), ffrom(0x3f7b985eu), ffrom(0x3fc90fdau)};
    const float low[4] = {ffrom(0x31ac3769u), ffrom(0x33222168u), ffrom(0x33140fb4u), ffrom(0x33a22168u)};
    if (bits >= 0x4c800000u) return isnan(x) ? x : copysignf(high[3], x);
    int id;
    if (bits < 0x3ee00000u) {
        if (bits < 0x39800000u) return x;
        id = -1;
    } else {
        x = fabsf(x);
        if (bits < 0x3f980000u) {
            if (bits < 0x3f300000u) { id = 0; x = (2.0f * x - 1.0f) / (2.0f + x); }
            else { id = 1; x = (x - 1.0f) / (x + 1.0f); }
        } else if (bits < 0x401c0000u) { id = 2; x = (x - 1.5f) / (1.0f + 1.5f * x); }
        else { id = 3; x = -1.0f / x; }
    }
    float z = x * x;
    float w = z * z;
    float s1 = z * (3.3333328366e-01f + w * (1.4253635705e-01f + w * 6.1687607318e-02f));
    float s2 = w * (-1.9999158382e-01f + w * -1.0648017377e-01f);
    if (id < 0) return x - x * (s1 + s2);
    float r = high[id] - ((x * (s1 + s2) - low[id]) - x);
    return negative ? -r : r;
}
__device__ inline float native_atan2(float y, float x) {
    const float pi = 3.14159274101257324f;
    const float pi_lo = ffrom(0xb3bbbd2eu);
    if (isnan(x) || isnan(y)) return altd_nan_operand(y, x);  // the CPU's `x + y` returns y when both are NaN
    uint32_t ix = fbits(x), iy = fbits(y);
    if (ix == 0x3f800000u) return native_atan(y);
    uint32_t m = ((iy >> 31) & 1u) | ((ix >> 30) & 2u);
    ix &= 0x7fffffffu; iy &= 0x7fffffffu;
    if (iy == 0) return m <= 1 ? y : (m == 2 ? pi : -pi);
    if (ix == 0) return copysignf(pi / 2.0f, y);
    if (ix == 0x7f800000u) {
        if (iy == 0x7f800000u) {
            const float v[4] = {pi / 4.0f, -pi / 4.0f, 3.0f * pi / 4.0f, -3.0f * pi / 4.0f};
            return v[m];
        }
        const float v[4] = {0.0f, -0.0f, pi, -pi};
        return v[m];
    }
    if (ix + (26u << 23) < iy || iy == 0x7f800000u) return copysignf(pi / 2.0f, y);
    float z = ((m & 2) && iy + (26u << 23) < ix) ? 0.0f : native_atan(fabsf(y / x));
    switch (m) { case 0: return z; case 1: return -z; case 2: return pi - (z - pi_lo); default: return (z - pi_lo) - pi; }
}

// ---- .NET MathF.SinCos (godot_math.rs managed_sin_cos) --------------------

// Straight-line for |value| <= 10000; larger arguments are flagged.
__device__ inline float packed_sine(float value, uint32_t& err) {
    float a = fabsf(value);
    if (!(a <= 10000.0f)) err |= ERR_MANAGED_LARGE;
    float rounded = a * ffrom(0x3ea2f983u) + ffrom(0x4b400000u);
    float n = rounded - ffrom(0x4b400000u);
    float r = a - n * ffrom(0x40490000u);
    r -= n * ffrom(0x3a7da000u);
    r -= n * ffrom(0x34222000u);
    r -= n * ffrom(0x2cb4611au);
    float square = r * r;
    r = ffrom(fbits(r) ^ (fbits(rounded) << 31));
    float p = ffrom(0x362edef8u) * square;
    p += ffrom(0xb94fb7ffu);
    p *= square;
    p += ffrom(0x3c088766u);
    p *= square;
    p += ffrom(0xbe2aaaa6u);
    float result = r + (square * p) * r;
    return ffrom(fbits(result) ^ (fbits(value) & 0x80000000u));
}
__device__ inline void managed_sin_cos(float angle, float& s, float& c, uint32_t& err) {
    if (isnan(angle)) { s = angle; c = ffrom((fbits(angle) & 0x7fffffffu) | 0x00400000u); return; }
    float a = fabsf(angle);
    if (!(a >= ffrom(0x39000000u))) { s = angle; c = 1.0f - a * a * 0.5f; return; }
    s = packed_sine(angle, err);
    c = packed_sine(a + 1.57079637050628662f, err);
}

// ---- x87 FSINCOS (native_math.rs engine_sin_cos) --------------------------
// fdlibm kernels in f64 after a Cody-Waite reduction; for float inputs up to
// ENGINE_DOMAIN the product n * PIO2_1 and the first subtraction are exact.
// The rounded float results equal x87 except for the inputs in
// engine_exceptions.h, found by an exhaustive scan (examples/gpu_engine_scan.rs).

__device__ inline void engine_sin_cos_raw(float xf, float& s, float& c) {
    double x = (double)xf;
    double n = rint(x * 6.36619772367581382433e-01);
    double y = (x - n * 1.57079632673412561417e+00) - n * 6.07710050650619224932e-11;
    y = n == 0.0 ? x : y;  // keeps -0
    double z = y * y, w = z * z;
    double r = 8.33333333332248946124e-03 + z * (-1.98412698298579493134e-04 + z * 2.75573137070700676789e-06)
        + z * w * (-2.50507602534068634195e-08 + z * 1.58969099521155010221e-10);
    // fdlibm returns tiny arguments unchanged, which also keeps the sign of -0.
    double sy = fabs(y) < 0x1p-27 ? y : y + z * y * (-1.66666666666666324348e-01 + z * r);
    double rc = z * (4.16666666666666019037e-02 + z * (-1.38888888888741095749e-03 + z * 2.48015872894767294178e-05))
        + w * w * (-2.75573143513906633035e-07 + z * (2.08757232129817482790e-09 + z * -1.13596475577881948265e-11));
    double hz = 0.5 * z, one_minus = 1.0 - hz;
    double cy = one_minus + (((1.0 - one_minus) - hz) + z * rc);
    int q = (int)n & 3;
    double sv = (q & 1) ? cy : sy, cv = (q & 1) ? sy : cy;
    sv = (q & 2) ? -sv : sv;
    cv = ((q + 1) & 2) ? -cv : cv;
    s = (float)sv;
    c = (float)cv;
}

constexpr float ENGINE_DOMAIN = 16.0f;

// Branch-free binary search over the sorted exception keys.
__device__ inline void engine_sin_cos(float x, float& s, float& c, uint32_t& err) {
    if (!(fabsf(x) <= ENGINE_DOMAIN)) err |= ERR_ENGINE_DOMAIN;
    engine_sin_cos_raw(x, s, c);
    if (ENGINE_EXCEPTION_COUNT == 0) return;
    uint32_t key = fbits(x);
    int lo = 0;
    for (int step = ENGINE_EXCEPTION_STEP; step > 0; step >>= 1) {
        int probe = lo + step;
        lo = (probe < ENGINE_EXCEPTION_COUNT && ENGINE_EXCEPTIONS[probe][0] <= key) ? probe : lo;
    }
    bool hit = ENGINE_EXCEPTIONS[lo][0] == key;
    s = hit ? ffrom(ENGINE_EXCEPTIONS[lo][1]) : s;
    c = hit ? ffrom(ENGINE_EXCEPTIONS[lo][2]) : c;
}

// ---- musl/Arm double exp and pow (double_math.rs) -------------------------

__device__ inline double exp_special(double tmp, uint64_t bits, uint64_t k) {
    if ((k & 0x80000000ull) == 0) {
        bits -= 1009ull << 52;
        double scale = dfrom(bits);
        return dfrom(2032ull << 52) * (scale + scale * tmp);
    }
    bits += 1022ull << 52;
    double scale = dfrom(bits);
    double y = scale + scale * tmp;
    if (fabs(y) < 1.0) {
        double one = y < 0.0 ? -1.0 : 1.0;
        double lo = scale - y + scale * tmp;
        double hi = one + y;
        lo = one - hi + y + lo;
        y = (hi + lo) - one;
        if (y == 0.0) y = dfrom(bits & (1ull << 63));
    }
    return 2.2250738585072014e-308 * y;
}
__device__ inline double exp_inner(double x, double xtail, bool negative) {
    uint64_t exponent = (dbits(x) >> 52) & 0x7ff;
    if (exponent < 969) return negative ? -(1.0 + x) : 1.0 + x;
    if (exponent >= 1033) {
        double y = signbit(x) ? 0.0 : INFINITY;
        return negative ? -y : y;
    }
    double kd = round(INV_LN2_N * x);
    uint64_t ki = (uint64_t)(int64_t)kd;
    double r = (x + kd * NEG_LN2_HI_N + kd * NEG_LN2_LO_N) + xtail;
    int idx = 2 * (int)(ki % 128);
    uint64_t bias = negative ? (0x800ull << 7) : 0;
    uint64_t top = (ki + bias) << 45;
    double tail = dfrom(EXP_TAB[idx]);
    uint64_t bits = EXP_TAB[idx + 1] + top;
    double r2 = r * r;
    double tmp = tail + r + r2 * (EXP_POLY[0] + r * EXP_POLY[1]) + r2 * r2 * (EXP_POLY[2] + r * EXP_POLY[3]);
    if (exponent >= 1032) return exp_special(tmp, bits, ki);
    double scale = dfrom(bits);
    return scale + scale * tmp;
}
__device__ inline double dexp(double x) {
    if (isnan(x)) return x;
    return exp_inner(x, 0.0, false);
}
__device__ inline void pow_log(uint64_t ix, double& out_hi, double& out_lo) {
    uint64_t tmp = ix - 0x3fe6955500000000ull;
    int i = (int)((tmp >> 45) % 128);
    double kd = (double)((int64_t)tmp >> 52);
    uint64_t iz = ix - (tmp & (0xfffull << 52));
    double z = dfrom(iz);
    double invc = POW_TAB[i][0], logc = POW_TAB[i][1], logctail = POW_TAB[i][2];
    double zhi = dfrom((iz + (1ull << 31)) & (~0ull << 32));
    double zlo = z - zhi;
    double rhi = zhi * invc - 1.0;
    double rlo = zlo * invc;
    double r = rhi + rlo;
    double t1 = kd * LN2_HI + logc;
    double t2 = t1 + r;
    double lo1 = kd * LN2_LO + logctail;
    double lo2 = t1 - t2 + r;
    const double* a = POW_POLY;
    double ar = a[0] * r;
    double ar2 = r * ar;
    double ar3 = r * ar2;
    double arhi = a[0] * rhi;
    double arhi2 = rhi * arhi;
    double hi = t2 + arhi2;
    double lo3 = rlo * (ar + arhi);
    double lo4 = t2 - hi + arhi2;
    double p = ar3 * (a[1] + r * a[2] + ar2 * (a[3] + r * a[4] + ar2 * (a[5] + r * a[6])));
    double lo = lo1 + lo2 + lo3 + lo4 + p;
    double y = hi + lo;
    out_hi = y;
    out_lo = hi - y + lo;
}
__device__ inline uint32_t integer_kind(uint64_t bits) {
    uint64_t e = (bits >> 52) & 0x7ff;
    if (e < 1023) return 0;
    if (e > 1075) return 2;
    uint64_t bit = 1ull << (1075 - e);
    return (bits & (bit - 1)) ? 0 : ((bits & bit) ? 1 : 2);
}
__device__ inline double dpow(double x, double y) {
    // pow(1, y) is 1 for every y, NaN included, as the paths below also give;
    // answering first skips the logarithm, e.g. for godot_ease(0, curve).
    if (dbits(x) == dbits(1.0)) return 1.0;
    const uint64_t SIGN = 1ull << 63;
    uint64_t ix = dbits(x), iy = dbits(y);
    uint32_t topx = (uint32_t)(ix >> 52), topy = (uint32_t)(iy >> 52);
    bool negative = false;
    if (topx - 1u >= 0x7feu || (topy & 0x7ffu) - 0x3beu >= 0x80u) {
        if (y == 0.0 || !isfinite(y)) {
            if (y == 0.0) return 1.0;
            if (x == 1.0) return 1.0;
            if (isnan(x) || isnan(y)) return altd_nan_operand(x, y);
            if (fabs(x) == 1.0) return 1.0;
            if ((fabs(x) < 1.0) == !signbit(y)) return 0.0;
            return y * y;
        }
        if (x == 0.0 || !isfinite(x)) {
            double x2 = isnan(x) ? altd_nan_operand(x, x) : x * x;
            if (signbit(x) && integer_kind(iy) == 1) x2 = dfrom(dbits(x2) ^ (1ull << 63));  // -x2, NaN included
            return signbit(y) && !isnan(x2) ? 1.0 / x2 : x2;
        }
        if (signbit(x)) {
            uint32_t kind = integer_kind(iy);
            if (kind == 0) return dfrom(0xfff8000000000000ull);
            negative = kind == 1;
            ix &= ~SIGN; topx &= 0x7ff;
        }
        if ((topy & 0x7ffu) - 0x3beu >= 0x80u) {
            if (ix == dbits(1.0)) return 1.0;
            if ((topy & 0x7ffu) < 0x3beu) return ix > dbits(1.0) ? 1.0 + y : 1.0 - y;
            return (ix > dbits(1.0)) == (topy < 0x800u) ? INFINITY : 0.0;
        }
        if (topx == 0) ix = (dbits(x * 4503599627370496.0) & ~SIGN) - (52ull << 52);
    }
    double hi, lo;
    pow_log(ix, hi, lo);
    double yhi = dfrom(iy & (~0ull << 27));
    double ylo = y - yhi;
    double lhi = dfrom(dbits(hi) & (~0ull << 27));
    double llo = hi - lhi + lo;
    double ehi = yhi * lhi;
    double elo = ylo * lhi + y * llo;
    return exp_inner(ehi, elo, negative);
}

// ---- Network activation (network.rs game_expm1 / game_tanh) ---------------

__device__ inline double game_expm1(double x) {
    uint32_t high = (uint32_t)(dbits(x) >> 32) & 0x7fffffffu;
    bool negative = signbit(x);
    if (high >= 0x4043687au) {
        if (isnan(x)) return x;
        if (isinf(x)) return negative ? -1.0 : x;
        if (negative) return -1.0;
        if (x > 7.09782712893383973096e2) return INFINITY;
    }
    int k; double c;
    if (high > 0x3fd62e42u) {
        double hi, lo;
        if (high < 0x3ff0a2b2u) {
            if (!negative) { hi = x - 6.93147180369123816490e-1; lo = 1.90821492927058770002e-10; k = 1; }
            else { hi = x + 6.93147180369123816490e-1; lo = -1.90821492927058770002e-10; k = -1; }
        } else {
            k = (int)(1.44269504088896338700 * x + (negative ? -0.5 : 0.5));
            hi = x - (double)k * 6.93147180369123816490e-1;
            lo = (double)k * 1.90821492927058770002e-10;
        }
        x = hi - lo; c = (hi - x) - lo;
    } else if (high < 0x3c900000u) {
        return x;
    } else { k = 0; c = 0.0; }
    double hfx = 0.5 * x;
    double hxs = x * hfx;
    double r1 = 1.0 + hxs * (-3.33333333333331316428e-2 + hxs * (1.58730158725481460165e-3 + hxs * (-7.93650757867487942473e-5
        + hxs * (4.00821782732936239552e-6 + hxs * (-2.01099218183624371326e-7)))));
    double t = 3.0 - r1 * hfx;
    double e = hxs * ((r1 - t) / (6.0 - x * t));
    if (k == 0) return x - (x * e - hxs);
    e = x * (e - c) - c;
    e -= hxs;
    if (k == -1) return 0.5 * (x - e) - 0.5;
    if (k == 1) return x < -0.25 ? -2.0 * (e - (x + 0.5)) : 1.0 + 2.0 * (x - e);
    double twopk = dfrom((uint64_t)(0x3ff + k) << 52);
    if (k < 0 || k > 56) {
        double y = x - e + 1.0;
        return (k == 1024 ? y * 2.0 * dfrom(0x7fe0000000000000ull) : y * twopk) - 1.0;
    }
    double inverse = dfrom((uint64_t)(0x3ff - k) << 52);
    return k < 20 ? (x - e + (1.0 - inverse)) * twopk : (x - (e + inverse) + 1.0) * twopk;
}
// The scalar tanh, one branch per range. A wave whose lanes fall in different
// ranges executes every taken branch in turn, and the three expm1 call sites
// are inlined separately, so the double-precision work (1/32 rate on RDNA3)
// runs up to three times per layer. Kept for comparison: -DALTD_TANH_BRANCHY.
__device__ inline double game_tanh_branchy(double value) {
    double x = fabs(value);
    uint32_t high = (uint32_t)(dbits(x) >> 32);
    double t;
    if (high > 0x3fe193eau) {
        if (high > 0x40340000u) t = 1.0 - 0.0 / x;
        else { double u = game_expm1(2.0 * x); t = 1.0 - 2.0 / (u + 2.0); }
    } else if (high > 0x3fd058aeu) {
        double u = game_expm1(2.0 * x); t = u / (u + 2.0);
    } else if (high >= 0x00100000u) {
        double u = game_expm1(-2.0 * x); t = -u / (u + 2.0);
    } else t = x;
    return signbit(value) ? -t : t;
}

// ---- branch-free game_tanh ------------------------------------------------
// One expm1 evaluation per lane on a selected argument, then selects, so the
// lanes of a wave never serialize the tanh ranges or the expm1 cases. Each
// lane's selected expression is exactly what the scalar code evaluates on
// that path; unselected values are computed and dropped. This is the scalar
// form of `tanh4`/`expm1_tanh` in src/nn/network_simd.rs, whose CPU tests
// compare it with the scalar functions bit for bit over every branch,
// threshold and special value.

// 2^k by exponent-field arithmetic, as the scalar code.
__device__ inline double pow2_of(int32_t k) {
    return dfrom((uint64_t)(int64_t)(k + 0x3ff) << 52);
}

// `game_expm1` for the arguments `game_tanh` selects: finite, below 709.78
// when positive, so the overflow block of the scalar code never returns early.
// The integer k is kept as an integer for its comparisons and powers of two
// (the scalar code's `k as f64` only enters the two reduction products).
__device__ inline double expm1_lane(double x) {
    uint32_t h = (uint32_t)(dbits(x) >> 32) & 0x7fffffffu;
    bool negative = signbit(x);
    bool reduce = h > 0x3fd62e42u, tiny = h < 0x3c900000u, near = h < 0x3ff0a2b2u;
    // k = +-1 near ln 2, else trunc(x / ln 2 +- 0.5); 0 without reduction.
    double rounded = 1.44269504088896338700 * x + (negative ? -0.5 : 0.5);
    int32_t ki = reduce ? (near ? (negative ? -1 : 1) : __double2int_rz(rounded)) : 0;
    double k = (double)ki;
    // For k = +-1, x - k*ln2_hi and k*ln2_lo equal the scalar x -+ ln2_hi, +-ln2_lo exactly.
    double hi = x - k * 6.93147180369123816490e-1;
    double lo = k * 1.90821492927058770002e-10;
    double xr = reduce ? hi - lo : x;
    double c = reduce ? (hi - xr) - lo : 0.0;
    double hfx = 0.5 * xr;
    double hxs = xr * hfx;
    double p = hxs * -2.01099218183624371326e-7;
    p = hxs * (4.00821782732936239552e-6 + p);
    p = hxs * (-7.93650757867487942473e-5 + p);
    p = hxs * (1.58730158725481460165e-3 + p);
    p = hxs * (-3.33333333333331316428e-2 + p);
    double r1 = 1.0 + p;
    double t = 3.0 - r1 * hfx;
    double e = hxs * ((r1 - t) / (6.0 - xr * t));
    double k0 = xr - (xr * e - hxs);
    e = (xr * (e - c) - c) - hxs;
    double xme = xr - e;
    double km1 = 0.5 * xme - 0.5;
    double k1 = xr < -0.25 ? -2.0 * (e - (xr + 0.5)) : 1.0 + 2.0 * xme;
    double twopk = pow2_of(ki), inverse = pow2_of(-ki);
    // k < 0 || k > 56 (k == 1024 needs x > 709, outside the domain).
    bool far = ki < 0 || ki > 56;
    double far_value = (xme + 1.0) * twopk - 1.0;
    double below20 = (xme + (1.0 - inverse)) * twopk;
    double from20 = ((xr - (e + inverse)) + 1.0) * twopk;
    double general = far ? far_value : (ki < 20 ? below20 : from20);
    double result = ki == 0 ? k0 : (ki == -1 ? km1 : (ki == 1 ? k1 : general));
    return tiny ? x : result;
}

__device__ inline double game_tanh(double value) {
#ifdef ALTD_TANH_BRANCHY
    return game_tanh_branchy(value);
#else
    double x = fabs(value);
    uint32_t h = (uint32_t)(dbits(x) >> 32);
    // Above 0x40340000 (|x| > 20, infinities, NaN) the scalar path stays a
    // branch: NaN must keep its payload, and a NaN routed through the selects
    // below can lose it on this hardware (fabs folded into the 32-bit select
    // as a float source modifier, which quiets or canonicalizes under IEEE
    // mode). Finite lanes never take it during inference.
    if (h > 0x40340000u) {
        double t = 1.0 - 0.0 / x;
        return signbit(value) ? -t : t;
    }
    bool big = h > 0x3fe193eau, mid = h > 0x3fd058aeu, normal = h > 0x000fffffu;
    double t = expm1_lane(mid ? 2.0 * x : -2.0 * x);
    double d = t + 2.0;
    // The range's own division: 2 / (t + 2) for the large range, t / (t + 2)
    // otherwise; the small range's (-t) / (t + 2) is the exact negation.
    double q = (big ? 2.0 : t) / d;
    double result = big ? 1.0 - q : (mid ? q : (normal ? -q : x));
    return signbit(value) ? -result : result;
#endif
}

// ---- math profiles (crate::math::profile) ----------------------------------
// The Windows UCRT functions of math/kernels/ucrt.h. World::math_profile picks a
// profile for the whole simulation, so the branches below never diverge.

#define ALTD_MATH_FN __device__ inline
#define ALTD_MATH_TABLE __device__ constexpr
#define ALTD_MATH_CONST constexpr
#include "../../math/kernels/ucrt.h"

// MathProfile::index.
enum : uint32_t { MATH_PROTON = 0, MATH_WIN10_FMA3 = 1, MATH_WIN11_FMA3 = 2 };

// sinf and cosf are musl's with the Windows exceptions. Inputs beyond
// WIN_TRIG_LIMIT are flagged by the musl reduction already.
__device__ inline float profile_sin(uint32_t math, float x, uint32_t& err) {
    float musl = native_sin(x, err);
    return math == MATH_PROTON ? musl : win_sinf_fix(x, musl);
}
__device__ inline float profile_cos(uint32_t math, float x, uint32_t& err) {
    float musl = native_cos(x, err);
    return math == MATH_PROTON ? musl : win_cosf_fix(x, musl);
}
// The profile is uniform, so these branch rather than select (WGSL's select
// evaluates both operands).
__device__ inline float profile_atan2(uint32_t math, float y, float x) {
    if (math == MATH_PROTON) return native_atan2(y, x);
    return win_atan2f(y, x);
}
__device__ inline double profile_exp(uint32_t math, double x) {
    if (math == MATH_PROTON) return dexp(x);
    if (math == MATH_WIN10_FMA3) return win10_exp(x);
    return win11_exp(x);
}
__device__ inline double profile_pow(uint32_t math, double x, double y) {
    if (math == MATH_PROTON) return dpow(x, y);
    if (math == MATH_WIN10_FMA3) return win10_pow(x, y);
    return win11_pow(x, y);
}
__device__ inline double profile_tanh(uint32_t math, double x) {
    if (math == MATH_PROTON) return game_tanh(x);
    if (math == MATH_WIN10_FMA3) return win10_tanh(x);
    return win11_tanh(x);
}

}  // namespace altd
