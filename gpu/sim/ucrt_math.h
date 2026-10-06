// Device port of src/math/ucrt_math.rs: the Windows UCRT (ucrtbase.dll) FMA3 sinf, cosf, atan2f,
// exp, tanh and pow. The Rust version is verified against the live DLL; keep the two in step.
// Fused multiply-adds are explicit fma() calls; build with -ffp-contract=off / -fmad=false so
// nothing else fuses. Arguments outside the ported domains of sinf and cosf set `large` and return
// NaN, like the other ported math (the CPU takes its slow path instead).
#pragma once
#include "compat.h"
#include <cstdint>
#include "ucrt_tables.h"

namespace altd {
namespace ucrt {

__device__ inline double D(uint64_t bits) { return __builtin_bit_cast(double, bits); }
__device__ inline uint64_t B(double x) { return __builtin_bit_cast(uint64_t, x); }
__device__ inline float F32(uint32_t bits) { return __builtin_bit_cast(float, bits); }
__device__ inline double fm(double a, double b, double c) { return fma(a, b, c); }

constexpr uint32_t QNAN32 = 0x7FC00000u;
constexpr uint32_t INDEFINITE32 = 0xFFC00000u;
constexpr uint64_t QNAN64 = 0x7FF8000000000000ull;
constexpr uint64_t INDEFINITE64 = 0xFFF8000000000000ull;
constexpr uint64_t INF64 = 0x7FF0000000000000ull;
constexpr uint64_t SIGN64 = 0x8000000000000000ull;
constexpr uint64_t ABS64 = 0x7FFFFFFFFFFFFFFFull;
constexpr uint64_t QUIET64 = 0x0008000000000000ull;

constexpr double S1 = -0.16666666666666666, S2 = 0.008333333333333333, S3 = -0.0001984126984126984, S4 = 2.7557319223985893E-06;
constexpr double C1 = -0.5, C2 = 0.041666666666666664, C3 = -0.0013888888888888887, C4 = 2.4801587301587298E-05, C5 = -2.755731922398589E-07;

__device__ inline double sin_kernel(double r) {
    double x2 = r * r;
    double p = fm(x2, S4, S3);
    p = fm(x2, p, S2);
    p = fm(x2, p, S1);
    return fm(p, r * x2, r);
}

__device__ inline double cos_kernel_sinf(double r) {
    double x2 = r * r;
    double a = fm(x2, C1, 1.0);
    double p = fm(x2, C5, C4);
    p = fm(x2, p, C3);
    p = fm(x2, p, C2);
    return fm(p, x2 * x2, a);
}

__device__ inline double cos_kernel_cosf(double r) {
    double x2 = r * r;
    double a = 1.0 - x2 * 0.5;
    double p = fm(x2, C5, C4);
    p = fm(x2, p, C3);
    p = fm(x2, p, C2);
    return fm(p, x2 * x2, a);
}

// sinf/cosf of an infinity or NaN: the indefinite NaN, or the input NaN made quiet.
__device__ inline uint32_t non_finite_trig(uint32_t bits) { return (bits & 0x007FFFFFu) ? (bits | 0x00400000u) : INDEFINITE32; }

__device__ inline float sinf_(float xf, bool& large) {
    uint32_t ib = __builtin_bit_cast(uint32_t, xf);
    if ((ib & 0x7F800000u) == 0x7F800000u) return F32(non_finite_trig(ib));
    double x = (double)xf;
    uint64_t ab = B(x) & ABS64;
    if (ab <= 0x3FE921FB54442D18ull) {
        if (ab >= 0x3F80000000000000ull) return (float)sin_kernel(x);
        if (ab >= 0x3F20000000000000ull) {
            double t = x * x * x;
            return (float)fm(-t, 0.16666666666666666, x);
        }
        return xf;
    }
    if (ab >= 0x4170008AC0000000ull) { large = true; return F32(QNAN32); }
    double ax = D(ab);
    double t0 = fm(ax, 0.6366197723675814, 0.5);
    int n = __double2int_rz(t0);
    int q = n & 3;
    double nd = (double)n;
    double r1 = fm(-nd, 1.5707963267341256, ax);
    double r = r1 - nd * 6.077100506506192E-11;
    double res = (q & 1) == 0 ? sin_kernel(r) : cos_kernel_sinf(r);
    bool flip = (q == 0 || q == 1) ^ ((ib >> 31) == 0);
    if (flip) res = -res;
    return (float)res;
}

__device__ inline float cosf_(float xf, bool& large) {
    uint32_t ib = __builtin_bit_cast(uint32_t, xf);
    if ((ib & 0x7F800000u) == 0x7F800000u) return F32(non_finite_trig(ib));
    double x = (double)xf;
    uint64_t ab = B(x) & ABS64;
    if (ab <= 0x3FE921FB54442D18ull) {
        if (ab >= 0x3F80000000000000ull) return (float)cos_kernel_cosf(x);
        if (ab >= 0x3F20000000000000ull) return (float)fm(-x, x * 0.5, 1.0);
        return 1.0f;
    }
    if (ab >= 0x41E921FB60000000ull) { large = true; return F32(QNAN32); }
    double ax = D(ab);
    double t0 = fm(ax, 0.6366197723675814, 0.5);
    int n = __double2int_rz(t0);
    int q = n & 3;
    double nd = (double)n;
    double r1 = fm(-nd, 1.5707963267341256, ax);
    double r = r1 - nd * 6.077100506506192E-11;
    double res = (q & 1) != 0 ? sin_kernel(r) : cos_kernel_cosf(r);
    if ((((q + 1) >> 1) & 1) != 0) res = -res;
    return (float)res;
}

__device__ inline float atan2f_(float yf, float xf) {
    double x7 = (double)xf, y6 = (double)yf;
    uint64_t xb = B(x7), yb = B(y6);
    int exp_x = (int)((xb >> 52) & 0x7FF), exp_y = (int)((yb >> 52) & 0x7FF);
    int esi = exp_y - exp_x;
    uint64_t ax = xb & ABS64, ay = yb & ABS64;
    bool xneg = (int64_t)xb < 0, yneg = (int64_t)yb < 0;
    if (ax > INF64 || ay > INF64) {
        float nan = ax > INF64 ? xf : yf;
        return F32(__builtin_bit_cast(uint32_t, nan) | 0x00400000u);
    }
    if (ay == 0) {
        if (xneg) return yneg ? -3.1415927f : 3.1415927f;
        return (float)y6;
    }
    if (ax == 0 && yneg) return -1.5707964f;
    if (esi > 0x1A) return yneg ? -1.5707964f : 1.5707964f;
    if (esi < -13 && !xneg) {
        if (esi < -150) return yneg ? -0.0f : 0.0f;
        if (esi < -126) {
            double v = D(0x4630000000000000ull) * y6 / x7;
            uint64_t vb = B(v);
            uint64_t sign = vb & SIGN64;
            uint64_t mag = vb & ABS64;
            uint32_t e = (uint32_t)(mag >> 52);
            uint64_t rbx;
            if (e > 0x64) {
                rbx = ((uint64_t)(e - 0x64) << 52) | (mag & 0x800FFFFFFFFFFFFFull);
            } else {
                uint64_t m = (mag & 0x801FFFFFFFFFFFFFull) | 0x10000000000000ull;
                int cnt = 0x65 - (int)e;
                if (cnt > 0x36) {
                    rbx = 0;
                } else {
                    cnt--;
                    m >>= cnt;
                    uint64_t carry = m & 1;
                    rbx = (m >> 1) + carry;
                }
            }
            rbx |= sign;
            return (float)D(rbx);
        }
        return (float)(y6 / x7);
    }
    if (esi < -26 && xneg) return yneg ? -3.1415927f : 3.1415927f;
    if (ay == INF64 && ax == INF64) {
        if (xneg) return yneg ? -2.3561945f : 2.3561945f;
        return yneg ? -0.7853982f : 0.7853982f;
    }
    double sx = xneg ? -x7 : x7;
    double sy = yneg ? -y6 : y6;
    bool swap = sy > sx;
    double small = swap ? sx : sy;
    double large = swap ? sy : sx;
    double t = small / large;
    double res;
    if (t > 0.0625) {
        double s256 = small * 256.0;
        double tt = t * 256.0 + 0.5;
        int n = __double2int_rz(tt);
        double nd = (double)n;
        double a = nd * small;
        double b = nd * large;
        double l256 = large * 256.0;
        double num = s256 - b;
        double den = a + l256;
        double u = num / den;
        double hi = u + D(ATAN2F_ATAN256[n - 16]);
        double u3 = u * u;
        u3 *= u;
        u3 *= 0.33333333333224097;
        res = hi - u3;
    } else if (0.0001 > t) {
        res = t;
    } else {
        double t2 = t * t;
        double p = 0.19999999999393223 - t2 * 0.1428571356180717;
        p *= t2;
        double t3 = t2 * t;
        double q = 0.3333333333333317 - p;
        q *= t3;
        res = t - q;
    }
    if (swap) res = 1.5707963267948966 - res;
    if (xneg) res = 3.141592653589793 - res;
    if (yneg) res = -res;
    return (float)res;
}

__device__ inline double exp_(double x) {
    uint64_t ab = B(x) & ABS64;
    if (ab >= INF64) {
        if (B(x) == INF64) return x;
        if (B(x) == 0xFFF0000000000000ull) return 0.0;
        return D(B(x) | QUIET64);
    }
    if (x > 709.782712893384) return D(INF64);
    if (x < -744.0346068132731) return 0.0;
    if (ab <= 0x3E50000000000000ull) return x + 1.0;
    double x1 = x * 92.33248261689366;
    double nd = trunc(x1);
    int nn = __double2int_rz(x1);
    double r1 = fm(nd, -0.010830424696223417, x);
    double r2 = nd * -2.5728046223276688E-14;
    double r = r2 + r1;
    int j = nn & 0x3F;
    int k = nn >> 6;
    double p = fm(0.001388888888888889, r, 0.008333333333333333);
    p = fm(p, r, 0.041666666666666664);
    p = fm(p, r, 0.16666666666666666);
    p = fm(p, r, 0.5);
    double rr = r * r;
    double e = fm(rr, p, r);
    double t = e * D(EXP_MUL[j]);
    t += D(EXP_LO[j]);
    double res = t + D(EXP_HI[j]);
    if (k > -1022 || (k == -1022 && res >= 1.0)) return D(B(res) + ((uint64_t)(int64_t)k << 52));
    int cl = (k + 0x432) & 63;
    return res * D(1ull << cl);
}

__device__ inline double tanh_(double x) {
    uint64_t xb = B(x);
    uint64_t ab = xb & ABS64;
    if (ab < 0x3E30000000000000ull) return x;
    if (ab > INF64) return D(xb | QUIET64);
    bool neg = ab != xb;
    double a = neg ? -x : x;
    if (a > 20.0) return neg ? -1.0 : 1.0;
    const bool large = 1.0 < a;
    double num, den;
    if (large) {
        double a2 = a + a;
        double c = a2 * 46.16624130844683;
        c = c > 0 ? c + 0.5 : c - 0.5;
        int nn = __double2int_rz(c);
        int j = nn & 0x1F;
        double nd = (double)nn;
        double hi = D(TANH_HI[j]);
        double lo = D(TANH_LO[j]);
        int e = nn - j;
        int k = (e + ((e >> 31) & 0x1F)) >> 5;
        double r6 = (a2 - nd * 0.021660849335603416) * 1.0;
        double r1 = (double)(-nn) * 5.689487495325456E-11 * 1.0;
        double r = r1 + r6;
        double p = r * 0.001388894908637772 + 0.008333367984342196;
        p = p * r + 0.04166666666622608;
        p = p * r + 0.16666666666526087;
        p *= r;
        double rr = r * r;
        p += 0.5;
        p *= rr;
        double hl = lo + hi;
        p += r1;
        p += r6;
        p *= hl;
        p += lo;
        int k1 = (k + (k < 0 ? 1 : 0)) >> 1;
        int k2 = k - k1;
        double s1 = D((uint64_t)(int64_t)(k1 + 0x3FF) << 52);
        double s2 = D((uint64_t)(int64_t)(k2 + 0x3FF) << 52);
        double t = (hi + p) * s1;
        t *= s2;
        t += 1.0;
        num = 2.0;
        den = t;
    } else {
        double x2 = a * a;
        double x3 = x2 * a;
        bool up = 0.9 <= a;
        const double k0 = up ? -0.00016559704390354995 : -0.0002000476210719095;
        const double k1 = up ? 1.154758789961434E-08 : 1.4207792637883471E-08;
        const double k2 = up ? 0.00017307605012622596 : 0.00020911402625291644;
        const double k3 = up ? 0.016735877546189656 : 0.020156216602693764;
        const double k4 = up ? 0.014617304728873168 : 0.017601634900304468;
        const double k5 = up ? 0.3172045589772944 : 0.3816414142883289;
        const double k6 = up ? 0.2277938706590883 : 0.27403042465617977;
        const double k7 = up ? 0.6833816119772959 : 0.8220912739685393;
        double p = k0 - x2 * k1;
        double q = x2 * k2;
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
    const double dd = num / den;
    const double res = large ? 1.0 - dd : dd + a;
    return neg ? -res : res;
}

// pow for finite x > 0 and finite y != 0; the DLL's code for |y| up to 2^63.
__device__ inline double pow_core(double x, double y) {
    uint64_t xb = B(x), yb = B(y);
    if (yb == 0x3FF0000000000000ull) return x;
    int64_t yab = (int64_t)(yb & 0x7FF0000000000000ull);
    if (yab < 0x3C00000000000000ll) return y + 1.0;

    uint64_t x8 = xb;
    int e_int = (int)(xb >> 52) - 0x3FF;
    uint64_t mant = xb & 0x000FFFFFFFFFFFFFull;
    if (e_int == -1023) {
        double m2 = D(mant | 0x3FF0000000000000ull) - 1.0;
        uint64_t mb = B(m2);
        mant = mb & 0x000FFFFFFFFFFFFFull;
        e_int = (int)(mb >> 52) - 0x7FD;
        x8 = mb;
    }
    double e6 = (double)e_int;
    uint64_t r8 = x8 & 0x000FF00000000000ull;
    uint64_t r9 = (x8 & 0x0000080000000000ull) << 1;
    r8 += r9;
    double c = D(r8 | 0x3FE0000000000000ull);
    int j = (int)(r8 >> 44);
    double m = D(mant | 0x3FE0000000000000ull);
    double d4 = c - m;
    double hiv = d4 * D(POW_T1[j]);
    double lov = d4 * D(POW_T2[j]);
    double z = hiv + lov;
    double err = hiv - z;
    double zz = z * z;
    err += lov;
    double pp = fm(0.1428571428571429, z, 0.16666666666666666);
    pp = fm(pp, z, 0.2);
    pp = fm(pp, z, 0.25);
    pp = fm(pp, z, 0.3333333333333333);
    pp = fm(pp, z, 0.5);
    pp = fm(pp, zz, err);
    double s5 = fm(e6, 5.7699990475432854E-08, -pp);
    double t0 = D(POW_T3[j]);
    double s3 = s5 + D(POW_T4[j]);
    double s1 = s3;
    s3 -= z;
    double lg = fm(e6, 0.6931471228599548, t0);
    double s7 = lg;
    lg += s3;
    double lg_full = lg;
    double lg_hi = D(B(lg) & 0xFFFFFFFFF8000000ull);
    double zs = z + s3;
    s7 -= lg_full;
    s1 -= zs;
    s7 += s3;
    double s5b = lg_full - lg_hi;
    s7 += s1;
    s7 += s5b;
    double y_hi = D(yb & 0xFFFFFFFFF8000000ull);
    double y_lo = y - y_hi;
    double a3 = y_lo * s7;
    double a4 = y_lo * lg_hi;
    double a5 = s7 * y_hi;
    double a6 = lg_hi * y_hi;
    a3 += a4;
    a3 += a5;
    double prod_hi = a6 + a3;
    double prod_lo = a6 - prod_hi;
    prod_lo += a3;
    double w = prod_hi * 92.33248261689366;
    if (w > 65536.0) return D(INF64);
    if (w < -68800.0) return 0.0;
    int nn = __double2int_rn(w);
    double nd = (double)nn;
    double r0 = fm(-nd, 0.010830424260348082, prod_hi);
    int jj = nn & 0x3F;
    int k = (nn - jj) >> 6;
    double lo2 = nd * -4.359010638708991E-10;
    double rr2 = r0 + lo2;
    rr2 += prod_lo;
    double q = fm(0.001388888888888889, rr2, 0.008333333333333333);
    q = fm(q, rr2, 0.041666666666666664);
    q = fm(q, rr2, 0.16666666666666666);
    q = fm(q, rr2, 0.5);
    q = fm(q, rr2, 1.0);
    q *= rr2;
    double u5 = q * D(EXP_LO[jj]);
    double u1 = q * D(EXP_HI[jj]);
    u5 += D(EXP_LO[jj]);
    u1 += u5;
    u1 += D(EXP_HI[jj]);
    uint64_t scale_bits = (uint64_t)(int64_t)(k + 0x3FF) << 52;
    if (k <= -1022) {
        if (k == -1022 && u1 >= 1.0) return u1 * D(scale_bits);
        if ((int64_t)B(prod_hi) > (int64_t)0xC0874046DFEFD9D0ull) return D(1);
        int sh = k + 0x432;
        if (sh < 0) sh = 0;
        return u1 * D(1ull << (sh & 63));
    }
    if (k + 0x3FF == 0x7FF) {
        if (u1 >= 1.0) return D(INF64);
        return D(B(u1) | 0x7FE0000000000000ull);
    }
    return u1 * D(scale_bits);
}

// 0 not an integer, 1 an odd integer, 2 an even integer.
__device__ inline uint32_t int_kind(uint64_t bits) {
    uint64_t e = (bits >> 52) & 0x7FF;
    if (e < 1023) return 0;
    if (e > 1075) return 2;
    uint64_t bit = 1ull << (1075 - e);
    return (bits & (bit - 1)) ? 0 : ((bits & bit) ? 1 : 2);
}

// The NaN pow returns for NaN operand(s): the only NaN, or for two the first unless its sign is set
// and the second's is not.
__device__ inline uint64_t nan_operand(uint64_t xb, uint64_t yb) {
    bool xn = (xb & ABS64) > INF64, yn = (yb & ABS64) > INF64;
    return (xn && (!yn || (xb & SIGN64) == 0 || (yb & SIGN64) != 0)) ? xb : yb;
}

__device__ inline double pow_(double x, double y) {
    uint64_t xb = B(x), yb = B(y);
    bool xnan = (xb & ABS64) > INF64, ynan = (yb & ABS64) > INF64;
    // A signalling NaN is invalid even where C99 gives 1.
    bool xs = xnan && (xb & QUIET64) == 0, ys = ynan && (yb & QUIET64) == 0;
    if (xs || ys) return D(nan_operand(xb, yb) | QUIET64);
    if ((yb & ABS64) == 0 || xb == 0x3FF0000000000000ull) return 1.0;
    if (xnan || ynan) return D(nan_operand(xb, yb) | QUIET64);
    bool xneg = (xb & SIGN64) != 0;
    double ax = fabs(x);
    if (ax == 1.0) {
        if ((yb & ABS64) == INF64 || int_kind(yb) == 2) return 1.0;
    }
    if ((yb & ABS64) == INF64) return ((ax < 1.0) == (y > 0.0)) ? 0.0 : D(INF64);
    uint32_t kind = int_kind(yb);
    bool negate = xneg && kind == 1;
    if (ax == 0.0) {
        double r = y < 0.0 ? D(INF64) : 0.0;
        return negate ? -r : r;
    }
    if ((B(ax) & ABS64) == INF64) {
        double r = y < 0.0 ? 0.0 : D(INF64);
        return negate ? -r : r;
    }
    if (xneg && kind == 0) return D(INDEFINITE64);
    if ((int64_t)(yb & 0x7FF0000000000000ull) > 0x43E0000000000000ll) return ((ax < 1.0) == (y > 0.0)) ? 0.0 : D(INF64);  // |y| beyond 2^63 is even
    double r = pow_core(ax, y);
    return negate ? -r : r;
}

}  // namespace ucrt
}  // namespace altd
