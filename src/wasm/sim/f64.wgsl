// IEEE 754 binary64 in integer arithmetic, correctly rounded to nearest,
// ties to even: WebGPU has no f64. `F64` holds the bits (x: low word, y: high
// word). Addition, subtraction and multiplication give the results of Berkeley
// SoftFloat 3e (addMagsF64, subMagsF64, f64_mul, roundPackToF64), but select
// between the cases instead of branching, since one thread runs per car and
// lanes that disagree run both sides; wasm/test/f64-reference.wgsl keeps the
// SoftFloat structure that wasm/test/exact.js compares them with. Division
// divides the significands bit by bit with a sticky remainder. NaNs propagate
// as on x86: the first NaN operand, quieted; invalid operations give the x86
// default NaN. Subnormals are exact.
//
// Double-float ("f64 polyfill") libraries keep a value as an unevaluated sum
// of two f32 and are not correctly rounded, so they cannot reproduce the CPU
// bit for bit; they also lose the binary64 exponent range.

alias F64 = vec2<u32>;
alias U64 = vec2<u32>;

const F64_SIGN: u32 = 0x80000000u;
const F64_ZERO: F64 = F64(0u, 0u);
const F64_ONE: F64 = F64(0u, 0x3ff00000u);
const F64_DEFAULT_NAN: F64 = F64(0u, 0xfff80000u);

// ---- 64-bit integers ---------------------------------------------------

fn u64_add(a: U64, b: U64) -> U64 {
    let lo = a.x + b.x;
    return U64(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}
fn u64_sub(a: U64, b: U64) -> U64 {
    return U64(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}
fn u64_lt(a: U64, b: U64) -> bool { return a.y < b.y || (a.y == b.y && a.x < b.x); }
fn u64_le(a: U64, b: U64) -> bool { return a.y < b.y || (a.y == b.y && a.x <= b.x); }
fn u64_eq(a: U64, b: U64) -> bool { return a.x == b.x && a.y == b.y; }
fn u64_nonzero(a: U64) -> bool { return (a.x | a.y) != 0u; }
// Shifts by counts that may differ between lanes select rather than branch, so
// a wave never runs several cases in turn. `(v << 1u) << (31u - s)` is
// v << (32 - s) for s in 1..31 and 0 for s = 0.
fn u64_shl(a: U64, n: u32) -> U64 {
    let s = n & 31u;
    let lo = a.x << s;
    let hi = (a.y << s) | ((a.x >> 1u) >> (31u - s));
    return select(select(U64(lo, hi), U64(0u, lo), n >= 32u), U64(0u, 0u), n >= 64u);
}
fn u64_shr(a: U64, n: u32) -> U64 {
    let s = n & 31u;
    let lo = (a.x >> s) | ((a.y << 1u) << (31u - s));
    let hi = a.y >> s;
    return select(select(U64(lo, hi), U64(hi, 0u), n >= 32u), U64(0u, 0u), n >= 64u);
}
// SoftFloat shiftRightJam64: any bit shifted out sets bit 0.
fn u64_shr_jam(a: U64, n: u32) -> U64 {
    let z = u64_shr(a, n);
    let lost = select(!u64_eq(u64_shl(z, n), a), u64_nonzero(a), n >= 64u);
    return U64(z.x | select(0u, 1u, lost), z.y);
}
// countLeadingZeros from the binary32 exponent of x (or of x >> 8 from 2^24),
// converted exactly: Tint lowers the builtin to a ~20 instruction dependent
// polyfill, this takes a handful. Zero gives 158, clamped to 32.
fn clz32(x: u32) -> u32 {
    let big = x >= 0x1000000u;
    let v = select(x, x >> 8u, big);
    return min(select(158u, 150u, big) - (bitcast<u32>(f32(v)) >> 23u), 32u);
}
fn u64_clz(a: U64) -> u32 {
    let high = a.y != 0u;
    return clz32(select(a.x, a.y, high)) + select(32u, 0u, high);
}
fn u64_from_u32(v: u32) -> U64 { return U64(v, 0u); }
fn u64_inc(a: U64) -> U64 { return u64_add(a, U64(1u, 0u)); }
// 32 x 32 -> 64 from 16-bit halves.
fn mul32(a: u32, b: u32) -> U64 {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = p01 + p10;
    let mid_carry = select(0u, 0x10000u, mid < p01);
    let lo = p00 + (mid << 16u);
    return U64(lo, p11 + (mid >> 16u) + mid_carry + select(0u, 1u, lo < p00));
}
// The low 64 bits of a product.
fn u64_mul(a: U64, b: U64) -> U64 {
    let p = mul32(a.x, b.x);
    return U64(p.x, p.y + a.x * b.y + a.y * b.x);
}
// The 128-bit product as (w0, w1, w2, w3), least significant first.
fn u64_mul_wide(a: U64, b: U64) -> vec4<u32> {
    let p00 = mul32(a.x, b.x);
    let p01 = mul32(a.x, b.y);
    let p10 = mul32(a.y, b.x);
    let p11 = mul32(a.y, b.y);
    let mid = u64_add(p01, p10);
    let mid_carry = select(0u, 1u, u64_lt(mid, p01));
    let w1 = p00.y + mid.x;
    let carry = select(0u, 1u, w1 < p00.y);
    let hi = u64_add(u64_add(p11, U64(mid.y, mid_carry)), U64(carry, 0u));
    return vec4<u32>(p00.x, w1, hi.x, hi.y);
}
// Unsigned quotient and remainder by a divisor below 2^16.
fn u64_div_small(a: U64, d: u32) -> U64 {
    let h1 = (a.y >> 16u);
    let q1 = h1 / d;
    let r1 = h1 % d;
    let h2 = (r1 << 16u) | (a.y & 0xffffu);
    let q2 = h2 / d;
    let r2 = h2 % d;
    let h3 = (r2 << 16u) | (a.x >> 16u);
    let q3 = h3 / d;
    let r3 = h3 % d;
    let h4 = (r3 << 16u) | (a.x & 0xffffu);
    let q4 = h4 / d;
    return U64((q3 << 16u) | q4, (q1 << 16u) | q2);
}

// ---- classification ----------------------------------------------------

fn f64_exp(a: F64) -> i32 { return i32((a.y >> 20u) & 0x7ffu); }
fn f64_frac(a: F64) -> U64 { return U64(a.x, a.y & 0xfffffu); }
fn f64_signbit(a: F64) -> bool { return (a.y & F64_SIGN) != 0u; }
fn f64_isnan(a: F64) -> bool {
    let h = a.y & 0x7fffffffu;
    return h > 0x7ff00000u || (h == 0x7ff00000u && a.x != 0u);
}
fn f64_isinf(a: F64) -> bool { return (a.y & 0x7fffffffu) == 0x7ff00000u && a.x == 0u; }
fn f64_isfinite(a: F64) -> bool { return (a.y & 0x7ff00000u) != 0x7ff00000u; }
fn f64_neg(a: F64) -> F64 { return F64(a.x, a.y ^ F64_SIGN); }
fn f64_abs(a: F64) -> F64 { return F64(a.x, a.y & 0x7fffffffu); }
fn f64_inf(sign: bool) -> F64 { return F64(0u, select(0u, F64_SIGN, sign) | 0x7ff00000u); }
fn f64_signed_zero(sign: bool) -> F64 { return F64(0u, select(0u, F64_SIGN, sign)); }
fn f64_quiet(a: F64) -> F64 { return F64(a.x, a.y | 0x80000u); }
fn f64_propagate_nan(a: F64, b: F64) -> F64 { return f64_quiet(select(b, a, f64_isnan(a))); }
// Copies the sign of `s` onto `a`.
fn f64_copysign(a: F64, s: F64) -> F64 { return F64(a.x, (a.y & 0x7fffffffu) | (s.y & F64_SIGN)); }

// ---- rounding and packing (SoftFloat conventions) -----------------------
// `exp` is the biased exponent minus one; `sig` has its leading bit at 62.

fn f64_pack(sign: bool, exp: i32, sig: U64) -> F64 {
    return u64_add(U64(0u, select(0u, F64_SIGN, sign) | (u32(exp) << 20u)), sig);
}
fn f64_round_pack(sign: bool, exp_in: i32, sig_in: U64) -> F64 {
    var exp = exp_in;
    var sig = sig_in;
    var round_bits = sig.x & 0x3ffu;
    if (exp < 0 || exp >= 0x7fd) {
        if (exp < 0) {
            sig = u64_shr_jam(sig, u32(-exp));
            exp = 0;
            round_bits = sig.x & 0x3ffu;
        } else if (exp > 0x7fd || !u64_lt(u64_add(sig, U64(0x200u, 0u)), U64(0u, 0x80000000u))) {
            return f64_inf(sign);
        }
    }
    sig = u64_shr(u64_add(sig, U64(0x200u, 0u)), 10u);
    if (round_bits == 0x200u) { sig.x = sig.x & 0xfffffffeu; }
    if (!u64_nonzero(sig)) { exp = 0; }
    return f64_pack(sign, exp, sig);
}
fn f64_norm_round_pack(sign: bool, exp: i32, sig: U64) -> F64 {
    let shift = i32(u64_clz(sig)) - 1;
    let e = exp - shift;
    if (shift >= 10 && e >= 0 && e < 0x7fd) {
        return f64_pack(sign, select(0, e, u64_nonzero(sig)), u64_shl(sig, u32(shift - 10)));
    }
    return f64_round_pack(sign, e, u64_shl(sig, u32(shift)));
}
// Normalizes a nonzero subnormal fraction: (exponent, significand with bit 52).
fn f64_norm_subnormal(frac: U64) -> vec3<u32> {
    let shift = i32(u64_clz(frac)) - 11;
    let sig = u64_shl(frac, u32(shift));
    return vec3<u32>(u32(1 - shift), sig.x, sig.y);
}

// ---- arithmetic ----------------------------------------------------------

// a + b with b's sign flipped by `flip` (F64_SIGN for subtraction). Lanes of
// a wave see different signs, exponent gaps and cancellation, and branching
// on those ran every case in turn (three times the uniform cost), so every
// finite case takes one path that selects instead. The larger magnitude `x`
// and the smaller `y` get their significands at bit 61 (subnormals have
// exponent 1 and no hidden bit); y is shifted right by the exponent gap with
// a sticky bit. The sum or difference is exact above its sticky bit, which
// lies at least 7 bits below the rounding position after normalizing
// (cancellation beyond one bit only happens when the gap is at most 1 and
// nothing was shifted out), so rounding it once is correctly rounded. The
// result cannot underflow inexactly; overflow gives infinity.
fn f64_add_signed(a: F64, b: F64, flip: u32) -> F64 {
    let bs = F64(b.x, b.y ^ flip);
    let swap = u64_lt(f64_abs(a), f64_abs(bs));
    let x = select(a, bs, swap);
    let y = select(bs, a, swap);
    let subtract = ((a.y ^ bs.y) & F64_SIGN) != 0u;
    let ex_raw = f64_exp(x);
    let ey_raw = f64_exp(y);
    if (ex_raw == 0x7ff) {
        // NaN operands propagate (b unflipped), inf - inf is invalid,
        // otherwise the infinite operand with its effective sign.
        let inf = select(x, F64_DEFAULT_NAN, subtract && ey_raw == 0x7ff);
        return select(inf, f64_propagate_nan(a, b), f64_isnan(a) || f64_isnan(b));
    }
    let ex = max(ex_raw, 1);
    let mx = U64(x.x, (x.y & 0xfffffu) | select(0x100000u, 0u, ex_raw == 0));
    let my = U64(y.x, (y.y & 0xfffffu) | select(0x100000u, 0u, ey_raw == 0));
    let sx = U64(mx.x << 9u, (mx.y << 9u) | (mx.x >> 23u));
    let sy = u64_shr_jam(U64(my.x << 9u, (my.y << 9u) | (my.x >> 23u)), u32(ex - max(ey_raw, 1)));
    let z = select(u64_add(sx, sy), u64_sub(sx, sy), subtract);
    // Leading bit to 62, as far as the exponent allows (subnormal results),
    // then round to nearest even at bit 10.
    let shift = min(u64_clz(z) - 1u, u32(ex));
    let n = u64_shl(z, shift);
    let rounded = u64_add(n, U64(0x200u, 0u));
    let sig = U64(((rounded.x >> 10u) | (rounded.y << 22u)) & select(0xffffffffu, 0xfffffffeu, (n.x & 0x3ffu) == 0x200u), rounded.y >> 10u);
    // An exact zero is +0 for a difference and keeps the operands' sign for
    // a sum; the leading bit of `sig` carries into the exponent field.
    let nonzero = u64_nonzero(z);
    let sign = (x.y & F64_SIGN) != 0u && (nonzero || !subtract);
    let exp = select(0, ex - i32(shift), nonzero);
    let finite = u64_add(U64(0u, select(0u, F64_SIGN, sign) | (u32(exp) << 20u)), sig);
    return select(finite, f64_inf(sign), exp + i32(sig.y >> 20u) >= 0x7ff);
}
fn f64_add(a: F64, b: F64) -> F64 { return f64_add_signed(a, b, 0u); }
fn f64_sub(a: F64, b: F64) -> F64 { return f64_add_signed(a, b, F64_SIGN); }
// Normal operands whose product is normal before rounding and cannot round
// to infinity take a straight path; zeros, subnormals, infinities, NaNs and
// results near the ends of the range branch to SoftFloat's general code,
// which lanes rarely need.
fn f64_mul(a: F64, b: F64) -> F64 {
    let ea = f64_exp(a);
    let eb = f64_exp(b);
    let ez = ea + eb - 0x3ff;
    if (ea == 0 || eb == 0 || ea == 0x7ff || eb == 0x7ff || ez < 1 || ez > 0x7fc) { return f64_mul_general(a, b); }
    let sa = U64(a.x << 10u, (((a.y & 0xfffffu) | 0x100000u) << 10u) | (a.x >> 22u));
    let sb = U64(b.x << 11u, (((b.y & 0xfffffu) | 0x100000u) << 11u) | (b.x >> 21u));
    let p = u64_mul_wide(sa, sb);
    // The product's leading bit is at 125 or 126; keep the top 64 bits with
    // a sticky bit and move the leading bit to 62.
    let low = p.w < 0x40000000u;
    let top = U64(p.z | select(0u, 1u, (p.x | p.y) != 0u), p.w);
    let n = select(top, U64(top.x << 1u, (top.y << 1u) | (top.x >> 31u)), low);
    let rounded = u64_add(n, U64(0x200u, 0u));
    let sig = U64(((rounded.x >> 10u) | (rounded.y << 22u)) & select(0xffffffffu, 0xfffffffeu, (n.x & 0x3ffu) == 0x200u), rounded.y >> 10u);
    let exp = u32(ez - select(0, 1, low));
    return u64_add(U64(0u, ((a.y ^ b.y) & F64_SIGN) | (exp << 20u)), sig);
}
fn f64_mul_general(a: F64, b: F64) -> F64 {
    let sign = ((a.y ^ b.y) & F64_SIGN) != 0u;
    var ea = f64_exp(a);
    var eb = f64_exp(b);
    var sa = f64_frac(a);
    var sb = f64_frac(b);
    if (ea == 0x7ff) {
        if (u64_nonzero(sa) || (eb == 0x7ff && u64_nonzero(sb))) { return f64_propagate_nan(a, b); }
        if (eb == 0 && !u64_nonzero(sb)) { return F64_DEFAULT_NAN; }
        return f64_inf(sign);
    }
    if (eb == 0x7ff) {
        if (u64_nonzero(sb)) { return f64_propagate_nan(a, b); }
        if (ea == 0 && !u64_nonzero(sa)) { return F64_DEFAULT_NAN; }
        return f64_inf(sign);
    }
    if (ea == 0) {
        if (!u64_nonzero(sa)) { return f64_signed_zero(sign); }
        let n = f64_norm_subnormal(sa);
        ea = i32(n.x);
        sa = n.yz;
    }
    if (eb == 0) {
        if (!u64_nonzero(sb)) { return f64_signed_zero(sign); }
        let n = f64_norm_subnormal(sb);
        eb = i32(n.x);
        sb = n.yz;
    }
    var ez = ea + eb - 0x3ff;
    sa = u64_shl(U64(sa.x, sa.y | 0x100000u), 10u);
    sb = u64_shl(U64(sb.x, sb.y | 0x100000u), 11u);
    let p = u64_mul_wide(sa, sb);
    var sz = U64(p.z | select(0u, 1u, (p.x | p.y) != 0u), p.w);
    if (u64_lt(sz, U64(0u, 0x40000000u))) {
        ez = ez - 1;
        sz = u64_shl(sz, 1u);
    }
    return f64_round_pack(sign, ez, sz);
}
// One digit of f64_div: floor(r * 2^18 / sb) for r < sb < 2^53, and the new
// remainder, as (digit, remainder low, remainder high, 1), or w = 0 if the f32
// estimate (inv = 2^18 / sb) was further off than two steps either way. The
// remainder is computed modulo 2^64, and exactly when its top word beyond 64
// bits matches its sign.
fn f64_div_digit(r: U64, sb: U64, inv: f32) -> vec4<u32> {
    var digit = min(u32((f32(r.y) * 4294967296.0 + f32(r.x)) * inv), 0x7ffffu);
    let low = mul32(digit, sb.x);
    let high = mul32(digit, sb.y);
    let mid = low.y + high.x;
    let top = high.y + select(0u, 1u, mid < low.y);
    let shifted = u64_shl(r, 18u);
    let product = U64(low.x, mid);
    var remainder = u64_sub(shifted, product);
    let remainder_top = (r.y >> 14u) - top - select(0u, 1u, u64_lt(shifted, product));
    var fits = remainder_top == select(0u, 0xffffffffu, i64_is_negative(remainder));
    for (var i = 0u; i < 2u; i = i + 1u) {
        let low_digit = i64_is_negative(remainder);
        digit = select(digit, digit - 1u, low_digit);
        remainder = select(remainder, u64_add(remainder, sb), low_digit);
    }
    for (var i = 0u; i < 2u; i = i + 1u) {
        let high_digit = !i64_is_negative(remainder) && !u64_lt(remainder, sb);
        digit = select(digit, digit + 1u, high_digit);
        remainder = select(remainder, u64_sub(remainder, sb), high_digit);
    }
    fits = fits && !i64_is_negative(remainder) && u64_lt(remainder, sb);
    return vec4<u32>(digit, remainder.x, remainder.y, select(0u, 1u, fits));
}
fn f64_div(a: F64, b: F64) -> F64 {
    let sign = ((a.y ^ b.y) & F64_SIGN) != 0u;
    var ea = f64_exp(a);
    var eb = f64_exp(b);
    var sa = f64_frac(a);
    var sb = f64_frac(b);
    if (ea == 0x7ff) {
        if (u64_nonzero(sa)) { return f64_propagate_nan(a, b); }
        if (eb == 0x7ff) {
            if (u64_nonzero(sb)) { return f64_propagate_nan(a, b); }
            return F64_DEFAULT_NAN;
        }
        return f64_inf(sign);
    }
    if (eb == 0x7ff) {
        if (u64_nonzero(sb)) { return f64_propagate_nan(a, b); }
        return f64_signed_zero(sign);
    }
    if (eb == 0) {
        if (!u64_nonzero(sb)) {
            if (ea == 0 && !u64_nonzero(sa)) { return F64_DEFAULT_NAN; }
            return f64_inf(sign);
        }
        let n = f64_norm_subnormal(sb);
        eb = i32(n.x);
        sb = n.yz;
    }
    if (ea == 0) {
        if (!u64_nonzero(sa)) { return f64_signed_zero(sign); }
        let n = f64_norm_subnormal(sa);
        ea = i32(n.x);
        sa = n.yz;
    }
    var ez = ea - eb + 0x3fe;
    sa = U64(sa.x, sa.y | 0x100000u);
    sb = U64(sb.x, sb.y | 0x100000u);
    if (u64_lt(sa, sb)) {
        ez = ez - 1;
        sa = u64_add(sa, sa);
    }
    // sa / sb lies in [1, 2): 55 quotient bits, then the remainder as sticky.
    // The leading bit is 1; three 18-bit digits of long division follow, each
    // estimated from an f32 reciprocal and corrected with its exact remainder.
    let inv = 262144.0 / (f32(sb.y) * 4294967296.0 + f32(sb.x));
    var r = u64_sub(sa, sb);
    var q = U64(1u, 0u);
    var exact = true;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let digit = f64_div_digit(r, sb, inv);
        q = U64((q.x << 18u) | digit.x, (q.y << 18u) | (q.x >> 14u));
        r = digit.yz;
        exact = exact && digit.w != 0u;
    }
    // A device whose f32 division is further off than the correction steps
    // reach divides bit by bit.
    if (!exact) {
        r = sa;
        q = U64(0u, 0u);
        for (var i = 0u; i < 55u; i = i + 1u) {
            q = u64_add(q, q);
            if (!u64_lt(r, sb)) {
                r = u64_sub(r, sb);
                q.x = q.x | 1u;
            }
            r = u64_add(r, r);
        }
    }
    let sz = u64_shl(q, 8u);
    return f64_round_pack(sign, ez, U64(sz.x | select(0u, 1u, u64_nonzero(r)), sz.y));
}

// ---- fused multiply-add ----------------------------------------------------

struct U128 {
    lo: U64,
    hi: U64,
}

fn u128_add(a: U128, b: U128) -> U128 {
    let lo = u64_add(a.lo, b.lo);
    return U128(lo, u64_add(u64_add(a.hi, b.hi), U64(select(0u, 1u, u64_lt(lo, a.lo)), 0u)));
}
fn u128_sub(a: U128, b: U128) -> U128 {
    return U128(u64_sub(a.lo, b.lo), u64_sub(u64_sub(a.hi, b.hi), U64(select(0u, 1u, u64_lt(a.lo, b.lo)), 0u)));
}
fn u128_lt(a: U128, b: U128) -> bool { return u64_lt(a.hi, b.hi) || (u64_eq(a.hi, b.hi) && u64_lt(a.lo, b.lo)); }
fn u128_clz(a: U128) -> u32 { return select(u64_clz(a.hi), 64u + u64_clz(a.lo), !u64_nonzero(a.hi)); }
// Counts that wrap below zero are u64 shifts of 64 or more, which give zero.
fn u128_shl(a: U128, n: u32) -> U128 {
    return U128(u64_shl(a.lo, n), u64_shl(a.hi, n) | u64_shr(a.lo, 64u - n) | u64_shl(a.lo, n - 64u));
}
fn u128_shr_jam(a: U128, n: u32) -> U128 {
    let z = U128(u64_shr(a.lo, n) | u64_shl(a.hi, 64u - n) | u64_shr(a.hi, n - 64u), u64_shr(a.hi, n));
    let back = u128_shl(z, n);
    let lost = !u64_eq(back.lo, a.lo) || !u64_eq(back.hi, a.hi);
    return U128(U64(z.lo.x | select(0u, 1u, lost), z.lo.y), z.hi);
}
// a * b + c, rounded once (C fma, SoftFloat's f64_mulAdd). NaN operands
// propagate in the order a, b, c; an infinite product with a zero factor, or
// one added to the opposite infinity, gives the default NaN. The callers are
// the Windows math kernels, which branch on uniform data far more than here.
fn f64_fma(a: F64, b: F64, c: F64) -> F64 {
    let sign_p = ((a.y ^ b.y) & F64_SIGN) != 0u;
    let sign_c = f64_signbit(c);
    var ea = f64_exp(a);
    var eb = f64_exp(b);
    var ec = f64_exp(c);
    if (ea == 0x7ff || eb == 0x7ff || ec == 0x7ff) {
        if (f64_isnan(a) || f64_isnan(b) || f64_isnan(c)) {
            return f64_quiet(select(select(c, b, f64_isnan(b)), a, f64_isnan(a)));
        }
        if (ea == 0x7ff || eb == 0x7ff) {
            if (f64_is_zero(a) || f64_is_zero(b) || (ec == 0x7ff && sign_c != sign_p)) { return F64_DEFAULT_NAN; }
            return f64_inf(sign_p);
        }
        return c;
    }
    // An exact zero product adds as a signed zero; a zero c leaves the
    // product, rounded once.
    if (f64_is_zero(a) || f64_is_zero(b)) { return f64_add(f64_signed_zero(sign_p), c); }
    if (f64_is_zero(c)) { return f64_mul(a, b); }
    var ma = f64_frac(a);
    var mb = f64_frac(b);
    var mc = f64_frac(c);
    if (ea == 0) {
        let n = f64_norm_subnormal(ma);
        ea = i32(n.x);
        ma = n.yz;
    }
    if (eb == 0) {
        let n = f64_norm_subnormal(mb);
        eb = i32(n.x);
        mb = n.yz;
    }
    if (ec == 0) {
        let n = f64_norm_subnormal(mc);
        ec = i32(n.x);
        mc = n.yz;
    }
    // Both terms as 128-bit significands with the leading bit at 125, worth
    // sig * 2^(e - 1148): the product exactly, c shifted up by 73.
    let p = u64_mul_wide(u64_shl(U64(ma.x, ma.y | 0x100000u), 10u), u64_shl(U64(mb.x, mb.y | 0x100000u), 10u));
    var wp = U128(p.xy, p.zw);
    var e = ea + eb - 1022;
    if (wp.hi.y < 0x20000000u) {
        wp = u128_shl(wp, 1u);
        e = e - 1;
    }
    var wc = U128(U64(0u, 0u), u64_shl(U64(mc.x, mc.y | 0x100000u), 9u));
    // Align the smaller exponent with a sticky bit. Gaps of at most one shift
    // out only zeros (both terms end in zero bits), so a difference with
    // massive cancellation is exact; beyond that the sticky bit lies far below
    // the rounding position.
    let d = e - ec;
    if (d >= 0) {
        wc = u128_shr_jam(wc, u32(d));
    } else {
        wp = u128_shr_jam(wp, u32(-d));
        e = ec;
    }
    var w: U128;
    var sign = sign_p;
    if (sign_p == sign_c) {
        w = u128_add(wp, wc);
    } else if (u128_lt(wp, wc)) {
        w = u128_sub(wc, wp);
        sign = sign_c;
    } else {
        w = u128_sub(wp, wc);
    }
    if (!u64_nonzero(w.lo) && !u64_nonzero(w.hi)) { return F64_ZERO; }
    // Leading bit to 126: the top word then holds it at 62 for rounding.
    let shift = u128_clz(w) - 1u;
    w = u128_shl(w, shift);
    e = e - i32(shift);
    return f64_round_pack(sign, e, U64(w.hi.x | select(0u, 1u, u64_nonzero(w.lo)), w.hi.y));
}

// ---- comparisons (false for NaN operands) ---------------------------------

fn f64_both_zero(a: F64, b: F64) -> bool { return ((a.x | b.x) | ((a.y | b.y) & 0x7fffffffu)) == 0u; }
fn f64_lt(a: F64, b: F64) -> bool {
    if (f64_isnan(a) || f64_isnan(b)) { return false; }
    let sa = f64_signbit(a);
    if (sa != f64_signbit(b)) { return sa && !f64_both_zero(a, b); }
    return !u64_eq(a, b) && (sa != u64_lt(a, b));
}
fn f64_le(a: F64, b: F64) -> bool {
    if (f64_isnan(a) || f64_isnan(b)) { return false; }
    let sa = f64_signbit(a);
    if (sa != f64_signbit(b)) { return sa || f64_both_zero(a, b); }
    return u64_eq(a, b) || (sa != u64_lt(a, b));
}
fn f64_gt(a: F64, b: F64) -> bool { return f64_lt(b, a); }
fn f64_ge(a: F64, b: F64) -> bool { return f64_le(b, a); }
fn f64_eq(a: F64, b: F64) -> bool {
    if (f64_isnan(a) || f64_isnan(b)) { return false; }
    return u64_eq(a, b) || f64_both_zero(a, b);
}
fn f64_is_zero(a: F64) -> bool { return (a.x | (a.y & 0x7fffffffu)) == 0u; }

// ---- conversions ---------------------------------------------------------

fn f64_from_f32(f: f32) -> F64 {
    let u = bitcast<u32>(f);
    let sign = u & 0x80000000u;
    let e = (u >> 23u) & 0xffu;
    var frac = u & 0x7fffffu;
    if (e == 0xffu) {
        if (frac != 0u) { return F64(frac << 29u, sign | 0x7ff80000u | (frac >> 3u)); }
        return F64(0u, sign | 0x7ff00000u);
    }
    if (e == 0u) {
        if (frac == 0u) { return F64(0u, sign); }
        let shift = clz32(frac) - 8u;
        frac = frac << shift;
        // The significand's leading bit adds one to the exponent field.
        return F64(frac << 29u, (sign | ((0x380u - shift) << 20u)) + (frac >> 3u));
    }
    return F64(frac << 29u, sign | ((e + 0x380u) << 20u) | (frac >> 3u));
}
// SoftFloat roundPackToF32: `sig` has its leading bit at 30.
fn f32_round_pack(sign: u32, exp_in: i32, sig_in: u32) -> f32 {
    var exp = exp_in;
    var sig = sig_in;
    var round_bits = sig & 0x7fu;
    if (exp < 0 || exp >= 0xfd) {
        if (exp < 0) {
            let d = u32(-exp);
            if (d < 31u) {
                sig = (sig >> d) | select(0u, 1u, (sig << (32u - d)) != 0u);
            } else {
                sig = select(0u, 1u, sig != 0u);
            }
            exp = 0;
            round_bits = sig & 0x7fu;
        } else if (exp > 0xfd || sig + 0x40u >= 0x80000000u) {
            return bitcast<f32>(sign | 0x7f800000u);
        }
    }
    sig = (sig + 0x40u) >> 7u;
    if (round_bits == 0x40u) { sig = sig & 0xfffffffeu; }
    if (sig == 0u) { exp = 0; }
    return bitcast<f32>(sign + (u32(exp) << 23u) + sig);
}
fn f64_to_f32(a: F64) -> f32 {
    let sign = a.y & F64_SIGN;
    let e = f64_exp(a);
    let fh = a.y & 0xfffffu;
    let fl = a.x;
    if (e == 0x7ff) {
        if ((fh | fl) != 0u) { return bitcast<f32>(sign | 0x7fc00000u | (fh << 3u) | (fl >> 29u)); }
        return bitcast<f32>(sign | 0x7f800000u);
    }
    let frac = (fh << 10u) | (fl >> 22u) | select(0u, 1u, (fl & 0x3fffffu) != 0u);
    if ((u32(e) | frac) == 0u) { return bitcast<f32>(sign); }
    return f32_round_pack(sign, e - 0x381, frac | 0x40000000u);
}
fn f64_from_u32(v: u32) -> F64 {
    if (v == 0u) { return F64_ZERO; }
    let shift = clz32(v) + 21u;
    return f64_pack(false, 0x432 - i32(shift), u64_shl(U64(v, 0u), shift));
}
fn f64_from_i32(v: i32) -> F64 {
    let negative = v < 0;
    let r = f64_from_u32(select(u32(v), 0u - u32(v), negative));
    return F64(r.x, r.y | select(0u, F64_SIGN, negative));
}
fn f64_from_u64(a: U64) -> F64 {
    if (!u64_nonzero(a)) { return F64_ZERO; }
    if ((a.y & 0x80000000u) != 0u) { return f64_round_pack(false, 0x43d, u64_shr_jam(a, 1u)); }
    return f64_norm_round_pack(false, 0x43c, a);
}
// Two's complement 64-bit integers.
fn i64_neg(a: U64) -> U64 { return u64_sub(U64(0u, 0u), a); }
fn i64_is_negative(a: U64) -> bool { return (a.y & 0x80000000u) != 0u; }
fn i64_lt(a: U64, b: U64) -> bool {
    return select(u64_lt(a, b), i64_is_negative(a), i64_is_negative(a) != i64_is_negative(b));
}
fn i64_from_i32(v: i32) -> U64 { return U64(u32(v), select(0u, 0xffffffffu, v < 0)); }
fn f64_from_i64(a: U64) -> F64 {
    let negative = i64_is_negative(a);
    let m = select(a, i64_neg(a), negative);
    if (!u64_nonzero(m)) { return F64_ZERO; }
    if ((m.y & 0x80000000u) != 0u) { return f64_round_pack(negative, 0x43d, u64_shr_jam(m, 1u)); }
    return f64_norm_round_pack(negative, 0x43c, m);
}
// Rust `as i64`: truncation, saturating, NaN to 0.
fn f64_to_i64(a: F64) -> U64 {
    if (f64_isnan(a)) { return U64(0u, 0u); }
    let e = f64_exp(a);
    let negative = f64_signbit(a);
    if (e < 0x3ff) { return U64(0u, 0u); }
    if (e >= 0x43e) { return select(U64(0xffffffffu, 0x7fffffffu), U64(0u, 0x80000000u), negative); }
    let sig = U64(a.x, (a.y & 0xfffffu) | 0x100000u);
    var m: U64;
    if (e >= 0x433) { m = u64_shl(sig, u32(e - 0x433)); } else { m = u64_shr(sig, u32(0x433 - e)); }
    return select(m, i64_neg(m), negative);
}
// Rust `as i32`: truncation, saturating, NaN to 0.
fn f64_to_i32(a: F64) -> i32 {
    if (f64_isnan(a)) { return 0; }
    let e = f64_exp(a);
    let negative = f64_signbit(a);
    if (e < 0x3ff) { return 0; }
    if (e >= 0x41e) { return select(2147483647, -2147483647 - 1, negative); }
    let m = u64_shr(U64(a.x, (a.y & 0xfffffu) | 0x100000u), u32(0x433 - e)).x;
    return select(i32(m), -i32(m), negative);
}
// Rust `as u64` of a nonnegative double below 2^64 (truncation).
fn f64_to_u64(a: F64) -> U64 {
    let e = f64_exp(a);
    if (f64_signbit(a) || e < 0x3ff) { return U64(0u, 0u); }
    let sig = U64(a.x, (a.y & 0xfffffu) | 0x100000u);
    if (e >= 0x433) { return u64_shl(sig, u32(e - 0x433)); }
    return u64_shr(sig, u32(0x433 - e));
}

// ---- rounding to integers (SoftFloat f64_roundToInt) -------------------

fn f64_round_int(a: F64, ties_away: bool) -> F64 {
    let e = f64_exp(a);
    if (e <= 0x3fe) {
        if (f64_is_zero(a)) { return a; }
        var z = F64(0u, a.y & F64_SIGN);
        let half_or_more = e == 0x3fe;
        if (half_or_more && (ties_away || u64_nonzero(f64_frac(a)))) { z.y = z.y | 0x3ff00000u; }
        return z;
    }
    if (e >= 0x433) {
        if (e == 0x7ff && u64_nonzero(f64_frac(a))) { return f64_quiet(a); }
        return a;
    }
    let last = u64_shl(U64(1u, 0u), u32(0x433 - e));
    let round_mask = u64_sub(last, U64(1u, 0u));
    var z = u64_add(a, u64_shr(last, 1u));
    if (!ties_away && !u64_nonzero(U64(z.x & round_mask.x, z.y & round_mask.y))) {
        z = U64(z.x & ~last.x, z.y & ~last.y);
    }
    return U64(z.x & ~round_mask.x, z.y & ~round_mask.y);
}
// rint: nearest, ties to even.
fn f64_rint(a: F64) -> F64 { return f64_round_int(a, false); }
// round: nearest, ties away from zero.
fn f64_round(a: F64) -> F64 { return f64_round_int(a, true); }
fn f64_trunc(a: F64) -> F64 {
    let e = f64_exp(a);
    if (e < 0x3ff) { return F64(0u, a.y & F64_SIGN); }
    if (e >= 0x433) {
        if (e == 0x7ff && u64_nonzero(f64_frac(a))) { return f64_quiet(a); }
        return a;
    }
    let mask = u64_sub(u64_shl(U64(1u, 0u), u32(0x433 - e)), U64(1u, 0u));
    return U64(a.x & ~mask.x, a.y & ~mask.y);
}
fn f64_floor(a: F64) -> F64 {
    let t = f64_trunc(a);
    if (f64_signbit(a) && !f64_eq(t, a) && !f64_isnan(a)) { return f64_sub(t, F64_ONE); }
    return t;
}
fn f64_ceil(a: F64) -> F64 {
    let t = f64_trunc(a);
    if (!f64_signbit(a) && !f64_eq(t, a) && !f64_isnan(a)) { return f64_add(t, F64_ONE); }
    return t;
}
