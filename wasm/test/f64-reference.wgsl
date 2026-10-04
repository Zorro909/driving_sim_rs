// The SoftFloat 3e addition, subtraction and multiplication (with their
// shift, rounding and NaN helpers) that src/wasm/sim/f64.wgsl used before its
// branch-free versions, verified bit for bit against the CPU. wasm/test/exact.js
// compares the production functions with these `ref_` copies. Uses the U64
// helpers and constants of f64.wgsl.
fn ref_u64_shl(a: U64, n: u32) -> U64 {
    if (n == 0u) { return a; }
    if (n >= 64u) { return U64(0u, 0u); }
    if (n >= 32u) { return U64(0u, a.x << (n - 32u)); }
    return U64(a.x << n, (a.y << n) | (a.x >> (32u - n)));
}
fn ref_u64_shr(a: U64, n: u32) -> U64 {
    if (n == 0u) { return a; }
    if (n >= 64u) { return U64(0u, 0u); }
    if (n >= 32u) { return U64(a.y >> (n - 32u), 0u); }
    return U64((a.x >> n) | (a.y << (32u - n)), a.y >> n);
}
fn ref_u64_shr_jam(a: U64, n: u32) -> U64 {
    if (n == 0u) { return a; }
    if (n >= 64u) { return U64(select(0u, 1u, u64_nonzero(a)), 0u); }
    let z = ref_u64_shr(a, n);
    let lost = !u64_eq(ref_u64_shl(z, n), a);
    return U64(z.x | select(0u, 1u, lost), z.y);
}
fn ref_f64_propagate_nan(a: F64, b: F64) -> F64 {
    if (f64_isnan(a)) { return f64_quiet(a); }
    return f64_quiet(b);
}
fn ref_f64_round_pack(sign: bool, exp_in: i32, sig_in: U64) -> F64 {
    var exp = exp_in;
    var sig = sig_in;
    var round_bits = sig.x & 0x3ffu;
    if (exp < 0 || exp >= 0x7fd) {
        if (exp < 0) {
            sig = ref_u64_shr_jam(sig, u32(-exp));
            exp = 0;
            round_bits = sig.x & 0x3ffu;
        } else if (exp > 0x7fd || !u64_lt(u64_add(sig, U64(0x200u, 0u)), U64(0u, 0x80000000u))) {
            return f64_inf(sign);
        }
    }
    sig = ref_u64_shr(u64_add(sig, U64(0x200u, 0u)), 10u);
    if (round_bits == 0x200u) { sig.x = sig.x & 0xfffffffeu; }
    if (!u64_nonzero(sig)) { exp = 0; }
    return f64_pack(sign, exp, sig);
}
fn ref_f64_norm_round_pack(sign: bool, exp: i32, sig: U64) -> F64 {
    let shift = i32(u64_clz(sig)) - 1;
    let e = exp - shift;
    if (shift >= 10 && e >= 0 && e < 0x7fd) {
        return f64_pack(sign, select(0, e, u64_nonzero(sig)), ref_u64_shl(sig, u32(shift - 10)));
    }
    return ref_f64_round_pack(sign, e, ref_u64_shl(sig, u32(shift)));
}
fn ref_f64_norm_subnormal(frac: U64) -> vec3<u32> {
    let shift = i32(u64_clz(frac)) - 11;
    let sig = ref_u64_shl(frac, u32(shift));
    return vec3<u32>(u32(1 - shift), sig.x, sig.y);
}

fn ref_f64_add_mags(a: F64, b: F64, sign: bool) -> F64 {
    let ea = f64_exp(a);
    let eb = f64_exp(b);
    var sa = f64_frac(a);
    var sb = f64_frac(b);
    let diff = ea - eb;
    var ez: i32;
    var sz: U64;
    if (diff == 0) {
        if (ea == 0) { return u64_add(a, sb); }
        if (ea == 0x7ff) {
            if (u64_nonzero(sa) || u64_nonzero(sb)) { return ref_f64_propagate_nan(a, b); }
            return a;
        }
        ez = ea;
        sz = ref_u64_shl(u64_add(U64(0u, 0x200000u), u64_add(sa, sb)), 9u);
    } else {
        sa = ref_u64_shl(sa, 9u);
        sb = ref_u64_shl(sb, 9u);
        if (diff < 0) {
            if (eb == 0x7ff) {
                if (u64_nonzero(sb)) { return ref_f64_propagate_nan(a, b); }
                return f64_inf(sign);
            }
            ez = eb;
            if (ea != 0) { sa = u64_add(sa, U64(0u, 0x20000000u)); } else { sa = ref_u64_shl(sa, 1u); }
            sa = ref_u64_shr_jam(sa, u32(-diff));
        } else {
            if (ea == 0x7ff) {
                if (u64_nonzero(sa)) { return ref_f64_propagate_nan(a, b); }
                return a;
            }
            ez = ea;
            if (eb != 0) { sb = u64_add(sb, U64(0u, 0x20000000u)); } else { sb = ref_u64_shl(sb, 1u); }
            sb = ref_u64_shr_jam(sb, u32(diff));
        }
        sz = u64_add(U64(0u, 0x20000000u), u64_add(sa, sb));
        if (u64_lt(sz, U64(0u, 0x40000000u))) {
            ez = ez - 1;
            sz = ref_u64_shl(sz, 1u);
        }
    }
    return ref_f64_round_pack(sign, ez, sz);
}
fn ref_f64_sub_mags(a: F64, b: F64, sign_in: bool) -> F64 {
    var sign = sign_in;
    let ea = f64_exp(a);
    let eb = f64_exp(b);
    var sa = f64_frac(a);
    var sb = f64_frac(b);
    let diff = ea - eb;
    if (diff == 0) {
        if (ea == 0x7ff) {
            if (u64_nonzero(sa) || u64_nonzero(sb)) { return ref_f64_propagate_nan(a, b); }
            return F64_DEFAULT_NAN;
        }
        if (u64_eq(sa, sb)) { return F64_ZERO; }
        var e = ea;
        if (e != 0) { e = e - 1; }
        var d: U64;
        if (u64_lt(sa, sb)) {
            sign = !sign;
            d = u64_sub(sb, sa);
        } else {
            d = u64_sub(sa, sb);
        }
        var shift = i32(u64_clz(d)) - 11;
        var ez = e - shift;
        if (ez < 0) {
            shift = e;
            ez = 0;
        }
        return f64_pack(sign, ez, ref_u64_shl(d, u32(shift)));
    }
    sa = ref_u64_shl(sa, 10u);
    sb = ref_u64_shl(sb, 10u);
    var ez: i32;
    var sz: U64;
    if (diff < 0) {
        sign = !sign;
        if (eb == 0x7ff) {
            if (u64_nonzero(sb)) { return ref_f64_propagate_nan(a, b); }
            return f64_inf(sign);
        }
        if (ea != 0) { sa = u64_add(sa, U64(0u, 0x40000000u)); } else { sa = u64_add(sa, sa); }
        sa = ref_u64_shr_jam(sa, u32(-diff));
        sb = U64(sb.x, sb.y | 0x40000000u);
        ez = eb;
        sz = u64_sub(sb, sa);
    } else {
        if (ea == 0x7ff) {
            if (u64_nonzero(sa)) { return ref_f64_propagate_nan(a, b); }
            return a;
        }
        if (eb != 0) { sb = u64_add(sb, U64(0u, 0x40000000u)); } else { sb = u64_add(sb, sb); }
        sb = ref_u64_shr_jam(sb, u32(diff));
        sa = U64(sa.x, sa.y | 0x40000000u);
        ez = ea;
        sz = u64_sub(sa, sb);
    }
    return ref_f64_norm_round_pack(sign, ez - 1, sz);
}
fn ref_f64_add(a: F64, b: F64) -> F64 {
    let sa = f64_signbit(a);
    if (sa == f64_signbit(b)) { return ref_f64_add_mags(a, b, sa); }
    return ref_f64_sub_mags(a, b, sa);
}
fn ref_f64_sub(a: F64, b: F64) -> F64 {
    let sa = f64_signbit(a);
    if (sa != f64_signbit(b)) { return ref_f64_add_mags(a, b, sa); }
    return ref_f64_sub_mags(a, b, sa);
}
fn ref_f64_mul(a: F64, b: F64) -> F64 {
    let sign = ((a.y ^ b.y) & F64_SIGN) != 0u;
    var ea = f64_exp(a);
    var eb = f64_exp(b);
    var sa = f64_frac(a);
    var sb = f64_frac(b);
    if (ea == 0x7ff) {
        if (u64_nonzero(sa) || (eb == 0x7ff && u64_nonzero(sb))) { return ref_f64_propagate_nan(a, b); }
        if (eb == 0 && !u64_nonzero(sb)) { return F64_DEFAULT_NAN; }
        return f64_inf(sign);
    }
    if (eb == 0x7ff) {
        if (u64_nonzero(sb)) { return ref_f64_propagate_nan(a, b); }
        if (ea == 0 && !u64_nonzero(sa)) { return F64_DEFAULT_NAN; }
        return f64_inf(sign);
    }
    if (ea == 0) {
        if (!u64_nonzero(sa)) { return f64_signed_zero(sign); }
        let n = ref_f64_norm_subnormal(sa);
        ea = i32(n.x);
        sa = n.yz;
    }
    if (eb == 0) {
        if (!u64_nonzero(sb)) { return f64_signed_zero(sign); }
        let n = ref_f64_norm_subnormal(sb);
        eb = i32(n.x);
        sb = n.yz;
    }
    var ez = ea + eb - 0x3ff;
    sa = ref_u64_shl(U64(sa.x, sa.y | 0x100000u), 10u);
    sb = ref_u64_shl(U64(sb.x, sb.y | 0x100000u), 11u);
    let p = u64_mul_wide(sa, sb);
    var sz = U64(p.z | select(0u, 1u, (p.x | p.y) != 0u), p.w);
    if (u64_lt(sz, U64(0u, 0x40000000u))) {
        ez = ez - 1;
        sz = ref_u64_shl(sz, 1u);
    }
    return ref_f64_round_pack(sign, ez, sz);
}
