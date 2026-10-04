// params.pad is uploaded as zero. Passing every operation through an integer
// XOR with this runtime value prevents contraction across rounding boundaries.
fn fp_round(value: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(value) ^ params.pad);
}
fn fp_add(a: f32, b: f32) -> f32 { return fp_round(a + b); }
fn fp_sub(a: f32, b: f32) -> f32 { return fp_round(a - b); }
fn fp_mul(a: f32, b: f32) -> f32 { return fp_round(a * b); }
// WGSL division permits 2.5 ULP error. Divide binary32 significands exactly
// with integers instead, then round once to nearest, ties to even. Normal
// operands whose quotient is normal take this branch-free path; zeros,
// infinities, NaNs, subnormals and quotients near the exponent limits take
// fp_div_general. The exponent field is added to the rounded 24-bit quotient,
// so a quotient that rounds up to 2^24 carries into the exponent.
fn fp_div(a: f32, b: f32) -> f32 {
    let ab = bitcast<u32>(a);
    let bb = bitcast<u32>(b);
    let ae = (ab >> 23u) & 0xffu;
    let be = (bb >> 23u) & 0xffu;
    let biased = i32(ae) - i32(be) + 127;
    if (ae == 0u || be == 0u || ae == 0xffu || be == 0xffu || biased < 2 || biased > 254) {
        return fp_div_general(a, b);
    }
    var numerator = (ab & 0x7fffffu) | 0x800000u;
    let divisor = (bb & 0x7fffffu) | 0x800000u;
    let shift = numerator < divisor;
    numerator = select(numerator, numerator << 1u, shift);
    let exponent = biased - select(0, 1, shift);
    // As in fp_div_general: a hardware estimate corrected by two steps each way.
    var quotient = u32(f32(numerator) / f32(divisor) * 8388608.0);
    let off = fp_sub(fp_mul(f32(quotient), f32(divisor)), f32(numerator) * 8388608.0);
    var estimate = i32((numerator << 23u) - quotient * divisor);
    for (var i = 0u; i < 2u; i = i + 1u) {
        let low = estimate < 0;
        quotient = select(quotient, quotient - 1u, low);
        estimate = select(estimate, estimate + i32(divisor), low);
    }
    for (var i = 0u; i < 2u; i = i + 1u) {
        let high = estimate >= i32(divisor);
        quotient = select(quotient, quotient + 1u, high);
        estimate = select(estimate, estimate - i32(divisor), high);
    }
    if (!(abs(off) < 1073741824.0) || estimate < 0 || u32(estimate) >= divisor) { return fp_div_general(a, b); }
    let twice_remainder = u32(estimate) << 1u;
    quotient = quotient + select(0u, 1u, twice_remainder > divisor || (twice_remainder == divisor && (quotient & 1u) != 0u));
    return bitcast<f32>(((ab ^ bb) & 0x80000000u) | ((u32(exponent - 1) << 23u) + quotient));
}
fn fp_div_general(a: f32, b: f32) -> f32 {
    let ab = bitcast<u32>(a);
    let bb = bitcast<u32>(b);
    let sign = (ab ^ bb) & 0x80000000u;
    let am = ab & 0x7fffffffu;
    let bm = bb & 0x7fffffffu;
    let infinity = 0x7f800000u;
    if (am > infinity || bm > infinity || (am == infinity && bm == infinity) || (am == 0u && bm == 0u)) {
        return bitcast<f32>(0x7fc00000u ^ params.pad);
    }
    if (am == infinity || bm == 0u) { return bitcast<f32>(sign | infinity); }
    if (am == 0u || bm == infinity) { return bitcast<f32>(sign); }

    var ae = i32(am >> 23u);
    var be = i32(bm >> 23u);
    var numerator = am & 0x7fffffu;
    var divisor = bm & 0x7fffffu;
    if (ae == 0) {
        let shift = countLeadingZeros(numerator) - 8u;
        numerator = numerator << shift;
        ae = 1 - i32(shift);
    } else { numerator = numerator | 0x800000u; }
    if (be == 0) {
        let shift = countLeadingZeros(divisor) - 8u;
        divisor = divisor << shift;
        be = 1 - i32(shift);
    } else { divisor = divisor | 0x800000u; }

    var exponent = ae - be + 127;
    if (numerator < divisor) {
        numerator = numerator << 1u;
        exponent = exponent - 1;
    }
    if (exponent > 254) { return bitcast<f32>(sign | infinity); }
    if (exponent < -23) { return bitcast<f32>(sign); }
    // quotient = floor(numerator * 2^23 / divisor), a 24-bit integer, and the
    // remainder, from the hardware quotient corrected by up to two steps each
    // way without branching. The remainder is computed modulo 2^32, exact while
    // the true one fits an i32. The correctly rounded product proves that: it
    // is within 2^26 of the true remainder. A device whose division is further
    // off than the steps reach takes the bit loop.
    var quotient = u32(f32(numerator) / f32(divisor) * 8388608.0);
    let off = fp_sub(fp_mul(f32(quotient), f32(divisor)), f32(numerator) * 8388608.0);
    var estimate = i32((numerator << 23u) - quotient * divisor);
    for (var i = 0u; i < 2u; i = i + 1u) {
        let low = estimate < 0;
        quotient = select(quotient, quotient - 1u, low);
        estimate = select(estimate, estimate + i32(divisor), low);
    }
    for (var i = 0u; i < 2u; i = i + 1u) {
        let high = estimate >= i32(divisor);
        quotient = select(quotient, quotient + 1u, high);
        estimate = select(estimate, estimate - i32(divisor), high);
    }
    var remainder = u32(estimate);
    if (!(abs(off) < 1073741824.0) || estimate < 0 || remainder >= divisor) {
        remainder = numerator - divisor;
        quotient = 1u;
        for (var i = 0u; i < 23u; i = i + 1u) {
            remainder = remainder << 1u;
            quotient = quotient << 1u;
            if (remainder >= divisor) {
                remainder = remainder - divisor;
                quotient = quotient | 1u;
            }
        }
    }
    if (exponent <= 0) {
        // Shift the unrounded quotient into the subnormal range. The exact
        // remainder contributes a sticky bit, avoiding double rounding.
        let shift = u32(1 - exponent);
        let halfway = 1u << (shift - 1u);
        let discarded = quotient & ((1u << shift) - 1u);
        quotient = quotient >> shift;
        if (discarded > halfway || (discarded == halfway && (remainder != 0u || (quotient & 1u) != 0u))) {
            quotient = quotient + 1u;
        }
        return bitcast<f32>(sign | quotient);
    }
    let twice_remainder = remainder << 1u;
    if (twice_remainder > divisor || (twice_remainder == divisor && (quotient & 1u) != 0u)) {
        quotient = quotient + 1u;
        if (quotient == 0x1000000u) {
            quotient = quotient >> 1u;
            exponent = exponent + 1;
        }
    }
    return bitcast<f32>(sign | (u32(exponent) << 23u) | (quotient & 0x7fffffu));
}
