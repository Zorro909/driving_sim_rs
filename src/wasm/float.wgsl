// params.pad is uploaded as zero. Passing every operation through an integer
// XOR with this runtime value prevents contraction across rounding boundaries.
fn fp_round(value: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(value) ^ params.pad);
}
fn fp_add(a: f32, b: f32) -> f32 { return fp_round(a + b); }
fn fp_sub(a: f32, b: f32) -> f32 { return fp_round(a - b); }
fn fp_mul(a: f32, b: f32) -> f32 { return fp_round(a * b); }
// WGSL division permits 2.5 ULP error. Divide binary32 significands with
// integers instead, then round once to nearest, ties to even. Each remainder
// is smaller than a 24-bit divisor, so doubling it never overflows u32.
fn fp_div(a: f32, b: f32) -> f32 {
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
    var remainder = numerator - divisor;
    var quotient = 1u;
    for (var i = 0u; i < 23u; i = i + 1u) {
        remainder = remainder << 1u;
        quotient = quotient << 1u;
        if (remainder >= divisor) {
            remainder = remainder - divisor;
            quotient = quotient | 1u;
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
