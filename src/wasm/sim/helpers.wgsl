struct SimParams {
    population: u32, ticks: u32, start_tick: u32, start_batch: u32,
    batches_per_tick: u32, stats_phase: u32, stop_inactive: u32, eliminate_wall: u32,
    eliminate_idle: u32, has_limit: u32, limit: F64,
    world_offset: u32, vehicle_offset: u32, sensors_offset: u32, sensor_count: u32,
    layers: u32, stride: u32, shape_offset: u32, control_offset: u32,
    needs_path: u32, pad: u32, reserved_a: u32, reserved_b: u32,
}
@group(0) @binding(0) var<storage, read_write> cars: array<u32>;
@group(0) @binding(1) var<storage, read_write> agents: array<u32>;
@group(0) @binding(2) var<storage, read> world_data: array<u32>;
@group(0) @binding(3) var<storage, read> networks: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> results: array<u32>;
@group(0) @binding(5) var<uniform> params: SimParams;
fn f64_ne(a: F64,b: F64) -> bool { return !f64_eq(a,b); }
fn f64_max(a: F64,b: F64) -> F64 { return select(a,b,f64_lt(a,b) || f64_isnan(a)); }
fn u64_ne(a: U64,b: U64) -> bool { return !u64_eq(a,b); }
fn u64_gt(a: U64,b: U64) -> bool { return u64_lt(b,a); }
fn u64_ge(a: U64,b: U64) -> bool { return u64_le(b,a); }
fn i64_gt(a: U64,b: U64) -> bool { return i64_lt(b,a); }
fn i64_le(a: U64,b: U64) -> bool { return !i64_lt(b,a); }
fn i64_ge(a: U64,b: U64) -> bool { return !i64_lt(a,b); }
fn i64_shr(a: U64,n: u32) -> U64 {
 if (n == 0u) { return a; }
 let sign = select(0u,0xffffffffu,i64_is_negative(a));
 if (n >= 32u) { return U64(u32(i32(a.y) >> (n-32u)),sign); }
 return U64((a.x >> n) | (a.y << (32u-n)),u32(i32(a.y) >> n));
}
fn u64_mul_low(a: U64,b: U64) -> U64 { return u64_mul(a,b); }
fn u64_mod_small(a: U64,b: u32) -> U64 { return u64_sub(a,u64_mul(u64_div_small(a,b),U64(b,0u))); }
fn fp_isnan(a: f32) -> bool { return (bitcast<u32>(a) & 0x7fffffffu) > 0x7f800000u; }
fn fp_isinf(a: f32) -> bool { return (bitcast<u32>(a) & 0x7fffffffu) == 0x7f800000u; }
fn fp_isfinite(a: f32) -> bool { return (bitcast<u32>(a) & 0x7f800000u) != 0x7f800000u; }
fn fp_signbit(a: f32) -> bool { return (bitcast<u32>(a) & 0x80000000u) != 0u; }
fn fp_copysign(a: f32,b: f32) -> f32 { return bitcast<f32>((bitcast<u32>(a) & 0x7fffffffu) | (bitcast<u32>(b) & 0x80000000u)); }
// x86 NaN propagation: the first NaN operand, quieted. altd_invalid is the
// default NaN 0xffc00000 for non-NaN operands.
fn fp_nan_operand(a: f32,b: f32) -> f32 { return bitcast<f32>(select(bitcast<u32>(b),bitcast<u32>(a),fp_isnan(a)) | 0x00400000u); }
fn fp_invalid(a: f32) -> f32 { return bitcast<f32>(select(0xffc00000u,bitcast<u32>(a) | 0x00400000u,fp_isnan(a))); }
fn f64_nan_operand(a: F64,b: F64) -> F64 { let r=select(b,a,f64_isnan(a)); return F64(r.x,r.y | 0x00080000u); }
fn f64_invalid(a: F64) -> F64 { return select(F64(0u,0xfff80000u),F64(a.x,a.y | 0x00080000u),f64_isnan(a)); }
fn fp_mod(a: f32,b: f32) -> f32 {
 let ad=f64_from_f32(a); let bd=f64_from_f32(b);
 return f64_to_f32(f64_sub(ad,f64_mul(f64_trunc(f64_div(ad,bd)),bd)));
}
// Integer square root of a normalized binary32 significand, rounded once.
// Positive normal inputs take this branch-free path; the rest, and devices
// whose estimate is off, fp_sqrt_general. The root stays below 2^24, and the
// exponent field is added to it with its leading bit.
fn fp_sqrt(a: f32) -> f32 {
 let bits=bitcast<u32>(a);
 if (bits-0x800000u >= 0x7f000000u) { return fp_sqrt_general(a); }
 let exponent=i32(bits >> 23u)-127;
 let radicand=u64_shl(U64((bits & 0x7fffffu) | 0x800000u,0u),23u+u32(exponent & 1));
 let wide=f32(radicand.y)*4294967296.0+f32(radicand.x);
 var root=u32(sqrt(wide));
 let off=fp_sub(fp_mul(f32(root),f32(root)),wide);
 var remainder=i32(radicand.x-root*root);
 for (var i=0u; i<2u; i=i+1u) { let low=remainder<0; root=select(root,root-1u,low); remainder=select(remainder,remainder+i32(2u*root+1u),low); }
 for (var i=0u; i<2u; i=i+1u) { let high=remainder>i32(2u*root); remainder=select(remainder,remainder-i32(2u*root+1u),high); root=select(root,root+1u,high); }
 if (!(abs(off)<1073741824.0) || remainder<0 || u32(remainder)>2u*root) { return fp_sqrt_general(a); }
 root=root+select(0u,1u,u32(remainder)>root);
 return bitcast<f32>((u32((exponent >> 1u)+126) << 23u)+root);
}
fn fp_sqrt_general(a: f32) -> f32 {
 let bits=bitcast<u32>(a); let mag=bits & 0x7fffffffu;
 if (mag == 0u || bits == 0x7f800000u) { return a; }
 if ((bits & 0x80000000u) != 0u || mag > 0x7f800000u) { return bitcast<f32>(0x7fc00000u ^ params.pad); }
 var exponent=i32(mag >> 23u)-127;
 var sig=(mag & 0x7fffffu) | 0x800000u;
 if (mag < 0x800000u) {
   let shift=clz32(mag)-8u;
   sig=mag << shift; exponent=-126-i32(shift);
 }
 let radicand=u64_shl(U64(sig,0u),23u+u32(exponent & 1));
 // root=floor(sqrt(radicand)) from the hardware estimate, corrected as in
 // fp_div: the remainder is exact modulo 2^32 while the correctly rounded
 // square (exact radicand, 24 significant bits) bounds it within an i32, and
 // a device that is further off than the steps reach takes the bit loop.
 let wide=f32(radicand.y)*4294967296.0+f32(radicand.x);
 var root=u32(sqrt(wide));
 let off=fp_sub(fp_mul(f32(root),f32(root)),wide);
 var remainder=i32(radicand.x-root*root);
 for (var i=0u; i<2u; i=i+1u) { let low=remainder<0; root=select(root,root-1u,low); remainder=select(remainder,remainder+i32(2u*root+1u),low); }
 for (var i=0u; i<2u; i=i+1u) { let high=remainder>i32(2u*root); remainder=select(remainder,remainder-i32(2u*root+1u),high); root=select(root,root+1u,high); }
 if (!(abs(off)<1073741824.0) || remainder<0 || u32(remainder)>2u*root) {
   root=0u;
   for (var bit=0x800000u; bit != 0u; bit=bit >> 1u) {
     let probe=root | bit;
     if (u64_le(mul32(probe,probe),radicand)) { root=probe; }
   }
   remainder=i32(u64_sub(radicand,mul32(root,root)).x);
 }
 if (u32(remainder)>root) { root=root+1u; }
 var out_exp=(exponent >> 1u)+127;
 if (root == 0x1000000u) { root=root >> 1u; out_exp=out_exp+1; }
 return bitcast<f32>((u32(out_exp) << 23u) | (root & 0x7fffffu));
}
fn v2_add(a: vec2<f32>,b: vec2<f32>) -> vec2<f32> { return vec2<f32>(fp_add(a.x,b.x),fp_add(a.y,b.y)); }
fn v2_sub(a: vec2<f32>,b: vec2<f32>) -> vec2<f32> { return vec2<f32>(fp_sub(a.x,b.x),fp_sub(a.y,b.y)); }
fn v2_mul(a: vec2<f32>,b: vec2<f32>) -> vec2<f32> { return vec2<f32>(fp_mul(a.x,b.x),fp_mul(a.y,b.y)); }
fn v2_div(a: vec2<f32>,b: vec2<f32>) -> vec2<f32> { return vec2<f32>(fp_div(a.x,b.x),fp_div(a.y,b.y)); }
