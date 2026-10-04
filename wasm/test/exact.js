// Checks clz32 against countLeadingZeros over every u32, and the
// estimate-and-correct fp_div, fp_sqrt and f64_div against their
// bit-loop fallbacks (the previous implementation, verified bit for bit against
// the CPU): fp_sqrt over every binary32 input, the divisions over random,
// near-special and structured operand pairs. The branch-free f64_add, f64_sub
// and f64_mul are compared with the SoftFloat code they replaced
// (wasm/test/f64-reference.wgsl), NaN payloads included. `PAGE=exact.html GPU=hardware node wasm/test/run.mjs`.
// Work is submitted in small batches so the desktop stays responsive.
const text = async (url) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.text();
};

const KERNEL = `
struct Params { base: u32, pad: u32, mode: u32, seed: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> report: array<atomic<u32>, 8>;
fn hash(x: u32) -> u32 {
  var v = x * 747796405u + 2891336453u;
  v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
  return (v >> 22u) ^ v;
}
fn fail(a: u32, b: u32, got: u32, want: u32) {
  if (atomicAdd(&report[0], 1u) == 0u) {
    atomicStore(&report[1], a); atomicStore(&report[2], b);
    atomicStore(&report[3], got); atomicStore(&report[4], want);
  }
}
@compute @workgroup_size(64) fn sqrt_main(@builtin(global_invocation_id) id: vec3<u32>) {
  if ((id.x & 63u) == 0u) { atomicAdd(&report[5], 64u); }
  let bits = params.base + id.y * 2097152u + id.x;
  let x = bitcast<f32>(bits);
  let got = bitcast<u32>(fp_sqrt(x));
  let want = bitcast<u32>(old_fp_sqrt(x));
  if (got != want) { fail(bits, 0u, got, want); }
}
@compute @workgroup_size(64) fn clz_main(@builtin(global_invocation_id) id: vec3<u32>) {
  if ((id.x & 63u) == 0u) { atomicAdd(&report[5], 64u); }
  let x = params.base + id.y * 2097152u + id.x;
  let got = clz32(x);
  let want = countLeadingZeros(x);
  if (got != want) { fail(x, 0u, got, want); }
}
@compute @workgroup_size(64) fn div_main(@builtin(global_invocation_id) id: vec3<u32>) {
  if ((id.x & 63u) == 0u) { atomicAdd(&report[5], 64u); }
  let n = params.base + id.y * 2097152u + id.x;
  let h1 = hash(n ^ params.seed);
  let h2 = hash(h1 ^ 0x9e3779b9u);
  let h3 = hash(h2 + n);
  var a = h1;
  var b = h2;
  switch params.mode {
    // Uniform random bit patterns (all classes).
    case 0u: {}
    // Normal operands with nearby exponents: every quotient normal.
    case 1u: { a = (h1 & 0x807fffffu) | ((100u + (h3 & 63u)) << 23u); b = (h2 & 0x807fffffu) | ((100u + ((h3 >> 8u) & 63u)) << 23u); }
    // Significands within a few units of each other or of twice the other:
    // quotients next to 1 and 2, where the estimate is least certain.
    case 2u: {
      let sb = h2 & 0x7fffffu;
      let delta = (h3 & 15u) - 8u;
      let sa = select(sb + delta, (sb << 1u) + delta, (h3 & 16u) != 0u) & 0x7fffffu;
      a = (h1 & 0x80000000u) | (127u << 23u) | sa; b = (h2 & 0x80000000u) | (127u << 23u) | sb;
    }
    // Subnormal and overflow/underflow boundaries of the result.
    case 3u: { a = (h1 & 0x80ffffffu); b = (h2 & 0x807fffffu) | ((1u + (h3 & 255u)) << 23u); if ((h3 & 256u) != 0u) { a = (h1 & 0x807fffffu) | ((200u + (h3 >> 24u) % 55u) << 23u); b = h2 & 0x80ffffffu; } }
    // Exact and halfway quotients: a = b * q with short significands.
    default: {
      let q = (h3 & 0xfffu) | 0x1000u;
      let sb = (h2 & 0xfffu) | 0x1000u;
      let product = q * sb; // < 2^26
      let shift = countLeadingZeros(product) - 8u;
      let sa = select(product >> (0u - shift), product << shift, shift < 32u);
      a = (h1 & 0x80000000u) | ((110u + (h1 & 31u)) << 23u) | ((sa << ((h3 >> 13u) & 1u)) & 0x7fffffu);
      b = (h2 & 0x80000000u) | ((110u + (h2 & 31u)) << 23u) | ((sb << 11u) & 0x7fffffu);
    }
  }
  let got = bitcast<u32>(fp_div(bitcast<f32>(a), bitcast<f32>(b)));
  let want = bitcast<u32>(old_fp_div(bitcast<f32>(a), bitcast<f32>(b)));
  let nan = (got & 0x7fffffffu) > 0x7f800000u && (want & 0x7fffffffu) > 0x7f800000u;
  if (got != want && !nan) { fail(a, b, got, want); }
}
@compute @workgroup_size(64) fn div64_main(@builtin(global_invocation_id) id: vec3<u32>) {
  if ((id.x & 63u) == 0u) { atomicAdd(&report[5], 64u); }
  let n = params.base + id.y * 2097152u + id.x;
  let h1 = hash(n ^ params.seed);
  let h2 = hash(h1 ^ 0x9e3779b9u);
  let h3 = hash(h2 + n);
  let h4 = hash(h3 ^ 0x85ebca6bu);
  var a = F64(h1, h2);
  var b = F64(h3, h4);
  switch params.mode {
    // Uniform random bit patterns (all classes).
    case 0u: {}
    // Normal operands with nearby exponents.
    case 1u: { a.y = (h2 & 0x800fffffu) | ((1000u + (h4 & 63u)) << 20u); b.y = (h4 & 0x800fffffu) | ((1000u + ((h4 >> 8u) & 63u)) << 20u); }
    // Significands within a few units of each other or of twice the other.
    case 2u: {
      b.y = (h4 & 0x800fffffu) | (1023u << 20u);
      let delta = (h2 & 15u) - 8u;
      let twice = (h2 & 16u) != 0u;
      let sb = select(b, U64(b.x << 1u, (b.y << 1u) | (b.x >> 31u)), twice);
      let sa = u64_add(sb, U64(delta, select(0u, 0xffffffffu, (delta & 0x80000000u) != 0u)));
      a = F64(sa.x, (h1 & 0x80000000u) | (1023u << 20u) | (sa.y & 0xfffffu));
    }
    // Subnormal operands and results.
    case 3u: { a.y = h2 & 0x801fffffu; b.y = (h4 & 0x800fffffu) | ((1u + (h1 & 1023u)) << 20u); if ((h1 & 1024u) != 0u) { a.y = (h2 & 0x800fffffu) | ((1500u + (h1 >> 22u) % 540u) << 20u); b.y = h4 & 0x801fffffu; } }
    // Exact and near-halfway quotients: a = b * q with a short q, low bit nudged.
    default: {
      b = F64(h3 & 0xfffff000u, (h4 & 0x800fffffu) | ((900u + (h1 & 255u)) << 20u));
      a = f64_mul(b, f64_from_u32((h2 & 0xffffu) | 1u));
      a.x = a.x + (h1 >> 30u) - 1u;
    }
  }
  let got = f64_div(a, b);
  let want = old_f64_div(a, b);
  let nan = f64_isnan(got) && f64_isnan(want);
  if (any(got != want) && !nan) { fail(a.y, b.y, got.y, want.y); }
}
// A special or boundary operand: signed zeros, infinities, NaNs with random
// payloads, extreme subnormals and normals, or a random value.
fn edge64(h: u32, r: F64) -> F64 {
  let s = h & 0x80000000u;
  switch (h & 15u) {
    case 0u: { return F64(0u, s); }
    case 1u: { return F64(0u, s | 0x7ff00000u); }
    case 2u: { return F64(r.x, s | 0x7ff00000u | (r.y & 0xfffffu) | 0x1u); }
    case 3u: { return F64(1u, s); }
    case 4u: { return F64(0xffffffffu, s | 0xfffffu); }
    case 5u: { return F64(0u, s | 0x100000u); }
    case 6u: { return F64(0xffffffffu, s | 0x7fefffffu); }
    case 7u: { return F64(0u, s | 0x3ff00000u); }
    default: { return r; }
  }
}
@compute @workgroup_size(64) fn arith_main(@builtin(global_invocation_id) id: vec3<u32>) {
  if ((id.x & 63u) == 0u) { atomicAdd(&report[5], 64u); }
  let n = params.base + id.y * 2097152u + id.x;
  let h1 = hash(n ^ params.seed);
  let h2 = hash(h1 ^ 0x9e3779b9u);
  let h3 = hash(h2 + n);
  let h4 = hash(h3 ^ 0x85ebca6bu);
  let h5 = hash(h4 + 0x27d4eb2fu);
  var a = F64(h1, h2);
  var b = F64(h3, h4);
  let op = params.mode >> 3u;
  let ea = 1u + (h5 & 2045u);
  switch params.mode & 7u {
    // Uniform random bit patterns (all classes).
    case 0u: {}
    // Normal operands whose exponents differ by 0 to 63; products reach
    // overflow and underflow.
    case 1u: {
      let gap = (h5 >> 11u) & 63u;
      let eb = select(ea - gap, ea + gap, ea < 1024u);
      a.y = (h2 & 0x800fffffu) | (ea << 20u); b.y = (h4 & 0x800fffffu) | (eb << 20u);
    }
    // Near-equal magnitudes: b = a plus or minus a few units, or the
    // neighbouring binade, with any signs (massive cancellation).
    case 2u: {
      a.y = (h2 & 0x800fffffu) | (ea << 20u);
      let delta = (h5 >> 12u) & 31u;
      let m = u64_add(F64(a.x, a.y & 0x7fffffffu), U64(delta - 16u, select(0u, 0xffffffffu, delta < 16u)));
      b = F64(m.x, (m.y & 0x7fffffffu) | (h4 & 0x80000000u));
      if ((h5 & 0x20000u) != 0u) { b = F64(b.x ^ (h3 & 7u), b.y); }
    }
    // Subnormal, tiny and huge operands: underflow, overflow, gradual loss.
    case 3u: {
      let lowa = (h5 & 0x100u) != 0u;
      a.y = (h2 & 0x800fffffu) | (select(2046u - (h5 >> 20u) % 64u, (h5 >> 20u) % 64u, lowa) << 20u);
      let lowb = (h5 & 0x200u) != 0u;
      b.y = (h4 & 0x800fffffu) | (select(select(2046u - (h5 >> 26u), (h5 >> 26u), lowb), 1023u - (h5 >> 26u), (h5 & 0x400u) != 0u) << 20u);
    }
    // Specials against specials and random values.
    case 4u: { a = edge64(h5, a); b = edge64(h5 >> 4u, b); }
    // Short significands: exact sums and products, and exact ties (a gap of
    // 52 to 55 for sums; 27-bit significands for products).
    default: {
      let gap = 52u + ((h5 >> 11u) & 3u);
      let short_a = h1 & 0xfc000000u;
      a = F64(select(h1, short_a, op == 2u), (h2 & 0x800fffffu) | (ea << 20u));
      let eb = select(u32(max(i32(ea) - i32(gap), 1)), 900u + (h5 >> 24u), op == 2u);
      b = F64(select(h3 & (0xffffffffu << ((h5 >> 13u) & 31u)), h3 & 0xfc000000u, op == 2u), (h4 & 0x800fffffu & select(0xffffffffu, 0xfff00000u | (h4 & 0xfu), (h5 & 0x10000u) != 0u)) | (eb << 20u));
    }
  }
  var got: F64;
  var want: F64;
  switch op {
    case 0u: { got = f64_add(a, b); want = ref_f64_add(a, b); }
    case 1u: { got = f64_sub(a, b); want = ref_f64_sub(a, b); }
    default: { got = f64_mul(a, b); want = ref_f64_mul(a, b); }
  }
  // NaN payloads count: the CPU's x86 propagation is reproduced bit for bit.
  if (any(got != want)) { fail(a.y, b.y, got.y, want.y); }
}`;

export async function run(root) {
  const opts = window.altdTestOptions ?? {};
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const [f64, float, helpers, reference] = await Promise.all(
    ["src/wasm/sim/f64.wgsl", "src/wasm/float.wgsl", "src/wasm/sim/helpers.wgsl", "wasm/test/f64-reference.wgsl"].map((p) => text(root + p)),
  );
  const between = (source, start) => {
    const at = source.indexOf(start);
    const end = source.indexOf("\nfn ", at + 1);
    if (at < 0) throw new Error(`missing ${start}`);
    return source.slice(at, end < 0 ? undefined : end + 1);
  };
  const sqrt = between(helpers, "fn fp_sqrt(") + between(helpers, "fn fp_sqrt_general(");
  const div = between(float, "fn fp_div_general(");
  const force = (source, from, to, guard) => {
    if (!source.includes(guard)) throw new Error(`missing ${guard}`);
    return source.replace(from, to).replace(guard, guard.replace(/if \(/, "if (true || "));
  };
  const oldSqrt = force(between(helpers, "fn fp_sqrt_general("), "fn fp_sqrt_general(", "fn old_fp_sqrt(", "if (!(abs(off)<");
  const oldDiv = force(div, "fn fp_div_general(", "fn old_fp_div(", "if (!(abs(off) <");
  const div64 = between(f64, "fn f64_div(");
  const oldDiv64 = force(div64, "fn f64_div(", "fn old_f64_div(", "if (!exact) {");
  const code = f64 + reference + oldDiv64 + KERNEL + float + oldDiv + sqrt + oldSqrt;
  const module = device.createShaderModule({ code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === "error");
  if (errors.length) throw new Error(errors.map((m) => `${m.lineNum}: ${m.message}`).join("\n"));
  const uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const report = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const read = device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const PER = 2097152 * 8; // invocations per dispatch (2^24)
  const out = { ok: true };
  const check = async (entry, mode, total, label) => {
    if (opts.kernels && !opts.kernels.includes(entry)) return;
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: entry } });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [uniform, report].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    device.queue.writeBuffer(report, 0, new Uint32Array(8));
    const t = performance.now();
    for (let base = 0; base < total; base += PER) {
      device.queue.writeBuffer(uniform, 0, new Uint32Array([base >>> 0, 0, mode, 0x5eed + mode]));
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(32768, 8); // 2^21 x 8 = 2^24
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      await new Promise((r) => setTimeout(r, 2));
    }
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(report, 0, read, 0, 32);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const r = new Uint32Array(read.getMappedRange().slice(0));
    read.unmap();
    const hex = (v) => "0x" + v.toString(16).padStart(8, "0");
    // The counter wraps at 2^32 (the exhaustive sqrt run): compare modulo.
    out[label] = { checked: total, ran: r[5], mismatches: r[0], ms: Math.round(performance.now() - t) };
    if (r[0] || r[5] !== total % 2 ** 32) {
      out.ok = false;
      out[label].first = { a: hex(r[1]), b: hex(r[2]), got: hex(r[3]), want: hex(r[4]) };
    }
  };
  const divCount = opts.ticks ? opts.ticks * PER : 2 ** 28;
  await check("clz_main", 0, 2 ** 32, "clz_all");
  await check("sqrt_main", 0, 2 ** 32, "sqrt_all");
  for (let mode = 0; mode < 5; mode++) await check("div_main", mode, divCount, `div_mode${mode}`);
  for (let mode = 0; mode < 5; mode++) await check("div64_main", mode, divCount, `div64_mode${mode}`);
  for (const [op, name] of ["add", "sub", "mul"].entries())
    for (let mode = 0; mode < 6; mode++) await check("arith_main", op * 8 + mode, divCount, `${name}64_mode${mode}`);
  device.destroy();
  return out;
}
