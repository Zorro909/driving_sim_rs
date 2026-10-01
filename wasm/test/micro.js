// Single-thread latency microbenchmarks for the WGSL soft-float and memory
// patterns the simulator kernels are built from. Each kernel runs one
// invocation through a dependent chain, so GPU time / iterations is the
// serial latency of one operation. `PAGE=micro.html GPU=hardware node wasm/test/run.mjs`.
const text = async (url) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.text();
};

const HEADER = `
struct Params { n: u32, pad: u32, a: u32, b: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<storage, read_write> table: array<u32>;
var<private> scratch: array<u32, 1024>;
`;

const repeat = (k, body) => Array.from({ length: k }, () => body).join("\n");
// Loop bodies; x is an F64 (or f32 for fp_*), y a constant operand the
// compiler cannot see through (it comes from the uniform). x is per lane
// (params.pad is zero) so the chain runs on the vector ALU like the simulator
// kernels, rather than being scalarized; one wave of 64 lanes runs.
const f64Kernel = (body, unroll = 1) => `
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane: u32) {
  var x = F64(params.a + (lane & params.pad), 0x3ff80000u);
  let y = F64(params.b, 0x3ff00001u);
  for (var i = 0u; i < params.n; i = i + 1u) {
${repeat(unroll, body)}
  }
  out[0] = x.x; out[1] = x.y;
}`;
const f32Kernel = (body, unroll = 1) => `
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane: u32) {
  var x = bitcast<f32>(0x3fc00000u + params.a + (lane & params.pad));
  let y = bitcast<f32>(0x3f800001u + params.b);
  for (var i = 0u; i < params.n; i = i + 1u) {
${repeat(unroll, body)}
  }
  out[0] = bitcast<u32>(x);
}`;
// Divergent variants: every lane has its own operands with random signs and
// exponents, so data-dependent branches split the wave as in the simulator.
// Two operations per iteration keep the magnitudes bounded.
const mixedKernel = (body, z) => `
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane: u32) {
  let h = (lane + params.a) * 2654435761u;
  let g = h * 747796405u + 2891336453u;
  var x = F64(h, 0x3ff00000u | (h >> 12u));
  let y = F64(g, ((g & 1u) << 31u) | ((0x3fdu + (g >> 8u) % 5u) << 20u) | ((g >> 11u) & 0xfffffu) | params.pad);
  let z = ${z};
  for (var i = 0u; i < params.n; i = i + 1u) {
${body}
  }
  out[0] = x.x; out[1] = x.y;
}`;
const u32Kernel = (body) => `
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane: u32) {
  var x = 0x00123456u + params.a + (lane & params.pad);
  let y = 0x00800000u + params.b;
  for (var i = 0u; i < params.n; i = i + 1u) {
${body}
  }
  out[0] = x;
}`;
const chainKernel = (load, fill) => `
@compute @workgroup_size(1) fn main() {
  ${fill}
  var j = params.a;
  for (var i = 0u; i < params.n; i = i + 1u) { j = ${load}; }
  out[0] = j;
}`;

const KERNELS = {
  f64_add: f64Kernel("x = f64_add(x, y);"),
  f64_mul: f64Kernel("x = f64_mul(x, y);"),
  f64_div: f64Kernel("x = f64_div(x, y);"),
  f64_add_mixed: mixedKernel("x = f64_add(x, y); x = f64_sub(x, z);", "F64(h ^ g, ((h & 2u) << 30u) | ((0x3fdu + (h >> 9u) % 5u) << 20u) | (g & 0xfffffu))"),
  f64_mul_mixed: mixedKernel("x = f64_mul(x, y); x = f64_mul(x, z);", "f64_div(F64_ONE, y)"),
  fp_add: f32Kernel("x = fp_add(x, y);"),
  fp_div: f32Kernel("x = fp_div(x, y);"),
  fp_sqrt: f32Kernel("x = fp_sqrt(fp_add(x, y));"),
  // Leading-zero counts on a dependent chain (x stays below 2^24): the
  // builtins are polyfilled, f64.wgsl's clz32 is not.
  clz: u32Kernel("x = (x ^ y) >> (countLeadingZeros(x | y) - 7u);"),
  first_leading_bit: u32Kernel("x = (x ^ y) >> (24u - firstLeadingBit(x | y));"),
  clz32: u32Kernel("x = (x ^ y) >> (clz32(x | y) - 7u);"),
  shift_only: u32Kernel("x = (x ^ y) >> (x & 1u);"),
  // The same dependent f64_add chain, but 64 / 512 copies straight-line per
  // iteration: per-op time grows only if instruction fetch misses.
  f64_add_x64: f64Kernel("x = f64_add(x, y);", 64),
  f64_add_x512: f64Kernel("x = f64_add(x, y);", 512),
  f64_mul_x512: f64Kernel("x = f64_mul(x, y);", 512),
  private_chain: chainKernel(
    "scratch[j & 1023u]",
    "for (var k = 0u; k < 1024u; k = k + 1u) { scratch[k] = table[k]; }",
  ),
  storage_chain: chainKernel("table[j & 1023u]", ""),
};
// Iterations of the loop per kernel (each run is also repeated below).
const UNROLL = { f64_add_x64: 64, f64_add_x512: 512, f64_mul_x512: 512, f64_add_mixed: 2, f64_mul_mixed: 2 };

export async function run(root) {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
  const f64 = await text(root + "src/wasm/sim/f64.wgsl");
  const float = await text(root + "src/wasm/float.wgsl");
  const helpers = await text(root + "src/wasm/sim/helpers.wgsl");
  const sqrtAt = helpers.indexOf("fn fp_sqrt(");
  const generalAt = helpers.indexOf("fn fp_sqrt_general(");
  const sqrt = helpers.slice(sqrtAt, helpers.indexOf("\nfn ", sqrtAt) + 1) + helpers.slice(generalAt, helpers.indexOf("\nfn ", generalAt) + 1);
  const uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const out = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
  const table = device.createBuffer({ size: 4096, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  // A single 1024-cycle permutation so every load depends on the previous one.
  const order = Array.from({ length: 1024 }, (_, i) => i);
  for (let i = 1023; i > 0; i--) {
    const k = (i * 7919 + 13) % (i + 1);
    [order[i], order[k]] = [order[k], order[i]];
  }
  const next = new Uint32Array(1024);
  for (let i = 0; i < 1024; i++) next[order[i]] = order[(i + 1) % 1024];
  device.queue.writeBuffer(table, 0, next);
  const query = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const layout = device.createPipelineLayout({
    bindGroupLayouts: [
      device.createBindGroupLayout({
        entries: ["uniform", "storage", "storage"].map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })),
      }),
    ],
  });
  const only = window.altdTestOptions?.kernels;
  const results = {};
  for (const [name, body] of Object.entries(KERNELS)) {
    if (only && !only.includes(name)) continue;
    const module = device.createShaderModule({ code: f64 + HEADER + float + sqrt + body });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === "error");
    if (errors.length) throw new Error(name + ": " + errors.map((m) => m.message).join("\n"));
    const pipeline = await device.createComputePipelineAsync({ layout, compute: { module, entryPoint: "main" } });
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [uniform, out, table].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const time = async (n) => {
      device.queue.writeBuffer(uniform, 0, new Uint32Array([n, 0, 1, 0]));
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass({ timestampWrites: { querySet: query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(1);
      pass.end();
      encoder.resolveQuerySet(query, 0, 2, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, read, 0, 16);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(read.getMappedRange().slice(0));
      read.unmap();
      return Number(t[1] - t[0]) / 1000; // µs
    };
    await time(1);
    // Grow the iteration count until one dispatch takes ~2 ms (short enough
    // to leave the desktop responsive), then take the best of three.
    let n = 1;
    let us = await time(n);
    while (us < 2000 && n < 1 << 24) {
      n *= Math.max(2, Math.min(16, Math.floor(2000 / Math.max(us, 1))));
      us = await time(n);
    }
    const base = await time(1);
    let best = Infinity;
    for (let r = 0; r < 3; r++) best = Math.min(best, await time(n));
    const ops = n * (UNROLL[name] || 1);
    results[name] = { iterations: n, ops, us: +best.toFixed(1), nsPerOp: +(((best - base) * 1000) / ops).toFixed(2) };
  }
  device.destroy();
  return { ok: true, results };
}
