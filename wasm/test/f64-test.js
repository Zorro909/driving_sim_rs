// The browser side of `PAGE=f64.html node wasm/test/run.mjs`: runs the WGSL
// binary64 library (src/wasm/sim/f64.wgsl) on the WebGPU device over special,
// random and cancelling operands and compares every result bit for bit with
// JavaScript's IEEE doubles. NaN results only need to be NaN.
const text = async (url) => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: ${response.status}`);
    return response.text();
};

const OPS = ['add', 'sub', 'mul', 'div', 'lt', 'le', 'eq', 'from_f32', 'to_f32', 'rint', 'round', 'floor', 'trunc', 'to_i32',
    'from_i32', 'from_u64', 'from_i64', 'to_i64', 'ceil', 'u64_div6'];

const KERNEL = `
@group(0) @binding(0) var<storage, read> inputs: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> test: vec4<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= test.y) { return; }
    let q = inputs[i];
    let a = q.xy;
    let b = q.zw;
    var r = vec4<u32>(0u);
    switch test.x {
        case 0u: { r = vec4<u32>(f64_add(a, b), 0u, 0u); }
        case 1u: { r = vec4<u32>(f64_sub(a, b), 0u, 0u); }
        case 2u: { r = vec4<u32>(f64_mul(a, b), 0u, 0u); }
        case 3u: { r = vec4<u32>(f64_div(a, b), 0u, 0u); }
        case 4u: { r.x = select(0u, 1u, f64_lt(a, b)); r.y = select(0u, 1u, f64_gt(a, b)); }
        case 5u: { r.x = select(0u, 1u, f64_le(a, b)); r.y = select(0u, 1u, f64_ge(a, b)); }
        case 6u: { r.x = select(0u, 1u, f64_eq(a, b)); }
        case 7u: { r = vec4<u32>(f64_from_f32(bitcast<f32>(a.x)), 0u, 0u); }
        case 8u: { r.x = bitcast<u32>(f64_to_f32(a)); }
        case 9u: { r = vec4<u32>(f64_rint(a), 0u, 0u); }
        case 10u: { r = vec4<u32>(f64_round(a), 0u, 0u); }
        case 11u: { r = vec4<u32>(f64_floor(a), 0u, 0u); }
        case 12u: { r = vec4<u32>(f64_trunc(a), 0u, 0u); }
        case 13u: { r.x = u32(f64_to_i32(a)); }
        case 14u: { r = vec4<u32>(f64_from_i32(i32(a.x)), 0u, 0u); }
        case 15u: { r = vec4<u32>(f64_from_u64(a), 0u, 0u); }
        case 16u: { r = vec4<u32>(f64_from_i64(a), 0u, 0u); }
        case 17u: { r = vec4<u32>(f64_to_i64(a), 0u, 0u); }
        case 18u: { r = vec4<u32>(f64_ceil(a), 0u, 0u); }
        case 19u: { r = vec4<u32>(u64_div_small(a, 6u), 0u, 0u); }
        default: {}
    }
    outputs[i] = r;
}`;

const buffer = new ArrayBuffer(8);
const f64 = new Float64Array(buffer);
const u32 = new Uint32Array(buffer);
const f32 = new Float32Array(buffer, 0, 1);
const bitsOf = (x) => { f64[0] = x; return [u32[0], u32[1]]; };
const fromBits = (lo, hi) => { u32[0] = lo; u32[1] = hi; return f64[0]; };
const f32Bits = (x) => { f32[0] = x; return u32[0]; };

function rint(x) {
    if (!Number.isFinite(x)) return x;
    const f = Math.floor(x), d = x - f;
    const r = d < 0.5 ? f : d > 0.5 ? f + 1 : (f % 2 === 0 ? f : f + 1);
    return r === 0 ? (x < 0 || Object.is(x, -0) ? -0 : 0) : r;
}
function roundAway(x) {
    if (!Number.isFinite(x)) return x;
    let t = Math.trunc(x);
    if (Math.abs(x - t) >= 0.5) t += Math.sign(x);
    return t === 0 ? (x < 0 || Object.is(x, -0) ? -0 : 0) : t;
}
function toI32(x) {
    if (Number.isNaN(x)) return 0;
    return Math.max(-2147483648, Math.min(2147483647, Math.trunc(x))) | 0;
}
function toI64(x) {
    if (Number.isNaN(x)) return 0n;
    if (x >= 9223372036854775807) return 9223372036854775807n;
    if (x <= -9223372036854775808) return -9223372036854775808n;
    return BigInt(Math.trunc(x));
}
const big = (lo, hi) => (BigInt(hi) << 32n) | BigInt(lo);
const bigBits = (v) => { const m = BigInt.asUintN(64, v); return [Number(m & 0xffffffffn), Number(m >> 32n)]; };

// Expected result words for op and operand bits; `nan` marks a NaN result.
function expected(op, a, b, q) {
    const d = (x) => ({ words: [...bitsOf(x), 0, 0], nan: Number.isNaN(x) });
    switch (OPS[op]) {
        case 'add': return d(a + b);
        case 'sub': return d(a - b);
        case 'mul': return d(a * b);
        case 'div': return d(a / b);
        case 'lt': return { words: [+(a < b), +(a > b), 0, 0] };
        case 'le': return { words: [+(a <= b), +(a >= b), 0, 0] };
        case 'eq': return { words: [+(a === b), 0, 0, 0] };
        case 'from_f32': { u32[0] = q[0]; u32[1] = 0; return d(f32[0]); }
        case 'to_f32': { const r = Math.fround(a); return { words: [f32Bits(r), 0, 0, 0], nan: Number.isNaN(r), f32: true }; }
        case 'rint': return d(rint(a));
        case 'round': return d(roundAway(a));
        case 'floor': return d(Math.floor(a));
        case 'trunc': return d(Math.trunc(a));
        case 'to_i32': return { words: [toI32(a) >>> 0, 0, 0, 0] };
        case 'from_i32': return d(q[0] | 0);
        case 'from_u64': return d(Number(big(q[0], q[1])));
        case 'from_i64': return d(Number(BigInt.asIntN(64, big(q[0], q[1]))));
        case 'to_i64': return { words: [...bigBits(toI64(a)), 0, 0] };
        case 'ceil': return d(Math.ceil(a));
        case 'u64_div6': return { words: [...bigBits(big(q[0], q[1]) / 6n), 0, 0] };
    }
    throw new Error(`op ${op}`);
}

// A deterministic xorshift128+ stream of u32 words.
function generator(seed) {
    let s = [seed >>> 0 || 1, 0x9e3779b9, 0x7f4a7c15, 0x85ebca6b];
    return () => {
        let t = s[3];
        t ^= t << 11; t ^= t >>> 8;
        s = [t ^ s[0] ^ (s[0] >>> 19), s[0], s[1], s[2]];
        return s[0] >>> 0;
    };
}

const SPECIALS = [0, -0, 1, -1, 0.5, -0.5, 1.5, 2.5, -2.5, 3, 1 / 3, 0.1, 0.49999999999999994, 4503599627370495.5,
    4503599627370496, 9007199254740992, 9007199254740993, -4503599627370497, Infinity, -Infinity, NaN,
    5e-324, -5e-324, 2.2250738585072014e-308, 2.225073858507201e-308, 1.7976931348623157e308, -1.7976931348623157e308,
    2147483647.5, -2147483648.5, 2147483648, -2147483649, 9223372036854775807, -9223372036854775808, 1e-310, 1.401298464324817e-45,
    3.4028234663852886e38, 3.4028235677973366e38, 1.1754942807573643e-38, 7.006492321624085e-46, 709.782712893384, -745.1332191019411];

function operands(random, count) {
    const out = [];
    const specials = SPECIALS.map(bitsOf);
    for (const a of specials) for (const b of specials) out.push([...a, ...b]);
    const scaled = (range) => {
        const lo = random(), hi = random();
        const exponent = (1023 - range + (random() % (2 * range + 1))) & 0x7ff;
        return [lo, (hi & 0x800fffff) | (exponent << 20)];
    };
    while (out.length < count) {
        const kind = random() % 6;
        if (kind === 0) out.push([random(), random(), random(), random()]);
        else if (kind === 1) out.push([...scaled(60), ...scaled(60)]);
        else if (kind === 2) {
            // Close operands: massive cancellation in add/sub.
            const a = scaled(40);
            const b = [a[0] ^ (random() & 0xfff), a[1] ^ (random() & 0x80000000)];
            out.push([...a, ...b]);
        } else if (kind === 3) out.push([...scaled(1100), ...scaled(1100)]);
        else if (kind === 4) {
            // Subnormal and tiny operands.
            out.push([random(), random() & 0x801fffff, random(), (random() & 0x803fffff) | ((random() % 80) << 20)]);
        } else {
            // Halfway cases for rounding to integers and f32.
            const whole = Math.floor((random() / 4294967296 - 0.5) * 2 ** (random() % 60));
            const x = whole + [0.5, -0.5, 0.25, 0.75][random() % 4];
            out.push([...bitsOf(x), ...bitsOf(Math.fround(x) + 2 ** -(random() % 40) * Math.fround(x) * 2 ** -24)]);
        }
    }
    return out.map((q) => q.map((w) => w >>> 0));
}

export async function run(root) {
    const report = { ok: false, ops: {} };
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU: no adapter');
    report.adapter = adapter.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description } : null;
    const device = await adapter.requestDevice();
    const source = await text(new URL('src/wasm/sim/f64.wgsl', root)) + KERNEL;
    const module = device.createShaderModule({ code: source });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length) throw new Error(errors.map((m) => `${m.lineNum}:${m.linePos}: ${m.message}`).join('\n'));
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });

    const count = Number(window.altdTestOptions?.f64Count || 200000);
    const random = generator(window.altdTestOptions?.verifySeed || 7);
    const cases = operands(random, count);
    const input = new Uint32Array(cases.flat());
    const size = input.byteLength;
    const inputs = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const outputs = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inputs, 0, input);
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [inputs, outputs, uniform].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    let failures = 0;
    for (let op = 0; op < OPS.length; op++) {
        device.queue.writeBuffer(uniform, 0, new Uint32Array([op, cases.length, 0, 0]));
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(cases.length / 64));
        pass.end();
        encoder.copyBufferToBuffer(outputs, 0, readback, 0, size);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const got = new Uint32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        let mismatches = 0, first = null;
        for (let i = 0; i < cases.length; i++) {
            const q = cases[i];
            const want = expected(op, fromBits(q[0], q[1]), fromBits(q[2], q[3]), q);
            const words = got.subarray(4 * i, 4 * i + 4);
            let same;
            if (want.nan) {
                same = want.f32 ? ((words[0] & 0x7f800000) === 0x7f800000 && (words[0] & 0x7fffff) !== 0)
                    : Number.isNaN(fromBits(words[0], words[1]));
            } else {
                same = want.words.every((w, k) => (w >>> 0) === words[k]);
            }
            if (!same) {
                mismatches++;
                first ??= { index: i, a: fromBits(q[0], q[1]), b: fromBits(q[2], q[3]), operand: Array.from(q, (w) => w.toString(16)),
                    expected: want.words.map((w) => (w >>> 0).toString(16)), got: Array.from(words, (w) => w.toString(16)) };
            }
        }
        failures += mismatches;
        report.ops[OPS[op]] = first ? { mismatches, first } : { mismatches };
    }
    report.cases = cases.length;
    report.ok = failures === 0;
    return report;
}
