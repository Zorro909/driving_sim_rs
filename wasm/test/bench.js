// GPU-only throughput profile for GpuSimulation. Instruments WebGPU from the
// outside (prototype wrappers), so the library runs unmodified:
//  - wall time per advance() call, time awaiting the readback map, and the rest
//    (Rust decode/import, JS encoding);
//  - with 'timestamp-query', per-kernel GPU time and the first-to-last pass span.
// Options (window.altdTestOptions): population, ticks, warmup, populations,
// step (ticks per advance call), keepAlive, load (GpuSimulation.setLoad),
// patches ([search, replacement] shader source edits for ablation runs).
const text = async (url) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.text();
};

const labels = new WeakMap();
let profiling = null; // { device, encoders: [] } while measuring

const patched = new Set();
function instrument(patches) {
  // Experiments: plain [search, replacement] edits applied to every shader.
  const createModule = GPUDevice.prototype.createShaderModule;
  GPUDevice.prototype.createShaderModule = function (desc) {
    let code = desc.code;
    for (const [from, to] of patches) {
      if (code.includes(from)) patched.add(from);
      code = code.split(from).join(to);
    }
    return createModule.call(this, { ...desc, code });
  };
  const createAsync = GPUDevice.prototype.createComputePipelineAsync;
  GPUDevice.prototype.createComputePipelineAsync = async function (desc) {
    const p = await createAsync.call(this, desc);
    labels.set(p, desc.compute.entryPoint);
    return p;
  };
  const begin = GPUCommandEncoder.prototype.beginComputePass;
  const setPipeline = GPUComputePassEncoder.prototype.setPipeline;
  const states = new WeakMap();
  GPUCommandEncoder.prototype.beginComputePass = function (desc = {}) {
    if (!profiling?.timestamps) return begin.call(this, desc);
    let s = states.get(this);
    if (!s) {
      s = {
        set: profiling.device.createQuerySet({ type: "timestamp", count: 4096 }),
        names: [],
      };
      states.set(this, s);
    }
    const n = s.names.length;
    if (2 * n + 1 >= 4096) return begin.call(this, desc);
    s.names.push("?");
    const pass = begin.call(this, {
      ...desc,
      timestampWrites: {
        querySet: s.set,
        beginningOfPassWriteIndex: 2 * n,
        endOfPassWriteIndex: 2 * n + 1,
      },
    });
    pass.setPipeline = function (p) {
      s.names[n] = labels.get(p) ?? "?";
      return setPipeline.call(this, p);
    };
    return pass;
  };
  const finish = GPUCommandEncoder.prototype.finish;
  GPUCommandEncoder.prototype.finish = function (desc) {
    const s = states.get(this);
    if (s && s.names.length) {
      const bytes = s.names.length * 16;
      const resolve = profiling.device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      const read = profiling.device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      this.resolveQuerySet(s.set, 0, s.names.length * 2, resolve, 0);
      this.copyBufferToBuffer(resolve, 0, read, 0, bytes);
      profiling.encoders.push({ read, resolve, set: s.set, names: s.names });
    }
    return finish.call(this, desc);
  };
  const mapAsync = GPUBuffer.prototype.mapAsync;
  GPUBuffer.prototype.mapAsync = async function (...args) {
    const t = performance.now();
    try {
      return await mapAsync.apply(this, args);
    } finally {
      if (profiling && !profiling.inRead) profiling.mapMs += performance.now() - t;
    }
  };
}

async function collect(report) {
  profiling.inRead = true;
  for (const e of profiling.encoders) {
    await e.read.mapAsync(GPUMapMode.READ);
    const t = new BigUint64Array(e.read.getMappedRange().slice(0));
    e.read.unmap();
    let first = null,
      last = null;
    for (let i = 0; i < e.names.length; i++) {
      const a = t[2 * i],
        b = t[2 * i + 1];
      if (a === 0n || b < a) continue;
      const ns = Number(b - a);
      const k = (report.kernels[e.names[i]] ??= { passes: 0, ms: 0 });
      k.passes++;
      k.ms += ns / 1e6;
      first = first === null || a < first ? a : first;
      last = last === null || b > last ? b : last;
    }
    if (first !== null) report.spanMs += Number(last - first) / 1e6;
    e.read.destroy();
    e.resolve.destroy();
    e.set.destroy();
  }
  profiling.encoders = [];
  profiling.inRead = false;
}

export async function run(root) {
  instrument(window.altdTestOptions?.patches ?? []);
  const lib = await import("../pkg/altd_sim.js");
  await lib.default();
  const opts = window.altdTestOptions ?? {};
  const scenario = JSON.parse(
    await text(root + (opts.scenario || "wasm/test/scenario.json")),
  );
  const [scene, network, model] = await Promise.all(
    [scenario.scene, scenario.network, scenario.model].map((p) => text(root + p)),
  );
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter");
  const timestamps = adapter.features.has("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: timestamps ? ["timestamp-query"] : [],
  });
  const populations = opts.populations?.length
    ? opts.populations
    : [opts.population || 32];
  const ticks = opts.ticks || 720;
  const warmup = opts.warmup ?? 120;
  const out = { ok: true, timestamps, adapter: adapter.info.description || adapter.info.vendor, runs: [] };
  for (const population of populations) {
    const options = { ...scenario.options, population, gpuVerifyEvery: 0 };
    // Idle elimination would stop most cars early; keep the whole field
    // running so the profile reflects a full population.
    if (opts.keepAlive) {
      options.eliminateOnWall = false;
      options.eliminateWhenIdle = false;
    }
    const sim = new lib.Simulation(scene, network, model, options);
    sim.startWithShape(Uint32Array.from(scenario.shape));
    let t = performance.now();
    const gpu = await lib.GpuSimulation.create(device, sim);
    const compileMs = performance.now() - t;
    if (opts.load) gpu.setLoad(opts.load);
    if (warmup) await gpu.advance(warmup, false);
    const cpu = new lib.Simulation(scene, network, model, options);
    cpu.startWithShape(Uint32Array.from(scenario.shape));
    if (warmup) cpu.advance(warmup, false);
    t = performance.now();
    cpu.advance(ticks, false);
    const cpuMs = performance.now() - t;
    cpu.free();

    profiling = { device, encoders: [], mapMs: 0, timestamps, inRead: false };
    const report = {
      population,
      ticks,
      compileMs: Math.round(compileMs),
      cpuMs: Math.round(cpuMs),
      wallMs: 0,
      mapWaitMs: 0,
      otherMs: 0,
      spanMs: 0,
      kernels: {},
      windows: [],
      active: 0,
    };
    const step = opts.step || ticks;
    for (let done = 0; done < ticks; done += step) {
      profiling.mapMs = 0;
      t = performance.now();
      await gpu.advance(Math.min(step, ticks - done), false);
      const wall = performance.now() - t;
      report.windows.push({ wall: +wall.toFixed(2), map: +profiling.mapMs.toFixed(2) });
      report.wallMs += wall;
      report.mapWaitMs += profiling.mapMs;
    }
    report.active = sim.activeCount;
    const measured = profiling;
    await collect(report);
    profiling = null;
    report.otherMs = report.wallMs - report.mapWaitMs;
    for (const k of Object.values(report.kernels)) {
      k.usPerPass = +((k.ms * 1000) / k.passes).toFixed(1);
      k.ms = +k.ms.toFixed(2);
    }
    for (const key of ["wallMs", "mapWaitMs", "otherMs", "spanMs"])
      report[key] = +report[key].toFixed(1);
    report.usPerTick = +((report.wallMs * 1000) / ticks).toFixed(1);
    report.cpuUsPerTick = +((cpuMs * 1000) / ticks).toFixed(1);
    if (report.windows.length > 12) report.windows = report.windows.slice(0, 12);
    out.runs.push(report);
    gpu.free();
    sim.free();
    void measured;
  }
  device.destroy();
  const missing = (opts.patches ?? []).filter(([from]) => !patched.has(from));
  if (missing.length) throw new Error(`patch not found: ${missing[0][0]}`);
  return out;
}
