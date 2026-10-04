const text = async (url) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.text();
};
export async function run(root) {
  const lib = await import("../pkg/altd_sim.js");
  await lib.default();
  if (!lib.GpuSimulation)
    throw new Error("The full WebGPU simulator is not exported");
  const scenario = JSON.parse(
    await text(
      root + (window.altdTestOptions?.scenario || "wasm/test/scenario.json"),
    ),
  );
  if (window.altdTestOptions?.population)
    scenario.options.population = window.altdTestOptions.population;
  const [scene, network, model] = await Promise.all(
    [scenario.scene, scenario.network, scenario.model].map((p) =>
      text(root + p),
    ),
  );
  const cpu = new lib.Simulation(scene, network, model, scenario.options);
  const gpu = new lib.Simulation(scene, network, model, scenario.options);
  cpu.startWithShape(Uint32Array.from(scenario.shape));
  gpu.startWithShape(Uint32Array.from(scenario.shape));
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter");
  if (window.altdGpuMode === "hardware" && adapter.info.isFallbackAdapter)
    throw new Error("Hardware GPU required");
  const device = await adapter.requestDevice();
  const started = performance.now();
  const backend = await lib.GpuSimulation.create(device, gpu);
  if (window.altdTestOptions?.verify) await backend.verify(12);
  const report = {
    ok: true,
    adapter: {
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
    },
    compileMs: Math.round(performance.now() - started),
    checks: 0,
    mismatches: 0,
    first: null,
    steps: [],
    cpuMs: 0,
    gpuMs: 0,
  };
  const check = (where, a, b) => {
    report.checks++;
    if (!Object.is(a, b)) {
      report.ok = false;
      report.mismatches++;
      report.first ??= { where, cpu: a, gpu: b };
    }
  };
  const array = (where, a, b) => {
    check(where + ".length", a.length, b.length);
    for (let i = 0; i < a.length; i++) check(`${where}[${i}]`, a[i], b[i]);
  };
  const snapshot = (where) => {
    check(where + ".tick", cpu.tick, gpu.tick);
    check(where + ".generation", cpu.generation, gpu.generation);
    array(where + ".cars", cpu.carStates(), gpu.carStates());
    for (let i = 0; i < cpu.population; i++) {
      array(`${where}.controls.${i}`, cpu.controls(i), gpu.controls(i));
      array(`${where}.metrics.${i}`, cpu.metrics(i), gpu.metrics(i));
      array(`${where}.sensors.${i}`, cpu.sensors(i), gpu.sensors(i));
    }
    check(
      where + ".summary",
      JSON.stringify(cpu.generationSummary()),
      JSON.stringify(gpu.generationSummary()),
    );
  };
  const tickCheck = window.altdTestOptions?.tickCheck || 0;
  const advance = async (ticks, stop = false, handle = backend) => {
    const tick = gpu.tick;
    try {
      return await handle.advance(ticks, stop);
    } catch (error) {
      throw new Error(`GPU advance ${ticks} from tick ${tick}: ${error}`);
    }
  };
  check("zero.window", await backend.advance(0, false), cpu.advance(0, false));
  const steps = tickCheck
    ? Array.from({ length: tickCheck }, () => ({ advance: 1 }))
    : scenario.steps;
  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    let a, b, t;
    if (step.nextGeneration) {
      a = cpu.nextGeneration();
      b = gpu.nextGeneration();
      array(`step${i}.rewards`, a.rewards, b.rewards);
      check(`step${i}.preserved`, a.preservedCount, b.preservedCount);
      array(
        `step${i}.checkpoint`,
        cpu.checkpointBytes(),
        gpu.checkpointBytes(),
      );
    } else {
      t = performance.now();
      a =
        step.advanceGeneration === undefined
          ? cpu.advance(step.advance, false)
          : cpu.advanceGeneration(step.advanceGeneration);
      report.cpuMs += performance.now() - t;
      t = performance.now();
      b =
        step.advanceGeneration === undefined
          ? await advance(step.advance)
          : await backend.advanceGeneration(step.advanceGeneration);
      report.gpuMs += performance.now() - t;
      check(`step${i}.executed`, a, b);
    }
    snapshot(`step${i}`);
    report.steps.push({ step, tick: gpu.tick, mismatches: report.mismatches });
    if (!report.ok) break;
  }
  if (report.ok && !tickCheck && cpu.population <= 64) {
    const rejects = async (where, operation) => {
      let rejected = false;
      try {
        await operation();
      } catch {
        rejected = true;
      }
      check(where, rejected, true);
      check(where + ".busy", gpu.busy, false);
    };
    const checkpoint = cpu.checkpointBytes();
    cpu.restoreCheckpointBytes(checkpoint);
    gpu.restoreCheckpointBytes(checkpoint);
    check("verify.count", (await backend.verify(12)) > 0, true);
    snapshot("verify.restored");
    cpu.advance(7, false);
    gpu.advance(7, false);
    cpu.advance(13, false);
    await advance(13);
    snapshot("cpu.to.gpu");
    // An in-flight window owns mutation; rejection must not unlock it.
    const pending = backend.advance(6, false);
    let rejected = false;
    try {
      await backend.advance(1, false);
    } catch {
      rejected = true;
    }
    check("concurrent.gpu", rejected, true);
    rejected = false;
    try {
      gpu.nextGeneration();
    } catch {
      rejected = true;
    }
    check("concurrent.cpu", rejected, true);
    cpu.advance(6, false);
    await pending;
    snapshot("concurrent.completed");
    const other = await lib.GpuSimulation.create(device, gpu);
    cpu.advance(6, false);
    await advance(6, false, other);
    cpu.advance(6, false);
    await advance(6);
    snapshot("multiple.handles");
    other.free();
    cpu.advance(3607, false);
    await advance(3607);
    snapshot("split.window");
    cpu.setPaused(true);
    gpu.setPaused(true);
    await rejects("paused", () => backend.advance(6, false));
    cpu.setPaused(false);
    gpu.setPaused(false);
    await rejects("invalid.verify", () => backend.verify(0));
    snapshot("failed.window.restored");
    const settings = JSON.stringify({
      ...scenario.options.settings,
      population: cpu.population + 3,
    });
    cpu.setEvolutionSettings(settings);
    gpu.setEvolutionSettings(settings);
    cpu.nextGeneration();
    gpu.nextGeneration();
    cpu.advance(6, false);
    await advance(6);
    snapshot("population.resized");
    // A new network layout must not reuse the old GPU parameter indexing.
    const shape = Uint32Array.from([
      scenario.shape[0],
      8,
      scenario.shape.at(-1),
    ]);
    cpu.startWithShape(shape);
    gpu.startWithShape(shape);
    await rejects("changed.shape", () => backend.advance(6, false));
    cpu.advance(6, false);
    gpu.advance(6, false);
    snapshot("cpu.recovery");
    const recreated = await lib.GpuSimulation.create(device, gpu);
    device.destroy();
    await rejects("lost.device", () => recreated.advance(6, false));
    recreated.free();
    snapshot("lost.device.restored");
  }
  backend.free();
  cpu.free();
  gpu.free();
  device.destroy();
  report.cpuMs = Math.round(report.cpuMs);
  report.gpuMs = Math.round(report.gpuMs);
  return report;
}
