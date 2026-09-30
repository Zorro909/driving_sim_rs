// The browser side of wasm/test/run.mjs: runs scenario.json through the
// WebAssembly library and compares every car state with reference.json
// (written natively by examples/wasm_reference.rs), then, with WebGPU,
// checks the GPU raycaster against the CPU and drives the same scenario with
// GPU rays. `root` is the URL of the repository root.
import init, { Simulation, GpuRaycaster, version } from '../pkg/altd_sim.js';

const text = async (url) => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: ${response.status}`);
    return response.text();
};

// Runs the scenario steps on `sim`; with `raycaster`, the windows cast their rays on the GPU.
async function runSteps(sim, scenario, raycaster) {
    sim.startWithShape(Uint32Array.from(scenario.shape));
    const result = { states: [], generations: [], ticks: [], milliseconds: 0 };
    const started = performance.now();
    for (const step of scenario.steps) {
        if (step.advance !== undefined) {
            result.ticks.push(raycaster ? await sim.advanceWithGpuRays(raycaster, step.advance, false) : sim.advance(step.advance, false));
        } else if (step.advanceGeneration !== undefined) {
            result.ticks.push(raycaster ? await sim.advanceGenerationWithGpuRays(raycaster, step.advanceGeneration) : sim.advanceGeneration(step.advanceGeneration));
        } else if (step.nextGeneration) {
            const g = sim.nextGeneration();
            result.generations.push({ preservedCount: g.preservedCount, rewards: Array.from(g.rewards) });
            result.ticks.push(0);
        } else {
            throw new Error(`unknown step ${JSON.stringify(step)}`);
        }
        result.states.push(Array.from(sim.carStates()));
    }
    result.milliseconds = performance.now() - started;
    result.generation = sim.generation;
    result.tick = sim.tick;
    return result;
}

// Bit-for-bit comparison of two runs (JSON keeps every double exactly).
function compare(got, expected) {
    const report = { values: 0, mismatches: 0, first: null, maxAbsoluteError: 0, ticks: got.ticks.join(','), expectedTicks: expected.ticks.join(',') };
    const check = (where, a, b) => {
        report.values += 1;
        if (!Object.is(a, b)) {
            report.mismatches += 1;
            if (!report.first) report.first = { where, got: a, expected: b };
            if (Number.isFinite(a) && Number.isFinite(b)) report.maxAbsoluteError = Math.max(report.maxAbsoluteError, Math.abs(a - b));
        }
    };
    check('generation', got.generation, expected.generation);
    check('tick', got.tick, expected.tick);
    check('ticks.length', got.ticks.length, expected.ticks.length);
    check('generations.length', got.generations.length, expected.generations.length);
    check('states.length', got.states.length, expected.states.length);
    got.ticks.forEach((t, i) => check(`ticks[${i}]`, t, expected.ticks[i]));
    got.generations.forEach((g, i) => {
        check(`generations[${i}].preservedCount`, g.preservedCount, expected.generations[i]?.preservedCount);
        check(`generations[${i}].rewards.length`, g.rewards.length, expected.generations[i]?.rewards.length);
        g.rewards.forEach((r, j) => check(`generations[${i}].rewards[${j}]`, r, expected.generations[i]?.rewards[j]));
    });
    got.states.forEach((s, i) => {
        check(`states[${i}].length`, s.length, expected.states[i]?.length);
        s.forEach((v, j) => check(`states[${i}][${j}]`, v, expected.states[i]?.[j]));
    });
    report.ok = report.mismatches === 0;
    return report;
}

export async function run(root) {
    await init();
    const scenario = JSON.parse(await text(root + 'wasm/test/scenario.json'));
    const reference = JSON.parse(await text(root + 'wasm/test/reference.json'));
    const [scene, network, model] = await Promise.all([scenario.scene, scenario.network, scenario.model].map((p) => text(root + p)));
    const report = { version: version(), cpu: null, gpu: null, ok: false };

    const cpuSim = new Simulation(scene, network, model, scenario.options);
    report.rayCount = cpuSim.rayCount();
    report.sensorNames = cpuSim.sensorNames();
    const cpu = await runSteps(cpuSim, scenario, null);
    report.population = cpuSim.population;
    report.cpu = compare(cpu, reference);
    report.cpu.milliseconds = Math.round(cpu.milliseconds);
    report.cpu.bestLap = cpuSim.bestLap();
    report.cpu.checkpointBytes = cpuSim.checkpointJson().length;
    report.ok = report.cpu.ok;

    if (window.altdGpuMode === 'none') {
        report.gpu = { skipped: true };
        return report;
    }

    try {
        // Headless Chromium can return null while its Vulkan instance starts.
        let adapter;
        for (let attempt = 0; attempt < 10; attempt++) {
            adapter = await navigator.gpu?.requestAdapter();
            if (adapter || !navigator.gpu) break;
            await new Promise((resolve) => setTimeout(resolve, 250));
        }
        if (!adapter) throw new Error('WebGPU: no adapter');
        const info = adapter.info;
        const adapterInfo = {
            vendor: info.vendor, architecture: info.architecture, device: info.device,
            description: info.description, isFallbackAdapter: info.isFallbackAdapter,
        };
        if (window.altdGpuMode === 'hardware' && (info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test(Object.values(adapterInfo).join(' ')))) {
            throw new Error(`hardware GPU required: ${JSON.stringify(adapterInfo)}`);
        }
        const device = await adapter.requestDevice();
        const raycaster = await GpuRaycaster.create(device, cpuSim);
        report.gpu = { adapter: adapterInfo, nodeCount: raycaster.nodeCount, wallCount: raycaster.wallCount, depth: raycaster.depth };
        report.gpu.verify = await raycaster.verify(cpuSim, 4000, 4000, 7);
        const gpuSim = new Simulation(scene, network, model, scenario.options);
        const gpu = await runSteps(gpuSim, scenario, raycaster);
        report.gpu.compare = compare(gpu, cpu);
        report.gpu.milliseconds = Math.round(gpu.milliseconds);
        report.gpu.raysChecked = gpuSim.gpuRaysChecked();
        report.gpu.rayMismatches = gpuSim.gpuRayMismatches();
        report.gpu.ok = report.gpu.compare.ok && report.gpu.verify.rayMismatches === 0 && report.gpu.verify.pointMismatches === 0 && report.gpu.rayMismatches === 0;
        report.ok = report.ok && report.gpu.ok;
    } catch (error) {
        report.gpu = { unavailable: String(error && error.message || error) };
        report.ok = false;
    }
    return report;
}
