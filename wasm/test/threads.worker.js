const bits = (values) => Array.from(new BigUint64Array(Float64Array.from(values).buffer), (value) => value.toString(16).padStart(16, '0'));
function snapshot(sim) {
    return {
        generation: sim.generation, tick: sim.tick, states: bits(sim.carStates()),
        controls: Array.from({ length: sim.population }, (_, i) => bits(sim.controls(i))),
        sensors: Array.from({ length: sim.population }, (_, i) => bits(sim.sensors(i))),
        metrics: Array.from({ length: sim.population }, (_, i) => bits(sim.metrics(i))),
        summary: sim.generationSummary(), checkpoint: Array.from(sim.checkpointBytes()),
    };
}
function compare(actual, expected, path = '', report = { values: 0, mismatches: 0, first: null }) {
    if (actual && typeof actual === 'object' && expected && typeof expected === 'object') {
        const keys = new Set([...Object.keys(actual), ...Object.keys(expected)]);
        for (const key of keys) compare(actual[key], expected[key], `${path}/${key}`, report);
    } else {
        report.values++;
        if (!Object.is(actual, expected)) {
            report.mismatches++;
            const parts = path.split("/");
            const field = parts[2] === "state" ? parts[3] : parts[2];
            report.byField ??= {};
            report.byField[field] = (report.byField[field] || 0) + 1;
            report.first ??= { path, actual, expected };
        }
    }
    return report;
}
async function parity(lib, single, scene, network, model, scenario, reference) {
    const results = {};
    for (const mode of ['independent', 'lockstep']) {
        const sim = new lib.Simulation(scene, network, model, { ...scenario.options, mode });
        const steps = [], serialSteps = [];
        const baseline = new single.Simulation(scene, network, model, { ...scenario.options, mode });
        try {
            sim.startWithShape(Uint32Array.from(scenario.shape));
            baseline.startWithShape(Uint32Array.from(scenario.shape));
            for (let generation = 0; generation < 3; generation++) {
                for (const ticks of [1, 5, 54, 180]) {
                    steps.push({ executed: sim.advance(ticks, false), state: snapshot(sim) });
                    serialSteps.push({ executed: baseline.advance(ticks, false), state: snapshot(baseline) });
                }
                const next = sim.nextGeneration();
                steps.push({ preserved: next.preservedCount, rewards: bits(next.rewards), state: snapshot(sim) });
                const serialNext = baseline.nextGeneration();
                serialSteps.push({ preserved: serialNext.preservedCount, rewards: bits(serialNext.rewards), state: snapshot(baseline) });
                // Both directions use the same binary RNG/network checkpoint.
                const serial = new single.Simulation(scene, network, model, { ...scenario.options, mode });
                const resumed = new lib.Simulation(scene, network, model, { ...scenario.options, mode });
                try {
                    serial.restoreCheckpointBytes(sim.checkpointBytes());
                    resumed.restoreCheckpointBytes(serial.checkpointBytes());
                    serial.advance(12, false); resumed.advance(12, false);
                    const check = compare(snapshot(resumed), snapshot(serial));
                    if (check.mismatches) throw new Error(`checkpoint resume differed: ${JSON.stringify(check.first)}`);
                } finally { serial.free(); resumed.free(); }
            }
            results[mode] = { ...compare(steps, serialSteps), native: compare(steps, reference[mode]), serialNative: compare(serialSteps, reference[mode]) };
        } finally { sim.free(); baseline.free(); }
    }
    return results;
}
const yieldChannel = new MessageChannel();
const yields = [];
yieldChannel.port1.onmessage = () => yields.shift()?.();
const yieldTask = () => new Promise((resolve) => { yields.push(resolve); yieldChannel.port2.postMessage(null); });
const text = async (url) => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: ${response.status}`);
    return response.text();
};
self.onmessage = async ({ data: options }) => {
    if (options.type === 'ping') { self.postMessage({ type: 'pong', sent: options.sent }); return; }
    try {
        const started = performance.now();
        const threaded = options.threads !== 0;
        const packageBase = `/${options.packageBase || "target/wasm-thread-test"}`;
        const lib = await import(`${packageBase}/${threaded ? 'threads' : 'single'}/altd_sim.js`);
        const wasm = await lib.default();
        let children = 0, sharedChildMemories = 0;
        const WorkerClass = self.Worker;
        self.Worker = class extends WorkerClass {
            constructor(...args) { super(...args); children++; }
            postMessage(data, ...args) {
                if (data.memory?.buffer instanceof SharedArrayBuffer) sharedChildMemories++;
                return super.postMessage(data, ...args);
            }
        };
        if (threaded) await lib.initThreadPool(options.threads);
        const indices = lib.testThreadIndices ? Array.from(lib.testThreadIndices()) : null;
        const report = {
            ok: self.crossOriginIsolated && (wasm.memory.buffer instanceof SharedArrayBuffer) === threaded &&
                lib.cpuThreadCount() === (options.threads || 1) && children === options.threads &&
                sharedChildMemories === options.threads && (!indices || indices.length === new Set(indices).size),
            isolated: self.crossOriginIsolated, shared: wasm.memory.buffer instanceof SharedArrayBuffer,
            threads: lib.cpuThreadCount(), threaded, children, sharedChildMemories, indices,
            memoryBytes: wasm.memory.buffer.byteLength, startupMs: performance.now() - started,
        };
        const scenario = JSON.parse(await text('/' + options.scenario));
        const [scene, network, model] = await Promise.all([scenario.scene, scenario.network, scenario.model].map((url) => text('/' + url)));
        if (options.benchmark) {
            report.benchmarks = [];
            const device = options.gpuMode === "hardware" ? await lib.requestGpuDevice(true) : null;
            for (const population of options.populations.length ? options.populations : [32, 256, 1024, 4096]) {
                const setup = performance.now();
                const sim = new lib.Simulation(scene, network, model, { ...scenario.options, population, eliminateOnWall: false, eliminateWhenIdle: false });
                sim.startWithShape(Uint32Array.from(scenario.shape));
                const gpuStarted = performance.now();
                const gpu = device ? await lib.GpuSimulation.create(device, sim) : null;
                if (gpu) await gpu.verify(12);
                const result = { population, setupMs: performance.now() - setup, windowsMs: [], evolutionMs: [], generationsMs: [], gpuStartupMs: gpu ? performance.now() - gpuStarted : null, backend: gpu ? "webgpu" : "cpu", maxWindowMs: 0, peakLinearMemoryBytes: wasm.memory.buffer.byteLength };
                try {
                    for (let generation = 0; generation < 4; generation++) {
                        const generationStart = performance.now();
                        let advanceMs = 0;
                        let rate = 120;
                        while (sim.tick < 180) {
                            const chunk = Math.min(180 - sim.tick, Math.max(6, Math.floor(rate * 50 / 6000) * 6));
                            const windowStart = performance.now();
                            if (gpu) await gpu.advance(chunk, false);
                            else sim.advance(chunk, false);
                            const windowMs = performance.now() - windowStart;
                            advanceMs += windowMs;
                            rate = chunk * 1000 / Math.max(0.1, windowMs);
                            result.maxWindowMs = Math.max(result.maxWindowMs, windowMs);
                            await yieldTask();
                        }
                        const evolutionStart = performance.now();
                        sim.nextGeneration();
                        const evolutionMs = performance.now() - evolutionStart;
                        if (generation) {
                            result.windowsMs.push(advanceMs);
                            result.evolutionMs.push(evolutionMs);
                            result.generationsMs.push(performance.now() - generationStart);
                        }
                        result.peakLinearMemoryBytes = Math.max(result.peakLinearMemoryBytes, wasm.memory.buffer.byteLength);
                        await yieldTask();
                    }
                } finally { gpu?.free(); sim.free(); }
                report.benchmarks.push(result);
            }
            device?.destroy();
        } else {
            const single = await import(`${packageBase}/single/altd_sim.js`);
            await single.default();
            report.parity = await parity(lib, single, scene, network, model, scenario, JSON.parse(await text('/target/wasm-thread-reference.json')));
            report.nativeExact = Object.values(report.parity).every((result) => result.native.mismatches === 0);
            report.ok &&= Object.values(report.parity).every((result) => result.mismatches === 0 && result.native.mismatches === result.serialNative.mismatches);
            if (options.strictNative) report.ok &&= report.nativeExact;
            if (options.gpuMode === 'hardware') {
                const cpu = new lib.Simulation(scene, network, model, scenario.options);
                const sim = new lib.Simulation(scene, network, model, scenario.options);
                const device = await lib.requestGpuDevice(true);
                let gpu;
                try {
                    cpu.startWithShape(Uint32Array.from(scenario.shape));
                    sim.startWithShape(Uint32Array.from(scenario.shape));
                    gpu = await lib.GpuSimulation.create(device, sim);
                    await gpu.verify(12);
                    for (let generation = 0; generation < 3; generation++) {
                        cpu.advance(120, true);
                        await gpu.advance(120, true);
                        const check = compare(snapshot(sim), snapshot(cpu));
                        if (check.mismatches) throw new Error(`Threaded WebGPU mismatch: ${JSON.stringify(check.first)}`);
                        cpu.nextGeneration(); sim.nextGeneration();
                    }
                    report.gpu = { verifiedTicks: 12, generationWindows: 3, threads: lib.cpuThreadCount(), ok: true };
                } finally { gpu?.free(); sim.free(); cpu.free(); device.destroy(); }
            }
        }
        report.memoryBytes = wasm.memory.buffer.byteLength;
        self.postMessage(report);
    } catch (error) { self.postMessage({ ok: false, error: String(error.stack || error) }); }
};
