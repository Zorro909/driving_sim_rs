import { Simulation } from '../pkg/altd_sim.js';

// Compare every exported field on every tick, including the controls computed
// from GPU rays. Turnover also compares offspring and complete RNG checkpoints.
export async function verifyTicks(scene, network, model, scenario, raycaster, ticks) {
    const report = { ticksPerGeneration: ticks, values: 0, mismatches: 0, first: null, modes: [], raysChecked: 0, rayMismatches: 0 };
    const check = (where, got, expected) => {
        report.values++;
        if (!Object.is(got, expected)) {
            report.mismatches++;
            report.first ??= { where, got, expected };
        }
    };
    const array = (where, got, expected) => {
        check(`${where}.length`, got.length, expected.length);
        for (let i = 0; i < Math.max(got.length, expected.length); i++) check(`${where}[${i}]`, got[i], expected[i]);
    };
    for (const mode of ['independent', 'lockstep']) {
        const options = { ...scenario.options, mode, gpuVerifyEvery: 1 };
        const cpu = new Simulation(scene, network, model, options);
        const gpu = new Simulation(scene, network, model, options);
        try {
            cpu.startWithShape(Uint32Array.from(scenario.shape));
            gpu.startWithShape(Uint32Array.from(scenario.shape));
            for (let generation = 0; generation < 3; generation++) {
                for (let tick = 0; tick < ticks; tick++) {
                    const where = `${mode}/g${generation}/t${tick}`;
                    check(`${where}/executed`, await gpu.advanceWithGpuRays(raycaster, 1, false), cpu.advance(1, false));
                    check(`${where}/tick`, gpu.tick, cpu.tick);
                    array(`${where}/states`, gpu.carStates(), cpu.carStates());
                    for (let i = 0; i < cpu.population; i++) {
                        array(`${where}/car${i}/controls`, gpu.controls(i), cpu.controls(i));
                        array(`${where}/car${i}/metrics`, gpu.metrics(i), cpu.metrics(i));
                        array(`${where}/car${i}/sensors`, gpu.sensors(i), cpu.sensors(i));
                    }
                    if (report.mismatches) break;
                }
                if (report.mismatches) break;
                const a = gpu.nextGeneration();
                const b = cpu.nextGeneration();
                check(`${mode}/g${generation}/preserved`, a.preservedCount, b.preservedCount);
                array(`${mode}/g${generation}/rewards`, a.rewards, b.rewards);
                check(`${mode}/g${generation}/checkpoint`, gpu.checkpointJson(), cpu.checkpointJson());
            }
            report.modes.push(mode);
            report.raysChecked += gpu.gpuRaysChecked();
            report.rayMismatches += gpu.gpuRayMismatches();
        } finally {
            cpu.free(); gpu.free();
        }
        if (report.mismatches) break;
    }
    report.ok = report.mismatches === 0 && report.rayMismatches === 0;
    return report;
}
