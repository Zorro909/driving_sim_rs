#!/usr/bin/env node
// Validate the exact WGSL modules assembled by src/wasm/gpu.rs (raycaster) and
// src/wasm/sim/mod.rs (full simulator) with Naga, Firefox's shader frontend,
// and translate them to SPIR-V, where Firefox's pipeline creation once
// crashed. No browser or GPU adapter is required.
import { readFile, mkdtemp, writeFile, rm } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { shaderForEntry, specializeMath } from '../../src/wasm/sim/runtime.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const read = (...names) => Promise.all(names.map(name => readFile(path.join(root, 'src/wasm', name), 'utf8')));
const shaders = {
    Ray: (await read('float.wgsl', 'rays.wgsl')).join('\n'),
    Simulation: (await read('sim/f64.wgsl', 'sim/helpers.wgsl', 'float.wgsl', 'sim/generated.wgsl', 'sim/window.wgsl')).join('\n'),
};
for (const profile of [0, 1, 2]) {
    for (const entry of ['sensors_kernel', 'forward_kernel', 'stats_kernel', 'stop_kernel', 'step_kernel', 'drive_kernel', 'reset_kernel']) {
        shaders[`Simulation-${profile}-${entry}`] = shaderForEntry(specializeMath(shaders.Simulation, profile), entry);
    }
}
const scratch = await mkdtemp(path.join(tmpdir(), 'altd-shader-'));
try {
    for (const [name, code] of Object.entries(shaders)) {
        const source = path.join(scratch, `${name}.wgsl`);
        await writeFile(source, code);
        const result = spawnSync(process.env.NAGA || 'naga', [source, path.join(scratch, `${name}.spv`)], { encoding: 'utf8' });
        if (result.error) throw new Error(`Could not run Naga: ${result.error.message}. Install naga-cli or set NAGA to its executable.`);
        if (result.status !== 0) {
            process.stderr.write(`${name} shader:\n${result.stderr || result.stdout}`);
            process.exitCode = result.status ?? 1;
        } else {
            console.log(`${name} shader passes Naga validation and SPIR-V translation.`);
        }
    }
} finally {
    await rm(scratch, { recursive: true, force: true });
}
