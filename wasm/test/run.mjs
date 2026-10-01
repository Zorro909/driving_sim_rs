#!/usr/bin/env node
// Runs wasm/test/index.html in headless Chromium: serves the repository root
// on a local port and prints the page's report. Exits 1 on any mismatch.
// Needs wasm/pkg (wasm/build.sh), wasm/test/reference.json
// (cargo run --release --example wasm_reference) and Playwright with its
// Chromium (npm i -g playwright && npx playwright install chromium).
// CHROMIUM=/path/to/chrome overrides the browser; GPU=hardware uses Vulkan
// and rejects software adapters, GPU=software selects SwiftShader (default).
// NO_GPU=1 runs only the CPU comparison. PAGE=f64.html runs the soft-float
// unit test (wasm/test/f64.html) instead of the simulator comparison.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { execSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const require = createRequire(import.meta.url);

function loadPlaywright() {
    try {
        return require('playwright');
    } catch {
        const globalRoot = execSync('npm root -g').toString().trim();
        return require(path.join(globalRoot, 'playwright'));
    }
}

const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json' };

const server = createServer(async (request, response) => {
    const file = path.join(root, decodeURIComponent(new URL(request.url, 'http://localhost').pathname));
    if (!file.startsWith(root)) { response.writeHead(403).end(); return; }
    try {
        const body = await readFile(file);
        response.writeHead(200, { 'content-type': types[path.extname(file)] ?? 'application/octet-stream' }).end(body);
    } catch {
        response.writeHead(404).end();
    }
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const port = server.address().port;

const { chromium } = loadPlaywright();
const gpuMode = process.env.NO_GPU ? 'none' : process.env.GPU || 'software';
if (!['none', 'hardware', 'software'].includes(gpuMode)) throw new Error(`unknown GPU mode: ${gpuMode}`);
const args = gpuMode === 'none' ? [] : [
    '--enable-unsafe-webgpu', '--enable-blink-features=WebGPU', '--ignore-gpu-blocklist',
    '--enable-features=Vulkan',
    ...(gpuMode === 'hardware'
        ? ['--use-angle=vulkan', '--disable-vulkan-surface', '--enable-webgpu-developer-features']
        : ['--enable-unsafe-swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader']),
];
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM || undefined, args });
let report;
try {
    const page = await browser.newPage();
    await page.addInitScript((mode) => { window.altdGpuMode = mode; }, gpuMode);
    await page.addInitScript((fixture) => { window.altdDivisionFixture = fixture; }, process.env.DIVISION_FIXTURE);
    await page.addInitScript((options) => { window.altdTestOptions = options; }, {
        scenario: process.env.SCENARIO || 'wasm/test/scenario.json',
        reference: process.env.REFERENCE || 'wasm/test/reference.json',
        verifyRays: Number(process.env.VERIFY_RAYS || 4000),
        verifyPoints: Number(process.env.VERIFY_POINTS || 4000),
        verifySeed: Number(process.env.VERIFY_SEED || 7),
        f64Count: Number(process.env.F64_COUNT || 0),
        tickCheck: Number(process.env.TICK_CHECK || 0),
        verify: process.env.VERIFY_SIM==='1',
        population: Number(process.env.POPULATION || 0),
        populations: (process.env.POPULATIONS || '').split(',').filter(Boolean).map(Number),
        ticks: Number(process.env.TICKS || 0),
        step: Number(process.env.STEP || 0),
        keepAlive: process.env.KEEP_ALIVE === '1',
        load: Number(process.env.LOAD || 0),
        patches: process.env.PATCHES ? JSON.parse(process.env.PATCHES) : [],
        kernels: process.env.KERNELS ? process.env.KERNELS.split(',') : undefined,
    });
    page.on('console', (message) => { if (message.type() === 'error' || message.type() === 'warning') console.error(`[browser ${message.type()}] ${message.text()}`); });
    page.on('pageerror', (error) => console.error(`[browser error] ${error}`));
    await page.goto(`http://127.0.0.1:${port}/wasm/test/${process.env.PAGE || 'index.html'}`);
    await page.waitForFunction(() => window.altdReport !== undefined, null, { timeout: 600000 });
    report = await page.evaluate(() => window.altdReport);
    report.browser = browser.version();
    report.gpuMode = gpuMode;
    if (gpuMode === 'hardware') {
        const cdp = await browser.newBrowserCDPSession();
        const { gpu } = await cdp.send('SystemInfo.getInfo');
        report.hardware = { devices: gpu.devices, renderer: gpu.auxAttributes.glRenderer };
    }
} finally {
    await browser.close();
    server.close();
}
console.log(JSON.stringify(report, null, 2));
process.exit(report && report.ok ? 0 : 1);
