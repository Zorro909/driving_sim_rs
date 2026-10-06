// A new iframe gives every case an independent WASM instance and navigator.
// Exercise actual package initialization with delayed/rejected UA hints.
async function checkCase(root, scenario, packageBase) {
    let hints = 0;
    Object.defineProperty(navigator, 'userAgent', {
        configurable: true, value: scenario.platform === 'Windows' ? 'Windows NT 10.0' : 'Linux',
    });
    Object.defineProperty(navigator, 'userAgentData', {
        configurable: true,
        value: scenario.noHints ? undefined : {
            platform: scenario.platform,
            getHighEntropyValues: async (requested) => {
                hints++;
                if (requested.length !== 1 || requested[0] !== 'platformVersion') throw new Error('Wrong hints');
                await new Promise(resolve => setTimeout(resolve, 20));
                if (scenario.reject) throw new Error('Hint denied');
                return { platformVersion: scenario.version };
            },
        },
    });
    const lib = await import(root + packageBase + '/altd_sim.js');
    if (scenario.sync) {
        const bytes = await (await fetch(root + packageBase + '/altd_sim_bg.wasm')).arrayBuffer();
        lib.initSync({ module: bytes });
        // The async initializer must still await detection after initSync.
    }
    await lib.default();
    if (lib.initThreadPool) await lib.initThreadPool(1);
    const fixture = await (await fetch(root + 'wasm/test/scenario.json')).json();
    const [scene, network, model] = await Promise.all(
        [fixture.scene, fixture.network, fixture.model].map(async file => (await fetch(root + file)).text()),
    );
    const sim = new lib.Simulation(scene, network, model, { ...fixture.options, population: 1 });
    let actual;
    try {
        sim.startWithShape(fixture.shape);
        actual = JSON.parse(sim.checkpointJson()).rng.mathProfile || 'proton';
    } finally {
        sim.free();
    }
    if (actual !== scenario.expected) throw new Error(`${scenario.name}: ${actual}, expected ${scenario.expected}`);
    // Module start, async init and callers must all share the same query.
    const first = lib.detectMathProfile(), second = lib.detectMathProfile();
    if (first !== second) throw new Error(`${scenario.name}: detection promise was not cached`);
    const detected = await first;
    if (detected !== actual) throw new Error(`${scenario.name}: detection ${detected} differs from session ${actual}`);
    const expectedHints = scenario.platform === 'Windows' && !scenario.noHints ? 1 : 0;
    if (hints !== expectedHints) throw new Error(`${scenario.name}: ${hints} hint calls, expected ${expectedHints}`);
    return { ok: true, name: scenario.name, profile: actual, hintCalls: hints };
}

export async function run(root) {
    const args = new URLSearchParams(location.search);
    if (args.has('case')) {
        const scenario = JSON.parse(args.get('case'));
        let report;
        try {
            report = await checkCase(root, scenario, args.get('package'));
        } catch (error) {
            report = { ok: false, name: scenario.name, error: String(error.stack || error) };
        }
        parent.postMessage({ type: 'math-profile-init', report }, location.origin);
        return report;
    }
    const cases = [
        { name: 'win10', platform: 'Windows', version: '10.0.0', expected: 'win10-fma3' },
        { name: 'win11-23h2', platform: 'Windows', version: '15.0.0', expected: 'win10-fma3' },
        { name: 'win11-24h2', platform: 'Windows', version: '19.0.0', expected: 'win11-fma3' },
        { name: 'linux', platform: 'Linux', version: '99.0.0', expected: 'proton' },
        { name: 'windows-no-hints', platform: 'Windows', noHints: true, expected: 'win11-fma3' },
        { name: 'windows-denied-hints', platform: 'Windows', reject: true, expected: 'win11-fma3' },
        { name: 'windows-invalid-hints', platform: 'Windows', version: 'invalid', expected: 'win11-fma3' },
        { name: 'sync-then-async', platform: 'Windows', version: '15.0.0', sync: true, expected: 'win10-fma3' },
    ];
    const packageBase = window.altdTestOptions?.packageBase || 'wasm/pkg';
    const reports = [];
    for (const scenario of cases) {
        const frame = document.createElement('iframe');
        const url = new URL('math-profile-init.html', import.meta.url);
        url.searchParams.set('case', JSON.stringify(scenario));
        url.searchParams.set('package', packageBase);
        frame.src = url.href;
        const result = new Promise((resolve, reject) => {
            const receive = event => {
                if (event.source !== frame.contentWindow || event.origin !== location.origin ||
                    event.data?.type !== 'math-profile-init') return;
                clearTimeout(timer);
                window.removeEventListener('message', receive);
                resolve(event.data.report);
            };
            const timer = setTimeout(() => {
                window.removeEventListener('message', receive);
                reject(new Error(`${scenario.name}: initialization timed out`));
            }, 30000);
            window.addEventListener('message', receive);
        });
        document.body.append(frame);
        try {
            reports.push(await result);
        } finally {
            frame.remove();
        }
    }
    return { ok: reports.every(report => report.ok), checks: reports.length, cases: reports };
}
