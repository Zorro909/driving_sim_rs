// Exercise the public WebGPU runtime with a driver that delivers completed
// promises only when polled. No browser or GPU is needed.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';

globalThis.GPUBufferUsage = { STORAGE: 1, COPY_DST: 2, UNIFORM: 4, COPY_SRC: 8, MAP_READ: 16 };
globalThis.GPUMapMode = { READ: 1 };
globalThis.GPUShaderStage = { COMPUTE: 1 };
const nativeTimers = { setTimeout, clearTimeout, setInterval, clearInterval };
const probe = globalThis.__runtimeWaitProbe = { current: null };
const source = await readFile(process.env.RUNTIME_SOURCE ?? new URL('../../src/wasm/sim/runtime.js', import.meta.url), 'utf8');
const instrumentation = `
const probe = globalThis.__runtimeWaitProbe;
const performance = { now: () => probe.current.time };
const setTimeout = (...args) => probe.current.timer('timeout', ...args);
const clearTimeout = id => probe.current.clearTimer(id);
const setInterval = (...args) => probe.current.timer('interval', ...args);
const clearInterval = id => probe.current.clearTimer(id);
`;
const { createSimulator } = await import(`data:text/javascript;base64,${Buffer.from(instrumentation + source).toString('base64')}`);
const entries = ['sensors_kernel', 'forward_kernel', 'stats_kernel', 'stop_kernel', 'step_kernel', 'drive_kernel', 'reset_kernel'];
// Only command orchestration is under test. Compilation is an external API;
// its stub accepts minimal entry points and profile/sensor declarations.
const shader = `
fn profile_atan2(math:u32,y_arg:f32,x_arg:f32)->f32 { return y_arg; }
fn profile_exp(math:u32,x_arg:f32)->f32 { return x_arg; }
fn profile_pow(math:u32,x_arg:f32,y_arg:f32)->f32 { return x_arg; }
fn profile_tanh(math:u32,x_arg:f32)->f32 { return x_arg; }
fn sensor_value()->f32 { return 0.0; }
fn after_sensor() {}
${entries.map(name => `@compute @workgroup_size(1) fn ${name}() {}`).join('\n')}
`;

class Fixture {
    constructor(options = {}) {
        this.options = options;
        this.time = 0;
        this.pending = [];
        this.commands = [];
        this.polls = [];
        this.activeTimers = new Set();
        this.scopes = [];
        this.buffers = [];
        this.abort = false;
        this.workRequests = 0;
        this.mappingRequests = 0;
        this.inWindow = false;
        this.world = new Uint32Array(40);
        this.world[15] = options.profile ?? 0;
        this.world[16] = 2;
        this.world[22] = 3;
        this.world[28] = 10;
        this.base = new Uint32Array(24);
        this.base[0] = 33;
        this.base[14] = 16;
        this.base[15] = 3;
        this.base[23] = 120;
        const fixture = this;
        this.device = {
            limits: { maxBufferSize: 1e6, maxStorageBufferBindingSize: 1e6 },
            lost: new Promise(resolve => { fixture.signalLoss = resolve; }),
            pushErrorScope(scope) { fixture.scopes.push(scope); },
            popErrorScope() {
                const scope = fixture.scopes.pop();
                return Promise.resolve(fixture.inWindow
                    ? (scope === 'validation' ? options.validation : options.allocation) ?? null : null);
            },
            createBindGroupLayout() { return {}; },
            createPipelineLayout() { return {}; },
            createBindGroup() { return {}; },
            createShaderModule() { return { getCompilationInfo: async () => ({ messages: [] }) }; },
            async createComputePipelineAsync(desc) { return desc.compute.entryPoint; },
            createBuffer(desc) { return fixture.buffer(desc.size); },
            createCommandEncoder() { return fixture.encoder(); },
            queue: {
                writeBuffer(buffer, offset, data) {
                    new Uint8Array(buffer.bytes, offset, data.byteLength).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
                },
                submit(commands) {
                    if (!commands.length) {
                        fixture.polls.push(fixture.pending.map(item => item.kind));
                        if (options.pollError) throw new Error(options.pollError);
                        fixture.flush();
                        return;
                    }
                    for (const command of commands) for (const operation of command) {
                        fixture.commands.push(operation.record);
                        operation.run?.();
                    }
                },
                onSubmittedWorkDone() {
                    const request = fixture.workRequests++;
                    return fixture.promise('work', request === 0 ? options.workError : undefined);
                },
            },
        };
        probe.current = this;
    }
    async initialize() {
        this.sim = await createSimulator(this.device, shader, this.world, this.base);
        this.names = ['cars', 'agents', 'networks', 'results', 'snapshot'];
        this.cars = new Uint32Array([0x80000000, 0x7ff80000, 0x12345678, 0xffffffff]);
        this.agents = new Uint32Array([0xdeadbeef, 1, 0]);
        await this.sim.upload(this.cars, this.agents, new Uint32Array(4), this.base);
        this.sim.setLoad(this.options.load ?? 1);
        this.inWindow = true;
        return this;
    }
    timer(kind, callback, delay, ...args) {
        let id;
        const wrapped = () => {
            if (kind === 'timeout') this.activeTimers.delete(id);
            callback(...args);
        };
        id = kind === 'timeout' ? nativeTimers.setTimeout(wrapped, delay) : nativeTimers.setInterval(wrapped, delay);
        this.activeTimers.add(id);
        return id;
    }
    clearTimer(id) {
        nativeTimers.clearTimeout(id);
        nativeTimers.clearInterval(id);
        this.activeTimers.delete(id);
    }
    promise(kind, message) {
        if (this.abort || this.options.immediate) return message ? Promise.reject(new Error(message)) : Promise.resolve();
        return new Promise((resolve, reject) => this.pending.push({ kind, resolve, reject, message }));
    }
    flush() {
        for (const item of this.pending.splice(0)) item.message ? item.reject(new Error(item.message)) : item.resolve();
    }
    buffer(size) {
        const fixture = this;
        const buffer = {
            name: this.names?.shift() ?? 'setup', bytes: new ArrayBuffer(size), mapState: 'unmapped', unmapped: 0,
            async mapAsync(mode) {
                assert.equal(mode, GPUMapMode.READ);
                assert.equal(this.mapState, 'unmapped');
                fixture.mappingRequests++;
                this.mapState = 'pending';
                try { await fixture.promise('map', fixture.options.mapError); this.mapState = 'mapped'; }
                catch (error) { this.mapState = 'unmapped'; throw error; }
            },
            getMappedRange() {
                assert.equal(this.mapState, 'mapped');
                if (fixture.options.rangeError) throw new Error(fixture.options.rangeError);
                return this.bytes;
            },
            unmap() { this.mapState = 'unmapped'; this.unmapped++; },
            destroy() {},
        };
        this.buffers.push(buffer);
        return buffer;
    }
    encoder() {
        if (this.options.load < 1 && !this.commands.length) this.time = 30;
        const operations = [];
        return {
            beginComputePass() {
                let pipeline, offset;
                return {
                    setPipeline(value) { pipeline = value; },
                    setBindGroup(index, group, offsets) { assert.equal(index, 0); [offset] = offsets; },
                    dispatchWorkgroups(groups) { operations.push({ record: ['dispatch', pipeline, offset, groups] }); },
                    end() {},
                };
            },
            copyBufferToBuffer(from, start, to, offset, size) {
                operations.push({ record: ['copy', from.name, start, to.name, offset, size],
                    run: () => new Uint8Array(to.bytes, offset, size).set(new Uint8Array(from.bytes, start, size)) });
            },
            finish() { return operations; },
        };
    }
    snapshot(ticks) { return [0x80000000, 0x7ff80000, 0x12345678, 0xffffffff, 0xdeadbeef, 1, 0, ticks, 0, 0, 0]; }
    async run(operation) {
        let timeout;
        const promise = operation();
        try {
            return await Promise.race([promise, new Promise((_, reject) => {
                timeout = nativeTimers.setTimeout(() => reject(new Error('GPU completion stalled without polling')), 100);
            })]);
        } finally {
            nativeTimers.clearTimeout(timeout);
            this.abort = true;
            this.flush();
            for (const timer of this.activeTimers) this.clearTimer(timer);
            await promise.catch(() => {});
        }
    }
    assertFinished() {
        assert.equal(this.scopes.length, 0);
        assert.equal(this.activeTimers.size, 0, 'completion wait timer must be cancelled');
    }
}

test('readback completes promptly on a poll-gated device and preserves every state bit', async () => {
    const fixture = await new Fixture().initialize();
    await fixture.run(async () => {
        const result = await fixture.sim.advance([1, 0], false);
        assert.deepEqual(Array.from(result), fixture.snapshot(1));
        assert.ok(fixture.polls.some(pending => pending.includes('map')));
        fixture.assertFinished();
    });
});

for (const immediate of [false, true]) {
    test(`reset, statistics, inference, physics and readback retain command order (${immediate ? 'immediate' : 'poll-gated'})`, async () => {
        const fixture = await new Fixture({ immediate }).initialize();
        await fixture.run(async () => {
            const result = await fixture.sim.advance([2, 4], true);
            assert.deepEqual(Array.from(result), fixture.snapshot(2));
            assert.deepEqual(fixture.commands, [
                ['dispatch', 'reset_kernel', 0, 2],
                ['dispatch', 'sensors_kernel', 0, 6],
                ['dispatch', 'forward_kernel', 0, 33],
                ['dispatch', 'drive_kernel', 0, 2],
                ['dispatch', 'step_kernel', 0, 2],
                ['dispatch', 'stats_kernel', 256, 2],
                ['dispatch', 'stop_kernel', 256, 1],
                ['dispatch', 'sensors_kernel', 256, 6],
                ['dispatch', 'forward_kernel', 256, 33],
                ['dispatch', 'drive_kernel', 256, 2],
                ['dispatch', 'step_kernel', 256, 2],
                ['copy', 'cars', 0, 'snapshot', 0, 16],
                ['copy', 'agents', 0, 'snapshot', 16, 12],
                ['copy', 'results', 0, 'snapshot', 28, 16],
            ]);
            fixture.assertFinished();
            new Uint32Array(fixture.buffers.find(buffer => buffer.name === 'cars').bytes).fill(42);
            assert.deepEqual(Array.from(result), fixture.snapshot(2), 'returned snapshot owns its bytes');
        });
});
}

test('a long window polls both queued backpressure and its final readback', async () => {
    const fixture = await new Fixture().initialize();
    await fixture.run(async () => {
        const result = await fixture.sim.advance([12, 0], false);
        assert.deepEqual(Array.from(result), fixture.snapshot(12));
        assert.ok(fixture.polls.some(pending => pending.includes('work')), 'queued completion must progress');
        assert.ok(fixture.polls.some(pending => pending.includes('map')), 'readback must progress separately');
        assert.ok(fixture.polls.every(pending => pending.length > 0), 'poll only while completion is pending');
        fixture.assertFinished();
    });
});

test('the load idle path polls its queued work before yielding', async () => {
    const fixture = await new Fixture({ load: 0.85 }).initialize();
    await fixture.run(async () => {
        const result = await fixture.sim.advance([4, 0], false);
        assert.deepEqual(Array.from(result), fixture.snapshot(4));
        assert.ok(fixture.polls.some(pending => pending.includes('work')));
        assert.ok(fixture.polls.some(pending => pending.includes('map')));
        fixture.assertFinished();
    });
});

test('already completed work cancels timers without unnecessary polling', async () => {
    const fixture = await new Fixture({ immediate: true }).initialize();
    await fixture.run(async () => {
        assert.deepEqual(Array.from(await fixture.sim.advance([12, 0], false)), fixture.snapshot(12));
        fixture.assertFinished();
        await new Promise(resolve => nativeTimers.setTimeout(resolve, 12));
        assert.equal(fixture.polls.length, 0);
    });
});

for (const immediate of [false, true]) {
    test(`mapping rejection unlocks the simulator and cancels timers (${immediate ? 'immediate' : 'poll-gated'})`, async () => {
        const fixture = await new Fixture({ immediate, mapError: 'mapping rejected' }).initialize();
        await fixture.run(async () => {
            await assert.rejects(fixture.sim.advance([1, 0], false), /mapping rejected/);
            fixture.assertFinished();
            const readback = fixture.buffers.find(buffer => buffer.name === 'snapshot');
            assert.equal(readback.mapState, 'unmapped');
            const polls = fixture.polls.length;
            await new Promise(resolve => nativeTimers.setTimeout(resolve, 12));
            assert.equal(fixture.polls.length, polls);
            fixture.options.mapError = undefined;
            const result = await fixture.sim.advance([1, 0], false);
            assert.deepEqual(Array.from(result), fixture.snapshot(1));
            fixture.assertFinished();
        });
    });
}

test('mapped-range failure unmaps the buffer and releases window ownership', async () => {
    const fixture = await new Fixture({ immediate: true, rangeError: 'range failed' }).initialize();
    await fixture.run(async () => {
        await assert.rejects(fixture.sim.advance([1, 0], false), /range failed/);
        const readback = fixture.buffers.find(buffer => buffer.name === 'snapshot');
        assert.equal(readback.mapState, 'unmapped');
        assert.equal(readback.unmapped, 1);
        fixture.assertFinished();
        fixture.options.rangeError = undefined;
        assert.deepEqual(Array.from(await fixture.sim.advance([1, 0], false)), fixture.snapshot(1));
    });
});

test('validation errors retain precedence over allocation and mapping failures', async () => {
    const fixture = await new Fixture({ immediate: true, mapError: 'mapping failed',
        validation: { message: 'validation failed' }, allocation: { message: 'allocation failed' } }).initialize();
    await fixture.run(async () => {
        await assert.rejects(fixture.sim.advance([1, 0], false), /^Error: validation failed$/);
        fixture.assertFinished();
    });
});

test('allocation errors are reported after window cleanup', async () => {
    const fixture = await new Fixture({ immediate: true, allocation: { message: 'allocation failed' } }).initialize();
    await fixture.run(async () => {
        await assert.rejects(fixture.sim.advance([1, 0], false), /^Error: allocation failed$/);
        fixture.assertFinished();
    });
});

test('queue completion rejection cancels polling and leaves subsequent windows usable', async () => {
    const fixture = await new Fixture({ workError: 'queue completion rejected' }).initialize();
    await fixture.run(async () => {
        await assert.rejects(fixture.sim.advance([12, 0], false), /queue completion rejected/);
        assert.equal(fixture.mappingRequests, 0);
        fixture.assertFinished();
        const polls = fixture.polls.length;
        await new Promise(resolve => nativeTimers.setTimeout(resolve, 12));
        assert.equal(fixture.polls.length, polls);
        assert.deepEqual(Array.from(await fixture.sim.advance([1, 0], false)), fixture.snapshot(1));
        fixture.assertFinished();
    });
});

test('poll submission failure stops polling and keeps ownership until the original mapping completes', async () => {
    const fixture = await new Fixture({ pollError: 'empty submit rejected' }).initialize();
    await fixture.run(async () => {
        const pending = fixture.sim.advance([1, 0], false);
        await new Promise(resolve => nativeTimers.setTimeout(resolve, 12));
        assert.equal(fixture.polls.length, 1);
        assert.equal(fixture.activeTimers.size, 0);
        assert.equal(fixture.scopes.length, 2, 'keep error scopes until original completion');
        await assert.rejects(fixture.sim.advance([1, 0], false), /in flight/);
        fixture.flush();
        assert.deepEqual(Array.from(await pending), fixture.snapshot(1));
        fixture.assertFinished();
    });
});

test('device loss stops new polling and retains the original mapping rejection and ownership', async () => {
    const fixture = await new Fixture({ mapError: 'original map rejected after loss' }).initialize();
    await fixture.run(async () => {
        const pending = fixture.sim.advance([1, 0], false);
        fixture.signalLoss({ message: 'synthetic device loss' });
        await new Promise(resolve => nativeTimers.setTimeout(resolve, 12));
        assert.equal(fixture.polls.length, 0);
        assert.equal(fixture.activeTimers.size, 0);
        assert.equal(fixture.scopes.length, 2);
        await assert.rejects(fixture.sim.advance([1, 0], false), /in flight/);
        fixture.flush();
        await assert.rejects(pending, /original map rejected after loss/);
        fixture.assertFinished();
        await assert.rejects(fixture.sim.advance([1, 0], false), /device lost.*synthetic device loss/);
    });
});

test('pending readback rejects concurrent upload, track replacement and advancement', async () => {
    const fixture = await new Fixture().initialize();
    await fixture.run(async () => {
        const pending = fixture.sim.advance([1, 0], false);
        await assert.rejects(fixture.sim.advance([1, 0], false), /in flight/);
        await assert.rejects(fixture.sim.upload(fixture.cars, fixture.agents, new Uint32Array(4), fixture.base), /in flight/);
        assert.throws(() => fixture.sim.setWorld(fixture.world), /in flight/);
        await pending;
        fixture.assertFinished();
    });
});

test('all math profiles survive polled windows and reject incompatible track replacement', async () => {
    for (const profile of [0, 1, 2]) {
        const fixture = await new Fixture({ profile }).initialize();
        await fixture.run(async () => {
            assert.deepEqual(Array.from(await fixture.sim.advance([1, 0], false)), fixture.snapshot(1));
            const wrong = fixture.world.slice();
            wrong[15] = (profile + 1) % 3;
            assert.throws(() => fixture.sim.setWorld(wrong), /Math profile changed/);
            fixture.sim.setWorld(fixture.world);
            await fixture.sim.upload(fixture.cars, fixture.agents, new Uint32Array(4), fixture.base);
            assert.deepEqual(Array.from(await fixture.sim.advance([1, 0], false)), fixture.snapshot(1));
            fixture.assertFinished();
        });
    }
});
