// Runs the same fp_div helper used by the raycaster against native bit patterns.
export async function verifyDivision(device, root, fixture) {
    const source = await (await fetch(root + 'src/wasm/float.wgsl')).text();
    const response = await fetch(root + fixture);
    if (!response.ok) throw new Error(`${fixture}: ${response.status}`);
    const data = new Uint32Array(await response.arrayBuffer());
    if (data.length === 0 || data.length % 4 !== 0) throw new Error('invalid division fixture');
    const code = source + `
struct Params { count: u32, pad: u32, unused: vec2<u32> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> outputs: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.count) { return; }
    let q = inputs[id.x];
    outputs[id.x] = bitcast<u32>(fp_div(bitcast<f32>(q.x), bitcast<f32>(q.y)));
}`;
    const pipeline = await device.createComputePipelineAsync({
        layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
    });
    const count = data.length / 4;
    const uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const input = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    try {
        device.queue.writeBuffer(uniform, 0, Uint32Array.from([count, 0, 0, 0]));
        device.queue.writeBuffer(input, 0, data);
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries:
            [uniform, input, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
        encoder.copyBufferToBuffer(output, 0, readback, 0, count * 4);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const got = new Uint32Array(readback.getMappedRange());
        const report = { count, mismatches: 0, first: null };
        const nan = (bits) => (bits & 0x7fffffff) > 0x7f800000;
        for (let i = 0; i < count; i++) {
            const expected = data[i * 4 + 2];
            // NaN payload/sign propagation varies across CPU architectures.
            if (got[i] !== expected && !(nan(got[i]) && nan(expected))) {
                report.mismatches++;
                report.first ??= { index: i, a: data[i * 4], b: data[i * 4 + 1], got: got[i], expected };
            }
        }
        readback.unmap();
        report.ok = report.mismatches === 0;
        return report;
    } finally {
        for (const buffer of [uniform, input, output, readback]) buffer.destroy();
    }
}
