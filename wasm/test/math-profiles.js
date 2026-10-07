// Compare the generated device math against the captured UCRT fixture bits.
// The simulation tests separately cover WASM CPU math and CPU evolution.
import { shaderForEntry, specializeMath } from '../../src/wasm/sim/runtime.js';

const functions = {
  atan2f: { arity: 2, width: 4, call: 'F64(bitcast<u32>(profile_atan2(0u, bitcast<f32>(world_data[at]), bitcast<f32>(world_data[at+1u]))), 0u)' },
  exp: { arity: 1, width: 8, call: 'profile_exp(0u, F64(world_data[at], world_data[at+1u]))' },
  pow: { arity: 2, width: 8, call: 'profile_pow(0u, F64(world_data[at], world_data[at+1u]), F64(world_data[at+2u], world_data[at+3u]))' },
  tanh: { arity: 1, width: 8, call: 'profile_tanh(0u, F64(world_data[at], world_data[at+1u]))' },
};

export async function run(root) {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  if (window.altdGpuMode === 'hardware' && adapter.info.isFallbackAdapter)
    throw new Error('Hardware GPU required');
  const device = await adapter.requestDevice();
  const paths = ['sim/f64.wgsl', 'sim/helpers.wgsl', 'float.wgsl', 'sim/generated.wgsl', 'sim/window.wgsl'];
  const source = (await Promise.all(paths.map(async path => {
    const response = await fetch(root + 'src/wasm/' + path);
    if (!response.ok) throw new Error(`${path}: ${response.status}`);
    return response.text();
  }))).join('\n').replaceAll('i < 55u', 'i < 55u + params.pad').replaceAll('i < 23u', 'i < 23u + params.pad');
  const report = { ok: true, checks: 0, mismatches: 0, first: null, profiles: {} };
  try {
    for (const [name, { arity, width, call }] of Object.entries(functions)) {
      const response = await fetch(root + `math/fixtures/${name}.bin`);
      if (!response.ok) throw new Error(`${name}: ${response.status}`);
      const bytes = await response.arrayBuffer();
      const stride = (arity + 3) * width;
      const count = bytes.byteLength / stride;
      if (!Number.isInteger(count) || count < 100) throw new Error(`Invalid ${name} fixture`);
      const fixture = new DataView(bytes);
      const input = device.createBuffer({ size: bytes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const output = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const read = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const uniform = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const params = new Uint32Array(24);
      params[0] = count;
      device.queue.writeBuffer(uniform, 0, params);
      device.queue.writeBuffer(input, 0, bytes);
      try {
        for (const [profile, label] of ['proton', 'win10-fma3', 'win11-fma3'].entries()) {
          const kernel = `
@compute @workgroup_size(64) fn fixture_kernel(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x; if (i >= params.population) { return; }
  let at = i * ${stride / 4}u;
  let value = ${call};
  results[i * 2u] = value.x; results[i * 2u + 1u] = value.y;
}`;
          const code = shaderForEntry(specializeMath(source, profile) + kernel, 'fixture_kernel');
          const module = device.createShaderModule({ code });
          const info = await module.getCompilationInfo();
          const errors = info.messages.filter(message => message.type === 'error');
          if (errors.length) throw new Error(errors.map(message => message.message).join('\n'));
          const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'fixture_kernel' } });
          const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
            { binding: 2, resource: { buffer: input } },
            { binding: 4, resource: { buffer: output } },
            { binding: 5, resource: { buffer: uniform } },
          ] });
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline); pass.setBindGroup(0, group);
          pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
          encoder.copyBufferToBuffer(output, 0, read, 0, count * 8);
          device.queue.submit([encoder.finish()]);
          await read.mapAsync(GPUMapMode.READ);
          const got = new Uint32Array(read.getMappedRange().slice(0));
          read.unmap();
          let mismatches = 0;
          for (let row = 0; row < count; row++) {
            const expected = row * stride + (arity + profile) * width;
            const want = [fixture.getUint32(expected, true), width === 8 ? fixture.getUint32(expected + 4, true) : 0];
            report.checks++;
            if (got[row * 2] !== want[0] || got[row * 2 + 1] !== want[1]) {
              mismatches++; report.mismatches++;
              report.first ??= { name, profile: label, row, got: Array.from(got.slice(row * 2, row * 2 + 2)), want };
            }
          }
          report.profiles[label] ??= {};
          report.profiles[label][name] = { checks: count, mismatches };
          report.ok &&= mismatches === 0;
        }
      } finally { for (const buffer of [input, output, read, uniform]) buffer.destroy(); }
    }
  } finally { device.destroy(); }
  return report;
}
