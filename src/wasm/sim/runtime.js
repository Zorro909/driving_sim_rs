// WebGPU command and buffer glue used only by the WASM GpuSimulation handle.
// All numerical work is WGSL. Buffers persist across windows and networks are
// uploaded only when the WASM population changes at a generation boundary.

// Share of wall time the simulator keeps the GPU fed (setLoad changes it),
// the target GPU time of one submission, the shortest idle pause, and the
// submissions kept queued so completion latency does not starve the GPU.
const DEFAULT_LOAD = 0.85;
const SUBMISSION_MS = 4;
const MIN_IDLE_MS = 4;
const QUEUED_SUBMISSIONS = 3;
// `progress(done, total, kernel)` reports each pipeline before it compiles,
// then once more when all have.
export async function createSimulator(device, source, world, base, progress) {
  source = specializeSensors(source, world, base);
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  let simulator;
  try {
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform", hasDynamicOffset: true },
        },
      ],
    });
    const layout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });
    const pipelines = [];
    const entryPoints = [
      "sensors_kernel",
      "forward_kernel",
      "stats_kernel",
      "stop_kernel",
      "step_kernel",
      "drive_kernel",
      "reset_kernel",
    ];
    for (const entryPoint of entryPoints) {
      progress?.(pipelines.length, entryPoints.length, entryPoint);
      const module = device.createShaderModule({
        code: shaderForEntry(source, entryPoint),
      });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === "error");
      if (errors.length)
        throw new Error(
          errors
            .map((m) => `${m.lineNum}:${m.linePos}: ${m.message}`)
            .join("\n"),
        );
      pipelines.push(
        await device.createComputePipelineAsync({
          layout,
          compute: { module, entryPoint },
        }),
      );
    }
    progress?.(pipelines.length, entryPoints.length, null);
    simulator = new Simulator(device, bindGroupLayout, pipelines, world, base);
  } finally {
    const validation = await device.popErrorScope();
    const allocation = await device.popErrorScope();
    if (validation || allocation) {
      simulator?.destroy();
      throw new Error((validation || allocation).message);
    }
  }
  return simulator;
}
class Simulator {
  constructor(device, layout, pipelines, world, base) {
    this.device = device;
    this.layout = layout;
    this.pipelines = pipelines;
    this.base = base;
    this.buffers = [];
    this.world = this.buffer(
      world.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );
    device.queue.writeBuffer(this.world, 0, world);
    this.uniform = this.buffer(
      base[23] * 256,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    this.lost = null;
    device.lost.then((info) => {
      this.lost = new Error(`WebGPU device lost: ${info.message}`);
    });
    this.load = DEFAULT_LOAD;
    this.tickMs = 2; // GPU time per tick, refined after each window
  }
  // A replaced track: the next upload binds the new world buffer.
  setWorld(world) {
    if (this.busy) throw new Error("WebGPU window is in flight");
    const old = this.world;
    this.world = this.buffer(
      world.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );
    this.device.queue.writeBuffer(this.world, 0, world);
    old.destroy();
    this.buffers.splice(this.buffers.indexOf(old), 1);
    this.populationSizes = null;
  }
  setLoad(load) {
    if (!(load > 0 && load <= 1))
      throw new Error("GPU load must be in (0, 1]");
    this.load = load;
  }
  buffer(size, usage) {
    if (
      size > this.device.limits.maxBufferSize ||
      (usage & GPUBufferUsage.STORAGE &&
        size > this.device.limits.maxStorageBufferBindingSize)
    )
      throw new Error("Population exceeds WebGPU buffer limits");
    const b = this.device.createBuffer({ size: Math.max(4, size), usage });
    this.buffers.push(b);
    return b;
  }
  async upload(cars, agents, networks, base) {
    if (this.busy) throw new Error("WebGPU window is in flight");
    if (this.lost) throw this.lost;
    this.device.pushErrorScope("out-of-memory");
    this.device.pushErrorScope("validation");
    try {
      this.resultSize = 16;
      const sizes = [
        cars.byteLength,
        agents.byteLength,
        networks.byteLength,
        (4 + base[0] * 192) * 4,
        cars.byteLength + agents.byteLength + this.resultSize,
      ];
      if (!this.populationSizes?.every((size, i) => size === sizes[i])) {
        // Reproduction can resize a population. Replace its allocations then.
        for (const b of this.populationBuffers ?? []) {
          b.destroy();
          const index = this.buffers.indexOf(b);
          if (index >= 0) this.buffers.splice(index, 1);
        }
        this.populationBuffers = [];
        this.populationSizes = null;
        const first = this.buffers.length;
        try {
          const usage =
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC;
          this.cars = this.buffer(cars.byteLength, usage);
          this.agents = this.buffer(agents.byteLength, usage);
          this.networks = this.buffer(
            networks.byteLength,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          );
          this.resultSize = 16;
          // Per car: 128 words of sensor inputs or step carry, then 64
          // sensor error words.
          this.results = this.buffer((4 + base[0] * 192) * 4, usage);
          this.readback = this.buffer(
            cars.byteLength + agents.byteLength + this.resultSize,
            GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          );
          this.populationBuffers = this.buffers.slice(first);
          this.bindGroup = this.device.createBindGroup({
            layout: this.layout,
            entries: [
              this.cars,
              this.agents,
              this.world,
              this.networks,
              this.results,
              this.uniform,
            ].map((buffer, binding) => ({
              binding,
              resource: { buffer, ...(binding === 5 ? { size: 96 } : {}) },
            })),
          });
          this.populationSizes = sizes;
        } finally {
          this.populationBuffers = this.buffers.slice(first);
        }
      }
      this.carSize = cars.byteLength;
      this.agentSize = agents.byteLength;
      this.base = base;
      this.device.queue.writeBuffer(this.cars, 0, cars);
      this.device.queue.writeBuffer(this.agents, 0, agents);
      this.device.queue.writeBuffer(this.networks, 0, networks);
    } finally {
      const validation = await this.device.popErrorScope();
      const allocation = await this.device.popErrorScope();
      if (validation || allocation) {
        this.populationSizes = null;
        throw new Error((validation || allocation).message);
      }
    }
  }
  async advance(window, reset) {
    if (this.busy) throw new Error("WebGPU window is in flight");
    if (this.lost) throw this.lost;
    this.busy = true;
    this.device.pushErrorScope("out-of-memory");
    this.device.pushErrorScope("validation");
    let value;
    try {
      const uniform = this.base.slice();
      uniform.set(window, 1);
      if (uniform[1] > this.base[23])
        throw new Error("GPU window exceeds its submission limit");
      const uniforms = new Uint32Array(uniform[1] * 64);
      for (let k = 1; k <= uniform[1]; k++) {
        uniforms.set(uniform, (k - 1) * 64);
        uniforms[(k - 1) * 64 + 22] = k;
      }
      this.device.queue.writeBuffer(this.uniform, 0, uniforms);
      this.device.queue.writeBuffer(
        this.results,
        0,
        new Uint32Array([uniform[1], 0, 0, 0]),
      );
      // Split the window into short submissions so a desktop compositor
      // sharing the GPU queue waits a few milliseconds at most, and leave
      // the GPU idle for (1 - load) of the time.
      const ticks = uniform[1];
      const size = Math.max(
        1,
        Math.min(ticks, Math.floor(SUBMISSION_MS / this.tickMs)),
      );
      const started = performance.now();
      let idle = 0;
      const queued = [];
      for (let first = 1; first <= ticks; first += size) {
        const last = Math.min(ticks, first + size - 1);
        const encoder = this.device.createCommandEncoder();
        const dispatch = (pipeline, k, count) => {
          const pass = encoder.beginComputePass();
          pass.setPipeline(this.pipelines[pipeline]);
          pass.setBindGroup(0, this.bindGroup, [(k - 1) * 256]);
          pass.dispatchWorkgroups(count);
          pass.end();
        };
        if (reset && first === 1)
          dispatch(6, 1, Math.ceil(uniform[0] / 32));
        for (let k = first; k <= last; k++) {
          const groups = Math.ceil(uniform[0] / 32);
          if ((uniform[2] + k) % 6 === uniform[5]) {
            dispatch(2, k, groups);
            dispatch(3, k, 1);
          }
          dispatch(0, k, groups * uniform[15]);
          dispatch(1, k, uniform[0]);
          dispatch(5, k, groups);
          dispatch(4, k, groups);
        }
        if (last === ticks) {
          encoder.copyBufferToBuffer(
            this.cars,
            0,
            this.readback,
            0,
            this.carSize,
          );
          encoder.copyBufferToBuffer(
            this.agents,
            0,
            this.readback,
            this.carSize,
            this.agentSize,
          );
          encoder.copyBufferToBuffer(
            this.results,
            0,
            this.readback,
            this.carSize + this.agentSize,
            this.resultSize,
          );
        }
        this.device.queue.submit([encoder.finish()]);
        if (last === ticks) break;
        // Bound the queued submissions.
        queued.push(this.device.queue.onSubmittedWorkDone());
        if (queued.length > QUEUED_SUBMISSIONS) await queued.shift();
        const owed =
          ((performance.now() - started - idle) * (1 - this.load)) /
            this.load -
          idle;
        if (owed >= MIN_IDLE_MS) {
          await queued.at(-1);
          queued.length = 0;
          const t = performance.now();
          await new Promise((resolve) => setTimeout(resolve, owed));
          idle += performance.now() - t;
        }
      }
      await this.readback.mapAsync(GPUMapMode.READ);
      const busy = performance.now() - started - idle;
      this.tickMs =
        (this.tickMs + Math.min(50, Math.max(0.05, busy / ticks))) / 2;
      try {
        value = new Uint32Array(this.readback.getMappedRange().slice(0));
      } finally {
        this.readback.unmap();
      }
    } finally {
      const validation = await this.device.popErrorScope();
      const allocation = await this.device.popErrorScope();
      this.busy = false;
      if (validation || allocation)
        throw new Error((validation || allocation).message);
    }
    return value;
  }
  destroy() {
    for (const b of this.buffers) b.destroy();
    this.buffers = [];
  }
}

// Remove unreachable functions before handing each pipeline to a driver.
// Large soft-float modules otherwise consume several GB during compilation.
function shaderForEntry(source, entry) {
  const functions = new Map();
  const regex = /(@compute\s+@workgroup_size\([^)]*\)\s*)?\bfn\s+(\w+)\s*\(/g;
  let match;
  while ((match = regex.exec(source))) {
    const start = match.index;
    const body = source.indexOf("{", regex.lastIndex);
    let end = body + 1,
      depth = 1;
    while (depth) {
      if (source[end] === "{") depth++;
      else if (source[end] === "}") depth--;
      end++;
    }
    functions.set(match[2], { start, end, body: source.slice(body, end) });
    regex.lastIndex = end;
  }
  const needed = new Set([entry]);
  const queue = [entry];
  while (queue.length) {
    const name = queue.pop();
    const body = functions.get(name)?.body;
    if (!body) throw new Error(`Missing shader function ${name}`);
    for (const call of body.matchAll(/\b(\w+)\s*\(/g)) {
      if (functions.has(call[1]) && !needed.has(call[1])) {
        needed.add(call[1]);
        queue.push(call[1]);
      }
    }
  }
  for (const [name, fn] of [...functions].reverse())
    if (!needed.has(name))
      source = source.slice(0, fn.start) + source.slice(fn.end);
  return source;
}
function specializeSensors(source, world, base) {
  const names = [
    "S_RAYCAST",
    "S_DISTANCE_FROM_WALL",
    "S_SPEED",
    "S_VELOCITY_FRONT",
    "S_VELOCITY_SIDE",
    "S_ACCELERATION_FRONT",
    "S_ACCELERATION_SIDE",
    "S_ANGULAR_VELOCITY",
    "S_WHEEL_ANGLE",
    "S_BOOST_CAPACITY",
    "S_GRIP",
    "S_CORRECT_DIRECTION",
    "S_TRACK_CURVATURE",
  ];
  const used = new Set();
  for (let i = 0; i < base[15]; i++) used.add(names[world[base[14] + i * 6]]);
  const start = source.indexOf("fn sensor_value("),
    end = source.indexOf("\nfn ", start + 1);
  let sensor = source.slice(start, end);
  const ranges = [];
  for (const m of sensor.matchAll(/case (S_\w+): \{/g)) {
    if (used.has(m[1])) continue;
    let depth = 1,
      end = m.index + m[0].length;
    while (depth) {
      if (sensor[end] === "{") depth++;
      else if (sensor[end] === "}") depth--;
      end++;
    }
    ranges.push([m.index, end]);
  }
  for (const [start, end] of ranges.reverse())
    sensor = sensor.slice(0, start) + sensor.slice(end);
  return source.slice(0, start) + sensor + source.slice(end);
}
