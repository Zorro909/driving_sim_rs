#include <chrono>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <ctime>
// libaltd_gpu.so: GPU parts of the fidelity simulator behind a C ABI.
// Rust loads it with dlopen (src/gpu.rs); examples/gpu_check.rs compares
// every part bit for bit with the CPU code.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstddef>
#include "math.h"
#include "rays.h"
#include "sensors.h"
#include "step.h"
#include "stats.h"

using namespace altd;

#define CHECK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) { \
    fprintf(stderr, "altd_gpu %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); return (int)e_; } } while (0)

namespace {

enum MathOp : int {
    OP_NATIVE_SIN = 0,     // f32 -> f32
    OP_NATIVE_COS = 1,     // f32 -> f32
    OP_NATIVE_ATAN2 = 2,   // (f32 y, f32 x) -> f32
    OP_MANAGED_SINCOS = 3, // f32 -> (f32, f32)
    OP_DEXP = 4,           // f64 -> f64
    OP_DPOW = 5,           // (f64, f64) -> f64
    OP_ENGINE_SINCOS = 6,  // f32 -> (f32, f32), with exceptions
    OP_ENGINE_RAW = 7,     // f32 -> (f32, f32), without exceptions
    OP_GAME_TANH = 8,      // f64 -> f64
};

__global__ void math_kernel(int op, int n, const void* in, void* out, uint32_t* err) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t e = 0;
    const float* fi = (const float*)in;
    const double* di = (const double*)in;
    float* fo = (float*)out;
    double* dout = (double*)out;
    switch (op) {
        case OP_NATIVE_SIN: fo[i] = native_sin(fi[i], e); break;
        case OP_NATIVE_COS: fo[i] = native_cos(fi[i], e); break;
        case OP_NATIVE_ATAN2: fo[i] = native_atan2(fi[2 * i], fi[2 * i + 1]); break;
        case OP_MANAGED_SINCOS: managed_sin_cos(fi[i], fo[2 * i], fo[2 * i + 1], e); break;
        case OP_DEXP: dout[i] = dexp(di[i]); break;
        case OP_DPOW: dout[i] = dpow(di[2 * i], di[2 * i + 1]); break;
        case OP_ENGINE_SINCOS: engine_sin_cos(fi[i], fo[2 * i], fo[2 * i + 1], e); break;
        case OP_ENGINE_RAW: engine_sin_cos_raw(fi[i], fo[2 * i], fo[2 * i + 1]); break;
        case OP_GAME_TANH: dout[i] = game_tanh(di[i]); break;
    }
    err[i] = e;
}

// Engine sin/cos of the float with bits `first + i` (wrapping), for the
// exhaustive x87 scan. Output: interleaved sine and cosine bits.
__global__ void engine_scan_kernel(uint32_t first, uint32_t n, uint32_t* out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s, c;
    engine_sin_cos_raw(ffrom(first + i), s, c);
    out[2 * (size_t)i] = fbits(s);
    out[2 * (size_t)i + 1] = fbits(c);
}

size_t in_size(int op) {
    switch (op) {
        case OP_NATIVE_ATAN2: return 8;
        case OP_DEXP: case OP_GAME_TANH: return 8;
        case OP_DPOW: return 16;
        default: return 4;
    }
}
size_t out_size(int op) {
    switch (op) {
        case OP_MANAGED_SINCOS: case OP_ENGINE_SINCOS: case OP_ENGINE_RAW: return 8;
        case OP_DEXP: case OP_DPOW: case OP_GAME_TANH: return 8;
        default: return 4;
    }
}

// Query kinds of altd_gpu_query.
enum QueryOp : int {
    Q_RAYCAST = 0,  // in (sx, sy, ex, ey) -> out (hx, hy, hit, 0)
    Q_CLOSEST = 1,  // in (px, py, -, -)   -> out (cx, cy, 1, 0)
};

__global__ void query_kernel(World w, int op, int n, const float4* in, float4* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 q = in[i];
    float2 r = make_float2(0.0f, 0.0f);
    bool ok = true;
    if (op == Q_RAYCAST) ok = raycast(w, make_float2(q.x, q.y), make_float2(q.z, q.w), r);
    else r = closest_wall_point(w, make_float2(q.x, q.y));
    out[i] = make_float4(r.x, r.y, ok ? 1.0f : 0.0f, 0.0f);
}

template <typename T>
int upload(const T* host, size_t count, const T*& device) {
    T* d = nullptr;
    if (count) {
        CHECK(hipMalloc(&d, sizeof(T) * count));
        CHECK(hipMemcpy(d, host, sizeof(T) * count, hipMemcpyHostToDevice));
    }
    device = d;
    return 0;
}

int upload_grid(const NearGrid& host, NearGrid& device) {
    size_t cells = host.start ? (size_t)host.nx * host.ny : 0;
    size_t items = cells ? host.start[cells] : 0;
    return upload(host.start, cells ? cells + 1 : 0, device.start) || upload(host.items, items, device.items);
}

// ---- simulation ----

// Network layer widths, and for each control (Agent::controls order:
// acceleration, steering, brake, handbrake, boost) the output neuron that
// sets it: the last one with that name, or -1 for 0.0.
struct NetShape {
    uint32_t layers, width[MAX_LAYERS];
    int32_t control_src[5];
};

struct WindowArgs {
    uint64_t start_tick, ticks;
    uint32_t start_batch, batches_per_tick, stats_phase, stop_when_inactive;
    uint32_t eliminate_on_wall, eliminate_when_idle, has_time_limit, pad;
    double time_limit;
};

struct WindowState {
    uint32_t live_count[2][8], step_count[2];
    uint32_t stopped;
    uint32_t items;     // SAT items of the split step (cleared by its end kernel)
    uint64_t executed;  // the stopping tick's k
    uint64_t base;      // k before the current period; kernels add their offset
};

uint32_t live_total(const WindowState& st, uint32_t p) {
    uint32_t count = 0;
    for (uint32_t b = 0; b < 8; b++) count += st.live_count[p][b];
    return count;
}

struct Sim {
    World world;  // device pointers, owned by the world handle
    VehicleDesc vehicle;
    Sensors sensors;
    uint32_t capacity, n;
    Car* cars;
    Agent* agents;
    float* here;     // sensor path offset per car
    double* inputs;  // n * sensors.count, car-major
    NetShape shape;
    size_t stride;   // parameters per network
    double* params;  // capacity * stride, one network per car
    uint32_t network_count;
    double* mean;
    double* novelty;
    // altd_gpu_sim_window: work lists, settling, device state, stop polling.
    uint32_t* live[2];
    uint32_t* steps[2];
    uint32_t* settled_at;
    Car* shadow;
    WindowState* state;
    // The split step: carried state per car, SAT items (car * MAX_PAIRS + pair) and their pushes.
    StepCarry* carry;
    uint32_t* items;
    Vec2* pushes;
    uint32_t* host_stopped;  // pinned
    hipEvent_t chunk_done[4];
    hipStream_t stream;  // blocking, so ordered with the null stream
    hipGraphExec_t graph_exec;
    WindowArgs graph_args;
    uint32_t graph_n;
};

// Kernel arguments captured by the period graph. The requested window length
// only affects the host's replay count and is not part of the captured graph.
bool same_schedule(const WindowArgs& a, const WindowArgs& b) {
    return a.start_tick == b.start_tick && a.start_batch == b.start_batch
        && a.batches_per_tick == b.batches_per_tick && a.stats_phase == b.stats_phase
        && a.stop_when_inactive == b.stop_when_inactive && a.eliminate_on_wall == b.eliminate_on_wall
        && a.eliminate_when_idle == b.eliminate_when_idle && a.has_time_limit == b.has_time_limit
        && a.time_limit == b.time_limit;
}

// Pass 1 of the sensor read: the path offset (a nearest-segment search) per reading car.
__global__ void path_offset_kernel(World w, ReadMask m, Car* cars, uint32_t n, float* here) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !reads(m, cars[i], i)) return;
    here[i] = sensor_path_offset(w, cars[i]);
}

// Pass 2: one thread per (sensor, car), sensor-major so a wave reads one sensor kind.
__global__ void sensor_kernel(World w, VehicleDesc v, Sensors s, ReadMask m, Car* cars, uint32_t n,
                              const float* here, double* inputs) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t k = t / n, i = t % n;
    if (k >= s.count || !reads(m, cars[i], i)) return;
    uint32_t err = 0;
    inputs[(size_t)i * s.count + k] = sensor_value(w, v, cars[i], s.s[k], s.needs_path ? here[i] : 0.0f, err);
    if (err) atomicOr(&cars[i].error, err);
}

// `Network::forward_into` for the reading cars, then Controls(**outputs).
// A block holds CARS_PER_BLOCK cars. Within a layer of `cols` outputs, thread
// t computes output t % cols of the block's car t / cols, so the block's
// CARS_PER_BLOCK * cols outputs occupy the first threads and the waves past
// them skip the layer entirely. With one car per LANES threads, a 12-, 8- or
// 5-wide layer would leave 4, 8 or 11 idle lanes in every wave, and idle
// lanes still cost the double-precision issue slots that bound this kernel.
// Each output adds its products in input order, then the bias, as the CPU does.
constexpr int LANES = 16;
constexpr int CARS_PER_BLOCK = 16;
// ALTD_FORWARD_DIAG (timing builds only, results are wrong): 1 skips the
// activation, 2 reads one weight row for every input (no weight traffic),
// 3 skips the products and applies the activation to an input, 4 skips both
// (the layer loop's own cost), 5 additionally drops the per-layer barriers.
#ifndef ALTD_FORWARD_DIAG
#define ALTD_FORWARD_DIAG 0
#endif
#if ALTD_FORWARD_DIAG == 5
#define FORWARD_LAYER_BARRIER() ((void)0)
#else
#define FORWARD_LAYER_BARRIER() __syncthreads()
#endif
// Weight rows are loaded ROW_CHUNK at a time, one chunk ahead of the products
// that use them, and a layer's first chunk and bias during the previous
// layer's activation. Every row's weights are a fresh cache line streamed
// from memory, so without the lookahead each row waits for its own load.
constexpr uint32_t ROW_CHUNK = 8;

// Rows first .. first + ROW_CHUNK - 1 of output j (zero past `rows` or when
// not `active`; nothing is read there).
__device__ inline void load_rows(double (&w)[ROW_CHUNK], const double* __restrict__ matrix, uint32_t rows, uint32_t cols, uint32_t j,
                                 uint32_t first, bool active) {
#pragma unroll
    for (uint32_t u = 0; u < ROW_CHUNK; u++) {
        bool have = active && first + u < rows;
#if ALTD_FORWARD_DIAG == 2
        w[u] = have ? matrix[j] : 0.0;
#else
        w[u] = have ? matrix[(size_t)(first + u) * cols + j] : 0.0;
#endif
    }
}

// Every thread of the block must call it (barriers); `car` and `live` are the
// caller's slot (threadIdx.x / LANES).
__device__ inline void forward_car(const NetShape& shape, const double* __restrict__ params, size_t stride, bool live, uint32_t car,
                                   const double* __restrict__ inputs, Agent* agents) {
    // Two activation buffers: a layer reads one and writes the other, so one
    // barrier per layer orders the writes before the next layer's reads (the
    // previous barrier already ordered this layer's reads before those writes).
    // Row stride of MAX_WIDTH + 1 doubles: the cars of one wave read the same
    // column at the same time, and rows 256 bytes apart would share LDS banks.
    __shared__ double buffers[2][CARS_PER_BLOCK][MAX_WIDTH + 1];
    __shared__ uint32_t slot_car[CARS_PER_BLOCK];
    __shared__ uint32_t slot_live[CARS_PER_BLOCK];
    uint32_t t = threadIdx.x, slot = t / LANES, lane = t % LANES;
    uint32_t n_in = shape.width[0];
    if (lane == 0) {
        slot_car[slot] = car;
        slot_live[slot] = live;
    }
    for (uint32_t i = lane; live && i < n_in; i += LANES) buffers[0][slot][i] = inputs[(size_t)car * n_in + i];
    __syncthreads();
    uint32_t current = 0;  // the buffer holding this layer's inputs
    // This thread's assignment for the current layer: car slot s, output j.
    uint32_t rows = shape.width[0], cols = shape.width[1];
    uint32_t s = t / cols, j = t % cols;
    bool active = t < CARS_PER_BLOCK * cols && slot_live[s];
    const double* matrix = active ? params + (size_t)slot_car[s] * stride : params;
    double w[ROW_CHUNK];
    load_rows(w, matrix, rows, cols, j, 0, active);
    size_t position = 0;
    for (uint32_t layer = 0; layer + 1 < shape.layers; layer++) {
        double (&values)[CARS_PER_BLOCK][MAX_WIDTH + 1] = buffers[current];
        double acc = 0.0, bias = 0.0, out = 0.0;
        if (active) {
            bias = matrix[(size_t)rows * cols + j];
#if ALTD_FORWARD_DIAG == 3 || ALTD_FORWARD_DIAG >= 4
            acc = values[s][j < rows ? j : 0];
#else
            for (uint32_t i = 0; i < rows; i += ROW_CHUNK) {
                double next[ROW_CHUNK];
                load_rows(next, matrix, rows, cols, j, i + ROW_CHUNK, true);
#pragma unroll
                for (uint32_t u = 0; u < ROW_CHUNK; u++)
                    if (i + u < rows) acc += values[s][i + u] * w[u];
#pragma unroll
                for (uint32_t u = 0; u < ROW_CHUNK; u++) w[u] = next[u];
            }
#endif
        }
        // The next layer's assignment and first rows, issued before this
        // layer's activation so their latency overlaps it.
        bool has_next = layer + 2 < shape.layers;
        uint32_t rows_n = cols, cols_n = has_next ? shape.width[layer + 2] : 1;
        uint32_t s_n = t / cols_n, j_n = t % cols_n;
        bool active_n = has_next && t < CARS_PER_BLOCK * cols_n && slot_live[s_n];
        size_t position_n = position + (size_t)(rows + 1) * cols;
        const double* matrix_n = active_n ? params + (size_t)slot_car[s_n] * stride + position_n : params;
        load_rows(w, matrix_n, rows_n, cols_n, j_n, 0, active_n);
        if (active) {
#if ALTD_FORWARD_DIAG == 1 || ALTD_FORWARD_DIAG >= 4
            out = acc + bias;
#else
            out = game_tanh(acc + bias);
#endif
        }
        if (active) buffers[current ^ 1][s][j] = out;
        FORWARD_LAYER_BARRIER();
        current ^= 1;
        rows = rows_n; cols = cols_n; s = s_n; j = j_n; active = active_n; matrix = matrix_n; position = position_n;
    }
    if (live && lane < 5) {
        int32_t src = shape.control_src[lane];
        agents[car].controls[lane] = src >= 0 ? buffers[current][slot][src] : 0.0;
    }
}

__global__ void __launch_bounds__(LANES * CARS_PER_BLOCK)
    forward_kernel(NetShape shape, const double* __restrict__ params, size_t stride, ReadMask m, const Car* cars, uint32_t n,
                   const double* __restrict__ inputs, Agent* agents) {
    uint32_t car = blockIdx.x * CARS_PER_BLOCK + threadIdx.x / LANES;
    bool live = car < n && reads(m, cars[car], car);
    forward_car(shape, params, stride, live, car, inputs, agents);
}

// The agent's side of a step of car i: a reported wall contact sets
// AGENT_PENDING_CONTACT; a car deactivated by the step records `tick`
// (Schedule::drive_agent).
__device__ inline void record(const Car& car, Agent& agent, bool was_active, bool report, uint64_t tick) {
    uint32_t flags = agent.flags | (report ? AGENT_PENDING_CONTACT : 0u);
    if (was_active && !(car.flags & CAR_ACTIVE)) {
        agent.deactivated_at = tick;
        flags |= AGENT_HAS_DEACTIVATED;
    }
    agent.flags = flags;
}

// `Car::step` (drive) or `Car::passive_step` of car i with the agent's controls.
__device__ inline void step_car(const World& w, const VehicleDesc& v, Car& car, Agent& agent, bool drive, bool eliminate_on_wall,
                                uint64_t tick) {
    bool was_active = car.flags & CAR_ACTIVE;
    bool report = car_step(w, v, car, agent.controls, drive, eliminate_on_wall);
    record(car, agent, was_active, report, tick);
}

constexpr uint32_t STEP_THREADS = 64;

// Bound all physics kernels to their actual block size so the solver
// contact-index array reserves LDS for STEP_THREADS threads, rather than 1024.
__global__ void __launch_bounds__(STEP_THREADS) step_kernel(World w, VehicleDesc v, Car* cars, Agent* agents, uint32_t n, uint32_t drive, uint32_t eliminate_on_wall,
                            uint64_t tick) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    step_car(w, v, cars[i], agents[i], drive != 0, eliminate_on_wall != 0, tick);
}

// ---- the split step ----
//
// `step_car` in three kernels, so the SAT tests of all broad pairs run side
// by side instead of in a loop per car: `step_begin` per car reserves one
// SAT item per broad pair (consecutive from the carry's base), the SAT kernel
// runs one item per thread, and `step_end` per car reads the pushes. The
// work per SAT thread varies only with its pair's axes, not with the pair
// counts of the other cars of the wave.

// Reserves `count` consecutive slots of `*total` per lane, one atomic per
// wave; returns the lane's first slot. Every lane must call it.
__device__ inline uint32_t reserve(uint32_t count, uint32_t* total) {
    uint32_t lane = __lane_id(), prefix = count;  // inclusive scan over the wave
    for (uint32_t o = 1; o < (uint32_t)warpSize; o <<= 1) {
        uint32_t x = (uint32_t)__shfl_up((int)prefix, o);
        prefix += lane >= o ? x : 0u;
    }
    uint32_t sum = (uint32_t)__shfl((int)prefix, warpSize - 1), base = 0;
    if (lane == 0 && sum) base = atomicAdd(total, sum);
    return (uint32_t)__shfl((int)base, 0) + prefix - count;
}

// Every lane must call it.
__device__ inline void begin_car(const World& w, const VehicleDesc& v, Car* cars, Agent* agents, bool valid, uint32_t i, bool drive,
                                 StepCarry* carry, uint32_t* items, uint32_t* item_count) {
    StepCarry k{};
    uint32_t count = 0;
    if (valid) {
        k = step_begin(w, v, cars[i], agents[i].controls, drive);
        count = cars[i].pair_count;
    }
    uint32_t base = reserve(count, item_count);
    if (!valid) return;
    k.base = base;
    carry[i] = k;
    for (uint32_t c = 0; c < count; c++) items[base + c] = i * MAX_PAIRS + c;
}

// The SAT items, grid-stride (the grid is sized for the cars, not the items).
__global__ void __launch_bounds__(STEP_THREADS) sat_kernel(World w, VehicleDesc v, Car* cars, const uint32_t* items, const WindowState* st, Vec2* pushes) {
    uint32_t total = st->items;
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += gridDim.x * blockDim.x) {
        uint32_t item = items[t];
        pushes[t] = sat_pair(w, v, cars[item / MAX_PAIRS], item % MAX_PAIRS);
    }
}

__device__ inline bool end_car(const World& w, const VehicleDesc& v, Car& car, Agent& agent, const StepCarry& k, const Vec2* pushes,
                               bool drive, bool eliminate_on_wall, uint64_t tick) {
    bool report = step_end(w, v, car, k, [&](uint32_t ci) { return pushes[k.base + ci]; }, drive, eliminate_on_wall);
    record(car, agent, k.was_active, report, tick);
    return report;
}

__global__ void __launch_bounds__(STEP_THREADS) step_begin_kernel(World w, VehicleDesc v, Car* cars, Agent* agents, uint32_t n, uint32_t drive, StepCarry* carry,
                                  uint32_t* items, WindowState* st) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    begin_car(w, v, cars, agents, i < n, i, drive != 0, carry, items, &st->items);
}

__global__ void __launch_bounds__(STEP_THREADS) step_end_kernel(World w, VehicleDesc v, Car* cars, Agent* agents, uint32_t n, uint32_t drive, uint32_t eliminate_on_wall,
                                uint64_t tick, const StepCarry* carry, const Vec2* pushes, WindowState* st) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) st->items = 0;  // the SAT kernel is done with them
    if (i >= n) return;
    end_car(w, v, cars[i], agents[i], carry[i], pushes, drive != 0, eliminate_on_wall != 0, tick);
}

uint32_t blocks(size_t n, uint32_t size = 256) { return (uint32_t)((n + size - 1) / size); }

// ALTD_GPU_SPLIT=0: the step in one kernel (car_step) instead of the split step.
bool split_step() {
    static const char* env = getenv("ALTD_GPU_SPLIT");
    static const bool split = !(env && strcmp(env, "0") == 0);
    return split;
}

// The step kernels need ~300 MB of scratch at full occupancy, over the HSA
// runtime's single-dispatch limit, so by default every launch allocates and
// frees it (0.2-1.7 ms per launch). Keeping the scratch avoids that. Runs at
// load, before the first HIP call initialises HSA; an explicit setting wins.
__attribute__((constructor)) void keep_scratch() { setenv("HSA_NO_SCRATCH_RECLAIM", "1", 0); }

}  // namespace

extern "C" {

// Copies the track data to the device; returns an opaque handle or null.
// `desc` holds host pointers and has the layout of World.
World* altd_gpu_world_create(const World* desc) {
    if (desc->ray_depth > (uint32_t)MAX_RAY_DEPTH) {
        fprintf(stderr, "altd_gpu: BSP depth %u exceeds MAX_RAY_DEPTH %d\n", desc->ray_depth, MAX_RAY_DEPTH);
        return nullptr;
    }
    World* w = new World(*desc);
    int bad = upload(desc->ray_nodes, desc->ray_node_count, w->ray_nodes)
        || upload(desc->ray_walls, desc->ray_wall_count, w->ray_walls)
        || upload(desc->surfaces, desc->surface_count, w->surfaces)
        || upload(desc->tiles, (size_t)(desc->tile_nx * desc->tile_ny), w->tiles)
        || upload(desc->path_segments, desc->path_points ? desc->path_points - 1 : 0, w->path_segments)
        || upload(desc->path_offsets, desc->path_points, w->path_offsets)
        || upload_grid(desc->path_grid, w->path_grid)
        || upload(desc->curve_position, desc->curve_points, w->curve_position)
        || upload(desc->curve_offsets, desc->curve_points, w->curve_offsets)
        || upload(desc->curve_forwards, desc->curve_points, w->curve_forwards)
        || upload_grid(desc->curve_grid, w->curve_grid)
        || upload(desc->shapes, desc->shape_count, w->shapes)
        || upload(desc->shape_local, desc->shape_point_count, w->shape_local)
        || upload(desc->shape_normals, desc->shape_point_count, w->shape_normals)
        || upload_grid(desc->shape_grid, w->shape_grid)
        || upload(desc->path_xy, (size_t)desc->path_points * 2, w->path_xy);
    if (bad) {
        fprintf(stderr, "altd_gpu: world upload failed\n");
        return nullptr;
    }
    return w;
}

void altd_gpu_world_free(World* w) {
    if (!w) return;
    for (const void* p : {(const void*)w->ray_nodes, (const void*)w->ray_walls, (const void*)w->surfaces,
                          (const void*)w->tiles, (const void*)w->path_segments, (const void*)w->path_offsets,
                          (const void*)w->path_grid.start, (const void*)w->path_grid.items,
                          (const void*)w->curve_position, (const void*)w->curve_offsets, (const void*)w->curve_forwards,
                          (const void*)w->curve_grid.start, (const void*)w->curve_grid.items,
                          (const void*)w->shapes, (const void*)w->shape_local, (const void*)w->shape_normals,
                          (const void*)w->shape_grid.start, (const void*)w->shape_grid.items, (const void*)w->path_xy})
        if (p) (void)hipFree((void*)p);
    delete w;
}

// Runs `n` BSP queries (see QueryOp) on host arrays.
int altd_gpu_query(const World* w, int op, int n, const float4* in, float4* out) {
    float4 *d_in, *d_out;
    CHECK(hipMalloc(&d_in, sizeof(float4) * n));
    CHECK(hipMalloc(&d_out, sizeof(float4) * n));
    CHECK(hipMemcpy(d_in, in, sizeof(float4) * n, hipMemcpyHostToDevice));
    query_kernel<<<(n + 255) / 256, 256>>>(*w, op, n, d_in, d_out);
    CHECK(hipGetLastError());
    CHECK(hipMemcpy(out, d_out, sizeof(float4) * n, hipMemcpyDeviceToHost));
    CHECK(hipFree(d_in));
    CHECK(hipFree(d_out));
    return 0;
}


// Struct sizes and a few offsets, for the Rust layout check.
int altd_gpu_layout(uint64_t* out, int n) {
    uint64_t v[] = {sizeof(World), sizeof(Car), sizeof(Contact), sizeof(Agent), sizeof(VehicleDesc), sizeof(SensorDesc),
                    sizeof(WheelDesc), sizeof(ShapeDev), sizeof(TileCell), sizeof(PathSeg), sizeof(NearGrid),
                    offsetof(World, tiles), offsetof(World, curve_grid), offsetof(World, shape_grid),
                    offsetof(Car, tick), offsetof(Car, pairs), offsetof(Car, contacts), offsetof(Agent, recent),
                    offsetof(VehicleDesc, steering_speed), offsetof(VehicleDesc, gravity)};
    int count = (int)(sizeof(v) / sizeof(v[0]));
    for (int i = 0; i < n && i < count; i++) out[i] = v[i];
    return count;
}

Sim* altd_gpu_sim_create(const World* world, const VehicleDesc* vehicle, const SensorDesc* sensors, uint32_t sensor_count,
                         uint32_t capacity) {
    if (sensor_count > (uint32_t)MAX_SENSORS || vehicle->wheel_count > (uint32_t)MAX_WHEELS) return nullptr;
    Sim* s = new Sim{};
    s->world = *world;
    s->vehicle = *vehicle;
    s->sensors.count = sensor_count;
    for (uint32_t k = 0; k < sensor_count; k++) {
        s->sensors.s[k] = sensors[k];
        s->sensors.needs_path |= sensors[k].kind == S_CORRECT_DIRECTION || sensors[k].kind == S_TRACK_CURVATURE;
    }
    s->capacity = capacity;
    if (hipMalloc(&s->cars, sizeof(Car) * capacity) || hipMalloc(&s->agents, sizeof(Agent) * capacity)
        || hipMalloc(&s->here, sizeof(float) * capacity) || hipMalloc(&s->inputs, sizeof(double) * capacity * MAX_SENSORS)
        || hipMalloc(&s->live[0], sizeof(uint32_t) * ((capacity + 7) / 8) * 8) || hipMalloc(&s->live[1], sizeof(uint32_t) * ((capacity + 7) / 8) * 8)
        || hipMalloc(&s->steps[0], sizeof(uint32_t) * capacity) || hipMalloc(&s->steps[1], sizeof(uint32_t) * capacity)
        || hipMalloc(&s->settled_at, sizeof(uint32_t) * capacity) || hipMalloc(&s->shadow, sizeof(Car) * capacity)
        || hipMalloc(&s->carry, sizeof(StepCarry) * capacity) || hipMalloc(&s->items, sizeof(uint32_t) * capacity * MAX_PAIRS)
        || hipMalloc(&s->pushes, sizeof(Vec2) * capacity * MAX_PAIRS)
        || hipMalloc(&s->novelty, sizeof(double) * capacity)
        || hipMalloc(&s->state, sizeof(WindowState)) || hipHostMalloc(&s->host_stopped, sizeof(uint32_t) * 4)
        || hipEventCreateWithFlags(&s->chunk_done[0], hipEventDisableTiming) || hipEventCreateWithFlags(&s->chunk_done[1], hipEventDisableTiming)
        || hipEventCreateWithFlags(&s->chunk_done[2], hipEventDisableTiming) || hipEventCreateWithFlags(&s->chunk_done[3], hipEventDisableTiming)
        || hipStreamCreate(&s->stream)) {
        fprintf(stderr, "altd_gpu: sim allocation failed\n");
        return nullptr;
    }
    return s;
}

void altd_gpu_sim_free(Sim* s) {
    if (!s) return;
    if (s->graph_exec) (void)hipGraphExecDestroy(s->graph_exec);
    for (void* p : {(void*)s->cars, (void*)s->agents, (void*)s->here, (void*)s->inputs, (void*)s->params, (void*)s->live[0],
                    (void*)s->live[1], (void*)s->steps[0], (void*)s->steps[1], (void*)s->settled_at, (void*)s->shadow, (void*)s->state,
                    (void*)s->carry, (void*)s->items, (void*)s->pushes, (void*)s->mean, (void*)s->novelty})
        if (p) (void)hipFree(p);
    if (s->host_stopped) (void)hipHostFree(s->host_stopped);
    for (hipEvent_t e : s->chunk_done)
        if (e) (void)hipEventDestroy(e);
    if (s->stream) (void)hipStreamDestroy(s->stream);
    delete s;
}

// Uploads `n` cars and (if not null) agents.
int altd_gpu_sim_upload(Sim* s, uint32_t n, const Car* cars, const Agent* agents) {
    if (n > s->capacity) return -1;
    s->n = n;
    CHECK(hipMemcpy(s->cars, cars, sizeof(Car) * n, hipMemcpyHostToDevice));
    if (agents) CHECK(hipMemcpy(s->agents, agents, sizeof(Agent) * n, hipMemcpyHostToDevice));
    return 0;
}

int altd_gpu_sim_upload_agents(Sim* s, uint32_t n, const Agent* agents) {
    if (n != s->n) return -1;
    CHECK(hipMemcpy(s->agents, agents, sizeof(Agent) * n, hipMemcpyHostToDevice));
    return 0;
}

int altd_gpu_sim_download(Sim* s, Car* cars, Agent* agents) {
    CHECK(hipDeviceSynchronize());
    if (cars) CHECK(hipMemcpy(cars, s->cars, sizeof(Car) * s->n, hipMemcpyDeviceToHost));
    if (agents) CHECK(hipMemcpy(agents, s->agents, sizeof(Agent) * s->n, hipMemcpyDeviceToHost));
    return 0;
}

// Reads the sensors of the cars selected by `mask`; if `inputs` is not null,
// copies all n * sensor_count values back (unread cars keep stale values).
int altd_gpu_sim_sensors(Sim* s, ReadMask mask, double* inputs) {
    if (s->n == 0) return 0;
    if (s->sensors.needs_path) path_offset_kernel<<<blocks(s->n, 64), 64>>>(s->world, mask, s->cars, s->n, s->here);
    sensor_kernel<<<blocks((size_t)s->n * s->sensors.count, 64), 64>>>(s->world, s->vehicle, s->sensors, mask, s->cars, s->n,
                                                                    s->here, s->inputs);
    CHECK(hipGetLastError());
    if (inputs)
        CHECK(hipMemcpy(inputs, s->inputs, sizeof(double) * s->n * s->sensors.count, hipMemcpyDeviceToHost));
    return 0;
}

// Installs one network per car: `layers` widths, then n * parameter_count(widths)
// f64 parameters (Network::params, car-major). `control_src` as in NetShape.
int altd_gpu_sim_networks(Sim* s, uint32_t layers, const uint32_t* width, uint32_t n, const double* params,
                          const int32_t* control_src) {
    if (layers < 2 || layers > (uint32_t)MAX_LAYERS || n > s->capacity || width[0] != s->sensors.count) return -1;
    NetShape shape{};
    shape.layers = layers;
    size_t stride = 0;
    for (uint32_t l = 0; l < layers; l++) {
        if (width[l] == 0 || width[l] > (uint32_t)MAX_WIDTH || (l > 0 && width[l] > (uint32_t)LANES)) return -1;
        shape.width[l] = width[l];
        if (l > 0) stride += (size_t)(width[l - 1] + 1) * width[l];
    }
    for (int c = 0; c < 5; c++) {
        if (control_src[c] >= (int32_t)width[layers - 1]) return -1;
        shape.control_src[c] = control_src[c];
    }
    // The graph embeds the network shape and parameter pointer. Parameter
    // contents may change in place without invalidating it.
    if (s->graph_exec && memcmp(&shape, &s->shape, sizeof shape) != 0) {
        CHECK(hipGraphExecDestroy(s->graph_exec));
        s->graph_exec = nullptr;
    }
    if (stride != s->stride || !s->params) {
        if (s->params) CHECK(hipFree(s->params));
        if (s->mean) CHECK(hipFree(s->mean));
        s->params = nullptr;
        s->mean = nullptr;
        CHECK(hipMalloc(&s->params, sizeof(double) * stride * s->capacity));
        CHECK(hipMalloc(&s->mean, sizeof(double) * stride));
    }
    s->shape = shape;
    s->stride = stride;
    CHECK(hipMemcpy(s->params, params, sizeof(double) * stride * n, hipMemcpyHostToDevice));
    s->network_count = n;
    return 0;
}

// One thread per parameter preserves the CPU's left-to-right population sum.
// Only `stride` such threads exist (a few dozen waves), so with one load in
// flight per add the population's 8 bytes x n x stride stream far below the
// memory bandwidth. Each block therefore stages MEAN_ROWS networks' slice of
// its MEAN_COLS parameters in LDS with all MEAN_THREADS threads loading
// (coalesced rows, many loads in flight), then the MEAN_COLS summing threads
// add the staged rows in ascending network order, exactly as before.
constexpr uint32_t MEAN_COLS = 64, MEAN_ROWS = 64, MEAN_THREADS = 256;

__global__ void __launch_bounds__(MEAN_THREADS)
population_mean_kernel(const double* params, size_t stride, uint32_t n, double* mean) {
    __shared__ double tile[MEAN_ROWS][MEAN_COLS + 1];
    uint32_t t = threadIdx.x;
    size_t col0 = (size_t)blockIdx.x * MEAN_COLS;
    bool sums = t < MEAN_COLS && col0 + t < stride;
    double sum = 0.0;
    for (uint32_t row0 = 0; row0 < n; row0 += MEAN_ROWS) {
        uint32_t rows = min(MEAN_ROWS, n - row0);
        for (uint32_t e = t; e < rows * MEAN_COLS; e += MEAN_THREADS) {
            uint32_t r = e / MEAN_COLS, c = e % MEAN_COLS;
            if (col0 + c < stride) tile[r][c] = params[(size_t)(row0 + r) * stride + col0 + c];
        }
        __syncthreads();
        if (sums)
            for (uint32_t r = 0; r < rows; r++) sum += tile[r][t];
        __syncthreads();
    }
    if (sums) mean[col0 + t] = sum / (double)n;
}

// Parallelize the expensive pow calls, then add their results in the exact
// parameter order used by TrainingRunner::install. pow(x, 2) is the game's
// routine and must not be replaced by x*x or a parallel sum.
__global__ void __launch_bounds__(64)
population_novelty_kernel(const double* params, size_t stride, const double* mean, double* novelty) {
    __shared__ double terms[64];
    size_t car = blockIdx.x;
    uint32_t lane = threadIdx.x;
    double sum = 0.0;
    for (size_t base = 0; base < stride; base += 64) {
        size_t j = base + lane;
        if (j < stride) terms[lane] = dpow(params[car * stride + j] - mean[j], 2.0);
        __syncthreads();
        if (lane == 0)
            for (uint32_t k = 0; k < 64 && base + k < stride; k++) sum += terms[k];
        __syncthreads();
    }
    if (lane == 0) novelty[car] = sqrt(sum);
}

int altd_gpu_sim_novelty(Sim* s, uint32_t n, double* novelty) {
    if (n != s->network_count || !s->params) return -1;
    if (n == 0) return 0;
    population_mean_kernel<<<blocks(s->stride, MEAN_COLS), MEAN_THREADS, 0, s->stream>>>(s->params, s->stride, n, s->mean);
    population_novelty_kernel<<<n, 64, 0, s->stream>>>(s->params, s->stride, s->mean, s->novelty);
    CHECK(hipGetLastError());
    CHECK(hipMemcpyAsync(novelty, s->novelty, sizeof(double) * n, hipMemcpyDeviceToHost, s->stream));
    CHECK(hipStreamSynchronize(s->stream));
    return 0;
}

// Sensors and network for the cars in `mask`: sets their Agent::controls.
int altd_gpu_sim_infer(Sim* s, ReadMask mask) {
    if (!s->params) return -1;
    if (s->n == 0) return 0;
    int status = altd_gpu_sim_sensors(s, mask, nullptr);
    if (status) return status;
    forward_kernel<<<blocks(s->n, CARS_PER_BLOCK), LANES * CARS_PER_BLOCK>>>(s->shape, s->params, s->stride, mask, s->cars, s->n,
                                                                            s->inputs, s->agents);
    CHECK(hipGetLastError());
    return 0;
}

// Counts the cars still active afterwards into `active` if not null.
__global__ void stats_kernel(World w, Car* cars, Agent* agents, uint32_t n, uint64_t tick, uint64_t stats_tick,
                             uint32_t eliminate_when_idle, uint32_t* active) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t err = 0;
    update_stats(w, cars[i], agents[i], tick, stats_tick, eliminate_when_idle != 0, err);
    cars[i].error |= err;
    if (active && (cars[i].flags & CAR_ACTIVE)) atomicAdd(active, 1u);
}

// One physics tick for all cars (see step_kernel). Needs physics shapes
// (the shapeless resolve_walls path is not ported).
int altd_gpu_sim_step(Sim* s, uint32_t drive, uint32_t eliminate_on_wall, uint64_t tick) {
    if (s->world.shape_count == 0) return -2;
    if (s->n == 0) return 0;
    if (!split_step()) {
        step_kernel<<<blocks(s->n, STEP_THREADS), STEP_THREADS>>>(s->world, s->vehicle, s->cars, s->agents, s->n, drive, eliminate_on_wall, tick);
        CHECK(hipGetLastError());
        return 0;
    }
    CHECK(hipMemset(s->state, 0, sizeof(WindowState)));
    step_begin_kernel<<<blocks(s->n, STEP_THREADS), STEP_THREADS>>>(s->world, s->vehicle, s->cars, s->agents, s->n, drive, s->carry, s->items, s->state);
    sat_kernel<<<blocks(s->n, STEP_THREADS), STEP_THREADS>>>(s->world, s->vehicle, s->cars, s->items, s->state, s->pushes);
    step_end_kernel<<<blocks(s->n, STEP_THREADS), STEP_THREADS>>>(s->world, s->vehicle, s->cars, s->agents, s->n, drive, eliminate_on_wall, tick, s->carry,
                                              s->pushes, s->state);
    CHECK(hipGetLastError());
    return 0;
}

// Schedule::update_stats for all cars on a statistics tick.
int altd_gpu_sim_stats(Sim* s, uint64_t tick, uint64_t stats_tick, uint32_t eliminate_when_idle) {
    if (s->world.path_points < 2) return -2;  // Track::closest_path needs a path
    if (s->n == 0) return 0;
    stats_kernel<<<blocks(s->n, 64), 64>>>(s->world, s->cars, s->agents, s->n, tick, stats_tick, eliminate_when_idle, nullptr);
    CHECK(hipGetLastError());
    return 0;
}

// ---- whole windows ----
//
// TrainingRunner::advance_window (not paused, no native broadphase), in the
// tick-major order: per tick the statistics (on statistics ticks, followed by
// the stop test), then the batch's sensors + forward, then the physics step.
//
// The host only enqueues: the stop test runs on the device and every later
// kernel of the window returns at once. Kernels walk compact work lists,
// double-buffered and rebuilt on statistics ticks:
//  - live: the cars that may read sensors or change statistics (active, or
//    with a pending contact at the window start); after each statistics
//    update exactly the active cars. Cars deactivated by a wall stay listed
//    until the next rebuild and are filtered by their active flag. Each live
//    list has eight segments, retaining the cars' original inference batches.
//  - step: the cars still stepped. An inactive car whose step leaves every
//    field but `tick` bitwise unchanged is settled: `car_step` is a pure
//    function of that state for inactive cars (controls, agent and tick are
//    not read), so all its later steps repeat it. Settled cars leave the list
//    and get the skipped `tick` increments when the window ends.
constexpr uint32_t NOT_SETTLED = 0xffffffffu;

// Appends `value` where `take`, one atomic per wave; every lane must call it.
__device__ inline void append(bool take, uint32_t value, uint32_t* list, uint32_t* count) {
    uint64_t mask = __ballot(take);
    if (!mask) return;
    uint32_t lane = __lane_id();
    uint32_t leader = (uint32_t)__ffsll((unsigned long long)mask) - 1;
    uint32_t base = 0;
    if (lane == leader) base = atomicAdd(count, (uint32_t)__popcll(mask));
    base = (uint32_t)__shfl((int)base, (int)leader);
    if (take) list[base + (uint32_t)__popcll(mask & ((1ull << lane) - 1))] = value;
}

// The Car state `car_step` reads: everything but `tick`, and the used pairs and contacts.
__device__ inline void snapshot(const Car& c, Car& s) {
    __builtin_memcpy(&s, &c, offsetof(Car, pairs));
    __builtin_memcpy(s.pairs, c.pairs, sizeof(uint32_t) * c.pair_count);
    __builtin_memcpy(s.axes, c.axes, sizeof(Vec2) * c.pair_count);
    __builtin_memcpy(s.contacts, c.contacts, sizeof(Contact) * c.contact_count);
}

__device__ inline bool same_words(const void* a, const void* b, size_t bytes) {
    uint32_t differ = 0;
    for (size_t k = 0; k < bytes; k += 4) {
        uint32_t x, y;
        __builtin_memcpy(&x, (const char*)a + k, 4);
        __builtin_memcpy(&y, (const char*)b + k, 4);
        differ |= x ^ y;
    }
    return differ == 0;
}

__device__ inline bool same_state(const Car& s, const Car& c) {
    if (s.pair_count != c.pair_count || s.contact_count != c.contact_count) return false;
    return same_words(&s, &c, offsetof(Car, tick))
        && same_words((const char*)&s + offsetof(Car, tick) + 8, (const char*)&c + offsetof(Car, tick) + 8,
                      offsetof(Car, pairs) - offsetof(Car, tick) - 8)
        && same_words(s.pairs, c.pairs, sizeof(uint32_t) * c.pair_count)
        && same_words(s.axes, c.axes, sizeof(Vec2) * c.pair_count)
        && same_words(s.contacts, c.contacts, sizeof(Contact) * c.contact_count);
}

// Clears HAS_DEACTIVATED and builds the first live and step lists (index 0).
__global__ void w_start_kernel(Car* cars, Agent* agents, uint32_t n, uint32_t* live, uint32_t* steps, uint32_t* settled_at,
                               WindowState* st) {
    uint32_t batch_size = (n + 7) / 8, batch = blockIdx.y;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t i = batch * batch_size + j;
    bool valid = j < batch_size && i < n, listed = false;
    if (valid) {
        uint32_t flags = agents[i].flags & ~AGENT_HAS_DEACTIVATED;
        agents[i].flags = flags;
        settled_at[i] = NOT_SETTLED;
        listed = (cars[i].flags & CAR_ACTIVE) || (flags & AGENT_PENDING_CONTACT);
    }
    append(listed, i, live + batch * batch_size, &st->live_count[0][batch]);
    append(valid, i, steps, &st->step_count[0]);
}

// Statistics of the live cars in list p; the still active ones form list 1 - p.
__global__ void w_stats_kernel(World w, Car* cars, Agent* agents, const uint32_t* in, uint32_t* out, WindowState* st, uint32_t p, uint32_t batch_size,
                               uint64_t start_tick, uint32_t off, uint64_t first, uint32_t eliminate_when_idle) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch = blockIdx.y;
    bool valid = !st->stopped && j < st->live_count[p][batch], keep = false;
    uint32_t i = valid ? in[batch * batch_size + j] : 0;
    if (valid) {
        uint32_t err = 0;
        uint64_t tick = start_tick + st->base + off;
        update_stats(w, cars[i], agents[i], tick, tick - first, eliminate_when_idle != 0, err);
        cars[i].error |= err;
        keep = cars[i].flags & CAR_ACTIVE;
    }
    append(keep, i, out + batch * batch_size, &st->live_count[1 - p][batch]);
}

// The stop test after the statistics of tick k (live list 1 - p holds the
// active cars); clears the counts the next rebuilds append to.
__global__ void w_stop_kernel(WindowState* st, uint32_t p, uint64_t start_tick, uint32_t off, uint64_t first,
                              uint32_t has_time_limit, double time_limit, uint32_t stop_when_inactive) {
    if (st->stopped) return;
    uint64_t k = st->base + off;
    bool time_stop = has_time_limit && (double)((start_tick + k - first) / 6) * 0.1 >= time_limit;
    uint32_t live = 0;
    for (uint32_t b = 0; b < 8; b++) live += st->live_count[1 - p][b];
    if (time_stop || (stop_when_inactive && live == 0)) {
        st->stopped = 1;
        st->executed = k;
        return;
    }
    for (uint32_t b = 0; b < 8; b++) st->live_count[p][b] = 0;
    st->step_count[1 - p] = 0;
}

// The live lists retain original batch identity. Each y slice handles one
// scheduled batch; z slices of the sensor grid handle individual sensors.
__global__ void w_path_offset_kernel(World w, ReadMask m, Car* cars, const uint32_t* list, const WindowState* st, uint32_t p,
                                     float* here) {
    uint32_t batch = (m.batch_index + blockIdx.y) % 8;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (st->stopped || j >= st->live_count[p][batch]) return;
    uint32_t i = list[batch * m.batch_size + j];
    if (!(cars[i].flags & CAR_ACTIVE)) return;
    here[i] = sensor_path_offset(w, cars[i]);
}

__global__ void w_sensor_kernel(World w, VehicleDesc v, Sensors s, ReadMask m, Car* cars, const uint32_t* list,
                                const WindowState* st, uint32_t p, const float* here, double* inputs) {
    uint32_t batch = (m.batch_index + blockIdx.y) % 8;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x, k = blockIdx.z;
    if (st->stopped || j >= st->live_count[p][batch]) return;
    uint32_t i = list[batch * m.batch_size + j];
    if (!(cars[i].flags & CAR_ACTIVE)) return;
    uint32_t err = 0;
    inputs[(size_t)i * s.count + k] = sensor_value(w, v, cars[i], s.s[k], s.needs_path ? here[i] : 0.0f, err);
    if (err) atomicOr(&cars[i].error, err);
}

__global__ void __launch_bounds__(LANES * CARS_PER_BLOCK)
w_forward_kernel(NetShape shape, const double* __restrict__ params, size_t stride, ReadMask m, const Car* cars,
                 const uint32_t* list, const WindowState* st, uint32_t p, const double* __restrict__ inputs, Agent* agents) {
    uint32_t batch = (m.batch_index + blockIdx.y) % 8;
    uint32_t count = st->live_count[p][batch];
    if (st->stopped || blockIdx.x * CARS_PER_BLOCK >= count) return;  // uniform per block
    uint32_t j = blockIdx.x * CARS_PER_BLOCK + threadIdx.x / LANES;
    uint32_t car = j < count ? list[batch * m.batch_size + j] : 0;
    bool live = j < count && (cars[car].flags & CAR_ACTIVE);
    forward_car(shape, params, stride, live, car, inputs, agents);
}

// Steps the cars of step list q. With `out`, inactive cars are checked for
// settling and the others form step list 1 - q.
__global__ void __launch_bounds__(STEP_THREADS) w_step_kernel(World w, VehicleDesc v, Car* cars, Agent* agents, const uint32_t* in, uint32_t* out, WindowState* st,
                              uint32_t q, uint32_t* settled_at, Car* shadow, uint32_t eliminate_on_wall, uint64_t start_tick,
                              uint32_t off) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    bool valid = !st->stopped && j < st->step_count[q], keep = valid;
    uint32_t i = valid ? in[j] : 0;
    if (valid) {
        uint64_t k = st->base + off;  // the step index: one step per tick
        uint64_t tick = start_tick + k;
        uint32_t step_index = (uint32_t)k;
        Car& car = cars[i];
        bool check = out && !(car.flags & CAR_ACTIVE);
        if (check) snapshot(car, shadow[i]);
        step_car(w, v, car, agents[i], true, eliminate_on_wall != 0, tick);
        if (check && same_state(shadow[i], car)) {
            settled_at[i] = step_index;
            keep = false;
        }
    }
    if (out) append(keep, i, out, &st->step_count[1 - q]);
}

// w_step_kernel split (see sat_kernel): begin, then the SAT items, then end.
__global__ void __launch_bounds__(STEP_THREADS) w_step_begin_kernel(World w, VehicleDesc v, Car* cars, Agent* agents, const uint32_t* in, WindowState* st, uint32_t q,
                                    uint32_t settle, Car* shadow, StepCarry* carry, uint32_t* items) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (st->stopped) return;  // uniform
    bool valid = j < st->step_count[q];
    uint32_t i = valid ? in[j] : 0;
    if (valid && settle && !(cars[i].flags & CAR_ACTIVE)) snapshot(cars[i], shadow[i]);
    begin_car(w, v, cars, agents, valid, i, true, carry, items, &st->items);
}

__global__ void __launch_bounds__(STEP_THREADS) w_step_end_kernel(World w, VehicleDesc v, Car* cars, Agent* agents, const uint32_t* in, uint32_t* out, WindowState* st,
                                  uint32_t q, uint32_t* settled_at, const Car* shadow, uint32_t eliminate_on_wall, uint64_t start_tick,
                                  uint32_t off, const StepCarry* carry, const Vec2* pushes) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j == 0) st->items = 0;  // the SAT kernel is done with them
    bool valid = !st->stopped && j < st->step_count[q], keep = valid;
    uint32_t i = valid ? in[j] : 0;
    if (valid) {
        uint64_t k = st->base + off;
        const StepCarry& c = carry[i];
        bool check = out && !c.was_active;
        end_car(w, v, cars[i], agents[i], c, pushes, true, eliminate_on_wall != 0, start_tick + k);
        if (check && same_state(shadow[i], cars[i])) {
            settled_at[i] = (uint32_t)k;
            keep = false;
        }
    }
    if (out) append(keep, i, out, &st->step_count[1 - q]);
}

__global__ void w_advance_kernel(WindowState* st, uint32_t by) { st->base += by; }

// Adds the skipped steps to the settled cars' tick.
__global__ void w_finish_kernel(Car* cars, uint32_t n, const uint32_t* settled_at, const WindowState* st, uint64_t ticks) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || settled_at[i] == NOT_SETTLED) return;
    uint64_t steps = st->stopped ? st->executed - 1 : ticks;
    cars[i].tick += steps - settled_at[i];
}

int altd_gpu_sim_window(Sim* s, const WindowArgs* a, uint64_t* executed, uint32_t* transition_without_drive) {
    *executed = 0;
    *transition_without_drive = 0;
    if (a->ticks == 0) return 0;
    if (!s->params || s->world.shape_count == 0 || s->world.path_points < 2) return -2;
    uint32_t n = s->n;
    hipStream_t stream = s->stream;
    // ALTD_GPU_PROFILE=1: GPU time per phase (event pairs around direct launches).
    // ALTD_GPU_PROFILE=host: only the host thread's CPU time and the wall time.
    // ALTD_GPU_GRAPH=0: direct launches instead of the period graph.
    static const char* profile_env = getenv("ALTD_GPU_PROFILE");
    static const bool profile = profile_env && strcmp(profile_env, "host") != 0;
    static const char* graph_env = getenv("ALTD_GPU_GRAPH");
    static const bool use_graph = !profile && !(graph_env && strcmp(graph_env, "0") == 0);
    timespec cpu0, wall0;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu0);
    clock_gettime(CLOCK_MONOTONIC, &wall0);
    double spent[4] = {0, 0, 0, 0};  // stats, sensors, forward, step
    struct Timed { int slot; uint64_t k; hipEvent_t start, stop; };
    std::vector<Timed> timed;
    uint64_t enqueued = 0;  // ticks enqueued so far (profile timeline)
    auto phase = [&](int slot, auto&& launch) -> int {
        if (!profile) { launch(); return 0; }
        Timed t{slot, enqueued, nullptr, nullptr};
        CHECK(hipEventCreate(&t.start));
        CHECK(hipEventCreate(&t.stop));
        CHECK(hipEventRecord(t.start, stream));
        launch();
        CHECK(hipEventRecord(t.stop, stream));
        timed.push_back(t);
        return 0;
    };
    WindowState* st = s->state;
    CHECK(hipMemsetAsync(st, 0, sizeof(WindowState), stream));
    if (n) w_start_kernel<<<dim3(blocks((n + 7) / 8, 256), 8), 256, 0, stream>>>(s->cars, s->agents, n, s->live[0], s->steps[0], s->settled_at, st);
    uint64_t first = a->stats_phase == 0 ? 6 : a->stats_phase;
    // The time limit stops the window at a known tick; the host runs up to it.
    uint64_t limit = a->ticks;
    if (a->has_time_limit)
        for (uint64_t k = 1; k <= a->ticks; k++) {
            uint64_t tick = a->start_tick + k;
            if (tick % 6 == a->stats_phase && (double)((tick - first) / 6) * 0.1 >= a->time_limit) {
                limit = k;
                break;
            }
        }
    // Tick k = base + off of a period of PERIOD ticks: the statistics ticks,
    // read batches and list parity depend only on off, so a period is one graph.
    constexpr uint32_t PERIOD = 24;  // lcm of the statistics (6) and batch (8) cycles
    static_assert(PERIOD % 6 == 0 && PERIOD % 8 == 0, "period");
    auto enqueue_tick = [&](uint32_t off, uint32_t& p) {
        enqueued++;
        if (profile && enqueued % 600 == 1) {  // list sizes along the window
            WindowState now;
            (void)hipMemcpyAsync(&now, st, sizeof now, hipMemcpyDeviceToHost, stream);  // no CHECK: this lambda returns void
            (void)hipStreamSynchronize(stream);
            fprintf(stderr, "  tick %5llu: live %u/%u, steps %u/%u\n", (unsigned long long)enqueued, live_total(now, 0), live_total(now, 1),
                    now.step_count[0], now.step_count[1]);
        }
        uint64_t tick_mod = a->start_tick + off;
        uint32_t batch_index = (uint32_t)((a->start_batch + ((off - 1) % 8) * a->batches_per_tick) % 8);
        bool stats = tick_mod % 6 == a->stats_phase;
        if (stats) {
            if (n) phase(0, [&] {
                w_stats_kernel<<<dim3(blocks((n + 7) / 8, 64), 8), 64, 0, stream>>>(s->world, s->cars, s->agents, s->live[p], s->live[1 - p], st, p, (n + 7) / 8,
                                                                 a->start_tick, off, first, a->eliminate_when_idle);
                w_stop_kernel<<<1, 1, 0, stream>>>(st, p, a->start_tick, off, first, a->has_time_limit, a->time_limit,
                                                   a->stop_when_inactive);
            });
            else w_stop_kernel<<<1, 1, 0, stream>>>(st, p, a->start_tick, off, first, a->has_time_limit, a->time_limit,
                                                    a->stop_when_inactive);
            p = 1 - p;
        }
        if (n) {
            ReadMask mask{(uint32_t)((n + 7) / 8), batch_index, a->batches_per_tick, 1};
            phase(1, [&] {
                if (s->sensors.needs_path)
                    w_path_offset_kernel<<<dim3(blocks(mask.batch_size, 64), mask.batches_per_tick), 64, 0, stream>>>(s->world, mask, s->cars, s->live[p], st, p, s->here);
                w_sensor_kernel<<<dim3(blocks(mask.batch_size, 64), mask.batches_per_tick, s->sensors.count), 64, 0, stream>>>(s->world, s->vehicle, s->sensors, mask,
                                                                                           s->cars, s->live[p], st, p, s->here, s->inputs);
            });
            phase(2, [&] {
                w_forward_kernel<<<dim3(blocks(mask.batch_size, CARS_PER_BLOCK), mask.batches_per_tick), LANES * CARS_PER_BLOCK, 0, stream>>>(
                    s->shape, s->params, s->stride, mask, s->cars, s->live[p], st, p, s->inputs, s->agents);
            });
            phase(3, [&] {
                // On statistics ticks list 1 - p is read and list p rebuilt.
                uint32_t q = stats ? 1 - p : p;
                uint32_t* out = stats ? s->steps[p] : nullptr;
                if (!split_step()) {
                    w_step_kernel<<<blocks(n, STEP_THREADS), STEP_THREADS, 0, stream>>>(s->world, s->vehicle, s->cars, s->agents, s->steps[q], out, st, q,
                                                                    s->settled_at, s->shadow, a->eliminate_on_wall, a->start_tick, off);
                    return;
                }
                w_step_begin_kernel<<<blocks(n, STEP_THREADS), STEP_THREADS, 0, stream>>>(s->world, s->vehicle, s->cars, s->agents, s->steps[q], st, q,
                                                                      out != nullptr, s->shadow, s->carry, s->items);
                sat_kernel<<<blocks(n, STEP_THREADS), STEP_THREADS, 0, stream>>>(s->world, s->vehicle, s->cars, s->items, st, s->pushes);
                w_step_end_kernel<<<blocks(n, STEP_THREADS), STEP_THREADS, 0, stream>>>(s->world, s->vehicle, s->cars, s->agents, s->steps[q], out, st, q,
                                                                    s->settled_at, s->shadow, a->eliminate_on_wall, a->start_tick, off,
                                                                    s->carry, s->pushes);
            });
        }
    };
    auto enqueue_period = [&] {
        uint32_t p = 0;
        for (uint32_t off = 1; off <= PERIOD; off++) enqueue_tick(off, p);
        w_advance_kernel<<<1, 1, 0, stream>>>(st, PERIOD);
    };
    uint64_t periods = limit / PERIOD, rest = limit % PERIOD;
    hipGraphExec_t exec = nullptr;
    if (use_graph && periods > 1) {
        if (s->graph_exec && (s->graph_n != n || !same_schedule(s->graph_args, *a))) {
            CHECK(hipGraphExecDestroy(s->graph_exec));
            s->graph_exec = nullptr;
        }
        if (!s->graph_exec) {
            hipGraph_t graph;
            CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
            enqueue_period();
            CHECK(hipStreamEndCapture(stream, &graph));
            CHECK(hipGraphInstantiate(&s->graph_exec, graph, nullptr, nullptr, 0));
            CHECK(hipGraphDestroy(graph));
            s->graph_n = n;
            s->graph_args = *a;
        }
        exec = s->graph_exec;
    }
    // Polls the device stop flag every POLL periods, keeping DEPTH polls queued.
    constexpr uint64_t POLL = 2;
    constexpr int DEPTH = 3;
    uint32_t* seen = s->host_stopped;  // pinned, one slot per queued poll
    for (int c = 0; c < DEPTH; c++) seen[c] = 0;
    int polls = 0;
    bool host_stop = false;
    double launch_wall = 0, wait_wall = 0;  // host profile: time in launches and in polls
    auto now = [] { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec; };
    for (uint64_t m = 0; m < periods && !host_stop; m++) {
        double t0 = profile_env ? now() : 0;
        if (exec) CHECK(hipGraphLaunch(exec, stream));
        else enqueue_period();
        if (profile_env) launch_wall += now() - t0;
        CHECK(hipGetLastError());
        if (a->stop_when_inactive && (m + 1) % POLL == 0) {
            int slot = polls % DEPTH;
            CHECK(hipMemcpyAsync(&seen[slot], &st->stopped, sizeof(uint32_t), hipMemcpyDeviceToHost, stream));
            CHECK(hipEventRecord(s->chunk_done[slot], stream));
            polls++;
            if (polls >= DEPTH) {  // wait for the oldest queued poll
                int oldest = polls % DEPTH;
                double t1 = profile_env ? now() : 0;
                CHECK(hipEventSynchronize(s->chunk_done[oldest]));
                if (profile_env) wait_wall += now() - t1;
                host_stop = seen[oldest] != 0;
            }
        }
    }
    if (!host_stop) {
        uint32_t p = 0;
        for (uint32_t off = 1; off <= rest; off++) enqueue_tick(off, p);
    }
    if (n) w_finish_kernel<<<blocks(n, 256), 256, 0, stream>>>(s->cars, n, s->settled_at, st, a->ticks);
    CHECK(hipGetLastError());
    WindowState result;
    CHECK(hipMemcpyAsync(&result, st, sizeof result, hipMemcpyDeviceToHost, stream));
    CHECK(hipStreamSynchronize(stream));
    *executed = result.stopped ? result.executed : limit;
    *transition_without_drive = result.stopped;
    constexpr uint64_t SPAN = 600;  // profile timeline bucket, ticks
    std::vector<double> line(4 * (limit / SPAN + 1), 0.0);
    for (const Timed& t : timed) {
        float ms = 0.0f;
        CHECK(hipEventElapsedTime(&ms, t.start, t.stop));
        spent[t.slot] += ms * 1e-3;
        line[4 * ((t.k - 1) / SPAN) + t.slot] += ms * 1e-3;
        CHECK(hipEventDestroy(t.start));
        CHECK(hipEventDestroy(t.stop));
    }
    if (profile)
        fprintf(stderr, "altd_gpu window: %llu ticks, %u cars: stats %.3f s, sensors %.3f s, forward %.3f s, step %.3f s\n",
                (unsigned long long)*executed, n, spent[0], spent[1], spent[2], spent[3]);
    if (profile)
        for (size_t b = 0; 4 * b < line.size(); b++)
            fprintf(stderr, "  ticks %5zu..: stats %.3f sensors %.3f forward %.3f step %.3f\n", b * SPAN + 1, line[4 * b], line[4 * b + 1],
                    line[4 * b + 2], line[4 * b + 3]);
#ifdef ALTD_STEP_TIMING
    {
        unsigned long long c[16];
        int rate_khz = 0;
        CHECK(hipMemcpyFromSymbol(c, HIP_SYMBOL(g_step_cycles), sizeof c));
        CHECK(hipDeviceGetAttribute(&rate_khz, hipDeviceAttributeWallClockRate, 0));
        const char* names[6] = {"drive", "validate", "contacts+merge", "active", "solver", "finish"};
        double us = 1e3 / (double)rate_khz, steps = (double)(c[8] ? c[8] : 1);
        fprintf(stderr, "altd_gpu step sections (%llu car steps, %llu solving, %llu pair updates, %llu full scans), us per car step:",
                c[8], c[9], c[7], c[6]);
        for (int k = 0; k < 6; k++) fprintf(stderr, " %s %.2f", names[k], (double)c[k] * us / steps);
        fprintf(stderr, " (pairs %.2f, merge %.2f)", (double)c[10] * us / steps, (double)c[11] * us / steps);
        fprintf(stderr, "; SAT pairs %.2f per step, axes %.2f per pair, separated %.2f, wall points %.2f",
                (double)c[12] / steps, (double)c[13] / (double)(c[12] ? c[12] : 1), (double)c[14] / (double)(c[12] ? c[12] : 1),
                (double)c[15] / (double)(c[12] ? c[12] : 1));
        fprintf(stderr, "\n");
        unsigned long long zero[16] = {};
        CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_step_cycles), zero, sizeof zero));
    }
#endif
    if (profile_env) {
        timespec cpu1, wall1;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu1);
        clock_gettime(CLOCK_MONOTONIC, &wall1);
        auto seconds = [](timespec a, timespec b) { return (double)(b.tv_sec - a.tv_sec) + 1e-9 * (double)(b.tv_nsec - a.tv_nsec); };
        fprintf(stderr, "altd_gpu window: host thread cpu %.3f s of %.3f s wall (%s): launches %.3f s, polls %.3f s\n",
                seconds(cpu0, cpu1), seconds(wall0, wall1), exec ? "graph" : "direct", launch_wall, wait_wall);
    }
    return 0;
}

int altd_gpu_version() { return 8; }

// Runs one math primitive over `n` host inputs; `err` gets the per-element error bits.
int altd_gpu_math(int op, int n, const void* in, void* out, uint32_t* err) {
    void *d_in, *d_out;
    uint32_t* d_err;
    CHECK(hipMalloc(&d_in, in_size(op) * n));
    CHECK(hipMalloc(&d_out, out_size(op) * n));
    CHECK(hipMalloc(&d_err, 4 * (size_t)n));
    CHECK(hipMemcpy(d_in, in, in_size(op) * n, hipMemcpyHostToDevice));
    math_kernel<<<(n + 255) / 256, 256>>>(op, n, d_in, d_out, d_err);
    CHECK(hipGetLastError());
    CHECK(hipMemcpy(out, d_out, out_size(op) * n, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(err, d_err, 4 * (size_t)n, hipMemcpyDeviceToHost));
    CHECK(hipFree(d_in));
    CHECK(hipFree(d_out));
    CHECK(hipFree(d_err));
    return 0;
}

// Raw engine sin/cos for floats with bits first .. first + n - 1.
int altd_gpu_engine_scan(uint32_t first, uint32_t n, uint32_t* out) {
    static uint32_t* d_out = nullptr;
    static uint32_t capacity = 0;
    if (n > capacity) {
        if (d_out) CHECK(hipFree(d_out));
        CHECK(hipMalloc(&d_out, 8 * (size_t)n));
        capacity = n;
    }
    engine_scan_kernel<<<(n + 255) / 256, 256>>>(first, n, d_out);
    CHECK(hipGetLastError());
    CHECK(hipMemcpy(out, d_out, 8 * (size_t)n, hipMemcpyDeviceToHost));
    return 0;
}

}  // extern "C"
