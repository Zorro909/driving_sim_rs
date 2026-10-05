#!/usr/bin/env python3
"""Check HIP constructor cleanup without allocating GPU memory.

Usage: python3 gpu/test_constructor_cleanup.py [path/to/libaltd_gpu.so]

Requires Linux, a C++17 compiler, and the ROCm runtime needed to load the
library's GPU registration symbols. A temporary LD_PRELOAD shim replaces HIP
memory, event, and stream calls with ordinary host allocations. No kernels run.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


SHIM = r"""
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace {
enum Operation { DEVICE_ALLOC, COPY, HOST_ALLOC, EVENT_CREATE, STREAM_CREATE, OPERATIONS };
enum Resource { DEVICE, HOST, EVENT, STREAM, RESOURCES };
struct Allocation { Resource kind; size_t bytes; };
std::unordered_map<void*, Allocation> allocations;
std::array<size_t, OPERATIONS> calls{};
std::array<size_t, RESOURCES> outstanding{};
int failure_operation = -1;
size_t failure_call = 0;
size_t invalid_calls = 0;
constexpr int SUCCESS = 0;
constexpr int FAILURE = 2;

bool fail(Operation operation) {
    ++calls[operation];
    return failure_operation == operation && calls[operation] == failure_call;
}

int allocate(void** pointer, size_t bytes, Operation operation, Resource resource) {
    if (fail(operation)) return FAILURE;
    void* memory = std::malloc(bytes ? bytes : 1);
    if (!memory) std::abort();
    allocations.emplace(memory, Allocation{resource, bytes});
    ++outstanding[resource];
    *pointer = memory;
    return SUCCESS;
}

int release(void* pointer, Resource resource) {
    if (!pointer) return SUCCESS;
    auto allocation = allocations.find(pointer);
    if (allocation == allocations.end() || allocation->second.kind != resource) {
        ++invalid_calls;
        return FAILURE;
    }
    --outstanding[resource];
    allocations.erase(allocation);
    std::free(pointer);
    return SUCCESS;
}
}

extern "C" {
// These Linux C ABI signatures avoid a dependency on ROCm development headers.
int hipMalloc(void** pointer, size_t bytes) {
    return allocate(pointer, bytes, DEVICE_ALLOC, DEVICE);
}
int hipMemcpy(void* destination, const void* source, size_t bytes, int kind) {
    if (fail(COPY)) return FAILURE;
    auto allocation = allocations.find(destination);
    if (kind != 1 || allocation == allocations.end()
        || allocation->second.kind != DEVICE || allocation->second.bytes < bytes
        || (!source && bytes)) {
        ++invalid_calls;
        return FAILURE;
    }
    if (bytes) std::memcpy(destination, source, bytes);
    return SUCCESS;
}
int hipHostMalloc(void** pointer, size_t bytes, unsigned int) {
    return allocate(pointer, bytes, HOST_ALLOC, HOST);
}
int hipEventCreateWithFlags(void** event, unsigned int) {
    return allocate(event, 1, EVENT_CREATE, EVENT);
}
int hipStreamCreate(void** stream) {
    return allocate(stream, 1, STREAM_CREATE, STREAM);
}
int hipFree(void* pointer) { return release(pointer, DEVICE); }
int hipHostFree(void* pointer) { return release(pointer, HOST); }
int hipEventDestroy(void* event) { return release(event, EVENT); }
int hipStreamDestroy(void* stream) { return release(stream, STREAM); }
const char* hipGetErrorString(int) { return "injected HIP constructor test failure"; }

void cleanup_test_reset(int operation, size_t call) {
    calls.fill(0);
    failure_operation = operation;
    failure_call = call;
}
size_t cleanup_test_calls(int operation) { return calls.at(operation); }
size_t cleanup_test_outstanding(int resource) { return outstanding.at(resource); }
size_t cleanup_test_invalid_calls() { return invalid_calls; }
}
"""


RUNNER = r"""
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

struct alignas(16) float4 { float x, y, z, w; };
#include "world.h"

namespace {
enum Operation { DEVICE_ALLOC, COPY, HOST_ALLOC, EVENT_CREATE, STREAM_CREATE, OPERATIONS };
constexpr int RESOURCES = 4;
const char* operation_names[] = {
    "hipMalloc", "hipMemcpy", "hipHostMalloc", "hipEventCreateWithFlags", "hipStreamCreate"
};
const char* resource_names[] = {"device", "host", "event", "stream"};

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "constructor cleanup test: %s\n", message);
    std::exit(1);
}

template <typename T> T symbol(void* library, const char* name) {
    dlerror();
    void* address = dlsym(library, name);
    if (const char* error = dlerror()) fail(error);
    return reinterpret_cast<T>(address);
}

// Every World upload has a nonzero count and a distinct host pointer. The
// copied grid descriptors must release their uploaded starts and items too.
struct Fixture {
    altd::RayNode ray_nodes[1]{};
    float4 ray_walls[1]{};
    altd::SurfaceDev surfaces[1]{};
    altd::TileCell tiles[1]{};
    altd::PathSeg path_segments[1]{};
    double path_offsets[2]{};
    uint32_t path_start[2]{0, 1}, path_items[1]{};
    altd::Vec2 curve_position[2]{};
    float curve_offsets[2]{};
    altd::Vec2 curve_forwards[2]{};
    uint32_t curve_start[2]{0, 1}, curve_items[1]{};
    altd::ShapeDev shapes[1]{};
    altd::Vec2 shape_local[1]{}, shape_normals[1]{};
    uint32_t shape_start[2]{0, 1}, shape_items[1]{};
    double path_xy[4]{};
    altd::World world{};

    Fixture() {
        world.ray_node_count = world.ray_wall_count = world.ray_depth = 1;
        world.ray_nodes = ray_nodes;
        world.ray_walls = ray_walls;
        world.surface_count = 1;
        world.surfaces = surfaces;
        world.tile_nx = world.tile_ny = 1;
        world.tiles = tiles;
        world.path_points = 2;
        world.path_segments = path_segments;
        world.path_offsets = path_offsets;
        world.path_grid = {0, 0, 1, 1, 1, path_start, path_items};
        world.curve_points = 2;
        world.curve_position = curve_position;
        world.curve_offsets = curve_offsets;
        world.curve_forwards = curve_forwards;
        world.curve_grid = {0, 0, 1, 1, 1, curve_start, curve_items};
        world.shape_count = world.shape_point_count = 1;
        world.shapes = shapes;
        world.shape_local = shape_local;
        world.shape_normals = shape_normals;
        world.shape_grid = {0, 0, 1, 1, 1, shape_start, shape_items};
        world.path_xy = path_xy;
    }
};
}

int main(int argc, char** argv) {
    if (argc != 2) fail("expected library path");
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) fail(dlerror());
    auto world_create = symbol<altd::World* (*)(const altd::World*)>(library, "altd_gpu_world_create");
    auto world_free = symbol<void (*)(altd::World*)>(library, "altd_gpu_world_free");
    auto sim_create = symbol<void* (*)(const altd::World*, const altd::VehicleDesc*,
                                     const altd::SensorDesc*, uint32_t, uint32_t)>(library, "altd_gpu_sim_create");
    auto sim_free = symbol<void (*)(void*)>(library, "altd_gpu_sim_free");
    auto reset = symbol<void (*)(int, size_t)>(RTLD_DEFAULT, "cleanup_test_reset");
    auto calls = symbol<size_t (*)(int)>(RTLD_DEFAULT, "cleanup_test_calls");
    auto outstanding = symbol<size_t (*)(int)>(RTLD_DEFAULT, "cleanup_test_outstanding");
    auto invalid_calls = symbol<size_t (*)()>(RTLD_DEFAULT, "cleanup_test_invalid_calls");

    auto check_resources = [&](const std::array<size_t, RESOURCES>& expected, const char* context) {
        if (invalid_calls()) fail("constructor cleanup freed an unowned pointer or made an invalid copy");
        for (int resource = 0; resource < RESOURCES; ++resource) {
            size_t count = outstanding(resource);
            if (count != expected[resource]) {
                std::fprintf(stderr, "%s: %zu outstanding %s resources, expected %zu\n",
                             context, count, resource_names[resource], expected[resource]);
                fail("resource leak");
            }
        }
    };
    const std::array<size_t, RESOURCES> none{};
    check_resources(none, "before constructors");
    world_free(nullptr);
    sim_free(nullptr);
    check_resources(none, "null destructors");

    Fixture fixture;
    reset(-1, 0);
    altd::World* world = world_create(&fixture.world);
    if (!world) fail("successful world constructor returned null");
    std::array<size_t, OPERATIONS> world_calls{};
    for (int operation = 0; operation < OPERATIONS; ++operation) world_calls[operation] = calls(operation);
    if (world_calls[DEVICE_ALLOC] != 19 || world_calls[COPY] != 19)
        fail("world fixture did not exercise all 19 uploads through the fake runtime");
    world_free(world);
    check_resources(none, "successful world destruction");

    size_t world_failures = 0;
    for (int operation = 0; operation < OPERATIONS; ++operation) {
        for (size_t call = 1; call <= world_calls[operation]; ++call) {
            reset(operation, call);
            world = world_create(&fixture.world);
            if (world) fail("failed world constructor returned a handle");
            if (calls(operation) != call) fail("world failure injection was not reached");
            char context[128];
            std::snprintf(context, sizeof context, "world %s failure %zu", operation_names[operation], call);
            check_resources(none, context);
            ++world_failures;
        }
    }

    reset(-1, 0);
    world = world_create(&fixture.world);
    if (!world) fail("sim test world constructor returned null");
    std::array<size_t, RESOURCES> world_resources{};
    for (int resource = 0; resource < RESOURCES; ++resource) world_resources[resource] = outstanding(resource);
    altd::VehicleDesc vehicle{};
    reset(-1, 0);
    void* sim = sim_create(world, &vehicle, nullptr, 0, 2);
    if (!sim) fail("successful sim constructor returned null");
    std::array<size_t, OPERATIONS> sim_calls{};
    for (int operation = 0; operation < OPERATIONS; ++operation) sim_calls[operation] = calls(operation);
    if (sim_calls[DEVICE_ALLOC] != 16 || sim_calls[HOST_ALLOC] != 1
        || sim_calls[EVENT_CREATE] != 4 || sim_calls[STREAM_CREATE] != 1)
        fail("sim fixture did not exercise all allocations through the fake runtime");
    sim_free(sim);
    check_resources(world_resources, "successful sim destruction");

    size_t sim_failures = 0;
    for (int operation = 0; operation < OPERATIONS; ++operation) {
        for (size_t call = 1; call <= sim_calls[operation]; ++call) {
            reset(operation, call);
            sim = sim_create(world, &vehicle, nullptr, 0, 2);
            if (sim) fail("failed sim constructor returned a handle");
            if (calls(operation) != call) fail("sim failure injection was not reached");
            char context[128];
            std::snprintf(context, sizeof context, "sim %s failure %zu", operation_names[operation], call);
            check_resources(world_resources, context);
            ++sim_failures;
        }
    }
    reset(-1, 0);
    world_free(world);
    world_free(nullptr);
    sim_free(nullptr);
    check_resources(none, "final world destruction and null destructors");
    dlclose(library);
    check_resources(none, "library unload");
    std::printf("HIP constructor cleanup passed: %zu world failures, %zu sim failures, successful and null destructors\n",
                world_failures, sim_failures);
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    gpu_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "library", nargs="?", type=Path, default=gpu_dir.parent / "target/gpu/libaltd_gpu.so"
    )
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("this test requires Linux LD_PRELOAD")
    library = args.library.resolve()
    if not library.is_file():
        parser.error(f"library does not exist: {library}; build it with gpu/build.sh")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    with tempfile.TemporaryDirectory(prefix="altd-constructor-cleanup-") as directory:
        temporary = Path(directory)
        shim_source = temporary / "fake_hip.cpp"
        runner_source = temporary / "constructor_cleanup.cpp"
        shim_library = temporary / "fake_hip.so"
        runner = temporary / "constructor_cleanup"
        shim_source.write_text(SHIM)
        runner_source.write_text(RUNNER)
        common = [*compiler, "-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror"]
        subprocess.run(
            [*common, "-fPIC", "-shared", str(shim_source), "-o", str(shim_library)], check=True
        )
        subprocess.run(
            [*common, "-I", str(gpu_dir / "sim"), str(runner_source), "-ldl", "-o", str(runner)],
            check=True,
        )
        environment = os.environ.copy()
        environment["LD_PRELOAD"] = str(shim_library)
        result = subprocess.run(
            [str(runner), str(library)], env=environment, text=True, capture_output=True
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode:
            print(result.stderr, end="", file=sys.stderr)
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
