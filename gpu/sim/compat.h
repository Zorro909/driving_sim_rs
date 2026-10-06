// One source for AMD (hipcc) and NVIDIA (nvcc -x cu). Under HIP this header only includes the
// runtime and forwards the warp helpers to the HIP builtins; under CUDA it maps the hip* runtime
// names the simulator uses to their cuda* equivalents.
//
// Warp helpers: kernels call altd_lane_id/altd_ballot/altd_shfl*. On AMD a wave is warpSize
// (32 or 64) lanes and __ballot returns a 64-bit mask. On NVIDIA a warp is 32 lanes and the
// *_sync forms with a full mask are required; callers already have every lane of the wave call them.
#pragma once

// Marks the C ABI functions of the library. A Windows DLL exports nothing unless told to.
#if defined(_WIN32)
#define ALTD_API __declspec(dllexport)
#else
#define ALTD_API
#endif

#if defined(__CUDACC__)

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

// A clang builtin that nvcc's front end does not know.
#define __builtin_memcpy memcpy

#define hipError_t cudaError_t
#define hipSuccess cudaSuccess
#define hipErrorInvalidValue cudaErrorInvalidValue
#define hipGetErrorString cudaGetErrorString
#define hipGetLastError cudaGetLastError
#define hipGetDevice cudaGetDevice
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipDeviceProp_t cudaDeviceProp
#define hipGetDeviceProperties cudaGetDeviceProperties
#define hipDeviceGetAttribute cudaDeviceGetAttribute
#define hipDeviceAttributeWallClockRate cudaDevAttrClockRate

#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipHostMalloc(ptr, size) cudaMallocHost(ptr, size)
#define hipHostFree cudaFreeHost
#define hipMemcpy cudaMemcpy
#define hipMemcpyAsync cudaMemcpyAsync
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipMemset cudaMemset
#define hipMemsetAsync cudaMemsetAsync
#define hipMemcpyToSymbol cudaMemcpyToSymbol
#define hipMemcpyFromSymbol cudaMemcpyFromSymbol
#define HIP_SYMBOL(symbol) (symbol)

#define hipStream_t cudaStream_t
#define hipStreamCreate cudaStreamCreate
#define hipStreamDestroy cudaStreamDestroy
#define hipStreamSynchronize cudaStreamSynchronize
#define hipStreamBeginCapture cudaStreamBeginCapture
#define hipStreamEndCapture cudaStreamEndCapture
#define hipStreamCaptureModeThreadLocal cudaStreamCaptureModeThreadLocal

#define hipEvent_t cudaEvent_t
#define hipEventCreate cudaEventCreate
#define hipEventCreateWithFlags cudaEventCreateWithFlags
#define hipEventDisableTiming cudaEventDisableTiming
#define hipEventRecord cudaEventRecord
#define hipEventSynchronize cudaEventSynchronize
#define hipEventElapsedTime cudaEventElapsedTime
#define hipEventDestroy cudaEventDestroy

#define hipGraph_t cudaGraph_t
#define hipGraphExec_t cudaGraphExec_t
#define hipGraphLaunch cudaGraphLaunch
#define hipGraphDestroy cudaGraphDestroy
#define hipGraphExecDestroy cudaGraphExecDestroy
// HIP's (exec, graph, error node, log buffer, buffer size) form; CUDA 12+ takes flags instead.
static inline cudaError_t hipGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph, void*, void*, size_t) {
    return cudaGraphInstantiate(exec, graph, 0ull);
}

#define wall_clock64 clock64

// NVIDIA arithmetic turns every NaN result into its canonical NaN, dropping payload and sign. The
// CPU (x86 SSE) returns the first NaN operand with its quiet bit set, and the default NaN
// 0xffc00000 (0xfff8000000000000) for an invalid operation such as inf - inf.
// altd_nan_operand(a, b): the result of an add, multiply or divide when a or b is a NaN.
// altd_invalid(x): the result of x - x for a NaN or infinite x.
__device__ inline float altd_nan_operand(float a, float b) {
    return __uint_as_float(__float_as_uint(isnan(a) ? a : b) | 0x00400000u);
}
__device__ inline double altd_nan_operand(double a, double b) {
    return __longlong_as_double(__double_as_longlong(isnan(a) ? a : b) | 0x0008000000000000ll);
}
__device__ inline float altd_invalid(float x) { return isnan(x) ? altd_nan_operand(x, x) : __uint_as_float(0xffc00000u); }

__device__ inline uint32_t altd_lane_id() {
    uint32_t lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}
__device__ inline uint64_t altd_ballot(bool predicate) { return (uint64_t)__ballot_sync(0xffffffffu, predicate); }
__device__ inline uint32_t altd_shfl(uint32_t value, uint32_t lane) { return __shfl_sync(0xffffffffu, value, lane); }
__device__ inline uint32_t altd_shfl_up(uint32_t value, uint32_t delta) { return __shfl_up_sync(0xffffffffu, value, delta); }

#else

#include <hip/hip_runtime.h>
#include <cstdint>

// AMD hardware already matches the CPU's NaN results.
__device__ inline float altd_nan_operand(float a, float b) { return a + b; }
__device__ inline double altd_nan_operand(double a, double b) { return a + b; }
__device__ inline float altd_invalid(float x) { return x - x; }

__device__ inline uint32_t altd_lane_id() { return __lane_id(); }
__device__ inline uint64_t altd_ballot(bool predicate) { return __ballot(predicate); }
__device__ inline uint32_t altd_shfl(uint32_t value, uint32_t lane) { return (uint32_t)__shfl((int)value, (int)lane); }
__device__ inline uint32_t altd_shfl_up(uint32_t value, uint32_t delta) { return (uint32_t)__shfl_up((int)value, delta); }

#endif
