// clock_gettime for the profiling timers of altd_gpu.hip on Windows. Include it after every other
// header: windows.h defines `near` and `far`, which the simulator headers use as variable names.
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <time.h>

#define CLOCK_MONOTONIC 1
#define CLOCK_THREAD_CPUTIME_ID 3

static inline int clock_gettime(int clock, timespec* t) {
    if (clock == CLOCK_MONOTONIC) {
        LARGE_INTEGER frequency, count;
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&count);
        t->tv_sec = (time_t)(count.QuadPart / frequency.QuadPart);
        t->tv_nsec = (long)((count.QuadPart % frequency.QuadPart) * 1000000000ll / frequency.QuadPart);
        return 0;
    }
    FILETIME created, exited, kernel, user;  // 100 ns units
    if (!GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user)) return -1;
    ULARGE_INTEGER k, u;
    k.LowPart = kernel.dwLowDateTime, k.HighPart = kernel.dwHighDateTime;
    u.LowPart = user.dwLowDateTime, u.HighPart = user.dwHighDateTime;
    unsigned long long ticks = k.QuadPart + u.QuadPart;
    t->tv_sec = (time_t)(ticks / 10000000ull);
    t->tv_nsec = (long)((ticks % 10000000ull) * 100ull);
    return 0;
}
