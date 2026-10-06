// Checks math/kernels/ucrt.h against outputs of the genuine DLLs (ucrt_harness.c rows of
// wine/musl, MS FMA3 and MS SSE2 result bits), compiled as C and as C++:
//
//   cc  -std=c11   -O2 -ffp-contract=off -I math/kernels tools/ucrt/check_kernels.c -o /tmp/ck -lm
//   c++ -std=c++20 -O2 -ffp-contract=off -I math/kernels -x c++ tools/ucrt/check_kernels.c -o /tmp/ck++
//   /tmp/ck {win10|win11} <in-dir> <results-dir>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

static inline uint32_t fbits(float x) { uint32_t b; memcpy(&b, &x, 4); return b; }
static inline float ffrom(uint32_t b) { float x; memcpy(&x, &b, 4); return x; }
static inline uint64_t dbits(double x) { uint64_t b; memcpy(&b, &x, 8); return b; }
static inline double dfrom(uint64_t b) { double x; memcpy(&x, &b, 8); return x; }

#define ALTD_MATH_FN static inline
#define ALTD_MATH_TABLE static const
#define ALTD_MATH_CONST static const
#include "ucrt.h"

static void *slurp(const char *dir, const char *name, size_t *size) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("cannot open %s\n", path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    *size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc(*size);
    if (fread(buf, 1, *size, f) != *size) exit(2);
    fclose(f);
    return buf;
}

static int failures;

static void report(const char *name, size_t n, size_t bad) {
    printf("%-7s %9zu inputs, %zu mismatches\n", name, n, bad);
    if (bad) failures++;
}

static void miss64(const char *name, size_t *bad, uint64_t a, uint64_t b, uint64_t got, uint64_t want) {
    if (++*bad <= 8) printf("  %s(%016llx, %016llx) = %016llx, want %016llx\n", name, (unsigned long long)a, (unsigned long long)b, (unsigned long long)got, (unsigned long long)want);
}

static void check_d1(int win11, const char *in, const char *res, const char *name, const char *file) {
    size_t size;
    const uint64_t *x = (const uint64_t *)slurp(in, file, &size);
    size_t n = size / 8;
    char out[64];
    snprintf(out, sizeof out, "%s.out", name);
    const uint64_t *r = (const uint64_t *)slurp(res, out, &size);
    if (size != n * 24) exit(3);
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) {
        double v = dfrom(x[i]), got;
        if (!strcmp(name, "exp")) got = win11 ? win11_exp(v) : win10_exp(v);
        else if (!strcmp(name, "tanh")) got = win11 ? win11_tanh(v) : win10_tanh(v);
        else got = win_log(v);
        if (dbits(got) != r[i * 3 + 1]) miss64(name, &bad, x[i], 0, dbits(got), r[i * 3 + 1]);
    }
    report(name, n, bad);
}

static void check_pow(int win11, const char *in, const char *res) {
    size_t size;
    const uint64_t *x = (const uint64_t *)slurp(in, "pow.f64x2", &size);
    size_t n = size / 16;
    const uint64_t *r = (const uint64_t *)slurp(res, "pow.out", &size);
    if (size != n * 24) exit(3);
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) {
        double a = dfrom(x[i * 2]), b = dfrom(x[i * 2 + 1]);
        double got = win11 ? win11_pow(a, b) : win10_pow(a, b);
        if (dbits(got) != r[i * 3 + 1]) miss64("pow", &bad, x[i * 2], x[i * 2 + 1], dbits(got), r[i * 3 + 1]);
    }
    report("pow", n, bad);
}

static void check_f(const char *in, const char *res, const char *name) {
    size_t size;
    int two = !strcmp(name, "atan2f");
    const uint32_t *x = (const uint32_t *)slurp(in, two ? "atan2.f32x2" : "trig.f32", &size);
    size_t n = size / (two ? 8 : 4);
    char out[64];
    snprintf(out, sizeof out, "%s.out", name);
    const uint32_t *r = (const uint32_t *)slurp(res, out, &size);
    if (size != n * 12) exit(3);
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) {
        float got;
        // Column 0 is Wine's builtin, which is musl.
        float musl = ffrom(r[i * 3]);
        if (two) got = win_atan2f(ffrom(x[i * 2]), ffrom(x[i * 2 + 1]));
        else if (!strcmp(name, "sinf")) got = win_sinf_fix_all(ffrom(x[i]), musl);
        else got = win_cosf_fix_all(ffrom(x[i]), musl);
        if (fbits(got) != r[i * 3 + 1] && ++bad <= 8)
            printf("  %s(%08x, %08x) = %08x, want %08x\n", name, x[i * (two ? 2 : 1)], two ? x[i * 2 + 1] : 0u, fbits(got), r[i * 3 + 1]);
    }
    report(name, n, bad);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("usage: %s {win10|win11} <in-dir> <results-dir>\n", argv[0]);
        return 2;
    }
    int win11 = !strcmp(argv[1], "win11");
    check_f(argv[2], argv[3], "sinf");
    check_f(argv[2], argv[3], "cosf");
    check_f(argv[2], argv[3], "atan2f");
    check_d1(win11, argv[2], argv[3], "exp", "exp.f64");
    check_d1(win11, argv[2], argv[3], "log", "log.f64");
    check_d1(win11, argv[2], argv[3], "tanh", "tanh.f64");
    check_pow(win11, argv[2], argv[3]);
    return failures != 0;
}
