// Every float32: genuine ucrtbase sinf/cosf (FMA3) vs Wine's builtin (musl). Writes the
// differing inputs as (input bits, genuine bits, builtin bits) and an order-independent hash of
// the builtin outputs, so a port of musl can be checked against the builtin without the table.
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef float (*f1_t)(float);
typedef int (*setfma_t)(int);
static uint64_t mix(uint64_t z) { z += 0x9e3779b97f4a7c15ull; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }
typedef struct { uint64_t lo, hi, h; f1_t g, b; uint32_t *out; size_t n, cap; } job_t;
static DWORD WINAPI run(void *p) {
    job_t *j = p;
    for (uint64_t i = j->lo; i < j->hi; i++) {
        uint32_t bits = (uint32_t)i, rg, rb; float v; memcpy(&v, &bits, 4);
        float og = j->g(v), ob = j->b(v); memcpy(&rg, &og, 4); memcpy(&rb, &ob, 4);
        j->h += mix(((uint64_t)bits << 32) | rb);
        if (rg != rb) {
            if (j->n == j->cap) { j->cap = j->cap ? j->cap * 2 : 1024; j->out = realloc(j->out, j->cap * 12); }
            j->out[3 * j->n] = bits; j->out[3 * j->n + 1] = rg; j->out[3 * j->n + 2] = rb; j->n++;
        }
    }
    return 0;
}
int main(int argc, char **argv) {
    HMODULE m = LoadLibraryA(argv[1]);
    HMODULE w = LoadLibraryA("ucrtbase.dll");
    if (!m || !w || m == w) { printf("load failed %p %p\n", m, w); return 2; }
    ((setfma_t)GetProcAddress(m, "_set_FMA3_enable"))(atoi(argv[3]));
    const char *names[] = {"sinf", "cosf"};
    for (int fn = 0; fn < 2; fn++) {
        enum { T = 32 }; job_t jobs[T]; HANDLE th[T];
        for (int t = 0; t < T; t++) {
            jobs[t] = (job_t){(1ull << 32) * t / T, (1ull << 32) * (t + 1) / T, 0, (f1_t)GetProcAddress(m, names[fn]), (f1_t)GetProcAddress(w, names[fn]), 0, 0, 0};
            th[t] = CreateThread(0, 0, run, &jobs[t], 0, 0);
        }
        WaitForMultipleObjects(T, th, TRUE, INFINITE);
        char path[512]; snprintf(path, sizeof path, "%s\\%s.diff", argv[2], names[fn]);
        FILE *f = fopen(path, "wb");
        uint64_t h = 0, n = 0;
        for (int t = 0; t < T; t++) { h += jobs[t].h; n += jobs[t].n; fwrite(jobs[t].out, 12, jobs[t].n, f); }
        fclose(f);
        printf("%s diffs=%llu builtin_hash=%016llx\n", names[fn], (unsigned long long)n, (unsigned long long)h);
    }
    return 0;
}
