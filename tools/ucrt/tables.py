#!/usr/bin/env python3
"""Extract the tables of the Windows UCRT math kernels (math/kernels/ucrt.h).

Reads two genuine x64 ucrtbase.dll builds and the exhaustive sinf/cosf scans of
scan_sincos.c, and writes math/kernels/ucrt_tables.h. The DLLs are not part of
the repository; only the extracted values are committed.

  --win10  ucrtbase.dll 10.0.22621.7510 (Windows 10, Windows 11 up to 23H2)
  --win11  ucrtbase.dll 10.0.26100.9444 (Windows 11 24H2 and later)
  --scan   directory with sinf.diff and cosf.diff of scan_sincos.c for the
           win11 build (the win10 build has the same differences)

Each scan row is (input bits, genuine result bits, Wine builtin result bits) as
little-endian u32. Wine's builtin sinf/cosf are musl's, so the genuine function
is musl plus the rows: src/math/native_math.rs over every float32 reproduces
the builtin outputs that scan_sincos.c hashes.
"""

import argparse
import hashlib
from pathlib import Path
import struct

root = Path(__file__).resolve().parents[2]

WIN10_SHA256 = "3a1af295ce81bba0d351b192d19fabf0f1bdf81ae044c374a62a4704343b0a2b"
WIN11_SHA256 = "5c52e3a303ba"  # prefix, versions.tsv

# musl sinf/cosf on the GPU stop at |x| >= 0x4dc90fdb (gpu/sim/math.h
# ERR_NATIVE_LARGE), so rows below it are split from the wide rest.
GPU_LIMIT = 0x4DC90FDB


class Image:
    def __init__(self, path):
        self.data = path.read_bytes()
        d = self.data
        pe = struct.unpack_from("<I", d, 0x3C)[0]
        sections = struct.unpack_from("<H", d, pe + 6)[0]
        optional = struct.unpack_from("<H", d, pe + 20)[0]
        o = pe + 24
        self.base = struct.unpack_from("<Q", d, o + 24)[0]
        self.sections = []
        for i in range(sections):
            _, size, va, _, raw = struct.unpack_from("<8sIIII", d, o + optional + 40 * i)
            self.sections.append((va, size, raw))

    def read(self, va, n):
        rva = va - self.base
        for start, size, raw in self.sections:
            if start <= rva and rva + n <= start + size:
                return self.data[rva - start + raw : rva - start + raw + n]
        raise ValueError(f"{va:#x} is not mapped")

    def u64(self, va, n, stride=8):
        return [struct.unpack("<Q", self.read(va + stride * i, 8))[0] for i in range(n)]


def scan(path):
    d = path.read_bytes()
    return [struct.unpack_from("<III", d, i) for i in range(0, len(d), 12)]


def exceptions(rows, odd):
    """Sorted (|x| bits, result bits for +|x|) of a scan. sinf is odd, cosf even."""
    table = {}
    for x, genuine, _ in rows:
        key = x & 0x7FFFFFFF
        value = genuine ^ (x & 0x80000000) if odd else genuine
        assert table.setdefault(key, value) == value, hex(x)
    # Both signs of every input differ.
    assert len(rows) == 2 * len(table)
    return sorted(table.items())


def emit_array(out, name, kind, values):
    out.append(f"ALTD_MATH_TABLE {kind} {name}[{len(values)}] = {{")
    suffix = "ull" if kind == "uint64_t" else "u"
    width = 16 if kind == "uint64_t" else 8
    per = 4 if kind == "uint64_t" else 6
    for i in range(0, len(values), per):
        out.append("    " + " ".join(f"0x{v:0{width}x}{suffix}," for v in values[i : i + per]))
    out.append("};")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--win10", type=Path, required=True)
    parser.add_argument("--win11", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=root / "math/kernels/ucrt_tables.h")
    args = parser.parse_args()
    assert hashlib.sha256(args.win10.read_bytes()).hexdigest() == WIN10_SHA256, "not ucrtbase 10.0.22621.7510"
    assert hashlib.sha256(args.win11.read_bytes()).hexdigest().startswith(WIN11_SHA256), "not ucrtbase 10.0.26100.9444"
    w10 = Image(args.win10)
    w11 = Image(args.win11)

    arrays = []  # (comment, name, kind, values)
    arrays.append(("atan2f: atan(n / 256) for n = 16..256", "UCRT_ATAN2F_ATAN256", "uint64_t", w10.u64(0x1800D46D0, 241)))
    arrays.append(("tanh: 2^(j/32) split in high and low parts", "UCRT_TANH_HI", "uint64_t", w10.u64(0x1800D4EA0, 32)))
    arrays.append(("", "UCRT_TANH_LO", "uint64_t", w10.u64(0x1800D4FA0, 32)))
    exp = [w10.u64(0x1800D3410, 64), w10.u64(0x1800D3210, 64), w10.u64(0x1800D3010, 64)]
    assert exp == [w11.u64(0x18010C580, 64), w11.u64(0x18010C380, 64), w11.u64(0x18010C180, 64)]
    arrays.append(("exp and the win10 pow: 2^(j/64) as a multiplier and low and high parts", "UCRT_EXP_MUL", "uint64_t", exp[0]))
    arrays.append(("", "UCRT_EXP_LO", "uint64_t", exp[1]))
    arrays.append(("", "UCRT_EXP_HI", "uint64_t", exp[2]))
    arrays.append(("win10 pow: logarithm tables over 257 mantissa intervals", "UCRT_POW_T1", "uint64_t", w10.u64(0x1800D0CF0, 257)))
    arrays.append(("", "UCRT_POW_T2", "uint64_t", w10.u64(0x1800D1500, 257)))
    arrays.append(("", "UCRT_POW_T3", "uint64_t", w10.u64(0x1800CEF00, 257)))
    arrays.append(("", "UCRT_POW_T4", "uint64_t", w10.u64(0x1800CF710, 257)))
    arrays.append(("win11 pow (ARM optimized-routines pow): log table, N = 128", "UCRT_POW11_INVC", "uint64_t", w11.u64(0x180102DE8, 128, 32)))
    arrays.append(("", "UCRT_POW11_LOGC", "uint64_t", w11.u64(0x180102DF8, 128, 32)))
    arrays.append(("", "UCRT_POW11_LOGCTAIL", "uint64_t", w11.u64(0x180102E00, 128, 32)))
    arrays.append(("win11 pow: exp table, N = 256, (tail, scale bits) pairs", "UCRT_POW11_EXP", "uint64_t", w11.u64(0x180103EB0, 512)))
    cpu_only = len(arrays)
    log = [w11.u64(0x18010A650, 257), w11.u64(0x180109420, 257), w11.u64(0x180109C30, 257)]
    for table in log:
        assert w10.data.find(b"".join(struct.pack("<Q", v) for v in table)) >= 0
    arrays.append(("log (both groups): 1/F, and log(F) in lead and tail parts, over 257 intervals", "UCRT_LOG_INV", "uint64_t", log[0]))
    arrays.append(("", "UCRT_LOG_LEAD", "uint64_t", log[1]))
    arrays.append(("", "UCRT_LOG_TAIL", "uint64_t", log[2]))

    counts = []
    for fn, odd in [("sinf", True), ("cosf", False)]:
        table = exceptions(scan(args.scan / f"{fn}.diff"), odd)
        low = [e for e in table if e[0] < GPU_LIMIT]
        wide = [e for e in table if e[0] >= GPU_LIMIT]
        up = fn.upper()
        what = "the bits of sinf(|x|), negated for negative x" if odd else "result bits"
        arrays.append((f"{fn}: musl's result is wrong by 1 ulp for these |x| bits; values are {what}. |x| < {GPU_LIMIT:#x}", f"UCRT_{up}_KEYS", "uint32_t", [k for k, _ in low]))
        arrays.append(("", f"UCRT_{up}_VALUES", "uint32_t", [v for _, v in low]))
        arrays.append((f"{fn}, |x| >= {GPU_LIMIT:#x} (CPU only)", f"UCRT_{up}_WIDE_KEYS", "uint32_t", [k for k, _ in wide]))
        arrays.append(("", f"UCRT_{up}_WIDE_VALUES", "uint32_t", [v for _, v in wide]))
        counts += [(f"UCRT_{up}_COUNT", len(low)), (f"UCRT_{up}_WIDE_COUNT", len(wide))]

    out = [
        "// Generated by tools/ucrt/tables.py from ucrtbase.dll 10.0.22621.7510 and 10.0.26100.9444. Do not edit.",
        "// Bit patterns; math/kernels/ucrt.h documents their use.",
        "#pragma once",
    ]
    for comment, name, kind, values in arrays[:cpu_only]:
        if comment:
            out.append(f"// {comment}")
        emit_array(out, name, kind, values)
    # WGSL has no sinf/cosf or log of its own; gen-sim.py drops the region.
    out.append("#ifndef ALTD_MATH_WGSL")
    for name, n in counts:
        step = 1
        while step * 2 < n:
            step *= 2
        out.append(f"ALTD_MATH_CONST int {name} = {n};")
        out.append(f"ALTD_MATH_CONST int {name[:-6]}_STEP = {step if n > 1 else 0};")
    for comment, name, kind, values in arrays[cpu_only:]:
        if comment:
            out.append(f"// {comment}")
        emit_array(out, name, kind, values)
    out.append("#endif")
    args.output.write_text("\n".join(out) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
