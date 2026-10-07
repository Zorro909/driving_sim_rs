#!/usr/bin/env python3
"""Sample the harness outputs (ucrt_harness.c) into math/fixtures/<function>.bin.

Each row holds the input bits and then the result bits for proton (Wine's musl
column), win10-fma3 (the G01 FMA3 column) and win11-fma3 (the G02 FMA3 column),
little-endian, u32 for the float functions and u64 for the double functions.
The rows are every input on which the profiles differ, thinned evenly to --diff
rows, plus --even rows spread over all inputs. sinf and cosf differ on few of
the harness inputs, so their rows also sample the exhaustive scans of
scan_sincos.c (the inputs of math/kernels/ucrt_tables.h, whose sinf and cosf
are one for both Windows profiles).

    python3 tools/ucrt/fixtures.py /path/to/altd-wine --scan /path/to/scans
"""

import argparse
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
OUT = root / "math/fixtures"

# (function, input file, input words per row, word type)
FUNCTIONS = [
    ("sinf", "trig.f32", 1, np.uint32),
    ("cosf", "trig.f32", 1, np.uint32),
    ("atan2f", "atan2.f32x2", 2, np.uint32),
    ("exp", "exp.f64", 1, np.uint64),
    ("log", "log.f64", 1, np.uint64),
    ("tanh", "tanh.f64", 1, np.uint64),
    ("pow", "pow.f64x2", 2, np.uint64),
]


def thin(rows, count):
    if len(rows) <= count:
        return rows
    return rows[np.linspace(0, len(rows) - 1, count).astype(np.int64)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wine", type=Path, help="the altd-wine directory (harness/in*, versions/results*)")
    parser.add_argument("--diff", type=int, default=2000)
    parser.add_argument("--even", type=int, default=500)
    parser.add_argument("--scan", type=Path, required=True, help="directory with sinf.diff and cosf.diff")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, file, arity, word in FUNCTIONS:
        tables = []
        for inputs, results in [("in", "results"), ("in_wide", "results_in_wide")]:
            x = np.fromfile(args.wine / "harness" / inputs / file, dtype=word).reshape(-1, arity)
            g01 = np.fromfile(args.wine / "versions" / results / "G01" / f"{name}.out", dtype=word).reshape(-1, 3)
            g02 = np.fromfile(args.wine / "versions" / results / "G02" / f"{name}.out", dtype=word).reshape(-1, 3)
            assert len(x) == len(g01) == len(g02), (name, inputs)
            assert (g01[:, 0] == g02[:, 0]).all(), (name, "musl columns differ")
            tables.append(np.column_stack([x, g01[:, 0], g01[:, 1], g02[:, 1]]))
        if name in ["sinf", "cosf"]:
            # Scan rows: input, genuine, Wine builtin.
            scan = np.fromfile(args.scan / f"{name}.diff", dtype=np.uint32).reshape(-1, 3)
            tables.append(np.column_stack([scan[:, 0], scan[:, 2], scan[:, 1], scan[:, 1]]))
        rows = np.concatenate(tables)
        rows = np.unique(rows, axis=0)
        p, w10, w11 = rows[:, arity], rows[:, arity + 1], rows[:, arity + 2]
        differ = (p != w10) | (p != w11) | (w10 != w11)
        picked = np.concatenate([thin(rows[differ], args.diff), thin(rows, args.even)])
        picked = np.unique(picked, axis=0)
        picked.astype("<u4" if word is np.uint32 else "<u8").tofile(OUT / f"{name}.bin")
        print(f"{name}: {len(rows)} inputs, {differ.sum()} differ, {len(picked)} rows")


if __name__ == "__main__":
    main()
