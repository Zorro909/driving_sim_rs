#!/usr/bin/env python3
"""Benchmark identical GPU generations; build examples/gpu_check first."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cars", type=int, nargs="+", default=[8192, 32768, 262144])
    parser.add_argument("--ticks", type=int, default=5400)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--no-elimination", action="store_true")
    parser.add_argument("--training", action="store_true", help="include turnover across consecutive generations")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--binary", type=Path, help="gpu_check executable (also supports saved baselines)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    library = args.library.resolve()
    binary = args.binary.resolve() if args.binary else root / "target/release/examples/gpu_check"
    if args.samples < 1 or args.warmups < 0 or args.generations < 1:
        parser.error("samples and generations must be positive; warmups must be nonnegative")
    if args.training and args.no_elimination:
        parser.error("the training benchmark uses the fixture's elimination settings")
    report = {
        "benchmark": "training" if args.training else "fixed_generation",
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "library": str(library),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "requested_ticks": args.ticks,
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "default"),
        "graph": os.environ.get("ALTD_GPU_GRAPH", "default"),
        "split": os.environ.get("ALTD_GPU_SPLIT", "default"),
        "profile": os.environ.get("ALTD_GPU_PROFILE", "off"),
        "training_profile": os.environ.get("ALTD_TRAIN_PROFILE", "off"),
        "generations": args.generations if args.training else 1,
        "samples": args.samples,
        "warmups": args.warmups,
        "eliminate": not args.no_elimination,
        "results": [],
    }
    checkpoint = Path(os.environ.get("ALTD_GPU_CKPT", "/tmp/altd-gpu-ckpt"))
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    report["checkpoint_sha256"] = hashlib.sha256((checkpoint / metadata["population_file"]).read_bytes()).hexdigest()
    for cars in args.cars:
        env = dict(os.environ, ALTD_GPU_LIB=str(library), ALTD_GPU_BENCH_CARS=str(cars),
                   ALTD_GPU_BENCH_TICKS=str(args.ticks), ALTD_GPU_BENCH_SAMPLES=str(args.samples),
                   ALTD_GPU_BENCH_WARMUPS=str(args.warmups),
                   ALTD_GPU_BENCH_GENERATIONS=str(args.generations),
                   ALTD_GPU_BENCH_ELIMINATE=str(int(not args.no_elimination)))
        command = [str(binary), "benchtrain" if args.training else "benchfixed"]
        records = []
        for sample in range(args.warmups + args.samples if args.training else 1):
            trial = []
            with subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE, text=True) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    if line.startswith("{"):
                        record = json.loads(line)
                        if args.training:
                            record.update(sample=sample, warmup=sample < args.warmups)
                        trial.append(record)
                if process.wait():
                    raise SystemExit(f"benchmark failed for {cars} cars")
            if args.training and [r["generation"] for r in trial] != list(range(args.generations)):
                raise SystemExit("missing training generations")
            records.extend(trial)
        if args.training:
            generations = []
            for generation in range(args.generations):
                runs = [r for r in records if r["generation"] == generation]
                timed = [r for r in runs if not r["warmup"]]
                if len(timed) != args.samples or len({r["digest"] for r in runs}) != 1:
                    raise SystemExit("missing samples or inconsistent training results")
                generations.append({"generation": generation, "digest": runs[0]["digest"], **{
                    "median_" + key: statistics.median(r[key] for r in timed)
                    for key in ["simulate_seconds", "turnover_seconds", "total_seconds"]}})
            timed = [sum(r["total_seconds"] for r in records if r["sample"] == sample) / args.generations
                     for sample in range(args.warmups, args.warmups + args.samples)]
            result = {"cars": cars, "median_seconds": statistics.median(timed), "generations": generations, "runs": records}
        else:
            timed = [r["seconds"] for r in records if not r["warmup"]]
            if len(timed) != args.samples or len({r["digest"] for r in records}) != 1:
                raise SystemExit("missing samples or inconsistent results")
            result = {"cars": cars, "median_seconds": statistics.median(timed), "runs": records}
        report["results"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
