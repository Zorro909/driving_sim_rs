#!/usr/bin/env python3
"""Measure the public generated-track CPU workload, retaining every sample."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/altd-sim")
    parser.add_argument("--scene", type=Path, default=ROOT / "wasm/test/scenarios/rally_mixed.scene.json")
    parser.add_argument("--spawn-trace", type=Path, default=ROOT / "wasm/test/scenarios/rally_mixed.spawn.json")
    parser.add_argument("--network", type=Path, default=ROOT / "wasm/test/scenarios/benchmark.network.json")
    parser.add_argument("--model", type=Path, default=ROOT / "assets/models/rally.json")
    parser.add_argument("--population", type=int, default=1024)
    parser.add_argument("--ticks", type=int, default=3600)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, nargs="+", default=sorted({1, min(8, os.cpu_count() or 1)}))
    parser.add_argument("--modes", choices=["independent", "lockstep"], nargs="+", default=["independent"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if min(args.population, args.ticks, args.repeats, *args.threads) < 1:
        parser.error("population, ticks, repeats and thread counts must be positive")
    common = ["--scene", str(args.scene.resolve()), "--spawn-trace", str(args.spawn_trace.resolve()),
              "--network", str(args.network.resolve()), "--model", str(args.model.resolve()),
              "--population", str(args.population), "--ticks", str(args.ticks),
              "--warmup-ticks", "2", "--seed", str(args.seed)]
    report = {"workload": "generated-track", "results": []}
    for mode in args.modes:
        for threads in args.threads:
            command = [str(args.binary.resolve()), "--mode", mode, "--threads", str(threads), "bench", *common]
            def run():
                return json.loads(subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout)
            run()  # Untimed process warmup, in addition to each run's simulation warmup.
            runs = [run() for _ in range(args.repeats)]
            median = statistics.median(result["timed_wall_seconds"] for result in runs)
            report["results"].append({"mode": mode, "threads": threads, "command": command,
                                      "median_seconds": median, "runs": runs})
            print(f"{mode}, {threads} threads: {median:.6f} seconds, {args.population * args.ticks / median:.0f} car-ticks/s", flush=True)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
