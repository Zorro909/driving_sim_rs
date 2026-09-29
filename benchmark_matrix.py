"""Run the Rust population benchmarks recorded in docs/benchmarks/README.md.

Each configuration runs `--repeats` times; the report keeps every run and the
run with the median timed wall time.

Run: python driving_sim_rs/benchmark_matrix.py --report docs/benchmarks/rust_population_1000.json
"""

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "driving_sim_rs" / "target" / "release" / "altd-sim"
TRACES = ROOT / "docs" / "traces"


def run(args: list[str]) -> dict:
    output = subprocess.run([str(BINARY), *args], check=True, capture_output=True, text=True).stdout
    return json.loads(output)


def bench_cases(threads: list[int]) -> list[tuple[str, list[str]]]:
    cases = [
        ("python_window_early", ["--population", "1000", "--ticks", "12", "--warmup-ticks", "2"]),
        ("python_window_contact", ["--population", "1000", "--ticks", "6", "--warmup-ticks", "0",
                                   "--spawn-index", "1231"]),
    ]
    for mode in ("independent", "lockstep"):
        for count in threads:
            cases.append((f"a07_30s_{mode}_{count}t",
                          ["--mode", mode, "--threads", str(count), "--population", "1000",
                           "--ticks", "1800", "--warmup-ticks", "2"]))
    cases.append(("a07_contact_10s_independent",
                  ["--population", "1000", "--ticks", "600", "--warmup-ticks", "0",
                   "--spawn-index", "1231"]))
    cases.append(("a07_30s_independent_10000_cars",
                  ["--population", "10000", "--ticks", "1800", "--warmup-ticks", "2"]))
    return cases


def train_cases() -> list[tuple[str, list[str]]]:
    common = ["--scene", str(ROOT / "driving_sim_rs/scenes_exact/rally_a07_contact_scene.json"),
              "--network", str(TRACES / "rally_a01_live_network.json"),
              "--model", str(ROOT / "driving_sim_rs/rally_trained_model_exact.json"),
              "--spawn-trace", str(TRACES / "rally_a07_contact_reference.json"),
              "--population", "1000", "--generations", "5", "--ticks", "1800", "--seed", "7"]
    return [("train_a07_5gen_30s_no_elimination",
             common + ["--no-eliminate-on-wall", "--no-idle-eliminate"]),
            ("train_a07_5gen_30s_elimination", common + ["--eliminate-on-wall", "--idle-eliminate"])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, nargs="+",
                        default=sorted({1, 2, 4, 8, 16, os.cpu_count()}))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for name, case_args in bench_cases(args.threads):
        runs = [run(["bench", *case_args]) for _ in range(args.repeats)]
        median = statistics.median(r["timed_wall_seconds"] for r in runs)
        chosen = min(runs, key=lambda r: abs(r["timed_wall_seconds"] - median))
        results[name] = {"args": case_args, "median_run": chosen, "timed_wall_seconds_all": [r["timed_wall_seconds"] for r in runs]}
        print(f"{name:40s} {chosen['realtime_multiplier']:10.2f}x  "
              f"{chosen['car_ticks_per_wall_second']:14.0f} car ticks/s  "
              f"contacts {chosen['contacts_after_ticks']}", flush=True)
    for name, case_args in train_cases():
        runs = []
        for _ in range(args.repeats):
            with_output = case_args + ["--output", "/dev/null"]
            runs.append(run(["train", *with_output]))
        median = statistics.median(r["timing"]["simulation_wall_seconds"] for r in runs)
        chosen = min(runs, key=lambda r: abs(r["timing"]["simulation_wall_seconds"] - median))
        results[name] = {"args": case_args, "median_run": chosen}
        timing = chosen["timing"]
        print(f"{name:40s} {timing['realtime_multiplier']:10.2f}x simulation, "
              f"{timing['complete_realtime_multiplier']:.2f}x including turnover", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
