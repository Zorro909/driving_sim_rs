"""Summarizes results.jsonl: mean car ticks/s, .NET pool CPU per car tick and path cache hit rates per case and config."""
import json, os, statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
runs = defaultdict(list)
hits = defaultdict(lambda: [0, 0, 0, 0])  # offset calls, hits, sample calls, hits
for line in open(os.path.join(HERE, "results.jsonl")):
    r = json.loads(line)
    if "opt" in r:
        h = hits[tuple(r["label"].split("/"))]
        for i, k in enumerate(["offset_calls", "offset_hits", "sample_calls", "sample_hits"]):
            h[i] += r["opt"][k]
    elif "car_ticks_per_s" in r and not r["label"].startswith("verify"):
        case, config = r["label"].split("/")
        runs[case, config].append(r)

print("| Case | Config | Runs | Car ticks/s (each) | Mean | vs off | Pool CPU µs per car tick | Total cores | Path cache hits (offset / sample) |")
print("|---|---|---:|---|---:|---:|---:|---:|---|")
for case in dict.fromkeys(c for c, _ in runs):
    base = statistics.mean(r["car_ticks_per_s"] for r in runs[case, "off"])
    for (c, config), rs in runs.items():
        if c != case:
            continue
        ticks = [r["car_ticks_per_s"] for r in rs]
        mean = statistics.mean(ticks)
        pool = statistics.mean(r["threads"]["dotnet_pool"] * 1e6 / r["car_ticks_per_s"] for r in rs)
        cores = statistics.mean(r["threads"]["total_cores"] for r in rs)
        oc, oh, sc, sh = hits[case, config]
        rate = f"{oh / oc:.0%} / {sh / sc:.0%}" if oc else ""
        each = ", ".join(f"{t / 1000:.1f}k" for t in ticks)
        print(f"| {case} | {config} | {len(rs)} | {each} | {mean:,.0f} | {mean / base - 1:+.1%} | {pool:.1f} | {cores:.2f} | {rate} |")
