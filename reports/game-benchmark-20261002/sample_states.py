"""Sample run state and kernel wait channel of the busiest game threads (no profiler needed)."""
import collections, json, sys, time
from sweep import game_pid, threads, measure, reset_network

pid = game_pid()
pop = int(sys.argv[1]) if len(sys.argv) > 1 else 400
reset_network()
from mcp import call
call("start_training", {"networks": ["zz-benchmark-20261002"], "track": {"name": "A07 Three Terrains", "source": "campaign_a"},
                        "settings": {"population": pop, "mutation_rate": 0, "unlimited_time": True, "eliminate": False,
                                     "idle_eliminate": False}, "multiplier": 16, "replace_active": True})
time.sleep(4)
a = threads(pid); time.sleep(2); b = threads(pid)
busy = sorted(b, key=lambda t: -(b[t][1] - a.get(t, b[t])[1]))[:6]
counts = {t: collections.Counter() for t in busy}
end = time.time() + 6
n = 0
while time.time() < end:
    for t in busy:
        try:
            st = open(f"/proc/{pid}/task/{t}/stat").read().rsplit(")", 1)[1].split()[0]
            wc = open(f"/proc/{pid}/task/{t}/wchan").read().strip() if st != "R" else ""
            counts[t][st + (":" + wc if wc else "")] += 1
        except FileNotFoundError:
            pass
    n += 1
    time.sleep(0.0005)
out = {f"{t} {b[t][0]}{' (main)' if t == pid else ''}": {k: round(v / n, 3) for k, v in c.most_common(5)} for t, c in counts.items()}
print(json.dumps({"population": pop, "samples": n, "states": out}, indent=1))
