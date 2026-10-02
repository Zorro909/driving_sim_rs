"""Joint run states of the main thread, thread 2327051-like helper and the .NET pool, sampled together."""
import collections, json, os, sys, time
from sweep import game_pid, threads, reset_network
from mcp import call

pid = game_pid()
pop = int(sys.argv[1])
helper = int(sys.argv[2])
reset_network()
call("start_training", {"networks": ["zz-benchmark-20261002"], "track": {"name": "A07 Three Terrains", "source": "campaign_a"},
                        "settings": {"population": pop, "mutation_rate": 0, "unlimited_time": True, "eliminate": False,
                                     "idle_eliminate": False}, "multiplier": 16, "replace_active": True})
time.sleep(4)
pool = [t for t, v in threads(pid).items() if v[0].startswith(".NET TP")]
def st(t):
    try: return open(f"/proc/{pid}/task/{t}/stat").read().rsplit(")", 1)[1].split()[0] == "R"
    except FileNotFoundError: return False
c = collections.Counter(); n = 0; end = time.time() + 8
while time.time() < end:
    m, h = st(pid), st(helper)
    k = sum(st(t) for t in pool)
    c[(("main" if m else "-"), ("helper" if h else "-"), "pool>=4" if k >= 4 else f"pool{k}")] += 1
    n += 1
print(json.dumps({"population": pop, "pool_threads": len(pool), "samples": n,
                  "joint": {" ".join(k): round(v / n, 3) for k, v in c.most_common(12)}}, indent=1))
