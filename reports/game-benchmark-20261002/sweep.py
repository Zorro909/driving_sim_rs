"""One-factor-at-a-time throughput sweep of the running game through its MCP server.

Each case starts a fresh training session of the benchmark network at 16x, waits,
then measures simulated time over running wall time (from the mod's ThroughputMeter)
and the CPU time of every game thread (from /proc). Results go to results.jsonl.

Usage: python3 sweep.py <group> [<group> ...]   (groups: see GROUPS below)
"""
import json, os, subprocess, sys, time
from mcp import call

NET = "zz-benchmark-20261002"
SOURCE_NET = "20260930-230014-generalized-model-rally-long-edaf30-latest"
TRACK = {"name": "A07 Three Terrains", "source": "campaign_a"}
BASE = {"mutation_rate": 0, "unlimited_time": True, "eliminate": False,
        "idle_eliminate": False, "auto_save": False}
TICK = os.sysconf("SC_CLK_TCK")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl")

ALL_RAYS = [0, -10, 10, -20, 20, -35, 35, -45, 45, -60, 60, -90, 90]
ALL_SENSORS = ["boost_capacity", "correct_direction", "grip", "velocity_front",
               "velocity_side", "track_curvature", "wheel_angle"]
NO_PATH = [s for s in ALL_SENSORS if s not in ("correct_direction", "track_curvature")]
DEFAULT_LAYERS = [16, 16, 16, 16, 12, 12, 12, 8]


def game_pid():
    out = subprocess.run(["pgrep", "-f", "AI Learns To Drive.exe"], capture_output=True, text=True).stdout.split()
    pids = [int(p) for p in out if open(f"/proc/{p}/comm").read().strip().startswith("AI Learns")]
    return pids[0]


def threads(pid):
    """tid -> (name, cpu s, kernel cpu s, voluntary switches, involuntary switches)."""
    res = {}
    for tid in os.listdir(f"/proc/{pid}/task"):
        try:
            base = f"/proc/{pid}/task/{tid}"
            name = open(f"{base}/comm").read().strip()
            fields = open(f"{base}/stat").read().rsplit(")", 1)[1].split()
            status = dict(l.split(":", 1) for l in open(f"{base}/status").read().splitlines() if ":" in l)
            res[int(tid)] = (name, (int(fields[11]) + int(fields[12])) / TICK, int(fields[12]) / TICK,
                             int(status["voluntary_ctxt_switches"]), int(status["nonvoluntary_ctxt_switches"]))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return res


def thread_deltas(pid, a, b, wall):
    """Busy fraction (cores) per thread over the interval, grouped; kernel share and switches per second."""
    per = []
    for tid, (name, *vals) in b.items():
        prev = a.get(tid, (name, 0.0, 0.0, 0, 0))[1:]
        d = [(v - p) / wall for v, p in zip(vals, prev)]
        if d[0] > 0:
            per.append((tid, name, *d))
    per.sort(key=lambda x: -x[2])
    groups = {k: [0.0, 0.0, 0.0, 0.0, 0] for k in ("main", "dotnet_pool", "other")}
    for tid, name, *d in per:
        g = groups["main" if tid == pid else "dotnet_pool" if name.startswith(".NET T") else "other"]
        for i in range(4):
            g[i] += d[i]
        g[4] += 1
    out = {"total_cores": round(sum(p[2] for p in per), 3)}
    for k, (cpu, sys_, vcs, nvcs, n) in groups.items():
        out[k] = round(cpu, 3)
        out[k + "_detail"] = {"kernel_cores": round(sys_, 3), "vol_switches_per_s": round(vcs),
                              "invol_switches_per_s": round(nvcs), "threads": n}
    out["top"] = [(tid, name, round(cpu, 3)) for tid, name, cpu, *_ in per[:8]]
    return out


def throughput():
    s = call("get_training_status")["training"]
    t = s["speed"]["throughput"]
    cur = s["networks"][0]["current"]
    return t, cur, s


def stop():
    if call("get_training_status").get("training"):
        call("stop_training")


def edit(**changes):
    stop()
    call("update_network", {"name": NET, **changes})


def reset_network(**changes):
    stop()
    call("duplicate_network", {"name": SOURCE_NET, "source": "user", "new_name": NET, "overwrite": True})
    if changes:
        call("update_network", {"name": NET, **changes})


def measure(label, population, settings=None, warmup=5.0, seconds=15.0, multiplier=16, track=TRACK):
    pid = game_pid()
    t0 = time.time()
    call("start_training", {"networks": [NET], "track": track,
                            "settings": {**BASE, "population": population, **(settings or {})},
                            "multiplier": multiplier, "replace_active": True})
    start_s = time.time() - t0
    time.sleep(warmup)
    a_t, _, _ = throughput()
    a_th, a_wall = threads(pid), time.monotonic()
    time.sleep(seconds)
    b_t, cur, status = throughput()
    b_th, b_wall = threads(pid), time.monotonic()
    wall = b_wall - a_wall
    run = b_t["running_wall_time_s"] - a_t["running_wall_time_s"]
    over = b_t["overhead_wall_time_s"] - a_t["overhead_wall_time_s"]
    sim = b_t["simulated_time_s"] - a_t["simulated_time_s"]
    gens = b_t["completed_generations"] - a_t["completed_generations"]
    net = call("get_network", {"name": NET, "source": "user"})["summary"]
    row = {
        "label": label, "population": population, "settings": settings or {},
        "shape": net["shape"], "rays": len(net["vision_angles"]), "sensors": net["sensors"],
        "start_call_s": round(start_s, 2), "wall_s": round(wall, 2),
        "running_multiplier": round(sim / run, 3) if run > 0 else None,
        "effective_multiplier": round(sim / (run + over), 3) if run + over > 0 else None,
        "overhead_s": round(over, 3), "generations": gens,
        "car_ticks_per_s": round(sim * 60 * population / (run + over)) if run + over > 0 else None,
        "active": cur["active"], "best_score": cur["best_score"],
        "best_average_speed": cur["best_average_speed"],
        "threads": thread_deltas(pid, a_th, b_th, wall),
    }
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    th = row["threads"]
    print(f"{label:28s} pop={population:5d} x{row['effective_multiplier']} run_x{row['running_multiplier']} "
          f"ticks/s={row['car_ticks_per_s']} gens={gens} over={over:.2f}s cores={th['total_cores']} "
          f"main={th['main']} pool={th['dotnet_pool']} other={th['other']}", flush=True)
    return row


def g_population():
    reset_network()
    for p in [1, 25, 50, 100, 200, 400, 800, 1600]:
        measure(f"pop{p}", p)


def g_population_small_net():
    # Smallest network: 1 ray, 0 hidden layers, speed only. Behaviour changes, cost floor.
    reset_network(vision_angles=[0], sensors=["speed"], hidden_layers=[])
    for p in [100, 400, 1600]:
        measure(f"tiny_pop{p}", p)


def g_rays():
    # Keep behaviour fixed: trained weights for ray 0 only, every added ray starts at zero weight.
    reset_network(vision_angles=[0])
    for k in [1, 3, 5, 7, 9, 13]:
        edit(vision_angles=ALL_RAYS[:k], init="zeros")
        measure(f"rays{k}", 400)


def g_layers():
    # Hidden layers change behaviour; all cars still drive the same (mutation 0).
    for layers in [[], [16], [16, 16], [16] * 4, DEFAULT_LAYERS, [16] * 10]:
        reset_network(hidden_layers=layers)
        measure(f"layers{len(layers)}", 400)


def g_path_sensors():
    # Without path sensors, then with them added at zero weight: same driving, extra queries.
    reset_network(sensors=NO_PATH)
    measure("no_path_sensors", 400)
    edit(sensors=ALL_SENSORS, init="zeros")
    measure("path_sensors_zero_w", 400)


def g_turnover():
    # Short generations: the meter's overhead time is the turnover pause.
    reset_network()
    for p in [100, 400, 1600]:
        measure(f"turnover_pop{p}", p, settings={"unlimited_time": False, "time_limit_s": 5},
                warmup=6.0, seconds=30.0)


def g_random_tracks():
    reset_network()
    for p in [100, 400]:
        measure(f"random_turnover_pop{p}", p, settings={"unlimited_time": False, "time_limit_s": 5},
                warmup=6.0, seconds=30.0, track={"random": {"length": 20}})


def g_multiplier():
    # Does the requested speed matter once the machine is the limit?
    reset_network()
    for m in [4, 8, 16]:
        measure(f"pop400_x{m}", 400, multiplier=m)


def g_spread():
    # mutation 0 stacks identical cars on one trajectory; mutated cars spread out like real training.
    reset_network()
    for p in [400, 1600]:
        measure(f"mutated_pop{p}", p, settings={"mutation_rate": 0.1}, warmup=8.0)



def g_sensors():
    # Start from the tiny network and add one sensor at a time with zero weights (same behaviour).
    reset_network(vision_angles=[0], sensors=["speed"], hidden_layers=[])
    measure("tiny_speed_only", 400)
    for s in ["grip", "boost_capacity", "velocity_front", "velocity_side", "wheel_angle",
              "correct_direction", "track_curvature"]:
        reset_network(vision_angles=[0], sensors=["speed"], hidden_layers=[])
        edit(sensors=["speed", s], init="zeros")
        measure(f"tiny_plus_{s}", 400)
    reset_network(vision_angles=[0], sensors=["speed"], hidden_layers=[])
    edit(vision_angles=ALL_RAYS, init="zeros")
    measure("tiny_plus_12_rays", 400)
    reset_network(vision_angles=[0], sensors=["speed"], hidden_layers=[])
    edit(hidden_layers=DEFAULT_LAYERS, init="zeros")
    measure("tiny_plus_8_layers", 400)


GROUPS = {k[2:]: v for k, v in globals().items() if k.startswith("g_")}

if __name__ == "__main__":
    for g in sys.argv[1:]:
        print(f"== {g}", flush=True)
        GROUPS[g]()
