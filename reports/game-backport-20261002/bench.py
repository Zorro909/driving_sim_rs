"""A/B benchmark of the backported optimizations (tools/AltdOpt) in the running game.

The hooked game reads altd-opt.json next to its DLL (forward, path_memo, verify) and reports
counters in altd-opt-status.json. Each case reuses measure() from the earlier benchmark
(game-benchmark-20261002/sweep.py) and runs the configurations interleaved, so drift over
time affects them equally. Results go to results.jsonl in this folder.

Usage: python3 bench.py <group> [<group> ...]   (groups: see GROUPS below)
"""
import json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "game-benchmark-20261002"))
import sweep
from mcp import call
from sweep import ALL_SENSORS, DEFAULT_LAYERS, edit, reset_network

sweep.OUT = os.path.join(HERE, "results.jsonl")
GAME = os.path.expanduser("~/.var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/"
                          "AI Learns To Race/data_AILearnsToDrive_windows_x86_64")
CONFIGS = {
    "off": {"forward": False, "path_memo": False},
    "forward": {"forward": True, "path_memo": False},
    "path_memo": {"forward": False, "path_memo": True},
    "both": {"forward": True, "path_memo": True},
}
TINY = dict(vision_angles=[0], sensors=["speed"], hidden_layers=[])


def status():
    with open(os.path.join(GAME, "altd-opt-status.json")) as f:
        return json.load(f)


def set_config(forward, path_memo, verify=False):
    with open(os.path.join(GAME, "altd-opt.json"), "w") as f:
        json.dump({"forward": forward, "path_memo": path_memo, "verify": verify}, f)
    deadline = time.time() + 5
    while time.time() < deadline:
        s = status()
        if (s["forward_on"], s["path_on"], s["verify"]) == (forward, path_memo, verify):
            return
        time.sleep(0.2)
    raise RuntimeError(f"altd-opt.json not applied: {status()}")


COUNTERS = ["forward_fast", "forward_fallback", "forward_verified", "forward_mismatches", "offset_calls",
            "offset_hits", "sample_calls", "sample_hits", "path_verified", "path_mismatches"]


def run(label, config, population, settings=None, verify=False, **kw):
    set_config(**CONFIGS[config], verify=verify)
    before = status()
    row = sweep.measure(f"{label}/{config}", population, settings, **kw)
    time.sleep(1.2)  # the status file is written once per second
    after = status()
    counters = {k: after[k] - before[k] for k in COUNTERS}
    counters["first_mismatch"] = after["first_mismatch"]
    # measure() already appended its row; append the opt counters as a companion row.
    with open(sweep.OUT, "a") as f:
        f.write(json.dumps({"label": row["label"], "config": config, "verify": verify, "opt": counters}) + "\n")
    print(f"    opt {counters}", flush=True)
    return row


def ab(label, configs, population, repeats, settings=None, **kw):
    for _ in range(repeats):
        for c in configs:
            run(label, c, population, settings, **kw)


def g_verify():
    # Exactness in the game: optimized results are compared bitwise with the original and the
    # original is returned. Mutated cars spread out, so the networks and track positions differ.
    reset_network()
    run("verify_trained_mutated", "both", 400, {"mutation_rate": 0.1}, verify=True, warmup=8.0, seconds=20.0)
    run("verify_trained_turnover", "both", 100, {"mutation_rate": 0.1, "unlimited_time": False,
        "time_limit_s": 5}, verify=True, warmup=6.0, seconds=20.0)


def g_trained():
    reset_network()
    ab("trained_pop400", ["off", "forward", "path_memo", "both"], 400, 3)


def g_trained_mutated():
    reset_network()
    ab("trained_mutated_pop400", ["off", "both"], 400, 2, {"mutation_rate": 0.1}, warmup=8.0)


def g_mutated():
    # With mutation 0 every car is identical and on the same spot, so the path cache also hits
    # across cars (97% instead of 50%/33%). Mutated cars spread out and give the realistic hit rate.
    reset_network()
    ab("trained_mutated_pop400", ["off", "forward", "path_memo", "both"], 400, 2, {"mutation_rate": 0.1}, warmup=8.0)
    ab("trained_mutated_pop1600", ["off", "both"], 1600, 2, {"mutation_rate": 0.1}, warmup=8.0)
    reset_network(**TINY)
    edit(sensors=["speed", "correct_direction", "track_curvature"], init="zeros")
    ab("tiny_plus_path_sensors_mutated", ["off", "path_memo"], 400, 2, {"mutation_rate": 0.1}, warmup=8.0)


def g_populations():
    reset_network()
    for p in [100, 1600]:
        ab(f"trained_pop{p}", ["off", "both"], p, 2)


def g_layers():
    # Same as tiny_plus_8_layers in the earlier benchmark: 8 zero-weight hidden layers.
    reset_network(**TINY)
    edit(hidden_layers=DEFAULT_LAYERS, init="zeros")
    ab("tiny_plus_8_layers", ["off", "forward"], 400, 3)


def g_path():
    # Tiny network plus both path sensors at zero weight: the cars stand still, only the queries cost.
    reset_network(**TINY)
    edit(sensors=["speed", "correct_direction", "track_curvature"], init="zeros")
    ab("tiny_plus_path_sensors", ["off", "path_memo"], 400, 3)


def g_tiny():
    # Hook overhead where nothing is saved: no hidden layers, no path sensors.
    reset_network(**TINY)
    ab("tiny_speed_only", ["off", "both"], 400, 2)


GROUPS = {k[2:]: v for k, v in globals().items() if k.startswith("g_")}

if __name__ == "__main__":
    for g in sys.argv[1:]:
        print(f"== {g}", flush=True)
        GROUPS[g]()
