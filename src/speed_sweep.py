"""Step 7, part A: replay every clip at several speeds and record what succeeds.

One line is appended to results/replay_speed_raw.csv per replay, so the run can
be stopped and restarted without losing anything.

Usage: python src/speed_sweep.py                 (layouts 10 to 14)
       python src/speed_sweep.py 10 20           (layouts 10 to 19)
"""
import csv
import glob
import os
import sys
import time

sys.path.insert(0, "src")
from replay_libero import make_env, replay
from retarget import retarget
from retime import add_dwell, nonuniform, uniform

OBJECT = "alphabet_soup_1_pos"
SPEEDS = [float(x) for x in os.environ.get("SPEEDS", "1.0,1.5,2.0,2.5,3.0").split(",")]
RAW = "results/replay_speed_raw.csv"


def main():
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    last = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    os.makedirs("results", exist_ok=True)
    done = set()
    if os.path.exists(RAW):
        done = {(r["clip"], r["layout"], r["method"], r["speedup"]) for r in csv.DictReader(open(RAW))}
    else:
        with open(RAW, "w", newline="") as f:
            csv.writer(f).writerow(["clip", "layout", "method", "speedup", "success", "steps"])

    trajs = {}
    for path in sorted(glob.glob("data/keypoints/*.npz")):
        traj = retarget(path)
        if traj is not None:
            trajs[os.path.splitext(os.path.basename(path))[0]] = add_dwell(traj)

    env, _, inits = make_env(task_id=0, image_size=64)   # small images: nothing is recorded here
    start, n = time.time(), 0
    for layout in range(first, last):                   # layout is the outer loop, so a
        for speed in SPEEDS:                            # partial run is still balanced
            methods = [("normal", None)] if speed == 1.0 else [("uniform", uniform), ("nonuniform", nonuniform)]
            for method, fn in methods:
                for clip, traj in trajs.items():
                    key = (clip, str(layout), method, str(speed))
                    if key in done:
                        continue
                    fast = traj if fn is None else fn(traj, speed)
                    ok, steps, _ = replay(env, inits[layout], fast, OBJECT)
                    with open(RAW, "a", newline="") as f:
                        csv.writer(f).writerow([clip, layout, method, speed, int(ok), steps])
                    n += 1
                    print(f"layout {layout} {method} {speed}x {clip}: {'ok' if ok else 'FAIL'} "
                          f"| {n} replays in {time.time() - start:.0f} s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
