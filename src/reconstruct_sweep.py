"""Rebuild results/replay_speed_raw.csv for the first speed sweep (4 Oct 2026).

That run finished, but its results file was lost when the Colab session ended.
Every outcome was printed in the notebook log, so the successes are copied here
from that log. Step counts are recomputed from the keypoints, which is exact
because retiming is deterministic. Later sweeps append real rows to the same file.
"""
import csv
import glob
import os
import sys

sys.path.insert(0, "src")
from retarget import retarget
from retime import add_dwell, nonuniform, uniform

ALL = [f"clip-{i}" for i in range(1, 15)]
NORMAL_OK = [c for c in ALL if c != "clip-7"]
# (layout, method, speedup) -> clips that succeeded. Anything not listed failed.
OK = {
    (10, "normal", 1.0): NORMAL_OK, (11, "normal", 1.0): NORMAL_OK, (12, "normal", 1.0): NORMAL_OK,
    (10, "uniform", 1.5): ["clip-6"],
    (11, "uniform", 1.5): ["clip-6", "clip-8"],
    (12, "uniform", 1.5): ["clip-6"],
    (10, "nonuniform", 1.5): ["clip-10", "clip-11", "clip-12", "clip-13", "clip-14", "clip-4", "clip-6", "clip-8"],
    (11, "nonuniform", 1.5): ["clip-11", "clip-12", "clip-13", "clip-14", "clip-4", "clip-6", "clip-8"],
    (12, "nonuniform", 1.5): ["clip-11", "clip-12", "clip-13", "clip-14", "clip-3", "clip-4", "clip-5",
                              "clip-6", "clip-8", "clip-9"],
    (10, "nonuniform", 2.0): ["clip-13", "clip-6"],
    (11, "nonuniform", 2.0): ["clip-13", "clip-6"],
    (12, "nonuniform", 2.0): ["clip-13", "clip-6"],
    (10, "nonuniform", 2.5): ["clip-13"], (11, "nonuniform", 2.5): ["clip-13"], (12, "nonuniform", 2.5): ["clip-13"],
}
FN = {"uniform": uniform, "nonuniform": nonuniform}


def main():
    trajs = {}
    for path in sorted(glob.glob("data/keypoints/*.npz")):
        t = retarget(path)
        if t is not None:
            trajs[os.path.splitext(os.path.basename(path))[0]] = add_dwell(t)
    os.makedirs("results", exist_ok=True)
    rows = []
    for layout in (10, 11, 12):
        for speed in (1.0, 1.5, 2.0, 2.5, 3.0):
            for method in (["normal"] if speed == 1.0 else ["uniform", "nonuniform"]):
                for clip, traj in trajs.items():
                    fast = traj if method == "normal" else FN[method](traj, speed)
                    ok = clip in OK.get((layout, method, speed), [])
                    rows.append([clip, layout, method, speed, int(ok), len(fast["s"])])
    with open("results/replay_speed_raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "layout", "method", "speedup", "success", "steps"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows, {sum(r[4] for r in rows)} successes")


if __name__ == "__main__":
    main()
