"""Turn results/replay_speed_raw.csv into the table and the file plot_results.py reads."""
import csv
from collections import defaultdict

rows = list(csv.DictReader(open("results/replay_speed_raw.csv")))
groups = defaultdict(list)
for r in rows:
    groups[(r["method"], float(r["speedup"]))].append(r)

out = []
for (method, speed), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    rate = sum(int(r["success"]) for r in rs) / len(rs)
    secs = sum(int(r["steps"]) for r in rs) / len(rs) / 20
    names = ["uniform", "nonuniform"] if method == "normal" else [method]
    for name in names:                      # normal speed is the 1x point of both curves
        out.append([name, speed, round(rate, 3), round(secs, 2), round(60 / secs * rate, 1), len(rs)])
    print(f"{method:10s} {speed:>4}x  success {rate:5.0%}  path {secs:5.2f} s  "
          f"{60 / secs * rate:5.1f} successful picks/min  (n={len(rs)})")

with open("results/replay_speed.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "speedup", "success_rate", "seconds_per_attempt", "successful_picks_per_minute", "n"])
    w.writerows(out)
