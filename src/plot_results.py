import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/figures", exist_ok=True)
rows = list(csv.DictReader(open("results/replay_speed.csv")))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
for method, style in [("uniform", "o--"), ("nonuniform", "s-")]:
    sel = [r for r in rows if r["method"] == method]
    k = [float(r["speedup"]) for r in sel]
    ax[0].plot(k, [float(r["success_rate"]) for r in sel], style, label=method)
    ax[1].plot(k, [float(r["successful_picks_per_minute"]) for r in sel], style, label=method)
ax[0].set_xlabel("speedup"); ax[0].set_ylabel("success rate"); ax[0].set_ylim(0, 1.05)
ax[1].set_xlabel("speedup"); ax[1].set_ylabel("successful picks per minute")
ax[0].legend()
fig.tight_layout()
fig.savefig("results/figures/speed.png", dpi=200)
print("saved results/figures/speed.png")
