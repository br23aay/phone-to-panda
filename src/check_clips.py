"""Run after extract_keypoints.py: says which clips are usable and plots one."""
import glob
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from retarget import retarget

good = []
for path in sorted(glob.glob("data/keypoints/*.npz")):
    traj = retarget(path)
    if traj is None:
        print(f"REJECTED {path}")
        continue
    good.append(path)
    held = int(traj["grip"].sum()) / 20
    lift = traj["h"][traj["grip"] > 0.5].max() * 100
    print(f"ok       {path}: {len(traj['s']) / 20:.1f} s, holding for {held:.1f} s, lift {lift:.0f} cm")
print(f"{len(good)} usable clips")

if good:
    traj = retarget(good[0])
    fig, ax = plt.subplots(3, 1, sharex=True)
    for a, key in zip(ax, ["s", "h", "grip"]):
        a.plot(traj[key])
        a.set_ylabel(key)
    ax[0].set_title(good[0])
    plt.show()
