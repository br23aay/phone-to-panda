"""Step 7: speed a trajectory up, uniformly or everywhere except grasp and release."""
import numpy as np

CONTROL_HZ = 20


def add_dwell(traj, steps=10):
    """Hold still at the grasp so the simulated gripper has time to close."""
    g = int(np.argmax(traj["grip"] > 0.5))
    out = {}
    for key, x in traj.items():
        out[key] = np.concatenate([x[:g + 1], np.repeat(x[g], steps), x[g + 1:]])
    out["grip"][g:g + steps + 1] = 1.0
    return out


def _warp(traj, speed):
    dt = 1.0 / CONTROL_HZ
    new_t = np.concatenate([[0.0], np.cumsum(dt / speed[:-1])])
    tu = np.arange(0.0, new_t[-1] + 1e-9, dt)
    nearest = np.clip(np.searchsorted(new_t, tu), 0, len(new_t) - 1)
    return {
        "s": np.interp(tu, new_t, traj["s"]),
        "h": np.interp(tu, new_t, traj["h"]),
        "grip": traj["grip"][nearest],
    }


def uniform(traj, k):
    """Naive speedup: everything k times faster, grasp included."""
    return _warp(traj, np.full(len(traj["s"]), float(k)))


def nonuniform(traj, k, pad_s=0.5):
    """Data-side speedup: k times faster, but grasp and release stay at 1x."""
    grip = traj["grip"] > 0.5
    change = np.flatnonzero(np.diff(grip.astype(int)) != 0) + 1
    speed = np.full(len(grip), float(k))
    pad = int(pad_s * CONTROL_HZ)
    for c in change:
        speed[max(0, c - pad):c + pad] = 1.0
    return _warp(traj, speed)
