"""Step 4: turn pixel keypoints into an object-centric end-effector trajectory.

Each clip becomes three signals sampled at the simulator's control rate:
  s    progress along the pick-to-place line (0 at the grasp, 1 at the release)
  h    height above the grasp point, in metres
  grip 1 while the hand is holding the object, else 0
"""
import numpy as np

CONTROL_HZ = 20          # LIBERO control frequency
PICK_TO_PLACE_M = 0.20   # distance between your two tape marks, in metres
MAX_GAP = 3              # frames of lost tracking that are bridged by interpolation


def _fill_and_smooth(x, win=5):
    idx = np.arange(len(x))
    good = ~np.isnan(x)
    x = np.interp(idx, idx[good], x[good])
    pad = win // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(win) / win, mode="valid")


def _longest_run(valid):
    """Longest stretch of detected frames, allowing gaps of up to MAX_GAP frames."""
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return None
    breaks = np.flatnonzero(np.diff(idx) > MAX_GAP + 1) + 1
    runs = np.split(idx, breaks)
    best = max(runs, key=len)
    return best[0], best[-1] + 1


def load_clip(path, min_seconds=2.0):
    """Keep only the longest continuous hand track. This drops stray detections,
    for example on the hand's shadow before the hand enters the frame."""
    rows = np.load(path)["rows"]
    run = _longest_run(~np.isnan(rows[:, 1]))
    if run is None:
        return None
    rows = rows[run[0]:run[1]]
    t = rows[:, 0]
    if t[-1] - t[0] < min_seconds:
        return None
    cols = [_fill_and_smooth(rows[:, c]) for c in range(1, 7)]
    thumb = np.stack([cols[2], cols[3]], axis=1)
    index = np.stack([cols[4], cols[5]], axis=1)
    return t, (thumb + index) / 2.0


def find_grasp_release(pinch):
    """Grasp and release from the motion itself, not from finger spacing.

    Around a rigid box the fingers barely close, so finger spacing is a weak signal.
    The hand's path is a strong one: it goes down to the box, across, and down again.
      release = lowest point of the hand at the far end of its sideways travel
      grasp   = lowest point of the hand in the first half of that travel
    """
    x, y = pinch[:, 0], pinch[:, 1]          # image y points down, so larger y is lower
    far = int(np.argmax(np.abs(x - x[0])))   # frame where the hand is furthest sideways
    span = x[far] - x[0]
    if abs(span) < 50:
        return None
    at_far_end = np.flatnonzero(np.abs(x - x[far]) < 0.1 * abs(span))
    release = int(at_far_end[np.argmax(y[at_far_end])])
    before = np.arange(release)
    first_half = before[np.abs(x[before] - x[release]) > 0.5 * abs(span)]
    if len(first_half) == 0:
        return None
    grasp = int(first_half[np.argmax(y[first_half])])
    if release - grasp < 5:
        return None
    return grasp, release


def retarget(path):
    clip = load_clip(path)
    if clip is None:
        return None
    t, pinch = clip
    found = find_grasp_release(pinch)
    if found is None:
        return None
    grasp, release = found
    dx = pinch[release, 0] - pinch[grasp, 0]
    metres_per_px = PICK_TO_PLACE_M / abs(dx)
    s = (pinch[:, 0] - pinch[grasp, 0]) / dx
    h = (pinch[grasp, 1] - pinch[:, 1]) * metres_per_px
    grip = np.zeros(len(t))
    grip[grasp:release] = 1.0

    tu = np.arange(t[0], t[-1], 1.0 / CONTROL_HZ)
    nearest = np.clip(np.searchsorted(t, tu), 0, len(t) - 1)
    return {
        "s": np.interp(tu, t, s),
        "h": np.interp(tu, t, h),
        "grip": grip[nearest],
    }


def to_sim(traj, pick_pos, place_pos, safe_h=0.06, blend_h=0.04):
    """Map (s, h) onto simulator coordinates. Returns an (N, 3) array of targets.

    Before the grasp the path is reshaped so the gripper comes down vertically:
    a hand can slide in from the side, but the Panda's fingers would hit the object.
    """
    pick_pos, place_pos = np.asarray(pick_pos, float), np.asarray(place_pos, float)
    s, h = traj["s"].copy(), np.maximum(traj["h"], 0.0)
    grasp = int(np.argmax(traj["grip"] > 0.5))
    s[:grasp] *= np.clip((h[:grasp] - safe_h) / blend_h, 0.0, 1.0)
    xy = pick_pos[None, :2] + s[:, None] * (place_pos[:2] - pick_pos[:2])[None, :]
    # reach the release height by the halfway point, so the object clears the basket wall
    base_z = pick_pos[2] + np.clip(s / 0.5, 0.0, 1.0) * (place_pos[2] - pick_pos[2])
    return np.column_stack([xy, base_z + h])
