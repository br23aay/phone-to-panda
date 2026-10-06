"""Step 5: replay every clip on several starting layouts and save the successes.

Each successful replay is saved as one .npz file, so nothing is lost if the
session stops. Run it again and it carries on where it left off.

Usage: python src/build_demos.py            (normal speed, layouts 0 to 4)
       python src/build_demos.py 2.0        (non-uniform 2x speedup)
"""
import glob
import os
import sys
import time

import numpy as np
import robosuite.utils.transform_utils as T

sys.path.insert(0, "src")
from replay_libero import make_env, replay
from retarget import retarget
from retime import add_dwell, nonuniform

OBJECT = "alphabet_soup_1_pos"
LAYOUTS = range(int(os.environ.get("N_LAYOUTS", 5)))


def pack(frames, language):
    obs = [o for o, _ in frames]
    state = [np.concatenate([o["robot0_eef_pos"], T.quat2axisangle(o["robot0_eef_quat"]),
                             o["robot0_gripper_qpos"]]) for o in obs]
    return {
        # rotated 180 degrees, the same way LeRobot's LIBERO wrapper shows images to a policy
        "image": np.stack([o["agentview_image"][::-1, ::-1] for o in obs]),
        "image2": np.stack([o["robot0_eye_in_hand_image"][::-1, ::-1] for o in obs]),
        "state": np.asarray(state, dtype=np.float32),
        "action": np.asarray([a for _, a in frames], dtype=np.float32),
        "language": language,
    }


def main():
    speed = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    out = "data/demos_1x_raw" if speed == 1.0 else f"data/demos_fast{speed:g}x_raw"
    os.makedirs(out, exist_ok=True)
    env, language, inits = make_env(task_id=0)
    kept = tried = 0
    start = time.time()
    for path in sorted(glob.glob("data/keypoints/*.npz")):
        name = os.path.splitext(os.path.basename(path))[0]
        traj = retarget(path)
        if traj is None:
            print(f"{name}: rejected by retarget", flush=True)
            continue
        traj = add_dwell(traj)
        if speed != 1.0:
            traj = nonuniform(traj, speed)
        for i in LAYOUTS:
            target = f"{out}/{name}_layout{i}.npz"
            tried += 1
            if os.path.exists(target):
                kept += 1
                continue
            ok, _, frames = replay(env, inits[i], traj, OBJECT, record=True)
            if ok:
                np.savez_compressed(target, **pack(frames, language))
                kept += 1
            print(f"{name} layout {i}: {'saved' if ok else 'FAILED'}, {len(frames)} frames "
                  f"| kept {kept} of {tried} | {time.time() - start:.0f} s", flush=True)
    print(f"DONE: kept {kept} of {tried} replays in {out}", flush=True)


if __name__ == "__main__":
    main()
