"""Step 5: replay a retargeted trajectory on the Panda in LIBERO.

The arm follows the (s, h, grip) trajectory with a proportional controller.
Object and basket positions are read from the simulator, so one human clip can
be replayed on any starting layout.
"""
import os

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from retarget import to_sim

MAX_DELTA = 0.05        # metres moved per unit of action (LIBERO's controller limit)
GAIN = 3.0              # proportional gain; the arm is compliant and lags without it
APPROACH_HEIGHT = 0.15  # fly in this far above the object before following the path
PLACE_HEIGHT = 0.20     # release this far above the basket origin, in metres
SETTLE_STEPS = 30       # steps to wait after the release before checking success


def make_env(suite_name="libero_object", task_id=0, image_size=256):
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=image_size, camera_widths=image_size)
    env.seed(0)
    return env, task.language, suite.get_task_init_states(task_id)


def step_towards(env, obs, target, grip, frames=None):
    action = np.zeros(7)
    action[:3] = np.clip(GAIN * (target - obs["robot0_eef_pos"]) / MAX_DELTA, -1.0, 1.0)
    action[6] = 1.0 if grip > 0.5 else -1.0   # +1 closes the gripper, -1 opens it
    if frames is not None:
        frames.append((obs, action))           # the observation the action was chosen from
    return env.step(action)[0]


def replay(env, init_state, traj, object_key, basket_key="basket_1_pos", record=False):
    """Returns (success, steps on the human path, recorded frames)."""
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(10):                        # let objects settle
        obs = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))[0]
    pick = obs[object_key].copy()
    place = obs[basket_key].copy()
    place[2] += PLACE_HEIGHT
    targets = to_sim(traj, pick, place)
    frames = [] if record else None

    above = targets[0].copy()
    above[2] = max(above[2], pick[2] + APPROACH_HEIGHT)
    for waypoint in (above, targets[0]):       # fly in from above, then to the path start
        for _ in range(80):
            if np.linalg.norm(waypoint - obs["robot0_eef_pos"]) < 0.01:
                break
            obs = step_towards(env, obs, waypoint, 0.0, frames)

    for target, grip in zip(targets, traj["grip"]):
        obs = step_towards(env, obs, target, grip, frames)
    for _ in range(SETTLE_STEPS):
        obs = step_towards(env, obs, targets[-1], 0.0, frames)
    return bool(env.check_success()), len(targets), frames
