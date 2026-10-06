"""Step 9: render replay videos, and a side-by-side of the phone clip and the robot.

Usage: python src/make_videos.py clip-13 [path/to/phone_clip.mov]
Writes mp4 files to results/videos/.
"""
import os
import sys

import cv2
import imageio
import numpy as np

sys.path.insert(0, "src")
from replay_libero import make_env, replay
from retarget import load_clip, retarget
from retime import add_dwell, nonuniform, uniform

OBJECT = "alphabet_soup_1_pos"
OUT = "results/videos"


def render(env, init, traj, path):
    ok, _, frames = replay(env, init, traj, OBJECT, record=True)
    video = [np.ascontiguousarray(o["agentview_image"][::-1, ::-1]) for o, _ in frames]
    imageio.mimsave(path, video, fps=20)
    print(f"{path}: {'success' if ok else 'FAILED'}, {len(video)} frames", flush=True)
    return video, ok


def read_video(path, height):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        w = int(f.shape[1] * height / f.shape[0]) // 2 * 2
        frames.append(cv2.cvtColor(cv2.resize(f, (w, height)), cv2.COLOR_BGR2RGB))
    return frames, fps


def side_by_side(phone_path, robot, traj, t0, path, height=360):
    """Phone clip on the left, robot on the right, kept in step with each other.

    The robot's extra frames (flying in at the start, the 10-step pause at the
    grasp, settling at the end) have no counterpart in the phone clip, so the
    phone picture is held still during them.
    """
    phone, fps = read_video(phone_path, height)
    robot = [cv2.resize(f, (height, height), interpolation=cv2.INTER_NEAREST) for f in robot]
    path_len = len(traj["s"])
    lead_in = max(len(robot) - path_len - 30, 0)
    grasp = int(np.argmax(traj["grip"] > 0.5))
    out = []
    for i, right in enumerate(robot):
        j = min(max(i - lead_in, 0), path_len - 1)          # step along the human path
        j = j if j <= grasp else max(grasp, j - 10)          # skip the added pause
        left = phone[min(int((t0 + j / 20) * fps), len(phone) - 1)]
        out.append(np.hstack([left, right]))
    imageio.mimsave(path, out, fps=20)
    imageio.mimsave(path.replace(".mp4", ".gif"), [f[::2, ::2] for f in out[::3]], duration=0.15, loop=0)
    print(f"{path}: {len(out)} frames", flush=True)


def main():
    clip = sys.argv[1]
    phone = sys.argv[2] if len(sys.argv) > 2 else None
    os.makedirs(OUT, exist_ok=True)
    env, _, inits = make_env(task_id=0)
    traj = add_dwell(retarget(f"data/keypoints/{clip}.npz"))
    robot, _ = render(env, inits[10], traj, f"{OUT}/{clip}_1x.mp4")
    for speed in (1.5, 2.0):
        render(env, inits[10], uniform(traj, speed), f"{OUT}/{clip}_uniform_{speed}x.mp4")
        render(env, inits[10], nonuniform(traj, speed), f"{OUT}/{clip}_nonuniform_{speed}x.mp4")
    if phone and os.path.exists(phone):
        t0 = load_clip(f"data/keypoints/{clip}.npz")[0][0]
        side_by_side(phone, robot, traj, t0, f"{OUT}/{clip}_side_by_side.mp4")


if __name__ == "__main__":
    main()
