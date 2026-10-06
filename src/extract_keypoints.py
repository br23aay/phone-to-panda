"""Step 3: extract wrist, thumb-tip and index-tip pixels from each phone clip.

Usage: python src/extract_keypoints.py --clips data/raw --out data/keypoints
Needs hand_landmarker.task in the repo root (download command is in the guide).
"""
import argparse
import glob
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

WRIST, THUMB_TIP, INDEX_TIP = 0, 4, 8


def process(video_path, model_path, min_conf=0.5):
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    rows = []
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int(1000 * i / fps))
            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                row = [i / fps]
                for k in (WRIST, THUMB_TIP, INDEX_TIP):
                    row += [lm[k].x * width, lm[k].y * height]
            else:
                row = [i / fps] + [np.nan] * 6
            rows.append(row)
            i += 1
    cap.release()
    return np.array(rows, dtype=np.float64), fps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/raw")
    ap.add_argument("--out", default="data/keypoints")
    ap.add_argument("--model", default="hand_landmarker.task")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    paths = []
    for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV"):
        paths += glob.glob(os.path.join(args.clips, ext))
    for path in sorted(set(paths)):
        rows, fps = process(path, args.model)
        if len(rows) == 0:
            print(f"SKIP {path}: no frames read")
            continue
        missing = float(np.isnan(rows[:, 1]).mean())
        name = os.path.splitext(os.path.basename(path))[0]
        np.savez(os.path.join(args.out, name + ".npz"), rows=rows, fps=fps)
        print(f"{name}: {len(rows)} frames, {missing:.0%} without a hand")
