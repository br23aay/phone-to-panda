"""Step 6: turn the saved replays into a LeRobot dataset for training.

Usage: python src/convert_demos.py data/demos_1x_raw data/demos_1x
"""
import glob
import shutil
import sys
import os

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

IMG = {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]}
FEATURES = {
    "observation.images.image": IMG,
    "observation.images.image2": IMG,
    "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
    "action": {"dtype": "float32", "shape": (7,), "names": None},
}


def main():
    raw, out = sys.argv[1], sys.argv[2]
    if os.path.exists(out):
        shutil.rmtree(out)
    ds = LeRobotDataset.create(repo_id="local/" + os.path.basename(out), fps=20,
                               features=FEATURES, root=out)
    files = sorted(glob.glob(os.path.join(raw, "*.npz")))
    for n, path in enumerate(files, 1):
        d = np.load(path, allow_pickle=True)
        task = str(d["language"])
        for i in range(len(d["action"])):
            ds.add_frame({
                "observation.images.image": d["image"][i],
                "observation.images.image2": d["image2"][i],
                "observation.state": d["state"][i],
                "action": d["action"][i],
                "task": task,
            })
        ds.save_episode()
        print(f"{n}/{len(files)} {os.path.basename(path)}: {len(d['action'])} frames", flush=True)
    ds.finalize()
    print(f"DONE: {len(files)} episodes written to {out}", flush=True)


if __name__ == "__main__":
    main()
