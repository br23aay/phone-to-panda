"""Second unattended Colab job: test the first policy with more frequent re-planning,
then keep training it and evaluate again. Everything is saved to Drive as it goes.

Usage in Colab: %run src/day_job2.py 35000
"""
import os
import shutil
import sys
import threading
import time

from google.colab import drive, runtime

sys.argv = [sys.argv[0], "1x"]            # day_job reads the dataset name from argv
from day_job import D, get_dataset, retry, run

EXTRA_STEPS = os.environ.get("EXTRA_STEPS", "35000")
OUT = "out/act_1x_more"
EVAL_COMMON = ("--env.type=libero --env.task=libero_object --env.task_ids=[0] --env.episode_length=400 "
               "--eval.batch_size=1 --policy.n_action_steps=20")


def to_drive(local, remote):
    if os.path.isdir(local):
        retry(lambda: shutil.copytree(local, remote, dirs_exist_ok=True), f"saving {local}")
        print(f"saved {local} to Drive", flush=True)


def sync_checkpoints():
    while True:
        time.sleep(120)
        try:
            for ckpt in sorted(os.listdir(f"{OUT}/checkpoints")):
                src = f"{OUT}/checkpoints/{ckpt}/pretrained_model"
                dst = f"{D}/act_1x_more/ckpt_{ckpt}"
                if ckpt == "last" or not os.path.isfile(f"{src}/model.safetensors") or os.path.isdir(dst):
                    continue
                time.sleep(30)
                shutil.copytree(src, dst + "_tmp", dirs_exist_ok=True)
                os.rename(dst + "_tmp", dst)
                shutil.copytree("logs", f"{D}/logs_1x_more", dirs_exist_ok=True)
                print(f"checkpoint {ckpt} copied to Drive", flush=True)
        except Exception as e:
            print("checkpoint sync skipped:", str(e)[:120], flush=True)


def evaluate(model, name, episodes):
    run(f"lerobot-eval --policy.path={model} {EVAL_COMMON} --eval.n_episodes={episodes} --output_dir=out/{name}",
        r"pc_success|End of eval|Error", f"logs/{name}.txt")
    to_drive(f"out/{name}", f"{D}/{name}")
    to_drive("logs", f"{D}/logs_1x_more")


def main():
    os.environ["MUJOCO_GL"] = "egl"
    os.system('echo N | python -c "import libero.libero" > /dev/null 2>&1')
    os.makedirs("logs", exist_ok=True)
    os.makedirs(f"{D}/act_1x_more", exist_ok=True)
    get_dataset("data/demos_1x")
    retry(lambda: shutil.copytree(f"{D}/act_1x/pretrained_model", "init_model", dirs_exist_ok=True),
          "copying the first policy")
    print("first policy copied from Drive", flush=True)

    evaluate("init_model", "eval_15k_replan20", 20)

    threading.Thread(target=sync_checkpoints, daemon=True).start()
    run(f"lerobot-train --policy.path=init_model --policy.push_to_hub=false --dataset.repo_id=local/demos_1x "
        f"--dataset.root=data/demos_1x --output_dir={OUT} --steps={EXTRA_STEPS} --batch_size=32 "
        f"--save_freq=5000 --log_freq=500 --wandb.enable=false "
        f"--policy.optimizer_lr=3e-5 --policy.optimizer_lr_backbone=3e-5",
        r"step:|Checkpoint|End of training|Error", "logs/train_more.txt")
    final = os.path.realpath(f"{OUT}/checkpoints/last/pretrained_model")
    to_drive(final, f"{D}/act_1x_more/pretrained_model")

    evaluate(final, "eval_50k_replan20", 50)


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            to_drive("logs", f"{D}/logs_1x_more")
            drive.flush_and_unmount()
        finally:
            print("RELEASING GPU", flush=True)
            runtime.unassign()
