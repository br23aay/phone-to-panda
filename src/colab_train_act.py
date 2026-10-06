"""Unattended Colab job: copy demos from Drive, convert, train ACT, evaluate, save to Drive.

Everything heavy runs on the local disk. Drive is only touched at the start and
the end, with retries, because the Drive mount can drop under load.
"""
import os
import re
import shutil
import subprocess
import sys
import threading
import time

from google.colab import drive, runtime

D = "/content/drive/MyDrive/phone-to-panda"
NAME = sys.argv[1] if len(sys.argv) > 1 else "1x"
STEPS = sys.argv[2] if len(sys.argv) > 2 else "15000"


def mount():
    try:
        drive.mount("/content/drive", force_remount=True)
    except Exception as e:
        print("mount failed:", e, flush=True)


def retry(fn, what):
    for _ in range(6):
        try:
            return fn()
        except OSError as e:
            print(f"Drive problem during {what}: {e}. Remounting.", flush=True)
            time.sleep(10)
            mount()
    raise RuntimeError(f"gave up on {what}")


def run(cmd, pattern, log):
    print(">>", cmd[:110], flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, errors="replace")
    with open(log, "w") as f:
        for line in p.stdout:
            for part in line.replace("\r", "\n").split("\n"):
                if re.search(pattern, part):
                    print(part.strip()[:300], flush=True)
                    f.write(part.strip() + "\n")
                    f.flush()
    return p.wait()


def copy_file(src, dst):
    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
        shutil.copy(src, dst)


def save_to_drive():
    for local, remote in [("logs", f"{D}/logs_{NAME}"), (f"out/eval_act_{NAME}", f"{D}/eval_act_{NAME}")]:
        if os.path.isdir(local):
            retry(lambda: shutil.copytree(local, remote, dirs_exist_ok=True), f"saving {local}")
    ckpt = os.path.realpath(f"out/act_{NAME}/checkpoints/last/pretrained_model")
    if os.path.isdir(ckpt):
        retry(lambda: shutil.copytree(ckpt, f"{D}/act_{NAME}/pretrained_model", dirs_exist_ok=True), "saving checkpoint")
    print("SAVED TO DRIVE", flush=True)


def sync_checkpoints():
    """Runs in the background: copies each new checkpoint to Drive as soon as it appears."""
    while True:
        time.sleep(120)
        try:
            for ckpt in sorted(os.listdir(f"out/act_{NAME}/checkpoints")):
                src = f"out/act_{NAME}/checkpoints/{ckpt}/pretrained_model"
                dst = f"{D}/act_{NAME}/ckpt_{ckpt}"
                if ckpt == "last" or not os.path.isfile(f"{src}/model.safetensors") or os.path.isdir(dst):
                    continue
                time.sleep(30)                      # let the checkpoint finish writing
                shutil.copytree(src, dst + "_tmp", dirs_exist_ok=True)
                os.rename(dst + "_tmp", dst)
                print(f"checkpoint {ckpt} copied to Drive", flush=True)
        except Exception as e:                      # never let a Drive hiccup stop training
            print("checkpoint sync skipped:", e, flush=True)


def get_dataset(data):
    """Use the converted dataset saved on Drive if there is one; otherwise build it and save it."""
    tar_drive, tar_local = f"{D}/demos_{NAME}_dataset.tar", f"/content/demos_{NAME}_dataset.tar"
    if retry(lambda: os.path.exists(tar_drive), "checking for saved dataset"):
        retry(lambda: copy_file(tar_drive, tar_local), "copying saved dataset")
        os.makedirs("data", exist_ok=True)
        os.system(f"tar -xf {tar_local} -C data")
        print("using the converted dataset saved on Drive", flush=True)
        return
    raw, local_raw = f"{D}/demos_{NAME}_raw", f"/content/demos_{NAME}_raw"
    os.makedirs(local_raw, exist_ok=True)
    names = retry(lambda: sorted(os.listdir(raw)), "listing demos")
    for i, n in enumerate(names, 1):
        retry(lambda: copy_file(f"{raw}/{n}", f"{local_raw}/{n}"), f"copying {n}")
        if i % 10 == 0 or i == len(names):
            print(f"copied {i}/{len(names)} demos to local disk", flush=True)
    run(f"python src/convert_demos.py {local_raw} {data}", r"^\d+/\d+ |DONE|Error", "logs/convert.txt")
    os.system(f"tar -cf {tar_local} -C data demos_{NAME}")
    retry(lambda: copy_file(tar_local, tar_drive), "saving converted dataset")
    print("converted dataset saved to Drive", flush=True)


def main():
    os.environ["MUJOCO_GL"] = "egl"
    os.system('echo N | python -c "import libero.libero" > /dev/null 2>&1')
    os.makedirs("logs", exist_ok=True)
    os.makedirs(f"{D}/act_{NAME}", exist_ok=True)
    data = f"data/demos_{NAME}"
    get_dataset(data)
    threading.Thread(target=sync_checkpoints, daemon=True).start()
    run(f"lerobot-train --policy.type=act --policy.push_to_hub=false --dataset.repo_id=local/demos_{NAME} "
        f"--dataset.root={data} --output_dir=out/act_{NAME} --steps={STEPS} --batch_size=32 "
        f"--save_freq=2500 --log_freq=250 --wandb.enable=false",
        r"step:|Checkpoint|End of training|Error", "logs/train.txt")
    save_to_drive()
    run(f"lerobot-eval --policy.path=out/act_{NAME}/checkpoints/last/pretrained_model --env.type=libero "
        f"--env.task=libero_object --env.task_ids=[0] --eval.batch_size=1 --eval.n_episodes=50 "
        f"--output_dir=out/eval_act_{NAME}",
        r"pc_success|End of eval|Error", "logs/eval.txt")


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            save_to_drive()
            drive.flush_and_unmount()
        finally:
            print("RELEASING GPU", flush=True)
            runtime.unassign()
