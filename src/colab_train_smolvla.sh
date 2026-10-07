#!/usr/bin/env bash
# Fine-tune SmolVLA on the 101 demonstrations and evaluate it in LIBERO.
# Run on a Colab L4 GPU. On a T4 one training step took 4.2 s, on an L4 about 0.47 s.
# Expects the converted dataset (src/convert_demos.py) at data/demos_1x.
set -e
export MUJOCO_GL=egl
pip install -q "lerobot[libero,smolvla,training]"

# SmolVLA's base checkpoint names its cameras camera1, camera2, camera3.
RM='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'

lerobot-train \
  --policy.path=lerobot/smolvla_base --policy.push_to_hub=false \
  --dataset.repo_id=local/demos_1x --dataset.root=data/demos_1x \
  --rename_map="$RM" \
  --output_dir=out/smolvla --steps=20000 --batch_size=16 \
  --log_freq=200 --save_freq=5000 --wandb.enable=false

# N_ACTION_STEPS is how many actions are executed before the policy looks at the
# cameras again. 50 is the SmolVLA default (20% success), 10 gave 58%.
N_ACTION_STEPS=${N_ACTION_STEPS:-10}
echo N | lerobot-eval \
  --policy.path=out/smolvla/checkpoints/last/pretrained_model \
  --policy.n_action_steps="$N_ACTION_STEPS" \
  --env.type=libero --env.task=libero_object --env.task_ids=[0] \
  --env.observation_height=256 --env.observation_width=256 \
  --env.episode_length=400 --eval.batch_size=1 --eval.n_episodes=50 \
  --rename_map="$RM" --output_dir=out/smolvla_eval_n"$N_ACTION_STEPS"
