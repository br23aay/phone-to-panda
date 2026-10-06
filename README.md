# phone-to-panda

**14 phone clips of my hand, one evening of recording, 101 robot demonstrations.** I filmed myself moving a small box into a tub, turned the video into gripper trajectories, and used them to drive a Panda arm in the LIBERO simulator. Then I measured how far that motion can be sped up before the robot starts failing, and why.

![My phone clip on the left, the simulated Panda following it on the right](results/videos/clip-13_side_by_side.gif)

*Left: one of my phone clips. Right: the Panda in LIBERO driven by that clip, kept in step.*

Submission for the Humanoid Robot Learning Research internship challenge. Every result below comes from data I recorded myself.

| | |
| --- | --- |
| Phone clips recorded | 14 |
| Clips that drive the robot to success | 13 |
| Demonstrations produced (8 layouts per clip) | 101 of 112 replays, 90% |
| Speed conditions measured | 42 replays each, on unseen layouts |
| Policy trained on the demonstrations | did not work, reported below |

## Results

**1. Phone video to robot motion.** 13 of my 14 clips make the simulated arm complete the task ("pick up the alphabet soup and place it in the basket"). Replayed on 8 different starting layouts each, 101 of 112 replays succeed (90%). One clip (clip-7) fails on every layout and one (clip-2) fails on 3 of 8.

**2. Speeding the motion up.** Each clip was replayed at several speeds in two ways: *uniform* (everything faster) and *non-uniform* (faster everywhere except half a second around the grasp and the release). 42 replays per condition, on layouts the demonstrations never used.

| Speedup | Uniform: success | Uniform: picks/min | Non-uniform: success | Non-uniform: picks/min |
| --- | --- | --- | --- | --- |
| 1x | 93% | 7.5 | 93% | 7.5 |
| 1.25x | 71% | 7.2 | 79% | 7.4 |
| 1.5x | 10% | 1.2 | 60% | 6.3 |
| 1.75x | 0% | 0.0 | 31% | 3.6 |
| 2x | 0% | 0.0 | 14% | 1.8 |
| 2.5x | 0% | 0.0 | 7% | 1.0 |
| 3x | 0% | 0.0 | 0% | 0.0 |

"Picks/min" is successful picks per minute of path time, so a fast motion that fails often scores low.

No speedup beat normal speed on successful picks per minute (7.5). The arm finishes sooner when sped up, but it fails often enough to lose overall.

Non-uniform speedup is better than uniform at every speed where anything works. The clips that survive speedup are my slow ones; my fastest clips fail first. So the limit is the absolute speed the arm can track, not the speedup factor.

![Success rate and throughput against speedup](results/figures/speed.png)

![The same clip at normal speed, uniform 2x and non-uniform 2x](results/videos/clip-13_speed_comparison.gif)

*The same clip three ways. Uniform 2x leaves the object behind; non-uniform 2x keeps the grasp.*

**3. Training a policy on the demonstrations.** This did not work. An ACT policy trained only on my 101 demonstrations never completed the task.

| Policy | Success | Episodes |
| --- | --- | --- |
| 15,000 steps, re-planning every 100 actions, 280-step limit | 0% | 50 |
| 15,000 steps, re-planning every 20 actions, 400-step limit | 0% | 20 |
| 50,000 steps (the last 35,000 at 3 times the learning rate), re-planning every 20 actions, 400-step limit | no result file | |

Details and the checks I ran are under "What did not work".

## How it works

1. **Record.** Phone fixed side-on at table height. Box and tub 20 cm apart. 14 clips of 5 to 10 seconds.
2. **Keypoints.** MediaPipe Hand Landmarker gives wrist, thumb tip and index tip in every frame (`src/extract_keypoints.py`).
3. **Retarget.** Each clip becomes three signals relative to the object, not the camera: progress `s` from pick (0) to place (1), height `h` in metres, and a grip flag (`src/retarget.py`).
4. **Replay.** A proportional controller makes the Panda follow that path in LIBERO, using the object and basket positions from the simulator (`src/replay_libero.py`). Successful replays are saved as demonstrations (`src/build_demos.py`).
5. **Speed up.** The same trajectories are retimed uniformly or non-uniformly and replayed again (`src/retime.py`, `src/speed_sweep.py`).

## Design choices

- **Object-centric retargeting.** Because the path is stored relative to the pick and place points, one phone clip can be replayed on any layout. 14 clips gave 101 demonstrations.
- **Side-on camera, planar motion.** Left-right in the image is progress, up-down is height. Depth is ignored. This needs no camera calibration: the hand travels a known 20 cm between grasp and release, which sets the scale.
- **End-effector space only.** A parallel gripper needs a position and open or closed. I use the midpoint of thumb and index, and ignore the other landmarks.
- **Why a speed experiment.** Humanoid's public technical write-up says naive speedup of a policy hurts grasping and that speeding up the data non-uniformly works better. I wanted to see whether that shows up with my own data. The challenge did not ask for this; it is my reading of what would be useful.

## What worked and what did not

**Hand to gripper needed four fixes.** With the raw retargeted path the arm succeeded 0 times out of 5.

| Problem | Cause | Fix |
| --- | --- | --- |
| Fingers hit the object on the way in | A hand can slide in from the side, a gripper cannot | Force a vertical approach before the grasp |
| Arm trailed about 5 cm behind the path | The simulator's controller is compliant | Proportional gain of 3 |
| Carried object knocked the basket over | The path came in too low | Reach release height by the halfway point |
| Object left behind | A hand pinches faster than the gripper closes | Hold still for 10 steps (0.5 s) at the grasp |

**Grasp detection from finger spacing failed.** Around a rigid 6 cm box my fingers only close from about 10 cm to 7 cm, too small a change to detect reliably. I switched to using the hand's path: the two lowest points, one at each end of the sideways travel.

**The hand's shadow was detected as a hand** for about a second in the first clip. Keeping only the longest continuous track removes it.

**clip-7 fails everywhere.** Its grasp is detected too early (it reports holding the box for 6.6 of 7.7 seconds), so the gripper closes in the wrong place.

**The policy does not work.** ACT trained on the 101 demonstrations scores 0%. In the evaluation videos of the first policy the arm drifts away from the objects instead of reaching for them. I checked that this is not a mismatch between training data and evaluation:

- the evaluation environment's camera images are pixel-identical to the saved demonstration frames,
- the robot state vector is identical,
- replaying a saved demonstration's actions inside the evaluation environment succeeds.

So the policy itself has not learned the task. Re-planning every 20 actions instead of 100 and raising the step limit from 280 to 400 made no difference. I do not know the cause. My guesses are too little data for this policy type, and demonstrations that are hard to imitate: they come from 14 different human timings, and the proportional controller produces saturated actions.

**Housekeeping that went wrong.** The results file of the first speed sweep was lost when its Colab session ended. Every outcome had been printed in the log, so `src/reconstruct_sweep.py` rebuilds the file from that log; the step counts are recomputed, which is exact because retiming is deterministic. Rows for 1.25x and 1.75x come from a later run and were written directly.

## Limits

- Depth and wrist rotation are ignored. The gripper always points straight down.
- One task, one object, one simulator.
- Building demonstrations reads object and basket positions from the simulator. A trained policy would not get those; it sees only camera images and its own state.
- The speed results are for open-loop replay with a simple controller, not for a learned policy.
- 14 clips from one person on one evening.

## How to run

Laptop (any OS), for steps 2 and 3:

```
pip install -r requirements.txt
# download hand_landmarker.task from the MediaPipe model page into the repo root
python src/extract_keypoints.py --clips data/raw --out data/keypoints
python src/check_clips.py
```

The keypoints for my 14 clips are already in `data/keypoints/`, so everything below runs without the videos.

Linux or Google Colab, for the simulator:

```
pip install "lerobot[libero]"
export MUJOCO_GL=egl
echo N | python -c "import libero.libero"
python src/build_demos.py            # replay every clip, save demonstrations
python src/speed_sweep.py 10 13      # speed comparison on layouts 10 to 12
python src/summarise_speed.py
python src/plot_results.py
python src/make_videos.py clip-13    # replay videos
```

Policy training used `src/convert_demos.py`, then `lerobot-train --policy.type=act`. The exact commands are in `src/day_job.py` and `src/day_job2.py`, which are the Colab job scripts I ran.

## Why this could matter for a humanoid robot company

I built this around problems Humanoid describes in its public technical write-up, not around a benchmark.

- **Cheap data.** A phone, a table and one evening produced 101 robot demonstrations. Storing the motion relative to the object is what turns 14 clips into 101, because each clip replays on any layout. The same idea should apply to human video collected at scale.
- **The same interface.** The pipeline works in end-effector space: where the gripper should be and whether it is closed. That is the action space Humanoid's VLA uses, so human hand video maps onto it without a hand-to-joint model.
- **Throughput, measured.** Industrial customers pay for picks per minute, not success rate alone. This repo reports successful picks per minute at each speed and shows where speedup stops paying off.
- **Evidence for retiming the data, not the playback.** Keeping the grasp at normal speed beat uniform speedup at every speed that worked at all. That matches what Humanoid reports for its own policies, here reproduced on 14 clips from a phone.
- **Known failure modes.** Every failure in this project has a cause written next to it, or an honest "I do not know yet".

## What I would build next

In the order I would do them:

1. **A tracking controller that anticipates the path.** The proportional controller lags at speed. Feed-forward velocity should move the point where speedup breaks, and would show whether the limit is the controller or the physics of the grasp.
2. **Debug the policy on the smallest case.** Train on one clip's demonstrations, confirm it works, then add clips one at a time to find where imitation breaks.
3. **Residual RL on top of the replay.** Keep the retargeted path as the base and let PPO learn small corrections with a time penalty. My earlier work was PPO on the Shadow Hand in MuJoCo, so this is the direction I know best.
4. **Post-train a small VLA** (SmolVLA) on the same demonstrations and compare it with ACT.
5. **Depth and rotation.** Add a second camera, or an egocentric view with a hand-pose model, so the retargeting is no longer planar.
6. **A second embodiment.** Replay the same object-centric trajectories on a different gripper to test how much of the pipeline is robot-specific.

## Note on AI assistance

This project was built with heavy use of an AI coding assistant (Claude), which the challenge allows. The phone data is my own recording.
