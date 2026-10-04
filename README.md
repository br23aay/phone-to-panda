# phone-to-panda

Work in progress. This README is a placeholder and will be replaced with results and a full write-up.

A phone video of my hand moving a box into a tub is turned into a trajectory that drives a Panda arm in the LIBERO simulator. The same trajectories are then sped up in two ways to see which one still completes the task.

## Status

Hand keypoints are extracted from the phone clips, each clip is retargeted to an object-centric gripper path, and the path is replayed on the Panda in LIBERO. The speed comparison and policy training are in progress.

## Files

| File | What it does |
| --- | --- |
| src/extract_keypoints.py | Phone clip to wrist, thumb-tip and index-tip pixels |
| src/retarget.py | Keypoints to a gripper trajectory relative to the object |
| src/check_clips.py | Reports which clips are usable |
| src/replay_libero.py | Replays a trajectory on the Panda in LIBERO |
| src/build_demos.py | Saves successful replays as demonstrations |
| src/retime.py | Uniform and non-uniform speedup |
| src/speed_sweep.py | Replays every clip at several speeds |
| src/summarise_speed.py | Success rate and picks per minute per speed |
| src/plot_results.py | Chart of the speed results |
