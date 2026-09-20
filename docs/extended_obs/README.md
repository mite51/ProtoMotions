# Extended observations on current main

This branch starts at `607ca7a0bb92e261120bcab8d9f97f28b3130ffc` (fork and upstream
main at integration). Features and FBX tools were ported from
`dev/jwylie/extended_obs_linux` (`cecfac01e`) and
`dev/jwylie/extend_obs` (`175ab820a`), including animation-tool changes unique to
the latter. Current-main MJCF conversion, joint naming,
IsaacLab 3 quaternion/Warp conventions, PPO storage, rewards and optimizers remain
the foundation. Old Windows launch patches, old USD robot copies and RunPod keys
were not imported. Local SMPL model files were preserved and are not committed.

## First comparison: observations only

| Setting | Normal | Extended |
|---|---|---|
| Experiment | `mimic/mlp.py` | `mimic/fight.py` |
| Dataset | `amass_smpl_train_new.pt` | same |
| Tracking rewards, smoothing, termination, PD, reference alignment | main | same |
| Networks, learning rates, rollout, evaluator | main | same, wider input |
| Added inputs | none | 8 contact flags, 6 × 17 collision features, 23 strength values |
| Strength / projectiles / opponents | nominal / off / off | nominal / off / off |

Expanded contact sensors **do not expand the contact-matching reward**: that reward
still compares the same four foot bodies. Nominal-strength observations do not
write gains or torque limits. Optional fighting computation is disabled for the
normal experiment. Neither experiment uses the old stronger articulation rewards,
relaxed action-smoothing penalty, short 100-step evaluation, or smooth reanchoring.

## Run after reviewing validation

The existing compatible interpreter is
`/home/jwylie/anaconda3/envs/protomotion_isaaclab_12/bin/python`
(PyTorch 2.11 CUDA 12.8, IsaacLab 12, Isaac Sim 6). Its editable ProtoMotions install
points elsewhere, so use the launcher below, or set `PYTHONPATH` to this checkout
and use `python -m protomotions.train_agent`. The old `env_isaaclab` directory in
this checkout is not the matching runtime.

```bash
cd /home/jwylie/Dev/ProtoMotions
scripts/train_fighting_comparison.sh baseline
# Run sequentially on the single GPU:
scripts/train_fighting_comparison.sh extended
```

Defaults: 1,024 environments, minibatch 4,096, seed 0, 500 million environment
transitions per run. This is an initial comparison budget, not a convergence
claim. Compare at equal transitions and evaluator budgets; extend both if still
improving. Keep run directories distinct. Existing `last.ckpt` means **resume**:
saved settings take precedence over new CLI overrides. To change environment
count or other settings, use a new `EXPERIMENT_NAME` and `--checkpoint` to warm-start.

For memory pressure, use the same reduced settings for both runs:

```bash
NUM_ENVS=512 BATCH_SIZE=2048 EXPERIMENT_NAME=smpl_baseline_512 \
  scripts/train_fighting_comparison.sh baseline
NUM_ENVS=512 BATCH_SIZE=2048 EXPERIMENT_NAME=smpl_extended_512 \
  scripts/train_fighting_comparison.sh extended
```

The packaged library alone occupies roughly 8.4 GiB before contact smoothing and
simulator/rollout allocations. Reducing environments cannot reduce this fixed cost.
Avoid running both jobs concurrently. Smaller environment counts change rollout
batch size and updates per transition, so keep them matched between runs.

A config-only check does not start the simulator or training:

```bash
EXPERIMENT_NAME=check_extended scripts/train_fighting_comparison.sh extended --create-config-only
```

## Compare behavior before introducing custom animation

Use main's full 600-step evaluator, including failure rate, global position and
orientation errors, and maximum joint error. Inspect the same clips and start
frames for knees, elbows, punches, crouches and getups. Record per-body position
and rotation errors: a good mean can hide a bad limb. Use the same inference PD,
robot assets and dataset. Retain the checkpoints, resolved configs and evaluator
outputs. A second seed is useful before treating a small difference as meaningful.

Do not compare the normal and extended models by loading one model's weights into
the other's differently sized input layer. Train both from scratch for this test.
Passing these comparisons gives evidence against an observation-induced regression;
it cannot prove that all clips or future damage conditions will work.

After the baseline A/B comparison passes, create a separate dataset variant with
the custom clips and repeat **both** models at nominal strength. Keep a fixed panel
of original clips for regression checking. Oversample rare custom moves explicitly
rather than assuming that adding a handful to 9,324 motions will teach them well.
Only then begin the weakness curriculum, so data and strength changes are isolated.

## Per-body strength and the curriculum

`fight_stamina.py` inherits the extended experiment directly. It starts with
per-episode strengths in `[0.75, 1.0]`, with no thrown objects, opponent reward,
extra tracking terms, or reference reanchoring. Use a new experiment name and an
**extended** checkpoint:

```bash
EXPERIMENT_NAME=smpl_stamina_mild scripts/train_fighting_comparison.sh stamina \
  --checkpoint results/smpl_extended_amass_s0/last.ckpt
```

Effective strength `s` scales nominal engine stiffness and torque capacity by `s`
and damping by `sqrt(s)`. This reduces response speed and available torque while
approximately preserving damping ratio. It is not a hard angular-speed cap, nor a
metabolic stamina model. At `s=0`, active drive is disabled; at `s=1`, cached
nominal engine properties are restored. Only the IsaacLab built-in PD path is the
validated strength model; the legacy proportional-PD helper is retained for reference.

Runtime API: `env.set_body_stamina(strength, env_ids=None)` takes
`[selected_environments, 23]` for SMPL, values in `[0, 1]`. Columns correspond to
sorted actuated body indices (`env._stamina_body_indices`); all three axes of a
body's joint share one value. To weaken an entire arm, update its shoulder, elbow,
wrist and hand columns together. A game can combine damage and fatigue into this
effective-strength value. Contact damage **does not yet accumulate automatically**
into strength loss; the caller controls that mapping and recovery law.

Recommended progression after the comparison: mild stationary weakness; coherent
whole-limb weakness; gradual changes during a clip; sudden damage and recovery;
then external impacts and opponents. Keep healthy episodes in later stages and
evaluate healthy tracking separately to catch forgetting. The old `[0.1, 2.0]`
range mixes near-paralysis and superhuman strength and is too broad for a first
weakness stage. Severe impairment can make a reference physically impossible;
measure balance/recovery as well as exact pose tracking before relaxing rewards.

Cautions in the inherited design:

- Constant strength and empty collider slots in stage 1 acquire tiny running
  variance. New values can initially hit observation-normalization clipping when
  warm-starting. Monitor normalization and introduce small changes gradually.
- Collision mass affects top-K selection but is not an input feature. The policy
  cannot directly distinguish otherwise identical objects with different mass.
- The 17-float representation lacks angular velocity and full box dimensions;
  top-K identity can switch abruptly. It is a coarse threat cue, not full geometry.
- Scene obstacles are not populated by the inherited collision candidate builder;
  currently ground, projectiles and opponent body proxies are supported.
- The inherited opponent-impact reward is a proximity/closing-speed proxy, not
  proof of an actual damaging contact. The projectile penalty multiplies body
  force by the maximum active threat, not the identity of the collider that hit it.
  These later experimental tiers should be revised before serious combat training.

## Animation tools

The animation tools `data/scripts/retarget_fbx_to_smpl.py`,
`_fbx_extract_blender.py`, `inspect_fbx.py` and `verify_motion.py` are restored.
This combines Linux support with the other branch's Biped mapping, Blender layered
actions, animated object root motion, and optional orientation correction.
Orientation remains unchanged by default. `--auto-correct` assumes the first
retained pelvis should be upright and face +X; do not use it blindly for getups,
lying poses, or leaning starts. `--rotation-euler X Y Z` gives explicit control.
The retargeter discovers `blender` on PATH or accepts `--blender-exe`. It writes
`.motion` files against the current MJCF; inspect those before building a training
library. Run its `--help` for rest-pose and axis options. Preserve rest poses and
coordinate conventions across clips; do not fix apparent articulation errors by
changing training rewards before inspecting the reference.

Read-only dataset audit:

```bash
PYTHONPATH=. /home/jwylie/anaconda3/envs/protomotion_isaaclab_12/bin/python \
  -m scripts.check_fighting_motion_data \
  /home/jwylie/Dev/ProtomotionsAnimData/amass_smpl_train_new.pt \
  --output results/extended_obs_validation/data_report.json
```

`ai/docs/` contains historical notes, not the current training contract. This
README supersedes the old fighting deployment curriculum description.

On this machine, `/usr/bin/blender` is missing NumPy. The working executable is
`/home/jwylie/Downloads/blender-5.2.1-linux-x64/blender`; pass it with
`--blender-exe`. Batch extraction uses factory startup so interactive add-ons do
not affect conversion. A complete Kimodo face-up getup conversion with the
separate `tpose.fbx` succeeded. It retained the source's 24 FPS (131 frames);
`--output-fps 30` does not upsample a 24 FPS source. Inspect visual quality before
adding the result to either training library.

## Validation on 2026-09-20

No full training or trained-policy behavior comparison has been performed.
The saved smoke checkpoints are startup/update checks, not usable trained models.

| Check | Result |
|---|---|
| Normal model, full dataset, 1,024 envs / batch 4,096 | 2 PPO iterations, saved checkpoint, no OOM |
| Extended model, same settings | 2 PPO iterations, saved checkpoint, no OOM |
| Mild randomized strength, full dataset, 32 envs / batch 32 | 2 PPO iterations, saved checkpoint |
| Runtime strength 0 / 0.25 / 1 | Actual engine stiffness, damping and torque limits verified; other 31 envs unchanged; original properties restored |
| Selected regression suite | 232 passed; 2 checkpoint-dependent tests excluded because their large LFS checkpoints are unavailable locally |
| Unified ONNX export and ONNX Runtime | Maximum action difference 1.12e-8; joint-target difference 5.96e-8 |
| FBX extraction and retargeting | Blender 5.2, Kimodo face-up getup + separate rest pose, 131 finite frames |
| Source hygiene | Ruff and git diff whitespace checks pass |

The data audit checked all tensor values in 9,324 motions / 4,094,531 frames,
motion boundaries, and contact labels. Forward kinematics on 2,048 sampled frames
matched stored body positions to a maximum of 1.67e-6 m using the expected
exponential-map joint convention. Interpreting those values as Euler hinge angles
instead gives large errors; preserve the declared joint convention across tools.
Windows paths in the library are source metadata; the packaged tensors load on
Linux without access to those paths. The packaged baseline contains no clips
identified by the custom Fighting/Kimodo/TEST path categories.

Local logs, data audit, runtime assertion script/report, exported ONNX and sample
retargeted motion are under `results/extended_obs_validation/` (git-ignored).
Smoke checkpoints use separate `results/extended_obs_*smoke*/` directories.
The contact-label smoothing path now temporarily disables cuDNN autotuning while
processing variable-length clips and restores it afterward. This avoids thousands
of per-length tuning searches without changing the smoothing calculation.
