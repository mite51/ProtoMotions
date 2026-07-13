<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
SPDX-License-Identifier: Apache-2.0
-->

# Fighting-Game Mimic — Deployment & Observation Spec

This document is the **portable contract** for the fighting-game mimic policy. It
describes everything that differs from the stock mimic tracker so the changes can
be mirrored exactly into other inference clients (the two non-ProtoMotions clients
and the ONNX path). If a client reproduces the observation layout, action contract,
and timing below, the same `.onnx` will drive it identically to ProtoMotions.

> Status: covers all tiers — base, thrown colliders, stamina, and **multi-character
> self-play** (Phase 5, `fight_multichar.py`). Opponents appear purely as additional
> **collision primitives** and require **no observation-shape change** (that is the
> whole reason the layout below is frozen). A client that never instantiates opponents
> is unaffected; the same `.onnx` drives single- and multi-character deployments.

---

## 1. What changed vs. the stock mimic tracker

| Area | Stock mimic | Fighting mimic |
|---|---|---|
| Robot / control | `smpl`, BUILT_IN_PD | **`smpl`, BUILT_IN_PD** (engine-side PD; per-joint gains are set from stamina, see below) |
| Ground sensing | root-height / terrain obs | **`collision_primitives`** — unified egocentric encoding of ground + obstacles + projectiles + (later) opponents |
| Contacts | usually off | **`observe_contacts=True`** on a frozen body set |
| Per-limb strength | n/a | **`stamina_obs`** = per-body joint-stiffness scale; the engine's per-joint gains are set to `nominal * stamina` |
| Reference tracking | world-anchored | **per-step XY re-anchoring** (shoves don't accumulate absolute-position error) |
| Action outputs | joint position targets | joint position targets (same contract; gains live in the engine, not the action output) |

Experiment files: `examples/experiments/mimic/fight.py` (Tier 1 base, defines the
frozen layout), `fight_throw.py` (Tier 2), `fight_stamina.py` (Tier 2b, stamina).

> **Control mode note.** Earlier iterations used `smpl-proportional` (PROPORTIONAL,
> Python-side PD) so stamina could scale gains per step. That path is unstable on
> IsaacLab (actuators run at zero gain + explicit effort torques → the character can
> "launch"). The model now uses the standard **BUILT_IN_PD** engine PD; stamina
> changes the **engine's** joint stiffness/damping on reset instead
> (`Simulator.set_joint_gain_scale`).

---

## 2. Frozen observation layout (policy `in_keys`)

The actor/critic consume these keys **in this order**. Widths are fixed for the
life of the model — do not add, remove, or reorder.

```
[ max_coords_obs, mimic_target_poses, previous_actions, collision_primitives, stamina_obs ]
```

| Key | Factory (in `fight.py`) | Binds to (context paths) | Notes |
|---|---|---|---|
| `max_coords_obs` | `max_coords_obs_factory(observe_contacts=True)` | `current.rigid_body_{pos,rot,vel,ang_vel}`, `body_contacts` | Stock max-coords proprio + a per-contact-body boolean channel. |
| `mimic_target_poses` | `mimic_target_poses_max_coords_factory(with_velocities=True)` | `current.*`, `mimic.future_{pos,rot,vel,ang_vel}` | Future reference frames relative to current root. Velocities included. |
| `previous_actions` | `previous_actions_factory(history_steps=1)` | `historical.actions` | Last raw action. |
| `collision_primitives` | `collision_primitives_obs_factory(num_obs_primitives=6)` | `current.rigid_body_{pos,rot,vel}`, `collision_primitives.*` | K=6 nearest primitives × 17 floats = **102 floats**. See §3. |
| `stamina_obs` | `stamina_obs_factory()` | `body_stamina` | One scalar per **actuated body**. See §5. |

Contact body set (frozen), from `configure_robot_and_simulator` in `fight.py`:

```
all_left_foot_bodies, all_right_foot_bodies,
all_left_hand_bodies, all_right_hand_bodies,
head_body_name, torso_body_name
```

---

## 3. Collision-primitive encoding (the `collision_primitives` obs)

This replaces ground observations with one uniform representation for **every**
collidable entity. The policy kernel
(`protomotions/envs/obs/collision_primitives.py`) does the egocentric transform,
distance sort, top-K selection, and zero-padding **inside the graph**. A client
therefore only needs to populate a **world-frame candidate buffer**; the model does
the rest.

### 3.1 Per-primitive 17-float layout (egocentric, heading-aligned hip frame)

| idx | name | dim | description |
|---|---|---|---|
| 0:3 | `rel_pos` | 3 | position relative to hips (heading frame) |
| 3:6 | `rel_lin_vel` | 3 | linear velocity relative to hips (heading frame) |
| 6:9 | `local_tan` | 3 | primitive local X-axis (tangent/forward), heading frame |
| 9:12 | `local_norm` | 3 | primitive local Z-axis (normal/up), heading frame |
| 12 | `radius` | 1 | sphere/capsule radius (0 for boxes) |
| 13 | `extent_z` | 1 | box height / capsule cylinder length (0 for spheres) |
| 14 | `damage` | 1 | threat scalar (replaces `danger_level`); higher = more harmful |
| 15:16 | `shape` | 2 | one-hot `[is_box, is_sphere]`; capsule = `[0, 0]` |

`local_tan` / `local_norm` together are the continuous 6-D (tan/norm) encoding of
the primitive's orientation (the same representation used elsewhere in the repo for
rotations), so there is no quaternion discontinuity.

### 3.2 World-frame candidate buffer (what the client fills)

The env/client maintains, per agent, a fixed-capacity buffer of **M** candidates
(`EnvConfig.max_collision_primitives`, default **16**). Each candidate has, in
WORLD frame:

| field | shape | meaning |
|---|---|---|
| `pos` | `[M, 3]` | world position of the primitive center |
| `rot` | `[M, 4]` | world orientation quaternion (xyzw) |
| `lin_vel` | `[M, 3]` | world linear velocity |
| `radius` | `[M]` | sphere/capsule radius (0 for box) |
| `extent_z` | `[M]` | box full height / capsule length (0 for sphere) |
| `damage` | `[M]` | threat scalar |
| `shape` | `[M, 2]` | one-hot `[is_box, is_sphere]` |
| `valid` | `[M]` | 1.0 = active, 0.0 = padding (sorted to the end, zeroed out) |

The kernel emits the **K = 6 nearest valid** candidates by egocentric distance to
the root, zero-padding if fewer than K are valid. `K ≤ M` always.

### 3.3 Candidate selection rules (mirror these on the client)

Per the spec, populate the buffer each step as:

1. **Slot 0 — ground:** a box directly beneath the root at the ground height
   (`pos = [root_x, root_y, ground_z]`, `shape = box`, `damage = 0`). `rel_pos.z`
   then encodes how high the character is above ground (a richer root-height).
2. **Obstacles / projectiles:** dynamic/static colliders, by distance. Projectiles
   are a randomized mix of primitive shapes thrown with a random orientation and mild
   tumble, so the client **must not** assume boxes — populate `radius`/`extent_z`/`shape`
   per the actual primitive:
   - **box:** `radius = 0`, `extent_z = full height`, `shape = [1, 0]`
   - **sphere:** `radius = r`, `extent_z = 0`, `shape = [0, 1]`
   - **capsule:** `radius = r`, `extent_z = cylinder length`, `shape = [0, 0]`
   The shape mix and per-slot sizes come from `ProjectileConfig.shapes` /
   `get_shape_specs()` (each pool slot is a fixed shape, round-robin over `shapes`);
   ProtoMotions fills these fields from `Simulator.get_active_projectile_states()`
   (`radius`/`extent_z`/`shape`) in `BaseEnv._build_collision_primitives`. Mixed
   geometry is realized on the IsaacLab backend; other backends fall back to box.
3. **Other characters (multi-char tier):** the opponents' key bodies only —
   **head, pelvis, arm, forearm, foot, shin** — distance-filtered. Wired in
   `BaseEnv._write_opponent_primitives`: each opponent key body becomes a sphere
   primitive (`radius = EnvConfig.opponent_body_radius`, `damage = 0`) placed in the
   striking character's candidate buffer after the ground/projectile slots. The set of
   opponent bodies is configured via `EnvConfig.opponent_key_body_names`.

If you have more live candidates than M, keep the nearest M (or any superset of the
nearest K); the kernel will pick the final K. Velocities can be supplied directly
or derived by finite difference on positions (ProtoMotions derives projectile
velocity by finite difference — see `BaseEnv._build_collision_primitives`).

---

## 4. Reference re-anchoring (affects how you feed `mimic_target_poses`)

`fight.py` enables `realign_motion_with_humanoid_on_each_step=True`. Each step the
reference root **XY** is shifted to coincide with the character's actual root, so an
unavoidable shove does **not** accumulate absolute-position tracking error; pose
shape, orientation, relative locomotion, and velocities are still tracked.

Client implication: provide the future reference frames (`mimic.future_*`) using the
**same convention the model was trained on** — i.e. expressed relative to the
character's *current* root each frame, not a fixed world anchor. (The existing
mimic obs already builds future targets relative to current state; re-anchoring just
guarantees the offset is recomputed every step.)

---

## 5. Stamina (per-body joint-stiffness scale → engine PD gains)

`stamina_obs` is one scalar per **actuated body** (a body with hinge DOFs). It is
the body's **joint-stiffness scale**: `1.0 = nominal gains`, `<1` weaker/more
compliant, `>1` stiffer. All of a body's DOF axes share the same scalar, and
**damping scales by the same factor as stiffness**.

- **Ordering** of the stamina vector: actuated bodies in **ascending body index**
  (`sorted(kinematic_info.hinge_axes_map.keys())`). The client must use the same
  order.
- **Body → DOF mapping:** DOFs are emitted in ascending-body-index traversal order,
  so each actuated body owns a contiguous run of DOFs equal to its hinge-DOF count.
  Per-DOF scale = repeat each body's stamina across its DOFs
  (see `BaseEnv._init_body_stamina` / `_apply_body_stamina_to_gains`).
- **Effect (BUILT_IN_PD):** the engine's per-DOF `stiffness` and `damping` are set to
  `nominal * stamina`. In ProtoMotions this happens on reset via
  `Simulator.set_joint_gain_scale` (IsaacLab: `write_joint_stiffness/damping_to_sim`).
  A deploy client must do the equivalent: **set its PD controller's per-joint stiffness
  and damping to `nominal * stamina`**, and feed the same per-body stamina into
  `stamina_obs`. There is no per-step gain output to read — the value is a persistent
  engine setting that changes when stamina changes.

During Tiers 1–2 stamina is inert (`1.0` = nominal). The stamina tier
(`fight_stamina.py`, Tier 2b) randomizes it per-episode in `body_stamina_range`
(default `(0.1, 2.0)`).

---

## 6. Action / control contract

Robot uses **`smpl`** (BUILT_IN_PD). The action stage (`make_pd_action_config`) maps
the policy output to PD **joint-position targets** (`pd_action_offset +
pd_action_scale * tanh(action)`). The physics engine runs the PD loop internally
using its configured per-joint stiffness/damping.

Runtime PD law (applied by the engine, per DOF, per substep):

```
torque = stiffness * (joint_pos_target - joint_pos) - damping * joint_vel
```

where `stiffness`/`damping` are the **engine's** current per-joint gains. For the
fighting model those gains are `nominal * stamina` (see §5) — set on reset, not per
step. A deploy client therefore:

1. Feeds `stamina` into `stamina_obs`.
2. Sets its PD controller's per-joint `stiffness`/`damping` to `nominal * stamina`.
3. Runs the policy to get `joint_pos_targets` and applies engine PD to reach them.

(Effort limits still clamp torque, so a `stamina = 2.0` body is stiffer but bounded.)

---

## 7. ONNX export & deploy contract

Use `deployment/export_bm_tracker_onnx.py` as the template. The unified pipeline
already produces the four outputs this model needs:

| ONNX output | shape | meaning |
|---|---|---|
| `actions` | `[B, num_dofs]` | raw policy action |
| `joint_pos_targets` | `[B, num_dofs]` | PD position targets (the value you drive the engine PD toward) |
| `stiffness_targets` | `[B, num_dofs]` | **nominal** per-DOF stiffness (constant; see §7.2) |
| `damping_targets` | `[B, num_dofs]` | **nominal** per-DOF damping (constant; see §7.2) |

Under BUILT_IN_PD only `joint_pos_targets` is needed to drive the engine; the two
gain outputs are the nominal constants and are provided for convenience (the YAML
sidecar also carries them as `default_joint_stiffness/damping`).

**Inputs** are the bound context attribute paths (sanitized: `.`→`_`). For the fight
model these include, in addition to the stock tracker inputs:

- `collision_primitives.pos|rot|lin_vel|radius|extent_z|damage|shape|valid`
- `body_stamina`
- `current.rigid_body_vel` (collision-primitive obs uses body velocity)

### 7.1 Export-script changes (Phase 6 — implemented & validated)

`export_bm_tracker_onnx.py`'s `MockContext` was extended with the fight bindings so
tracing resolves them (shapes for `num_envs=1`):

- `mock.collision_primitives` with `.pos [1,M,3]`, `.rot [1,M,4]`, `.lin_vel [1,M,3]`,
  `.radius [1,M]`, `.extent_z [1,M]`, `.damage [1,M]`, `.shape [1,M,2]`, `.valid [1,M]`
  (`M = env_config.max_collision_primitives`).
- `mock.body_stamina [1, num_stamina_bodies]` where `num_stamina_bodies =
  len(robot_config.kinematic_info.hinge_axes_map)`.
- `mock.body_contacts [1, num_contact_bodies]` sized to the **frozen contact-body
  subset** (`len(build_body_ids_tensor(body_names, robot_config.contact_bodies))`),
  NOT all bodies — `max_coords_obs(observe_contacts=True)` emits one channel per
  contact body, so mis-sizing this breaks the actor's first-layer width.
- `mock.current.rigid_body_vel` was already present.

The obs layout is unchanged by the control-mode switch, so the width math still holds
(24 bodies): actor input width `1136 = max_coords_obs(366) + mimic_target_poses(576) +
previous_actions(69) + collision_primitives(102) + stamina_obs(23)`, with 8 contact
bodies. The collision-primitive top-K (K=6) and candidate capacity (M=16) are baked as
constants in the traced graph, so the ONNX expects exactly M=16 candidate slots. The
YAML sidecar now reports `control_type: BUILT_IN_PD`. Re-run the export on the new
`smpl` base checkpoint and confirm the same 1136 width + ~1e-8 output match.

### 7.2 Stamina ⇄ gains caveat (important)

The exported `stiffness_targets` / `damping_targets` are the **nominal** per-DOF gains
(constant in the graph). Stamina is **not** a gain multiplier applied inside the ONNX
— it is a persistent engine setting. So the deploy client must:

1. Read `stamina` (the per-body values it feeds into `stamina_obs`, §5).
2. Expand to per-DOF and set the engine's per-joint gains to `nominal * stamina`
   (using the ONNX/YAML nominal gains as the base).
3. Drive engine PD toward `joint_pos_targets`.

This exactly matches training dynamics (`Simulator.set_joint_gain_scale` sets
`nominal * stamina` on reset). `stamina` never needs to be an ONNX input because the
gains are an engine-side setting, not a per-step network output.

### 7.3 Timing

The deploy host **must** tick at the policy's training cadence
(`control_dt = physics_dt * decimation`). The export reads these from the training
`simulator_config.sim.{fps,decimation}` and writes them into the YAML sidecar — do
not trust generic defaults.

---

## 8. Training tiers (warm-start chain)

All tiers share the **identical** frozen architecture, so each **warm-starts** from
the previous checkpoint; only environment curriculum knobs change.

> **Two warm-start gotchas (learned the hard way):**
>
> 1. **`--checkpoint` restores the epoch/step counter.** `--training-max-steps` is an
>    *absolute* cap (`max_epochs = training_max_steps // num_envs // num_steps`), and
>    warm-start restores `current_epoch`/`step_count` from the loaded checkpoint. So
>    each tier's cap must be **cumulative**: if Tier 1 trained to 1.5M, Tier 2 must
>    pass `--training-max-steps 3000000` to add another 1.5M, Tier 3 `4500000`, etc.
>    Passing the same 1.5M to a warm-started tier makes it exit immediately (0 epochs).
> 2. **A pre-existing `results/<experiment-name>/last.ckpt` forces RESUME mode**, which
>    loads the pickled config + saved CLI args and **ignores** `--checkpoint`,
>    `--overrides`, and `--training-max-steps`. To warm-start a *new* tier, use a fresh
>    `--experiment-name` (or delete the stale `results/<name>` dir first).

```bash
# Tier 1 — base (interference inert)
python protomotions/train_agent.py --robot-name smpl --simulator isaaclab \
  --experiment-path examples/experiments/mimic/fight.py \
  --experiment-name smpl_fight_base --motion-file <motions.pt> \
  --num-envs 8192 --batch-size 8192

# Tier 2 — thrown colliders (resume Tier 1)
python protomotions/train_agent.py --robot-name smpl --simulator isaaclab \
  --experiment-path examples/experiments/mimic/fight_throw.py \
  --experiment-name smpl_fight_throw --motion-file <motions.pt> \
  --num-envs 8192 --batch-size 8192 --checkpoint results/smpl_fight_base/last.ckpt
# ramp difficulty: --overrides simulator.projectile.auto_throw_prob=0.03

# Tier 2b — randomized per-body stamina / joint-stiffness scale (resume Tier 2)
python protomotions/train_agent.py --robot-name smpl --simulator isaaclab \
  --experiment-path examples/experiments/mimic/fight_stamina.py \
  --experiment-name smpl_fight_stamina --motion-file <motions.pt> \
  --num-envs 8192 --batch-size 8192 --checkpoint results/smpl_fight_throw/last.ckpt

# Tier 5 — multi-character self-play (warm-start Tier 2b)
# NUM_CHARACTERS (N) defaults to 2 in fight_multichar.py; override for more.
# IMPORTANT: --num-envs is the PHYSICAL scene count (E). The RL agent sees E * N rows
# (one per character), and there are E * N robots in the sim. So for the same VRAM /
# robot budget as the single-character tiers, HALVE --num-envs when N=2 (quarter for
# N=4, etc.). Example: --num-envs 2048 with N=2 -> 4096 agent rows / 4096 robots.
# training_max_steps is measured in agent rows too (max_epochs = steps // (E*N) //
# num_steps), so it stays comparable to the other tiers at the same robot budget.
python protomotions/train_agent.py --robot-name smpl --simulator isaaclab \
  --experiment-path examples/experiments/mimic/fight_multichar.py \
  --experiment-name smpl_fight_multichar --motion-file <motions.pt> \
  --num-envs 2048 --batch-size 4096 --checkpoint results/smpl_fight_stamina/last.ckpt
# more characters: --overrides simulator.num_characters=3   (and lower --num-envs to E)
```

### 8.1 Multi-character architecture (deploy-irrelevant, training-only)

Multi-character self-play spawns **N physically-interacting articulations per scene**
(`Robot_0..Robot_{N-1}`) and presents the RL agent a **flattened** view of `E × N` rows
(E = physical scenes = `--num-envs`; the agent's `num_envs` is `E × N`). All N characters
share one policy and all contribute to the experience buffer, so a 2-character run
roughly doubles samples per scene. Opponents are real PD-controlled bodies (not
kinematic), so contacts, strikes, and interference are physically correct.

This is **entirely a training/simulator concern** — it does not change the observation
layout, action contract, or ONNX graph. An inference client deploying the policy on a
single body needs nothing from this section. `num_characters = 1` is a strict no-op
that reproduces the single-character code path exactly.

The opponent-impact **reward** (`opponent_impact_rew_factory`, in
`fight_multichar.py`) rewards a character's fast-moving striking limbs
(`EnvConfig.striking_body_names`) for closing speed into an opponent's key bodies
within `EnvConfig.opponent_strike_radius`. Rewards never appear at deploy time.

---

## 8.2 Impact robustness (training stability)

Hard impacts (fast projectiles, opponent collisions) can rarely destabilize the PhysX
solver in a single env, producing extreme/non-finite contact forces or state. To keep
multi-hour runs from crashing on these rare events, the fighting tiers add three guards
(all deploy-irrelevant — they only affect training stability):

1. **Contact-force clamp** (`BaseEnv`, `_MAX_CONTACT_FORCE = 1e5 N`): per-body contact
   magnitudes are `nan_to_num`'d and clamped before any reward sees them. Far above any
   real humanoid contact, so normal gradients are untouched.
2. **Opt-in non-finite state recovery** (`EnvConfig.sanitize_non_finite_state=True`,
   frozen on from Tier 1): if the simulator returns non-finite state for some env, it is
   sanitized (rotations→identity, else→0) with a throttled warning instead of asserting;
   the affected env fails tracking and resets on the next step. Default-off elsewhere, so
   all other training keeps strict fail-fast validation.
3. **Reward sanitize** (`combine_rewards`, same opt-in flag): a non-finite reward from a
   near-blowup (e.g. power ∝ vel²) is zeroed for that env this step rather than asserting.
4. **Gentler projectiles** (`fight_throw.py`): `speed_range=(12,22) m/s`, `density=300`
   so a cube travels ≲ its own size per substep (avoids tunneling). Ramp back up via
   `--overrides simulator.projectile.{speed_range,auto_throw_prob,density}=...` as the
   policy hardens.

## 9. Quick checklist for a new inference client

- [ ] Robot uses BUILT_IN_PD (engine-side PD); drive the engine toward `joint_pos_targets` at the trained cadence.
- [ ] Build observations in the exact order of §2 with the exact widths.
- [ ] Maintain the M-slot world-frame candidate buffer (§3.2); fill ground + obstacles + (later) opponent key bodies (§3.3). Let the ONNX do selection/transform.
- [ ] Feed `body_stamina` (§5 ordering); set engine per-joint gains to `nominal * stamina` (stiffness and damping) whenever stamina changes.
- [ ] Provide future reference frames relative to the current root each step (§4).
- [ ] Tick at `control_dt` from the YAML sidecar (§7.3).
