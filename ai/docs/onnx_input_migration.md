# ONNX Input Migration: Pre-Computed Observations -> Raw Context Tensors

## TL;DR

The new ONNX export (`deployment/export_bm_tracker_onnx.py`) bakes the
**observation computation** into the ONNX graph. Where the old export consumed
already-cooked observation tensors (`max_coords_obs`, `mimic_target_poses`,
`historical_previous_actions`), the new export consumes the **raw context
tensors** the obs functions read from.

The training/inference behavior of the actor is unchanged - only the cut
between "host code" and "ONNX" has moved up the pipeline.

| | Old ONNX (actor-only) | New ONNX (`unified_pipeline`) |
|---|---|---|
| Inputs | Cooked observations | Raw simulator/mimic context |
| Obs computation | Done in host code | Inside ONNX graph |
| Action processing (PD targets, stiffness/damping) | Done in host code | Inside ONNX graph |
| Outputs | `actions` only | `actions`, `joint_pos_targets`, `stiffness_targets`, `damping_targets` |

Host code now only needs to feed simulator state + mimic reference state, and
read PD targets / gains directly. No need to re-implement obs functions.

---

## Old vs New Input Lists

### Old ONNX inputs (the previous `OVProtomotions/run_inference.py` contract)

```
max_coords_obs              # cooked obs tensor
mimic_target_poses          # cooked obs tensor
historical_previous_actions # cooked obs tensor
```

### New ONNX inputs (current `unified_pipeline.onnx`)

```
current_rigid_body_pos
current_rigid_body_rot
current_rigid_body_vel
current_rigid_body_ang_vel
ground_heights
historical_actions
mimic_future_pos
mimic_future_rot
mimic_future_vel
mimic_future_ang_vel
```

These names are the dotted context paths (`current.rigid_body_pos`,
`mimic.future_pos`, `historical.actions`, ...) sanitized for ONNX
(dots -> underscores).

---

## What each new input is

All shapes use `B = batch (num_envs)`, `N_b = num_bodies`, `N_dof = num_dofs`,
`F = num_future_steps`, `H = history_steps`. For the SMPL mimic mlp config
trained here: `N_b = 24`, `N_dof = 69`, `F = 4` (mimic future_steps), `H = 1`
(default `previous_actions_factory(history_steps=1)`).

### From `current.*` (current robot state, post-physics-step)

| Input | Shape | Frame / units | Description |
|---|---|---|---|
| `current_rigid_body_pos` | `[B, N_b, 3]` | World, meters | Rigid body positions of all bodies. |
| `current_rigid_body_rot` | `[B, N_b, 4]` | World, **xyzw** | Rigid body rotations (quaternion, w-last). |
| `current_rigid_body_vel` | `[B, N_b, 3]` | World, m/s | Rigid body linear velocities. |
| `current_rigid_body_ang_vel` | `[B, N_b, 3]` | World, rad/s | Rigid body angular velocities. |
| `ground_heights` | `[B]` | World, meters | Terrain height under each robot's pelvis (root). For flat terrain this is just `0`. |

> Quaternion convention is **xyzw** (w-last). This is the "common" format the
> `Simulator` base class converts everything to, regardless of whether the
> underlying simulator uses wxyz internally (IsaacGym/IsaacLab) or xyzw natively
> (Newton/Genesis). See `protomotions/simulator/base_simulator/simulator.py`.

### From `mimic.future_*` (target motion lookahead)

| Input | Shape | Frame / units | Description |
|---|---|---|---|
| `mimic_future_pos` | `[B, F, N_b, 3]` | World, meters | Target body positions at each of the `F` future control steps. |
| `mimic_future_rot` | `[B, F, N_b, 4]` | World, xyzw | Target body rotations. |
| `mimic_future_vel` | `[B, F, N_b, 3]` | World, m/s | Target body linear velocities. |
| `mimic_future_ang_vel` | `[B, F, N_b, 3]` | World, rad/s | Target body angular velocities. |

> The set of future steps comes from the `MimicControlConfig.future_steps`
> field (int N -> `[1, 2, ..., N]`, list -> explicit indices). For this
> checkpoint they are listed in `unified_pipeline.yaml` under
> `motion.future_step_indices`. Stride is one control step
> (`control_dt = 0.02 s` -> 50 Hz by default).

> The mimic component also subtracts a per-env XY offset on the first frame so
> all future targets are expressed in the robot's current footprint (see
> `MimicControl.step` below). If you precompute these in the host, replicate
> that offset.

### From `historical.actions`

| Input | Shape | Frame / units | Description |
|---|---|---|---|
| `historical_actions` | `[B, H, N_dof]` | Action space | Raw actions from the previous H steps (most recent past at index 0). |

> This is **raw** policy action (pre-PD-scaling). The factory has a
> `processed=True` variant that uses post-tanh/clamp values; this config does
> not use it. On the deployment side you typically just store the previous
> ONNX `actions` output and feed it back next step.

---

## Where the inputs come from in code

### Where the obs functions live (what was previously baked into ONNX)

These are the functions that turn raw context into the cooked obs tensors.
They are now called inside `ObservationExportModule.forward` rather than in
`OVProtomotions`.

- `protomotions/envs/obs/humanoid.py:241` -
  `compute_humanoid_max_coords_observations(body_pos, body_rot, body_vel,
  body_ang_vel, ground_height, body_contacts, *, local_obs, root_height_obs,
  observe_contacts, w_last)` produces `max_coords_obs`.
- `protomotions/envs/obs/target_poses.py:158` -
  `build_max_coords_target_poses(current_state_body_*, mimic_ref_*, *,
  with_velocities, with_relative, w_last, future_steps)` produces
  `mimic_target_poses`.
- `protomotions/envs/obs/humanoid_historical.py:234` -
  `compute_historical_actions_from_state(historical_actions, *,
  history_steps)` produces the `previous_actions` obs.

### Where the new ONNX inputs are populated at runtime

This is the side you actually need to mirror in `OVProtomotions`. The
ProtoMotions runtime fills them as follows:

#### `current.rigid_body_*` and `ground_heights`

Filled in `BaseEnv.post_physics_step()`:

- `protomotions/envs/base_env/env.py:698` - ground heights queried from
  terrain using the pelvis (root) XY:
  ```python
  ground_heights = self.terrain.get_ground_heights(
      current_state.rigid_body_pos[:, 0]
  )
  ```
- `protomotions/envs/base_env/env.py:731-738` - current rigid body state +
  ground heights pushed into `EnvContext.current` for obs binding.

In your test app, get rigid body pos/rot/vel/ang_vel from the simulator
wrapper once per control step (after the physics substeps), and either compute
ground heights from your terrain (if any) or pass zeros for flat-ground
scenes. Body ordering must match `kinematic_info.body_names` - the
`unified_pipeline.yaml` exposes this list explicitly.

#### `mimic.future_*`

Filled by the mimic control component at the start of each step:

- `protomotions/envs/control/mimic_control.py:158-198` - the component samples
  the motion library `F` steps ahead of the current motion time, reshapes to
  `[B, F, N_b, *]`, applies the per-env XY offset so the first future step is
  centered on the current pelvis XY, and writes them to `EnvContext.mimic`.

In your test app you need a motion player that, given current time `t`,
returns target body pose+vel at times `t + i * control_dt` for `i in
future_step_indices` (`unified_pipeline.yaml` -> `motion.future_step_indices`).
Reshape to `[1, F, N_b, *]` and apply the XY-offset trick from the file above
to match training distribution.

#### `historical.actions`

Maintained by the historical-state buffer:

- `protomotions/envs/obs/state_history_buffer.py:330-394` -
  `rotate_and_update(...)` rolls the buffer and writes the latest `actions` to
  index 0 each step.
- `protomotions/envs/obs/state_history_buffer.py:211-213` - the
  `historical_actions` property is `self.actions[:, 1:]`, i.e. everything
  *before* the current action - that is what binds to ONNX input
  `historical_actions`.

In your test app, keep a `[1, H, N_dof]` buffer initialized to zeros. After
each ONNX call, push the new `actions` output to the front and shift older
entries right (drop oldest). Feed the buffer back next step.

### Where the export is wired together

- `protomotions/utils/export_utils.py:1056` - `ObservationExportModule`
  collects the union of `dynamic_vars` paths across all configured obs
  components and turns them into the ONNX input list.
- `protomotions/utils/export_utils.py:390` - `UnifiedPipelineModule` chains
  obs -> policy -> action processing for export.
- `deployment/export_bm_tracker_onnx.py:368-501` - orchestrates the export and
  emits the YAML sidecar.
- `examples/experiments/mimic/mlp.py:71-75` - the obs component config that
  determines which `dynamic_vars` are needed. If you change the experiment
  config, the ONNX input list will change accordingly.

---

## Migration checklist for the test app

1. Read `unified_pipeline.yaml` once and use it as the deployment contract:
   - `joint_names`, `body_names` for ordering;
   - `robot.anchor_body_name`, `anchor_body_index`;
   - `timing.control_dt` for motion-time stepping;
   - `motion.future_step_indices` for which future steps to sample.
2. Each control step, build the 10 input tensors above.
3. Run the ONNX session with these inputs.
4. Read `joint_pos_targets`, `stiffness_targets`, `damping_targets` directly
   and pass them to the simulator's PD controller. (No more host-side action
   processing.)
5. Cache `actions` for the next-step `historical_actions` buffer.

Drop `max_coords_obs`, `mimic_target_poses`, and `historical_previous_actions`
entirely from the host code - they are no longer needed.

---

## Caveats / config-dependent details

- **History depth.** `H = 1` is only correct for this exact mlp config
  (`previous_actions_factory(history_steps=1)`). If you ever switch to a
  config with deeper history, the `historical_actions` shape becomes
  `[B, H, N_dof]` with `H > 1`. Read `static_params.history_steps` from the
  obs component (or just use the shape ONNX advertises for that input).
- **`ground_heights`.** Folded out of the graph entirely if
  `root_height_obs=False` is ever set; the current factory default has it
  `True`, which is why it appears in your error.
- **`body_contacts`.** Currently absent because `observe_contacts=False`
  causes ONNX constant-folding to strip the input. If you flip that flag in a
  future training run, an extra `body_contacts: bool[B, N_b]` input will
  appear.
- **Future-frame XY offset.** The mimic component recenters all future targets
  to the current pelvis XY (see `mimic_control.py:158-198`). Replicate this
  in the host or your trajectories will appear translationally offset to the
  policy.

---

## Optional: keep the old contract instead

If you'd rather not change the host code, export an actor-only ONNX with
`export_ppo_actor` (`protomotions/utils/export_utils.py:212`). It will accept
`max_coords_obs`, `mimic_target_poses`, `previous_actions` (note: the third
key in the current code is `previous_actions`, not
`historical_previous_actions` - that older name only appears in
`protomotions/utils/debug_exporter.py` and one doc file). You will then need
to keep computing the obs in the host, which means tracking any future
changes to `compute_humanoid_max_coords_observations`,
`build_max_coords_target_poses`, and `compute_historical_actions_from_state`.
