# Tracker export timing

Both `export_tracker_onnx.py` and `export_bm_tracker_onnx.py` use the checkpoint
directory's **`resolved_configs.pt` training simulator** for timing by default.
The actor, gains, body/joint order, and graph declarations still come from the
selected export configuration (`resolved_configs_inference.pt` when available).
An inference configuration is not evidence of the original training cadence.
Missing or invalid training timing stops export; neither exporter guesses
1000 Hz physics / 50 Hz control.

For an intentional simulator conversion, pass `--target-simulator mujoco`
(or another supported simulator). This uses the existing simulator conversion
on copies of the export configs. It selects deployment timing and records it
separately from saved training timing. A target preset is a deployment choice;
it does not establish the policy's training frequency or validate its behavior
at a different cadence. Conversion errors stop export.

The sidecar timing contract is:

- `timing.basis`: `training` by default, or `target_simulator` for an explicit
  conversion.
- `timing.training`: cadence from the saved training simulator, including its
  simulator class and config path, SHA-256, and source fields.
- `timing.target_simulator`: converted cadence and conversion provenance,
  including whether the parameters came from a saved robot preset or current
  simulator class defaults; `null` when no conversion was requested.
- `timing.export_config`: identity of the configuration used to build the graph.
- Top-level `dt` and `timing.dt` both equal `timing.control_dt`.
- `physics_dt = 1 / fps`; `control_dt = physics_dt * decimation`.
- `motion.future_dt_seconds[i] = motion.future_step_indices[i] * control_dt`.
  No decimal rounding is applied. Each cadence record also contains its own
  future offsets for comparison.

For the bundled `data/pretrained_models/motion_tracker/smpl/last.ckpt`, the
adjacent saved training binary and YAML both specify IsaacLab at `fps=120`,
`decimation=4`, with one future frame. Default export therefore declares
`physics_dt=1/120`, `dt=control_dt=1/30`, and `future_dt_seconds=[1/30]`.
The older exporter's 1000/50 values came from its automatic MuJoCo conversion.
That conversion also mutated the loaded simulator config in place.

The local `results/smpl_amass_flat` saved training config independently records
the same 120/30 cadence, while its existing compiled sidecar records 1000/50.
A fresh BM export of that checkpoint was validated against PyTorch in a separate
test output directory. Unity's older sidecar identifies
`results/smpl_amass_flat_v2/last.ckpt`; that exact directory was not found locally.
The verified configs above must not be treated as proof of the missing v2
checkpoint's historical training cadence.

This change does not replace existing sidecars. Unity's temporary behavior
remains to use scene physics FPS/decimation and derive lookahead from each
future-frame index times the scene policy interval, while parsing/checking
sidecar timing for consistency. Keep the original gains, ordering, and graph
declarations. The quick model's historical training configuration remains
unverified; its walk/jab failure and zero contact inputs (TD-15b) are a separate
investigation, not evidence that its 120/30 sidecar is incorrect.
