# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Run closed-loop inference in the ProtoMotions simulator using an exported
``unified_pipeline.onnx`` instead of a PyTorch checkpoint.

The ONNX bundle produced by ``deployment/export_bm_tracker_onnx.py`` (or
``protomotions.utils.export_utils.export_unified_pipeline``) is a complete
``Context -> Observations -> Policy -> ActionProcessing`` graph.  Its inputs
are raw context attribute paths (e.g. ``current.rigid_body_pos``,
``mimic.future_rot``, ``historical.actions``, ``ground_heights``) and its
outputs include the raw policy ``actions`` tensor.  Because ProtoMotions does
not apply observation normalization outside the actor, the ONNX is
self-contained and its ``actions`` output can be fed straight into
``env.step(action)``.

No agent / model checkpoint is loaded; we only need the ONNX file, its YAML
sidecar (for the input-name -> context-path map), and
``resolved_configs_inference.pt`` so the simulator/env are configured exactly
like training.

Example
-------
::

    python protomotions/inference_onnx_agent.py \\
        --onnx results/smpl_amass_flat_v2/compiled_models/unified_pipeline.onnx \\
        --simulator isaacgym \\
        --num-envs 16 \\
        --motion-file data/motion_for_trackers/soma23_bones_seed_mini.pt
"""


def create_parser():
    """Create and configure the argument parser for ONNX inference."""
    parser = argparse.ArgumentParser(
        description="Run trained policy via ONNX in the ProtoMotions simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="Path to exported unified_pipeline.onnx",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Simulator to use (e.g., 'isaacgym', 'isaaclab', 'newton', 'genesis', 'mujoco')",
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default=None,
        help=(
            "Directory containing resolved_configs_inference.pt. "
            "Defaults to parent of the ONNX file's directory."
        ),
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments to run",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        default=None,
        help="Path to motion file. If not provided, uses the one from resolved configs.",
    )
    parser.add_argument(
        "--scenes-file",
        type=str,
        default=None,
        help="Path to scenes file (optional)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run simulation in headless mode",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after this many environment steps (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides in format key=value (e.g., env.max_episode_length=5000)",
    )

    return parser


# Parse arguments first (argparse is safe, doesn't import torch).
import argparse  # noqa: E402

parser = create_parser()
args, unknown_args = parser.parse_known_args()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch.
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import everything else, including torch.
import logging  # noqa: E402
import sys  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

from protomotions.envs.context_paths import resolve_path  # noqa: E402
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger(__name__)


# ONNX element-type -> numpy dtype.  Generous mapping so we stay robust against
# non-float inputs (e.g. masked-mimic uses bool masks that are emitted as
# int/bool tensors).  Casting respects the actual input declaration so we don't
# silently coerce types and break model semantics.
_ORT_TYPE_TO_NP_DTYPE = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16,
    "tensor(bool)": np.bool_,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
}


def _to_numpy_for_onnx(value: Any, ort_dtype: str) -> np.ndarray:
    """Convert a context value (typically a torch.Tensor) to a numpy array
    matching the ONNX input's declared dtype.
    """
    target = _ORT_TYPE_TO_NP_DTYPE.get(ort_dtype, np.float32)
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.dtype != target:
        arr = arr.astype(target)
    # ORT requires contiguous arrays for some EPs; cheap insurance.
    return np.ascontiguousarray(arr)


def _trim_to_expected_shape(
    arr: np.ndarray,
    expected_shape: list,
    name: str,
    context_path: str,
) -> np.ndarray:
    """Slice non-batch dims of ``arr`` down to the ONNX's static dims.

    ONNX shapes come from ``session.get_inputs()[i].shape`` and contain a mix
    of ``int`` (static) and ``str`` (dynamic, e.g. ``"batch_size"``).  The
    env's runtime tensors may have larger non-batch dims than the policy was
    trained with — most commonly because the state-history buffer stores more
    timesteps than the policy actually consumes, or because the mimic buffer
    holds extra future steps for other observations.  We trim by slicing
    ``[:N]`` along each over-sized static dim (history index 0 is the most
    recent step, so this keeps the freshest data).
    """
    if len(arr.shape) != len(expected_shape):
        raise RuntimeError(
            f"ONNX input '{name}' (ctx.{context_path}) has rank "
            f"{len(arr.shape)} ({list(arr.shape)}) but the model expects rank "
            f"{len(expected_shape)} ({expected_shape}).  The exported policy "
            "and the live env shapes are incompatible."
        )

    slices = []
    for axis, (got, want) in enumerate(zip(arr.shape, expected_shape)):
        if isinstance(want, str) or want is None:
            slices.append(slice(None))
            continue
        if got < want:
            raise RuntimeError(
                f"ONNX input '{name}' (ctx.{context_path}) axis {axis} has "
                f"size {got} but the model requires at least {want}."
            )
        if got > want:
            slices.append(slice(0, want))
        else:
            slices.append(slice(None))

    if all(s == slice(None) for s in slices):
        return arr
    return np.ascontiguousarray(arr[tuple(slices)])


def main():
    global parser, args
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    yaml_path = onnx_path.with_suffix(".yaml")
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"YAML sidecar not found next to ONNX: expected {yaml_path}. "
            "The sidecar is required to map ONNX inputs to context paths."
        )

    # Default configs dir: parent of the ONNX file's directory
    # (i.e. results/<exp>/compiled_models/foo.onnx -> results/<exp>/).
    configs_dir = Path(args.configs_dir) if args.configs_dir else onnx_path.parent.parent
    resolved_configs_path = configs_dir / "resolved_configs_inference.pt"
    assert resolved_configs_path.exists(), (
        f"Could not find resolved_configs_inference.pt at {resolved_configs_path}. "
        "Pass --configs-dir to point at the training output directory."
    )

    # ------------------------------------------------------------------
    # 1. Load resolved configs (matches inference_agent.py)
    # ------------------------------------------------------------------
    log.info(f"Loading resolved configs from {resolved_configs_path}")
    resolved_configs = torch.load(
        resolved_configs_path, map_location="cpu", weights_only=False
    )

    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config = resolved_configs["env"]

    # Switch simulators if requested.
    current_simulator = simulator_config._target_.split(".")[-3]
    if args.simulator != current_simulator:
        log.info(
            f"Switching simulator from '{current_simulator}' (training) to "
            f"'{args.simulator}' (inference)"
        )
        from protomotions.simulator.factory import update_simulator_config_for_test

        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )

    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    # CLI runtime overrides
    if args.num_envs is not None:
        log.info(f"CLI override: num_envs = {args.num_envs}")
        simulator_config.num_envs = args.num_envs

    if args.motion_file is not None:
        log.info(f"CLI override: motion_file = {args.motion_file}")
        motion_lib_config.motion_file = args.motion_file

    if args.scenes_file is not None:
        log.info(f"CLI override: scenes_file = {args.scenes_file}")
        scene_lib_config.scene_file = args.scenes_file

    if args.headless is not None:
        log.info(f"CLI override: headless = {args.headless}")
        simulator_config.headless = args.headless

    from protomotions.utils.config_utils import (
        apply_config_overrides,
        parse_cli_overrides,
    )

    cli_overrides = parse_cli_overrides(args.overrides) if args.overrides else None
    if cli_overrides:
        apply_config_overrides(
            cli_overrides,
            env_config,
            simulator_config,
            robot_config,
            resolved_configs.get("agent"),
            terrain_config,
            motion_lib_config,
            scene_lib_config,
        )

    # ------------------------------------------------------------------
    # 2. Fabric setup (mirror inference_agent.py — CPU + Windows-friendly)
    # ------------------------------------------------------------------
    accelerator = "cpu"
    fabric_kwargs = dict(
        accelerator=accelerator,
        devices=1,
        num_nodes=1,
        loggers=[],
        callbacks=[],
    )
    if sys.platform == "win32":
        from lightning.fabric.strategies import SingleDeviceStrategy

        fabric_kwargs["strategy"] = SingleDeviceStrategy(device="cpu")
    fabric_config = FabricConfig(**fabric_kwargs)
    fabric: Fabric = Fabric(**asdict(fabric_config))
    fabric.launch()

    simulator_extra_params: Dict[str, Any] = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(fabric.device)}
        app_launcher = AppLauncher(app_launcher_flags)
        simulator_extra_params["simulation_app"] = app_launcher.app

    # ------------------------------------------------------------------
    # 3. Build env + simulator (no agent / no checkpoint)
    # ------------------------------------------------------------------
    from protomotions.simulator.base_simulator.utils import (
        convert_friction_for_simulator,
    )

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    from protomotions.utils.component_builder import build_all_components

    save_dir_for_weights = (
        getattr(env_config, "save_dir", None)
        if hasattr(env_config, "save_dir")
        else None
    )
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=save_dir_for_weights,
        **simulator_extra_params,
    )

    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]

    from protomotions.envs.base_env.env import BaseEnv

    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )

    # Optional one-shot scene snapshot. ``PPP_SCENE_DUMP_PATH=<file.usda>``
    # flattens the live IsaacLab USD stage to ASCII USD after the env is
    # built but before the first inference step, so we can diff its
    # PhysX-attribute set against OVProtomotions' ``_runtime_scene.usda``
    # to find any unidentified scene-config mismatch causing the
    # step-1+ integrator drift documented in
    # ``docs/chirality_debugging_2026-06-18.md``.
    import os as _os_scene
    scene_dump_path = _os_scene.environ.get("PPP_SCENE_DUMP_PATH")
    if scene_dump_path:
        try:
            import omni.usd  # type: ignore
            from pxr import UsdUtils  # type: ignore

            stage = omni.usd.get_context().get_stage()
            UsdUtils.FlattenLayerStack(stage).Export(scene_dump_path)
            log.warning(
                "Scene dump: wrote flattened IsaacLab USD stage to %s",
                scene_dump_path,
            )
        except Exception:
            # Fall back to a non-flattened export so we still get a
            # comparable scene snapshot even if FlattenLayerStack chokes
            # on IsaacLab's composition arcs.
            try:
                import omni.usd  # type: ignore

                stage = omni.usd.get_context().get_stage()
                stage.Export(scene_dump_path)
                log.warning(
                    "Scene dump: wrote IsaacLab stage to %s "
                    "(non-flattened — composition arcs preserved).",
                    scene_dump_path,
                )
            except Exception as e:
                log.exception("Scene dump failed: %s", e)

    # ------------------------------------------------------------------
    # 4. Load ONNX + YAML _runtime mapping
    # ------------------------------------------------------------------
    log.info(f"Loading ONNX model from {onnx_path}")
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )

    with open(yaml_path, "r") as f:
        yaml_meta = yaml.safe_load(f)

    runtime_meta = yaml_meta.get("_runtime", {})
    onnx_name_to_in_key: Dict[str, str] = dict(
        runtime_meta.get("onnx_name_to_in_key", {})
    )
    if not onnx_name_to_in_key:
        raise ValueError(
            f"YAML sidecar {yaml_path} is missing _runtime.onnx_name_to_in_key; "
            "cannot route ONNX inputs to context paths."
        )

    # Authoritative input list comes from the session itself; some exports
    # rename or reorder inputs.
    session_inputs = session.get_inputs()
    session_in_names = [inp.name for inp in session_inputs]
    session_in_dtypes = {inp.name: inp.type for inp in session_inputs}
    # Expected per-axis shape (mix of int and str; str = dynamic dim).
    session_in_shapes = {inp.name: list(inp.shape) for inp in session_inputs}

    missing = [n for n in session_in_names if n not in onnx_name_to_in_key]
    if missing:
        raise ValueError(
            f"ONNX inputs {missing} have no entry in _runtime.onnx_name_to_in_key "
            f"(YAML: {yaml_path}).  Re-export the model or update the YAML."
        )

    out_names = [out.name for out in session.get_outputs()]
    if "actions" not in out_names:
        raise ValueError(
            f"ONNX outputs {out_names} do not include 'actions'. "
            "Expected a unified_pipeline export with 'actions' output."
        )

    log.info(f"ONNX inputs ({len(session_in_names)}):")
    for name in session_in_names:
        log.info(
            f"  {name:40s} -> ctx.{onnx_name_to_in_key[name]} "
            f"({session_in_dtypes[name]}, shape={session_in_shapes[name]})"
        )
    log.info(f"ONNX outputs: {out_names}")

    # ------------------------------------------------------------------
    # 5. Closed-loop inference
    # ------------------------------------------------------------------
    done_indices = None
    step = 0
    metric_sums: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}

    max_steps = args.max_steps
    if max_steps is None:
        log.info("Running ONNX inference loop... (Ctrl+C to stop)")
    else:
        log.info(f"Running ONNX inference loop for {max_steps} steps...")

    # Optional per-step parity dump. Set PPP_PARITY_DUMP_PATH=<file.npz>
    # to capture every ONNX input, output, and eval metric so the value
    # can be diffed against an OVProtomotions run on the same motion.
    # Stops at the first env-reset boundary so the trace only covers a
    # single episode (the inference loop already calls env.reset on
    # done_indices each iter).
    import os as _os_dump
    parity_dump_path = _os_dump.environ.get("PPP_PARITY_DUMP_PATH")
    parity_records: list = []
    if parity_dump_path:
        log.warning(
            "Parity dump enabled: writing every ONNX input/output to %s",
            parity_dump_path,
        )

    try:
        while True:
            env.reset(done_indices)
            ctx = env.context

            ort_inputs: Dict[str, np.ndarray] = {}
            for name in session_in_names:
                context_path = onnx_name_to_in_key[name]
                value = resolve_path(ctx, context_path)
                if value is None:
                    raise RuntimeError(
                        f"Context path '{context_path}' (for ONNX input '{name}') "
                        "resolved to None.  Check that the env is configured "
                        "to populate this field (e.g. mimic/historical views)."
                    )
                arr = _to_numpy_for_onnx(value, session_in_dtypes[name])
                arr = _trim_to_expected_shape(
                    arr, session_in_shapes[name], name, context_path
                )
                ort_inputs[name] = arr

            outputs = session.run(out_names, ort_inputs)
            actions_np = outputs[out_names.index("actions")]
            action = torch.as_tensor(
                actions_np, device=env.device, dtype=torch.float32
            )

            obs, rewards, dones, terminated, extras = env.step(action)

            if "eval_values" in extras:
                for k, v in extras["eval_values"].items():
                    val = float(v.mean().item())
                    metric_sums[k] = metric_sums.get(k, 0.0) + val
                    metric_counts[k] = metric_counts.get(k, 0) + 1

            if parity_dump_path:
                rec: Dict[str, Any] = {
                    f"in_{k}": np.asarray(v, dtype=np.float32).copy()
                    for k, v in ort_inputs.items()
                }
                for i, oname in enumerate(out_names):
                    rec[f"out_{oname}"] = np.asarray(
                        outputs[i], dtype=np.float32
                    ).copy()
                if "eval_values" in extras:
                    for k, v in extras["eval_values"].items():
                        rec[f"eval_{k}"] = np.float32(v.mean().item())
                # Raw DOF state in kinematic_info order. The ONNX consumes
                # body-space (rigid_body_*) tensors, but the policy commands
                # joint targets, so any divergence between simulators shows
                # up most cleanly here. ``current.dof_pos`` / ``current.dof_vel``
                # exist on ``CurrentStateView`` (envs/context_views.py).
                rec["dump_dof_pos"] = _to_numpy_for_onnx(
                    resolve_path(ctx, "current.dof_pos"), "tensor(float)"
                ).copy()
                rec["dump_dof_vel"] = _to_numpy_for_onnx(
                    resolve_path(ctx, "current.dof_vel"), "tensor(float)"
                ).copy()
                parity_records.append(rec)

            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            step += 1

            if max_steps is not None and step >= max_steps:
                log.info(f"Reached --max-steps={max_steps}, stopping.")
                break

            # If dumping, stop on the first done to keep the trace
            # confined to a single episode (no reset discontinuities).
            if parity_dump_path and done_indices.numel() > 0:
                log.warning(
                    "Parity dump: stopping at step %d on first done (single-episode trace).",
                    step,
                )
                break
    except KeyboardInterrupt:
        print(f"\nStopped after {step} steps.")
    finally:
        if parity_dump_path and parity_records:
            save_dict: Dict[str, np.ndarray] = {}
            keys = list(parity_records[0].keys())
            for key in keys:
                arrs = [r[key] for r in parity_records]
                save_dict[key] = np.stack(arrs, axis=0)
            save_dict["meta_step_count"] = np.int64(len(parity_records))
            save_dict["meta_motion_file"] = np.array(
                str(motion_lib_config.motion_file), dtype=object
            )
            save_dict["meta_simulator"] = np.array(args.simulator, dtype=object)
            np.savez(parity_dump_path, **save_dict)
            log.warning(
                "Parity dump: wrote %d records to %s (keys=%s).",
                len(parity_records),
                parity_dump_path,
                sorted(k for k in save_dict.keys() if not k.startswith("meta_")),
            )

        if metric_counts:
            print("Average metrics:")
            for k in sorted(metric_counts.keys()):
                avg = metric_sums[k] / metric_counts[k]
                print(f"  {k}: {avg:.4f}")

        if hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()


if __name__ == "__main__":
    main()
