# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the complete SMPL hierarchical controller for the native C++ viewer.

Uses the mini-brain weights embedded in the actor checkpoint, saved observation
constants, action processing, and training cadence. No simulator is started.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from tensordict import TensorDict

from deployment.export_bm_tracker_onnx import MockContext
from protomotions.agents.ppo.hierarchical_model import HierarchicalPPOActor
from protomotions.components.pose_lib import build_body_ids_tensor
from protomotions.utils.export_timing import resolve_export_timing
from protomotions.utils.export_utils import (
    ActionExportModule, ObservationExportModule, UnifiedPipelineModule,
    _resolve_context_path,
)


def reconstruct(checkpoint: Path):
    """Reconstruct exactly the saved actor, including its frozen reflex network."""
    config_path = checkpoint.parent / "resolved_configs_inference.pt"
    if not config_path.exists():
        config_path = checkpoint.parent / "resolved_configs.pt"
    resolved = torch.load(config_path, map_location="cpu", weights_only=False)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    robot, env = resolved["robot"], resolved["env"]
    cfg = deepcopy(resolved["agent"].model.actor)
    if cfg._target_.rsplit(".", 1)[-1] != "HierarchicalPPOActor":
        raise ValueError("This exporter requires a HierarchicalPPOActor checkpoint")
    # Never depend on an external pretraining file which may have since changed.
    cfg.minibrain_checkpoint = None
    actor = HierarchicalPPOActor(cfg).eval()
    # Loading a trained policy must not invoke the training-time 0.01 rescale.
    actor._main_brain_last_layer_scaled = True
    body_names = list(robot.kinematic_info.body_names)
    joints = list(robot.kinematic_info.dof_names)
    if len(body_names) != 24 or len(joints) != 69 or cfg.minibrain_config.feature_dim not in (34, 46):
        raise ValueError("This viewer export requires the SMPL 24-body, 69-DOF, 34- or 46-feature layout")
    steps = env.control_components["mimic"].future_steps
    steps = list(range(1, steps + 1)) if isinstance(steps, int) else list(steps)
    contacts = build_body_ids_tensor(body_names, robot.contact_bodies, "cpu").tolist()
    history = max((v.get_params().get("history_steps", 1)
                   for k, v in env.observation_components.items() if k in actor.in_keys), default=1)
    mock = MockContext(1, len(joints), len(body_names), len(steps), robot.anchor_body_index,
                       history_steps=history, num_stamina_bodies=len(body_names)-1,
                       num_contact_bodies=len(contacts))
    components = {k: v for k, v in env.observation_components.items() if k in actor.in_keys}
    obs = ObservationExportModule(components, mock, "cpu").eval()
    keys = obs.get_input_keys()
    inputs = tuple(_resolve_context_path(k, mock) for k in keys)
    with torch.no_grad():
        values = obs(*inputs)
        actor(TensorDict(dict(zip(obs.get_output_keys(), values)), batch_size=[1]))
    actor.load_state_dict({k.removeprefix("_actor."): v for k, v in ckpt["model"].items()
                           if k.startswith("_actor.")}, strict=True)
    actor.minibrain.freeze()
    pipeline = UnifiedPipelineModule(obs, actor, ActionExportModule(env.action_config, "cpu"),
                                     list(actor.in_keys), "mean_action").eval()
    timing = resolve_export_timing(checkpoint, resolved, config_path, steps)
    return pipeline, keys, inputs, resolved, ckpt, timing, contacts, steps


def validate(pipeline, keys, inputs, session, replay=None):
    """Assert numerical parity on varied batches/stamina and optional C++ feeds."""
    errors = {o.name: 0.0 for o in session.get_outputs()}
    cases = []
    rng = torch.Generator().manual_seed(1203)
    for batch in (1, 2, 4):
        for strength in (1.0, 0.2, 0.0):
            values = []
            for key, value in zip(keys, inputs):
                value = value.repeat((batch,) + (1,) * (value.ndim - 1)).clone()
                if key == "body_stamina":
                    value.fill_(strength)
                elif value.dtype == torch.float32:
                    value += 0.03 * torch.randn(value.shape, generator=rng)
                    if "rot" in key:
                        value = torch.nn.functional.normalize(value, dim=-1)
                values.append(value)
            cases.append((f"batch_{batch}_stamina_{strength}", values))
    if replay:
        data = np.load(replay)
        # The C++ parity dump stores in_<onnx_name> with a leading time axis.
        count = len(data["out_actions"])
        live_names = {x.name for x in session.get_inputs()}
        for step in range(count):
            values = []
            for key, template in zip(keys, inputs):
                name = key.replace(".", "_")
                if name in live_names:
                    values.append(torch.from_numpy(data["in_" + name][step]).to(template.dtype))
                else:
                    # ONNX pruned this unused binding (e.g. disabled contacts).
                    values.append(template)
            cases.append((f"cpp_tick_{step}", values))
    with torch.no_grad():
        for label, values in cases:
            expected = pipeline(*values)
            feed = {k.replace(".", "_"): v.numpy() for k, v in zip(keys, values)}
            actual = session.run(None, {x.name: feed[x.name] for x in session.get_inputs()})
            for info, ref, result in zip(session.get_outputs(), expected, actual):
                np.testing.assert_allclose(result, ref.numpy(), atol=1e-4, rtol=1e-4,
                                           err_msg=f"{label}: {info.name}")
                if not np.isfinite(result).all():
                    raise AssertionError(f"Nonfinite output in {label}")
                errors[info.name] = max(errors[info.name], float(np.max(np.abs(result-ref.numpy()))))
                if label.startswith("cpp_tick_"):
                    recorded = data["out_" + info.name][int(label.removeprefix("cpp_tick_"))]
                    np.testing.assert_allclose(recorded, ref.numpy(), atol=1e-4, rtol=1e-4,
                                               err_msg=f"C++ runtime {label}: {info.name}")
                    errors[info.name] = max(errors[info.name],
                                           float(np.max(np.abs(recorded-ref.numpy()))))
    return {"cases": len(cases), "max_absolute_error": errors}


class LocalController(torch.nn.Module):
    """Standalone reflex API: 34 features per joint -> radian PD targets."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, joint_features):
        out = self.model(joint_features, q_current=joint_features[..., :3])
        return out["u_pd"], out["delta_u"]


def export_local(actor, output):
    local = LocalController(actor.minibrain).eval()
    feat_dim = actor.minibrain.config.feature_dim
    features = torch.randn(1, 23, feat_dim) * 0.1
    features[..., 26:29] = 1.0
    features[..., 29:31] = 0.8
    path = output / "shared_minibrain.onnx"
    names = ["joint_features", "joint_pos_targets", "delta_u"]
    torch.onnx.export(local, (features,), str(path), input_names=names[:1], output_names=names[1:],
                      dynamic_axes={n: {0: "batch_size"} for n in names},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    error = 0.0
    with torch.no_grad():
        for batch in (1, 2, 4):
            x = features.repeat(batch, 1, 1)
            x[:, ::2, 26] = 0  # Explicitly verify individual locked axes.
            x[:, 1::2, 27:29] = 0.2
            expected = local(x)
            actual = session.run(None, {"joint_features": x.numpy()})
            for ref, result in zip(expected, actual):
                np.testing.assert_allclose(result, ref.numpy(), atol=1e-5, rtol=1e-5)
                error = max(error, float(np.max(np.abs(result-ref.numpy()))))
            np.testing.assert_array_equal(actual[1][:, ::2, 0], 0)
            np.testing.assert_array_equal(actual[0].reshape(batch, 23, 3)[:, ::2, 0],
                                          x.numpy()[:, ::2, 0])
    return dict(cases=3, locked_axes_verified=True, max_absolute_error=error,
                onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def export(checkpoint: Path, output: Path, replay=None):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    pipeline, keys, inputs, resolved, ckpt, timing, contacts, steps = reconstruct(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "unified_pipeline.onnx"
    names = [k.replace(".", "_") for k in keys]
    outputs = ["actions", "joint_pos_targets", "stiffness_targets", "damping_targets"]
    torch.onnx.export(pipeline, inputs, str(path), input_names=names, output_names=outputs,
                      dynamic_axes={n: {0: "batch_size"} for n in names+outputs},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    report = validate(pipeline, keys, inputs, session, replay)
    actor = pipeline.policy_module
    report["standalone_minibrain"] = export_local(actor, output)
    robot, env = resolved["robot"], resolved["env"]
    body_names = list(robot.kinematic_info.body_names)
    joints = list(robot.kinematic_info.dof_names)
    spec = {n: (k, list(v.shape)) for n, k, v in zip(names, keys, inputs)}
    actual_inputs = []
    for info in session.get_inputs():
        key, shape = spec[info.name]
        item = dict(name=info.name, key=key, shape=shape)
        if key == "historical.actions":
            item.update(kind="last_actions", output_key="actions")
        actual_inputs.append(item)
    ac = env.action_config
    sidecar = dict(
        type="unified_pipeline", architecture="hierarchical_shared_joint_minibrain",
        dt=timing["control_dt"], timing=timing,
        body_names=body_names, joint_names=joints,
        contact_body_names=[body_names[i] for i in contacts],
        policy_inputs=actual_inputs,
        policy_outputs=[dict(name=n, key=n, shape=[1, len(joints)]) for n in outputs],
        control=dict(
            stiffness=ac["stiffness"].tolist(), damping=ac["damping"].tolist(),
            effort_limits=[float(robot.control.control_info[j].effort_limit) for j in joints],
            action_scale=ac["pd_action_scale"].tolist(),
            pd_action_offset=ac["pd_action_offset"].tolist(),
            action_transform=ac["action_transform"],
            clamp_value=ac.get("clamp_value", 1.0),
            pd_target_max_accel=getattr(resolved["simulator"], "pd_target_max_accel", None)),
        motion=dict(future_step_indices=steps),
        robot=dict(mjcf_path=robot.asset.asset_file_name, body_names=body_names, joint_names=joints),
        metadata=dict(checkpoint=str(checkpoint.resolve()), epoch=ckpt.get("epoch"),
                      step_count=ckpt.get("step_count"), minibrain_weights="embedded_in_actor_checkpoint",
                      main_residual_radians=actor.config.residual_pose_scale,
                      local_residual_radians=actor.minibrain.config.max_residual_offset),
    )
    (output / "unified_pipeline.yaml").write_text(yaml.safe_dump(sidecar, sort_keys=False))
    report.update(sidecar["metadata"])
    report.update(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  main_parameters=sum(p.numel() for p in actor.mu.parameters()),
                  minibrain_parameters=sum(p.numel() for p in actor.minibrain.parameters()))
    (output / "export_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", type=Path, help="Optional C++ parity NPZ")
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.replay)
