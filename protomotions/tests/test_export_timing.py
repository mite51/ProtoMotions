# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for training vs. deployment cadence in both exporters."""

import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from protomotions.utils.export_timing import resolve_export_timing


def _config(fps=120, decimation=4):
    return {
        "simulator": SimpleNamespace(
            _target_="training.Simulator",
            sim=SimpleNamespace(fps=fps, decimation=decimation),
        ),
        "robot": SimpleNamespace(
            simulation_params=SimpleNamespace(
                mujoco=SimpleNamespace(fps=1000, decimation=20)
            )
        ),
    }


@pytest.fixture
def saved(tmp_path):
    training_path = tmp_path / "resolved_configs.pt"
    inference_path = tmp_path / "resolved_configs_inference.pt"
    torch.save(_config(), training_path)
    inference = _config(1000, 20)
    torch.save(inference, inference_path)
    return tmp_path / "last.ckpt", inference, inference_path


def test_training_wins_over_different_inference_cadence(saved):
    timing = resolve_export_timing(*saved, [1, 3])
    assert timing["basis"] == "training"
    assert timing["target_simulator"] is None
    assert timing["physics_dt"] == 1 / 120
    assert timing["decimation"] == 4
    assert timing["dt"] == timing["control_dt"] == 1 / 30
    assert timing["training"]["future_dt_seconds"] == [1 / 30, 0.1]
    path = saved[0].parent / "resolved_configs.pt"
    provenance = timing["training"]["provenance"]
    assert provenance["config_path"] == str(path.resolve())
    assert provenance["config_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_target_conversion_is_explicit_and_does_not_mutate_graph_config(saved, monkeypatch):
    factory = importlib.import_module("protomotions.simulator.factory")

    def convert(simulator, name, robot):
        simulator._target_ = name
        simulator.sim = robot.simulation_params.mujoco
        robot.converted = True
        return simulator

    monkeypatch.setattr(factory, "update_simulator_config_for_test", convert)
    timing = resolve_export_timing(*saved, [1, 3], target_simulator="mujoco")
    assert timing["basis"] == "target_simulator"
    assert timing["physics_dt"] == 0.001
    assert timing["decimation"] == 20
    assert timing["dt"] == timing["control_dt"] == 0.02
    assert timing["training"]["control_dt"] == 1 / 30
    assert timing["target_simulator"]["future_dt_seconds"] == [0.02, 0.06]
    assert timing["target_simulator"]["provenance"]["parameters_source"] == "saved_robot_preset"
    assert saved[1]["simulator"]._target_ == "training.Simulator"
    assert not hasattr(saved[1]["robot"], "converted")


@pytest.mark.parametrize("fps,decimation", [
    (None, 4), (0, 4), (-120, 4), (float("nan"), 4),
    (float("inf"), 4), (120, None), (120, 0), (120, -4),
    (120, 4.5), (120, float("inf")), (True, 4), (120, True),
])
def test_invalid_training_timing_never_falls_back_to_inference(saved, fps, decimation):
    torch.save(_config(fps, decimation), saved[0].parent / "resolved_configs.pt")
    with pytest.raises(ValueError, match="Invalid or missing"):
        resolve_export_timing(*saved, [1])


def test_missing_training_config_is_not_inferred_from_inference(saved):
    (saved[0].parent / "resolved_configs.pt").unlink()
    with pytest.raises(FileNotFoundError, match="Saved training config required"):
        resolve_export_timing(*saved, [1])


def test_missing_simulator_is_not_replaced_by_a_preset(saved):
    torch.save({}, saved[0].parent / "resolved_configs.pt")
    with pytest.raises(ValueError, match="Invalid or missing"):
        resolve_export_timing(*saved, [1])


@pytest.mark.parametrize("module_name", ["export_tracker_onnx", "export_bm_tracker_onnx"])
@pytest.mark.parametrize("target", [None, "mujoco"])
def test_sidecars_use_one_interval_and_preserve_non_timing_fields(saved, module_name, target):
    module = importlib.import_module(f"deployment.{module_name}")
    timing = resolve_export_timing(*saved, [1, 3], target_simulator=target)
    kwargs = dict(
        onnx_in_names=["current_dof_pos"],
        onnx_out_names=["actions", "joint_pos_targets", "stiffness_targets", "damping_targets"],
        onnx_name_to_key={"current_dof_pos": "current.dof_pos"},
        input_shapes={"current.dof_pos": [1, 1]},
        joint_names=["hip"], body_names=["pelvis"],
        stiffness=[100.0], damping=[10.0], pd_target_max_accel=None,
        anchor_body_name="pelvis", anchor_body_index=0, root_body_index=0,
        num_bodies=1, num_dofs=1, mjcf_path="robot.xml", timing=timing,
        future_step_indices=[1, 3], checkpoint=str(saved[0]),
    )
    if module_name == "export_bm_tracker_onnx":
        kwargs.update(obs_input_keys=["current.dof_pos"], actor_obs_configs={},
                      effort_limits=[50.0], action_scale=[0.5], pd_action_offset=[0.0],
                      control_type="BUILT_IN_PD")
    else:
        kwargs["obs_future_step_selections"] = {}
    content = yaml.safe_load(yaml.safe_dump(module._build_yaml(**kwargs)))
    assert content["dt"] == content["timing"]["dt"] == content["timing"]["control_dt"]
    assert content["dt"] == content["timing"]["physics_dt"] * content["timing"]["decimation"]
    assert content["motion"]["future_dt_seconds"] == [content["dt"], 3 * content["dt"]]
    assert content["control"]["stiffness"] == [100.0]
    assert content["control"]["damping"] == [10.0]
    assert content["robot"]["joint_names"] == ["hip"]
    assert content["robot"]["body_names"] == ["pelvis"]
    assert content["_runtime"]["onnx_in_names"] == ["current_dof_pos"]


@pytest.mark.needs_lfs
def test_bundled_smpl_saved_training_config_matches_yaml():
    directory = Path(__file__).resolve().parents[2] / "data/pretrained_models/motion_tracker/smpl"
    path = directory / "resolved_configs_inference.pt"
    resolved = torch.load(path, map_location="cpu", weights_only=False)
    timing = resolve_export_timing(directory / "last.ckpt", resolved, path, [1])
    saved_yaml = yaml.safe_load((directory / "resolved_configs.yaml").read_text())
    sim = saved_yaml["simulator"]["sim"]
    assert timing["training"]["physics_fps"] == float(sim["fps"]) == 120
    assert timing["decimation"] == int(sim["decimation"]) == 4
    assert timing["control_dt"] == 1 / 30
    assert timing["training"]["future_dt_seconds"] == [1 / 30]
