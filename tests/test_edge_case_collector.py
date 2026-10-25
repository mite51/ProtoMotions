# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for protomotions.envs.diagnostics.edge_case_collector."""

import json
from pathlib import Path

import pytest
import torch

from protomotions.envs.diagnostics.edge_case_collector import (
    EdgeCaseCollector,
    EdgeCaseCollectorConfig,
    EdgeCaseLevel,
)


def _make_collector(num_envs: int = 4, window_length: int = 8) -> EdgeCaseCollector:
    cfg = EdgeCaseCollectorConfig(
        enabled=True,
        window_length=window_length,
        max_cases_per_level=64,
        num_joints=23,
        dofs_per_joint=3,
        num_bodies=24,
        level1_vel_err_threshold=8.0,
        level2_tracking_err_threshold=0.35,
        level3_tilt_threshold=0.6,
        contact_force_threshold=50.0,
        collect_on_termination_only=False,
        cooldown_steps=16,
    )
    return EdgeCaseCollector(config=cfg, num_envs=num_envs, device="cpu")


def test_level1_analytical_load_event():
    """
    Synthetic Level 1 event: high vel_pred_error (> 8.0 rad/s) with zero contact force
    is classified into the Level 1 bucket.
    """
    collector = _make_collector(num_envs=2, window_length=8)
    E, J, D, B = 2, 23, 69, 24

    vel_pred_error = torch.zeros((E, J, 3))
    # Env 1 has a high velocity prediction error spike on joint 12 in free air
    vel_pred_error[1, 12, 0] = 12.5
    contact_forces = torch.zeros((E, B))

    info = collector.record_step(
        dof_pos=torch.zeros((E, D)),
        q_des=torch.zeros((E, D)),
        vel_pred_error=vel_pred_error,
        contact_force_magnitudes=contact_forces,
        terminated=torch.tensor([False, False]),
    )

    assert int(info["levels"][0].item()) == int(EdgeCaseLevel.NONE)
    assert int(info["levels"][1].item()) == int(EdgeCaseLevel.LEVEL_1_ANALYTICAL_LOAD)

    level1_cases = collector.get_level_bucket(EdgeCaseLevel.LEVEL_1_ANALYTICAL_LOAD)
    assert len(level1_cases) == 1
    assert level1_cases[0]["env_id"] == 1
    assert level1_cases[0]["worst_joint_idx"] == 12
    assert pytest.approx(level1_cases[0]["max_vel_pred_err"], rel=1e-5) == 12.5
    assert level1_cases[0]["max_contact_force"] == 0.0


def test_level2_minibrain_tracking_event_joint5():
    """
    Synthetic Level 2 event: low contact force, low vel_pred_error, high |dof_pos - q_des|
    on joint 5 is classified into Level 2 bucket with worst_joint_idx == 5.
    """
    collector = _make_collector(num_envs=3, window_length=8)
    E, J, D, B = 3, 23, 69, 24

    # Record a pre-failure step first so frame K-2 has distinct state
    collector.record_step(
        root_pos=torch.full((E, 3), 1.0),
        dof_pos=torch.full((E, D), 0.1),
        q_des=torch.full((E, D), 0.1),
        vel_pred_error=torch.zeros((E, J, 3)),
        contact_force_magnitudes=torch.zeros((E, B)),
        motion_ids=torch.tensor([10, 20, 30]),
        motion_times=torch.tensor([0.5, 1.5, 2.5]),
    )

    # Step 2: Env 2 has high tracking error on joint 5 (DOFs 15..17)
    dof_pos = torch.zeros((E, D))
    q_des = torch.zeros((E, D))
    q_des[2, 5 * 3 + 1] = 0.85  # > 0.35 rad threshold on joint 5

    vel_pred_error = torch.full((E, J, 3), 0.2)  # well below 8.0 rad/s
    contact_forces = torch.full((E, B), 5.0)  # well below 50.0 N

    info = collector.record_step(
        root_pos=torch.full((E, 3), 2.0),
        dof_pos=dof_pos,
        q_des=q_des,
        vel_pred_error=vel_pred_error,
        contact_force_magnitudes=contact_forces,
        motion_ids=torch.tensor([10, 20, 30]),
        motion_times=torch.tensor([0.533, 1.533, 2.533]),
    )

    assert int(info["levels"][2].item()) == int(EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING)
    assert int(info["worst_joint_idx"][2].item()) == 5

    level2_cases = collector.get_level_bucket(EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING)
    assert len(level2_cases) == 1
    case = level2_cases[0]
    assert case["worst_joint_idx"] == 5
    assert pytest.approx(case["max_joint_tracking_err"], rel=1e-5) == 0.85
    assert len(collector.get_level_bucket(EdgeCaseLevel.LEVEL_1_ANALYTICAL_LOAD)) == 0
    assert len(collector.get_level_bucket(EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE)) == 0


def test_level3_mainbrain_balance_and_impact_events():
    """
    Synthetic Level 3 events:
      - Case A: All joints track q_des well (<= 0.35), low contact force, but terminated=True.
      - Case B: High external contact impact (> 50.0 N) and terminated=True.
    Both are classified into Level 3 bucket.
    """
    collector = _make_collector(num_envs=3, window_length=8)
    E, J, D, B = 3, 23, 69, 24

    dof_pos = torch.zeros((E, D))
    q_des = torch.zeros((E, D))
    # Env 0: small tracking error (0.05 rad <= 0.35), terminated=True
    q_des[0, 0] = 0.05
    # Env 1: high contact impact (180 N > 50 N) + large tracking error, terminated=True
    q_des[1, 3] = 0.75
    contact_forces = torch.zeros((E, B))
    contact_forces[1, 10] = 180.0

    terminated = torch.tensor([True, True, False])

    info = collector.record_step(
        dof_pos=dof_pos,
        q_des=q_des,
        vel_pred_error=torch.zeros((E, J, 3)),
        contact_force_magnitudes=contact_forces,
        terminated=terminated,
    )

    assert int(info["levels"][0].item()) == int(EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE)
    assert int(info["levels"][1].item()) == int(EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE)
    assert int(info["levels"][2].item()) == int(EdgeCaseLevel.NONE)

    level3_cases = collector.get_level_bucket(EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE)
    assert len(level3_cases) == 2
    assert level3_cases[0]["terminated"] is True
    assert level3_cases[1]["max_contact_force"] == pytest.approx(180.0)


def test_chronological_window_ordering_after_wraparound():
    """
    Verify that after recording more than K steps (e.g. 13 steps with K=8), the extracted
    window contains the exact last K steps in strictly increasing chronological order
    [5, 6, 7, 8, 9, 10, 11, 12].
    """
    K = 8
    collector = _make_collector(num_envs=2, window_length=K)
    E, J, D, B = 2, 23, 69, 24

    total_steps = 13
    for step in range(total_steps):
        root_pos = torch.full((E, 3), float(step))
        motion_times = torch.full((E,), float(step) * 0.033)
        terminated = torch.tensor([step == (total_steps - 1), False])

        collector.record_step(
            root_pos=root_pos,
            dof_pos=torch.full((E, D), float(step)),
            q_des=torch.full((E, D), float(step)),
            vel_pred_error=torch.zeros((E, J, 3)),
            contact_force_magnitudes=torch.zeros((E, B)),
            motion_ids=torch.full((E,), step, dtype=torch.long),
            motion_times=motion_times,
            terminated=terminated,
        )

    level3_cases = collector.get_level_bucket(EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE)
    assert len(level3_cases) == 1
    case = level3_cases[0]

    expected_steps = torch.arange(total_steps - K, total_steps, dtype=torch.float32)
    assert torch.allclose(case["root_pos"][:, 0], expected_steps)
    assert torch.allclose(case["dof_pos"][:, 0], expected_steps)
    assert torch.equal(
        case["motion_ids"],
        torch.arange(total_steps - K, total_steps, dtype=torch.long),
    )


def test_sampling_and_save_reload_roundtrip(tmp_path: Path):
    """
    Test sample_level2_minibrain_batch, sample_level3_mainbrain_resets, get_summary,
    and save_all / load_all round-trip.
    """
    K = 8
    collector = _make_collector(num_envs=3, window_length=K)
    E, J, D, B = 3, 23, 69, 24

    # Record 5 normal steps so pre-failure frames have distinct, verifiable timestamps
    for s in range(5):
        collector.record_step(
            root_pos=torch.full((E, 3), float(s)),
            dof_pos=torch.full((E, D), float(s) * 0.01),
            q_des=torch.full((E, D), float(s) * 0.01),
            vel_pred_error=torch.zeros((E, J, 3)),
            contact_force_magnitudes=torch.zeros((E, B)),
            motion_ids=torch.tensor([1, 2, 3], dtype=torch.long),
            motion_times=torch.full((E,), float(s) * 0.1),
        )

    # Step 5 (6th recorded step):
    # Env 0 -> Level 1 (vel_pred_error spike on joint 2)
    # Env 1 -> Level 2 (tracking error spike on joint 5)
    # Env 2 -> Level 3 (terminated=True with good tracking)
    vel_err = torch.zeros((E, J, 3))
    vel_err[0, 2, 0] = 15.0

    dof_pos = torch.full((E, D), 0.05)
    q_des = torch.full((E, D), 0.05)
    q_des[1, 5 * 3] = 0.95  # tracking error = 0.90 > 0.35 on joint 5

    collector.record_step(
        root_pos=torch.full((E, 3), 5.0),
        dof_pos=dof_pos,
        q_des=q_des,
        vel_pred_error=vel_err,
        contact_force_magnitudes=torch.zeros((E, B)),
        motion_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        motion_times=torch.full((E,), 0.5),
        terminated=torch.tensor([False, False, True]),
    )

    # Test sample_level2_minibrain_batch: should return state from frame before failure (step 4)
    # and target q_des from failure step (step 5)
    mb_batch = collector.sample_level2_minibrain_batch(batch_size=4, device="cpu")
    assert mb_batch["root_pos"].shape == (4, 3)
    assert mb_batch["root_rot"].shape == (4, 4)
    assert mb_batch["dof_pos"].shape == (4, D)
    assert mb_batch["q_des"].shape == (4, D)
    assert mb_batch["s_axis"].shape == (4, J, 3)
    assert torch.allclose(mb_batch["root_pos"], torch.full((4, 3), 4.0))
    assert torch.allclose(mb_batch["motion_times"], torch.full((4,), 0.4))
    assert torch.all(mb_batch["worst_joint_idx"] == 5)
    assert torch.allclose(mb_batch["q_des"][:, 5 * 3], torch.full((4,), 0.95))

    # Test sample_level3_mainbrain_resets: should return state several frames before failure
    l3_batch = collector.sample_level3_mainbrain_resets(batch_size=2, device="cpu")
    assert l3_batch["root_pos"].shape == (2, 3)
    assert l3_batch["dof_pos"].shape == (2, D)
    assert torch.all(l3_batch["motion_ids"] == 3)
    # Lookback of 4 frames from step 5 (at index K-1=7) lands on step 1 (at index K-5=3)
    assert torch.allclose(l3_batch["root_pos"], torch.full((2, 3), 1.0))

    # Test summary and save/load round-trip
    summary = collector.get_summary()
    assert summary["counts"]["level1_analytical_load"] == 1
    assert summary["counts"]["level2_minibrain_tracking"] == 1
    assert summary["counts"]["level3_mainbrain_balance"] == 1
    assert 5 in summary["levels"]["level2_minibrain_tracking"]["most_frequent_failing_joints"]

    saved_paths = collector.save_all(tmp_path)
    assert saved_paths["level1"].exists()
    assert saved_paths["level2"].exists()
    assert saved_paths["level3"].exists()
    assert saved_paths["summary"].exists()

    with open(saved_paths["summary"], "r", encoding="utf-8") as f:
        loaded_summary = json.load(f)
    assert loaded_summary["counts"]["level2_minibrain_tracking"] == 1

    # Reload into a fresh collector and verify buckets match
    reloaded = _make_collector(num_envs=3, window_length=K)
    reloaded.load_all(tmp_path)
    assert len(reloaded.get_level_bucket(EdgeCaseLevel.LEVEL_1_ANALYTICAL_LOAD)) == 1
    assert len(reloaded.get_level_bucket(EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING)) == 1
    assert len(reloaded.get_level_bucket(EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE)) == 1
    assert reloaded.get_level_bucket(EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING)[0]["worst_joint_idx"] == 5
