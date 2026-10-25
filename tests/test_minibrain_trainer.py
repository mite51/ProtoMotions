# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `scripts/train_smpl_minibrain.py` and `scripts/evaluate_edge_cases.py`."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from protomotions.agents.common.minibrain import (
    MINIBRAIN_FEATURE_DIM,
    SharedJointMiniBrain,
    SharedJointMiniBrainConfig,
)
from protomotions.envs.diagnostics.edge_case_collector import (
    EdgeCaseCollector,
    EdgeCaseCollectorConfig,
    EdgeCaseLevel,
)
from protomotions.envs.obs.joint_load_obs import build_subtree_topology
from protomotions.robot_configs.smpl import SmplRobotConfig
from scripts.evaluate_edge_cases import (
    evaluate_edge_cases_directory,
    format_edge_case_report,
    get_smpl_joint_names,
)
from scripts.train_smpl_minibrain import (
    BENCHMARK_REGIMES,
    CONTROLLER_NAMES,
    MiniBrainTrainerConfig,
    SyntheticSMPLPhysicsEnv,
    collect_short_horizon_rollout,
    compute_per_joint_gae,
    evaluate_minibrain_vs_baselines,
    format_benchmark_table,
    sample_perturbed_smpl_reset,
    sample_s_axis_with_lockouts,
    train_smpl_minibrain,
    update_mode_a_surrogate,
    update_mode_b_per_joint_ppo,
)


@pytest.fixture(scope="module")
def smpl_topology():
    cfg = SmplRobotConfig()
    return build_subtree_topology(cfg.kinematic_info, cfg.control, device="cpu")


def test_perturbation_sampler_and_dof_lockouts(smpl_topology):
    """Verify stamina range [0.2, 1.0], 1- or 2-axis DOF lockouts, and locked target invariants."""
    gen = torch.Generator().manual_seed(101)
    num_envs = 32
    num_joints = smpl_topology.num_joints

    body_stamina, dof_active_mask, s_axis = sample_s_axis_with_lockouts(
        num_envs=num_envs,
        num_joints=num_joints,
        stamina_min=0.2,
        stamina_max=1.0,
        lockout_prob=1.0,
        force_lock_1_or_2_all_joints=True,
        device="cpu",
        generator=gen,
    )

    assert body_stamina.shape == (num_envs, 23)
    assert dof_active_mask.shape == (num_envs, 23, 3)
    assert s_axis.shape == (num_envs, 23, 3)

    assert torch.all(body_stamina >= 0.2 - 1e-6)
    assert torch.all(body_stamina <= 1.0 + 1e-6)

    # Every joint must have either 1 or 2 axes locked (so 1 or 2 axes active, never 0 or 3)
    active_counts = (dof_active_mask > 0.5).sum(dim=-1)
    assert torch.all((active_counts == 1) | (active_counts == 2))

    # Full reset dictionary with horizon=4
    reset_data = sample_perturbed_smpl_reset(
        num_envs=16,
        horizon=4,
        topology=smpl_topology,
        stamina_min=0.2,
        stamina_max=1.0,
        lockout_prob=0.5,
        device="cpu",
        generator=gen,
    )
    assert reset_data["q_init"].shape == (16, 23, 3)
    assert reset_data["dq_init"].shape == (16, 23, 3)
    assert reset_data["q_des_seq"].shape == (16, 4, 23, 3)
    assert reset_data["dq_des_seq"].shape == (16, 4, 23, 3)

    locked_axes = reset_data["s_axis"] <= 1e-6
    assert locked_axes.any()
    assert torch.all(reset_data["dq_init"][locked_axes] == 0.0)
    for k in range(4):
        assert torch.allclose(
            reset_data["q_des_seq"][:, k][locked_axes],
            reset_data["q_init"][locked_axes],
            atol=0.0,
        )
        assert torch.all(reset_data["dq_des_seq"][:, k][locked_axes] == 0.0)


def test_synthetic_physics_locked_dofs_and_analytical_pd_lead(smpl_topology):
    """Verify locked DOFs remain stationary and Analytical PD Lead outperforms Baseline PD."""
    env = SyntheticSMPLPhysicsEnv(
        num_envs=24, dt=1.0 / 30.0, substeps=4, device="cpu", topology=smpl_topology
    )
    mb = SharedJointMiniBrain(
        SharedJointMiniBrainConfig(num_joints=23, feature_dim=MINIBRAIN_FEATURE_DIM)
    )
    mb.freeze()

    bench = evaluate_minibrain_vs_baselines(
        minibrain=mb,
        env=env,
        horizon=4,
        num_eval_envs=24,
        seed=77,
    )

    assert set(bench.keys()) == set(BENCHMARK_REGIMES)
    for regime in BENCHMARK_REGIMES:
        for c_name in CONTROLLER_NAMES:
            assert c_name in bench[regime]
            metrics = bench[regime][c_name]
            assert metrics["step1_pos_mse"] >= 0.0
            assert metrics["horizon_pos_mse"] >= 0.0

        # At zero-init, SharedJointMiniBrain == Baseline PD (u = q_des)
        assert math.isclose(
            bench[regime]["SharedJointMiniBrain"]["step1_pos_mse"],
            bench[regime]["Baseline PD (u = q_des)"]["step1_pos_mse"],
            rel_tol=1e-5,
            abs_tol=1e-7,
        )

        # Analytical PD Lead reduces 4-step position MSE compared to Baseline PD
        assert (
            bench[regime]["Analytical PD Lead (compute_analytical_pd_lead_action)"][
                "horizon_pos_mse"
            ]
            < bench[regime]["Baseline PD (u = q_des)"]["horizon_pos_mse"]
        )


def test_mode_a_surrogate_update_reduces_tracking_error(smpl_topology):
    """Verify Mode A surrogate training updates dynamics + policy and beats Baseline PD."""
    cfg = MiniBrainTrainerConfig(
        mode="surrogate",
        horizon=2,
        num_envs=16,
        num_iterations=12,
        lr_policy=4e-3,
        lr_dynamics=6e-3,
        dynamics_updates_per_iter=3,
        policy_updates_per_iter=2,
        seed=123,
    )
    env = SyntheticSMPLPhysicsEnv(
        num_envs=cfg.num_envs,
        dt=cfg.dt,
        substeps=cfg.substeps,
        device="cpu",
        topology=smpl_topology,
    )
    mb = SharedJointMiniBrain(
        SharedJointMiniBrainConfig(
            num_joints=23,
            feature_dim=MINIBRAIN_FEATURE_DIM,
            hidden_dims=[64, 64],
        )
    )
    mb.unfreeze()

    dyn_opt = torch.optim.Adam(mb.dynamics_head.parameters(), lr=cfg.lr_dynamics)
    pol_opt = torch.optim.Adam(
        [
            {
                "params": list(mb.trunk.parameters()) + list(mb.action_head.parameters()),
                "lr": cfg.lr_policy,
            },
            {"params": [mb.physics_basis_weights], "lr": 0.15},
        ]
    )
    gen = torch.Generator().manual_seed(cfg.seed)

    for it in range(cfg.num_iterations):
        rollout = collect_short_horizon_rollout(
            env=env,
            minibrain=mb,
            cfg=cfg,
            generator=gen,
            explore_std=cfg.surrogate_explore_std,
        )
        m = update_mode_a_surrogate(
            minibrain=mb,
            rollout=rollout,
            topology=smpl_topology,
            dyn_optimizer=dyn_opt,
            policy_optimizer=pol_opt,
            cfg=cfg,
            iteration=it,
        )
        assert math.isfinite(m["dynamics_loss"])
        assert math.isfinite(m["tracking_loss"])

    # Verify action_head weights moved away from zero
    assert mb.action_head.weight.abs().sum().item() > 0.0

    # Verify trained SharedJointMiniBrain achieves strictly lower tracking error than Baseline PD
    mb.freeze()
    eval_res = evaluate_minibrain_vs_baselines(
        minibrain=mb, env=env, horizon=2, num_eval_envs=16, seed=555
    )
    for regime in ("nominal", "low_stamina", "locked_1_or_2_dofs"):
        assert (
            eval_res[regime]["SharedJointMiniBrain"]["horizon_pos_mse"]
            < eval_res[regime]["Baseline PD (u = q_des)"]["horizon_pos_mse"]
        )


def test_mode_b_per_joint_ppo_update_and_gae(smpl_topology):
    """Verify Mode B per-joint GAE and clipped PPO update across E * 23 parallel joints."""
    cfg = MiniBrainTrainerConfig(
        mode="ppo",
        horizon=4,
        num_envs=12,
        num_iterations=4,
        ppo_epochs=3,
        lr_policy=3e-3,
        seed=321,
    )
    env = SyntheticSMPLPhysicsEnv(
        num_envs=cfg.num_envs,
        dt=cfg.dt,
        substeps=cfg.substeps,
        device="cpu",
        topology=smpl_topology,
    )
    mb = SharedJointMiniBrain(
        SharedJointMiniBrainConfig(
            num_joints=23,
            feature_dim=MINIBRAIN_FEATURE_DIM,
            hidden_dims=[64, 64],
            include_value_head=True,
        )
    )
    mb.unfreeze()

    ppo_log_std = nn.Parameter(torch.full((3,), math.log(cfg.ppo_init_std)))
    ppo_opt = torch.optim.Adam(
        list(mb.trunk.parameters())
        + list(mb.action_head.parameters())
        + list(mb.value_head.parameters())
        + [ppo_log_std],
        lr=cfg.lr_policy,
    )
    gen = torch.Generator().manual_seed(cfg.seed)

    rollout = collect_short_horizon_rollout(
        env=env,
        minibrain=mb,
        cfg=cfg,
        generator=gen,
        ppo_log_std=ppo_log_std,
    )
    advantages, returns = compute_per_joint_gae(
        rewards=rollout["rewards"],
        values=rollout["values"],
        bootstrap_value=rollout["bootstrap_value"],
        gamma=cfg.gamma,
        tau=cfg.gae_tau,
    )
    assert advantages.shape == (cfg.num_envs, cfg.horizon, 23)
    assert returns.shape == (cfg.num_envs, cfg.horizon, 23)

    for _ in range(cfg.num_iterations):
        rollout = collect_short_horizon_rollout(
            env=env,
            minibrain=mb,
            cfg=cfg,
            generator=gen,
            ppo_log_std=ppo_log_std,
        )
        ppo_metrics = update_mode_b_per_joint_ppo(
            minibrain=mb,
            ppo_log_std=ppo_log_std,
            rollout=rollout,
            ppo_optimizer=ppo_opt,
            cfg=cfg,
            topology=smpl_topology,
        )
        assert math.isfinite(ppo_metrics["ppo_policy_loss"])
        assert math.isfinite(ppo_metrics["ppo_value_loss"])
        assert math.isfinite(ppo_metrics["mean_joint_reward"])

    assert mb.action_head.weight.abs().sum().item() > 0.0


def test_end_to_end_hybrid_training_edge_case_replay_and_evaluator(
    tmp_path: Path, smpl_topology
):
    """Test Level 2 edge-case replay in `train_smpl_minibrain` and `evaluate_edge_cases_directory`."""
    edge_dir = tmp_path / "saved_edge_cases"
    collector = EdgeCaseCollector(
        config=EdgeCaseCollectorConfig(window_length=4, num_joints=23, num_bodies=24),
        num_envs=3,
        device="cpu",
    )

    # Record 2 steps so pre-failure frame (K-2) and failure frame (K-1) are populated
    E, J, D, B = 3, 23, 69, 24
    collector.record_step(
        root_pos=torch.tensor([[0.0, 0.0, 0.95]] * E),
        dof_pos=torch.zeros((E, D)),
        dof_vel=torch.zeros((E, D)),
        q_des=torch.zeros((E, D)),
        dq_des=torch.zeros((E, D)),
        vel_pred_error=torch.zeros((E, J, 3)),
        contact_force_magnitudes=torch.zeros((E, B)),
    )

    vel_err = torch.zeros((E, J, 3))
    vel_err[0, 1, 0] = 12.0  # Env 0 -> Level 1 on joint 1 (L_Knee)
    q_des = torch.zeros((E, D))
    q_des[1, 0] = 0.40  # Env 1 -> Level 2 on joint 0 (L_Hip), unsaturated (320 N*m < 500 N*m)
    dq_des = q_des * 10.0
    contact_forces = torch.zeros((E, B))
    contact_forces[2, 0] = 120.0  # Env 2 -> Level 3 impact + termination

    collector.record_step(
        root_pos=torch.tensor([[0.0, 0.0, 0.95]] * E),
        dof_pos=torch.zeros((E, D)),
        dof_vel=torch.zeros((E, D)),
        q_des=q_des,
        dq_des=dq_des,
        tau_grav_local=torch.full((E, J, 3), 4.0),
        ang_momentum_local=torch.full((E, J, 3), 1.5),
        vel_pred_error=vel_err,
        contact_force_magnitudes=contact_forces,
        terminated=torch.tensor([False, False, True]),
    )
    saved_edge_paths = collector.save_all(edge_dir)

    ckpt_path = tmp_path / "checkpoints" / "smpl_minibrain.pt"
    bench_path = tmp_path / "checkpoints" / "smpl_minibrain_benchmark.json"

    trainer_cfg = MiniBrainTrainerConfig(
        mode="hybrid",
        horizon=2,
        num_envs=8,
        num_iterations=6,
        eval_num_envs=8,
        edge_cases_file=str(saved_edge_paths["level2"]),
        edge_case_prob=0.5,
        checkpoint_path=str(ckpt_path),
        benchmark_json_path=str(bench_path),
        seed=99,
    )
    trained_mb, bench_payload = train_smpl_minibrain(trainer_cfg)

    assert ckpt_path.exists()
    assert bench_path.exists()
    with open(bench_path, "r", encoding="utf-8") as f:
        loaded_bench = json.load(f)
    assert "benchmark_results" in loaded_bench
    table_text = format_benchmark_table(
        bench_payload["benchmark_results"], horizon=trainer_cfg.horizon
    )
    assert "SharedJointMiniBrain" in table_text
    assert "locked_1_or_2_dofs" in table_text

    # Evaluate saved edge cases using `evaluate_edge_cases_directory`
    joint_names = get_smpl_joint_names(topology=smpl_topology)
    assert len(joint_names) == 23
    assert joint_names[0] == "L_Hip"

    export_dir = tmp_path / "exported_packs"
    report = evaluate_edge_cases_directory(
        edge_cases_dir=edge_dir,
        minibrain=trained_mb,
        export_hard_resets_dir=export_dir,
        device="cpu",
    )

    assert report["level_breakdown"]["total_cases"] == 3
    assert report["level_breakdown"]["level1_analytical_load"]["count"] == 1
    assert report["level_breakdown"]["level2_minibrain_tracking"]["count"] == 1
    assert report["level_breakdown"]["level3_mainbrain_balance"]["count"] == 1
    assert report["level2_analysis"]["minibrain_reduces_error_vs_pd"] is True
    assert report["level3_analysis"]["max_contact_force"] == pytest.approx(120.0)
    assert "level2_minibrain_pack" in report["exported_hard_reset_packs"]
    assert "level3_mainbrain_pack" in report["exported_hard_reset_packs"]

    formatted_report = format_edge_case_report(report)
    assert "MULTI-LEVEL SMPL EDGE-CASE DIAGNOSTIC REPORT" in formatted_report
    assert "L_Hip" in formatted_report
