#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline Diagnostic Evaluator & Hard-Reset Pack Exporter for Multi-Level SMPL Edge Cases.

Loads a saved `EdgeCaseCollector` directory containing:
  - `edge_cases_level1_load.pt`
  - `edge_cases_level2_minibrain.pt`
  - `edge_cases_level3_mainbrain.pt`
  - `edge_cases_summary.json`

Produces a structured diagnostic report and formatted summary:
  1. Counts and percentages per Level (Level 1 Analytical Load, Level 2 Mini-Brain
     Tracking, Level 3 Main Brain Balance/Impact).
  2. Top-5 worst SMPL joints by name (using the 23 articulated joint body names from
     `SmplRobotConfig` / `SubtreeTopology`).
  3. Level 1 analysis: compares analytical `tau_grav_local` and `ang_momentum_local`
     magnitudes against observed 1-step `vel_pred_error`.
  4. Level 2 evaluation: replays saved Level 2 pre-failure states through
     `SharedJointMiniBrain` vs. standard PD (`u = q_des`) and Analytical PD Lead to
     verify tracking error reduction.
  5. Level 3 analysis: reports contact force magnitudes and root tilt angles at failure.
  6. Optional export (`--export-hard-resets-dir`) of filtered hard-reset packs for
     `train_smpl_minibrain.py` and Main Brain PPO.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from protomotions.agents.common.minibrain import (
    MINIBRAIN_FEATURE_DIM,
    S_AXIS_ACTIVE_EPS,
    SharedJointMiniBrain,
    SharedJointMiniBrainConfig,
)
from protomotions.envs.diagnostics.edge_case_collector import (
    EdgeCaseCollector,
    EdgeCaseCollectorConfig,
    EdgeCaseLevel,
     compute_root_tilt_angle,
    wrap_to_pi,
)
from protomotions.envs.obs.joint_load_obs import (
    SubtreeTopology,
    build_subtree_topology,
    compute_analytical_pd_lead_action,
)
from protomotions.robot_configs.smpl import SmplRobotConfig
from scripts.train_smpl_minibrain import SyntheticSMPLPhysicsEnv


def get_smpl_joint_names(
    robot_cfg: Optional[SmplRobotConfig] = None,
    topology: Optional[SubtreeTopology] = None,
) -> List[str]:
    """Return the 23 articulated SMPL joint body names in kinematic order."""
    cfg = robot_cfg if robot_cfg is not None else SmplRobotConfig()
    kin = cfg.kinematic_info
    topo = (
        topology
        if topology is not None
        else build_subtree_topology(kin, cfg.control, device="cpu")
    )
    body_names = list(kin.body_names)
    return [body_names[int(b_idx)] for b_idx in topo.joint_body_indices.tolist()]


def compute_top_k_worst_joints(
    histogram: List[int],
    joint_names: List[str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Rank the top-k worst joints by failure count with SMPL joint names."""
    total = max(1, sum(int(c) for c in histogram))
    indexed = [
        (idx, int(count))
        for idx, count in enumerate(histogram)
        if idx < len(joint_names) and int(count) > 0
    ]
    indexed.sort(key=lambda pair: (-pair[1], pair[0]))
    top_entries: List[Dict[str, Any]] = []
    for idx, count in indexed[:top_k]:
        top_entries.append(
            {
                "joint_idx": int(idx),
                "joint_name": joint_names[idx],
                "count": int(count),
                "percentage": 100.0 * float(count) / float(total),
            }
        )
    return top_entries


def analyze_level1_analytical_load(
    collector: EdgeCaseCollector,
    joint_names: List[str],
) -> Dict[str, Any]:
    """Compare analytical `tau_grav_local` / `ang_momentum_local` against `vel_pred_error` on Level 1 cases."""
    cases = collector.get_level_bucket(EdgeCaseLevel.LEVEL_1_ANALYTICAL_LOAD)
    if len(cases) == 0:
        return {
            "num_cases": 0,
            "mean_tau_grav_norm": 0.0,
            "max_tau_grav_norm": 0.0,
            "mean_ang_momentum_norm": 0.0,
            "max_ang_momentum_norm": 0.0,
            "mean_vel_pred_err": 0.0,
            "max_vel_pred_err": 0.0,
            "momentum_to_gravity_ratio": 0.0,
            "top_5_worst_joints": [],
        }

    collated = collector.collate_level_bucket(EdgeCaseLevel.LEVEL_1_ANALYTICAL_LOAD)
    # Inspect failure frame K - 1
    tau_fail = collated["tau_grav_local"][:, -1]  # [N, J, 3]
    mom_fail = collated["ang_momentum_local"][:, -1]  # [N, J, 3]
    vel_err_fail = collated["vel_pred_error"][:, -1]  # [N, J, 3]
    worst_joints = collated["worst_joint_idx"]  # [N]

    tau_norms = torch.linalg.norm(tau_fail, dim=-1)  # [N, J]
    mom_norms = torch.linalg.norm(mom_fail, dim=-1)  # [N, J]
    vel_err_norms = torch.linalg.norm(vel_err_fail, dim=-1)  # [N, J]

    batch_arange = torch.arange(len(cases), dtype=torch.long)
    worst_j_clamped = worst_joints.clamp(0, len(joint_names) - 1)
    worst_tau = tau_norms[batch_arange, worst_j_clamped]
    worst_mom = mom_norms[batch_arange, worst_j_clamped]
    worst_vel_err = vel_err_norms[batch_arange, worst_j_clamped]

    hist = torch.bincount(worst_j_clamped, minlength=len(joint_names)).tolist()
    mean_tau = float(worst_tau.mean().item())
    mean_mom = float(worst_mom.mean().item())

    return {
        "num_cases": len(cases),
        "mean_tau_grav_norm": mean_tau,
        "max_tau_grav_norm": float(worst_tau.max().item()),
        "mean_ang_momentum_norm": mean_mom,
        "max_ang_momentum_norm": float(worst_mom.max().item()),
        "mean_vel_pred_err": float(worst_vel_err.mean().item()),
        "max_vel_pred_err": float(worst_vel_err.max().item()),
        "momentum_to_gravity_ratio": float(mean_mom / max(mean_tau, 1e-6)),
        "top_5_worst_joints": compute_top_k_worst_joints(hist, joint_names, top_k=5),
    }


@torch.no_grad()
def evaluate_level2_minibrain_vs_pd(
    collector: EdgeCaseCollector,
    minibrain: Optional[SharedJointMiniBrain] = None,
    joint_names: Optional[List[str]] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Evaluate whether `SharedJointMiniBrain` reduces tracking error on saved Level 2 states vs. standard PD."""
    dev = torch.device(device)
    cases = collector.get_level_bucket(EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING)
    if joint_names is None:
        joint_names = get_smpl_joint_names()

    if len(cases) == 0:
        return {
            "num_cases": 0,
            "recorded_mean_tracking_err": 0.0,
            "baseline_pd_pos_mse": 0.0,
            "analytical_pd_lead_pos_mse": 0.0,
            "minibrain_pos_mse": 0.0,
            "baseline_pd_vel_mse": 0.0,
            "analytical_pd_lead_vel_mse": 0.0,
            "minibrain_vel_mse": 0.0,
            "minibrain_reduces_error_vs_pd": False,
            "error_reduction_pct": 0.0,
            "top_5_worst_joints": [],
        }

    collated = collector.collate_level_bucket(EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING)
    N = int(collated["num_cases"].item())
    K = int(collated["dof_pos"].shape[1])
    pre_idx = max(0, K - 2)
    fail_idx = K - 1

    env = SyntheticSMPLPhysicsEnv(num_envs=N, device=dev)
    num_joints = env.num_joints

    if minibrain is None:
        minibrain = SharedJointMiniBrain(
            SharedJointMiniBrainConfig(
                num_joints=num_joints,
                dofs_per_joint=3,
                feature_dim=MINIBRAIN_FEATURE_DIM,
            )
        ).to(dev)
        minibrain.freeze()
    else:
        minibrain = minibrain.to(dev)
        minibrain.eval()

    q_init = collated["dof_pos"][:, pre_idx].reshape(N, num_joints, 3).to(dev)
    dq_init = collated["dof_vel"][:, pre_idx].reshape(N, num_joints, 3).to(dev)
    q_des = collated["q_des"][:, fail_idx].reshape(N, num_joints, 3).to(dev)
    dq_des = collated["dq_des"][:, fail_idx].reshape(N, num_joints, 3).to(dev)
    s_axis = collated["s_axis"][:, pre_idx].reshape(N, num_joints, 3).to(dev)
    active_mask = (s_axis > S_AXIS_ACTIVE_EPS).to(torch.float32)
    active_denom = active_mask.sum().clamp_min(1.0)

    q_des_masked = torch.where(active_mask > 0.5, q_des, q_init)
    dq_des_masked = dq_des * active_mask

    reset_dict = {
        "q_init": q_init,
        "dq_init": dq_init * active_mask,
        "root_pos": collated["root_pos"][:, pre_idx].to(dev),
        "root_rot": collated["root_rot"][:, pre_idx].to(dev),
        "root_vel": collated["root_vel"][:, pre_idx].to(dev),
        "root_ang_vel": collated["root_ang_vel"][:, pre_idx].to(dev),
        "s_axis": s_axis,
    }

    # 1. Baseline PD (u = q_des)
    env.reset_to_state(reset_dict)
    u_pd_base = q_des_masked.reshape(N, -1)
    q_next_pd, dq_next_pd = env.step(u_pd_base)
    pd_pos_mse = float(
        ((wrap_to_pi(q_next_pd - q_des_masked) * active_mask).pow(2).sum() / active_denom).item()
    )
    pd_vel_mse = float(
        (((dq_next_pd - dq_des_masked) * active_mask).pow(2).sum() / active_denom).item()
    )

    # 2. Analytical PD Lead
    env.reset_to_state(reset_dict)
    joint_features_0, tau_grav_0, _, _, _ = env.compute_load_and_features(
        q_des=q_des_masked, dq_des=dq_des_masked
    )
    u_lead = compute_analytical_pd_lead_action(
        q_j=q_init.reshape(N, -1),
        dq_j=dq_init.reshape(N, -1),
        q_des=q_des_masked.reshape(N, -1),
        dq_des=dq_des_masked.reshape(N, -1),
        kp=env.topology.kp,
        kd=env.topology.kd,
        tau_grav_local=tau_grav_0,
        s_axis=s_axis,
        dt=env.dt,
    ).reshape(N, num_joints, 3)
    u_lead_cmd = torch.where(active_mask > 0.5, u_lead, q_init).reshape(N, -1)
    q_next_lead, dq_next_lead = env.step(u_lead_cmd)
    lead_pos_mse = float(
        ((wrap_to_pi(q_next_lead - q_des_masked) * active_mask).pow(2).sum() / active_denom).item()
    )
    lead_vel_mse = float(
        (((dq_next_lead - dq_des_masked) * active_mask).pow(2).sum() / active_denom).item()
    )

    # 3. SharedJointMiniBrain
    env.reset_to_state(reset_dict)
    u_mb = minibrain.compute_pd_target(
        joint_features=joint_features_0,
        q_des=q_des_masked,
        q_current=q_init,
        s_axis=s_axis,
    )
    q_next_mb, dq_next_mb = env.step(u_mb)
    mb_pos_mse = float(
        ((wrap_to_pi(q_next_mb - q_des_masked) * active_mask).pow(2).sum() / active_denom).item()
    )
    mb_vel_mse = float(
        (((dq_next_mb - dq_des_masked) * active_mask).pow(2).sum() / active_denom).item()
    )

    worst_j = collated["worst_joint_idx"].clamp(0, len(joint_names) - 1)
    hist = torch.bincount(worst_j, minlength=len(joint_names)).tolist()

    reduces = bool(mb_pos_mse <= pd_pos_mse)
    reduction_pct = (
        100.0 * (pd_pos_mse - mb_pos_mse) / max(pd_pos_mse, 1e-8)
        if pd_pos_mse > 0.0
        else 0.0
    )

    return {
        "num_cases": N,
        "recorded_mean_tracking_err": float(
            collated["max_joint_tracking_err"].mean().item()
        ),
        "baseline_pd_pos_mse": pd_pos_mse,
        "analytical_pd_lead_pos_mse": lead_pos_mse,
        "minibrain_pos_mse": mb_pos_mse,
        "baseline_pd_vel_mse": pd_vel_mse,
        "analytical_pd_lead_vel_mse": lead_vel_mse,
        "minibrain_vel_mse": mb_vel_mse,
        "minibrain_reduces_error_vs_pd": reduces,
        "error_reduction_pct": float(reduction_pct),
        "top_5_worst_joints": compute_top_k_worst_joints(hist, joint_names, top_k=5),
    }


def analyze_level3_mainbrain_balance(
    collector: EdgeCaseCollector,
    joint_names: List[str],
) -> Dict[str, Any]:
    """Report contact force magnitudes and root tilt angles at failure for Level 3 cases."""
    cases = collector.get_level_bucket(EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE)
    if len(cases) == 0:
        return {
            "num_cases": 0,
            "mean_contact_force": 0.0,
            "max_contact_force": 0.0,
            "mean_root_tilt_rad": 0.0,
            "max_root_tilt_rad": 0.0,
            "mean_root_tilt_deg": 0.0,
            "max_root_tilt_deg": 0.0,
            "termination_rate": 0.0,
            "top_5_worst_joints": [],
        }

    collated = collector.collate_level_bucket(EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE)
    forces = collated["max_contact_force"]
    tilts_rad = collated["root_tilt_err"]
    # If root_tilt_err was stored as 0, recompute from failure-frame root_rot
    if torch.all(tilts_rad == 0.0) and "root_rot" in collated:
        tilts_rad = compute_root_tilt_angle(collated["root_rot"][:, -1])

    tilts_deg = tilts_rad * (180.0 / math.pi)
    terminated = collated["terminated"].to(torch.float32)
    worst_j = collated["worst_joint_idx"].clamp(0, len(joint_names) - 1)
    hist = torch.bincount(worst_j, minlength=len(joint_names)).tolist()

    return {
        "num_cases": len(cases),
        "mean_contact_force": float(forces.mean().item()),
        "max_contact_force": float(forces.max().item()),
        "mean_root_tilt_rad": float(tilts_rad.mean().item()),
        "max_root_tilt_rad": float(tilts_rad.max().item()),
        "mean_root_tilt_deg": float(tilts_deg.mean().item()),
        "max_root_tilt_deg": float(tilts_deg.max().item()),
        "termination_rate": float(terminated.mean().item()),
        "top_5_worst_joints": compute_top_k_worst_joints(hist, joint_names, top_k=5),
    }


def export_hard_reset_packs(
    collector: EdgeCaseCollector,
    output_dir: Union[str, Path],
    batch_size: Optional[int] = None,
) -> Dict[str, Path]:
    """Export filtered hard-reset packs for `train_smpl_minibrain.py` and Main Brain PPO."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exported: Dict[str, Path] = {}

    l2_cases = collector.get_level_bucket(EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING)
    if len(l2_cases) > 0:
        n2 = len(l2_cases) if batch_size is None else min(int(batch_size), len(l2_cases))
        l2_batch = collector.sample_level2_minibrain_batch(batch_size=n2, device="cpu")
        l2_file = out_path / "hard_reset_pack_level2_minibrain.pt"
        torch.save(
            {
                "level": int(EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING),
                "num_samples": n2,
                "cases": l2_cases,
                "sampled_batch": l2_batch,
            },
            l2_file,
        )
        exported["level2_minibrain_pack"] = l2_file

    l3_cases = collector.get_level_bucket(EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE)
    if len(l3_cases) > 0:
        n3 = len(l3_cases) if batch_size is None else min(int(batch_size), len(l3_cases))
        l3_batch = collector.sample_level3_mainbrain_resets(batch_size=n3, device="cpu")
        l3_file = out_path / "hard_reset_pack_level3_mainbrain.pt"
        torch.save(
            {
                "level": int(EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE),
                "num_samples": n3,
                "cases": l3_cases,
                "sampled_resets": l3_batch,
            },
            l3_file,
        )
        exported["level3_mainbrain_pack"] = l3_file

    return exported


def evaluate_edge_cases_directory(
    edge_cases_dir: Union[str, Path],
    minibrain_checkpoint: Optional[Union[str, Path]] = None,
    minibrain: Optional[SharedJointMiniBrain] = None,
    export_hard_resets_dir: Optional[Union[str, Path]] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Load a saved edge-case directory and compute the full 3-level diagnostic report."""
    in_dir = Path(edge_cases_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Edge-case directory not found: {in_dir}")

    robot_cfg = SmplRobotConfig()
    topology = build_subtree_topology(
        robot_cfg.kinematic_info, robot_cfg.control, device=device
    )
    joint_names = get_smpl_joint_names(robot_cfg=robot_cfg, topology=topology)

    collector = EdgeCaseCollector(
        config=EdgeCaseCollectorConfig(num_joints=len(joint_names)),
        num_envs=1,
        device="cpu",
    )
    collector.load_all(in_dir)

    if minibrain is None and minibrain_checkpoint is not None:
        ckpt_path = Path(minibrain_checkpoint)
        if ckpt_path.exists():
            minibrain = SharedJointMiniBrain.load_checkpoint(
                ckpt_path, device=device, freeze=True
            )

    summary = collector.get_summary()
    c1 = int(summary["counts"]["level1_analytical_load"])
    c2 = int(summary["counts"]["level2_minibrain_tracking"])
    c3 = int(summary["counts"]["level3_mainbrain_balance"])
    total_cases = c1 + c2 + c3
    denom = float(max(1, total_cases))

    level_breakdown = {
        "total_cases": total_cases,
        "level1_analytical_load": {
            "count": c1,
            "percentage": 100.0 * float(c1) / denom,
        },
        "level2_minibrain_tracking": {
            "count": c2,
            "percentage": 100.0 * float(c2) / denom,
        },
        "level3_mainbrain_balance": {
            "count": c3,
            "percentage": 100.0 * float(c3) / denom,
        },
    }

    global_top_5_joints = compute_top_k_worst_joints(
        summary["joint_failure_histogram"], joint_names, top_k=5
    )

    level1_report = analyze_level1_analytical_load(collector, joint_names)
    level2_report = evaluate_level2_minibrain_vs_pd(
        collector, minibrain=minibrain, joint_names=joint_names, device=device
    )
    level3_report = analyze_level3_mainbrain_balance(collector, joint_names)

    exported_packs: Dict[str, str] = {}
    if export_hard_resets_dir is not None:
        pack_paths = export_hard_reset_packs(collector, export_hard_resets_dir)
        exported_packs = {k: str(v) for k, v in pack_paths.items()}

    report = {
        "edge_cases_dir": str(in_dir),
        "joint_names": joint_names,
        "level_breakdown": level_breakdown,
        "top_5_worst_joints": global_top_5_joints,
        "level1_analysis": level1_report,
        "level2_analysis": level2_report,
        "level3_analysis": level3_report,
        "exported_hard_reset_packs": exported_packs,
    }
    return report


def format_edge_case_report(report: Dict[str, Any]) -> str:
    """Format the multi-level edge-case diagnostic breakdown for console output."""
    lines: List[str] = []
    sep = "=" * 96
    lines.append(sep)
    lines.append("MULTI-LEVEL SMPL EDGE-CASE DIAGNOSTIC REPORT")
    lines.append(sep)

    lb = report["level_breakdown"]
    lines.append(f"Total Stored Edge Cases: {lb['total_cases']}")
    lines.append(
        f"  - Level 1 (Analytical Load Mismatch)  : {lb['level1_analytical_load']['count']:>4d} "
        f"({lb['level1_analytical_load']['percentage']:>5.1f}%)"
    )
    lines.append(
        f"  - Level 2 (Mini-Brain Tracking Error) : {lb['level2_minibrain_tracking']['count']:>4d} "
        f"({lb['level2_minibrain_tracking']['percentage']:>5.1f}%)"
    )
    lines.append(
        f"  - Level 3 (Main Brain Balance/Impact) : {lb['level3_mainbrain_balance']['count']:>4d} "
        f"({lb['level3_mainbrain_balance']['percentage']:>5.1f}%)"
    )
    lines.append("-" * 96)

    lines.append("Top-5 Worst SMPL Joints Across All Levels:")
    if len(report["top_5_worst_joints"]) == 0:
        lines.append("  (No joint failures recorded)")
    else:
        for rank, item in enumerate(report["top_5_worst_joints"], start=1):
            lines.append(
                f"  {rank}. {item['joint_name']:<14} (joint #{item['joint_idx']:>2d}) : "
                f"{item['count']:>4d} cases ({item['percentage']:>5.1f}%)"
            )
    lines.append("-" * 96)

    l1 = report["level1_analysis"]
    lines.append(f"Level 1 Diagnostics (Analytical Subtree Load, N={l1['num_cases']}):")
    lines.append(
        f"  Mean ||tau_grav_local||     : {l1['mean_tau_grav_norm']:.4f} N*m "
        f"(max {l1['max_tau_grav_norm']:.4f})"
    )
    lines.append(
        f"  Mean ||ang_momentum_local|| : {l1['mean_ang_momentum_norm']:.4f} kg*m^2/s "
        f"(max {l1['max_ang_momentum_norm']:.4f})"
    )
    lines.append(
        f"  Mean ||vel_pred_error||     : {l1['mean_vel_pred_err']:.4f} rad/s "
        f"(max {l1['max_vel_pred_err']:.4f})"
    )
    lines.append("-" * 96)

    l2 = report["level2_analysis"]
    lines.append(
        f"Level 2 Diagnostics (SharedJointMiniBrain vs. Standard PD, N={l2['num_cases']}):"
    )
    lines.append(f"  Baseline PD 1-Step Pos MSE        : {l2['baseline_pd_pos_mse']:.6f} rad^2")
    lines.append(
        f"  Analytical PD Lead 1-Step Pos MSE : {l2['analytical_pd_lead_pos_mse']:.6f} rad^2"
    )
    lines.append(f"  SharedJointMiniBrain 1-Step MSE   : {l2['minibrain_pos_mse']:.6f} rad^2")
    lines.append(
        f"  Mini-Brain Reduces Error vs. PD   : {l2['minibrain_reduces_error_vs_pd']} "
        f"({l2['error_reduction_pct']:+.2f}%)"
    )
    lines.append("-" * 96)

    l3 = report["level3_analysis"]
    lines.append(f"Level 3 Diagnostics (Main Brain Balance & Impact, N={l3['num_cases']}):")
    lines.append(
        f"  Contact Force Magnitude : mean {l3['mean_contact_force']:.2f} N, "
        f"max {l3['max_contact_force']:.2f} N"
    )
    lines.append(
        f"  Root Tilt at Failure    : mean {l3['mean_root_tilt_rad']:.3f} rad "
        f"({l3['mean_root_tilt_deg']:.1f} deg), max {l3['max_root_tilt_rad']:.3f} rad "
        f"({l3['max_root_tilt_deg']:.1f} deg)"
    )
    lines.append(f"  Episode Termination Rate: {100.0 * l3['termination_rate']:.1f}%")
    lines.append(sep)
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate saved Level 1/2/3 SMPL edge cases and optionally export hard-reset packs."
    )
    parser.add_argument(
        "--edge-cases-dir",
        type=str,
        required=True,
        help="Directory containing saved edge_cases_level{1,2,3}_*.pt and edge_cases_summary.json.",
    )
    parser.add_argument(
        "--minibrain-checkpoint",
        type=str,
        default=None,
        help="Optional path to trained SharedJointMiniBrain checkpoint (.pt) for Level 2 evaluation.",
    )
    parser.add_argument(
        "--export-hard-resets-dir",
        type=str,
        default=None,
        help="Optional output directory to export filtered Level 2 and Level 3 hard-reset packs.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save the diagnostic report as JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device ('cpu' or 'cuda').",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = evaluate_edge_cases_directory(
        edge_cases_dir=args.edge_cases_dir,
        minibrain_checkpoint=args.minibrain_checkpoint,
        export_hard_resets_dir=args.export_hard_resets_dir,
        device=args.device,
    )
    print(format_edge_case_report(report))

    if args.output_json is not None:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    main()
