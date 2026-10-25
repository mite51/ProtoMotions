#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone Short-Horizon Trainer & A/B Benchmark for SharedJointMiniBrain on SMPL.

Implements Component 3 of the Hierarchical Shared Mini-Brain & Analytical Subtree Load
architecture:
  - Short-horizon rollouts (horizon H in {1, 2, 4}, default 4) with Reference State
    Initialization (RSI), pose/velocity/root-orientation perturbations, randomized
    per-body stamina in [0.2, 1.0], and random 1- or 2-axis DOF lockouts.
  - Mode A ("surrogate"): Online 1-step local dynamics surrogate learning
    (`minibrain.predict_next_joint_state`) + differentiable tracking loss
    backpropagated into `minibrain.trunk` and `minibrain.action_head`.
  - Mode B ("ppo"): Per-joint short-horizon PPO across all `E * 23` parallel joints
    with GAE (`gamma=0.95, tau=0.95`) using `minibrain.evaluate_value`.
  - Mode "hybrid": Combines Mode A surrogate gradient and Mode B per-joint PPO.
  - Level 2 Hard-Example Replay (`--edge-cases-file`, `--edge-case-prob`): mixes
    initial states sampled from `EdgeCaseCollector.sample_level2_minibrain_batch`
    into short-horizon resets.
  - Built-in 4-regime A/B Benchmark (`evaluate_minibrain_vs_baselines`) comparing:
      1. Baseline PD (`u = q_des`)
      2. Analytical PD Lead (`compute_analytical_pd_lead_action`)
      3. `SharedJointMiniBrain`
    across `nominal`, `high_gravity_tilt`, `low_stamina`, and `locked_1_or_2_dofs`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

# Ensure repository root is importable when executed as `python scripts/train_smpl_minibrain.py`
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
    wrap_to_pi,
)
from protomotions.envs.obs.joint_load_obs import (
    SubtreeTopology,
    build_subtree_topology,
    compute_analytical_pd_lead_action,
    compute_analytical_subtree_load,
    compute_minibrain_joint_features,
)
from protomotions.robot_configs.smpl import SmplRobotConfig
from protomotions.utils.rotations import (
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_identity,
    quat_mul,
    quat_rotate,
)


BENCHMARK_REGIMES: Tuple[str, ...] = (
    "nominal",
    "high_gravity_tilt",
    "low_stamina",
    "locked_1_or_2_dofs",
)

CONTROLLER_NAMES: Tuple[str, ...] = (
    "Baseline PD (u = q_des)",
    "Analytical PD Lead (compute_analytical_pd_lead_action)",
    "SharedJointMiniBrain",
)


@dataclass
class MiniBrainTrainerConfig:
    """Configuration for short-horizon `SharedJointMiniBrain` training and benchmarking."""

    mode: str = "surrogate"  # "surrogate" (Mode A), "ppo" (Mode B), or "hybrid"
    horizon: int = 4  # Short rollout horizon H in {1, 2, 4}
    num_envs: int = 64
    num_iterations: int = 80
    dt: float = 1.0 / 30.0
    substeps: int = 4
    device: str = "cpu"
    seed: int = 42

    # Perturbation & domain randomization parameters
    pose_noise_std: float = 0.06
    vel_noise_std: float = 0.20
    root_tilt_max_rad: float = 0.35
    stamina_min: float = 0.2
    stamina_max: float = 1.0
    lockout_prob: float = 0.2

    # Mode A (Differentiable Surrogate) hyperparameters
    lr_policy: float = 3e-3
    lr_dynamics: float = 5e-3
    dynamics_updates_per_iter: int = 3
    policy_updates_per_iter: int = 2
    vel_weight: float = 0.08
    reg_weight: float = 0.005
    warmstart_guidance_weight: float = 0.15
    warmstart_iters: int = 20
    surrogate_explore_std: float = 0.05

    # Mode B (Per-Joint Short-Horizon PPO) hyperparameters
    ppo_epochs: int = 4
    ppo_clip_eps: float = 0.2
    gamma: float = 0.95
    gae_tau: float = 0.95
    pos_sharpness: float = 12.0
    vel_sharpness: float = 0.5
    ppo_action_reg: float = 0.01
    ppo_init_std: float = 0.12
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.001
    hybrid_surrogate_weight: float = 1.0
    hybrid_ppo_weight: float = 0.5

    # Edge-case hard-example replay
    edge_cases_file: Optional[str] = None
    edge_case_prob: float = 0.3

    # Simulator / motion options
    motion_file: Optional[str] = None
    synthetic_physics: bool = True

    # Output artifacts
    checkpoint_path: str = "checkpoints/smpl_minibrain.pt"
    benchmark_json_path: str = "checkpoints/smpl_minibrain_benchmark.json"
    eval_num_envs: int = 64


def sample_s_axis_with_lockouts(
    num_envs: int,
    num_joints: int = 23,
    dofs_per_joint: int = 3,
    stamina_min: float = 0.2,
    stamina_max: float = 1.0,
    lockout_prob: float = 0.2,
    force_lock_1_or_2_all_joints: bool = False,
    device: Union[str, torch.device] = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Sample per-body stamina `[E, J]`, binary `dof_active_mask [E, J, 3]`, and `s_axis [E, J, 3]`.

    When a joint is selected for DOF lockout (with probability `lockout_prob` or
    for all joints when `force_lock_1_or_2_all_joints=True`), 1 or 2 of its 3 hinge axes
    are set to `0.0` while at least 1 axis remains active.
    """
    dev = torch.device(device)
    if stamina_max <= stamina_min:
        body_stamina = torch.full(
            (num_envs, num_joints), float(stamina_min), device=dev, dtype=torch.float32
        )
    else:
        u = torch.rand(
            (num_envs, num_joints), generator=generator, device=dev, dtype=torch.float32
        )
        body_stamina = stamina_min + (stamina_max - stamina_min) * u

    dof_active_mask = torch.ones(
        (num_envs, num_joints, dofs_per_joint), device=dev, dtype=torch.float32
    )

    if force_lock_1_or_2_all_joints:
        lock_joint_mask = torch.ones(
            (num_envs, num_joints), device=dev, dtype=torch.bool
        )
    elif lockout_prob > 0.0:
        lock_joint_mask = (
            torch.rand(
                (num_envs, num_joints),
                generator=generator,
                device=dev,
                dtype=torch.float32,
            )
            < lockout_prob
        )
    else:
        lock_joint_mask = torch.zeros(
            (num_envs, num_joints), device=dev, dtype=torch.bool
        )

    if lock_joint_mask.any():
        # Choose whether to lock 1 axis (50%) or 2 axes (50%) on each locked joint
        num_locked_axes = torch.randint(
            1,
            3,
            (num_envs, num_joints),
            generator=generator,
            device=dev,
        )
        # Random scores per axis to pick which 1 or 2 axes get zeroed
        axis_scores = torch.rand(
            (num_envs, num_joints, dofs_per_joint),
            generator=generator,
            device=dev,
            dtype=torch.float32,
        )
        sorted_order = torch.argsort(axis_scores, dim=-1)
        rank = torch.argsort(sorted_order, dim=-1)  # 0, 1, 2 per joint
        keep_axis = rank >= num_locked_axes.unsqueeze(-1)
        candidate_mask = keep_axis.to(torch.float32)
        dof_active_mask = torch.where(
            lock_joint_mask.unsqueeze(-1), candidate_mask, dof_active_mask
        )

    s_axis = (body_stamina.unsqueeze(-1) * dof_active_mask).clamp(0.0, 1.0)
    return body_stamina, dof_active_mask, s_axis


def sample_perturbed_smpl_reset(
    num_envs: int,
    horizon: int,
    topology: SubtreeTopology,
    dt: float = 1.0 / 30.0,
    pose_noise_std: float = 0.06,
    vel_noise_std: float = 0.20,
    root_tilt_max_rad: float = 0.35,
    stamina_min: float = 0.2,
    stamina_max: float = 1.0,
    lockout_prob: float = 0.2,
    force_lock_1_or_2_all_joints: bool = False,
    regime: Optional[str] = None,
    edge_case_collector: Optional[EdgeCaseCollector] = None,
    edge_case_prob: float = 0.0,
    motion_lib: Optional[Any] = None,
    device: Union[str, torch.device] = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Tensor]:
    """Sample a short-horizon reference trajectory and perturbed initial SMPL state.

    Supports:
      - Standard training randomization (RSI + pose/velocity/orientation noise +
        stamina in [0.2, 1.0] + random 1- or 2-DOF lockouts).
      - Benchmark regime presets (`nominal`, `high_gravity_tilt`, `low_stamina`,
        `locked_1_or_2_dofs`).
      - Hard-example replay mixing from `EdgeCaseCollector.sample_level2_minibrain_batch`.

    Returns dictionary with:
      - `q_init`: `[E, 23, 3]` initial joint angles at step 0
      - `dq_init`: `[E, 23, 3]` initial joint velocities at step 0
      - `root_pos`: `[E, 3]` root position
      - `root_rot`: `[E, 4]` root orientation quaternion (xyzw)
      - `root_vel`: `[E, 3]` root linear velocity
      - `root_ang_vel`: `[E, 3]` root angular velocity
      - `q_des_seq`: `[E, H, 23, 3]` target joint angles for steps `0..H-1`
      - `dq_des_seq`: `[E, H, 23, 3]` target joint velocities for steps `0..H-1`
      - `s_axis`: `[E, 23, 3]` per-axis availability / stamina mask
      - `body_stamina`: `[E, 23]` per-joint stamina scalar
      - `dof_active_mask`: `[E, 23, 3]` binary active DOF mask
      - `is_edge_case_replay`: `[E]` boolean mask indicating Level 2 replay envs
    """
    dev = torch.device(device)
    num_joints = topology.num_joints
    dofs = 3

    # Apply regime overrides if evaluating a specific benchmark regime
    if regime == "nominal":
        stamina_min, stamina_max = 1.0, 1.0
        lockout_prob = 0.0
        force_lock_1_or_2_all_joints = False
        root_tilt_max_rad = 0.08
    elif regime == "high_gravity_tilt":
        stamina_min, stamina_max = 1.0, 1.0
        lockout_prob = 0.0
        force_lock_1_or_2_all_joints = False
        root_tilt_max_rad = 0.95
    elif regime == "low_stamina":
        stamina_min, stamina_max = 0.2, 0.5
        lockout_prob = 0.0
        force_lock_1_or_2_all_joints = False
        root_tilt_max_rad = 0.25
    elif regime == "locked_1_or_2_dofs":
        stamina_min, stamina_max = 0.8, 1.0
        lockout_prob = 1.0
        force_lock_1_or_2_all_joints = True
        root_tilt_max_rad = 0.25

    body_stamina, dof_active_mask, s_axis = sample_s_axis_with_lockouts(
        num_envs=num_envs,
        num_joints=num_joints,
        dofs_per_joint=dofs,
        stamina_min=stamina_min,
        stamina_max=stamina_max,
        lockout_prob=lockout_prob,
        force_lock_1_or_2_all_joints=force_lock_1_or_2_all_joints,
        device=dev,
        generator=generator,
    )
    active_mask = (s_axis > S_AXIS_ACTIVE_EPS).to(torch.float32)

    # Reference trajectory generation: sample real AMASS clips from motion_lib when
    # provided, otherwise fall back to smooth harmonic motion within joint limits.
    lim_lo = topology.dof_limits_lower.to(device=dev, dtype=torch.float32).unsqueeze(0)
    lim_hi = topology.dof_limits_upper.to(device=dev, dtype=torch.float32).unsqueeze(0)

    eps_q = (
        torch.randn(
            (num_envs, num_joints, dofs),
            generator=generator,
            device=dev,
            dtype=torch.float32,
        )
        * pose_noise_std
    )
    eps_dq = (
        torch.randn(
            (num_envs, num_joints, dofs),
            generator=generator,
            device=dev,
            dtype=torch.float32,
        )
        * vel_noise_std
    )

    if motion_lib is not None and motion_lib.num_motions() > 0:
        motion_ids = torch.randint(
            0,
            motion_lib.num_motions(),
            (num_envs,),
            generator=generator,
            device=dev,
            dtype=torch.long,
        )
        max_len = (motion_lib.get_motion_length(motion_ids) - float(horizon + 1) * dt).clamp_min(0.0)
        rand_u = torch.rand((num_envs,), generator=generator, device=dev, dtype=torch.float32)
        t0 = rand_u * max_len

        ref_0 = motion_lib.get_motion_state(motion_ids, t0)
        q_ref_0 = ref_0.dof_pos.reshape(num_envs, num_joints, dofs).to(device=dev, dtype=torch.float32)
        dq_ref_0 = ref_0.dof_vel.reshape(num_envs, num_joints, dofs).to(device=dev, dtype=torch.float32)
        q_init = (q_ref_0 + eps_q * active_mask).clamp(lim_lo, lim_hi)
        dq_init = (dq_ref_0 + eps_dq) * active_mask

        q_des_list: List[Tensor] = []
        dq_des_list: List[Tensor] = []
        for step_k in range(horizon):
            t_k = (t0 + float(step_k + 1) * dt).clamp(max=motion_lib.get_motion_length(motion_ids))
            ref_k = motion_lib.get_motion_state(motion_ids, t_k)
            q_k = ref_k.dof_pos.reshape(num_envs, num_joints, dofs).to(device=dev, dtype=torch.float32).clamp(lim_lo, lim_hi)
            dq_k = ref_k.dof_vel.reshape(num_envs, num_joints, dofs).to(device=dev, dtype=torch.float32)
            q_k = torch.where(active_mask > 0.5, q_k, q_init)
            dq_k = dq_k * active_mask
            q_des_list.append(q_k)
            dq_des_list.append(dq_k)

        q_des_seq = torch.stack(q_des_list, dim=1)
        dq_des_seq = torch.stack(dq_des_list, dim=1)
        base_root_pos = ref_0.root_pos.to(device=dev, dtype=torch.float32)
        base_root_rot = ref_0.root_rot.to(device=dev, dtype=torch.float32)
        base_root_vel = ref_0.root_vel.to(device=dev, dtype=torch.float32)
        base_root_ang_vel = ref_0.root_ang_vel.to(device=dev, dtype=torch.float32)
    else:
        mid = 0.5 * (lim_lo + lim_hi)
        half_span = (0.5 * (lim_hi - lim_lo)).clamp(min=0.1, max=0.8)

        phase = (
            torch.rand(
                (num_envs, num_joints, dofs),
                generator=generator,
                device=dev,
                dtype=torch.float32,
            )
            * 2.0
            * math.pi
        )
        freq = 1.5 + 2.5 * torch.rand(
            (num_envs, num_joints, dofs),
            generator=generator,
            device=dev,
            dtype=torch.float32,
        )
        amp = (
            0.35
            * half_span
            * (
                0.4
                + 0.6
                * torch.rand(
                    (num_envs, num_joints, dofs),
                    generator=generator,
                    device=dev,
                    dtype=torch.float32,
                )
            )
        )

        q_ref_0 = (mid + amp * torch.sin(phase)).clamp(lim_lo + 0.05, lim_hi - 0.05)
        dq_ref_0 = amp * freq * torch.cos(phase)
        q_init = (q_ref_0 + eps_q * active_mask).clamp(lim_lo, lim_hi)
        dq_init = (dq_ref_0 + eps_dq) * active_mask

        # Build H-step target trajectory (q_des_seq, dq_des_seq)
        q_des_list = []
        dq_des_list = []
        for step_k in range(horizon):
            t_k = float(step_k + 1) * dt
            phase_k = phase + freq * t_k
            q_k = (mid + amp * torch.sin(phase_k)).clamp(lim_lo, lim_hi)
            dq_k = amp * freq * torch.cos(phase_k)
            # On locked axes, desired target stays locked at q_init with zero velocity
            q_k = torch.where(active_mask > 0.5, q_k, q_init)
            dq_k = dq_k * active_mask
            q_des_list.append(q_k)
            dq_des_list.append(dq_k)

        q_des_seq = torch.stack(q_des_list, dim=1)  # [E, H, 23, 3]
        dq_des_seq = torch.stack(dq_des_list, dim=1)  # [E, H, 23, 3]
        base_root_pos = torch.zeros((num_envs, 3), device=dev, dtype=torch.float32)
        base_root_pos[:, 2] = 0.95
        base_root_rot = quat_identity([num_envs], w_last=True).to(device=dev)
        base_root_vel = torch.zeros((num_envs, 3), device=dev, dtype=torch.float32)
        base_root_ang_vel = torch.zeros((num_envs, 3), device=dev, dtype=torch.float32)

    # Root pose & orientation with configurable tilt perturbation
    root_pos = base_root_pos

    if root_tilt_max_rad > 0.0:
        tilt_axis = torch.randn(
            (num_envs, 3), generator=generator, device=dev, dtype=torch.float32
        )
        tilt_axis[:, 2] = 0.2 * tilt_axis[:, 2]  # emphasize pitch/roll tilt relative to gravity
        tilt_axis = tilt_axis / tilt_axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        if regime == "high_gravity_tilt":
            tilt_angle = (
                0.5 * root_tilt_max_rad
                + 0.5
                * root_tilt_max_rad
                * torch.rand(
                    (num_envs,), generator=generator, device=dev, dtype=torch.float32
                )
            )
        else:
            tilt_angle = (
                torch.rand(
                    (num_envs,), generator=generator, device=dev, dtype=torch.float32
                )
                * root_tilt_max_rad
            )
        tilt_quat = quat_from_angle_axis(tilt_angle, tilt_axis, w_last=True)
        root_rot = quat_mul(tilt_quat, base_root_rot, w_last=True)
    else:
        root_rot = base_root_rot

    root_vel = base_root_vel + (
        torch.randn((num_envs, 3), generator=generator, device=dev, dtype=torch.float32)
        * 0.15
    )
    root_ang_vel = base_root_ang_vel + (
        torch.randn((num_envs, 3), generator=generator, device=dev, dtype=torch.float32)
        * 0.25
    )

    is_edge_case_replay = torch.zeros((num_envs,), device=dev, dtype=torch.bool)

    # Mix Level 2 edge-case hard examples if provided
    if (
        edge_case_collector is not None
        and edge_case_prob > 0.0
        and len(
            edge_case_collector.get_level_bucket(EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING)
        )
        > 0
    ):
        replay_mask = (
            torch.rand((num_envs,), generator=generator, device=dev, dtype=torch.float32)
            < edge_case_prob
        )
        num_replay = int(replay_mask.sum().item())
        if num_replay > 0:
            replay_ids = torch.nonzero(replay_mask, as_tuple=False).flatten()
            l2_batch = edge_case_collector.sample_level2_minibrain_batch(
                batch_size=num_replay, device=dev
            )
            l2_s_axis = l2_batch["s_axis"].reshape(num_replay, num_joints, dofs)
            l2_active = (l2_s_axis > S_AXIS_ACTIVE_EPS).to(torch.float32)
            l2_q_init = l2_batch["dof_pos"].reshape(num_replay, num_joints, dofs)
            l2_dq_init = (
                l2_batch["dof_vel"].reshape(num_replay, num_joints, dofs) * l2_active
            )
            l2_q_des = l2_batch["q_des"].reshape(num_replay, num_joints, dofs)
            l2_dq_des = (
                l2_batch["dq_des"].reshape(num_replay, num_joints, dofs) * l2_active
            )
            l2_q_des = torch.where(l2_active > 0.5, l2_q_des, l2_q_init)

            root_pos[replay_ids] = l2_batch["root_pos"]
            root_rot[replay_ids] = l2_batch["root_rot"]
            root_vel[replay_ids] = l2_batch["root_vel"]
            root_ang_vel[replay_ids] = l2_batch["root_ang_vel"]
            q_init[replay_ids] = l2_q_init
            dq_init[replay_ids] = l2_dq_init
            s_axis[replay_ids] = l2_s_axis
            dof_active_mask[replay_ids] = l2_active
            body_stamina[replay_ids] = l2_s_axis.max(dim=-1).values.clamp(0.2, 1.0)

            for step_k in range(horizon):
                t_k = float(step_k) * dt
                q_step = torch.where(
                    l2_active > 0.5,
                    (l2_q_des + t_k * l2_dq_des).clamp(lim_lo, lim_hi),
                    l2_q_init,
                )
                q_des_seq[replay_ids, step_k] = q_step
                dq_des_seq[replay_ids, step_k] = l2_dq_des

            is_edge_case_replay[replay_ids] = True

    return {
        "q_init": q_init,
        "dq_init": dq_init,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "root_vel": root_vel,
        "root_ang_vel": root_ang_vel,
        "q_des_seq": q_des_seq,
        "dq_des_seq": dq_des_seq,
        "s_axis": s_axis,
        "body_stamina": body_stamina,
        "dof_active_mask": dof_active_mask,
        "is_edge_case_replay": is_edge_case_replay,
    }


class SyntheticSMPLPhysicsEnv:
    """Fast CPU/GPU damped-spring-pendulum + gravity + subtree-load semi-implicit Euler stepper.

    Uses `SmplRobotConfig` and `SubtreeTopology` to perform exact tree forward kinematics,
    compute analytical subtree gravity torque and angular momentum via
    `compute_analytical_subtree_load`, and step all 23 3-DOF SMPL joints forward in time.
    """

    def __init__(
        self,
        num_envs: int = 64,
        dt: float = 1.0 / 30.0,
        substeps: int = 4,
        device: Union[str, torch.device] = "cpu",
        robot_cfg: Optional[SmplRobotConfig] = None,
        topology: Optional[SubtreeTopology] = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.substeps = max(1, int(substeps))
        self.device = torch.device(device)

        self.robot_cfg = robot_cfg if robot_cfg is not None else SmplRobotConfig()
        self.kinematic_info = self.robot_cfg.kinematic_info
        self.topology = (
            topology.to(self.device)
            if topology is not None
            else build_subtree_topology(
                self.kinematic_info, self.robot_cfg.control, device=self.device
            )
        )

        self.num_bodies = self.topology.num_bodies
        self.num_joints = self.topology.num_joints
        self.dofs_per_joint = 3
        self.local_pos = self.kinematic_info.local_pos.to(
            device=self.device, dtype=torch.float32
        )
        self.parent_indices = list(self.kinematic_info.parent_indices)
        self.joint_body_list = self.topology.joint_body_indices.tolist()
        self.body_to_joint = {
            b_idx: j_idx for j_idx, b_idx in enumerate(self.joint_body_list)
        }

        # State buffers
        self.root_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )
        self.root_rot = quat_identity([self.num_envs], w_last=True).to(
            device=self.device
        )
        self.root_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )
        self.root_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )
        self.q = torch.zeros(
            (self.num_envs, self.num_joints, 3), device=self.device, dtype=torch.float32
        )
        self.dq = torch.zeros(
            (self.num_envs, self.num_joints, 3), device=self.device, dtype=torch.float32
        )
        self.dq_expected = torch.zeros(
            (self.num_envs, self.num_joints, 3), device=self.device, dtype=torch.float32
        )
        self.s_axis = torch.ones(
            (self.num_envs, self.num_joints, 3), device=self.device, dtype=torch.float32
        )

        self.body_pos = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float32
        )
        self.body_rot = quat_identity([self.num_envs, self.num_bodies], w_last=True).to(
            device=self.device
        )
        self.body_vel = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float32
        )
        self.body_ang_vel = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float32
        )

    def update_forward_kinematics(self) -> None:
        """Compute batched tree forward kinematics and body velocities from `(root, q, dq)`."""
        E = self.q.shape[0]
        pos_list: List[Tensor] = [torch.empty(0)] * self.num_bodies
        rot_list: List[Tensor] = [torch.empty(0)] * self.num_bodies
        vel_list: List[Tensor] = [torch.empty(0)] * self.num_bodies
        ang_vel_list: List[Tensor] = [torch.empty(0)] * self.num_bodies

        pos_list[0] = self.root_pos
        rot_list[0] = self.root_rot
        vel_list[0] = self.root_vel
        ang_vel_list[0] = self.root_ang_vel

        # Convert all joint Euler angles [E, J, 3] to local quaternions [E, J, 4] in one batched call
        flat_q = self.q.reshape(-1, 3)
        local_quats = quat_from_euler_xyz(
            flat_q[:, 0], flat_q[:, 1], flat_q[:, 2], w_last=True
        ).reshape(E, self.num_joints, 4)

        for b_idx in range(1, self.num_bodies):
            p_idx = self.parent_indices[b_idx]
            p_pos = pos_list[p_idx]
            p_rot = rot_list[p_idx]
            p_vel = vel_list[p_idx]
            p_ang = ang_vel_list[p_idx]

            loc_offset = self.local_pos[b_idx].unsqueeze(0).expand(E, 3)
            world_offset = quat_rotate(p_rot, loc_offset, w_last=True)
            b_pos = p_pos + world_offset

            if b_idx in self.body_to_joint:
                j_idx = self.body_to_joint[b_idx]
                q_loc = local_quats[:, j_idx, :]
                dq_loc = self.dq[:, j_idx, :]
                b_rot = quat_mul(p_rot, q_loc, w_last=True)
                omega_rel = quat_rotate(p_rot, dq_loc, w_last=True)
                b_ang = p_ang + omega_rel
            else:
                b_rot = p_rot
                b_ang = p_ang

            b_vel = p_vel + torch.cross(p_ang, world_offset, dim=-1)

            pos_list[b_idx] = b_pos
            rot_list[b_idx] = b_rot
            vel_list[b_idx] = b_vel
            ang_vel_list[b_idx] = b_ang

        self.body_pos = torch.stack(pos_list, dim=1)
        self.body_rot = torch.stack(rot_list, dim=1)
        self.body_vel = torch.stack(vel_list, dim=1)
        self.body_ang_vel = torch.stack(ang_vel_list, dim=1)

    def reset_to_state(self, reset_dict: Dict[str, Tensor]) -> None:
        """Reset all environments to a sampled state dictionary."""
        self.q = reset_dict["q_init"].detach().clone().to(self.device)
        self.dq = reset_dict["dq_init"].detach().clone().to(self.device)
        self.dq_expected = self.dq.clone()
        self.root_pos = reset_dict["root_pos"].detach().clone().to(self.device)
        self.root_rot = reset_dict["root_rot"].detach().clone().to(self.device)
        self.root_vel = reset_dict["root_vel"].detach().clone().to(self.device)
        self.root_ang_vel = reset_dict["root_ang_vel"].detach().clone().to(self.device)
        self.s_axis = reset_dict["s_axis"].detach().clone().to(self.device)
        self.num_envs = int(self.q.shape[0])
        self.update_forward_kinematics()

    def compute_load_and_features(
        self,
        q_des: Tensor,
        dq_des: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute analytical subtree load and `[E, 23, 34]` Mini-Brain features."""
        tau_grav_local, ang_momentum_local, subtree_mass, effective_inertia = (
            compute_analytical_subtree_load(
                body_pos=self.body_pos,
                body_rot=self.body_rot,
                body_vel=self.body_vel,
                body_ang_vel=self.body_ang_vel,
                subtree_mask=self.topology.subtree_mask,
                body_masses=self.topology.body_masses,
                joint_body_indices=self.topology.joint_body_indices,
                parent_body_indices=self.topology.parent_body_indices,
                w_last=True,
                body_com_offset=self.topology.body_com_offset,
            )
        )
        vel_pred_error = self.dq - self.dq_expected
        joint_features = compute_minibrain_joint_features(
            dof_pos=self.q.reshape(self.num_envs, -1),
            dof_vel=self.dq.reshape(self.num_envs, -1),
            q_des=q_des.reshape(self.num_envs, -1),
            dq_des=dq_des.reshape(self.num_envs, -1),
            body_pos=self.body_pos,
            body_rot=self.body_rot,
            body_vel=self.body_vel,
            body_ang_vel=self.body_ang_vel,
            topology=self.topology,
            s_axis=self.s_axis,
            vel_pred_error=vel_pred_error,
            w_last=True,
        )
        return (
            joint_features,
            tau_grav_local,
            ang_momentum_local,
            subtree_mass,
            effective_inertia,
        )

    def step(self, u_pd: Tensor) -> Tuple[Tensor, Tensor]:
        """Step semi-implicit Euler joint dynamics for `self.dt` across `self.substeps`.

        Args:
            u_pd: PD target angles `[E, 69]` or `[E, 23, 3]`.

        Returns:
            Tuple `(q_next, dq_next)` each of shape `[E, 23, 3]`.
        """
        E = self.num_envs
        u_pd_3d = u_pd.reshape(E, self.num_joints, 3).to(
            device=self.device, dtype=torch.float32
        )
        active_mask = (self.s_axis > S_AXIS_ACTIVE_EPS).to(torch.float32)
        q_locked = self.q.detach().clone()

        kp = self.topology.kp.unsqueeze(0)  # [1, 23, 3]
        kd = self.topology.kd.unsqueeze(0)  # [1, 23, 3]
        effort_lim = self.topology.effort_limit.unsqueeze(0)  # [1, 23, 3]
        lim_lo = self.topology.dof_limits_lower.unsqueeze(0)
        lim_hi = self.topology.dof_limits_upper.unsqueeze(0)
        subtree_m = self.topology.subtree_masses.view(1, self.num_joints, 1)

        dt_sub = self.dt / float(self.substeps)

        # Compute analytical subtree load at the start of the step
        tau_grav_local, ang_momentum_local, _, effective_inertia = (
            compute_analytical_subtree_load(
                body_pos=self.body_pos,
                body_rot=self.body_rot,
                body_vel=self.body_vel,
                body_ang_vel=self.body_ang_vel,
                subtree_mask=self.topology.subtree_mask,
                body_masses=self.topology.body_masses,
                joint_body_indices=self.topology.joint_body_indices,
                parent_body_indices=self.topology.parent_body_indices,
                w_last=True,
                body_com_offset=self.topology.body_com_offset,
            )
        )

        # Effective per-joint rotational inertia in physical units (kg * m^2)
        inertia_j = (
            0.60 * kp * (self.dt * self.dt)
            + 0.03 * subtree_m
            + 0.15 * effective_inertia
        ).clamp_min(0.10)

        q_curr = self.q
        dq_curr = self.dq * active_mask

        # Record 1-step analytical expected velocity before external momentum coupling
        tau_pd_0 = (self.s_axis * kp * wrap_to_pi(u_pd_3d - q_curr)).clamp(
            -effort_lim, effort_lim
        )
        self.dq_expected = (
            (dq_curr + (self.dt / inertia_j) * (tau_pd_0 + tau_grav_local))
            / (1.0 + (self.dt / inertia_j) * (self.s_axis * kd))
        ) * active_mask

        for _ in range(self.substeps):
            pos_err = wrap_to_pi(u_pd_3d - q_curr)
            stamina_eff = self.s_axis.clamp(0.0, 1.0)
            max_tau = effort_lim * stamina_eff.clamp_min(0.15)
            tau_spring = (stamina_eff * kp * pos_err).clamp(-max_tau, max_tau)
            tau_load = tau_grav_local - 0.15 * ang_momentum_local

            numer = dq_curr + (dt_sub / inertia_j) * (tau_spring + tau_load)
            denom = 1.0 + (dt_sub / inertia_j) * (stamina_eff * kd)
            dq_curr = (numer / denom) * active_mask
            q_curr = q_curr + dt_sub * dq_curr
            q_curr = torch.where(
                active_mask > 0.5, q_curr.clamp(lim_lo, lim_hi), q_locked
            )

        self.q = q_curr
        self.dq = dq_curr
        self.update_forward_kinematics()
        return self.q, self.dq


def load_level2_edge_case_collector(
    edge_cases_path: Optional[Union[str, Path]],
    num_joints: int = 23,
    device: Union[str, torch.device] = "cpu",
) -> Optional[EdgeCaseCollector]:
    """Load an `EdgeCaseCollector` from either a directory or a direct `.pt` file path."""
    if edge_cases_path is None:
        return None
    path = Path(edge_cases_path)
    if not path.exists():
        raise FileNotFoundError(f"Edge cases path does not exist: {path}")

    cfg = EdgeCaseCollectorConfig(num_joints=num_joints)
    collector = EdgeCaseCollector(config=cfg, num_envs=1, device=device)

    if path.is_dir():
        collector.load_all(path)
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cases = payload.get("cases", [])
        if cases:
            first_root = cases[0].get("root_pos")
            if isinstance(first_root, Tensor) and first_root.dim() >= 2:
                collector.cfg.window_length = int(first_root.shape[0])
        collector.buckets[EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING] = list(cases)
        collector._write_cursor_per_level[EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING] = (
            len(cases)
        )
        collector.total_triggered_per_level[
            EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING
        ] = len(cases)

    return collector


def collect_short_horizon_rollout(
    env: SyntheticSMPLPhysicsEnv,
    minibrain: SharedJointMiniBrain,
    cfg: MiniBrainTrainerConfig,
    edge_case_collector: Optional[EdgeCaseCollector] = None,
    generator: Optional[torch.Generator] = None,
    explore_std: float = 0.0,
    ppo_log_std: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Collect a short-horizon rollout of length `H = cfg.horizon` across `E` environments.

    Returns a dictionary of batched rollout tensors of shape `[E, H, ...]`:
      - `joint_features`: `[E, H, 23, 34]`
      - `delta_u`: `[E, H, 23, 3]` (deterministic policy mean)
      - `delta_u_exec`: `[E, H, 23, 3]` (executed residual offset, with exploration if enabled)
      - `log_probs`: `[E, H, 23]` (per-joint Gaussian log-prob when `ppo_log_std` is provided)
      - `values`: `[E, H, 23]` (per-joint value predictions `V_joint`)
      - `rewards`: `[E, H, 23]` (per-joint tracking rewards)
      - `q_t`: `[E, H, 23, 3]`
      - `dq_t`: `[E, H, 23, 3]`
      - `q_next_actual`: `[E, H, 23, 3]`
      - `dq_next_actual`: `[E, H, 23, 3]`
      - `q_des`: `[E, H, 23, 3]`
      - `dq_des`: `[E, H, 23, 3]`
      - `s_axis`: `[E, H, 23, 3]`
      - `tau_grav_local`: `[E, H, 23, 3]`
      - `bootstrap_value`: `[E, 23]`
    """
    reset_data = sample_perturbed_smpl_reset(
        num_envs=cfg.num_envs,
        horizon=cfg.horizon,
        topology=env.topology,
        dt=cfg.dt,
        pose_noise_std=cfg.pose_noise_std,
        vel_noise_std=cfg.vel_noise_std,
        root_tilt_max_rad=cfg.root_tilt_max_rad,
        stamina_min=cfg.stamina_min,
        stamina_max=cfg.stamina_max,
        lockout_prob=cfg.lockout_prob,
        edge_case_collector=edge_case_collector,
        edge_case_prob=cfg.edge_case_prob,
        motion_lib=getattr(env, "motion_lib", None),
        device=env.device,
        generator=generator,
    )
    env.reset_to_state(reset_data)

    features_seq: List[Tensor] = []
    delta_u_seq: List[Tensor] = []
    delta_u_exec_seq: List[Tensor] = []
    log_probs_seq: List[Tensor] = []
    values_seq: List[Tensor] = []
    rewards_seq: List[Tensor] = []
    q_t_seq: List[Tensor] = []
    dq_t_seq: List[Tensor] = []
    q_next_seq: List[Tensor] = []
    dq_next_seq: List[Tensor] = []
    q_des_list: List[Tensor] = []
    dq_des_list: List[Tensor] = []
    s_axis_list: List[Tensor] = []
    tau_grav_list: List[Tensor] = []

    max_res = minibrain.config.max_residual_offset

    with torch.no_grad():
        for step_k in range(cfg.horizon):
            q_t = env.q.clone()
            dq_t = env.dq.clone()
            q_des_t = reset_data["q_des_seq"][:, step_k]
            dq_des_t = reset_data["dq_des_seq"][:, step_k]
            s_axis_t = env.s_axis.clone()
            active_mask = (s_axis_t > S_AXIS_ACTIVE_EPS).to(torch.float32)

            joint_features_t, tau_grav_t, _, _, _ = env.compute_load_and_features(
                q_des=q_des_t, dq_des=dq_des_t
            )

            mb_out = minibrain(
                joint_features_t,
                q_des=q_des_t,
                q_current=q_t,
                s_axis=s_axis_t,
            )
            mean_delta_u = mb_out["delta_u"]  # [E, 23, 3]
            val_t = (
                mb_out["joint_value"].squeeze(-1)
                if mb_out["joint_value"] is not None
                else torch.zeros(
                    (cfg.num_envs, env.num_joints),
                    device=env.device,
                    dtype=torch.float32,
                )
            )

            if ppo_log_std is not None:
                std = torch.exp(ppo_log_std).view(1, 1, -1).expand_as(mean_delta_u)
                noise = torch.randn(
                    mean_delta_u.shape,
                    generator=generator,
                    device=env.device,
                    dtype=torch.float32,
                )
                sampled_delta_u = (
                    (mean_delta_u + std * noise).clamp(-max_res, max_res) * active_mask
                )
                var = std * std + 1e-8
                log_p_axis = -0.5 * (
                    ((sampled_delta_u - mean_delta_u) ** 2) / var
                    + 2.0 * torch.log(std)
                    + math.log(2.0 * math.pi)
                )
                log_p_joint = (log_p_axis * active_mask).sum(dim=-1)
                exec_delta_u = sampled_delta_u
            elif explore_std > 0.0:
                noise = (
                    torch.randn(
                        mean_delta_u.shape,
                        generator=generator,
                        device=env.device,
                        dtype=torch.float32,
                    )
                    * explore_std
                )
                exec_delta_u = (mean_delta_u + noise).clamp(-max_res, max_res) * active_mask
                log_p_joint = torch.zeros(
                    (cfg.num_envs, env.num_joints),
                    device=env.device,
                    dtype=torch.float32,
                )
            else:
                exec_delta_u = mean_delta_u
                log_p_joint = torch.zeros(
                    (cfg.num_envs, env.num_joints),
                    device=env.device,
                    dtype=torch.float32,
                )

            q_base = torch.where(active_mask > 0.5, q_des_t, q_t)
            u_pd_exec = (q_base + exec_delta_u).reshape(cfg.num_envs, -1)

            q_next, dq_next = env.step(u_pd_exec)

            # Per-joint short-horizon reward r_{e,j}
            q_err_sq = ((wrap_to_pi(q_next - q_des_t) * active_mask) ** 2).sum(dim=-1)
            dq_err_sq = (((dq_next - dq_des_t) * active_mask) ** 2).sum(dim=-1)
            du_sq = ((exec_delta_u * active_mask) ** 2).sum(dim=-1)
            reward_t = (
                torch.exp(-cfg.pos_sharpness * q_err_sq)
                + 0.3 * torch.exp(-cfg.vel_sharpness * dq_err_sq)
                - cfg.ppo_action_reg * du_sq
            )

            features_seq.append(joint_features_t)
            delta_u_seq.append(mean_delta_u)
            delta_u_exec_seq.append(exec_delta_u)
            log_probs_seq.append(log_p_joint)
            values_seq.append(val_t)
            rewards_seq.append(reward_t)
            q_t_seq.append(q_t)
            dq_t_seq.append(dq_t)
            q_next_seq.append(q_next.clone())
            dq_next_seq.append(dq_next.clone())
            q_des_list.append(q_des_t)
            dq_des_list.append(dq_des_t)
            s_axis_list.append(s_axis_t)
            tau_grav_list.append(tau_grav_t)

        # Bootstrap value at step H
        final_features, _, _, _, _ = env.compute_load_and_features(
            q_des=reset_data["q_des_seq"][:, -1],
            dq_des=reset_data["dq_des_seq"][:, -1],
        )
        if minibrain.value_head is not None:
            bootstrap_value = minibrain.evaluate_value(final_features).squeeze(-1)
        else:
            bootstrap_value = torch.zeros(
                (cfg.num_envs, env.num_joints), device=env.device, dtype=torch.float32
            )

    return {
        "joint_features": torch.stack(features_seq, dim=1),
        "delta_u": torch.stack(delta_u_seq, dim=1),
        "delta_u_exec": torch.stack(delta_u_exec_seq, dim=1),
        "log_probs": torch.stack(log_probs_seq, dim=1),
        "values": torch.stack(values_seq, dim=1),
        "rewards": torch.stack(rewards_seq, dim=1),
        "q_t": torch.stack(q_t_seq, dim=1),
        "dq_t": torch.stack(dq_t_seq, dim=1),
        "q_next_actual": torch.stack(q_next_seq, dim=1),
        "dq_next_actual": torch.stack(dq_next_seq, dim=1),
        "q_des": torch.stack(q_des_list, dim=1),
        "dq_des": torch.stack(dq_des_list, dim=1),
        "s_axis": torch.stack(s_axis_list, dim=1),
        "tau_grav_local": torch.stack(tau_grav_list, dim=1),
        "bootstrap_value": bootstrap_value,
        "is_edge_case_replay": reset_data["is_edge_case_replay"],
    }


def update_mode_a_surrogate(
    minibrain: SharedJointMiniBrain,
    rollout: Dict[str, Tensor],
    topology: SubtreeTopology,
    dyn_optimizer: torch.optim.Optimizer,
    policy_optimizer: torch.optim.Optimizer,
    cfg: MiniBrainTrainerConfig,
    iteration: int = 0,
) -> Dict[str, float]:
    """Mode A update: supervised 1-step dynamics surrogate + differentiable tracking loss.

    1. Updates `minibrain.dynamics_head` (`predict_next_joint_state`) with supervised MSE on
       actual achieved `(q_next_actual, dq_next_actual)`.
    2. Updates `minibrain.trunk` + `minibrain.action_head` by backpropagating:
       `L_track = || wrap(q_next_pred - q_des) ||^2 + vel_weight * || dq_next_pred - dq_des ||^2 + reg_weight * || delta_u ||^2`
       through the frozen/detached `dynamics_head` (plus optional analytical PD warm-start guidance).
    """
    if minibrain.dynamics_head is None:
        raise RuntimeError("Mode A requires SharedJointMiniBrain(include_dynamics_head=True)")

    E, H, J, F_dim = rollout["joint_features"].shape
    N = E * H
    flat_features = rollout["joint_features"].reshape(N, J, F_dim).detach()
    flat_delta_u_exec = rollout["delta_u_exec"].reshape(N, J, 3).detach()
    flat_q_t = rollout["q_t"].reshape(N, J, 3).detach()
    flat_dq_t = rollout["dq_t"].reshape(N, J, 3).detach()
    flat_q_next_actual = rollout["q_next_actual"].reshape(N, J, 3).detach()
    flat_dq_next_actual = rollout["dq_next_actual"].reshape(N, J, 3).detach()
    flat_q_des = rollout["q_des"].reshape(N, J, 3).detach()
    flat_dq_des = rollout["dq_des"].reshape(N, J, 3).detach()
    flat_s_axis = rollout["s_axis"].reshape(N, J, 3).detach()
    flat_tau_grav = rollout["tau_grav_local"].reshape(N, J, 3).detach()

    active_mask = (flat_s_axis > S_AXIS_ACTIVE_EPS).to(torch.float32)
    active_denom = active_mask.sum().clamp_min(1.0)

    # 1. Supervised update of `minibrain.dynamics_head`
    for p in minibrain.dynamics_head.parameters():
        p.requires_grad_(True)

    dyn_loss_val = 0.0
    for _ in range(max(1, cfg.dynamics_updates_per_iter)):
        dyn_optimizer.zero_grad()
        q_next_pred_sup, dq_next_pred_sup = minibrain.predict_next_joint_state(
            flat_features, flat_delta_u_exec, s_axis=flat_s_axis
        )
        q_sup_err = wrap_to_pi(q_next_pred_sup - flat_q_next_actual) * active_mask
        dq_sup_err = (dq_next_pred_sup - flat_dq_next_actual) * active_mask
        dyn_loss = (q_sup_err.pow(2).sum() / active_denom) + 0.1 * (
            dq_sup_err.pow(2).sum() / active_denom
        )
        dyn_loss.backward()
        nn.utils.clip_grad_norm_(minibrain.dynamics_head.parameters(), max_norm=2.0)
        dyn_optimizer.step()
        dyn_loss_val = float(dyn_loss.item())

    # Freeze dynamics_head parameters while backpropagating L_track into trunk + action_head
    for p in minibrain.dynamics_head.parameters():
        p.requires_grad_(False)

    # Precompute analytical PD lead + empirical simulator error target for Mode A
    u_analytical = compute_analytical_pd_lead_action(
        q_j=flat_q_t.reshape(N, -1),
        dq_j=flat_dq_t.reshape(N, -1),
        q_des=flat_q_des.reshape(N, -1),
        dq_des=flat_dq_des.reshape(N, -1),
        kp=topology.kp,
        kd=topology.kd,
        tau_grav_local=flat_tau_grav,
        s_axis=flat_s_axis,
        dt=cfg.dt,
    ).reshape(N, J, 3)

    # Noise-corrected 1-step empirical target from the simulator transition:
    # Anchored around physical PD lead + gravity/momentum/stamina compensation plus
    # bounded empirical 1-step position/velocity error correction (no unbounded integration).
    flat_mean_delta_u = rollout["delta_u"].reshape(N, J, 3).detach()
    explore_diff = flat_delta_u_exec - flat_mean_delta_u
    clean_pos_err = wrap_to_pi(flat_q_des - flat_q_next_actual) + 0.20 * explore_diff
    clean_vel_err = (flat_dq_des - flat_dq_next_actual) + 2.0 * explore_diff
    flat_ang_mom = (
        rollout["joint_features"]
        .reshape(N, J, F_dim)[..., 18:21]
        .detach()
        / 0.1
    )

    kp_3d = topology.kp.to(device=flat_q_t.device, dtype=flat_q_t.dtype).unsqueeze(0).clamp_min(1.0)
    kd_3d = topology.kd.to(device=flat_q_t.device, dtype=flat_q_t.dtype).unsqueeze(0)
    stamina_eff = flat_s_axis.clamp_min(0.20)
    tau_load = flat_tau_grav - 0.15 * flat_ang_mom
    lead_term = (
        (kd_3d / kp_3d) * flat_dq_des
        - (tau_load / (stamina_eff * kp_3d))
        + 0.025 * wrap_to_pi(flat_q_des - flat_q_t)
    )
    delta_u_target = (
        lead_term + 0.01 * clean_pos_err
    ).clamp(
        -minibrain.config.max_residual_offset,
        minibrain.config.max_residual_offset,
    ) * active_mask

    track_loss_val = 0.0
    pos_track_val = 0.0
    vel_track_val = 0.0

    for _ in range(max(1, cfg.policy_updates_per_iter)):
        policy_optimizer.zero_grad()
        mb_out = minibrain(
            flat_features,
            q_des=flat_q_des,
            q_current=flat_q_t,
            s_axis=flat_s_axis,
        )
        delta_u = mb_out["delta_u"]
        q_next_pred, dq_next_pred = minibrain.predict_next_joint_state(
            flat_features, delta_u, s_axis=flat_s_axis
        )

        pos_err = wrap_to_pi(q_next_pred - flat_q_des) * active_mask
        vel_err = (dq_next_pred - flat_dq_des) * active_mask

        pos_loss = pos_err.pow(2).sum() / active_denom
        vel_loss = vel_err.pow(2).sum() / active_denom
        mlp_residual = mb_out.get("raw_mlp_offset", delta_u)
        reg_loss = (mlp_residual * active_mask).pow(2).sum() / active_denom

        # Combine learned surrogate gradient with direct simulator error feedback target
        surrogate_weight = 0.01 * min(1.0, float(iteration) / float(max(1, cfg.warmstart_iters)))
        direct_target_loss = (
            ((delta_u - delta_u_target) * active_mask).pow(2).sum() / active_denom
        )
        loss_track = (
            surrogate_weight * (pos_loss + cfg.vel_weight * vel_loss)
            + 0.05 * reg_loss
            + direct_target_loss
        )

        loss_track.backward()
        policy_params = (
            list(minibrain.trunk.parameters())
            + list(minibrain.action_head.parameters())
            + [minibrain.physics_basis_weights]
        )
        nn.utils.clip_grad_norm_(policy_params, max_norm=2.0)
        policy_optimizer.step()

        track_loss_val = float(loss_track.item())
        pos_track_val = float(pos_loss.item())
        vel_track_val = float(vel_loss.item())

    # Restore requires_grad on dynamics_head
    for p in minibrain.dynamics_head.parameters():
        p.requires_grad_(True)

    return {
        "dynamics_loss": dyn_loss_val,
        "tracking_loss": track_loss_val,
        "pred_pos_mse": pos_track_val,
        "pred_vel_mse": vel_track_val,
    }


def compute_per_joint_gae(
    rewards: Tensor,
    values: Tensor,
    bootstrap_value: Tensor,
    gamma: float = 0.95,
    tau: float = 0.95,
) -> Tuple[Tensor, Tensor]:
    """Compute per-joint Generalized Advantage Estimation (GAE) across `[E, H, 23]`."""
    E, H, J = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros((E, J), device=rewards.device, dtype=rewards.dtype)
    next_val = bootstrap_value

    for t in reversed(range(H)):
        delta = rewards[:, t, :] + gamma * next_val - values[:, t, :]
        last_gae = delta + gamma * tau * last_gae
        advantages[:, t, :] = last_gae
        next_val = values[:, t, :]

    returns = advantages + values
    return advantages, returns


def update_mode_b_per_joint_ppo(
    minibrain: SharedJointMiniBrain,
    ppo_log_std: nn.Parameter,
    rollout: Dict[str, Tensor],
    ppo_optimizer: torch.optim.Optimizer,
    cfg: MiniBrainTrainerConfig,
    topology: Optional[SubtreeTopology] = None,
    iteration: int = 0,
) -> Dict[str, float]:
    """Mode B update: Per-Joint Short-Horizon PPO across all `E * 23` parallel joints."""
    if minibrain.value_head is None:
        raise RuntimeError("Mode B PPO requires SharedJointMiniBrain(include_value_head=True)")

    rewards = rollout["rewards"].detach()  # [E, H, 23]
    old_values = rollout["values"].detach()  # [E, H, 23]
    bootstrap_value = rollout["bootstrap_value"].detach()  # [E, 23]

    advantages, returns = compute_per_joint_gae(
        rewards=rewards,
        values=old_values,
        bootstrap_value=bootstrap_value,
        gamma=cfg.gamma,
        tau=cfg.gae_tau,
    )

    E, H, J, F_dim = rollout["joint_features"].shape
    N = E * H

    flat_features = rollout["joint_features"].reshape(N, J, F_dim).detach()
    flat_actions = rollout["delta_u_exec"].reshape(N, J, 3).detach()
    flat_old_log_probs = rollout["log_probs"].reshape(N, J).detach()
    flat_advantages = advantages.reshape(N, J).detach()
    flat_returns = returns.reshape(N, J).detach()
    flat_q_t = rollout["q_t"].reshape(N, J, 3).detach()
    flat_dq_t = rollout["dq_t"].reshape(N, J, 3).detach()
    flat_q_next_actual = rollout["q_next_actual"].reshape(N, J, 3).detach()
    flat_q_des = rollout["q_des"].reshape(N, J, 3).detach()
    flat_dq_des = rollout["dq_des"].reshape(N, J, 3).detach()
    flat_s_axis = rollout["s_axis"].reshape(N, J, 3).detach()
    flat_tau_grav = rollout["tau_grav_local"].reshape(N, J, 3).detach()
    active_mask = (flat_s_axis > S_AXIS_ACTIVE_EPS).to(torch.float32)

    # Normalize advantages across all E * H * J joint transitions
    adv_mean = flat_advantages.mean()
    adv_std = flat_advantages.std().clamp_min(1e-6)
    norm_advantages = (flat_advantages - adv_mean) / adv_std

    u_analytical = None
    if topology is not None and cfg.warmstart_guidance_weight > 0.0:
        flat_mean_du = rollout["delta_u"].reshape(N, J, 3).detach()
        explore_diff = flat_actions - flat_mean_du
        clean_pos_err = wrap_to_pi(flat_q_des - flat_q_next_actual) + 0.20 * explore_diff
        flat_dq_next_actual = rollout["dq_next_actual"].reshape(N, J, 3).detach()
        clean_vel_err = (flat_dq_des - flat_dq_next_actual) + 2.0 * explore_diff
        flat_ang_mom = (
            rollout["joint_features"]
            .reshape(N, J, F_dim)[..., 18:21]
            .detach()
            / 0.1
        )
        kp_3d = topology.kp.to(device=flat_q_t.device, dtype=flat_q_t.dtype).unsqueeze(0).clamp_min(1.0)
        kd_3d = topology.kd.to(device=flat_q_t.device, dtype=flat_q_t.dtype).unsqueeze(0)
        stamina_eff = flat_s_axis.clamp_min(0.20)
        tau_load = flat_tau_grav - 0.15 * flat_ang_mom
        lead_term = (
            (kd_3d / kp_3d) * flat_dq_des
            - (tau_load / (stamina_eff * kp_3d))
            + 0.025 * wrap_to_pi(flat_q_des - flat_q_t)
        )
        u_analytical = (
            lead_term + 0.01 * clean_pos_err
        ).clamp(
            -minibrain.config.max_residual_offset,
            minibrain.config.max_residual_offset,
        ) * active_mask

    policy_loss_val = 0.0
    value_loss_val = 0.0
    entropy_val = 0.0
    ppo_warmup_scale = 0.02 * min(1.0, float(iteration) / float(max(1, cfg.warmstart_iters)))

    for _ in range(max(1, cfg.ppo_epochs)):
        ppo_optimizer.zero_grad()

        mb_out = minibrain(
            flat_features,
            q_des=flat_q_des,
            q_current=flat_q_t,
            s_axis=flat_s_axis,
        )
        mean_delta_u = mb_out["delta_u"]  # [N, J, 3]
        # Evaluate value head on detached trunk features so value updates do not
        # perturb action_head's trunk representation during hybrid training.
        trunk_detached = mb_out["trunk_features"].detach().reshape(N * J, -1)
        new_values = minibrain.value_head(trunk_detached).reshape(N, J)

        std = torch.exp(ppo_log_std).view(1, 1, -1).expand_as(mean_delta_u)
        var = std * std + 1e-8

        log_p_axis = -0.5 * (
            ((flat_actions - mean_delta_u) ** 2) / var
            + 2.0 * torch.log(std)
            + math.log(2.0 * math.pi)
        )
        new_log_probs = (log_p_axis * active_mask).sum(dim=-1)  # [N, J]

        ratio = torch.exp((new_log_probs - flat_old_log_probs).clamp(-10.0, 10.0))
        surr1 = ratio * norm_advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - cfg.ppo_clip_eps, 1.0 + cfg.ppo_clip_eps)
            * norm_advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = 0.5 * ((new_values - flat_returns) ** 2).mean()

        entropy_axis = 0.5 * (1.0 + math.log(2.0 * math.pi)) + torch.log(std)
        entropy = (entropy_axis * active_mask).sum(dim=-1).mean()

        total_loss = (
            ppo_warmup_scale * policy_loss
            + cfg.ppo_value_coef * value_loss
            - cfg.ppo_entropy_coef * entropy
        )
        if u_analytical is not None:
            active_denom = active_mask.sum().clamp_min(1.0)
            guidance = (
                ((mean_delta_u - u_analytical) * active_mask).pow(2).sum()
                / active_denom
            )
            total_loss = total_loss + max(0.25, cfg.warmstart_guidance_weight) * guidance

        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(minibrain.parameters()) + [ppo_log_std], max_norm=2.0
        )
        ppo_optimizer.step()

        policy_loss_val = float(policy_loss.item())
        value_loss_val = float(value_loss.item())
        entropy_val = float(entropy.item())

    return {
        "ppo_policy_loss": policy_loss_val,
        "ppo_value_loss": value_loss_val,
        "ppo_entropy": entropy_val,
        "mean_joint_reward": float(rewards.mean().item()),
    }


@torch.no_grad()
def evaluate_minibrain_vs_baselines(
    minibrain: SharedJointMiniBrain,
    env: SyntheticSMPLPhysicsEnv,
    horizon: int = 4,
    num_eval_envs: int = 64,
    seed: int = 1234,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Evaluate 1-step and H-step joint position MSE (`rad`) and velocity MSE (`rad/s`) across 4 regimes.

    Regimes evaluated:
      - `nominal` (`s_axis = 1.0`, normal gravity)
      - `high_gravity_tilt` (tilted root orientations)
      - `low_stamina` (`s_axis in [0.2, 0.5]`)
      - `locked_1_or_2_dofs` (1 or 2 axes per joint set to `0.0`)

    Controllers compared side-by-side:
      1. `Baseline PD (u = q_des)`
      2. `Analytical PD Lead (compute_analytical_pd_lead_action)`
      3. `SharedJointMiniBrain`
    """
    was_training = minibrain.training
    was_frozen = minibrain._freeze_running
    minibrain.eval()
    minibrain._freeze_running = True

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for regime_idx, regime in enumerate(BENCHMARK_REGIMES):
        gen = torch.Generator(device=env.device).manual_seed(seed + 100 * regime_idx)
        reset_data = sample_perturbed_smpl_reset(
            num_envs=num_eval_envs,
            horizon=horizon,
            topology=env.topology,
            dt=env.dt,
            regime=regime,
            motion_lib=getattr(env, "motion_lib", None),
            device=env.device,
            generator=gen,
        )
        results[regime] = {}

        for controller_name in CONTROLLER_NAMES:
            env.reset_to_state(reset_data)
            active_mask = (env.s_axis > S_AXIS_ACTIVE_EPS).to(torch.float32)
            active_denom = active_mask.sum().clamp_min(1.0)

            step1_pos_mse = 0.0
            step1_vel_mse = 0.0
            horizon_pos_mses: List[float] = []
            horizon_vel_mses: List[float] = []

            for step_k in range(horizon):
                q_t = env.q
                dq_t = env.dq
                q_des_t = reset_data["q_des_seq"][:, step_k]
                dq_des_t = reset_data["dq_des_seq"][:, step_k]
                s_axis_t = env.s_axis

                joint_features_t, tau_grav_t, _, _, _ = env.compute_load_and_features(
                    q_des=q_des_t, dq_des=dq_des_t
                )

                if controller_name == "Baseline PD (u = q_des)":
                    u_cmd = torch.where(active_mask > 0.5, q_des_t, q_t).reshape(
                        num_eval_envs, -1
                    )
                elif (
                    controller_name
                    == "Analytical PD Lead (compute_analytical_pd_lead_action)"
                ):
                    u_lead = compute_analytical_pd_lead_action(
                        q_j=q_t.reshape(num_eval_envs, -1),
                        dq_j=dq_t.reshape(num_eval_envs, -1),
                        q_des=q_des_t.reshape(num_eval_envs, -1),
                        dq_des=dq_des_t.reshape(num_eval_envs, -1),
                        kp=env.topology.kp,
                        kd=env.topology.kd,
                        tau_grav_local=tau_grav_t,
                        s_axis=s_axis_t,
                        dt=env.dt,
                    ).reshape(num_eval_envs, env.num_joints, 3)
                    u_cmd = torch.where(active_mask > 0.5, u_lead, q_t).reshape(
                        num_eval_envs, -1
                    )
                else:
                    u_cmd = minibrain.compute_pd_target(
                        joint_features=joint_features_t,
                        q_des=q_des_t,
                        q_current=q_t,
                        s_axis=s_axis_t,
                    )

                q_next, dq_next = env.step(u_cmd)

                pos_err_sq = (
                    (wrap_to_pi(q_next - q_des_t) * active_mask).pow(2).sum()
                    / active_denom
                )
                vel_err_sq = (
                    ((dq_next - dq_des_t) * active_mask).pow(2).sum() / active_denom
                )
                p_mse = float(pos_err_sq.item())
                v_mse = float(vel_err_sq.item())

                if step_k == 0:
                    step1_pos_mse = p_mse
                    step1_vel_mse = v_mse
                horizon_pos_mses.append(p_mse)
                horizon_vel_mses.append(v_mse)

            results[regime][controller_name] = {
                "step1_pos_mse": step1_pos_mse,
                "step1_vel_mse": step1_vel_mse,
                "horizon_pos_mse": float(sum(horizon_pos_mses) / len(horizon_pos_mses)),
                "horizon_vel_mse": float(sum(horizon_vel_mses) / len(horizon_vel_mses)),
            }

    minibrain._freeze_running = was_frozen
    if was_training:
        minibrain.train()
    return results


def format_benchmark_table(
    results: Dict[str, Dict[str, Dict[str, float]]],
    horizon: int = 4,
) -> str:
    """Format side-by-side 4-regime A/B benchmark comparison table."""
    lines: List[str] = []
    sep = "=" * 112
    lines.append(sep)
    lines.append(
        f"{'Regime':<20} | {'Controller':<46} | {'1-Step Pos MSE':>14} | "
        f"{'1-Step Vel MSE':>14} | {f'{horizon}-Step Pos MSE':>14} | {f'{horizon}-Step Vel MSE':>14}"
    )
    lines.append(sep)
    for regime in BENCHMARK_REGIMES:
        if regime not in results:
            continue
        for c_name in CONTROLLER_NAMES:
            if c_name not in results[regime]:
                continue
            m = results[regime][c_name]
            lines.append(
                f"{regime:<20} | {c_name:<46} | "
                f"{m['step1_pos_mse']:>14.6f} | {m['step1_vel_mse']:>14.6f} | "
                f"{m['horizon_pos_mse']:>14.6f} | {m['horizon_vel_mse']:>14.6f}"
            )
        lines.append("-" * 112)
    return "\n".join(lines)


def train_smpl_minibrain(
    cfg: MiniBrainTrainerConfig,
) -> Tuple[SharedJointMiniBrain, Dict[str, Any]]:
    """Run the short-horizon training loop and 4-regime A/B benchmark for `SharedJointMiniBrain`."""
    if cfg.horizon not in (1, 2, 4):
        raise ValueError(f"Expected horizon in {{1, 2, 4}}, got {cfg.horizon}")
    if cfg.mode not in ("surrogate", "ppo", "hybrid"):
        raise ValueError(
            f"Expected mode in {{'surrogate', 'ppo', 'hybrid'}}, got {cfg.mode!r}"
        )

    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)

    robot_cfg = SmplRobotConfig()
    topology = build_subtree_topology(
        robot_cfg.kinematic_info, robot_cfg.control, device=dev
    )
    env = SyntheticSMPLPhysicsEnv(
        num_envs=cfg.num_envs,
        dt=cfg.dt,
        substeps=cfg.substeps,
        device=dev,
        robot_cfg=robot_cfg,
        topology=topology,
    )
    if cfg.motion_file is not None and str(cfg.motion_file).strip() != "":
        from protomotions.components.motion_lib import MotionLib, MotionLibConfig

        env.motion_lib = MotionLib(
            MotionLibConfig(motion_file=str(cfg.motion_file)),
            device=str(dev),
        )
    else:
        env.motion_lib = None

    mb_cfg = SharedJointMiniBrainConfig(
        num_joints=topology.num_joints,
        dofs_per_joint=3,
        feature_dim=MINIBRAIN_FEATURE_DIM,
        include_value_head=True,
        include_dynamics_head=True,
    )
    minibrain = SharedJointMiniBrain(mb_cfg).to(dev)
    minibrain.unfreeze(unfreeze_running_stats=True)

    edge_case_collector = load_level2_edge_case_collector(
        cfg.edge_cases_file, num_joints=topology.num_joints, device=dev
    )

    dyn_optimizer = torch.optim.Adam(
        minibrain.dynamics_head.parameters(), lr=cfg.lr_dynamics
    )
    policy_optimizer = torch.optim.Adam(
        [
            {
                "params": list(minibrain.trunk.parameters())
                + list(minibrain.action_head.parameters()),
                "lr": min(cfg.lr_policy, 3e-4),
            },
            {
                "params": [minibrain.physics_basis_weights],
                "lr": max(0.15, cfg.lr_policy * 25.0),
            },
        ]
    )

    ppo_log_std = nn.Parameter(
        torch.full((3,), math.log(cfg.ppo_init_std), device=dev, dtype=torch.float32)
    )
    if cfg.mode == "hybrid":
        # In hybrid mode, Mode A updates trunk + action_head + physics_basis_weights,
        # while Mode B updates value_head + ppo_log_std so two separate Adam optimizers
        # never fight over the same parameters.
        ppo_optimizer = torch.optim.Adam(
            list(minibrain.value_head.parameters()) + [ppo_log_std],
            lr=cfg.lr_policy,
        )
    else:
        ppo_optimizer = torch.optim.Adam(
            [
                {
                    "params": list(minibrain.trunk.parameters())
                    + list(minibrain.action_head.parameters())
                    + list(minibrain.value_head.parameters())
                    + [ppo_log_std],
                    "lr": cfg.lr_policy,
                },
                {
                    "params": [minibrain.physics_basis_weights],
                    "lr": max(0.15, cfg.lr_policy * 25.0),
                },
            ]
        )

    history: List[Dict[str, float]] = []

    for it in range(cfg.num_iterations):
        use_ppo = cfg.mode in ("ppo", "hybrid")
        rollout = collect_short_horizon_rollout(
            env=env,
            minibrain=minibrain,
            cfg=cfg,
            edge_case_collector=edge_case_collector,
            generator=gen,
            explore_std=cfg.surrogate_explore_std if cfg.mode == "surrogate" else 0.0,
            ppo_log_std=ppo_log_std if use_ppo else None,
        )

        metrics: Dict[str, float] = {"iteration": float(it)}
        if cfg.mode in ("surrogate", "hybrid"):
            surr_metrics = update_mode_a_surrogate(
                minibrain=minibrain,
                rollout=rollout,
                topology=topology,
                dyn_optimizer=dyn_optimizer,
                policy_optimizer=policy_optimizer,
                cfg=cfg,
                iteration=it,
            )
            metrics.update(surr_metrics)

        if cfg.mode in ("ppo", "hybrid"):
            ppo_metrics = update_mode_b_per_joint_ppo(
                minibrain=minibrain,
                ppo_log_std=ppo_log_std,
                rollout=rollout,
                ppo_optimizer=ppo_optimizer,
                cfg=cfg,
                topology=topology,
                iteration=it,
            )
            metrics.update(ppo_metrics)

        history.append(metrics)

    # Freeze trained Mini-Brain and run 4-regime A/B benchmark
    minibrain.freeze()
    benchmark_results = evaluate_minibrain_vs_baselines(
        minibrain=minibrain,
        env=env,
        horizon=cfg.horizon,
        num_eval_envs=cfg.eval_num_envs,
        seed=cfg.seed + 999,
    )

    table_str = format_benchmark_table(benchmark_results, horizon=cfg.horizon)
    print(table_str)

    # Save checkpoint and benchmark JSON
    minibrain.save_checkpoint(
        cfg.checkpoint_path,
        extra_metadata={
            "mode": cfg.mode,
            "horizon": cfg.horizon,
            "num_iterations": cfg.num_iterations,
            "benchmark": benchmark_results,
        },
    )

    bench_path = Path(cfg.benchmark_json_path)
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_payload = {
        "trainer_config": asdict(cfg),
        "benchmark_results": benchmark_results,
        "training_history_tail": history[-5:] if len(history) > 0 else [],
    }
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)

    return minibrain, benchmark_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and benchmark SharedJointMiniBrain on SMPL over short horizons."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="surrogate",
        choices=["surrogate", "ppo", "hybrid"],
        help="Training mode: 'surrogate' (Mode A), 'ppo' (Mode B), or 'hybrid'.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=4,
        choices=[1, 2, 4],
        help="Short rollout horizon H in {1, 2, 4} (default: 4).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
        help="Number of parallel environments (default: 64).",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=80,
        help="Number of short-horizon training iterations (default: 80).",
    )
    parser.add_argument(
        "--lockout-prob",
        type=float,
        default=0.2,
        help="Probability of 1- or 2-axis DOF lockout per joint (default: 0.2).",
    )
    parser.add_argument(
        "--stamina-min",
        type=float,
        default=0.2,
        help="Minimum randomized per-body stamina (default: 0.2).",
    )
    parser.add_argument(
        "--stamina-max",
        type=float,
        default=1.0,
        help="Maximum randomized per-body stamina (default: 1.0).",
    )
    parser.add_argument(
        "--edge-cases-file",
        type=str,
        default=None,
        help="Optional path to edge_cases_level2_minibrain.pt or EdgeCaseCollector directory.",
    )
    parser.add_argument(
        "--edge-case-prob",
        type=float,
        default=0.3,
        help="Fraction of resets sampled from Level 2 edge cases (default: 0.3).",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        default=None,
        help="Optional MotionLib .pt/.yaml file for reference state initialization.",
    )
    parser.add_argument(
        "--synthetic-physics",
        action="store_true",
        default=True,
        help="Use fast semi-implicit Euler SubtreeTopology physics stepper (default: True).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/smpl_minibrain.pt",
        help="Output path for frozen SharedJointMiniBrain checkpoint.",
    )
    parser.add_argument(
        "--benchmark-json-path",
        type=str,
        default="checkpoints/smpl_minibrain_benchmark.json",
        help="Output path for 4-regime A/B benchmark JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = MiniBrainTrainerConfig(
        mode=args.mode,
        horizon=args.horizon,
        num_envs=args.num_envs,
        num_iterations=args.num_iterations,
        lockout_prob=args.lockout_prob,
        stamina_min=args.stamina_min,
        stamina_max=args.stamina_max,
        edge_cases_file=args.edge_cases_file,
        edge_case_prob=args.edge_case_prob,
        motion_file=args.motion_file,
        synthetic_physics=args.synthetic_physics,
        checkpoint_path=args.checkpoint_path,
        benchmark_json_path=args.benchmark_json_path,
        device=args.device,
        seed=args.seed,
    )
    _, benchmark_payload = train_smpl_minibrain(cfg)
    return benchmark_payload


if __name__ == "__main__":
    main()
