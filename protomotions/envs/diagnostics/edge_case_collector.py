# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-Level Edge-Case Collector & Diagnostic Triager for Hierarchical SMPL Control.

Maintains a lightweight ring buffer of the last K = window_length steps across all
parallel environments and automatically classifies error spikes and episode terminations
into three actionable diagnostic levels:
  - Level 1 (Analytical Load Mismatch): High 1-step velocity prediction error in free air
    (low external contact force). Points to unmodeled Coriolis / dynamic coupling in
    analytical subtree load features.
  - Level 2 (Mini-Brain Local Tracking Failure): Low external contact force and moderate
    analytical velocity prediction error, but at least one joint failed to track q_des.
    Points to SharedJointMiniBrain under/over-compensation.
  - Level 3 (Main Brain Balance / Impact Failure): External impact occurred OR all joints
    tracked q_des within tolerance, yet the character fell or terminated. Points to
    whole-body balance adjustment (delta_q) or impact bracing in the Main Brain.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor


class EdgeCaseLevel(IntEnum):
    """Diagnostic classification level for hierarchical controller failures."""

    NONE = 0
    LEVEL_1_ANALYTICAL_LOAD = 1
    LEVEL_2_MINIBRAIN_TRACKING = 2
    LEVEL_3_MAINBRAIN_BALANCE = 3


@dataclass
class EdgeCaseCollectorConfig:
    """Configuration for the multi-level edge-case ring buffer and diagnostic triager."""

    enabled: bool = True
    window_length: int = 8
    max_cases_per_level: int = 512
    num_joints: int = 23
    dofs_per_joint: int = 3
    num_bodies: int = 24
    level1_vel_err_threshold: float = 8.0
    level2_tracking_err_threshold: float = 0.35
    level3_tilt_threshold: float = 0.6
    contact_force_threshold: float = 50.0
    collect_on_termination_only: bool = False
    cooldown_steps: int = 16
    level3_lookback_frames: int = 4

    @property
    def num_dofs(self) -> int:
        return self.num_joints * self.dofs_per_joint


def compute_root_tilt_angle(root_rot: Tensor) -> Tensor:
    """
    Compute the root tilt angle (in radians) between the root's local up vector [0, 0, 1]
    and the world vertical [0, 0, 1], assuming xyzw quaternion ordering.

    Args:
        root_rot: Quaternion tensor of shape [..., 4] in xyzw order.

    Returns:
        Tilt angle in radians of shape [...] in [0, pi].
    """
    norm = torch.linalg.norm(root_rot, dim=-1, keepdim=True).clamp(min=1e-8)
    q = root_rot / norm
    x = q[..., 0]
    y = q[..., 1]
    # World z-component of rotated local [0, 0, 1] vector: 1 - 2*(x^2 + y^2)
    up_z = (1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0)
    return torch.acos(up_z)


def wrap_to_pi(angles: Tensor) -> Tensor:
    """Wrap angle differences to [-pi, pi]."""
    return torch.remainder(angles + torch.pi, 2.0 * torch.pi) - torch.pi


class EdgeCaseCollector:
    """
    Ring buffer and automatic 3-level diagnostic triager across parallel environments.
    """

    WINDOW_TENSOR_KEYS = (
        "root_pos",
        "root_rot",
        "root_vel",
        "root_ang_vel",
        "dof_pos",
        "dof_vel",
        "q_des",
        "dq_des",
        "main_brain_delta_q",
        "minibrain_delta_u",
        "s_axis",
        "tau_grav_local",
        "ang_momentum_local",
        "vel_pred_error",
        "contact_force_magnitudes",
        "motion_ids",
        "motion_times",
    )

    METADATA_KEYS = (
        "worst_joint_idx",
        "max_joint_tracking_err",
        "max_vel_pred_err",
        "max_contact_force",
        "root_tilt_err",
        "terminated",
        "valid_steps",
        "env_id",
        "level",
    )

    def __init__(
        self,
        config: Optional[EdgeCaseCollectorConfig] = None,
        num_envs: int = 1,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        if isinstance(config, int) and num_envs == 1:
            # Support EdgeCaseCollector(num_envs, ...) convenience invocation
            num_envs = int(config)
            config = EdgeCaseCollectorConfig()
        self.cfg = config if config is not None else EdgeCaseCollectorConfig()
        self.num_envs = int(num_envs)
        self.device = torch.device(device)

        E = self.num_envs
        K = self.cfg.window_length
        J = self.cfg.num_joints
        d = self.cfg.dofs_per_joint
        D = J * d
        B = self.cfg.num_bodies

        # Ring buffer tensors [E, K, ...]
        self.root_pos = torch.zeros((E, K, 3), dtype=torch.float32, device=self.device)
        self.root_rot = torch.zeros((E, K, 4), dtype=torch.float32, device=self.device)
        self.root_rot[..., 3] = 1.0  # Identity quaternion (xyzw)
        self.root_vel = torch.zeros((E, K, 3), dtype=torch.float32, device=self.device)
        self.root_ang_vel = torch.zeros((E, K, 3), dtype=torch.float32, device=self.device)

        self.dof_pos = torch.zeros((E, K, D), dtype=torch.float32, device=self.device)
        self.dof_vel = torch.zeros((E, K, D), dtype=torch.float32, device=self.device)
        self.q_des = torch.zeros((E, K, D), dtype=torch.float32, device=self.device)
        self.dq_des = torch.zeros((E, K, D), dtype=torch.float32, device=self.device)
        self.main_brain_delta_q = torch.zeros((E, K, D), dtype=torch.float32, device=self.device)
        self.minibrain_delta_u = torch.zeros((E, K, D), dtype=torch.float32, device=self.device)

        self.s_axis = torch.ones((E, K, J, d), dtype=torch.float32, device=self.device)
        self.tau_grav_local = torch.zeros((E, K, J, d), dtype=torch.float32, device=self.device)
        self.ang_momentum_local = torch.zeros((E, K, J, d), dtype=torch.float32, device=self.device)
        self.vel_pred_error = torch.zeros((E, K, J, d), dtype=torch.float32, device=self.device)

        self.contact_force_magnitudes = torch.zeros((E, K, B), dtype=torch.float32, device=self.device)
        self.motion_ids = torch.zeros((E, K), dtype=torch.long, device=self.device)
        self.motion_times = torch.zeros((E, K), dtype=torch.float32, device=self.device)

        # Per-environment ring buffer cursors & counters
        self.valid_steps = torch.zeros((E,), dtype=torch.long, device=self.device)
        self.write_head = torch.zeros((E,), dtype=torch.long, device=self.device)
        self.cooldown_counters = torch.zeros((E,), dtype=torch.long, device=self.device)

        # CPU reservoirs per diagnostic level
        self.buckets: Dict[EdgeCaseLevel, List[Dict[str, Any]]] = {
            EdgeCaseLevel.LEVEL_1_ANALYTICAL_LOAD: [],
            EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING: [],
            EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE: [],
        }
        self._write_cursor_per_level: Dict[EdgeCaseLevel, int] = {
            EdgeCaseLevel.LEVEL_1_ANALYTICAL_LOAD: 0,
            EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING: 0,
            EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE: 0,
        }
        self.total_triggered_per_level: Dict[EdgeCaseLevel, int] = {
            EdgeCaseLevel.LEVEL_1_ANALYTICAL_LOAD: 0,
            EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING: 0,
            EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE: 0,
        }

    def reset_envs(self, env_ids: Tensor) -> None:
        """Reset valid_steps, write_head, and cooldown counters for the specified environments."""
        if env_ids is None or env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.valid_steps[env_ids] = 0
        self.write_head[env_ids] = 0
        self.cooldown_counters[env_ids] = 0

        self.root_pos[env_ids] = 0.0
        self.root_rot[env_ids] = 0.0
        self.root_rot[env_ids, :, 3] = 1.0
        self.root_vel[env_ids] = 0.0
        self.root_ang_vel[env_ids] = 0.0
        self.dof_pos[env_ids] = 0.0
        self.dof_vel[env_ids] = 0.0
        self.q_des[env_ids] = 0.0
        self.dq_des[env_ids] = 0.0
        self.main_brain_delta_q[env_ids] = 0.0
        self.minibrain_delta_u[env_ids] = 0.0
        self.s_axis[env_ids] = 1.0
        self.tau_grav_local[env_ids] = 0.0
        self.ang_momentum_local[env_ids] = 0.0
        self.vel_pred_error[env_ids] = 0.0
        self.contact_force_magnitudes[env_ids] = 0.0
        self.motion_ids[env_ids] = 0
        self.motion_times[env_ids] = 0.0

    def _prepare_tensor(
        self,
        value: Optional[Tensor],
        shape: tuple,
        dtype: torch.dtype = torch.float32,
        default_val: float = 0.0,
    ) -> Tensor:
        if value is None:
            out = torch.full(shape, default_val, dtype=dtype, device=self.device)
            return out
        t = value.to(device=self.device, dtype=dtype)
        if t.shape != shape:
            t = t.reshape(shape)
        return t

    def extract_chronological_window(
        self, env_ids: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Extract the chronological K-step history [N, K, ...] for the requested env_ids
        (or all environments if env_ids is None).

        Chronological ordering rules:
          - Index K - 1 is always the most recently recorded step (frame t).
          - Index K - 2 is the immediately preceding step (frame t - 1).
          - When valid_steps[e] >= K, indices 0..K-1 correspond to frames (t - K + 1)..t
            unwrapped across the circular ring buffer boundary.
          - When 1 <= valid_steps[e] < K, the m valid frames occupy the trailing m slots
            [K - m .. K - 1] in chronological order, and the leading (K - m) slots replicate
            the earliest recorded frame (slot 0) so every frame is a valid simulator state.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)

        N = env_ids.numel()
        K = self.cfg.window_length
        heads = self.write_head[env_ids]  # number of steps written since reset (modulo wrapping)
        valid = self.valid_steps[env_ids]

        # Build index table [N, K] into the ring buffer axis (dim 1)
        steps_range = torch.arange(K, device=self.device, dtype=torch.long).unsqueeze(0)  # [1, K]
        # For wrapped envs (valid >= K), oldest is at (head - K) % K, newest at (head - 1) % K
        wrapped_idx = (heads.unsqueeze(1) - K + steps_range) % K  # [N, K]

        # For partially filled envs (1 <= valid < K), valid slots are 0 .. valid-1.
        # Place them at the tail [K - valid .. K - 1] and clamp leading positions to slot 0.
        partial_idx = (steps_range - (K - valid.unsqueeze(1))).clamp(min=0)
        partial_idx = torch.minimum(
            partial_idx, (valid.unsqueeze(1) - 1).clamp(min=0)
        )  # [N, K]

        is_wrapped = (valid >= K).unsqueeze(1)
        ring_indices = torch.where(is_wrapped, wrapped_idx, partial_idx)  # [N, K]

        batch_idx = env_ids.unsqueeze(1).expand(N, K)
        out: Dict[str, Tensor] = {}
        for key in self.WINDOW_TENSOR_KEYS:
            buf = getattr(self, key)
            out[key] = buf[batch_idx, ring_indices].clone()
        return out

    def triage_step(
        self,
        dof_pos: Tensor,
        q_des: Tensor,
        vel_pred_error: Tensor,
        contact_force_magnitudes: Tensor,
        root_rot: Optional[Tensor] = None,
        terminated: Optional[Tensor] = None,
        root_tilt_err: Optional[Tensor] = None,
        incoming_collider_impact: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Classify each environment into EdgeCaseLevel.{NONE, LEVEL_1, LEVEL_2, LEVEL_3}.

        Priority & Diagnostic Rules:
          - Level 1 (LEVEL_1_ANALYTICAL_LOAD):
            max_joint_vel_pred_err > level1_vel_err_threshold AND
            max_contact_force <= contact_force_threshold.
          - Level 2 (LEVEL_2_MINIBRAIN_TRACKING):
            max_contact_force <= contact_force_threshold AND
            max_joint_vel_pred_err <= level1_vel_err_threshold AND
            max_joint_tracking_err > level2_tracking_err_threshold.
          - Level 3 (LEVEL_3_MAINBRAIN_BALANCE):
            (terminated == True OR root_tilt_err > level3_tilt_threshold) AND
            (max_contact_force > contact_force_threshold OR incoming_collider_impact
             OR max_joint_tracking_err <= level2_tracking_err_threshold).
        """
        E = self.num_envs
        J = self.cfg.num_joints
        d = self.cfg.dofs_per_joint

        # 1. Per-joint position tracking error [E, J]
        q_diff = (dof_pos - q_des).reshape(E, J, d)
        joint_tracking_err = torch.linalg.norm(q_diff, dim=-1)  # [E, J]
        max_joint_tracking_err, worst_tracking_joint_idx = torch.max(joint_tracking_err, dim=-1)

        # 2. Per-joint 1-step velocity prediction error [E, J]
        vel_err_3d = vel_pred_error.reshape(E, J, d)
        joint_vel_err = torch.linalg.norm(vel_err_3d, dim=-1)  # [E, J]
        max_joint_vel_pred_err, worst_vel_joint_idx = torch.max(joint_vel_err, dim=-1)

        # 3. External contact force magnitude [E]
        if contact_force_magnitudes.dim() == 3:
            contact_mags = torch.linalg.norm(contact_force_magnitudes, dim=-1)
        else:
            contact_mags = contact_force_magnitudes.reshape(E, -1)
        max_contact_force = contact_mags.max(dim=-1).values

        # 4. Root tilt & termination flags [E]
        if root_tilt_err is None:
            if root_rot is not None:
                root_tilt_err = compute_root_tilt_angle(root_rot)
            else:
                root_tilt_err = torch.zeros((E,), dtype=torch.float32, device=self.device)
        else:
            root_tilt_err = root_tilt_err.to(device=self.device, dtype=torch.float32).reshape(E)

        if terminated is None:
            term_mask = torch.zeros((E,), dtype=torch.bool, device=self.device)
        else:
            term_mask = terminated.to(device=self.device).reshape(E).bool()

        if incoming_collider_impact is None:
            collider_mask = torch.zeros((E,), dtype=torch.bool, device=self.device)
        else:
            collider_mask = incoming_collider_impact.to(device=self.device).reshape(E).bool()

        low_contact = (max_contact_force <= self.cfg.contact_force_threshold) & (~collider_mask)
        high_contact = (max_contact_force > self.cfg.contact_force_threshold) | collider_mask

        high_vel_err = max_joint_vel_pred_err > self.cfg.level1_vel_err_threshold
        high_track_err = max_joint_tracking_err > self.cfg.level2_tracking_err_threshold
        fall_or_term = term_mask | (root_tilt_err > self.cfg.level3_tilt_threshold)

        # Triage into Level 1, Level 2, Level 3
        is_level1 = low_contact & high_vel_err
        is_level2 = low_contact & (~high_vel_err) & high_track_err
        is_level3 = (~is_level1) & (~is_level2) & fall_or_term & (
            high_contact | (~high_track_err)
        )

        levels = torch.full(
            (E,), int(EdgeCaseLevel.NONE), dtype=torch.long, device=self.device
        )
        levels = torch.where(
            is_level3,
            torch.tensor(int(EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE), device=self.device),
            levels,
        )
        levels = torch.where(
            is_level2,
            torch.tensor(int(EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING), device=self.device),
            levels,
        )
        levels = torch.where(
            is_level1,
            torch.tensor(int(EdgeCaseLevel.LEVEL_1_ANALYTICAL_LOAD), device=self.device),
            levels,
        )

        # Respect collect_on_termination_only and cooldown_steps
        if self.cfg.collect_on_termination_only:
            eligible = term_mask
        else:
            eligible = (self.cooldown_counters == 0) | term_mask

        levels = torch.where(
            eligible & (self.valid_steps > 0),
            levels,
            torch.tensor(int(EdgeCaseLevel.NONE), dtype=torch.long, device=self.device),
        )

        # For Level 1, worst_joint_idx reflects the joint with highest velocity prediction error
        # if tracking error is zero, otherwise for Level 2/3 it reflects worst tracking error joint.
        worst_joint_idx = torch.where(
            is_level1 & (~high_track_err),
            worst_vel_joint_idx,
            worst_tracking_joint_idx,
        )

        return {
            "levels": levels,
            "worst_joint_idx": worst_joint_idx,
            "worst_tracking_joint_idx": worst_tracking_joint_idx,
            "worst_vel_joint_idx": worst_vel_joint_idx,
            "max_joint_tracking_err": max_joint_tracking_err,
            "max_vel_pred_err": max_joint_vel_pred_err,
            "max_contact_force": max_contact_force,
            "root_tilt_err": root_tilt_err,
            "terminated": term_mask,
        }

    @torch.no_grad()
    def record_step(
        self,
        root_pos: Optional[Tensor] = None,
        root_rot: Optional[Tensor] = None,
        root_vel: Optional[Tensor] = None,
        root_ang_vel: Optional[Tensor] = None,
        dof_pos: Optional[Tensor] = None,
        dof_vel: Optional[Tensor] = None,
        q_des: Optional[Tensor] = None,
        dq_des: Optional[Tensor] = None,
        main_brain_delta_q: Optional[Tensor] = None,
        minibrain_delta_u: Optional[Tensor] = None,
        s_axis: Optional[Tensor] = None,
        tau_grav_local: Optional[Tensor] = None,
        ang_momentum_local: Optional[Tensor] = None,
        vel_pred_error: Optional[Tensor] = None,
        contact_force_magnitudes: Optional[Tensor] = None,
        motion_ids: Optional[Tensor] = None,
        motion_times: Optional[Tensor] = None,
        terminated: Optional[Tensor] = None,
        root_tilt_err: Optional[Tensor] = None,
        incoming_collider_impact: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Record the current step across all environments into the ring buffer, evaluate
        3-level diagnostic triage, and snapshot any triggered windows into CPU reservoirs.
        """
        E = self.num_envs
        K = self.cfg.window_length
        J = self.cfg.num_joints
        d = self.cfg.dofs_per_joint
        D = J * d
        B = self.cfg.num_bodies

        if not self.cfg.enabled:
            return {
                "levels": torch.zeros((E,), dtype=torch.long, device=self.device),
            }

        # Decrement active cooldown counters before evaluating current step
        self.cooldown_counters = (self.cooldown_counters - 1).clamp(min=0)

        root_pos_t = self._prepare_tensor(root_pos, (E, 3))
        if root_rot is None:
            root_rot_t = torch.zeros((E, 4), dtype=torch.float32, device=self.device)
            root_rot_t[:, 3] = 1.0
        else:
            root_rot_t = self._prepare_tensor(root_rot, (E, 4))
        root_vel_t = self._prepare_tensor(root_vel, (E, 3))
        root_ang_vel_t = self._prepare_tensor(root_ang_vel, (E, 3))

        dof_pos_t = self._prepare_tensor(dof_pos, (E, D))
        dof_vel_t = self._prepare_tensor(dof_vel, (E, D))
        q_des_t = self._prepare_tensor(q_des, (E, D))
        dq_des_t = self._prepare_tensor(dq_des, (E, D))
        main_brain_delta_q_t = self._prepare_tensor(main_brain_delta_q, (E, D))
        minibrain_delta_u_t = self._prepare_tensor(minibrain_delta_u, (E, D))

        s_axis_t = self._prepare_tensor(s_axis, (E, J, d), default_val=1.0)
        tau_grav_t = self._prepare_tensor(tau_grav_local, (E, J, d))
        ang_mom_t = self._prepare_tensor(ang_momentum_local, (E, J, d))
        vel_err_t = self._prepare_tensor(vel_pred_error, (E, J, d))

        if contact_force_magnitudes is not None and contact_force_magnitudes.dim() == 3:
            contact_mags_t = torch.linalg.norm(
                contact_force_magnitudes.to(device=self.device, dtype=torch.float32),
                dim=-1,
            ).reshape(E, B)
        else:
            contact_mags_t = self._prepare_tensor(contact_force_magnitudes, (E, B))

        motion_ids_t = self._prepare_tensor(motion_ids, (E,), dtype=torch.long)
        motion_times_t = self._prepare_tensor(motion_times, (E,), dtype=torch.float32)

        # Write into ring buffer slot (write_head % K) for each environment
        env_arange = torch.arange(E, device=self.device, dtype=torch.long)
        slots = self.write_head % K

        self.root_pos[env_arange, slots] = root_pos_t
        self.root_rot[env_arange, slots] = root_rot_t
        self.root_vel[env_arange, slots] = root_vel_t
        self.root_ang_vel[env_arange, slots] = root_ang_vel_t

        self.dof_pos[env_arange, slots] = dof_pos_t
        self.dof_vel[env_arange, slots] = dof_vel_t
        self.q_des[env_arange, slots] = q_des_t
        self.dq_des[env_arange, slots] = dq_des_t
        self.main_brain_delta_q[env_arange, slots] = main_brain_delta_q_t
        self.minibrain_delta_u[env_arange, slots] = minibrain_delta_u_t

        self.s_axis[env_arange, slots] = s_axis_t
        self.tau_grav_local[env_arange, slots] = tau_grav_t
        self.ang_momentum_local[env_arange, slots] = ang_mom_t
        self.vel_pred_error[env_arange, slots] = vel_err_t
        self.contact_force_magnitudes[env_arange, slots] = contact_mags_t
        self.motion_ids[env_arange, slots] = motion_ids_t
        self.motion_times[env_arange, slots] = motion_times_t

        self.write_head += 1
        self.valid_steps += 1

        # Evaluate triage rules on the current step
        triage_info = self.triage_step(
            dof_pos=dof_pos_t,
            q_des=q_des_t,
            vel_pred_error=vel_err_t,
            contact_force_magnitudes=contact_mags_t,
            root_rot=root_rot_t,
            terminated=terminated,
            root_tilt_err=root_tilt_err,
            incoming_collider_impact=incoming_collider_impact,
        )

        levels = triage_info["levels"]
        triggered_envs = torch.nonzero(levels != int(EdgeCaseLevel.NONE), as_tuple=False).flatten()

        if triggered_envs.numel() > 0:
            windows = self.extract_chronological_window(triggered_envs)
            windows_cpu = {k: v.detach().cpu() for k, v in windows.items()}

            levels_cpu = levels[triggered_envs].detach().cpu()
            worst_joint_cpu = triage_info["worst_joint_idx"][triggered_envs].detach().cpu()
            max_track_cpu = triage_info["max_joint_tracking_err"][triggered_envs].detach().cpu()
            max_vel_cpu = triage_info["max_vel_pred_err"][triggered_envs].detach().cpu()
            max_force_cpu = triage_info["max_contact_force"][triggered_envs].detach().cpu()
            tilt_cpu = triage_info["root_tilt_err"][triggered_envs].detach().cpu()
            term_cpu = triage_info["terminated"][triggered_envs].detach().cpu()
            valid_cpu = self.valid_steps[triggered_envs].detach().cpu()
            env_ids_cpu = triggered_envs.detach().cpu()

            for i in range(triggered_envs.numel()):
                lvl = EdgeCaseLevel(int(levels_cpu[i].item()))
                case_dict: Dict[str, Any] = {
                    k: windows_cpu[k][i] for k in self.WINDOW_TENSOR_KEYS
                }
                case_dict["worst_joint_idx"] = int(worst_joint_cpu[i].item())
                case_dict["max_joint_tracking_err"] = float(max_track_cpu[i].item())
                case_dict["max_vel_pred_err"] = float(max_vel_cpu[i].item())
                case_dict["max_contact_force"] = float(max_force_cpu[i].item())
                case_dict["root_tilt_err"] = float(tilt_cpu[i].item())
                case_dict["terminated"] = bool(term_cpu[i].item())
                case_dict["valid_steps"] = int(valid_cpu[i].item())
                case_dict["env_id"] = int(env_ids_cpu[i].item())
                case_dict["level"] = int(lvl)

                self._store_case(lvl, case_dict)

            # Set cooldown for triggered environments
            if self.cfg.cooldown_steps > 0:
                self.cooldown_counters[triggered_envs] = self.cfg.cooldown_steps

        return triage_info

    def _store_case(self, level: EdgeCaseLevel, case_dict: Dict[str, Any]) -> None:
        """Append or overwrite a case in the level's bounded CPU reservoir."""
        bucket = self.buckets[level]
        self.total_triggered_per_level[level] += 1
        max_cap = self.cfg.max_cases_per_level
        if len(bucket) < max_cap:
            bucket.append(case_dict)
        else:
            cursor = self._write_cursor_per_level[level] % max_cap
            bucket[cursor] = case_dict
        self._write_cursor_per_level[level] += 1

    def get_level_bucket(self, level: Union[int, EdgeCaseLevel]) -> List[Dict[str, Any]]:
        """Return list of recorded edge-case dicts for a given diagnostic level."""
        lvl = EdgeCaseLevel(int(level))
        return self.buckets[lvl]

    def collate_level_bucket(
        self, level: Union[int, EdgeCaseLevel]
    ) -> Dict[str, Tensor]:
        """
        Collate all cases in a level bucket into batched CPU tensors [N, K, ...]
        plus 1D metadata tensors [N].
        """
        lvl = EdgeCaseLevel(int(level))
        bucket = self.buckets[lvl]
        if len(bucket) == 0:
            return {"num_cases": torch.tensor(0, dtype=torch.long)}

        collated: Dict[str, Tensor] = {
            "num_cases": torch.tensor(len(bucket), dtype=torch.long)
        }
        for key in self.WINDOW_TENSOR_KEYS:
            collated[key] = torch.stack([c[key] for c in bucket], dim=0)

        collated["worst_joint_idx"] = torch.tensor(
            [c["worst_joint_idx"] for c in bucket], dtype=torch.long
        )
        collated["max_joint_tracking_err"] = torch.tensor(
            [c["max_joint_tracking_err"] for c in bucket], dtype=torch.float32
        )
        collated["max_vel_pred_err"] = torch.tensor(
            [c["max_vel_pred_err"] for c in bucket], dtype=torch.float32
        )
        collated["max_contact_force"] = torch.tensor(
            [c["max_contact_force"] for c in bucket], dtype=torch.float32
        )
        collated["root_tilt_err"] = torch.tensor(
            [c["root_tilt_err"] for c in bucket], dtype=torch.float32
        )
        collated["terminated"] = torch.tensor(
            [c["terminated"] for c in bucket], dtype=torch.bool
        )
        collated["valid_steps"] = torch.tensor(
            [c["valid_steps"] for c in bucket], dtype=torch.long
        )
        collated["env_id"] = torch.tensor(
            [c["env_id"] for c in bucket], dtype=torch.long
        )
        collated["level"] = torch.tensor(
            [c["level"] for c in bucket], dtype=torch.long
        )
        return collated

    def get_summary(self) -> Dict[str, Any]:
        """
        Compute diagnostic summary statistics across all three edge-case levels:
          - counts per level (stored and total triggered)
          - mean / max errors per level
          - most frequent failing joint indices (0..num_joints-1)
        """
        level_names = {
            EdgeCaseLevel.LEVEL_1_ANALYTICAL_LOAD: "level1_analytical_load",
            EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING: "level2_minibrain_tracking",
            EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE: "level3_mainbrain_balance",
        }

        summary: Dict[str, Any] = {
            "counts": {},
            "total_triggered": {},
            "levels": {},
            "joint_failure_histogram": [0] * self.cfg.num_joints,
            "most_frequent_failing_joints": [],
        }

        global_joint_hist = torch.zeros(self.cfg.num_joints, dtype=torch.long)

        for lvl, name in level_names.items():
            bucket = self.buckets[lvl]
            count = len(bucket)
            summary["counts"][name] = count
            summary["counts"][int(lvl)] = count
            summary["total_triggered"][name] = self.total_triggered_per_level[lvl]

            if count == 0:
                summary["levels"][name] = {
                    "count": 0,
                    "total_triggered": self.total_triggered_per_level[lvl],
                    "mean_joint_tracking_err": 0.0,
                    "max_joint_tracking_err": 0.0,
                    "mean_vel_pred_err": 0.0,
                    "max_vel_pred_err": 0.0,
                    "mean_contact_force": 0.0,
                    "max_contact_force": 0.0,
                    "most_frequent_failing_joints": [],
                    "joint_histogram": [0] * self.cfg.num_joints,
                }
                continue

            track_errs = torch.tensor(
                [c["max_joint_tracking_err"] for c in bucket], dtype=torch.float32
            )
            vel_errs = torch.tensor(
                [c["max_vel_pred_err"] for c in bucket], dtype=torch.float32
            )
            forces = torch.tensor(
                [c["max_contact_force"] for c in bucket], dtype=torch.float32
            )
            worst_joints = torch.tensor(
                [c["worst_joint_idx"] for c in bucket], dtype=torch.long
            )

            joint_hist = torch.bincount(
                worst_joints.clamp(0, self.cfg.num_joints - 1),
                minlength=self.cfg.num_joints,
            )
            global_joint_hist += joint_hist

            nonzero_joints = torch.nonzero(joint_hist > 0, as_tuple=False).flatten()
            sorted_order = torch.argsort(joint_hist[nonzero_joints], descending=True)
            top_joints = nonzero_joints[sorted_order].tolist()

            summary["levels"][name] = {
                "count": count,
                "total_triggered": self.total_triggered_per_level[lvl],
                "mean_joint_tracking_err": float(track_errs.mean().item()),
                "max_joint_tracking_err": float(track_errs.max().item()),
                "mean_vel_pred_err": float(vel_errs.mean().item()),
                "max_vel_pred_err": float(vel_errs.max().item()),
                "mean_contact_force": float(forces.mean().item()),
                "max_contact_force": float(forces.max().item()),
                "most_frequent_failing_joints": top_joints,
                "joint_histogram": joint_hist.tolist(),
            }

        summary["joint_failure_histogram"] = global_joint_hist.tolist()
        nonzero_global = torch.nonzero(global_joint_hist > 0, as_tuple=False).flatten()
        sorted_global = torch.argsort(global_joint_hist[nonzero_global], descending=True)
        summary["most_frequent_failing_joints"] = nonzero_global[sorted_global].tolist()

        return summary

    def save_all(self, directory: Union[str, Path]) -> Dict[str, Path]:
        """
        Save all three level reservoirs and the JSON diagnostic summary to `directory`:
          - edge_cases_level1_load.pt
          - edge_cases_level2_minibrain.pt
          - edge_cases_level3_mainbrain.pt
          - edge_cases_summary.json
        """
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "level1": out_dir / "edge_cases_level1_load.pt",
            "level2": out_dir / "edge_cases_level2_minibrain.pt",
            "level3": out_dir / "edge_cases_level3_mainbrain.pt",
            "summary": out_dir / "edge_cases_summary.json",
        }

        torch.save(
            {
                "config": asdict(self.cfg),
                "level": int(EdgeCaseLevel.LEVEL_1_ANALYTICAL_LOAD),
                "cases": self.buckets[EdgeCaseLevel.LEVEL_1_ANALYTICAL_LOAD],
                "collated": self.collate_level_bucket(EdgeCaseLevel.LEVEL_1_ANALYTICAL_LOAD),
            },
            paths["level1"],
        )
        torch.save(
            {
                "config": asdict(self.cfg),
                "level": int(EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING),
                "cases": self.buckets[EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING],
                "collated": self.collate_level_bucket(EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING),
            },
            paths["level2"],
        )
        torch.save(
            {
                "config": asdict(self.cfg),
                "level": int(EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE),
                "cases": self.buckets[EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE],
                "collated": self.collate_level_bucket(EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE),
            },
            paths["level3"],
        )

        summary = self.get_summary()
        # Convert integer keys in summary["counts"] to strings for JSON serialization
        json_summary = dict(summary)
        json_summary["counts"] = {str(k): v for k, v in summary["counts"].items()}
        with open(paths["summary"], "w", encoding="utf-8") as f:
            json.dump(json_summary, f, indent=2)

        return paths

    def load_all(self, directory: Union[str, Path]) -> None:
        """Load saved Level 1, Level 2, and Level 3 edge-case buckets from `directory`."""
        in_dir = Path(directory)
        file_map = {
            EdgeCaseLevel.LEVEL_1_ANALYTICAL_LOAD: in_dir / "edge_cases_level1_load.pt",
            EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING: in_dir / "edge_cases_level2_minibrain.pt",
            EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE: in_dir / "edge_cases_level3_mainbrain.pt",
        }
        for lvl, file_path in file_map.items():
            if file_path.exists():
                payload = torch.load(file_path, map_location="cpu", weights_only=False)
                cases = payload.get("cases", [])
                if cases:
                    first_root = cases[0].get("root_pos")
                    if isinstance(first_root, Tensor) and first_root.dim() >= 2:
                        self.cfg.window_length = int(first_root.shape[0])
                self.buckets[lvl] = list(cases)
                self._write_cursor_per_level[lvl] = len(self.buckets[lvl])
                self.total_triggered_per_level[lvl] = max(
                    self.total_triggered_per_level[lvl], len(self.buckets[lvl])
                )

    def sample_level2_minibrain_batch(
        self,
        batch_size: int,
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[str, Tensor]:
        """
        Sample a batch of initial states from the frame immediately *before* each Level 2
        failure (chronological window index K - 2) along with the target command so
        `train_smpl_minibrain.py` can directly replay and train on hard Level 2 edge cases.

        Returns a dict containing:
          (root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel,
           q_des, dq_des, s_axis, motion_ids, motion_times, worst_joint_idx)
        """
        bucket = self.buckets[EdgeCaseLevel.LEVEL_2_MINIBRAIN_TRACKING]
        if len(bucket) == 0:
            raise ValueError(
                "No Level 2 (Mini-Brain tracking) edge cases available to sample."
            )

        target_device = torch.device(device)
        idx = torch.randint(0, len(bucket), (batch_size,))

        def _pre_idx(case_dict: Dict[str, Any]) -> int:
            k_len = int(case_dict["root_pos"].shape[0])
            return max(0, k_len - 2)

        def _fail_idx(case_dict: Dict[str, Any]) -> int:
            k_len = int(case_dict["root_pos"].shape[0])
            return max(0, k_len - 1)

        batch: Dict[str, Tensor] = {
            "root_pos": torch.stack(
                [bucket[int(i)]["root_pos"][_pre_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
            "root_rot": torch.stack(
                [bucket[int(i)]["root_rot"][_pre_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
            "root_vel": torch.stack(
                [bucket[int(i)]["root_vel"][_pre_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
            "root_ang_vel": torch.stack(
                [bucket[int(i)]["root_ang_vel"][_pre_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
            "dof_pos": torch.stack(
                [bucket[int(i)]["dof_pos"][_pre_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
            "dof_vel": torch.stack(
                [bucket[int(i)]["dof_vel"][_pre_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
            "q_des": torch.stack(
                [bucket[int(i)]["q_des"][_fail_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
            "dq_des": torch.stack(
                [bucket[int(i)]["dq_des"][_fail_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
            "s_axis": torch.stack(
                [bucket[int(i)]["s_axis"][_pre_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
            "motion_ids": torch.stack(
                [bucket[int(i)]["motion_ids"][_pre_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
            "motion_times": torch.stack(
                [bucket[int(i)]["motion_times"][_pre_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
            "worst_joint_idx": torch.tensor(
                [bucket[int(i)]["worst_joint_idx"] for i in idx],
                dtype=torch.long,
                device=target_device,
            ),
        }
        return batch

    def sample_level3_mainbrain_resets(
        self,
        batch_size: int,
        device: Union[str, torch.device] = "cpu",
        lookback_frames: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """
        Sample reset states and (motion_ids, motion_times) several frames before a Level 3
        balance/impact failure so Main Brain PPO can prioritize hard balance transitions.

        Returns a dict containing:
          (root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel,
           motion_ids, motion_times, s_axis)
        """
        bucket = self.buckets[EdgeCaseLevel.LEVEL_3_MAINBRAIN_BALANCE]
        if len(bucket) == 0:
            raise ValueError(
                "No Level 3 (Main Brain balance/impact) edge cases available to sample."
            )

        target_device = torch.device(device)
        idx = torch.randint(0, len(bucket), (batch_size,))
        lookback = (
            self.cfg.level3_lookback_frames
            if lookback_frames is None
            else int(lookback_frames)
        )

        def _lb_idx(case_dict: Dict[str, Any]) -> int:
            k_len = int(case_dict["root_pos"].shape[0])
            return max(0, k_len - 1 - lookback)

        return {
            "root_pos": torch.stack(
                [bucket[int(i)]["root_pos"][_lb_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
            "root_rot": torch.stack(
                [bucket[int(i)]["root_rot"][_lb_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
            "root_vel": torch.stack(
                [bucket[int(i)]["root_vel"][_lb_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
            "root_ang_vel": torch.stack(
                [bucket[int(i)]["root_ang_vel"][_lb_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
            "dof_pos": torch.stack(
                [bucket[int(i)]["dof_pos"][_lb_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
            "dof_vel": torch.stack(
                [bucket[int(i)]["dof_vel"][_lb_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
            "s_axis": torch.stack(
                [bucket[int(i)]["s_axis"][_lb_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
            "motion_ids": torch.stack(
                [bucket[int(i)]["motion_ids"][_lb_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
            "motion_times": torch.stack(
                [bucket[int(i)]["motion_times"][_lb_idx(bucket[int(i)])] for i in idx], dim=0
            ).to(target_device),
        }
