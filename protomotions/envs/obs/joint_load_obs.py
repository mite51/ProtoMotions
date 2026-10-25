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
"""Analytical subtree load observations and per-joint feature kernels for SharedJointMiniBrain.

Provides:
1. Static kinematic subtree topology extraction (`SubtreeTopology`, `build_subtree_topology`,
   `estimate_mjcf_body_masses`) for articulated humanoids (e.g., SMPL with B=24 bodies, J=23 joints).
2. Pure-tensor Level-2 / TorchScript-compatible kernels for computing:
   - Per-joint gravitational torque in joint parent/local frame (`tau_grav_local`)
   - Per-joint subtree linear/orbital + spin angular momentum in joint local frame (`ang_momentum_local`)
   - Static subtree mass (`subtree_mass`) and dynamic effective inertia (`effective_inertia`)
   - Flat observation tensor `compute_joint_load_obs` (`[E, J * 8]`)
   - Full per-joint feature tensor `compute_minibrain_joint_features` (`[E, J, 34]`)
   - Analytical PD lead baseline action `compute_analytical_pd_lead_action` (`[E, J * 3]`)
"""

import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import mujoco
import torch
from torch import Tensor

from protomotions.components.pose_lib import ControlInfo, KinematicInfo
from protomotions.utils.rotations import quat_rotate, quat_rotate_inverse

# Per-joint analytical load features:
# [tau_grav_x, tau_grav_y, tau_grav_z, L_mom_x, L_mom_y, L_mom_z, subtree_mass, effective_inertia]
JOINT_LOAD_FEATURE_DIM: int = 8

# Per-joint static control & limit parameters:
# [kp_norm, kd_norm, effort_limit_norm, armature_norm,
#  limit_lo_x, limit_lo_y, limit_lo_z, limit_hi_x, limit_hi_y, limit_hi_z]
MINIBRAIN_STATIC_PARAM_DIM: int = 10

# Total per-joint feature width for SharedJointMiniBrain:
# q_j(3) + dq_j(3) + delta_q(3) + delta_dq(3) + gravity_dir_local(3)
# + tau_grav_local(3) + ang_momentum_local(3) + subtree_mass_norm(1) + effective_inertia_norm(1)
# + vel_pred_error(3) + s_axis(3) + static_joint_params(5) = 34
MINIBRAIN_DYNAMIC_FEATURE_DIM: int = 34
MINIBRAIN_FEATURE_DIM: int = 34


@dataclass
class SubtreeTopology:
    """Static subtree topology and control metadata for articulated joints."""

    joint_body_indices: Tensor  # [J] (long): articulated body indices (1..23 for SMPL)
    parent_body_indices: Tensor  # [J] (long): parent body index for each articulated body
    subtree_mask: Tensor  # [J, B] (float32): 1.0 if body b is in subtree of joint j
    body_masses: Tensor  # [B] (float32): mass of each rigid body
    subtree_masses: Tensor  # [J] (float32): subtree_mask @ body_masses
    kp: Tensor  # [J, 3] (float32): joint PD stiffness per axis
    kd: Tensor  # [J, 3] (float32): joint PD damping per axis
    effort_limit: Tensor  # [J, 3] (float32): actuator effort limit per axis
    dof_limits_lower: Tensor  # [J, 3] (float32): lower joint angle limits (rad)
    dof_limits_upper: Tensor  # [J, 3] (float32): upper joint angle limits (rad)
    body_com_offset: Optional[Tensor] = None  # [B, 3] (float32): local CoM offset (ipos) per body

    @property
    def num_joints(self) -> int:
        return int(self.joint_body_indices.shape[0])

    @property
    def num_bodies(self) -> int:
        return int(self.body_masses.shape[0])

    def to(
        self,
        device: Union[torch.device, str],
        dtype: Optional[torch.dtype] = None,
    ) -> "SubtreeTopology":
        """Move all topology tensors to the target device (and optional float dtype)."""
        float_kwargs = {"device": device}
        if dtype is not None:
            float_kwargs["dtype"] = dtype

        return SubtreeTopology(
            joint_body_indices=self.joint_body_indices.to(device=device),
            parent_body_indices=self.parent_body_indices.to(device=device),
            subtree_mask=self.subtree_mask.to(**float_kwargs),
            body_masses=self.body_masses.to(**float_kwargs),
            subtree_masses=self.subtree_masses.to(**float_kwargs),
            kp=self.kp.to(**float_kwargs),
            kd=self.kd.to(**float_kwargs),
            effort_limit=self.effort_limit.to(**float_kwargs),
            dof_limits_lower=self.dof_limits_lower.to(**float_kwargs),
            dof_limits_upper=self.dof_limits_upper.to(**float_kwargs),
            body_com_offset=(
                self.body_com_offset.to(**float_kwargs)
                if self.body_com_offset is not None
                else None
            ),
        )


def estimate_mjcf_body_masses(mjcf_path: str) -> Tensor:
    """Extract rigid-body masses [B] from an MJCF XML file using MuJoCo.

    Note: MuJoCo's `model.body_mass[0]` is the static `world` body, so the
    articulated robot bodies (starting at root `Pelvis`) correspond to `body_mass[1:]`.
    """
    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
    masses = mj_model.body_mass[1:].astype("float32")
    return torch.from_numpy(masses)


def estimate_mjcf_body_com_offsets(mjcf_path: str) -> Tensor:
    """Extract rigid-body local Center-of-Mass offsets `body_ipos` `[B, 3]` from an MJCF XML file."""
    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
    ipos = mj_model.body_ipos[1:].astype("float32")
    return torch.from_numpy(ipos)


def _resolve_dof_control_params(
    dof_names: list,
    control_info: Optional[Union[Dict[str, ControlInfo], object]],
    default_kp: float = 800.0,
    default_kd: float = 80.0,
    default_effort: float = 500.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Resolve per-DOF kp, kd, effort_limit tensors from control_info."""
    num_dofs = len(dof_names)
    kp = torch.full((num_dofs,), default_kp, dtype=torch.float32)
    kd = torch.full((num_dofs,), default_kd, dtype=torch.float32)
    effort = torch.full((num_dofs,), default_effort, dtype=torch.float32)

    if control_info is None:
        return kp, kd, effort

    # Unwrap ControlConfig if passed directly
    if hasattr(control_info, "control_info"):
        control_dict = getattr(control_info, "control_info")
    elif hasattr(control_info, "override_control_info"):
        control_dict = getattr(control_info, "override_control_info")
    elif isinstance(control_info, dict):
        control_dict = control_info
    else:
        return kp, kd, effort

    if not control_dict:
        return kp, kd, effort

    for idx, dof_name in enumerate(dof_names):
        matched_info: Optional[ControlInfo] = None
        if dof_name in control_dict:
            matched_info = control_dict[dof_name]
        else:
            for pattern, info in control_dict.items():
                if re.fullmatch(pattern, dof_name):
                    matched_info = info
                    break

        if matched_info is not None:
            if getattr(matched_info, "stiffness", None) is not None:
                kp[idx] = float(matched_info.stiffness)
            if getattr(matched_info, "damping", None) is not None:
                kd[idx] = float(matched_info.damping)
            if getattr(matched_info, "effort_limit", None) is not None:
                effort[idx] = float(matched_info.effort_limit)

    return kp, kd, effort


def build_subtree_topology(
    kinematic_info: KinematicInfo,
    control_info: Optional[Union[Dict[str, ControlInfo], object]] = None,
    body_masses: Optional[Tensor] = None,
    device: Union[torch.device, str] = "cpu",
) -> SubtreeTopology:
    """Construct `SubtreeTopology` from `KinematicInfo` and optional control/mass data.

    Args:
        kinematic_info: `KinematicInfo` (or `RobotConfig` containing `.kinematic_info`).
        control_info: Optional `Dict[str, ControlInfo]` or `ControlConfig` with PD gains.
        body_masses: Optional `[B]` body mass tensor. If None, attempts to load from
            `mjcf/smpl_humanoid.xml` when matching SMPL body count, else defaults to 3.0 kg/body.
        device: Target torch device for returned tensors.

    Returns:
        Populated `SubtreeTopology` on `device`.
    """
    if hasattr(kinematic_info, "kinematic_info"):
        robot_cfg = kinematic_info
        if control_info is None and hasattr(robot_cfg, "control"):
            control_info = robot_cfg.control
        kinematic_info = robot_cfg.kinematic_info

    num_bodies = kinematic_info.num_bodies
    parent_indices = list(kinematic_info.parent_indices)

    # Identify articulated joint bodies in kinematic order
    joint_body_list = [
        b_idx for b_idx in range(num_bodies) if b_idx in kinematic_info.hinge_axes_map
    ]
    num_joints = len(joint_body_list)

    joint_body_indices = torch.tensor(joint_body_list, dtype=torch.long)
    parent_body_indices = torch.tensor(
        [parent_indices[b_idx] for b_idx in joint_body_list], dtype=torch.long
    )

    # Build binary subtree mask S [J, B]: S[j, b] = 1.0 iff joint_body_indices[j] is an ancestor of b (or b itself)
    subtree_mask = torch.zeros((num_joints, num_bodies), dtype=torch.float32)
    joint_body_to_j = {b_idx: j_idx for j_idx, b_idx in enumerate(joint_body_list)}

    for b_idx in range(num_bodies):
        curr = b_idx
        visited = set()
        while curr != -1 and curr not in visited:
            visited.add(curr)
            if curr in joint_body_to_j:
                subtree_mask[joint_body_to_j[curr], b_idx] = 1.0
            curr = parent_indices[curr]

    # Resolve body masses [B] and local CoM offsets [B, 3]
    resolved_com_offsets: Optional[Tensor] = None
    if num_bodies == 24 and kinematic_info.body_names[0] == "Pelvis":
        try:
            from protomotions.assets import DEFAULT_ASSET_ROOT, resolve_asset_root

            smpl_xml = os.path.join(
                resolve_asset_root(DEFAULT_ASSET_ROOT), "mjcf/smpl_humanoid.xml"
            )
            if os.path.exists(smpl_xml):
                resolved_com_offsets = estimate_mjcf_body_com_offsets(smpl_xml)
                if body_masses is None:
                    body_masses = estimate_mjcf_body_masses(smpl_xml)
        except Exception:
            resolved_com_offsets = None

    if body_masses is None:
        body_masses = torch.full((num_bodies,), 3.0, dtype=torch.float32)
    else:
        body_masses = body_masses.detach().clone().to(dtype=torch.float32, device="cpu")
        if body_masses.numel() != num_bodies:
            raise ValueError(
                f"body_masses length ({body_masses.numel()}) != num_bodies ({num_bodies})"
            )

    if resolved_com_offsets is None or resolved_com_offsets.shape != (num_bodies, 3):
        body_com_offset = torch.zeros((num_bodies, 3), dtype=torch.float32)
    else:
        body_com_offset = resolved_com_offsets.to(dtype=torch.float32)

    subtree_masses = torch.matmul(subtree_mask, body_masses)

    # Extract per-DOF control parameters and reshape to [J, 3]
    kp_flat, kd_flat, effort_flat = _resolve_dof_control_params(
        kinematic_info.dof_names, control_info
    )
    limits_lo_flat = kinematic_info.dof_limits_lower.detach().cpu().to(torch.float32)
    limits_hi_flat = kinematic_info.dof_limits_upper.detach().cpu().to(torch.float32)

    kp_per_joint = torch.zeros((num_joints, 3), dtype=torch.float32)
    kd_per_joint = torch.zeros((num_joints, 3), dtype=torch.float32)
    effort_per_joint = torch.zeros((num_joints, 3), dtype=torch.float32)
    limits_lo_per_joint = torch.full((num_joints, 3), -math.pi, dtype=torch.float32)
    limits_hi_per_joint = torch.full((num_joints, 3), math.pi, dtype=torch.float32)

    dof_offset = 0
    for j_idx, b_idx in enumerate(joint_body_list):
        axes = kinematic_info.hinge_axes_map[b_idx]
        n_dofs = int(axes.shape[0])
        if n_dofs == 3:
            kp_per_joint[j_idx] = kp_flat[dof_offset : dof_offset + 3]
            kd_per_joint[j_idx] = kd_flat[dof_offset : dof_offset + 3]
            effort_per_joint[j_idx] = effort_flat[dof_offset : dof_offset + 3]
            limits_lo_per_joint[j_idx] = limits_lo_flat[dof_offset : dof_offset + 3]
            limits_hi_per_joint[j_idx] = limits_hi_flat[dof_offset : dof_offset + 3]
        elif n_dofs == 1:
            kp_per_joint[j_idx] = kp_flat[dof_offset]
            kd_per_joint[j_idx] = kd_flat[dof_offset]
            effort_per_joint[j_idx] = effort_flat[dof_offset]
            limits_lo_per_joint[j_idx] = limits_lo_flat[dof_offset]
            limits_hi_per_joint[j_idx] = limits_hi_flat[dof_offset]
        dof_offset += n_dofs

    topology = SubtreeTopology(
        joint_body_indices=joint_body_indices,
        parent_body_indices=parent_body_indices,
        subtree_mask=subtree_mask,
        body_masses=body_masses,
        subtree_masses=subtree_masses,
        kp=kp_per_joint,
        kd=kd_per_joint,
        effort_limit=effort_per_joint,
        dof_limits_lower=limits_lo_per_joint,
        dof_limits_upper=limits_hi_per_joint,
        body_com_offset=body_com_offset,
    )
    return topology.to(device=device)


@torch.jit.script
def compute_analytical_subtree_load(
    body_pos: Tensor,
    body_rot: Tensor,
    body_vel: Tensor,
    body_ang_vel: Tensor,
    subtree_mask: Tensor,
    body_masses: Tensor,
    joint_body_indices: Tensor,
    parent_body_indices: Tensor,
    gravity_vec: Optional[Tensor] = None,
    w_last: bool = True,
    body_com_offset: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute analytical subtree load quantities for each articulated joint.

    Args:
        body_pos: Rigid body world positions `[E, B, 3]`.
        body_rot: Rigid body world orientations `[E, B, 4]`.
        body_vel: Rigid body world linear velocities `[E, B, 3]`.
        body_ang_vel: Rigid body world angular velocities `[E, B, 3]`.
        subtree_mask: Binary descendant mask `[J, B]`.
        body_masses: Rigid body masses `[B]`.
        joint_body_indices: Articulated body indices `[J]`.
        parent_body_indices: Parent body indices for each joint `[J]`.
        gravity_vec: Optional world gravity vector `[3]` or `[E, 3]` or `[E, 1, 3]`.
            Defaults to `[0.0, 0.0, -9.81]`.
        w_last: Quaternion convention (`True` for xyzw, `False` for wxyz).

    Returns:
        Tuple of:
        - `tau_grav_local`: `[E, J, 3]` gravitational torque in parent body frame.
        - `ang_momentum_local`: `[E, J, 3]` subtree angular momentum in parent body frame.
        - `subtree_mass`: `[E, J, 1]` static subtree mass expanded across environments.
        - `effective_inertia`: `[E, J, 1]` dynamic subtree moment of inertia about joint `j`.
    """
    num_envs = body_pos.shape[0]
    num_joints = joint_body_indices.shape[0]

    if gravity_vec is None:
        g_world = torch.zeros(
            (num_envs, 1, 3), device=body_pos.device, dtype=body_pos.dtype
        )
        g_world[..., 2] = -9.81
    elif gravity_vec.dim() == 1:
        g_world = (
            gravity_vec.to(device=body_pos.device, dtype=body_pos.dtype)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(num_envs, 1, 3)
        )
    elif gravity_vec.dim() == 2:
        g_world = gravity_vec.to(
            device=body_pos.device, dtype=body_pos.dtype
        ).unsqueeze(1)
    else:
        g_world = gravity_vec.to(device=body_pos.device, dtype=body_pos.dtype)

    # Joint pivot positions & velocities in world frame: [E, J, 3]
    p_j = body_pos[:, joint_body_indices, :]
    v_j = body_vel[:, joint_body_indices, :]

    # Compute true body Center-of-Mass world positions and velocities when body_com_offset is provided
    if body_com_offset is not None:
        num_bodies = body_pos.shape[1]
        com_local = (
            body_com_offset.to(device=body_pos.device, dtype=body_pos.dtype)
            .unsqueeze(0)
            .expand(num_envs, num_bodies, 3)
        )
        com_world_offset = quat_rotate(
            body_rot.reshape(-1, 4), com_local.reshape(-1, 3), w_last
        ).reshape(num_envs, num_bodies, 3)
        c_b = body_pos + com_world_offset
        v_b = body_vel + torch.cross(body_ang_vel, com_world_offset, dim=-1)
    else:
        c_b = body_pos
        v_b = body_vel

    # Relative vectors from joint pivot j to body CoM b: [E, J, B, 3]
    r_jb = c_b.unsqueeze(1) - p_j.unsqueeze(2)
    v_rel = v_b.unsqueeze(1) - v_j.unsqueeze(2)

    # Subtree mass weights S_{j, b} * m_b: [J, B]
    mask_f = subtree_mask.to(device=body_pos.device, dtype=body_pos.dtype)
    masses_f = body_masses.to(device=body_pos.device, dtype=body_pos.dtype)
    mask_mass = mask_f * masses_f.unsqueeze(0)  # [J, B]

    # 1. Subtree gravitational torque in world frame:
    # sum_{b in Subtree(j)} (p_b - p_j) x (m_b * g) = (sum_{b} m_b (p_b - p_j)) x g
    weighted_r = (r_jb * mask_mass.unsqueeze(0).unsqueeze(-1)).sum(dim=2)  # [E, J, 3]
    g_expanded = g_world.expand(num_envs, num_joints, 3)
    tau_grav_world = torch.cross(weighted_r, g_expanded, dim=-1)  # [E, J, 3]

    # 2. Subtree linear/orbital + approximate body spin angular momentum in world frame:
    # sum_{b in Subtree(j)} m_b * ((p_b - p_j) x (v_b - v_j) + 0.02 * omega_b)
    orbital_cross = torch.cross(r_jb, v_rel, dim=-1)  # [E, J, B, 3]
    spin_term = 0.02 * body_ang_vel.unsqueeze(1).expand_as(orbital_cross)
    ang_mom_world = (
        (orbital_cross + spin_term) * mask_mass.unsqueeze(0).unsqueeze(-1)
    ).sum(dim=2)  # [E, J, 3]

    # Rotate world torque and angular momentum into parent body frame
    parent_rot = body_rot[:, parent_body_indices, :]  # [E, J, 4]
    flat_parent_rot = parent_rot.reshape(-1, 4)
    tau_grav_local = quat_rotate_inverse(
        flat_parent_rot, tau_grav_world.reshape(-1, 3), w_last
    ).reshape(num_envs, num_joints, 3)
    ang_momentum_local = quat_rotate_inverse(
        flat_parent_rot, ang_mom_world.reshape(-1, 3), w_last
    ).reshape(num_envs, num_joints, 3)

    # 3. Static subtree mass: [E, J, 1]
    subtree_mass_per_j = mask_mass.sum(dim=-1)  # [J]
    subtree_mass = (
        subtree_mass_per_j.unsqueeze(0)
        .unsqueeze(-1)
        .expand(num_envs, num_joints, 1)
        .clone()
    )

    # 4. Dynamic effective inertia: sum_{b in Subtree(j)} m_b * ||p_b - p_j||^2: [E, J, 1]
    r_sq = (r_jb * r_jb).sum(dim=-1)  # [E, J, B]
    effective_inertia = (r_sq * mask_mass.unsqueeze(0)).sum(dim=-1, keepdim=True)

    return tau_grav_local, ang_momentum_local, subtree_mass, effective_inertia


def compute_joint_load_obs(
    body_pos: Tensor,
    body_rot: Tensor,
    body_vel: Tensor,
    body_ang_vel: Tensor,
    subtree_mask: Tensor,
    body_masses: Tensor,
    joint_body_indices: Tensor,
    parent_body_indices: Tensor,
    gravity_vec: Optional[Tensor] = None,
    w_last: bool = True,
    torque_scale: float = 0.01,
    momentum_scale: float = 0.1,
    mass_scale: float = 0.05,
    inertia_scale: float = 0.5,
    body_com_offset: Optional[Tensor] = None,
) -> Tensor:
    """Compute flat `[E, J * 8]` analytical joint load observation for `MdpComponent` binding."""
    tau_grav_local, ang_momentum_local, subtree_mass, effective_inertia = (
        compute_analytical_subtree_load(
            body_pos=body_pos,
            body_rot=body_rot,
            body_vel=body_vel,
            body_ang_vel=body_ang_vel,
            subtree_mask=subtree_mask,
            body_masses=body_masses,
            joint_body_indices=joint_body_indices,
            parent_body_indices=parent_body_indices,
            gravity_vec=gravity_vec,
            w_last=w_last,
            body_com_offset=body_com_offset,
        )
    )
    per_joint_load = torch.cat(
        [
            tau_grav_local * torque_scale,
            ang_momentum_local * momentum_scale,
            subtree_mass * mass_scale,
            effective_inertia * inertia_scale,
        ],
        dim=-1,
    )
    return per_joint_load.reshape(body_pos.shape[0], -1)


def compute_minibrain_joint_features(
    dof_pos: Tensor,
    dof_vel: Tensor,
    q_des: Tensor,
    dq_des: Optional[Tensor] = None,
    body_pos: Optional[Tensor] = None,
    body_rot: Optional[Tensor] = None,
    body_vel: Optional[Tensor] = None,
    body_ang_vel: Optional[Tensor] = None,
    topology: Optional[SubtreeTopology] = None,
    subtree_mask: Optional[Tensor] = None,
    body_masses: Optional[Tensor] = None,
    joint_body_indices: Optional[Tensor] = None,
    parent_body_indices: Optional[Tensor] = None,
    kp: Optional[Tensor] = None,
    kd: Optional[Tensor] = None,
    effort_limit: Optional[Tensor] = None,
    dof_limits_lower: Optional[Tensor] = None,
    dof_limits_upper: Optional[Tensor] = None,
    body_com_offset: Optional[Tensor] = None,
    s_axis: Optional[Tensor] = None,
    body_stamina: Optional[Tensor] = None,
    dof_active_mask: Optional[Tensor] = None,
    dq_expected: Optional[Tensor] = None,
    vel_pred_error: Optional[Tensor] = None,
    gravity_vec: Optional[Tensor] = None,
    w_last: bool = True,
    torque_scale: float = 0.01,
    momentum_scale: float = 0.1,
    mass_scale: float = 0.05,
    inertia_scale: float = 0.5,
) -> Tensor:
    """Assemble the `[E, J, 34]` per-joint feature tensor for `SharedJointMiniBrain`.

    Feature layout per joint j (34 floats):
      0..3   (3): q_j                - current joint DOF angles [x, y, z]
      3..6   (3): dq_j               - current joint DOF velocities [x, y, z]
      6..9   (3): delta_q            - wrapped angle delta atan2(sin(q_des - q_j), cos(q_des - q_j))
      9..12  (3): delta_dq           - target velocity delta (dq_des - dq_j)
      12..15 (3): gravity_dir_local  - unit gravity direction in joint parent frame
      15..18 (3): tau_grav_local     - scaled analytical subtree gravity torque
      18..21 (3): ang_momentum_local - scaled analytical subtree angular momentum
      21..22 (1): subtree_mass_norm  - scaled static subtree mass
      22..23 (1): effective_inertia  - scaled dynamic effective inertia
      23..26 (3): vel_pred_error     - 1-step velocity prediction error (dq_j - dq_expected)
      26..29 (3): s_axis             - per-axis availability / stamina mask in [0, 1]
      29..34 (5): static_joint_params- [kp/1000, kd/100, effort/500, limit_span/pi, dist_to_limit_min]
    """
    device = dof_pos.device
    dtype = dof_pos.dtype
    num_envs = dof_pos.shape[0]

    if dof_pos.dim() == 2:
        num_joints = dof_pos.shape[1] // 3
        q_j = dof_pos.reshape(num_envs, num_joints, 3)
    elif dof_pos.dim() == 3:
        num_joints = dof_pos.shape[1]
        q_j = dof_pos
    else:
        raise ValueError(f"Expected dof_pos with 2 or 3 dims, got {dof_pos.shape}")

    dq_j = dof_vel.reshape(num_envs, num_joints, 3).to(device=device, dtype=dtype)
    q_des_3d = q_des.reshape(num_envs, num_joints, 3).to(device=device, dtype=dtype)

    if dq_des is None:
        dq_des_3d = torch.zeros_like(dq_j)
    else:
        dq_des_3d = dq_des.reshape(num_envs, num_joints, 3).to(
            device=device, dtype=dtype
        )

    # Unpack topology if provided
    if topology is not None:
        if subtree_mask is None:
            subtree_mask = topology.subtree_mask
        if body_masses is None:
            body_masses = topology.body_masses
        if joint_body_indices is None:
            joint_body_indices = topology.joint_body_indices
        if parent_body_indices is None:
            parent_body_indices = topology.parent_body_indices
        if kp is None:
            kp = topology.kp
        if kd is None:
            kd = topology.kd
        if effort_limit is None:
            effort_limit = topology.effort_limit
        if dof_limits_lower is None:
            dof_limits_lower = topology.dof_limits_lower
        if dof_limits_upper is None:
            dof_limits_upper = topology.dof_limits_upper
        if body_com_offset is None:
            body_com_offset = topology.body_com_offset

    # Resolve s_axis [E, J, 3] in [0, 1]
    if s_axis is not None:
        s_axis_t = s_axis.to(device=device, dtype=dtype)
        if s_axis_t.dim() == 1:
            if s_axis_t.numel() == num_joints:
                s_axis_3d = s_axis_t.view(1, num_joints, 1).expand(
                    num_envs, num_joints, 3
                )
            else:
                s_axis_3d = s_axis_t.view(1, num_joints, 3).expand(
                    num_envs, num_joints, 3
                )
        elif s_axis_t.dim() == 2:
            if s_axis_t.shape[1] == num_joints:
                s_axis_3d = s_axis_t.unsqueeze(-1).expand(num_envs, num_joints, 3)
            elif s_axis_t.shape[1] == num_joints * 3:
                s_axis_3d = s_axis_t.reshape(num_envs, num_joints, 3)
            elif s_axis_t.shape == (num_joints, 3):
                s_axis_3d = s_axis_t.unsqueeze(0).expand(num_envs, num_joints, 3)
            else:
                raise ValueError(f"Incompatible s_axis shape: {s_axis_t.shape}")
        else:
            s_axis_3d = s_axis_t.reshape(num_envs, num_joints, 3)
    elif body_stamina is not None:
        stamina_t = body_stamina.to(device=device, dtype=dtype)
        if stamina_t.shape[1] == num_joints:
            s_axis_3d = stamina_t.unsqueeze(-1).expand(num_envs, num_joints, 3)
        elif (
            joint_body_indices is not None
            and stamina_t.shape[1] > num_joints
        ):
            s_axis_3d = (
                stamina_t[:, joint_body_indices.to(device=device)]
                .unsqueeze(-1)
                .expand(num_envs, num_joints, 3)
            )
        else:
            s_axis_3d = stamina_t[:, :num_joints].unsqueeze(-1).expand(
                num_envs, num_joints, 3
            )
    else:
        s_axis_3d = torch.ones((num_envs, num_joints, 3), device=device, dtype=dtype)

    if dof_active_mask is not None:
        active_t = dof_active_mask.to(device=device, dtype=dtype)
        if active_t.dim() == 2 and active_t.shape == (num_joints, 3):
            active_t = active_t.unsqueeze(0)
        elif active_t.dim() == 2 and active_t.shape[1] == num_joints * 3:
            active_t = active_t.reshape(num_envs, num_joints, 3)
        s_axis_3d = s_axis_3d * active_t

    s_axis_3d = s_axis_3d.clamp(0.0, 1.0)

    # Wrapped target angle delta and velocity delta
    raw_delta_q = q_des_3d - q_j
    delta_q = torch.atan2(torch.sin(raw_delta_q), torch.cos(raw_delta_q))
    delta_dq = dq_des_3d - dq_j

    # Compute analytical subtree load and local gravity direction
    if (
        body_pos is not None
        and body_rot is not None
        and body_vel is not None
        and body_ang_vel is not None
        and subtree_mask is not None
        and body_masses is not None
        and joint_body_indices is not None
        and parent_body_indices is not None
    ):
        j_idx_dev = joint_body_indices.to(device=device)
        p_idx_dev = parent_body_indices.to(device=device)
        tau_grav_local, ang_momentum_local, subtree_mass, effective_inertia = (
            compute_analytical_subtree_load(
                body_pos=body_pos,
                body_rot=body_rot,
                body_vel=body_vel,
                body_ang_vel=body_ang_vel,
                subtree_mask=subtree_mask.to(device=device, dtype=dtype),
                body_masses=body_masses.to(device=device, dtype=dtype),
                joint_body_indices=j_idx_dev,
                parent_body_indices=p_idx_dev,
                gravity_vec=gravity_vec,
                w_last=w_last,
                body_com_offset=body_com_offset,
            )
        )

        # Unit gravity vector in joint parent frame
        if gravity_vec is None:
            unit_g = torch.zeros(
                (num_envs, num_joints, 3), device=device, dtype=dtype
            )
            unit_g[..., 2] = -1.0
        else:
            g_t = gravity_vec.to(device=device, dtype=dtype)
            if g_t.dim() == 1:
                g_t = g_t.view(1, 1, 3).expand(num_envs, num_joints, 3)
            elif g_t.dim() == 2:
                g_t = g_t.unsqueeze(1).expand(num_envs, num_joints, 3)
            unit_g = g_t / g_t.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)

        parent_rot = body_rot[:, p_idx_dev, :]
        gravity_dir_local = quat_rotate_inverse(
            parent_rot.reshape(-1, 4), unit_g.reshape(-1, 3), w_last
        ).reshape(num_envs, num_joints, 3)
    else:
        gravity_dir_local = torch.zeros(
            (num_envs, num_joints, 3), device=device, dtype=dtype
        )
        gravity_dir_local[..., 2] = -1.0
        tau_grav_local = torch.zeros(
            (num_envs, num_joints, 3), device=device, dtype=dtype
        )
        ang_momentum_local = torch.zeros(
            (num_envs, num_joints, 3), device=device, dtype=dtype
        )
        if subtree_mask is not None and body_masses is not None:
            sm = torch.matmul(
                subtree_mask.to(device=device, dtype=dtype),
                body_masses.to(device=device, dtype=dtype),
            )
            subtree_mass = sm.view(1, num_joints, 1).expand(num_envs, num_joints, 1)
        else:
            subtree_mass = torch.full(
                (num_envs, num_joints, 1), 3.0, device=device, dtype=dtype
            )
        effective_inertia = torch.zeros(
            (num_envs, num_joints, 1), device=device, dtype=dtype
        )

    # 1-step velocity prediction error
    if vel_pred_error is not None:
        vel_err_3d = vel_pred_error.reshape(num_envs, num_joints, 3).to(
            device=device, dtype=dtype
        )
    elif dq_expected is not None:
        vel_err_3d = dq_j - dq_expected.reshape(num_envs, num_joints, 3).to(
            device=device, dtype=dtype
        )
    else:
        vel_err_3d = torch.zeros_like(dq_j)

    # Static joint parameters (5 floats):
    # [kp / 1000.0, kd / 100.0, effort_limit / 500.0, limit_span_mean / pi, dist_to_limit_min]
    if kp is not None:
        kp_t = kp.to(device=device, dtype=dtype).reshape(-1, num_joints, 3)
        kp_norm = (kp_t.mean(dim=-1, keepdim=True) / 1000.0).expand(
            num_envs, num_joints, 1
        )
    else:
        kp_norm = torch.full(
            (num_envs, num_joints, 1), 0.8, device=device, dtype=dtype
        )

    if kd is not None:
        kd_t = kd.to(device=device, dtype=dtype).reshape(-1, num_joints, 3)
        kd_norm = (kd_t.mean(dim=-1, keepdim=True) / 100.0).expand(
            num_envs, num_joints, 1
        )
    else:
        kd_norm = torch.full(
            (num_envs, num_joints, 1), 0.8, device=device, dtype=dtype
        )

    if effort_limit is not None:
        eff_t = effort_limit.to(device=device, dtype=dtype).reshape(-1, num_joints, 3)
        effort_norm = (eff_t.mean(dim=-1, keepdim=True) / 500.0).expand(
            num_envs, num_joints, 1
        )
    else:
        effort_norm = torch.ones((num_envs, num_joints, 1), device=device, dtype=dtype)

    if dof_limits_lower is not None and dof_limits_upper is not None:
        lim_lo = (
            dof_limits_lower.to(device=device, dtype=dtype)
            .reshape(-1, num_joints, 3)
            .clamp(min=-2.0 * math.pi, max=2.0 * math.pi)
        )
        lim_hi = (
            dof_limits_upper.to(device=device, dtype=dtype)
            .reshape(-1, num_joints, 3)
            .clamp(min=-2.0 * math.pi, max=2.0 * math.pi)
        )
        limit_span_mean = (
            ((lim_hi - lim_lo).mean(dim=-1, keepdim=True) / math.pi)
            .expand(num_envs, num_joints, 1)
        )
        dist_lo = q_j - lim_lo
        dist_hi = lim_hi - q_j
        dist_to_limit_min = (
            torch.minimum(dist_lo, dist_hi).min(dim=-1, keepdim=True).values / math.pi
        )
    else:
        limit_span_mean = torch.full(
            (num_envs, num_joints, 1), 2.0, device=device, dtype=dtype
        )
        dist_to_limit_min = torch.ones(
            (num_envs, num_joints, 1), device=device, dtype=dtype
        )

    static_joint_params = torch.cat(
        [kp_norm, kd_norm, effort_norm, limit_span_mean, dist_to_limit_min], dim=-1
    )

    features = torch.cat(
        [
            q_j,  # 3
            dq_j,  # 3
            delta_q,  # 3
            delta_dq,  # 3
            gravity_dir_local,  # 3
            tau_grav_local * torque_scale,  # 3
            ang_momentum_local * momentum_scale,  # 3
            subtree_mass * mass_scale,  # 1
            effective_inertia * inertia_scale,  # 1
            vel_err_3d,  # 3
            s_axis_3d,  # 3
            static_joint_params,  # 5
        ],
        dim=-1,
    )
    return features


# Alias matching HIERARCHICAL_MINIBRAIN_PLAN.md naming
compute_joint_minibrain_features = compute_minibrain_joint_features


def compute_analytical_pd_lead_action(
    q_j: Tensor,
    dq_j: Tensor,
    q_des: Tensor,
    dq_des: Tensor,
    kp: Tensor,
    kd: Tensor,
    tau_grav_local: Optional[Tensor] = None,
    s_axis: Optional[Tensor] = None,
    dt: float = 1.0 / 30.0,
) -> Tensor:
    """Compute analytical PD lead + gravity compensation baseline action `[E, J * 3]`.

    Formula:
        a_analytical = q_des + ((kd / kp) * dq_des - (tau_grav_local / kp)) * (s_axis > 0)
    """
    del dq_j, dt  # Unused in closed-form PD lead law; kept for uniform controller signature
    num_envs = q_j.shape[0]
    device = q_des.device
    dtype = q_des.dtype

    q_des_3d = q_des.reshape(num_envs, -1, 3)
    num_joints = q_des_3d.shape[1]
    dq_des_3d = dq_des.reshape(num_envs, num_joints, 3).to(device=device, dtype=dtype)

    kp_3d = kp.to(device=device, dtype=dtype).reshape(-1, num_joints, 3).clamp_min(1.0)
    kd_3d = kd.to(device=device, dtype=dtype).reshape(-1, num_joints, 3)

    lead_offset = (kd_3d / kp_3d) * dq_des_3d
    if tau_grav_local is not None:
        tau_3d = tau_grav_local.reshape(num_envs, num_joints, 3).to(
            device=device, dtype=dtype
        )
        lead_offset = lead_offset - (tau_3d / kp_3d)

    if s_axis is not None:
        s_t = s_axis.to(device=device, dtype=dtype)
        if s_t.dim() == 2 and s_t.shape[1] == num_joints:
            active_mask = (s_t.unsqueeze(-1) > 0.0).to(dtype=dtype)
        else:
            active_mask = (s_t.reshape(num_envs, num_joints, 3) > 0.0).to(dtype=dtype)
        lead_offset = lead_offset * active_mask

    action = q_des_3d + lead_offset
    return action.reshape(num_envs, num_joints * 3)


__all__ = [
    "JOINT_LOAD_FEATURE_DIM",
    "MINIBRAIN_STATIC_PARAM_DIM",
    "MINIBRAIN_DYNAMIC_FEATURE_DIM",
    "MINIBRAIN_FEATURE_DIM",
    "SubtreeTopology",
    "estimate_mjcf_body_masses",
    "build_subtree_topology",
    "compute_analytical_subtree_load",
    "compute_joint_load_obs",
    "compute_minibrain_joint_features",
    "compute_joint_minibrain_features",
    "compute_analytical_pd_lead_action",
]
