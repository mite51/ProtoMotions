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
"""Unified collision-primitive observation kernel.

Encodes every collidable entity in the scene -- ground, static/dynamic obstacles,
thrown projectiles, and (in the multi-character tier) other characters' key bodies
-- into a single uniform egocentric representation, regardless of geometry.

Per-primitive 17-float layout (egocentric to the observing character's
heading-aligned hip frame):

    index  name         dim  description
    0:3    rel_pos       3   position relative to hips (heading frame)
    3:6    rel_lin_vel   3   linear velocity relative to hips (heading frame)
    6:9    local_tan     3   primitive local X-axis (tangent/forward), heading frame
    9:12   local_norm    3   primitive local Z-axis (normal/up), heading frame
    12     radius        1   sphere/capsule radius (0 for boxes)
    13     extent_z      1   box height / capsule cylinder length (0 for spheres)
    14     damage        1   threat scalar; higher == more harmful
    15:16  shape one-hot 2   [is_box, is_sphere]; capsule == [0, 0]

Aggregation: the environment maintains a fixed-capacity candidate buffer of M
primitives in WORLD frame (plus validity and physical-mass metadata). This kernel
range-gates candidates against every character body, ranks them uniformly by
distance, closing speed, and mass, keeps the K highest priorities, and zero-pads.
Output shape: ``[num_envs, K * 17]``.

The kernel is a pure tensor function (Level-2 / ONNX-exportable): all egocentric
math and top-K selection happen inside the graph, so external inference clients only
need to populate the world-frame candidate buffer.
"""

import torch
from torch import Tensor

from protomotions.utils import rotations

# Per-primitive feature width (see module docstring).
PRIMITIVE_FEATURE_DIM = 17


def compute_collision_priority(
    body_pos: Tensor,
    body_vel: Tensor,
    primitive_pos: Tensor,
    primitive_lin_vel: Tensor,
    primitive_mass: Tensor,
    primitive_valid: Tensor,
    selection_range: float,
    distance_weight: float,
    closing_speed_weight: float,
    mass_weight: float,
    distance_scale: float,
    speed_scale: float,
    mass_scale: float,
) -> Tensor:
    """Score candidates uniformly by proximity, contact likelihood, and mass.

    Distance and closing speed are evaluated against every character body rather
    than only the root. Invalid and out-of-range candidates receive a large
    negative score so they sort behind every eligible collider.
    """
    to_body = body_pos.unsqueeze(2) - primitive_pos.unsqueeze(1)
    body_distance = torch.norm(to_body, dim=-1).clamp_min(1.0e-6)
    nearest_distance = body_distance.amin(dim=1)

    relative_velocity = primitive_lin_vel.unsqueeze(1) - body_vel.unsqueeze(2)
    toward_body = (relative_velocity * (to_body / body_distance.unsqueeze(-1))).sum(
        dim=-1
    )
    max_closing_speed = toward_body.clamp_min(0.0).amax(dim=1)

    proximity = distance_scale / (nearest_distance + distance_scale)
    closing = max_closing_speed / (max_closing_speed + speed_scale)
    mass = primitive_mass.clamp_min(0.0)
    mass_priority = mass / (mass + mass_scale)
    score = (
        distance_weight * proximity
        + closing_speed_weight * closing
        + mass_weight * mass_priority
    )

    eligible = (primitive_valid > 0.5) & (nearest_distance <= selection_range)
    score = torch.where(eligible, score, torch.full_like(score, -1.0e9))

    # Stable preference for nearer and then earlier candidates when scores tie.
    candidate_index = torch.arange(
        primitive_pos.shape[1], device=primitive_pos.device, dtype=score.dtype
    ).unsqueeze(0)
    return score - nearest_distance * 1.0e-6 - candidate_index * 1.0e-8


def compute_collision_primitives_obs(
    body_pos: Tensor,
    body_rot: Tensor,
    body_vel: Tensor,
    primitive_pos: Tensor,
    primitive_rot: Tensor,
    primitive_lin_vel: Tensor,
    primitive_radius: Tensor,
    primitive_extent_z: Tensor,
    primitive_damage: Tensor,
    primitive_mass: Tensor,
    primitive_shape: Tensor,
    primitive_valid: Tensor,
    num_obs_primitives: int,
    selection_range: float = 8.0,
    distance_weight: float = 1.0,
    closing_speed_weight: float = 1.0,
    mass_weight: float = 0.25,
    distance_scale: float = 2.0,
    speed_scale: float = 10.0,
    mass_scale: float = 10.0,
    w_last: bool = True,
) -> Tensor:
    """Build the egocentric top-K collision-primitive observation.

    Args:
        body_pos: Robot body positions [num_envs, num_bodies, 3]. Root == index 0.
        body_rot: Robot body rotations [num_envs, num_bodies, 4] (quaternion).
        body_vel: Robot body linear velocities [num_envs, num_bodies, 3].
        primitive_pos: Candidate primitive world positions [num_envs, M, 3].
        primitive_rot: Candidate primitive world rotations [num_envs, M, 4].
        primitive_lin_vel: Candidate world linear velocities [num_envs, M, 3].
        primitive_radius: Sphere/capsule radius [num_envs, M] (0 for boxes).
        primitive_extent_z: Box height / capsule length [num_envs, M] (0 for spheres).
        primitive_damage: Threat scalar [num_envs, M].
        primitive_mass: Physical/effective mass [num_envs, M].
        primitive_shape: Shape one-hot [num_envs, M, 2] == [is_box, is_sphere].
        primitive_valid: Validity flag [num_envs, M] (1.0 active, 0.0 padding).
        num_obs_primitives: K, number of highest-priority primitives to emit.
        w_last: Quaternion convention (xyzw if True).

    Returns:
        Observation tensor [num_envs, num_obs_primitives * 17].
    """
    num_envs = primitive_pos.shape[0]
    num_candidates = primitive_pos.shape[1]
    k = num_obs_primitives

    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]
    root_vel = body_vel[:, 0, :]

    # Heading-only (yaw) inverse rotation defines the egocentric frame.
    heading_inv = rotations.calc_heading_quat_inv(root_rot, w_last)
    heading_inv_flat = (
        heading_inv.unsqueeze(1)
        .expand(num_envs, num_candidates, 4)
        .reshape(num_envs * num_candidates, 4)
    )

    # Relative position / velocity in the egocentric frame.
    rel_pos_w = (primitive_pos - root_pos.unsqueeze(1)).reshape(-1, 3)
    rel_vel_w = (primitive_lin_vel - root_vel.unsqueeze(1)).reshape(-1, 3)
    rel_pos = rotations.quat_rotate(heading_inv_flat, rel_pos_w, w_last).reshape(
        num_envs, num_candidates, 3
    )
    rel_lin_vel = rotations.quat_rotate(heading_inv_flat, rel_vel_w, w_last).reshape(
        num_envs, num_candidates, 3
    )

    # Primitive orientation in the egocentric frame -> continuous 6D tan/norm.
    prim_rot_local = rotations.quat_mul(
        heading_inv_flat, primitive_rot.reshape(-1, 4), w_last
    )
    tan_norm = rotations.quat_to_tan_norm(prim_rot_local, w_last).reshape(
        num_envs, num_candidates, 6
    )

    priority = compute_collision_priority(
        body_pos=body_pos,
        body_vel=body_vel,
        primitive_pos=primitive_pos,
        primitive_lin_vel=primitive_lin_vel,
        primitive_mass=primitive_mass,
        primitive_valid=primitive_valid,
        selection_range=selection_range,
        distance_weight=distance_weight,
        closing_speed_weight=closing_speed_weight,
        mass_weight=mass_weight,
        distance_scale=distance_scale,
        speed_scale=speed_scale,
        mass_scale=mass_scale,
    )

    # K highest-priority eligible colliders.
    k = min(k, num_candidates)
    _, topk_idx = torch.topk(priority, k, dim=1, largest=True)

    def _gather(x: Tensor) -> Tensor:
        if x.dim() == 2:
            return torch.gather(x, 1, topk_idx)
        channels = x.shape[-1]
        idx = topk_idx.unsqueeze(-1).expand(num_envs, k, channels)
        return torch.gather(x, 1, idx)

    g_rel_pos = _gather(rel_pos)
    g_rel_vel = _gather(rel_lin_vel)
    g_tan_norm = _gather(tan_norm)
    g_radius = _gather(primitive_radius).unsqueeze(-1)
    g_extent = _gather(primitive_extent_z).unsqueeze(-1)
    g_damage = _gather(primitive_damage).unsqueeze(-1)
    g_shape = _gather(primitive_shape)
    g_valid = _gather(primitive_valid)
    g_priority = _gather(priority)

    features = torch.cat(
        [g_rel_pos, g_rel_vel, g_tan_norm, g_radius, g_extent, g_damage, g_shape],
        dim=-1,
    )

    # Zero out padding slots (fewer than K valid primitives present).
    mask = (
        ((g_valid > 0.5) & (g_priority > -1.0e8))
        .to(features.dtype)
        .unsqueeze(-1)
    )
    features = features * mask

    return features.reshape(num_envs, k * PRIMITIVE_FEATURE_DIM)
