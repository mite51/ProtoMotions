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

from types import SimpleNamespace

import torch

from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.obs.collision_primitives import (
    compute_collision_primitives_obs,
    compute_collision_priority,
)


def _bare_env(num_scenes: int = 2, num_characters: int = 2) -> BaseEnv:
    env = BaseEnv.__new__(BaseEnv)
    env.device = torch.device("cpu")
    env.num_physical_envs = num_scenes
    env.num_characters = num_characters
    env.num_envs = num_scenes * num_characters
    return env


def test_partial_reset_expands_to_complete_physical_scene():
    env = _bare_env()
    expanded = env._expand_to_physical_scenes(torch.tensor([1, 2]))
    assert torch.equal(expanded, torch.tensor([0, 1, 2, 3]))


def test_scene_flags_are_coupled_across_characters():
    env = _bare_env()
    coupled = env._couple_scene_flags(
        torch.tensor([False, True, False, False])
    )
    assert torch.equal(coupled, torch.tensor([True, True, False, False]))


class _MotionLib:
    def __init__(self, future_root_pos: torch.Tensor):
        self.future_root_pos = future_root_pos

    def get_motion_length(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return torch.full(motion_ids.shape, 10.0)

    def get_motion_state(self, motion_ids: torch.Tensor, motion_times: torch.Tensor):
        return SimpleNamespace(root_pos=self.future_root_pos)


def test_trajectory_placement_converges_near_shared_target():
    torch.manual_seed(3)
    env = _bare_env(num_scenes=1)
    env.config = SimpleNamespace(
        character_spawn_radius=1.0,
        character_spawn_radius_variance=0.0,
        character_interaction_lookahead=1.0,
        character_interaction_target_radius=0.2,
        character_min_spawn_separation=0.4,
    )
    env.motion_manager = SimpleNamespace(
        motion_ids=torch.tensor([0, 1]),
        motion_times=torch.zeros(2),
    )
    current_root = torch.tensor([[10.0, 5.0, 1.0], [-7.0, 3.0, 1.0]])
    displacement = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    future_root = current_root.clone()
    future_root[:, :2] += displacement
    env.motion_lib = _MotionLib(future_root)
    env.respawn_root_offset = torch.tensor(
        [[-10.0, -5.0, 0.05], [-10.0, -5.0, 0.05]]
    )
    ref_state = SimpleNamespace(
        root_pos=current_root,
        root_rot=torch.tensor(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
        ),
    )

    env._place_multi_character_reference_states(torch.tensor([0, 1]), ref_state)

    reset_xy = current_root[:, :2] + env.respawn_root_offset[:, :2]
    predicted_xy = reset_xy + displacement
    assert torch.all(predicted_xy.norm(dim=-1) < 0.25)
    assert torch.dist(reset_xy[0], reset_xy[1]) >= 0.4


def _priority_inputs():
    body_pos = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    body_vel = torch.zeros_like(body_pos)
    primitive_pos = torch.tensor(
        [[[2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [20.0, 0.0, 0.0]]]
    )
    primitive_vel = torch.tensor(
        [[[-5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-20.0, 0.0, 0.0]]]
    )
    mass = torch.tensor([[1.0, 20.0, 100.0]])
    valid = torch.ones(1, 3)
    return body_pos, body_vel, primitive_pos, primitive_vel, mass, valid


def test_priority_uses_closing_speed_mass_and_range():
    args = _priority_inputs()
    score = compute_collision_priority(
        *args,
        selection_range=8.0,
        distance_weight=1.0,
        closing_speed_weight=1.0,
        mass_weight=0.25,
        distance_scale=2.0,
        speed_scale=10.0,
        mass_scale=10.0,
    )
    assert score[0, 0] > score[0, 1]  # Fast incoming beats heavier stationary.
    assert score[0, 2] < -1.0e8  # Hard range gate beats speed and mass.

    mass_only = compute_collision_priority(
        *args,
        selection_range=8.0,
        distance_weight=0.0,
        closing_speed_weight=0.0,
        mass_weight=1.0,
        distance_scale=2.0,
        speed_scale=10.0,
        mass_scale=10.0,
    )
    assert mass_only[0, 1] > mass_only[0, 0]


def test_observation_keeps_shape_and_zero_pads_out_of_range_candidates():
    body_pos, body_vel, primitive_pos, primitive_vel, mass, valid = _priority_inputs()
    primitive_rot = torch.zeros(1, 3, 4)
    primitive_rot[..., 3] = 1.0
    obs = compute_collision_primitives_obs(
        body_pos=body_pos,
        body_rot=torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]]
        ),
        body_vel=body_vel,
        primitive_pos=primitive_pos,
        primitive_rot=primitive_rot,
        primitive_lin_vel=primitive_vel,
        primitive_radius=torch.ones(1, 3),
        primitive_extent_z=torch.zeros(1, 3),
        primitive_damage=torch.zeros(1, 3),
        primitive_mass=mass,
        primitive_shape=torch.zeros(1, 3, 2),
        primitive_valid=valid,
        num_obs_primitives=3,
        selection_range=8.0,
    )
    features = obs.view(1, 3, 17)
    assert obs.shape == (1, 51)
    assert torch.count_nonzero(features[0, 2]) == 0
