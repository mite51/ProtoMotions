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

"""Unit tests for smooth velocity-error reference re-anchoring.

Covers the pure blend helper (blend strength, max-speed clamp) and confirms the
reset path still snaps the offset instantly (not smoothed).
"""

from types import SimpleNamespace

import torch

from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.base_env.utils import compute_smooth_realign_offset
from protomotions.envs.rewards.regularization import compute_realign_penalty


_PARAMS = dict(
    alpha_min=0.0,
    alpha_max=0.4,
    vel_err_low=0.3,
    vel_err_high=1.5,
    max_xy_speed=2.0,
    dt=1.0 / 30.0,
)


def test_zero_velocity_error_leaves_offset_unchanged():
    # Character tracks the clip velocity exactly -> vel_err = 0 -> alpha_min (0) ->
    # offset must not move even though roots are far apart (e.g. mid-getup).
    prev_offset = torch.tensor([[1.0, -2.0]])
    new_offset = compute_smooth_realign_offset(
        current_root_xy=torch.tensor([[5.0, 5.0]]),
        ref_root_xy=torch.tensor([[0.0, 0.0]]),
        current_root_xy_vel=torch.tensor([[1.0, 0.0]]),
        ref_root_xy_vel=torch.tensor([[1.0, 0.0]]),
        prev_offset_xy=prev_offset,
        **_PARAMS,
    )
    assert torch.allclose(new_offset, prev_offset)


def test_high_velocity_error_moves_offset_toward_target():
    # Large velocity mismatch (a shove) -> offset should move toward the instant-snap
    # target (current_root_xy - ref_root_xy).
    prev_offset = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[0.05, 0.0]])  # current - ref, small so clamp doesn't bind
    new_offset = compute_smooth_realign_offset(
        current_root_xy=target.clone(),
        ref_root_xy=torch.tensor([[0.0, 0.0]]),
        current_root_xy_vel=torch.tensor([[5.0, 0.0]]),
        ref_root_xy_vel=torch.tensor([[0.0, 0.0]]),
        prev_offset_xy=prev_offset,
        **_PARAMS,
    )
    # vel_err = 5 >> vel_err_high -> alpha = alpha_max = 0.4.
    expected = prev_offset + 0.4 * (target - prev_offset)
    assert torch.allclose(new_offset, expected)
    # Moved in the right direction, but not all the way (alpha < 1).
    assert 0.0 < new_offset[0, 0].item() < target[0, 0].item()


def test_max_xy_speed_clamps_offset_change():
    # Huge target + full alpha, but the per-step change is capped by max_xy_speed * dt.
    prev_offset = torch.tensor([[0.0, 0.0]])
    new_offset = compute_smooth_realign_offset(
        current_root_xy=torch.tensor([[100.0, 0.0]]),
        ref_root_xy=torch.tensor([[0.0, 0.0]]),
        current_root_xy_vel=torch.tensor([[50.0, 0.0]]),
        ref_root_xy_vel=torch.tensor([[0.0, 0.0]]),
        prev_offset_xy=prev_offset,
        **_PARAMS,
    )
    max_delta = _PARAMS["max_xy_speed"] * _PARAMS["dt"]
    delta_norm = torch.linalg.norm(new_offset - prev_offset, dim=-1)
    assert torch.allclose(delta_norm, torch.tensor([max_delta]), atol=1e-6)


def test_smoothstep_midpoint_uses_half_alpha_range():
    # At the midpoint of [vel_err_low, vel_err_high], smoothstep(0.5) = 0.5, so
    # alpha = alpha_min + 0.5 * (alpha_max - alpha_min).
    prev_offset = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[0.01, 0.0]])
    mid = 0.5 * (_PARAMS["vel_err_low"] + _PARAMS["vel_err_high"])
    new_offset = compute_smooth_realign_offset(
        current_root_xy=target.clone(),
        ref_root_xy=torch.tensor([[0.0, 0.0]]),
        current_root_xy_vel=torch.tensor([[mid, 0.0]]),
        ref_root_xy_vel=torch.tensor([[0.0, 0.0]]),
        prev_offset_xy=prev_offset,
        **_PARAMS,
    )
    expected_alpha = _PARAMS["alpha_min"] + 0.5 * (
        _PARAMS["alpha_max"] - _PARAMS["alpha_min"]
    )
    expected = prev_offset + expected_alpha * (target - prev_offset)
    assert torch.allclose(new_offset, expected, atol=1e-6)


def _bare_env() -> BaseEnv:
    env = BaseEnv.__new__(BaseEnv)
    env.device = torch.device("cpu")
    env.num_envs = 1
    env.respawn_root_offset = torch.zeros(1, 3)
    env._realign_offset_delta = torch.zeros(1)
    return env


class _MotionLib:
    def __init__(self, root_pos, root_vel):
        # Store as single-body rigid-body tensors [N, 1, 3].
        self._pos = root_pos.unsqueeze(1)
        self._vel = root_vel.unsqueeze(1)

    def get_motion_state(self, motion_ids, motion_times):
        return SimpleNamespace(rigid_body_pos=self._pos, rigid_body_vel=self._vel)


def test_reset_align_is_instant_not_smoothed():
    # align_motion_with_humanoid (used by the reset path) must fully snap the XY offset
    # in one call, independent of any velocity gating.
    env = _bare_env()
    env.motion_manager = SimpleNamespace(
        motion_ids=torch.tensor([0]),
        motion_times=torch.tensor([0.0]),
    )
    env.motion_lib = _MotionLib(
        root_pos=torch.tensor([[2.0, 3.0, 1.0]]),
        root_vel=torch.zeros(1, 3),
    )
    env_ids = torch.tensor([0])
    root_pos = torch.tensor([[10.0, -4.0, 1.0]])

    env.align_motion_with_humanoid(env_ids, root_pos)

    expected_xy = root_pos[:, :2] - torch.tensor([[2.0, 3.0]])
    assert torch.allclose(env.respawn_root_offset[:, :2], expected_xy)


def test_update_smooth_motion_alignment_wires_states():
    # End-to-end: the env method reads root + reference states and writes a smoothed
    # offset. With zero velocity error the offset stays put.
    env = _bare_env()
    env.respawn_root_offset[:, :2] = torch.tensor([[0.5, 0.5]])
    # The env method reads the ``realign_*`` config fields; dt is passed explicitly.
    cfg = SimpleNamespace(
        realign_alpha_min=_PARAMS["alpha_min"],
        realign_alpha_max=_PARAMS["alpha_max"],
        realign_vel_err_low=_PARAMS["vel_err_low"],
        realign_vel_err_high=_PARAMS["vel_err_high"],
        realign_max_xy_speed=_PARAMS["max_xy_speed"],
    )
    env.motion_manager = SimpleNamespace(
        config=cfg,
        motion_ids=torch.tensor([0]),
        motion_times=torch.tensor([0.0]),
    )
    env.motion_lib = _MotionLib(
        root_pos=torch.tensor([[0.0, 0.0, 1.0]]),
        root_vel=torch.tensor([[1.0, 0.0, 0.0]]),
    )
    env.simulator = SimpleNamespace(
        get_root_state=lambda env_ids: SimpleNamespace(
            root_pos=torch.tensor([[3.0, 3.0, 1.0]]),
            root_vel=torch.tensor([[1.0, 0.0, 0.0]]),
        )
    )

    env.update_smooth_motion_alignment(torch.tensor([0]), dt=_PARAMS["dt"])

    # vel_err == 0 -> alpha_min (0) -> offset unchanged and zero re-anchor delta.
    assert torch.allclose(env.respawn_root_offset[:, :2], torch.tensor([[0.5, 0.5]]))
    assert torch.allclose(env._realign_offset_delta, torch.zeros(1))


def test_update_smooth_motion_alignment_records_shift_delta():
    # Under a large velocity mismatch the offset moves and the recorded delta equals
    # the magnitude of that per-step shift.
    env = _bare_env()
    cfg = SimpleNamespace(
        realign_alpha_min=_PARAMS["alpha_min"],
        realign_alpha_max=_PARAMS["alpha_max"],
        realign_vel_err_low=_PARAMS["vel_err_low"],
        realign_vel_err_high=_PARAMS["vel_err_high"],
        realign_max_xy_speed=_PARAMS["max_xy_speed"],
    )
    env.motion_manager = SimpleNamespace(
        config=cfg,
        motion_ids=torch.tensor([0]),
        motion_times=torch.tensor([0.0]),
    )
    env.motion_lib = _MotionLib(
        root_pos=torch.tensor([[0.0, 0.0, 1.0]]),
        root_vel=torch.tensor([[0.0, 0.0, 0.0]]),
    )
    env.simulator = SimpleNamespace(
        get_root_state=lambda env_ids: SimpleNamespace(
            root_pos=torch.tensor([[0.05, 0.0, 1.0]]),
            root_vel=torch.tensor([[5.0, 0.0, 0.0]]),
        )
    )

    env.update_smooth_motion_alignment(torch.tensor([0]), dt=_PARAMS["dt"])

    # vel_err = 5 >> vel_err_high -> alpha_max = 0.4; target = [0.05, 0].
    expected_delta = torch.tensor([0.4 * 0.05])
    assert torch.allclose(env._realign_offset_delta, expected_delta, atol=1e-6)
    assert torch.allclose(
        env.respawn_root_offset[:, :2], torch.tensor([[0.4 * 0.05, 0.0]]), atol=1e-6
    )


def test_realign_penalty_kernel_returns_shift_magnitude():
    delta = torch.tensor([0.0, 0.1, 0.5])
    assert torch.allclose(compute_realign_penalty(delta), delta)


def test_realign_penalty_kernel_deadzone():
    delta = torch.tensor([0.05, 0.2, 0.5])
    out = compute_realign_penalty(delta, deadzone=0.1)
    # max(delta - 0.1, 0)
    assert torch.allclose(out, torch.tensor([0.0, 0.1, 0.4]), atol=1e-6)
