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
"""Per-body stamina observation.

Stamina is one scalar per actuated body -- the current joint-stiffness scale for
that body (all of the body's DOF axes share it). Under IsaacLab BUILT_IN_PD,
stiffness and torque capacity scale by strength and damping by its square root.
The properties are written on reset or a runtime strength update (see ``BaseEnv._apply_body_stamina_to_gains`` ->
``Simulator.set_joint_gain_scale``). Exposing it as an observation lets the policy
condition its actions on the current drive strength of each limb (e.g. push
harder / brace differently when a limb is "weak"). 1.0 == nominal gains.

This is a Level-2 / ONNX-exportable pure-tensor kernel.
"""
from torch import Tensor


def compute_stamina_obs(body_stamina: Tensor) -> Tensor:
    """Return per-body stamina as a flat observation.

    Args:
        body_stamina: Per-body stamina (joint-stiffness scale)
            [num_envs, num_stamina_bodies]; 1.0 == nominal gains.

    Returns:
        Stamina observation [num_envs, num_stamina_bodies]. Returned as-is (the
        policy applies its own input normalization); width is fixed by the robot's
        actuated-body count, so it is part of the frozen observation architecture.
    """
    return body_stamina
