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
"""SMPL humanoid configured for PROPORTIONAL (explicit Python PD) control.

This is a drop-in variant of :class:`SmplRobotConfig` that switches the control
type from ``BUILT_IN_PD`` (gains baked into the physics engine at setup) to
``PROPORTIONAL`` (PD torques computed each substep in Python). PROPORTIONAL
control is what enables runtime per-env, per-DOF gain modulation -- the
foundation for the "stamina" feature (see ``normalized_pd_stamina_action`` and
``BaseEnv.set_dof_stamina_scale``).

The per-DOF stiffness/damping/effort values are inherited unchanged from
``SmplRobotConfig`` so that, with stamina fixed at 1.0, behaviour matches the
``smpl`` robot as closely as the explicit-PD integration allows.

Register/select via ``--robot-name smpl-proportional``.
"""
from dataclasses import dataclass, field

from protomotions.robot_configs.base import ControlConfig, ControlType
from protomotions.robot_configs.smpl import SmplRobotConfig
from protomotions.components.pose_lib import ControlInfo


@dataclass
class SmplProportionalRobotConfig(SmplRobotConfig):
    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.PROPORTIONAL,
            override_control_info={
                ".*_(Hip|Knee|Ankle)_.*": ControlInfo(
                    stiffness=800,
                    damping=80,
                    effort_limit=500,
                    velocity_limit=100,
                ),
                ".*_Toe_.*": ControlInfo(
                    stiffness=500,
                    damping=50,
                    effort_limit=500,
                    velocity_limit=100,
                ),
                "(Torso|Spine|Chest)_.*": ControlInfo(
                    stiffness=1000,
                    damping=100,
                    effort_limit=500,
                    velocity_limit=100,
                ),
                "(Neck|Head|.*_Thorax|.*_Shoulder|.*_Elbow)_.*": ControlInfo(
                    stiffness=500,
                    damping=50,
                    effort_limit=500,
                    velocity_limit=100,
                ),
                ".*_(Wrist|Hand)_.*": ControlInfo(
                    stiffness=300,
                    damping=30,
                    effort_limit=500,
                    velocity_limit=100,
                ),
            },
        )
    )
