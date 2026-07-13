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
"""Fighting-game mimic environment (Tier 2b: randomized per-body stamina).

Warm-starts from the collider tier (``fight_throw.py``) -- identical frozen
observation/network architecture. This tier adds, on top of thrown colliders and
the dangerous-impact penalty:

  - Per-episode randomized per-body stamina (``randomize_body_stamina=True``).
    Each actuated body's joint-drive stiffness (and, proportionally, damping) is
    scaled by its stamina (sampled in ``body_stamina_range``), so the policy must
    stay controlled with weaker/stiffer/uneven joints. Under BUILT_IN_PD the
    scaled gains are written into the engine on reset. Stamina is already part of
    the observation (``stamina_obs``) from the Tier 1 base, so the network shape
    is unchanged.

Requires a BUILT_IN_PD robot (``--robot-name smpl``); the scaled gains are
applied via ``Simulator.set_joint_gain_scale``.

Train (resume/warm-start from the collider tier):
    python protomotions/train_agent.py \
        --robot-name smpl --simulator isaaclab \
        --experiment-path examples/experiments/mimic/fight_stamina.py \
        --experiment-name smpl_fight_stamina \
        --motion-file <motions.pt> --num-envs 8192 --batch-size 8192 \
        --checkpoint results/smpl_fight_throw/last.ckpt
"""
import argparse
import os
import importlib.util

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig


# Range from which per-body stamina is sampled each episode. Stamina is the
# per-body joint-stiffness scale (1.0 == nominal gains); it multiplies both
# stiffness and damping. 0.1 == very weak/compliant, 2.0 == twice as stiff.
BODY_STAMINA_RANGE = (0.1, 2.0)


def _load_base():
    """Load the collider tier (``fight_throw.py``) sitting beside this file."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fight_throw.py")
    spec = importlib.util.spec_from_file_location("fight_throw_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_base()

# Reuse the collider-tier definitions verbatim -- only stamina randomization changes.
terrain_config = _base.terrain_config
scene_lib_config = _base.scene_lib_config
motion_lib_config = _base.motion_lib_config
agent_config = _base.agent_config
configure_robot_and_simulator = _base.configure_robot_and_simulator
apply_inference_overrides = _base.apply_inference_overrides


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Collider tier + per-episode randomized per-body stamina."""
    cfg = _base.env_config(robot_cfg, args)
    cfg.randomize_body_stamina = True
    cfg.body_stamina_range = BODY_STAMINA_RANGE
    return cfg
