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
"""Fighting-game mimic environment (Tier 5: multi-character self-play).

Warm-starts from the stamina tier (``fight_stamina.py``) -- the observation and
network architecture are IDENTICAL and frozen since Tier 1 (the opponent key-body
collision-primitive slots were always present, just empty/zero until now). So you
can warm-start from an extended-observation checkpoint. This experimental tier
is outside the initial observation/strength comparison; see docs/extended_obs/README.md.

What Tier 5 adds on top of the frozen base:
  - N physically-interacting characters per scene (``simulator.num_characters``).
    All N share the single policy and every character's transition is a training
    sample (the env flattens to ``num_envs * N`` rows at the agent API). Characters
    reset as one scene-level episode. Each independently sampled clip is translated
    so its predicted root path approaches a shared interaction point, with bounded
    spacing and variance controlled by ``env.character_*`` settings.
  - Opponents appear in the collision-primitive observation as sphere primitives
    (their head/pelvis/limbs), so the policy perceives and can avoid/strike them.
  - ``opponent_impact`` reward: rewards landing fast hand/foot strikes on an
    opponent (closing speed gated by proximity). This is a geometric proxy,
    not a measured impact. Projectiles and their penalty are not inherited.

``simulator.num_characters`` is the only knob that sets N; it is NOT hard-limited
to 3 -- raise it for more characters per scene (subject to the frozen primitive
capacity ``env.max_collision_primitives`` and GPU memory). ``--num-envs`` remains
the number of physical scenes; the agent sees ``num_envs * num_characters`` rows.

Requires a BUILT_IN_PD robot (``--robot-name smpl``).

Train (resume/warm-start from the stamina tier):
    PYTHONPATH=. python -m protomotions.train_agent \
        --robot-name smpl --simulator isaaclab \
        --experiment-path examples/experiments/mimic/fight_multichar.py \
        --experiment-name smpl_fight_multichar \
        --motion-file <motions.pt> --num-envs 512 --batch-size 2048 \
        --checkpoint results/smpl_fight_stamina/last.ckpt

    # More characters per scene (configurable, not hard-limited):
    #   --overrides simulator.num_characters=3
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


# Number of physically-interacting characters per scene (shared policy). This is a
# setting, not a hard limit -- raise it (e.g. via --overrides simulator.num_characters=3)
# for more characters per scene.
NUM_CHARACTERS = 2
CHARACTER_SPAWN_RADIUS = 1.0

# Weight on the reward for landing fast limb strikes on opponents.
OPPONENT_IMPACT_WEIGHT = 0.1


def _load_base():
    """Load the stamina tier (``fight_stamina.py``) sitting beside this file."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fight_stamina.py")
    spec = importlib.util.spec_from_file_location("fight_stamina_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_base()

# Reuse the stamina-tier definitions verbatim -- only multi-character wiring changes.
terrain_config = _base.terrain_config
scene_lib_config = _base.scene_lib_config
motion_lib_config = _base.motion_lib_config
agent_config = _base.agent_config
apply_inference_overrides = _base.apply_inference_overrides


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Stamina tier env + opponent-impact reward."""
    from protomotions.envs.component_factories import opponent_impact_rew_factory

    cfg = _base.env_config(robot_cfg, args)
    cfg.character_spawn_radius = CHARACTER_SPAWN_RADIUS
    cfg.reward_components["opponent_impact"] = opponent_impact_rew_factory(
        weight=OPPONENT_IMPACT_WEIGHT,
        zero_during_grace_period=True,
    )
    return cfg


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Stamina-tier setup + spawn N interacting characters per scene."""
    _base.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    simulator_cfg.num_characters = NUM_CHARACTERS
    simulator_cfg.character_spawn_radius = CHARACTER_SPAWN_RADIUS
