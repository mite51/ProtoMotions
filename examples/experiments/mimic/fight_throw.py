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
"""Fighting-game mimic environment (Tier 2: thrown colliders).

Warm-starts from the Tier 1 base (``fight.py``) -- the observation architecture
is IDENTICAL (same K, contact set, M, action config, network shape), so you must
resume from a Tier 1 checkpoint. This tier only turns on environment curriculum
knobs; it does NOT change the policy input/output shapes.

What Tier 2 adds on top of the frozen base:
  - Projectile auto-throw: each env independently throws a cube at the character
    with probability ``simulator.projectile.auto_throw_prob`` per step. The cubes
    occupy the previously-empty collision-primitive slots, so the policy now sees
    real moving threats through the same obs it was already trained on.
  - Randomized per-episode "damage": a random fraction of the projectile pool is
    flagged dangerous (high damage); the rest stay at baseline. The policy must
    learn to discriminate threats, not treat every collider identically.
  - ``body_impact_penalty``: penalizes high-force impacts on vulnerable bodies
    (everything EXCEPT hands and feet), scaled by the incoming threat's damage.
    Hands/feet are exempt so striking/contact and footstep loads aren't punished.
  - Smooth velocity-error XY re-anchoring turns ON here (the Tier 1 base keeps it OFF
    to learn pure tracking first). Real interference begins in this tier, so the
    reference offset may now shift to follow an unavoidable shove/slide; the
    ``realign_penalty`` reward (inherited from the base, dormant there) becomes active
    and discourages the policy from leaning on it. Smooth params are inherited from
    ``fight.py``.

Curriculum: ramp ``auto_throw_prob`` from ~0 upward via overrides as the policy
stabilizes, e.g. ``--overrides simulator.projectile.auto_throw_prob=0.03``.

Train (resume/warm-start from Tier 1):
    python protomotions/train_agent.py \
        --robot-name smpl --simulator isaaclab \
        --experiment-path examples/experiments/mimic/fight_throw.py \
        --experiment-name smpl_fight_throw \
        --motion-file <motions.pt> --num-envs 8192 --batch-size 8192 \
        --checkpoint results/smpl_fight_base/last.ckpt
"""
import argparse
import os
import importlib.util

from protomotions.robot_configs.base import (
    RobotConfig,
    abstract_names_to_body_names,
)
from protomotions.simulator.base_simulator.config import (
    SimulatorConfig,
    ProjectileConfig,
)
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig


# Per-env, per-step probability of auto-throwing a projectile. Start low and ramp
# up over training via ``--overrides simulator.projectile.auto_throw_prob=...``.
AUTO_THROW_PROB = 0.01

# Semantic body groups that legitimately absorb large forces (excluded from the
# dangerous-impact penalty). Missing groups are skipped gracefully.
HAND_FOOT_BODY_GROUPS = [
    "all_left_hand_bodies",
    "all_right_hand_bodies",
    "all_left_foot_bodies",
    "all_right_foot_bodies",
]

# Force magnitude (N) below which impacts are ignored by the penalty.
IMPACT_FORCE_THRESHOLD = 50.0


def _load_base():
    """Load the Tier 1 base experiment (``fight.py``) sitting beside this file."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fight.py")
    spec = importlib.util.spec_from_file_location("fight_base_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_base()

# Reuse the Tier 1 definitions verbatim -- only env/sim wiring changes below.
terrain_config = _base.terrain_config
scene_lib_config = _base.scene_lib_config
motion_lib_config = _base.motion_lib_config
agent_config = _base.agent_config
apply_inference_overrides = _base.apply_inference_overrides


def _vulnerable_body_indices(robot_cfg: RobotConfig):
    """Indices of bodies to penalize on impact (all bodies minus hands/feet)."""
    body_names = list(robot_cfg.kinematic_info.body_names)
    exempt = set()
    available = robot_cfg.common_naming_to_robot_body_names.keys()
    for group in HAND_FOOT_BODY_GROUPS:
        if group in available:
            exempt.update(abstract_names_to_body_names(group, robot_cfg))
    return [i for i, name in enumerate(body_names) if name not in exempt]


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Tier 1 env + dangerous-impact penalty (projectile damage uses EnvConfig defaults)."""
    from protomotions.envs.component_factories import body_impact_penalty_rew_factory

    cfg = _base.env_config(robot_cfg, args)

    # Re-anchoring is DISABLED in the Tier 1 base (pure tracking, no crutch). Turn it
    # on here, the first interference tier (thrown colliders); fight_stamina.py and
    # fight_multichar.py inherit it through the tier chain. This also activates the
    # realign reliance penalty (already in reward_components, inherited from fight.py)
    # because the reference offset now actually moves.
    cfg.motion_manager.realign_motion_with_humanoid_on_each_step = True

    cfg.reward_components["body_impact_penalty"] = body_impact_penalty_rew_factory(
        vulnerable_body_indices=_vulnerable_body_indices(robot_cfg),
        weight=-0.01,
        threshold=IMPACT_FORCE_THRESHOLD,
        zero_during_grace_period=True,
    )
    return cfg


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Frozen contact set (from Tier 1) + enable projectile auto-throw."""
    _base.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    # Gentler than the ASE default (30-40 m/s, density 500): at the fight cadence
    # (physics_dt ~= 1/120 s) a 40 m/s cube travels ~0.33 m per substep -- larger than
    # its own size -- which tunnels and rarely destabilizes the PhysX solver. Capping
    # speed so per-substep travel stays near the cube size keeps strong, survivable
    # interference. Ramp speed/prob back up via --overrides once the policy is robust.
    # Mixed primitive shapes (box/sphere/capsule) with randomized orientation, so the
    # policy learns to brace for arbitrary colliders, not just cubes. Each pool slot is
    # one shape (round-robin), so num_projectiles=6 gives two of each. All three are
    # captured by the collision_primitives obs (radius/extent_z/shape), so the frozen
    # observation layout is unchanged.
    simulator_cfg.projectile = ProjectileConfig(
        auto_throw_enabled=True,
        auto_throw_prob=AUTO_THROW_PROB,
        speed_range=(12.0, 22.0),
        density=300.0,
        shapes=("box", "sphere", "capsule"),
        num_projectiles=6,
    )
