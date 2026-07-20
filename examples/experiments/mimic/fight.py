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
"""Fighting-game mimic environment (Tier 1 base).

Extends the standard mimic tracker (``mimic/mlp.py``) with the FINAL, FROZEN
observation architecture used across every fighting-game training tier:

  - ``collision_primitives``: unified egocentric 17-float encoding of the K
    nearest collidable entities (ground, obstacles, projectiles, other
    characters' key bodies). See ``protomotions/envs/obs/collision_primitives.py``.
  - contact observations enabled on a comprehensive, frozen body set.
  - ``stamina_obs``: per-body stamina scalars, each the current joint-stiffness
    scale for that body (all of a body's DOF axes share it; damping scales by the
    same factor). Under BUILT_IN_PD the scaled per-DOF gains are written into the
    engine on reset (see ``BaseEnv._apply_body_stamina_to_gains`` ->
    ``Simulator.set_joint_gain_scale``). Inert (==1.0, i.e. nominal gains) until
    the stamina tier turns on randomization.
  - smooth velocity-error XY motion re-anchoring
    (``realign_motion_with_humanoid_on_each_step``), which blends the reference offset
    toward the character with a strength proportional to the "unexpected" root XY
    velocity (character vs. clip) so an unavoidable shove doesn't accumulate
    absolute-position tracking error while pose/orientation/velocities are still
    tracked. This Tier 1 base trains PURE tracking with re-anchoring DISABLED (no
    crutch); the interference tiers (``fight_throw.py`` and below) turn it on. The
    smooth params are kept here (dormant) so those tiers inherit tuned values by only
    flipping the flag. See the deployment doc for the exact algorithm.
  - stronger whole-body articulation signal in the reward (TRAINING-ONLY; does not
    touch the frozen obs/action layout or the ONNX): the global body-position reward
    averages per-body error before the exponential, so a single under-articulated
    limb (an unbent knee, a short punch) is diluted across the ~24 bodies. To fix
    that without changing the network, the tracking bundle adds a per-body
    (exp-before-mean) orientation term and an explicit per-joint DOF-angle reward
    (``dof_pos_rew``), and relaxes ``action_smoothness`` so fast limb motion isn't
    suppressed. Rewards are inherited by all higher tiers (which only ADD terms).

In this Tier 1 base experiment the interference features are present but INERT:
only the ground primitive is populated (no projectiles/opponents) and stamina
defaults to 1.0. Training here yields a competent tracker whose network shape is
identical to all later tiers, so Phases 2-5 warm-start from this checkpoint.

IMPORTANT (frozen architecture): ``num_obs_primitives`` (K), the contact body
set, and ``max_collision_primitives`` (M) must not change once a tier is trained
-- they determine the policy input width and the ONNX input shapes.

Control: uses BUILT_IN_PD (engine-side PD), so run with ``--robot-name smpl``.
Stamina modulates the engine's joint gains directly (not a custom in-loop PD).

Train (Tier 1 base):
    python protomotions/train_agent.py \
        --robot-name smpl --simulator isaaclab \
        --experiment-path examples/experiments/mimic/fight.py \
        --experiment-name smpl_fight_base \
        --motion-file <motions.pt> --num-envs 8192 --batch-size 8192
"""
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
import argparse


# Number of nearest collision primitives included in the observation (K). Part of
# the frozen architecture: obs width contributed == NUM_OBS_PRIMITIVES * 17.
NUM_OBS_PRIMITIVES = 6

# Capacity (M) of the per-env candidate buffer the obs selects from. Must be >= K
# and large enough for the busiest later tier (obstacles + projectiles + opponents).
MAX_COLLISION_PRIMITIVES = 16

# Frozen contact-observation body set (expanded from common-naming keys).
CONTACT_BODIES = [
    "all_left_foot_bodies",
    "all_right_foot_bodies",
    "all_left_hand_bodies",
    "all_right_hand_bodies",
    "head_body_name",
    "torso_body_name",
]

# Keys consumed by the policy/critic. Order is part of the frozen architecture.
OBS_IN_KEYS = [
    "max_coords_obs",
    "mimic_target_poses",
    "previous_actions",
    "collision_primitives",
    "stamina_obs",
]


def terrain_config(args: argparse.Namespace):
    """Build terrain configuration."""
    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    """Build scene library configuration."""
    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build environment configuration (training defaults)."""
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.component_factories import (
        max_coords_obs_factory,
        collision_primitives_obs_factory,
        stamina_obs_factory,
        previous_actions_factory,
        mimic_target_poses_max_coords_factory,
        action_smoothness_factory,
        mimic_tracking_rewards_factory,
        dof_pos_rew_factory,
        pow_rew_factory,
        contact_match_rew_factory,
        realign_penalty_rew_factory,
        tracking_error_term_factory,
    )
    from protomotions.envs.action import make_pd_action_config

    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
        )
    }

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(observe_contacts=True),
        "previous_actions": previous_actions_factory(history_steps=1),
        "mimic_target_poses": mimic_target_poses_max_coords_factory(
            with_velocities=True
        ),
        "collision_primitives": collision_primitives_obs_factory(
            num_obs_primitives=NUM_OBS_PRIMITIVES
        ),
        # Per-body joint-stiffness scale (inert == 1.0/nominal in Tier 1; randomized
        # in the stamina tier). Part of the frozen layout so later tiers warm-start
        # without reshaping.
        "stamina_obs": stamina_obs_factory(),
    }

    termination_components = {
        "tracking_error": tracking_error_term_factory(threshold=0.5),
    }

    reward_components = {
        # Kept small so it doesn't suppress the fast limb articulation (punches,
        # knee flexion) the tracking terms below are meant to elicit.
        "action_smoothness": action_smoothness_factory(weight=-0.005),
        **mimic_tracking_rewards_factory(
            gt_weight=0.5,
            # Body-orientation tracking is the term that most directly encodes limb
            # bend/extension. Strengthened (weight 0.3->0.5, coef -5->-10) and, more
            # importantly, aggregated per-body (mean_before_exp=False) so a single
            # under-articulated limb (an unbent knee, a short punch) is no longer
            # diluted across the ~24 bodies before the exponential.
            gr_weight=0.5,
            gv_weight=0.1,
            gav_weight=0.2,
            rh_weight=0.2,
            gt_coef=-25.0,
            gr_coef=-10.0,
            gv_coef=-0.5,
            gav_coef=-0.1,
            rh_coef=-100.0,
            gr_mean_before_exp=False,
        ),
        # Direct per-joint angle tracking: weights every joint equally instead of
        # diluting extremities inside the Cartesian body-position mean. This is the
        # primary lever that makes knees flex and arms fully extend to match.
        "dof_pos_rew": dof_pos_rew_factory(weight=0.5, coefficient=-5.0),
        "pow_rew": pow_rew_factory(weight=-1e-5, min_value=-0.5),
        "contact_match_rew": contact_match_rew_factory(
            weight=-0.1, zero_during_grace_period=True
        ),
        # Discourage leaning on re-anchoring as a crutch: penalize how far the
        # reference was shifted each step. A no-op in this Tier 1 base (re-anchoring
        # disabled -> delta is 0); active in the interference tiers that turn
        # re-anchoring on, all of which inherit this reward via _base.env_config().
        "realign_penalty": realign_penalty_rew_factory(
            weight=-0.05, zero_during_grace_period=True
        ),
    }

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=2,
        max_collision_primitives=MAX_COLLISION_PRIMITIVES,
        # Hard projectile/opponent impacts can rarely trip a PhysX solver blowup in a
        # single env; tolerate it (sanitize + natural reset) so a long run survives.
        # Frozen on from Tier 1 so every fighting tier shares the behavior.
        sanitize_non_finite_state=True,
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
            # Tier 1 base trains PURE tracking with re-anchoring DISABLED so the policy
            # learns to hold its own position/balance with no crutch. The interference
            # tiers (fight_throw.py -> fight_stamina.py -> fight_multichar.py) flip this
            # flag to True where real interference begins. The smooth velocity-error
            # params below are kept (dormant) so those tiers inherit tuned values by
            # changing only the boolean. When enabled, the reference XY offset blends
            # toward the character's root with a strength proportional to the
            # "unexpected" root XY velocity (character vs. clip), so absolute XY drift
            # from an unavoidable shove is not penalized while pose/orientation/
            # velocities are still tracked, and balance-critical clips (e.g. getup) that
            # track the clip velocity leave the offset stable.
            realign_motion_with_humanoid_on_each_step=False,
            realign_alpha_min=0.0,
            realign_alpha_max=0.4,
            realign_vel_err_low=0.3,
            realign_vel_err_high=1.5,
            realign_max_xy_speed=2.0,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    """Build agent configuration."""
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOModelConfig,
        AdvantageNormalizationConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.envs.component_factories import (
        gt_error_factory,
        gr_error_factory,
        max_joint_error_factory,
    )

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=OBS_IN_KEYS,
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=OBS_IN_KEYS,
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=OBS_IN_KEYS,
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    agent_config: PPOAgentConfig = PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=OBS_IN_KEYS,
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        evaluator=MimicEvaluatorConfig(
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.5),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0,
            ),
            # Sanity-run eval budget: a forced full-dataset eval runs after every
            # warm-start (``just_loaded_checkpoint_should_evaluate``); the default
            # 600-step episodes over all ~9.5k motions (plus predicted-motion-lib
            # dumps) make each tier's startup very slow. Keep eval cheap for these
            # short curriculum runs. Does not affect obs/network/ONNX. Raise
            # ``max_eval_steps`` / re-enable the dump for full-quality training.
            max_eval_steps=100,
            save_predicted_motion_lib_every=None,
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True, use_ema=True
        ),
    )
    return agent_config


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Configure robot contact sensing for the frozen fighting observation set."""
    robot_cfg.update_fields(contact_bodies=CONTACT_BODIES)


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg,
    agent_cfg,
    terrain_cfg: TerrainConfig,
    motion_lib_cfg: MotionLibConfig,
    scene_lib_cfg: SceneLibConfig,
    args: argparse.Namespace,
):
    """Apply evaluation-specific overrides."""
    if hasattr(env_cfg, "termination_components") and env_cfg.termination_components:
        env_cfg.termination_components = {}

    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
