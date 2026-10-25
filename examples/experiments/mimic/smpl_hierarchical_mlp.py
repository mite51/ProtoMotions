# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Hierarchical Shared Mini-Brain & Analytical Subtree Load Mimic Configuration (SMPL)
===================================================================================

Full-body motion tracking experiment configuration for SMPL (`--robot-name=smpl`)
using Pattern B ("Spinal Cord") hierarchical motor control:
- Action space: `make_passthrough_pd_action_config(robot_cfg)` so the `u_pd` targets
  produced by `HierarchicalPPOActor` go directly to the simulator's PD controller in radians.
- Observations: `max_coords_obs`, `previous_actions`, `mimic_target_poses`,
  `collision_primitives`, `stamina_obs`, `joint_load_obs`, and `minibrain_features`.
- Agent: `HierarchicalPPOAgentConfig` with `HierarchicalPPOActorConfig`,
  `critic_warmup_epochs=10`, and `horizon_curriculum_enabled=True`.
"""

import argparse

from protomotions.agents.common.minibrain import SharedJointMiniBrainConfig
from protomotions.agents.ppo.hierarchical_model import (
    HierarchicalPPOActorConfig,
    HierarchicalPPOAgentConfig,
)
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.diagnostics.edge_case_collector import EdgeCaseCollectorConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


def terrain_config(args: argparse.Namespace):
    """Build terrain configuration."""
    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    """Build scene library configuration."""
    scene_file = getattr(args, "scenes_file", None)
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    motion_file = getattr(args, "motion_file", None)
    return MotionLibConfig(motion_file=motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace = None) -> EnvConfig:
    """Build hierarchical SMPL mimic environment configuration."""
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.component_factories import (
        action_smoothness_factory,
        collision_primitives_obs_factory,
        contact_match_rew_factory,
        dof_pos_rew_factory,
        joint_load_obs_factory,
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
        mimic_tracking_rewards_factory,
        minibrain_features_obs_factory,
        pow_rew_factory,
        previous_actions_factory,
        stamina_obs_factory,
        tracking_error_term_factory,
    )
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
        )
    }

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        "previous_actions": previous_actions_factory(history_steps=1),
        "mimic_target_poses": mimic_target_poses_max_coords_factory(
            with_velocities=True
        ),
        "collision_primitives": collision_primitives_obs_factory(),
        "stamina_obs": stamina_obs_factory(),
        "joint_load_obs": joint_load_obs_factory(robot_cfg),
        "minibrain_features": minibrain_features_obs_factory(robot_cfg),
    }

    termination_components = {
        "tracking_error": tracking_error_term_factory(threshold=0.5),
    }

    reward_components = {
        "action_smoothness": action_smoothness_factory(weight=-0.02),
        **mimic_tracking_rewards_factory(
            gt_weight=0.5,
            gr_weight=0.3,
            gv_weight=0.1,
            gav_weight=0.2,
            rh_weight=0.2,
            gt_coef=-25.0,
            gr_coef=-5.0,
            gv_coef=-0.5,
            gav_coef=-0.1,
            rh_coef=-100.0,
        ),
        "dof_pos_rew": dof_pos_rew_factory(weight=0.15, coefficient=-5.0),
        "pow_rew": pow_rew_factory(weight=-1e-5, min_value=-0.5),
    }
    motion_file = str(getattr(args, "motion_file", "") or "")
    use_contact_match = getattr(args, "use_contact_match_rew", None)
    if use_contact_match is None:
        use_contact_match = not motion_file.endswith(".motion")
    if use_contact_match:
        reward_components["contact_match_rew"] = contact_match_rew_factory(
            weight=-0.1, zero_during_grace_period=True
        )

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=2,
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        action_config=make_pd_action_config(
            robot_cfg, action_transform="clamp", clamp_value=1.0
        ),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )


def agent_config(
    robot_config: RobotConfig,
    env_config: EnvConfig,
    args: argparse.Namespace = None,
) -> HierarchicalPPOAgentConfig:
    """Build HierarchicalPPOAgentConfig for SMPL."""
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import MLPLayerConfig, MLPWithConcatConfig
    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.agents.ppo.config import (
        AdvantageNormalizationConfig,
        PPOModelConfig,
    )
    from protomotions.envs.component_factories import (
        gr_error_factory,
        gt_error_factory,
        max_joint_error_factory,
    )

    batch_size = getattr(args, "batch_size", 4096) if args is not None else 4096
    training_max_steps = (
        getattr(args, "training_max_steps", 1_000_000_000)
        if args is not None
        else 1_000_000_000
    )
    import os

    minibrain_checkpoint = (
        getattr(args, "minibrain_checkpoint", None) if args is not None else None
    )
    if minibrain_checkpoint is None and os.path.exists("checkpoints/smpl_minibrain.pt"):
        minibrain_checkpoint = "checkpoints/smpl_minibrain.pt"
    freeze_minibrain = (
        getattr(args, "freeze_minibrain", True) if args is not None else True
    )
    expose_minibrain_base_to_main = (
        getattr(args, "expose_minibrain_base_to_main", True)
        if args is not None
        else True
    )

    from protomotions.envs.action import build_pd_action_offset_scale

    enable_extended_obs = (
        getattr(args, "enable_extended_obs", False) if args is not None else False
    )
    main_obs_keys = [
        "max_coords_obs",
        "mimic_target_poses",
        "previous_actions",
        "joint_load_obs",
    ]
    if enable_extended_obs:
        main_obs_keys.extend(["collision_primitives", "stamina_obs"])

    actor_in_keys = main_obs_keys + ["minibrain_features"]
    mu_in_keys = (
        main_obs_keys + ["minibrain_base_cmd"]
        if expose_minibrain_base_to_main
        else list(main_obs_keys)
    )

    num_dofs = robot_config.kinematic_info.num_dofs
    num_joints = num_dofs // 3

    pd_action_offset, pd_action_scale = build_pd_action_offset_scale(
        robot_config.kinematic_info.hinge_axes_map,
        robot_config.kinematic_info.dof_limits_lower,
        robot_config.kinematic_info.dof_limits_upper,
        1.0,
        "cpu",
    )

    actor_config = HierarchicalPPOActorConfig(
        num_out=num_dofs,
        actor_logstd=-2.9,
        in_keys=actor_in_keys,
        mu_key="actor_trunk_out",
        minibrain_config=SharedJointMiniBrainConfig(
            num_joints=num_joints,
            dofs_per_joint=3,
        ),
        minibrain_checkpoint=minibrain_checkpoint,
        freeze_minibrain=freeze_minibrain,
        minibrain_features_key="minibrain_features",
        residual_pose_scale=1.8,
        init_main_brain_last_layer_scale=0.01,
        expose_minibrain_base_to_main=expose_minibrain_base_to_main,
        pd_action_offset=pd_action_offset,
        pd_action_scale=pd_action_scale,
        convert_to_normalized_action_space=True,
        mu_model=MLPWithConcatConfig(
            in_keys=mu_in_keys,
            normalize_obs=True,
            norm_clamp_value=5.0,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=main_obs_keys,
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5.0,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    return HierarchicalPPOAgentConfig(
        model=PPOModelConfig(
            in_keys=actor_in_keys,
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=batch_size,
        training_max_steps=training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        critic_warmup_epochs=4,
        horizon_curriculum_enabled=True,
        initial_max_episode_length=300,
        target_max_episode_length=1000,
        horizon_ramp_epochs=60,
        edge_case_collector_config=EdgeCaseCollectorConfig(
            enabled=True,
            num_joints=num_joints,
            dofs_per_joint=3,
            num_bodies=robot_config.kinematic_info.num_bodies,
        ),
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
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True, use_ema=True
        ),
    )


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Configure robot to add contact sensors for foot contact tracking."""
    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
    )


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
