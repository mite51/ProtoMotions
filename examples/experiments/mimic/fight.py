# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Main's MLP tracker plus fighting observations, with unchanged tracking physics.

Use mimic/mlp.py as the baseline. Rewards, optimizer, network hidden layers,
reference alignment, evaluation horizon and action processing inherit from it.
Only the input width and contact sensor coverage change. Stamina is nominal;
projectiles, gain randomization and multi-character interaction are disabled.
"""
import argparse

import torch

from examples.experiments.mimic import mlp as _base
from protomotions.envs.component_factories import (
    collision_primitives_obs_factory,
    max_coords_obs_factory,
    stamina_obs_factory,
)
from protomotions.robot_configs.base import abstract_names_to_body_names

NUM_OBS_PRIMITIVES = 6
MAX_COLLISION_PRIMITIVES = 16
CONTACT_BODIES = [
    "all_left_foot_bodies", "all_right_foot_bodies",
    "all_left_hand_bodies", "all_right_hand_bodies",
    "head_body_name", "torso_body_name",
]
OBS_IN_KEYS = [
    "max_coords_obs", "mimic_target_poses", "previous_actions",
    "collision_primitives", "stamina_obs",
]

terrain_config = _base.terrain_config
scene_lib_config = _base.scene_lib_config
motion_lib_config = _base.motion_lib_config
apply_inference_overrides = _base.apply_inference_overrides


def env_config(robot_cfg, args: argparse.Namespace):
    cfg = _base.env_config(robot_cfg, args)
    cfg.max_collision_primitives = MAX_COLLISION_PRIMITIVES
    cfg.observation_components.update(
        max_coords_obs=max_coords_obs_factory(observe_contacts=True),
        collision_primitives=collision_primitives_obs_factory(NUM_OBS_PRIMITIVES),
        stamina_obs=stamina_obs_factory(),
    )
    # Expanded sensing must not silently expand the contact-matching reward.
    # Main compares only feet, regardless of the additional contacts in the obs.
    foot_names = set()
    for group in ("all_left_foot_bodies", "all_right_foot_bodies"):
        foot_names.update(abstract_names_to_body_names(group, robot_cfg))
    foot_ids = [i for i, name in enumerate(robot_cfg.kinematic_info.body_names) if name in foot_names]
    contact_reward = cfg.reward_components["contact_match_rew"]
    del contact_reward.dynamic_vars["contact_body_ids"]
    contact_reward.static_params["contact_body_ids"] = torch.tensor(foot_ids, dtype=torch.long)
    return cfg


def agent_config(robot_config, env_config, args: argparse.Namespace):
    cfg = _base.agent_config(robot_config, env_config, args)
    for component in (cfg.model, cfg.model.actor, cfg.model.actor.mu_model, cfg.model.critic):
        component.in_keys = list(OBS_IN_KEYS)
    return cfg


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args: argparse.Namespace):
    _base.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    robot_cfg.update_fields(contact_bodies=CONTACT_BODIES)
