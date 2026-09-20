# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""First weakness curriculum: same tracking task with mild per-joint weakness.

Warm-start from fight.py in a NEW experiment directory. No projectiles, extra
rewards, or reference realignment. Expand the weakness range only after evaluating
tracking and recovery. Runtime damage can set the same per-body strength input.
"""
from examples.experiments.mimic import fight as _base

terrain_config = _base.terrain_config
scene_lib_config = _base.scene_lib_config
motion_lib_config = _base.motion_lib_config
agent_config = _base.agent_config
configure_robot_and_simulator = _base.configure_robot_and_simulator
apply_inference_overrides = _base.apply_inference_overrides


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.randomize_body_stamina = True
    cfg.body_stamina_range = (0.75, 1.0)
    return cfg
