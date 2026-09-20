# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for motion manager components.

This module contains all configuration dataclasses for motion manager functionality,
co-located with the motion manager implementations in the same directory.
"""

from typing import Optional, List, Union
from dataclasses import dataclass, field


@dataclass
class MotionManagerConfig:
    """Configuration for motion management."""

    _target_: str = "protomotions.envs.motion_manager.motion_manager.MotionManager"

    init_start_prob: float = field(
        default=0.2,
        metadata={
            "help": "Probability to sample an initial pose instead of random time. Helps prevent local-minima in AMP.",
            "min": 0.0,
            "max": 1.0,
        }
    )

    sample_time_truncate_s: Optional[float] = field(
        default=None,
        metadata={
            "help": "Optional extra seconds to remove from the end of random reset time sampling.",
            "min": 0.0,
        },
    )

    subset_method: Optional[Union[str, List[int]]] = field(
        default=None,
        metadata={
            "help": "Motion subset for evaluation: 'first', 'last', 'random', or list of motion IDs. None uses all motions.",
            "options": ["first", "last", "random"],
        }
    )

    exclude_motion_ids: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "Motion IDs to exclude from sampling. Useful for removing problematic motions.",
        }
    )

    exclude_motions_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to file with motion IDs to exclude (one per line). Can also be an expert training directory.",
        }
    )

    realign_motion_with_humanoid_on_each_step: bool = field(
        default=False,
        metadata={
            "help": "Realign motion with humanoid each step. Prevents tracking error accumulation for imperfect retargeting.",
        }
    )


@dataclass
class MimicMotionManagerConfig(MotionManagerConfig):
    """Configuration for mimic motion management."""

    _target_: str = (
        "protomotions.envs.motion_manager.mimic_motion_manager.MimicMotionManager"
    )

    resample_on_reset: bool = field(
        default=True,
        metadata={"help": "Whether to resample motion on environment reset."}
    )

    smooth_realign_enabled: bool = False

    # --- Smooth velocity-error re-anchoring ---------------------------------
    # When realign_motion_with_humanoid_on_each_step is enabled, the reference XY
    # offset is not snapped instantly to the character each step. Instead it is
    # blended toward the character with a strength proportional to the
    # "unexpected" root XY velocity (||current_xy_vel - ref_xy_vel||). This lets
    # an external shove/slide re-anchor the reference (so it stays reachable)
    # while a character tracking the clip velocity leaves the offset stable --
    # important for balance-critical clips like getup. See fight.py / the
    # deployment doc for the exact contract.
    realign_alpha_min: float = field(
        default=0.0,
        metadata={
            "help": "Blend factor when velocity error <= realign_vel_err_low (0 == frozen offset).",
            "min": 0.0,
            "max": 1.0,
        },
    )

    realign_alpha_max: float = field(
        default=0.4,
        metadata={
            "help": "Blend factor when velocity error >= realign_vel_err_high.",
            "min": 0.0,
            "max": 1.0,
        },
    )

    realign_vel_err_low: float = field(
        default=0.3,
        metadata={
            "help": "Root XY velocity error (m/s) below which re-anchoring uses realign_alpha_min.",
            "min": 0.0,
        },
    )

    realign_vel_err_high: float = field(
        default=1.5,
        metadata={
            "help": "Root XY velocity error (m/s) at/above which re-anchoring uses realign_alpha_max.",
            "min": 0.0,
        },
    )

    realign_max_xy_speed: float = field(
        default=2.0,
        metadata={
            "help": "Cap (m/s) on how fast the reference XY offset may change per step.",
            "min": 0.0,
        },
    )
