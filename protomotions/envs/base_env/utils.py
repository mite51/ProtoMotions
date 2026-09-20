# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BaseEnv utility functions.

Contains RL combining logic for rewards/terminations.
Look here when debugging reward or termination behavior.

This module handles:
- Reward combining (multiplicative, additive, grace periods, clamping)
- Termination combining (OR logic, terminate_on_true inversion)
- Body indices resolution from names to indices
"""

from typing import Any, Dict, Optional, Tuple
import logging

import torch
from torch import Tensor

from protomotions.envs.mdp_component import is_mdp_component
from protomotions.simulator.base_simulator.simulator_state import (
    get_sanitize_non_finite_state,
)

logger = logging.getLogger(__name__)
_reward_sanitize_warn_count = 0


# =============================================================================
# Motion Re-Anchoring
# =============================================================================


def compute_smooth_realign_offset(
    current_root_xy: Tensor,
    ref_root_xy: Tensor,
    current_root_xy_vel: Tensor,
    ref_root_xy_vel: Tensor,
    prev_offset_xy: Tensor,
    alpha_min: float,
    alpha_max: float,
    vel_err_low: float,
    vel_err_high: float,
    max_xy_speed: float,
    dt: float,
    eps: float = 1e-8,
) -> Tensor:
    """Smooth velocity-error reference re-anchoring (XY only).

    Blends the persisted reference XY offset toward the instant-snap target with a
    strength proportional to the "unexpected" root XY velocity (the mismatch between
    the character's velocity and the reference clip's velocity). A character that
    tracks the clip velocity leaves the offset stable (important for balance-critical
    clips such as getup); an external shove/slide produces velocity mismatch and pulls
    the reference toward the character so it stays reachable.

    Args:
        current_root_xy: Character root XY [N, 2].
        ref_root_xy: Reference clip root XY at the current playback time [N, 2].
        current_root_xy_vel: Character root XY velocity [N, 2].
        ref_root_xy_vel: Reference clip root XY velocity [N, 2].
        prev_offset_xy: Current persisted reference XY offset [N, 2].
        alpha_min: Blend factor at/below ``vel_err_low``.
        alpha_max: Blend factor at/above ``vel_err_high``.
        vel_err_low: Velocity error (m/s) below which ``alpha_min`` is used.
        vel_err_high: Velocity error (m/s) at/above which ``alpha_max`` is used.
        max_xy_speed: Cap (m/s) on offset change magnitude per step.
        dt: Control timestep (s), used with ``max_xy_speed`` to cap offset drift.
        eps: Numerical-stability epsilon.

    Returns:
        The new reference XY offset [N, 2].
    """
    # Instant-snap target: what the reference offset would be to co-locate roots.
    target_xy = current_root_xy - ref_root_xy

    # Unexpected root XY velocity -> smoothstep -> blend factor alpha.
    vel_err = torch.linalg.norm(current_root_xy_vel - ref_root_xy_vel, dim=-1)
    denom = max(vel_err_high - vel_err_low, eps)
    t = torch.clamp((vel_err - vel_err_low) / denom, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    alpha = alpha_min + (alpha_max - alpha_min) * t

    new_xy = prev_offset_xy + alpha.unsqueeze(-1) * (target_xy - prev_offset_xy)

    # Cap how fast the offset may move per step.
    delta = new_xy - prev_offset_xy
    max_delta = max_xy_speed * dt
    delta_norm = torch.linalg.norm(delta, dim=-1, keepdim=True)
    scale = torch.clamp(max_delta / delta_norm.clamp_min(eps), max=1.0)
    return prev_offset_xy + delta * scale




# =============================================================================
# Reward Combining
# =============================================================================


def combine_rewards(
    raw_rewards: Dict[str, Tensor],
    configs: Dict[str, Any],
    grace_mask: Optional[Tensor] = None,
    num_envs: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Combine raw rewards into final reward.
    
    RL Semantics:
    - multiplicative=True: rewards multiplied together (e.g., alive bonus)
    - Otherwise: weighted sum with optional clamping
    - grace_mask: zero rewards during grace period after reset
    
    Args:
        raw_rewards: Dict of {name: reward_tensor} from component execution.
        configs: Dict of {name: MdpComponent} where params contain metadata.
        grace_mask: Boolean mask [num_envs] - True where in grace period.
        num_envs: Number of environments.
        device: Device for tensors.
    
    Returns:
        Tuple of (combined_reward, logging_dict) where logging_dict contains
        raw and scaled rewards for debugging.
    """
    mult_reward = torch.ones(num_envs, device=device, dtype=torch.float)
    add_reward = torch.zeros(num_envs, device=device, dtype=torch.float)
    any_mult = False
    logging_dict: Dict[str, Tensor] = {}
    
    for name, reward in raw_rewards.items():
        router = configs[name]
        # Extract params from MdpComponent
        cfg = router.get_params() if is_mdp_component(router) else router
        
        # Apply grace period zeroing
        if cfg.get("zero_during_grace_period", False) and grace_mask is not None:
            reward = reward.clone()
            reward[grace_mask] = 0.0
        
        # Sanity check. A non-finite reward normally indicates a bug and fails fast.
        # In the opt-in impact-robustness mode (set_sanitize_non_finite_state), a rare
        # physics blowup can produce a huge-but-finite state that overflows a reward
        # (e.g. power ~ vel^2); tolerate it by zeroing the offending env's reward this
        # step. That env has already been flagged non-finite at the state level and is
        # reset by the termination path, so a single zeroed reward is harmless.
        if not torch.all(torch.isfinite(reward)):
            if not get_sanitize_non_finite_state():
                raise AssertionError(f"Reward '{name}' not finite")
            global _reward_sanitize_warn_count
            if _reward_sanitize_warn_count % 200 == 0:
                bad = (~torch.isfinite(reward)).sum().item()
                logger.warning(
                    "Sanitizing non-finite reward '%s' (%d envs) to 0.", name, bad
                )
            _reward_sanitize_warn_count += 1
            reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        logging_dict[f"raw_r/{name}"] = reward.clone()
        
        # Apply multiplicative or additive combining
        if cfg.get("multiplicative", False):
            mult_reward *= reward
            any_mult = True
        else:
            weight = cfg.get("weight", 0.0)
            if weight != 0:
                scaled = reward * weight
                
                # Apply clamping
                min_val = cfg.get("min_value")
                max_val = cfg.get("max_value")
                if min_val is not None:
                    scaled = torch.clamp(scaled, min=min_val)
                if max_val is not None:
                    scaled = torch.clamp(scaled, max=max_val)
                
                logging_dict[f"scaled_r/{name}"] = scaled.clone()
                add_reward += scaled
    
    # Combine multiplicative and additive
    if any_mult:
        logging_dict["multiplicative_reward"] = mult_reward.clone()
        logging_dict["additive_reward"] = add_reward.clone()
        return add_reward + mult_reward, logging_dict
    
    return add_reward, logging_dict


# =============================================================================
# Termination Combining
# =============================================================================


def combine_terminations(
    raw_terms: Dict[str, Tensor],
    configs: Dict[str, Any],
    num_envs: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """Combine termination conditions into reset/terminate buffers.
    
    RL Semantics:
    - terminate_on_true=False: inverts condition (terminate when function returns False)
    - All conditions OR'd together (any termination triggers reset)
    
    Args:
        raw_terms: Dict of {name: bool_tensor} from component execution.
        configs: Dict of {name: MdpComponent} where params contain metadata.
        num_envs: Number of environments.
        device: Device for tensors.
    
    Returns:
        Tuple of (reset_buf, terminate_buf, logging_dict) where:
        - reset_buf: Boolean [num_envs] - environments to reset
        - terminate_buf: Boolean [num_envs] - environments terminated (for bootstrapping)
        - logging_dict: Per-termination condition flags for debugging.
    """
    reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
    terminate_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
    logging_dict: Dict[str, Tensor] = {}
    
    for name, should_term in raw_terms.items():
        router = configs[name]
        # Extract params from MdpComponent
        cfg = router.get_params() if is_mdp_component(router) else router
        
        # Invert if terminate_on_true is False
        if not cfg.get("terminate_on_true", True):
            should_term = ~should_term
        
        # OR all conditions together
        reset_buf = reset_buf | should_term
        terminate_buf = terminate_buf | should_term
        logging_dict[f"termination/{name}"] = should_term.float().clone()
    
    return reset_buf, terminate_buf, logging_dict


# =============================================================================
# Evaluation Combining
# =============================================================================


def combine_evaluation(
    raw_values: Dict[str, Tensor],
    configs: Dict[str, Any],
    num_envs: int,
    device: torch.device,
) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
    """Combine evaluation component results into failure flags.

    Evaluation components return numeric values [num_envs]. If a component has
    a ``threshold`` in its static_params, the value is compared against it to
    determine failure. ``fail_above`` (default True) controls the comparison
    direction.

    Args:
        raw_values: Dict of {name: value_tensor} from ComponentManager.execute_all().
        configs: Dict of {name: MdpComponent} where static_params may contain
                 ``threshold`` and ``fail_above`` metadata.
        num_envs: Number of environments.
        device: Device for tensors.

    Returns:
        Tuple of (failed_buf, component_values, component_failures) where:
        - failed_buf: Boolean [num_envs] - environments that failed any component
        - component_values: Dict[str, Tensor] - raw numeric values per component
        - component_failures: Dict[str, Tensor] - boolean failure per component
    """
    failed_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
    component_values: Dict[str, Tensor] = {}
    component_failures: Dict[str, Tensor] = {}

    for name, value in raw_values.items():
        router = configs[name]
        cfg = router.get_params() if is_mdp_component(router) else router

        component_values[name] = value.clone()

        threshold = cfg.get("threshold", None)
        if threshold is not None:
            fail_above = cfg.get("fail_above", True)
            if fail_above:
                failed = value > threshold
            else:
                failed = value < threshold
            component_failures[name] = failed
            failed_buf = failed_buf | failed

    return failed_buf, component_values, component_failures
