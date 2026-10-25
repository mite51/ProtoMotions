# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Per-Joint Mini-Brain network for hierarchical motor control on SMPL.

This module implements Component 2 of the Hierarchical Shared Mini-Brain &
Analytical Subtree Load architecture:
- Evaluates all ``[E, J, F_in]`` joints in a single batched GEMM pass by
  reshaping to ``[E * J, F_in]`` for normalization and MLP trunk execution.
- Pools input normalization statistics across all ``E * J`` joints via
  ``RunningMeanStd``.
- Zero-initializes ``action_head`` so at initialization ``delta_u == 0`` and
  ``u_pd == q_des`` on all active axes.
- Enforces strict per-axis availability / stamina masking (``s_axis``) so
  unavailable or locked hinge axes output zero residual offset and zero gradient.
- Provides an optional per-joint value head ``V_joint: [E, J, 1]`` for
  short-horizon per-joint PPO / bandit updates.
- Provides an optional local 1-step forward dynamics surrogate head
  ``predict_next_joint_state`` for differentiable supervised in-simulator
  Mini-Brain training (Mode A).
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from protomotions.agents.utils.normalization import RunningMeanStd
from protomotions.agents.utils.training import get_activation_func


# Default 34-D per-joint feature layout matching joint_load_obs.py:
# [q_j(3), dq_j(3), delta_q(3), delta_dq(3), gravity_dir_local(3),
#  tau_grav_local(3), ang_momentum_local(3), subtree_mass(1),
#  effective_inertia(1), vel_pred_error(3), s_axis(3), static_joint_params(5)]
MINIBRAIN_FEATURE_DIM: int = 34

FEAT_Q_SLICE = slice(0, 3)
FEAT_DQ_SLICE = slice(3, 6)
FEAT_DELTA_Q_SLICE = slice(6, 9)
FEAT_DELTA_DQ_SLICE = slice(9, 12)
FEAT_GRAVITY_DIR_SLICE = slice(12, 15)
FEAT_TAU_GRAV_SLICE = slice(15, 18)
FEAT_ANG_MOMENTUM_SLICE = slice(18, 21)
FEAT_SUBTREE_MASS_SLICE = slice(21, 22)
FEAT_EFFECTIVE_INERTIA_SLICE = slice(22, 23)
FEAT_VEL_PRED_ERROR_SLICE = slice(23, 26)
FEAT_S_AXIS_SLICE = slice(26, 29)
FEAT_STATIC_PARAMS_SLICE = slice(29, 34)

S_AXIS_ACTIVE_EPS: float = 1e-6


@dataclass
class SharedJointMiniBrainConfig:
    """Configuration for ``SharedJointMiniBrain``."""

    _target_: str = "protomotions.agents.common.minibrain.SharedJointMiniBrain"
    num_joints: int = 23
    dofs_per_joint: int = 3
    feature_dim: int = MINIBRAIN_FEATURE_DIM
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    activation: str = "silu"
    use_layer_norm: bool = True
    normalize_obs: bool = True
    norm_clamp_value: float = 5.0
    norm_ema_decay: Optional[float] = 0.999
    max_residual_offset: float = 0.75
    include_value_head: bool = True
    include_dynamics_head: bool = True


class MiniBrainOutput(dict):
    """Dictionary output of ``SharedJointMiniBrain.forward`` with attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class SharedJointMiniBrain(nn.Module):
    """Shared per-joint spinal reflex network evaluated across ``[E, J, F_in]``."""

    def __init__(self, config: Optional[SharedJointMiniBrainConfig] = None):
        super().__init__()
        self.config = config if config is not None else SharedJointMiniBrainConfig()
        self._freeze_running: bool = False
        self.checkpoint_metadata: Dict[str, Any] = {}

        if self.config.normalize_obs:
            self.obs_norm: Optional[RunningMeanStd] = RunningMeanStd(
                fabric=None,
                shape=(self.config.feature_dim,),
                device="cpu",
                clamp_value=self.config.norm_clamp_value,
                ema_decay=self.config.norm_ema_decay,
            )
        else:
            self.obs_norm = None
        self.running_obs_norm = self.obs_norm

        trunk_layers: List[nn.Module] = []
        in_dim = self.config.feature_dim
        for hidden_dim in self.config.hidden_dims:
            linear = nn.Linear(in_dim, hidden_dim)
            nn.init.orthogonal_(linear.weight)
            nn.init.zeros_(linear.bias)
            trunk_layers.append(linear)
            if self.config.use_layer_norm:
                trunk_layers.append(nn.LayerNorm(hidden_dim))
            trunk_layers.append(
                get_activation_func(self.config.activation, return_type="nn")
            )
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)
        self.trunk_out_dim = in_dim

        # Policy Action Head: zero-initialized so raw_offset == 0 and u_pd == q_des at init
        self.action_head = nn.Linear(self.trunk_out_dim, self.config.dofs_per_joint)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        self.action_head._is_zero_init_head = True

        # Zero-initialized physical basis weights [4, dofs_per_joint] for:
        #   0: (kd / kp) * dq_des          (velocity damping lead)
        #   1: -tau_grav / (s_axis * kp)   (stamina-scaled gravity torque compensation)
        #   2: 0.15 * L_mom / (s_axis * kp)(stamina-scaled subtree momentum compensation)
        #   3: delta_q                     (proportional position lead)
        # Zero at init preserves exact u_pd == q_des identity at step 0 while giving
        # fast, zero-bias physical convergence alongside the nonlinear MLP residual.
        self.physics_basis_weights = nn.Parameter(
            torch.zeros(4, self.config.dofs_per_joint)
        )
        self.physics_basis_weights._is_zero_init_head = True

        # Optional Per-Joint Value Head: outputs V_joint [E, J, 1]
        if self.config.include_value_head:
            self.value_head: Optional[nn.Linear] = nn.Linear(self.trunk_out_dim, 1)
            nn.init.orthogonal_(self.value_head.weight, gain=1.0)
            nn.init.zeros_(self.value_head.bias)
        else:
            self.value_head = None

        # Optional Local 1-Step Forward Dynamics Surrogate Head:
        # Maps (norm_feat_j, delta_u_j) -> (pred_delta_q_next[3], pred_delta_dq_next[3])
        if self.config.include_dynamics_head:
            dyn_in_dim = self.config.feature_dim + self.config.dofs_per_joint
            dyn_hidden = (
                self.config.hidden_dims[-1] if len(self.config.hidden_dims) > 0 else 128
            )
            dyn_fc1 = nn.Linear(dyn_in_dim, dyn_hidden)
            dyn_fc2 = nn.Linear(dyn_hidden, 2 * self.config.dofs_per_joint)
            nn.init.orthogonal_(dyn_fc1.weight)
            nn.init.zeros_(dyn_fc1.bias)
            nn.init.orthogonal_(dyn_fc2.weight, gain=0.1)
            nn.init.zeros_(dyn_fc2.bias)
            dyn_layers: List[nn.Module] = [dyn_fc1]
            if self.config.use_layer_norm:
                dyn_layers.append(nn.LayerNorm(dyn_hidden))
            dyn_layers.append(
                get_activation_func(self.config.activation, return_type="nn")
            )
            dyn_layers.append(dyn_fc2)
            self.dynamics_head: Optional[nn.Sequential] = nn.Sequential(*dyn_layers)
        else:
            self.dynamics_head = None

    def reset_action_head_to_zero(self) -> None:
        """Reset ``action_head`` and ``physics_basis_weights`` to exact zeros."""
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        self.action_head._is_zero_init_head = True
        nn.init.zeros_(self.physics_basis_weights)
        self.physics_basis_weights._is_zero_init_head = True

    def compute_physics_basis_offset(
        self, joint_features_3d: Tensor, s_axis_3d: Tensor
    ) -> Tensor:
        """Compute linear combination of the 4 physical basis vectors ``[E, J, D]``."""
        if joint_features_3d.shape[-1] < 31:
            return torch.zeros_like(s_axis_3d)

        dq_j = joint_features_3d[..., FEAT_DQ_SLICE]
        delta_q = joint_features_3d[..., FEAT_DELTA_Q_SLICE]
        delta_dq = joint_features_3d[..., FEAT_DELTA_DQ_SLICE]
        dq_des = dq_j + delta_dq

        # Unscale tau_grav (stored * 0.01) and ang_momentum (stored * 0.1)
        tau_grav = joint_features_3d[..., FEAT_TAU_GRAV_SLICE] * 100.0
        ang_mom = joint_features_3d[..., FEAT_ANG_MOMENTUM_SLICE] * 10.0

        # Extract kp (stored / 1000) and kd (stored / 100)
        kp = (joint_features_3d[..., 29:30] * 1000.0).clamp_min(50.0)
        kd = (joint_features_3d[..., 30:31] * 100.0).clamp_min(1.0)
        stamina_eff = s_axis_3d.clamp_min(0.20)

        b_vel_lead = (kd / kp) * dq_des
        b_grav_comp = -tau_grav / (stamina_eff * kp)
        b_mom_comp = (0.15 * ang_mom) / (stamina_eff * kp)
        b_pos_lead = 0.10 * delta_q

        w = self.physics_basis_weights  # [4, D]
        scale_inv = 1.0 / max(float(self.config.max_residual_offset), 1e-6)
        return scale_inv * (
            w[0] * b_vel_lead
            + w[1] * b_grav_comp
            + w[2] * b_mom_comp
            + w[3] * b_pos_lead
        )

    def freeze(self) -> "SharedJointMiniBrain":
        """Freeze all parameters and running normalization statistics."""
        self._freeze_running = True
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()
        return self

    def unfreeze(self, unfreeze_running_stats: bool = True) -> "SharedJointMiniBrain":
        """Unfreeze parameters and optionally resume running normalization updates."""
        self._freeze_running = not unfreeze_running_stats
        for param in self.parameters():
            param.requires_grad_(True)
        self.train()
        return self

    def _reshape_joint_features(self, joint_features: Tensor) -> Tuple[Tensor, int, int]:
        """Validate and reshape ``joint_features`` into ``[E, J, F_in]``."""
        if joint_features.dim() == 2:
            num_envs = joint_features.shape[0]
            expected_flat = self.config.num_joints * self.config.feature_dim
            if joint_features.shape[1] != expected_flat:
                raise ValueError(
                    f"Expected 2D joint_features of shape [E, {expected_flat}], "
                    f"got {tuple(joint_features.shape)}"
                )
            joint_features = joint_features.reshape(
                num_envs, self.config.num_joints, self.config.feature_dim
            )
        elif joint_features.dim() == 3:
            if joint_features.shape[-1] != self.config.feature_dim:
                raise ValueError(
                    f"Expected joint_features last dim {self.config.feature_dim}, "
                    f"got {tuple(joint_features.shape)}"
                )
        else:
            raise ValueError(
                f"Expected 2D or 3D joint_features tensor, got shape {tuple(joint_features.shape)}"
            )
        return joint_features, joint_features.shape[0], joint_features.shape[1]

    def _normalize_flat_features(
        self, flat_features: Tensor, update_stats: bool = True
    ) -> Tensor:
        """Normalize flattened ``[E * J, F_in]`` features with shared ``RunningMeanStd``."""
        if not self.config.normalize_obs or self.obs_norm is None:
            return flat_features

        if self.obs_norm.mean.device != flat_features.device:
            self.obs_norm.to(flat_features.device)

        norm_features = self.obs_norm.normalize(flat_features)
        if update_stats and self.training and not self._freeze_running:
            self.obs_norm.record_moments(flat_features.detach())
        return norm_features

    def extract_s_axis(
        self,
        joint_features: Tensor,
        s_axis: Optional[Tensor] = None,
    ) -> Tensor:
        """Return ``s_axis`` shaped ``[E, J, D]`` from explicit input or ``joint_features``."""
        joint_features_3d, num_envs, num_joints = self._reshape_joint_features(
            joint_features
        )
        dofs = self.config.dofs_per_joint
        if s_axis is None:
            if joint_features_3d.shape[-1] >= FEAT_S_AXIS_SLICE.stop:
                return joint_features_3d[..., FEAT_S_AXIS_SLICE]
            return torch.ones(
                num_envs,
                num_joints,
                dofs,
                device=joint_features_3d.device,
                dtype=joint_features_3d.dtype,
            )

        if s_axis.dim() == 2:
            s_axis = s_axis.reshape(num_envs, num_joints, dofs)
        elif s_axis.dim() != 3 or s_axis.shape != (num_envs, num_joints, dofs):
            raise ValueError(
                f"Expected s_axis of shape {(num_envs, num_joints, dofs)}, "
                f"got {tuple(s_axis.shape)}"
            )
        return s_axis.to(device=joint_features_3d.device, dtype=joint_features_3d.dtype)

    def compute_active_mask(
        self,
        joint_features: Tensor,
        s_axis: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute ``(s_axis, active_mask)`` each of shape ``[E, J, D]``."""
        s_axis_3d = self.extract_s_axis(joint_features, s_axis=s_axis)
        active_mask = (s_axis_3d > S_AXIS_ACTIVE_EPS).to(joint_features.dtype)
        return s_axis_3d, active_mask

    def forward(
        self,
        joint_features: Tensor,
        q_des: Optional[Tensor] = None,
        q_current: Optional[Tensor] = None,
        s_axis: Optional[Tensor] = None,
    ) -> MiniBrainOutput:
        """Evaluate the Shared Mini-Brain across all ``[E, J]`` joints in one batched GEMM.

        Args:
            joint_features: Per-joint features ``[E, J, F_in]`` (or flattened ``[E, J * F_in]``).
            q_des: Desired joint angles ``[E, J, 3]`` or ``[E, J * 3]``. If ``None``,
                inferred from ``joint_features[..., 0:3] + joint_features[..., 6:9]``.
            q_current: Current joint angles ``[E, J, 3]`` or ``[E, J * 3]``. When provided,
                unavailable axes (``active_mask == 0``) lock to ``q_current``; otherwise
                they default to ``q_des * active_mask``.
            s_axis: Optional per-axis availability/stamina tensor ``[E, J, 3]``. If ``None``,
                extracted from ``joint_features[..., 26:29]``.

        Returns:
            ``MiniBrainOutput`` containing:
                - ``u_pd``: Flattened PD target ``[E, J * 3]``
                - ``u_pd_3d``: Per-joint PD target ``[E, J, 3]``
                - ``delta_u``: Masked residual PD offset ``[E, J, 3]``
                - ``delta_u_flat``: Flattened masked residual PD offset ``[E, J * 3]``
                - ``raw_offset``: Unmasked action head output ``[E, J, 3]``
                - ``active_mask``: Binary axis availability mask ``[E, J, 3]``
                - ``s_axis``: Per-axis availability/stamina ``[E, J, 3]``
                - ``q_base``: Masked base pose ``[E, J, 3]``
                - ``joint_value``: Per-joint value estimate ``[E, J, 1]`` (or ``None``)
                - ``trunk_features``: Hidden features ``[E, J, H]``
                - ``norm_features``: Normalized input features ``[E, J, F_in]``
        """
        joint_features_3d, num_envs, num_joints = self._reshape_joint_features(
            joint_features
        )
        dofs = self.config.dofs_per_joint
        feature_dim = self.config.feature_dim

        s_axis_3d, active_mask = self.compute_active_mask(
            joint_features_3d, s_axis=s_axis
        )

        # Single batched GPU GEMM across [E * J, F_in]
        flat_features = joint_features_3d.reshape(num_envs * num_joints, feature_dim)
        norm_flat = self._normalize_flat_features(flat_features, update_stats=True)
        trunk_flat = self.trunk(norm_flat)
        raw_offset_flat = self.action_head(trunk_flat)

        raw_mlp_offset = raw_offset_flat.reshape(num_envs, num_joints, dofs)
        physics_offset = self.compute_physics_basis_offset(
            joint_features_3d, s_axis_3d
        )
        raw_offset = raw_mlp_offset + physics_offset
        trunk_features = trunk_flat.reshape(num_envs, num_joints, self.trunk_out_dim)
        norm_features = norm_flat.reshape(num_envs, num_joints, feature_dim)

        # Masked residual offset computed directly from raw_offset
        delta_u = (
            self.config.max_residual_offset * torch.tanh(raw_offset) * active_mask
        )

        # Resolve q_des_3d
        if q_des is None:
            q_des_3d = (
                joint_features_3d[..., FEAT_Q_SLICE]
                + joint_features_3d[..., FEAT_DELTA_Q_SLICE]
            )
        else:
            q_des_3d = q_des.reshape(num_envs, num_joints, dofs).to(
                device=joint_features_3d.device, dtype=joint_features_3d.dtype
            )

        # Resolve q_base on active vs locked axes
        if q_current is not None:
            q_current_3d = q_current.reshape(num_envs, num_joints, dofs).to(
                device=joint_features_3d.device, dtype=joint_features_3d.dtype
            )
            q_base = torch.where(active_mask > 0.5, q_des_3d, q_current_3d)
        else:
            q_base = q_des_3d * active_mask

        u_pd_3d = q_base + delta_u
        u_pd = u_pd_3d.reshape(num_envs, num_joints * dofs)
        delta_u_flat = delta_u.reshape(num_envs, num_joints * dofs)

        if self.value_head is not None:
            joint_value = self.value_head(trunk_flat).reshape(num_envs, num_joints, 1)
        else:
            joint_value = None

        return MiniBrainOutput(
            u_pd=u_pd,
            u_pd_3d=u_pd_3d,
            delta_u=delta_u,
            delta_u_flat=delta_u_flat,
            raw_offset=raw_offset,
            raw_mlp_offset=raw_mlp_offset,
            physics_offset=physics_offset,
            active_mask=active_mask,
            s_axis=s_axis_3d,
            q_base=q_base,
            joint_value=joint_value,
            trunk_features=trunk_features,
            norm_features=norm_features,
        )

    def compute_pd_target(
        self,
        joint_features: Tensor,
        q_des: Tensor,
        q_current: Optional[Tensor] = None,
        s_axis: Optional[Tensor] = None,
    ) -> Tensor:
        """Convenience method returning flattened ``u_pd`` of shape ``[E, J * 3]``."""
        return self.forward(
            joint_features=joint_features,
            q_des=q_des,
            q_current=q_current,
            s_axis=s_axis,
        )["u_pd"]

    def evaluate_value(self, joint_features: Tensor) -> Tensor:
        """Evaluate per-joint value head ``V_joint`` of shape ``[E, J, 1]``."""
        if self.value_head is None:
            raise RuntimeError(
                "SharedJointMiniBrain was initialized with include_value_head=False"
            )
        joint_features_3d, num_envs, num_joints = self._reshape_joint_features(
            joint_features
        )
        flat_features = joint_features_3d.reshape(
            num_envs * num_joints, self.config.feature_dim
        )
        norm_flat = self._normalize_flat_features(flat_features, update_stats=False)
        trunk_flat = self.trunk(norm_flat)
        return self.value_head(trunk_flat).reshape(num_envs, num_joints, 1)

    def predict_next_joint_state(
        self,
        joint_features: Tensor,
        delta_u: Tensor,
        s_axis: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Predict 1-step next joint state ``(q_next_pred, dq_next_pred)`` each ``[E, J, 3]``.

        Masks unavailable/locked axes (``active_mask == 0``) so locked axes predict
        zero state delta and receive zero gradient with respect to ``delta_u``.
        """
        if self.dynamics_head is None:
            raise RuntimeError(
                "SharedJointMiniBrain was initialized with include_dynamics_head=False"
            )

        joint_features_3d, num_envs, num_joints = self._reshape_joint_features(
            joint_features
        )
        dofs = self.config.dofs_per_joint
        feature_dim = self.config.feature_dim

        delta_u_3d = delta_u.reshape(num_envs, num_joints, dofs).to(
            device=joint_features_3d.device, dtype=joint_features_3d.dtype
        )
        _, active_mask = self.compute_active_mask(joint_features_3d, s_axis=s_axis)

        masked_delta_u = delta_u_3d * active_mask

        flat_features = joint_features_3d.reshape(num_envs * num_joints, feature_dim)
        norm_flat = self._normalize_flat_features(flat_features, update_stats=False)
        flat_delta_u = masked_delta_u.reshape(num_envs * num_joints, dofs)

        dyn_in = torch.cat([norm_flat, flat_delta_u], dim=-1)
        dyn_out = self.dynamics_head(dyn_in).reshape(num_envs, num_joints, 2 * dofs)
        pred_delta_q_raw, pred_delta_dq_raw = torch.split(dyn_out, dofs, dim=-1)

        # Residual PD target coupling + learned dynamics adjustment on active axes
        pred_delta_q = (pred_delta_q_raw + 0.20 * masked_delta_u) * active_mask
        pred_delta_dq = (pred_delta_dq_raw + 2.0 * masked_delta_u) * active_mask

        q_j = joint_features_3d[..., FEAT_Q_SLICE]
        dq_j = joint_features_3d[..., FEAT_DQ_SLICE]

        q_next_pred = q_j + pred_delta_q
        dq_next_pred = dq_j + pred_delta_dq
        return q_next_pred, dq_next_pred

    def save_checkpoint(
        self,
        path: Union[str, Path],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Save ``SharedJointMiniBrain`` config, weights, and normalizer stats."""
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "config": asdict(self.config),
            "model_state_dict": self.state_dict(),
            "freeze_running": self._freeze_running,
            "extra_metadata": extra_metadata if extra_metadata is not None else {},
        }
        torch.save(payload, ckpt_path)
        return payload

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        freeze: bool = True,
    ) -> "SharedJointMiniBrain":
        """Load ``SharedJointMiniBrain`` from a checkpoint file."""
        ckpt_path = Path(path)
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)

        if isinstance(payload, dict) and "model_state_dict" in payload:
            cfg_data = payload.get("config", {})
            if isinstance(cfg_data, SharedJointMiniBrainConfig):
                config = cfg_data
            else:
                config = SharedJointMiniBrainConfig(**cfg_data)
            model = cls(config=config)
            model = model.to(device)
            model.load_state_dict(payload["model_state_dict"])
            model.checkpoint_metadata = dict(payload.get("extra_metadata", {}))
            if freeze:
                model.freeze()
            else:
                model.unfreeze(
                    unfreeze_running_stats=not bool(
                        payload.get("freeze_running", False)
                    )
                )
        else:
            model = cls()
            model = model.to(device)
            model.load_state_dict(payload)
            if freeze:
                model.freeze()
            else:
                model.unfreeze()

        return model
