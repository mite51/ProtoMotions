# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical PPO Actor & Agent ("Pattern B" Spinal Cord Hierarchy).

Implements Component 4 of the Hierarchical Shared Mini-Brain & Analytical Subtree
Load architecture on SMPL:
1. ``HierarchicalPPOActorConfig`` & ``HierarchicalPPOActor``:
   - Wraps the Main Brain policy trunk (``self.mu``) and the shared per-joint
     spinal reflex network (``self.minibrain: SharedJointMiniBrain``).
   - Optionally exposes the unadjusted Mini-Brain reflex offset
     (``minibrain_base_cmd`` / ``minibrain_base_delta_u``) to the Main Brain before
     computing the Main Brain's residual pose adjustment ``delta_q``.
   - Evaluates the Mini-Brain on the adjusted target pose ``q_des_adj = q_ref + delta_q``
     to produce the final PD target ``u_pd`` in radians (or normalized action space).
   - Preserves frozen / zero-initialized Mini-Brain weights when ``model.apply(weight_init_trainable)``
     is invoked during ``PPO.create_model()``.
2. ``HierarchicalPPOAgentConfig`` & ``HierarchicalPPO``:
   - Critic Warmup schedule (``critic_warmup_epochs``): skips actor optimizer steps
     while training the value critic during early epochs to prevent initial value-gradient
     corruption of the pretrained coordination policy.
   - Horizon Curriculum schedule (``compute_curriculum_episode_length``): linearly
     ramps ``env.max_episode_length`` from ``initial_max_episode_length`` to
     ``target_max_episode_length`` across ``horizon_ramp_epochs``.
   - Optional ``EdgeCaseCollector`` integration during rollout collection.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

import torch
from torch import Tensor, distributions
from tensordict import TensorDict
from lightning.fabric import Fabric

from protomotions.agents.base_agent.model import ProtoMotionsTensorDictModule
from protomotions.agents.common.common import (
    MODULE_INTERNALS_KEY,
    weight_init,
    weight_init_trainable,
)
from protomotions.agents.common.minibrain import (
    FEAT_ANG_MOMENTUM_SLICE,
    FEAT_DELTA_DQ_SLICE,
    FEAT_DELTA_Q_SLICE,
    FEAT_DQ_SLICE,
    FEAT_Q_SLICE,
    FEAT_S_AXIS_SLICE,
    FEAT_TAU_GRAV_SLICE,
    FEAT_VEL_PRED_ERROR_SLICE,
    SharedJointMiniBrain,
    SharedJointMiniBrainConfig,
)
from protomotions.agents.ppo.agent import PPO
from protomotions.agents.ppo.config import PPOActorConfig, PPOAgentConfig, PPOModelConfig
from protomotions.agents.ppo.model import PPOActor
from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.diagnostics.edge_case_collector import (
    EdgeCaseCollector,
    EdgeCaseCollectorConfig,
)

log = logging.getLogger(__name__)


@dataclass
class HierarchicalPPOActorConfig(PPOActorConfig):
    """Configuration for Pattern B Hierarchical PPO Actor."""

    mu_key: str = field(
        default="actor_trunk_out",
        metadata={"help": "The key of the output of the Main Brain mu model."},
    )
    _target_: str = "protomotions.agents.ppo.hierarchical_model.HierarchicalPPOActor"
    minibrain_config: SharedJointMiniBrainConfig = field(
        default_factory=SharedJointMiniBrainConfig,
        metadata={"help": "Configuration for the per-joint SharedJointMiniBrain."},
    )
    minibrain_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to pretrained SharedJointMiniBrain checkpoint."},
    )
    freeze_minibrain: bool = field(
        default=True,
        metadata={"help": "Freeze SharedJointMiniBrain weights and running stats."},
    )
    minibrain_features_key: str = field(
        default="minibrain_features",
        metadata={"help": "TensorDict key for per-joint Mini-Brain features [E, J*34] or [E, J, 34]."},
    )
    residual_pose_scale: float = field(
        default=1.8,
        metadata={"help": "Max radians adjustment delta_q = residual_pose_scale * tanh(raw_main_mu)."},
    )
    init_main_brain_last_layer_scale: float = field(
        default=0.01,
        metadata={
            "help": "Scale applied to the final linear layer of self.mu after weight_init "
            "so step-0 residual pose adjustments delta_q start near zero."
        },
    )
    expose_minibrain_base_to_main: bool = field(
        default=True,
        metadata={
            "help": "If True, evaluates minibrain on unadjusted q_ref under no_grad to "
            "populate minibrain_base_cmd / minibrain_base_delta_u before running mu_model."
        },
    )
    pd_action_offset: Optional[Tensor] = field(
        default=None,
        metadata={"help": "Optional PD action offset [num_dofs] for normalized action conversion."},
    )
    pd_action_scale: Optional[Tensor] = field(
        default=None,
        metadata={"help": "Optional PD action scale [num_dofs] for normalized action conversion."},
    )
    convert_to_normalized_action_space: bool = field(
        default=False,
        metadata={
            "help": "When True and pd_action_offset/scale are provided, converts u_pd in "
            "radians to normalized [-1, 1] space (u_pd - offset) / scale."
        },
    )


class HierarchicalPPOActor(PPOActor):
    """Pattern B Hierarchical Actor combining Main Brain (``self.mu``) and ``SharedJointMiniBrain``."""

    config: HierarchicalPPOActorConfig

    def __init__(self, config: HierarchicalPPOActorConfig):
        if config.num_out is None:
            config.num_out = (
                config.minibrain_config.num_joints
                * config.minibrain_config.dofs_per_joint
            )
        super().__init__(config)

        if self.config.minibrain_features_key not in self.in_keys:
            self.in_keys = list(self.in_keys) + [self.config.minibrain_features_key]

        if (
            self.config.minibrain_checkpoint is not None
            and str(self.config.minibrain_checkpoint).strip() != ""
        ):
            self.minibrain = SharedJointMiniBrain.load_checkpoint(
                self.config.minibrain_checkpoint,
                freeze=self.config.freeze_minibrain,
            )
        else:
            self.minibrain = SharedJointMiniBrain(config=self.config.minibrain_config)
            if self.config.freeze_minibrain:
                self.minibrain.freeze()
        self._main_brain_last_layer_scaled: bool = False

    def _scale_main_brain_last_layer_if_initialized(self) -> bool:
        """Scale the final linear layer of ``self.mu`` once its weights are materialized."""
        if getattr(self, "_main_brain_last_layer_scaled", False):
            return True
        last_scale = getattr(self.config, "init_main_brain_last_layer_scale", None)
        if last_scale is None:
            self._main_brain_last_layer_scaled = True
            return True
        linear_layers = [
            m
            for m in self.mu.modules()
            if isinstance(m, (torch.nn.Linear, torch.nn.modules.lazy.LazyModuleMixin))
            and hasattr(m, "weight")
        ]
        if not linear_layers:
            return False
        last_linear = linear_layers[-1]
        if isinstance(last_linear.weight, torch.nn.parameter.UninitializedParameter):
            return False
        if not getattr(last_linear, "_is_zero_init_head", False):
            with torch.no_grad():
                last_linear.weight.mul_(float(last_scale))
                if getattr(last_linear, "bias", None) is not None:
                    last_linear.bias.zero_()
        self._main_brain_last_layer_scaled = True
        return True

    def train(self, mode: bool = True) -> "HierarchicalPPOActor":
        """Keep ``self.minibrain`` in eval mode with frozen stats when ``freeze_minibrain=True``."""
        super().train(mode)
        if self.config.freeze_minibrain and hasattr(self, "minibrain"):
            self.minibrain.eval()
            self.minibrain._freeze_running = True
        return self

    def apply(self, fn):
        """Apply ``fn`` to submodules while protecting ``self.minibrain`` from weight re-initialization."""
        fn_name = getattr(fn, "__name__", "")
        is_weight_init_fn = fn in (weight_init, weight_init_trainable) or fn_name in (
            "weight_init",
            "weight_init_trainable",
        )

        if is_weight_init_fn:
            # Only apply generic weight initialization to the Main Brain mu network.
            # SharedJointMiniBrain manages its own orthogonal trunk + zero-init action_head
            # (or pretrained checkpoint weights).
            self.mu.apply(fn)
            self._scale_main_brain_last_layer_if_initialized()
            fn(self)
            if self.config.freeze_minibrain:
                self.minibrain.freeze()
            return self

        if self.config.freeze_minibrain and hasattr(self, "minibrain"):
            saved_params = [
                (p, p.data.clone()) for p in self.minibrain.parameters()
            ]
            super().apply(fn)
            for p, data_copy in saved_params:
                p.data.copy_(data_copy)
                p.requires_grad_(False)
            self.minibrain.eval()
            self.minibrain._freeze_running = True
            return self

        return super().apply(fn)

    def forward(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        """Pattern B forward pass:
        1. Extract ``minibrain_features`` [E, J, 34], ``q_current`` [E, J, 3], and ``q_ref`` [E, J, 3].
        2. Optionally query ``self.minibrain`` on unadjusted ``q_ref`` under ``torch.no_grad()``
           to expose ``minibrain_base_cmd`` / ``minibrain_base_delta_u`` [E, 69] to ``self.mu``.
        3. Evaluate Main Brain ``self.mu`` to get residual pose adjustment ``delta_q``.
        4. Update ``q_des_adj = q_ref + delta_q`` and ``adj_features[..., 6:9]``, then evaluate
           ``self.minibrain`` to produce ``mean_u_pd`` [E, 69].
        5. Sample Gaussian exploration action and compute ``neglogp``.
        """
        raw_features = tensordict[self.config.minibrain_features_key]
        minibrain_features, num_envs, num_joints = (
            self.minibrain._reshape_joint_features(raw_features)
        )
        dofs_per_joint = self.minibrain.config.dofs_per_joint
        action_dim = num_joints * dofs_per_joint

        # 2. Extract current joint angles and reference target pose
        q_current = minibrain_features[..., FEAT_Q_SLICE]
        ref_delta_q = minibrain_features[..., FEAT_DELTA_Q_SLICE]
        q_ref = q_current + ref_delta_q

        # 3. Optionally expose unadjusted Mini-Brain reflex command to Main Brain
        if self.config.expose_minibrain_base_to_main:
            with torch.no_grad():
                prev_freeze_running = self.minibrain._freeze_running
                self.minibrain._freeze_running = True
                try:
                    base_out = self.minibrain(
                        minibrain_features,
                        q_des=q_ref,
                        q_current=q_current,
                    )
                finally:
                    self.minibrain._freeze_running = prev_freeze_running
            base_delta_u = base_out["delta_u_flat"]
            tensordict["minibrain_base_cmd"] = base_delta_u
            tensordict["minibrain_base_delta_u"] = base_delta_u

        # 4. Evaluate Main Brain mu network
        was_scaled = getattr(self, "_main_brain_last_layer_scaled", True)
        if isinstance(self.mu, ProtoMotionsTensorDictModule):
            tensordict = self.mu(tensordict, log_internals=log_internals)
        else:
            tensordict = self.mu(tensordict)
        if not was_scaled and self._scale_main_brain_last_layer_if_initialized():
            if isinstance(self.mu, ProtoMotionsTensorDictModule):
                tensordict = self.mu(tensordict, log_internals=log_internals)
            else:
                tensordict = self.mu(tensordict)
        mu_raw = tensordict[self.config.mu_key]

        # 5. Bounded residual pose adjustment delta_q
        _, active_mask_3d = self.minibrain.compute_active_mask(minibrain_features)
        active_mask_flat = active_mask_3d.reshape(num_envs, action_dim)
        delta_q_flat = (
            self.config.residual_pose_scale * torch.tanh(mu_raw) * active_mask_flat
        )
        delta_q_3d = delta_q_flat.reshape(num_envs, num_joints, dofs_per_joint)

        # 6. Adjusted desired pose for the Mini-Brain
        q_des_adj = q_ref + delta_q_3d

        # 7. Update delta_q slice [..., 6:9] of minibrain_features (out-of-place) and evaluate Mini-Brain
        raw_adj_err = q_des_adj - q_current
        adj_delta_q = torch.atan2(torch.sin(raw_adj_err), torch.cos(raw_adj_err))
        adj_features = torch.cat(
            [
                minibrain_features[..., : FEAT_DELTA_Q_SLICE.start],
                adj_delta_q,
                minibrain_features[..., FEAT_DELTA_Q_SLICE.stop :],
            ],
            dim=-1,
        )
        adj_out = self.minibrain(
            adj_features,
            q_des=q_des_adj,
            q_current=q_current,
        )

        # 8. Obtain mean PD target mean_u_pd [E, 69]
        active_mask = adj_out["active_mask"].reshape(num_envs, action_dim)
        q_curr_flat = q_current.reshape(num_envs, action_dim)
        mean_u_pd = adj_out["u_pd"] * active_mask + q_curr_flat * (1.0 - active_mask)
        if (
            self.config.convert_to_normalized_action_space
            and self.config.pd_action_offset is not None
            and self.config.pd_action_scale is not None
        ):
            offset = self.config.pd_action_offset.to(
                device=mean_u_pd.device, dtype=mean_u_pd.dtype
            )
            scale = self.config.pd_action_scale.to(
                device=mean_u_pd.device, dtype=mean_u_pd.dtype
            ).clamp_min(1e-6)
            mean_u_pd = (mean_u_pd - offset) / scale

        # 9. Sample exploration noise
        std = torch.exp(self.logstd)
        dist = distributions.Normal(mean_u_pd, mean_u_pd * 0 + std)
        action = dist.sample()
        neglogp = -dist.log_prob(action).sum(dim=-1)

        # 10. Store outputs in TensorDict
        tensordict["action"] = action
        tensordict["mean_action"] = mean_u_pd
        tensordict["neglogp"] = neglogp
        tensordict["main_brain_delta_q"] = delta_q_flat
        tensordict["minibrain_delta_u"] = adj_out["delta_u_flat"]

        if log_internals:
            internals = tensordict.get(MODULE_INTERNALS_KEY, None)
            if internals is None:
                internals = TensorDict({}, batch_size=tensordict.batch_size)
            internals["main_brain_delta_q_abs_mean"] = (
                delta_q_flat.abs().mean(dim=-1).detach()
            )
            internals["minibrain_delta_u_abs_mean"] = (
                adj_out["delta_u_flat"].abs().mean(dim=-1).detach()
            )
            tensordict[MODULE_INTERNALS_KEY] = internals

        return tensordict


def compute_curriculum_episode_length(
    current_epoch: Union[int, "HierarchicalPPOAgentConfig"],
    initial_max_episode_length: int = 32,
    target_max_episode_length: int = 1000,
    horizon_ramp_epochs: int = 200,
    horizon_curriculum_enabled: bool = True,
) -> int:
    """Compute linearly ramped ``max_episode_length`` for the given epoch.

    Can be called either with explicit parameters:
        ``compute_curriculum_episode_length(epoch, initial_max_episode_length=32, ...)``
    or with ``(config, epoch)`` where ``config`` is a ``HierarchicalPPOAgentConfig``.
    """
    if isinstance(current_epoch, HierarchicalPPOAgentConfig):
        cfg = current_epoch
        epoch = int(initial_max_episode_length)
        initial_len = int(cfg.initial_max_episode_length)
        target_len = int(cfg.target_max_episode_length)
        ramp_epochs = int(cfg.horizon_ramp_epochs)
        enabled = bool(cfg.horizon_curriculum_enabled)
    else:
        epoch = int(current_epoch)
        initial_len = int(initial_max_episode_length)
        target_len = int(target_max_episode_length)
        ramp_epochs = int(horizon_ramp_epochs)
        enabled = bool(horizon_curriculum_enabled)

    if not enabled or ramp_epochs <= 0:
        return target_len
    if epoch <= 0:
        return initial_len
    if epoch >= ramp_epochs:
        return target_len

    alpha = float(epoch) / float(ramp_epochs)
    return int(round(initial_len + alpha * (target_len - initial_len)))


@dataclass
class HierarchicalPPOAgentConfig(PPOAgentConfig):
    """Configuration for ``HierarchicalPPO`` agent with Critic Warmup and Horizon Curriculum."""

    _target_: str = "protomotions.agents.ppo.hierarchical_model.HierarchicalPPO"
    model: PPOModelConfig = field(
        default_factory=lambda: PPOModelConfig(actor=HierarchicalPPOActorConfig()),
        metadata={"help": "Hierarchical PPO model configuration."},
    )
    batch_size: int = field(
        default=4096,
        metadata={"help": "Minibatch size for PPO updates."},
    )
    training_max_steps: int = field(
        default=1_000_000_000,
        metadata={"help": "Maximum environment steps for training."},
    )
    critic_warmup_epochs: int = field(
        default=10,
        metadata={
            "help": "Number of initial epochs to train only the value critic while skipping actor updates."
        },
    )
    horizon_curriculum_enabled: bool = field(
        default=True,
        metadata={"help": "Whether to linearly ramp env.max_episode_length across epochs."},
    )
    initial_max_episode_length: int = field(
        default=32,
        metadata={"help": "Starting max_episode_length at epoch 0 when curriculum is enabled."},
    )
    target_max_episode_length: int = field(
        default=1000,
        metadata={"help": "Final max_episode_length reached at horizon_ramp_epochs."},
    )
    horizon_ramp_epochs: int = field(
        default=200,
        metadata={"help": "Number of epochs over which max_episode_length ramps linearly."},
    )
    edge_case_collector_config: Optional[EdgeCaseCollectorConfig] = field(
        default=None,
        metadata={"help": "Optional configuration for online 3-level EdgeCaseCollector."},
    )

    def compute_curriculum_episode_length(self, current_epoch: int) -> int:
        """Compute curriculum episode length at ``current_epoch`` from this config."""
        return compute_curriculum_episode_length(
            current_epoch=current_epoch,
            initial_max_episode_length=self.initial_max_episode_length,
            target_max_episode_length=self.target_max_episode_length,
            horizon_ramp_epochs=self.horizon_ramp_epochs,
            horizon_curriculum_enabled=self.horizon_curriculum_enabled,
        )


class HierarchicalPPO(PPO):
    """Hierarchical PPO agent with Critic Warmup, Horizon Curriculum, and EdgeCaseCollector hook."""

    config: HierarchicalPPOAgentConfig

    def __init__(
        self,
        fabric: Fabric,
        env: BaseEnv,
        config: HierarchicalPPOAgentConfig,
        root_dir: Optional[Path] = None,
    ):
        super().__init__(fabric, env, config, root_dir)
        self._last_rollout_step_cache: Dict[str, Tensor] = {}

        if (
            self.config.edge_case_collector_config is not None
            and self.config.edge_case_collector_config.enabled
        ):
            self.edge_case_collector: Optional[EdgeCaseCollector] = EdgeCaseCollector(
                config=self.config.edge_case_collector_config,
                num_envs=self.num_envs,
                device=self.device,
            )
        else:
            self.edge_case_collector = None

        # Initialize curriculum episode length at epoch 0 if enabled
        if self.config.horizon_curriculum_enabled:
            self.env.max_episode_length = self.compute_curriculum_episode_length(
                self.current_epoch
            )

    def is_critic_warmup_active(self, epoch: Optional[int] = None) -> bool:
        """Return True if ``epoch`` is within ``critic_warmup_epochs``."""
        ep = self.current_epoch if epoch is None else int(epoch)
        return ep < int(self.config.critic_warmup_epochs)

    def compute_curriculum_episode_length(self, epoch: Optional[int] = None) -> int:
        """Compute the target ``max_episode_length`` for ``epoch`` (defaults to ``self.current_epoch``)."""
        ep = self.current_epoch if epoch is None else int(epoch)
        return compute_curriculum_episode_length(
            current_epoch=ep,
            initial_max_episode_length=self.config.initial_max_episode_length,
            target_max_episode_length=self.config.target_max_episode_length,
            horizon_ramp_epochs=self.config.horizon_ramp_epochs,
            horizon_curriculum_enabled=self.config.horizon_curriculum_enabled,
        )

    def update_horizon_curriculum(self, epoch: Optional[int] = None) -> int:
        """Update ``self.env.max_episode_length`` for the given epoch and return it."""
        if self.config.horizon_curriculum_enabled:
            new_len = self.compute_curriculum_episode_length(epoch=epoch)
            self.env.max_episode_length = new_len
        return int(self.env.max_episode_length)

    def pre_collect_step(self, step: int) -> None:
        """At step 0 of each epoch rollout, update ``self.env.max_episode_length`` from curriculum."""
        super().pre_collect_step(step)
        if step == 0:
            self.update_horizon_curriculum(self.current_epoch)

    def collect_rollout_step(self, obs_td: TensorDict, step: int) -> TensorDict:
        """Collect rollout step and cache hierarchical actor diagnostics for ``EdgeCaseCollector``."""
        output_td = super().collect_rollout_step(obs_td, step)
        if self.edge_case_collector is not None:
            cache: Dict[str, Tensor] = {}
            for key in (
                "minibrain_features",
                "main_brain_delta_q",
                "minibrain_delta_u",
            ):
                if key in output_td:
                    cache[key] = output_td[key].detach()
                elif key in obs_td:
                    cache[key] = obs_td[key].detach()
            self._last_rollout_step_cache = cache
        return output_td

    def record_rollout_step(
        self,
        next_obs_td: TensorDict,
        actions,
        rewards,
        dones,
        terminated,
        done_indices,
        extras,
        step,
    ):
        """Record rollout step and forward telemetry to ``self.edge_case_collector`` if attached."""
        super().record_rollout_step(
            next_obs_td,
            actions,
            rewards,
            dones,
            terminated,
            done_indices,
            extras,
            step,
        )

        if self.edge_case_collector is not None:
            mb_feats = self._last_rollout_step_cache.get("minibrain_features", None)
            main_delta_q = self._last_rollout_step_cache.get("main_brain_delta_q", None)
            mini_delta_u = self._last_rollout_step_cache.get("minibrain_delta_u", None)

            dof_pos = None
            dof_vel = None
            q_des = None
            dq_des = None
            s_axis = None
            tau_grav = None
            ang_mom = None
            vel_err = None

            if mb_feats is not None:
                num_joints = self.edge_case_collector.cfg.num_joints
                feat_3d = mb_feats.reshape(self.num_envs, num_joints, -1)
                q_curr_3d = feat_3d[..., FEAT_Q_SLICE]
                dq_curr_3d = feat_3d[..., FEAT_DQ_SLICE]
                delta_q_3d = feat_3d[..., FEAT_DELTA_Q_SLICE]
                delta_dq_3d = feat_3d[..., FEAT_DELTA_DQ_SLICE]

                dof_pos = q_curr_3d.reshape(self.num_envs, -1)
                dof_vel = dq_curr_3d.reshape(self.num_envs, -1)
                q_des = (q_curr_3d + delta_q_3d).reshape(self.num_envs, -1)
                dq_des = (dq_curr_3d + delta_dq_3d).reshape(self.num_envs, -1)
                tau_grav = feat_3d[..., FEAT_TAU_GRAV_SLICE]
                ang_mom = feat_3d[..., FEAT_ANG_MOMENTUM_SLICE]
                vel_err = feat_3d[..., FEAT_VEL_PRED_ERROR_SLICE]
                s_axis = feat_3d[..., FEAT_S_AXIS_SLICE]

            root_pos = None
            root_rot = None
            root_vel = None
            root_ang_vel = None
            contact_forces = None
            motion_ids = None
            motion_times = None

            if self.env is not None and hasattr(self.env, "simulator"):
                try:
                    robot_state = self.env.simulator.get_robot_state()
                    root_pos = robot_state.root_pos
                    root_rot = robot_state.root_rot
                    root_vel = robot_state.root_vel
                    root_ang_vel = robot_state.root_ang_vel
                    if robot_state.rigid_body_contact_forces is not None:
                        contact_forces = torch.norm(
                            robot_state.rigid_body_contact_forces, dim=-1
                        )
                except Exception:
                    pass
                motion_mgr = getattr(self.env, "motion_manager", None)
                if motion_mgr is not None:
                    motion_ids = getattr(motion_mgr, "motion_ids", None)
                    motion_times = getattr(motion_mgr, "motion_times", None)

            self.edge_case_collector.record_step(
                root_pos=root_pos,
                root_rot=root_rot,
                root_vel=root_vel,
                root_ang_vel=root_ang_vel,
                dof_pos=dof_pos,
                dof_vel=dof_vel,
                q_des=q_des,
                dq_des=dq_des,
                main_brain_delta_q=main_delta_q,
                minibrain_delta_u=mini_delta_u,
                s_axis=s_axis,
                tau_grav_local=tau_grav,
                ang_momentum_local=ang_mom,
                vel_pred_error=vel_err,
                contact_force_magnitudes=contact_forces,
                motion_ids=motion_ids,
                motion_times=motion_times,
                terminated=terminated,
            )
            if done_indices is not None and done_indices.numel() > 0:
                self.edge_case_collector.reset_envs(done_indices)

    def perform_optimization_step(self, batch_dict, batch_idx) -> Dict:
        """Perform one PPO minibatch step, skipping actor updates during critic warmup."""
        warmup_active = self.is_critic_warmup_active(self.current_epoch)
        if warmup_active:
            self._skip_actor_for_epoch = True
            iter_log_dict: Dict[str, Tensor] = {
                "info/critic_warmup_active": torch.tensor(1.0, device=self.device),
                "actor/update_skipped": torch.tensor(1.0, device=self.device),
            }
            critic_loss, critic_loss_dict = self.critic_step(batch_dict)
            iter_log_dict.update(critic_loss_dict)
            critic_grad_clip_dict = self._step_optimizer(
                loss=critic_loss,
                model=self.critic,
                optimizer=self.critic_optimizer,
                model_name="critic",
            )
            iter_log_dict.update(critic_grad_clip_dict)
            return iter_log_dict

        iter_log_dict = super().perform_optimization_step(batch_dict, batch_idx)
        iter_log_dict["info/critic_warmup_active"] = torch.tensor(
            0.0, device=self.device
        )
        return iter_log_dict

    def optimize_model(self) -> Dict:
        """Run epoch optimization and attach horizon curriculum & critic warmup metrics."""
        max_ep_len = self.update_horizon_curriculum(self.current_epoch)
        warmup_active = self.is_critic_warmup_active(self.current_epoch)

        training_log_dict = super().optimize_model()
        training_log_dict["info/critic_warmup_active"] = torch.tensor(
            1.0 if warmup_active else 0.0, device=self.device
        )
        training_log_dict["info/max_episode_length"] = torch.tensor(
            float(max_ep_len), device=self.device
        )
        if self.edge_case_collector is not None:
            summary = self.edge_case_collector.get_summary()
            counts = summary.get("counts", {})
            for lvl in (1, 2, 3):
                c = counts.get(lvl, counts.get(str(lvl), 0))
                training_log_dict[f"edge_cases/level{lvl}_count"] = torch.tensor(
                    float(c), device=self.device
                )
        return training_log_dict

    def get_state_dict(self, state_dict):
        """Save PPO state dict and persist collected edge cases to ``root_dir / 'edge_cases'``."""
        state_dict = super().get_state_dict(state_dict)
        if (
            self.edge_case_collector is not None
            and getattr(self, "root_dir", None) is not None
            and getattr(self.fabric, "global_rank", 0) == 0
        ):
            try:
                self.edge_case_collector.save_all(Path(self.root_dir) / "edge_cases")
            except Exception as exc:
                log.warning("Failed to save edge cases to disk: %s", exc)
        return state_dict


__all__ = [
    "HierarchicalPPOActorConfig",
    "HierarchicalPPOActor",
    "HierarchicalPPOAgentConfig",
    "HierarchicalPPO",
    "compute_curriculum_episode_length",
]
