# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Component 4: HierarchicalPPOActor, HierarchicalPPO, and SMPL config."""

import argparse
from pathlib import Path

import torch
from torch import nn
from tensordict import TensorDict

from protomotions.agents.common.common import weight_init, weight_init_trainable
from protomotions.agents.common.config import MLPLayerConfig, MLPWithConcatConfig
from protomotions.agents.common.minibrain import (
    FEAT_DELTA_Q_SLICE,
    FEAT_Q_SLICE,
    FEAT_S_AXIS_SLICE,
    MINIBRAIN_FEATURE_DIM,
    SharedJointMiniBrain,
    SharedJointMiniBrainConfig,
)
from protomotions.agents.ppo.config import PPOModelConfig
from protomotions.agents.ppo.hierarchical_model import (
    HierarchicalPPO,
    HierarchicalPPOActor,
    HierarchicalPPOActorConfig,
    HierarchicalPPOAgentConfig,
    compute_curriculum_episode_length,
)
from protomotions.agents.ppo.model import PPOModel
from protomotions.envs.action import passthrough_pd_action
from protomotions.robot_configs.smpl import SmplRobotConfig


def _make_actor_config(
    num_joints: int = 23,
    dofs_per_joint: int = 3,
    expose_minibrain_base_to_main: bool = True,
    freeze_minibrain: bool = True,
    minibrain_checkpoint: str = None,
) -> HierarchicalPPOActorConfig:
    num_dofs = num_joints * dofs_per_joint
    in_keys = ["max_coords_obs", "minibrain_features"]
    mu_in_keys = (
        ["max_coords_obs", "minibrain_base_cmd"]
        if expose_minibrain_base_to_main
        else ["max_coords_obs"]
    )
    return HierarchicalPPOActorConfig(
        num_out=num_dofs,
        actor_logstd=-2.9,
        in_keys=in_keys,
        mu_key="actor_trunk_out",
        minibrain_config=SharedJointMiniBrainConfig(
            num_joints=num_joints,
            dofs_per_joint=dofs_per_joint,
            feature_dim=MINIBRAIN_FEATURE_DIM,
        ),
        minibrain_checkpoint=minibrain_checkpoint,
        freeze_minibrain=freeze_minibrain,
        minibrain_features_key="minibrain_features",
        residual_pose_scale=0.5,
        expose_minibrain_base_to_main=expose_minibrain_base_to_main,
        mu_model=MLPWithConcatConfig(
            in_keys=mu_in_keys,
            normalize_obs=True,
            norm_clamp_value=5.0,
            out_keys=["actor_trunk_out"],
            num_out=num_dofs,
            layers=[MLPLayerConfig(units=64, activation="relu") for _ in range(2)],
        ),
    )


def _make_synthetic_minibrain_features(
    batch_size: int = 8,
    num_joints: int = 23,
    flat: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create synthetic [E, J, 34] (or flat [E, J*34]) features along with (q_current, q_ref)."""
    torch.manual_seed(42)
    feats = torch.randn(batch_size, num_joints, MINIBRAIN_FEATURE_DIM) * 0.1
    q_current = torch.randn(batch_size, num_joints, 3) * 0.25
    ref_delta_q = torch.randn(batch_size, num_joints, 3) * 0.15
    q_ref = q_current + ref_delta_q

    feats[..., FEAT_Q_SLICE] = q_current
    feats[..., FEAT_DELTA_Q_SLICE] = ref_delta_q
    feats[..., FEAT_S_AXIS_SLICE] = 1.0  # All axes active

    if flat:
        return feats.reshape(batch_size, num_joints * MINIBRAIN_FEATURE_DIM), q_current, q_ref
    return feats, q_current, q_ref


def test_hierarchical_ppo_actor_forward_shapes():
    """Verify HierarchicalPPOActor forward pass shapes for both flat and 3D minibrain_features."""
    batch_size = 8
    num_joints = 23
    num_dofs = num_joints * 3

    actor_cfg = _make_actor_config(
        num_joints=num_joints,
        expose_minibrain_base_to_main=True,
    )
    actor = HierarchicalPPOActor(actor_cfg)

    for flat in (True, False):
        mb_feats, _, _ = _make_synthetic_minibrain_features(
            batch_size=batch_size, num_joints=num_joints, flat=flat
        )
        td = TensorDict(
            {
                "max_coords_obs": torch.randn(batch_size, 128),
                "minibrain_features": mb_feats,
            },
            batch_size=[batch_size],
        )

        out_td = actor(td, log_internals=True)

        assert out_td["action"].shape == (batch_size, num_dofs)
        assert out_td["mean_action"].shape == (batch_size, num_dofs)
        assert out_td["neglogp"].shape == (batch_size,)
        assert out_td["main_brain_delta_q"].shape == (batch_size, num_dofs)
        assert out_td["minibrain_delta_u"].shape == (batch_size, num_dofs)
        assert out_td["minibrain_base_cmd"].shape == (batch_size, num_dofs)
        assert out_td["minibrain_base_delta_u"].shape == (batch_size, num_dofs)
        assert torch.all(torch.isfinite(out_td["action"]))
        assert torch.all(torch.isfinite(out_td["mean_action"]))
        assert torch.all(torch.isfinite(out_td["neglogp"]))


def test_hierarchical_ppo_actor_step0_zero_init_property():
    """If Main Brain mu_model outputs 0 and Mini-Brain action_head is zero-init, mean_action == q_ref."""
    batch_size = 6
    num_joints = 23
    num_dofs = num_joints * 3

    actor_cfg = _make_actor_config(num_joints=num_joints)
    actor = HierarchicalPPOActor(actor_cfg)

    mb_feats, _, q_ref = _make_synthetic_minibrain_features(
        batch_size=batch_size, num_joints=num_joints, flat=True
    )
    td = TensorDict(
        {
            "max_coords_obs": torch.randn(batch_size, 128),
            "minibrain_features": mb_feats,
        },
        batch_size=[batch_size],
    )

    # Materialize lazy layers in mu_model first, then zero out the final Linear layer of mu_model
    _ = actor(td)
    last_linear = actor.mu.mlp[-1]
    assert isinstance(last_linear, nn.Linear)
    nn.init.zeros_(last_linear.weight)
    nn.init.zeros_(last_linear.bias)

    # Ensure Mini-Brain action_head is zero-initialized
    actor.minibrain.reset_action_head_to_zero()

    out_td = actor(td)
    q_ref_flat = q_ref.reshape(batch_size, num_dofs)

    assert torch.allclose(out_td["main_brain_delta_q"], torch.zeros_like(q_ref_flat), atol=1e-6)
    assert torch.allclose(out_td["minibrain_delta_u"], torch.zeros_like(q_ref_flat), atol=1e-6)
    assert torch.allclose(out_td["minibrain_base_cmd"], torch.zeros_like(q_ref_flat), atol=1e-6)
    assert torch.allclose(out_td["mean_action"], q_ref_flat, atol=1e-6)


def test_weight_init_trainable_preserves_minibrain_weights(tmp_path: Path):
    """model.apply(weight_init_trainable) must NOT clobber frozen Mini-Brain weights or zero-init action_head."""
    batch_size = 4
    num_joints = 23
    num_dofs = num_joints * 3

    # Create a pretrained Mini-Brain with known non-zero action_head & trunk weights and save checkpoint
    pretrained_mb = SharedJointMiniBrain(
        SharedJointMiniBrainConfig(num_joints=num_joints, dofs_per_joint=3)
    )
    with torch.no_grad():
        pretrained_mb.action_head.weight.fill_(0.1234)
        pretrained_mb.action_head.bias.fill_(-0.0567)
    ckpt_path = tmp_path / "minibrain_pretrained.pt"
    pretrained_mb.save_checkpoint(ckpt_path)

    expected_action_w = pretrained_mb.action_head.weight.detach().clone()
    expected_action_b = pretrained_mb.action_head.bias.detach().clone()
    expected_trunk_w = pretrained_mb.trunk[0].weight.detach().clone()

    actor_cfg = _make_actor_config(
        num_joints=num_joints,
        freeze_minibrain=True,
        minibrain_checkpoint=str(ckpt_path),
    )
    critic_cfg = MLPWithConcatConfig(
        in_keys=["max_coords_obs"],
        out_keys=["value"],
        normalize_obs=True,
        num_out=1,
        layers=[MLPLayerConfig(units=64, activation="relu")],
    )
    model_cfg = PPOModelConfig(
        in_keys=["max_coords_obs", "minibrain_features"],
        out_keys=["action", "mean_action", "neglogp", "value"],
        actor=actor_cfg,
        critic=critic_cfg,
    )
    model = PPOModel(model_cfg)

    # Materialize lazy modules so mu_model has concrete nn.Linear parameters
    mb_feats, _, _ = _make_synthetic_minibrain_features(batch_size=batch_size, num_joints=num_joints)
    td = TensorDict(
        {
            "max_coords_obs": torch.randn(batch_size, 128),
            "minibrain_features": mb_feats,
        },
        batch_size=[batch_size],
    )
    model.materialize(td)

    # Apply weight_init_trainable and weight_init(orthogonal=True) as PPO.create_model() does
    model.apply(weight_init_trainable)
    model.apply(lambda m: weight_init(m, orthogonal=True))

    mb = model._actor.minibrain
    assert torch.allclose(mb.action_head.weight, expected_action_w)
    assert torch.allclose(mb.action_head.bias, expected_action_b)
    assert torch.allclose(mb.trunk[0].weight, expected_trunk_w)
    assert all(not p.requires_grad for p in mb.parameters())

    # Also verify fresh (non-checkpoint) zero-init action_head is preserved under weight_init_trainable
    fresh_actor = HierarchicalPPOActor(_make_actor_config(num_joints=num_joints, freeze_minibrain=False))
    fresh_actor(td)
    fresh_actor.apply(weight_init_trainable)
    assert torch.count_nonzero(fresh_actor.minibrain.action_head.weight) == 0
    assert torch.count_nonzero(fresh_actor.minibrain.action_head.bias) == 0


def test_critic_warmup_and_horizon_curriculum_schedule():
    """Verify compute_curriculum_episode_length and HierarchicalPPO critic warmup schedule."""
    cfg = HierarchicalPPOAgentConfig(
        critic_warmup_epochs=10,
        horizon_curriculum_enabled=True,
        initial_max_episode_length=32,
        target_max_episode_length=1000,
        horizon_ramp_epochs=200,
    )

    # Test curriculum schedule at start, midpoint, and after ramp completion
    assert compute_curriculum_episode_length(0, 32, 1000, 200, True) == 32
    assert cfg.compute_curriculum_episode_length(0) == 32
    assert cfg.compute_curriculum_episode_length(100) == 516
    assert cfg.compute_curriculum_episode_length(200) == 1000
    assert cfg.compute_curriculum_episode_length(350) == 1000

    # Monotonicity across all epochs 0..250
    lengths = [cfg.compute_curriculum_episode_length(ep) for ep in range(251)]
    assert lengths[0] == 32
    assert lengths[-1] == 1000
    assert all(b >= a for a, b in zip(lengths, lengths[1:]))

    # Disabled curriculum returns target_max_episode_length immediately
    assert compute_curriculum_episode_length(0, 32, 1000, 200, False) == 1000

    # Test critic warmup schedule helper on HierarchicalPPO instance without spinning up a simulator
    dummy_agent = object.__new__(HierarchicalPPO)
    dummy_agent.config = cfg
    dummy_agent.current_epoch = 0
    assert dummy_agent.is_critic_warmup_active() is True
    assert dummy_agent.is_critic_warmup_active(9) is True
    assert dummy_agent.is_critic_warmup_active(10) is False
    assert dummy_agent.is_critic_warmup_active(50) is False


def test_smpl_hierarchical_mlp_experiment_config_on_cpu():
    """Verify examples/experiments/mimic/smpl_hierarchical_mlp.py config construction with SmplRobotConfig on CPU."""
    from examples.experiments.mimic.smpl_hierarchical_mlp import (
        agent_config,
        env_config,
    )

    robot_cfg = SmplRobotConfig()
    args = argparse.Namespace(
        batch_size=256,
        training_max_steps=100_000,
        motion_file="dummy.pt",
    )

    from protomotions.envs.action import normalized_pd_fixed_gains_action

    e_cfg = env_config(robot_cfg, args)
    assert e_cfg.action_config["fn"] is normalized_pd_fixed_gains_action
    assert e_cfg.action_config["action_transform"] == "clamp"
    expected_obs_keys = {
        "max_coords_obs",
        "previous_actions",
        "mimic_target_poses",
        "collision_primitives",
        "stamina_obs",
        "joint_load_obs",
        "minibrain_features",
    }
    assert expected_obs_keys.issubset(set(e_cfg.observation_components.keys()))

    a_cfg = agent_config(robot_cfg, e_cfg, args)
    assert isinstance(a_cfg, HierarchicalPPOAgentConfig)
    assert isinstance(a_cfg.model.actor, HierarchicalPPOActorConfig)
    assert a_cfg.critic_warmup_epochs == 4
    assert a_cfg.horizon_curriculum_enabled is True
    assert a_cfg.model.actor.num_out == 69
    assert a_cfg.model.actor.convert_to_normalized_action_space is True
    assert a_cfg.model.actor.residual_pose_scale == 1.8

    # Instantiate full PPOModel from config and run CPU forward pass
    model = PPOModel(a_cfg.model)
    batch_size = 4
    mb_feats, _, _ = _make_synthetic_minibrain_features(
        batch_size=batch_size, num_joints=23, flat=True
    )
    td = TensorDict(
        {
            "max_coords_obs": torch.randn(batch_size, 358),
            "mimic_target_poses": torch.randn(batch_size, 645),
            "previous_actions": torch.randn(batch_size, 69),
            "collision_primitives": torch.randn(batch_size, 6 * 17),
            "stamina_obs": torch.ones(batch_size, 23),
            "joint_load_obs": torch.randn(batch_size, 23 * 8),
            "minibrain_features": mb_feats,
        },
        batch_size=[batch_size],
    )
    out_td = model(td)
    assert out_td["action"].shape == (batch_size, 69)
    assert out_td["mean_action"].shape == (batch_size, 69)
    assert out_td["neglogp"].shape == (batch_size,)
    assert out_td["value"].shape == (batch_size, 1)
