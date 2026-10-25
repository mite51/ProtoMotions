# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SharedJointMiniBrain (Component 2)."""

from pathlib import Path

import pytest
import torch
from torch import nn

from protomotions.agents.common.minibrain import (
    FEAT_DELTA_Q_SLICE,
    FEAT_DQ_SLICE,
    FEAT_Q_SLICE,
    FEAT_S_AXIS_SLICE,
    MINIBRAIN_FEATURE_DIM,
    SharedJointMiniBrain,
    SharedJointMiniBrainConfig,
)


def _make_random_joint_features(
    num_envs: int = 8,
    num_joints: int = 23,
    feature_dim: int = MINIBRAIN_FEATURE_DIM,
    seed: int = 42,
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    features = torch.randn(num_envs, num_joints, feature_dim, generator=gen)
    # Default all axes to available (s_axis = 1.0)
    features[..., FEAT_S_AXIS_SLICE] = 1.0
    return features


def test_zero_initialization_identity():
    """At initialization, delta_u == 0 and u_pd == q_des for active axes."""
    cfg = SharedJointMiniBrainConfig(num_joints=23, feature_dim=MINIBRAIN_FEATURE_DIM)
    model = SharedJointMiniBrain(cfg)

    assert getattr(model.action_head, "_is_zero_init_head", False) is True
    assert torch.all(model.action_head.weight == 0.0)
    assert torch.all(model.action_head.bias == 0.0)

    num_envs, num_joints = 6, 23
    features = _make_random_joint_features(num_envs=num_envs, num_joints=num_joints)
    q_des = torch.randn(num_envs, num_joints, 3)
    q_current = torch.randn(num_envs, num_joints, 3)

    out = model(features, q_des=q_des, q_current=q_current)

    assert out["delta_u"].shape == (num_envs, num_joints, 3)
    assert out["u_pd_3d"].shape == (num_envs, num_joints, 3)
    assert out["u_pd"].shape == (num_envs, num_joints * 3)
    assert out["joint_value"].shape == (num_envs, num_joints, 1)

    assert torch.all(out["raw_offset"] == 0.0)
    assert torch.all(out["delta_u"] == 0.0)
    assert torch.allclose(out["u_pd_3d"], q_des, atol=0.0, rtol=0.0)
    assert torch.allclose(out["u_pd"], q_des.reshape(num_envs, -1), atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "axis_mask_pattern",
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
)
def test_unavailable_dof_masking_and_gradients(axis_mask_pattern):
    """Verify strict zero delta_u, pose locking, and zero gradients on masked axes."""
    cfg = SharedJointMiniBrainConfig(num_joints=23, feature_dim=MINIBRAIN_FEATURE_DIM)
    model = SharedJointMiniBrain(cfg)

    # Set non-zero weights on action_head so raw_offset is non-zero everywhere
    with torch.no_grad():
        nn.init.orthogonal_(model.action_head.weight, gain=0.5)
        model.action_head.bias.fill_(0.25)

    num_envs, num_joints = 4, 23
    features = _make_random_joint_features(num_envs=num_envs, num_joints=num_joints)
    s_axis = torch.tensor(axis_mask_pattern, dtype=features.dtype).view(1, 1, 3)
    s_axis = s_axis.expand(num_envs, num_joints, 3).clone()

    # Embed s_axis inside joint_features[..., 26:29] to also test implicit extraction
    features[..., FEAT_S_AXIS_SLICE] = s_axis

    q_des = torch.randn(num_envs, num_joints, 3)
    q_current = torch.randn(num_envs, num_joints, 3)

    out = model(features, q_des=q_des, q_current=q_current)
    out["raw_offset"].retain_grad()
    delta_u = out["delta_u"]
    u_pd_3d = out["u_pd_3d"]

    active_bool = torch.tensor(axis_mask_pattern, dtype=torch.bool)
    inactive_bool = ~active_bool

    # 1. Unavailable axes must have strictly 0.0 delta_u and lock to q_current
    if inactive_bool.any():
        assert torch.all(delta_u[..., inactive_bool] == 0.0)
        assert torch.allclose(
            u_pd_3d[..., inactive_bool], q_current[..., inactive_bool], atol=0.0
        )

    # 2. Available axes must be active (non-zero delta_u) and equal q_des + delta_u
    if active_bool.any():
        assert torch.all(delta_u[..., active_bool].abs() > 0.0)
        assert torch.allclose(
            u_pd_3d[..., active_bool],
            q_des[..., active_bool] + delta_u[..., active_bool],
            atol=1e-6,
        )
        assert torch.all(
            delta_u[..., active_bool].abs() <= cfg.max_residual_offset + 1e-6
        )

    # 3. Verify gradient wrt raw_offset is strictly zero on masked axes and non-zero on active axes
    loss = (u_pd_3d**2).sum()
    loss.backward()
    raw_grad = out["raw_offset"].grad
    assert raw_grad is not None
    if inactive_bool.any():
        assert torch.all(raw_grad[..., inactive_bool] == 0.0)
    if active_bool.any():
        assert torch.all(raw_grad[..., active_bool].abs() > 0.0)

    # 4. Also verify fallback when q_current is None: q_base = q_des * active_mask
    out_no_qcurr = model(features, q_des=q_des, q_current=None, s_axis=s_axis)
    if inactive_bool.any():
        assert torch.all(out_no_qcurr["u_pd_3d"][..., inactive_bool] == 0.0)


def test_batching_permutation_invariance():
    """Permuting environments or joints gives identical per-joint outputs when frozen."""
    cfg = SharedJointMiniBrainConfig(num_joints=23, feature_dim=MINIBRAIN_FEATURE_DIM)
    model = SharedJointMiniBrain(cfg)

    with torch.no_grad():
        nn.init.orthogonal_(model.action_head.weight, gain=0.4)
        model.action_head.bias.uniform_(-0.1, 0.1)

    num_envs, num_joints = 8, 23
    features = _make_random_joint_features(num_envs=num_envs, num_joints=num_joints)
    q_des = torch.randn(num_envs, num_joints, 3)
    q_current = torch.randn(num_envs, num_joints, 3)

    # Warm up running stats on one batch, then freeze
    _ = model(features, q_des=q_des, q_current=q_current)
    model.freeze()

    out_orig = model(features, q_des=q_des, q_current=q_current)
    q_next_orig, dq_next_orig = model.predict_next_joint_state(
        features, out_orig["delta_u"]
    )

    # Permute environments
    perm_e = torch.tensor([5, 2, 7, 0, 4, 1, 6, 3], dtype=torch.long)
    out_perm_e = model(
        features[perm_e], q_des=q_des[perm_e], q_current=q_current[perm_e]
    )
    assert torch.allclose(
        out_perm_e["delta_u"], out_orig["delta_u"][perm_e], atol=1e-6
    )
    assert torch.allclose(
        out_perm_e["u_pd_3d"], out_orig["u_pd_3d"][perm_e], atol=1e-6
    )
    assert torch.allclose(
        out_perm_e["joint_value"], out_orig["joint_value"][perm_e], atol=1e-6
    )

    # Permute joints
    perm_j = torch.randperm(num_joints, generator=torch.Generator().manual_seed(99))
    out_perm_j = model(
        features[:, perm_j],
        q_des=q_des[:, perm_j],
        q_current=q_current[:, perm_j],
    )
    q_next_perm_j, dq_next_perm_j = model.predict_next_joint_state(
        features[:, perm_j], out_perm_j["delta_u"]
    )

    assert torch.allclose(
        out_perm_j["delta_u"], out_orig["delta_u"][:, perm_j], atol=1e-6
    )
    assert torch.allclose(
        out_perm_j["u_pd_3d"], out_orig["u_pd_3d"][:, perm_j], atol=1e-6
    )
    assert torch.allclose(
        out_perm_j["joint_value"], out_orig["joint_value"][:, perm_j], atol=1e-6
    )
    assert torch.allclose(q_next_perm_j, q_next_orig[:, perm_j], atol=1e-6)
    assert torch.allclose(dq_next_perm_j, dq_next_orig[:, perm_j], atol=1e-6)


def test_predict_next_joint_state_differentiable_path():
    """Loss on (q_next_pred - q_des)^2 backpropagates cleanly into action_head and trunk."""
    cfg = SharedJointMiniBrainConfig(num_joints=23, feature_dim=MINIBRAIN_FEATURE_DIM)
    model = SharedJointMiniBrain(cfg)

    num_envs, num_joints = 6, 23
    features = _make_random_joint_features(num_envs=num_envs, num_joints=num_joints)
    # Lock axis 2 on joint 0 to also verify masked dynamics predictions
    features[:, 0, FEAT_S_AXIS_SLICE] = torch.tensor([1.0, 1.0, 0.0])

    q_des = features[..., FEAT_Q_SLICE] + features[..., FEAT_DELTA_Q_SLICE] + 0.2
    dq_des = features[..., FEAT_DQ_SLICE] + 0.1

    policy_params = list(model.trunk.parameters()) + list(
        model.action_head.parameters()
    )
    optimizer = torch.optim.Adam(policy_params, lr=1e-2)

    # Step 1 (from zero-init action_head): breaks symmetry on action_head
    optimizer.zero_grad()
    out0 = model(features, q_des=q_des)
    q_next0, dq_next0 = model.predict_next_joint_state(features.detach(), out0["delta_u"])
    loss0 = ((q_next0 - q_des) ** 2).mean() + 0.1 * ((dq_next0 - dq_des) ** 2).mean()
    loss0.backward()

    # Locked axis 2 on joint 0 must predict exact zero state change
    assert torch.allclose(
        q_next0[:, 0, 2], features[:, 0, FEAT_Q_SLICE.start + 2], atol=0.0
    )
    assert torch.allclose(
        dq_next0[:, 0, 2], features[:, 0, FEAT_DQ_SLICE.start + 2], atol=0.0
    )

    assert model.action_head.weight.grad is not None
    assert model.action_head.weight.grad.abs().sum() > 0.0
    assert model.action_head.bias.grad is not None
    assert model.action_head.bias.grad.abs().sum() > 0.0
    optimizer.step()

    # Step 2: With non-zero action_head weights, gradients flow cleanly into all trunk layers
    optimizer.zero_grad()
    out1 = model(features, q_des=q_des)
    q_next1, dq_next1 = model.predict_next_joint_state(features.detach(), out1["delta_u"])
    loss1 = ((q_next1 - q_des) ** 2).mean() + 0.1 * ((dq_next1 - dq_des) ** 2).mean()
    loss1.backward()

    assert model.action_head.weight.grad.abs().sum() > 0.0
    for module in model.trunk.modules():
        if isinstance(module, nn.Linear):
            assert module.weight.grad is not None
            assert torch.isfinite(module.weight.grad).all()
            assert module.weight.grad.abs().sum() > 0.0

    # Take a few optimization steps and verify tracking loss decreases
    for _ in range(5):
        optimizer.step()
        optimizer.zero_grad()
        out_k = model(features, q_des=q_des)
        q_next_k, dq_next_k = model.predict_next_joint_state(
            features.detach(), out_k["delta_u"]
        )
        loss_k = ((q_next_k - q_des) ** 2).mean() + 0.1 * (
            (dq_next_k - dq_des) ** 2
        ).mean()
        loss_k.backward()

    assert loss_k.item() < loss0.item()


def test_freeze_unfreeze_and_checkpoint_roundtrip(tmp_path: Path):
    """Test freeze/unfreeze behavior and save_checkpoint / load_checkpoint."""
    cfg = SharedJointMiniBrainConfig(
        num_joints=23,
        feature_dim=MINIBRAIN_FEATURE_DIM,
        hidden_dims=[64, 64],
    )
    model = SharedJointMiniBrain(cfg)

    with torch.no_grad():
        nn.init.orthogonal_(model.action_head.weight, gain=0.3)
        model.action_head.bias.fill_(-0.05)

    features = _make_random_joint_features(num_envs=5, num_joints=23, seed=7)
    q_des = torch.randn(5, 23, 3)

    # Unfrozen training pass updates running mean
    model.unfreeze()
    assert model.training is True
    assert model._freeze_running is False
    assert all(p.requires_grad for p in model.parameters())

    _ = model(features, q_des=q_des)
    mean_after_step1 = model.obs_norm.mean.clone()
    assert torch.any(mean_after_step1 != 0.0)

    # Freezing stops parameter grads and freezes running stats
    model.freeze()
    assert model.training is False
    assert model._freeze_running is True
    assert all(not p.requires_grad for p in model.parameters())

    features_shifted = features + 10.0
    _ = model(features_shifted, q_des=q_des)
    assert torch.allclose(model.obs_norm.mean, mean_after_step1, atol=0.0, rtol=0.0)

    # Save checkpoint and reload
    ckpt_file = tmp_path / "minibrain_test.pt"
    model.save_checkpoint(ckpt_file, extra_metadata={"epoch": 12, "mode": "A"})
    assert ckpt_file.exists()

    loaded_frozen = SharedJointMiniBrain.load_checkpoint(
        ckpt_file, device="cpu", freeze=True
    )
    assert loaded_frozen.training is False
    assert loaded_frozen._freeze_running is True
    assert all(not p.requires_grad for p in loaded_frozen.parameters())
    assert loaded_frozen.checkpoint_metadata == {"epoch": 12, "mode": "A"}
    assert loaded_frozen.config.hidden_dims == [64, 64]

    out_ref = model(features, q_des=q_des)
    out_loaded = loaded_frozen(features, q_des=q_des)
    assert torch.allclose(out_ref["u_pd"], out_loaded["u_pd"], atol=1e-6)
    assert torch.allclose(out_ref["delta_u"], out_loaded["delta_u"], atol=1e-6)
    assert torch.allclose(out_ref["joint_value"], out_loaded["joint_value"], atol=1e-6)

    # Unfreezing loaded model re-enables gradients
    loaded_frozen.unfreeze()
    assert loaded_frozen.training is True
    assert loaded_frozen._freeze_running is False
    assert all(p.requires_grad for p in loaded_frozen.parameters())
