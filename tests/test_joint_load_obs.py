# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for analytical subtree load observations and Mini-Brain joint features."""

import math
import torch

from protomotions.robot_configs.smpl import SmplRobotConfig
from protomotions.utils.rotations import (
    quat_from_angle_axis,
    quat_identity,
    quat_mul,
    quat_rotate,
)
from protomotions.envs.obs.joint_load_obs import (
    JOINT_LOAD_FEATURE_DIM,
    MINIBRAIN_DYNAMIC_FEATURE_DIM,
    MINIBRAIN_FEATURE_DIM,
    MINIBRAIN_STATIC_PARAM_DIM,
    SubtreeTopology,
    build_subtree_topology,
    compute_analytical_pd_lead_action,
    compute_analytical_subtree_load,
    compute_joint_load_obs,
    compute_minibrain_joint_features,
)


def _get_smpl_config_and_topology():
    cfg = SmplRobotConfig()
    topo = build_subtree_topology(cfg.kinematic_info, cfg.control)
    return cfg, topo


def test_constants():
    assert JOINT_LOAD_FEATURE_DIM == 8
    assert MINIBRAIN_STATIC_PARAM_DIM == 10
    assert MINIBRAIN_DYNAMIC_FEATURE_DIM == 34
    assert MINIBRAIN_FEATURE_DIM == 34


def test_smpl_subtree_topology():
    cfg, topo = _get_smpl_config_and_topology()
    kin = cfg.kinematic_info

    assert isinstance(topo, SubtreeTopology)
    assert topo.num_joints == 23
    assert topo.num_bodies == 24
    assert topo.joint_body_indices.shape == (23,)
    assert topo.parent_body_indices.shape == (23,)
    assert topo.subtree_mask.shape == (23, 24)
    assert topo.body_masses.shape == (24,)
    assert topo.subtree_masses.shape == (23,)
    assert topo.kp.shape == (23, 3)
    assert topo.kd.shape == (23, 3)
    assert topo.effort_limit.shape == (23, 3)
    assert topo.dof_limits_lower.shape == (23, 3)
    assert topo.dof_limits_upper.shape == (23, 3)

    body_names = kin.body_names
    joint_body_names = [body_names[i] for i in topo.joint_body_indices.tolist()]

    # Verify Pelvis (index 0) is parent of L_Hip, R_Hip, Torso
    pelvis_idx = body_names.index("Pelvis")
    assert pelvis_idx == 0
    for root_child_name in ("L_Hip", "R_Hip", "Torso"):
        j_idx = joint_body_names.index(root_child_name)
        assert int(topo.parent_body_indices[j_idx].item()) == pelvis_idx

    # Verify L_Hip subtree has 4 bodies: L_Hip, L_Knee, L_Ankle, L_Toe
    l_hip_j = joint_body_names.index("L_Hip")
    l_hip_subtree_bodies = [
        body_names[b]
        for b in range(topo.num_bodies)
        if topo.subtree_mask[l_hip_j, b].item() > 0.5
    ]
    assert set(l_hip_subtree_bodies) == {"L_Hip", "L_Knee", "L_Ankle", "L_Toe"}
    assert len(l_hip_subtree_bodies) == 4

    # Verify L_Toe has 1 body: L_Toe
    l_toe_j = joint_body_names.index("L_Toe")
    l_toe_subtree_bodies = [
        body_names[b]
        for b in range(topo.num_bodies)
        if topo.subtree_mask[l_toe_j, b].item() > 0.5
    ]
    assert l_toe_subtree_bodies == ["L_Toe"]

    # Subtree mass of L_Hip must equal sum of masses of L_Hip, L_Knee, L_Ankle, L_Toe
    expected_l_hip_mass = sum(
        topo.body_masses[body_names.index(name)].item()
        for name in ("L_Hip", "L_Knee", "L_Ankle", "L_Toe")
    )
    assert math.isclose(
        topo.subtree_masses[l_hip_j].item(), expected_l_hip_mass, rel_tol=1e-5
    )

    # Verify .to() preserves shapes and values
    topo_f64 = topo.to(device="cpu", dtype=torch.float64)
    assert topo_f64.subtree_mask.dtype == torch.float64
    assert topo_f64.joint_body_indices.dtype == torch.long


def test_analytical_subtree_load_horizontal_vs_vertical_and_zero_vel():
    cfg, topo = _get_smpl_config_and_topology()
    kin = cfg.kinematic_info
    body_names = kin.body_names
    joint_body_names = [body_names[i] for i in topo.joint_body_indices.tolist()]
    l_hip_j = joint_body_names.index("L_Hip")

    num_envs = 2
    num_bodies = topo.num_bodies

    # Env 0: L_Hip limb hanging vertically straight down along -Z from L_Hip at (0, 0, 1)
    # Env 1: L_Hip limb extended horizontally along +X from L_Hip at (0, 0, 1)
    body_pos = torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32)
    body_rot = quat_identity([num_envs, num_bodies], w_last=True)
    body_vel = torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32)
    body_ang_vel = torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32)

    l_chain = ["L_Hip", "L_Knee", "L_Ankle", "L_Toe"]
    distances = [0.0, 0.4, 0.8, 0.95]
    for name, dist in zip(l_chain, distances):
        b_idx = body_names.index(name)
        # Env 0: vertical hanging along -Z (parallel to gravity [0, 0, -9.81])
        body_pos[0, b_idx] = torch.tensor([0.0, 0.0, 1.0 - dist])
        # Env 1: horizontal extended along +X (perpendicular to gravity [0, 0, -9.81])
        body_pos[1, b_idx] = torch.tensor([dist, 0.0, 1.0])

    tau_grav, ang_mom, subtree_mass, eff_inertia = compute_analytical_subtree_load(
        body_pos=body_pos,
        body_rot=body_rot,
        body_vel=body_vel,
        body_ang_vel=body_ang_vel,
        subtree_mask=topo.subtree_mask,
        body_masses=topo.body_masses,
        joint_body_indices=topo.joint_body_indices,
        parent_body_indices=topo.parent_body_indices,
        w_last=True,
    )

    assert tau_grav.shape == (num_envs, 23, 3)
    assert ang_mom.shape == (num_envs, 23, 3)
    assert subtree_mass.shape == (num_envs, 23, 1)
    assert eff_inertia.shape == (num_envs, 23, 1)

    # Extended horizontal limb produces strictly larger ||tau_grav_local|| than vertical hanging limb
    vertical_tau_norm = tau_grav[0, l_hip_j].norm().item()
    horizontal_tau_norm = tau_grav[1, l_hip_j].norm().item()
    assert vertical_tau_norm < 1e-5
    assert horizontal_tau_norm > 10.0
    assert horizontal_tau_norm > vertical_tau_norm

    # Zero velocity produces zero ang_momentum_local
    assert torch.allclose(ang_mom, torch.zeros_like(ang_mom), atol=1e-7)


def test_analytical_subtree_load_frame_invariance():
    _, topo = _get_smpl_config_and_topology()
    torch.manual_seed(42)

    num_envs = 4
    num_bodies = topo.num_bodies

    body_pos = torch.randn((num_envs, num_bodies, 3), dtype=torch.float32)
    rand_axis = torch.randn((num_envs, num_bodies, 3), dtype=torch.float32)
    rand_angle = torch.rand((num_envs, num_bodies), dtype=torch.float32) * math.pi
    body_rot = quat_from_angle_axis(rand_angle, rand_axis, w_last=True)
    body_vel = torch.randn((num_envs, num_bodies, 3), dtype=torch.float32)
    body_ang_vel = torch.randn((num_envs, num_bodies, 3), dtype=torch.float32)
    gravity_vec = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32)

    tau_0, mom_0, mass_0, inertia_0 = compute_analytical_subtree_load(
        body_pos=body_pos,
        body_rot=body_rot,
        body_vel=body_vel,
        body_ang_vel=body_ang_vel,
        subtree_mask=topo.subtree_mask,
        body_masses=topo.body_masses,
        joint_body_indices=topo.joint_body_indices,
        parent_body_indices=topo.parent_body_indices,
        gravity_vec=gravity_vec,
        w_last=True,
    )

    # Rotate the entire character AND gravity vector by a global 3D quaternion per env
    global_axis = torch.tensor(
        [[1.0, 2.0, -0.5], [-0.3, 1.0, 0.8], [0.5, -1.2, 0.7], [0.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    global_angle = torch.tensor([0.7, 1.4, -1.1, 2.2], dtype=torch.float32)
    q_global = quat_from_angle_axis(global_angle, global_axis, w_last=True)  # [E, 4]
    q_global_bodies = q_global.unsqueeze(1).expand(num_envs, num_bodies, 4)

    rot_body_pos = quat_rotate(q_global_bodies, body_pos, w_last=True)
    rot_body_rot = quat_mul(q_global_bodies, body_rot, w_last=True)
    rot_body_vel = quat_rotate(q_global_bodies, body_vel, w_last=True)
    rot_body_ang_vel = quat_rotate(q_global_bodies, body_ang_vel, w_last=True)
    rot_gravity = quat_rotate(
        q_global, gravity_vec.unsqueeze(0).expand(num_envs, 3), w_last=True
    )

    tau_1, mom_1, mass_1, inertia_1 = compute_analytical_subtree_load(
        body_pos=rot_body_pos,
        body_rot=rot_body_rot,
        body_vel=rot_body_vel,
        body_ang_vel=rot_body_ang_vel,
        subtree_mask=topo.subtree_mask,
        body_masses=topo.body_masses,
        joint_body_indices=topo.joint_body_indices,
        parent_body_indices=topo.parent_body_indices,
        gravity_vec=rot_gravity,
        w_last=True,
    )

    assert torch.allclose(tau_0, tau_1, atol=1e-4, rtol=1e-4)
    assert torch.allclose(mom_0, mom_1, atol=1e-4, rtol=1e-4)
    assert torch.allclose(mass_0, mass_1, atol=1e-6)
    assert torch.allclose(inertia_0, inertia_1, atol=1e-4, rtol=1e-4)


def test_minibrain_joint_features_and_locked_dofs():
    _, topo = _get_smpl_config_and_topology()
    torch.manual_seed(7)

    num_envs = 5
    num_joints = topo.num_joints
    num_bodies = topo.num_bodies

    dof_pos = torch.randn((num_envs, num_joints * 3)) * 0.3
    dof_vel = torch.randn((num_envs, num_joints * 3)) * 0.5
    q_des = dof_pos + torch.randn_like(dof_pos) * 0.1
    dq_des = torch.randn_like(dof_vel) * 0.2

    body_pos = torch.randn((num_envs, num_bodies, 3))
    body_rot = quat_identity([num_envs, num_bodies], w_last=True)
    body_vel = torch.randn((num_envs, num_bodies, 3))
    body_ang_vel = torch.randn((num_envs, num_bodies, 3))

    # Construct s_axis with 1 or 2 DOFs zeroed on specific joints
    s_axis = torch.ones((num_envs, num_joints, 3), dtype=torch.float32)
    s_axis[:, 0, 1] = 0.0  # 1 DOF locked on joint 0 (y-axis)
    s_axis[:, 1, 0] = 0.0  # 2 DOFs locked on joint 1 (x and z axes)
    s_axis[:, 1, 2] = 0.0

    feats = compute_minibrain_joint_features(
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        q_des=q_des,
        dq_des=dq_des,
        body_pos=body_pos,
        body_rot=body_rot,
        body_vel=body_vel,
        body_ang_vel=body_ang_vel,
        topology=topo,
        s_axis=s_axis,
    )

    assert feats.shape == (num_envs, 23, MINIBRAIN_FEATURE_DIM)
    assert torch.isfinite(feats).all()

    # s_axis sits at feature slice 26:29
    s_axis_slice = feats[:, :, 26:29]
    assert torch.allclose(s_axis_slice, s_axis)
    assert (s_axis_slice[:, 0, 1] == 0.0).all()
    assert (s_axis_slice[:, 0, 0] == 1.0).all()
    assert (s_axis_slice[:, 1, 0] == 0.0).all()
    assert (s_axis_slice[:, 1, 2] == 0.0).all()
    assert (s_axis_slice[:, 1, 1] == 1.0).all()

    # Also verify flat load obs and analytical PD lead action
    load_obs = compute_joint_load_obs(
        body_pos=body_pos,
        body_rot=body_rot,
        body_vel=body_vel,
        body_ang_vel=body_ang_vel,
        subtree_mask=topo.subtree_mask,
        body_masses=topo.body_masses,
        joint_body_indices=topo.joint_body_indices,
        parent_body_indices=topo.parent_body_indices,
    )
    assert load_obs.shape == (num_envs, 23 * JOINT_LOAD_FEATURE_DIM)
    assert torch.isfinite(load_obs).all()

    pd_action = compute_analytical_pd_lead_action(
        q_j=dof_pos,
        dq_j=dof_vel,
        q_des=q_des,
        dq_des=dq_des,
        kp=topo.kp,
        kd=topo.kd,
        s_axis=s_axis,
    )
    assert pd_action.shape == (num_envs, 69)
    assert torch.isfinite(pd_action).all()

    # Locked axes in pd_action must equal q_des exactly (zero lead offset)
    pd_action_3d = pd_action.reshape(num_envs, 23, 3)
    q_des_3d = q_des.reshape(num_envs, 23, 3)
    assert torch.allclose(pd_action_3d[:, 0, 1], q_des_3d[:, 0, 1])
    assert torch.allclose(pd_action_3d[:, 1, 0], q_des_3d[:, 1, 0])
    assert torch.allclose(pd_action_3d[:, 1, 2], q_des_3d[:, 1, 2])


def test_analytical_gravity_torque_matches_mujoco_rne_exact():
    """Verify compute_analytical_subtree_load with body_com_offset matches MuJoCo RNE qfrc_bias to <1e-4 N*m."""
    import mujoco
    import numpy as np
    from protomotions.utils.rotations import wxyz_to_xyzw

    _, topo = _get_smpl_config_and_topology()
    assert topo.body_com_offset is not None
    assert topo.body_com_offset.shape == (24, 3)

    mj_model = mujoco.MjModel.from_xml_path("protomotions/data/assets/mjcf/smpl_humanoid.xml")
    mj_data = mujoco.MjData(mj_model)

    rng = np.random.default_rng(12345)
    max_err_with_com = 0.0
    max_err_without_com = 0.0

    for _ in range(20):
        mj_data.qpos[:3] = [0.0, 0.0, 1.0]
        quat_rand = rng.standard_normal(4)
        quat_rand /= np.linalg.norm(quat_rand)
        mj_data.qpos[3:7] = quat_rand
        mj_data.qpos[7:] = rng.uniform(-0.6, 0.6, size=(69,))
        mj_data.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

        body_pos = torch.from_numpy(mj_data.xpos[1:].copy()).float().unsqueeze(0)
        body_rot_wxyz = torch.from_numpy(mj_data.xquat[1:].copy()).float().unsqueeze(0)
        body_rot = wxyz_to_xyzw(body_rot_wxyz)
        zero_vel = torch.zeros_like(body_pos)

        # 1. With body_com_offset (true CoM)
        tau_with_com, _, _, _ = compute_analytical_subtree_load(
            body_pos=body_pos,
            body_rot=body_rot,
            body_vel=zero_vel,
            body_ang_vel=zero_vel,
            subtree_mask=topo.subtree_mask,
            body_masses=topo.body_masses,
            joint_body_indices=topo.joint_body_indices,
            parent_body_indices=topo.parent_body_indices,
            w_last=True,
            body_com_offset=topo.body_com_offset,
        )

        # 2. Without body_com_offset (joint pivot origin approximation)
        tau_without_com, _, _, _ = compute_analytical_subtree_load(
            body_pos=body_pos,
            body_rot=body_rot,
            body_vel=zero_vel,
            body_ang_vel=zero_vel,
            subtree_mask=topo.subtree_mask,
            body_masses=topo.body_masses,
            joint_body_indices=topo.joint_body_indices,
            parent_body_indices=topo.parent_body_indices,
            w_last=True,
            body_com_offset=None,
        )

        parent_rot = body_rot[:, topo.parent_body_indices, :]
        tau_world_with = quat_rotate(
            parent_rot.reshape(-1, 4), tau_with_com.reshape(-1, 3), w_last=True
        ).reshape(23, 3)
        tau_world_without = quat_rotate(
            parent_rot.reshape(-1, 4), tau_without_com.reshape(-1, 3), w_last=True
        ).reshape(23, 3)

        for j in range(23):
            for axis_k in range(3):
                dof_idx = 6 + j * 3 + axis_k
                jnt_idx = 1 + j * 3 + axis_k
                axis_world = torch.from_numpy(mj_data.xaxis[jnt_idx].copy()).float()
                mj_grav_torque = float(-mj_data.qfrc_bias[dof_idx])

                pred_with = float(torch.dot(tau_world_with[j], axis_world).item())
                pred_without = float(torch.dot(tau_world_without[j], axis_world).item())

                max_err_with_com = max(max_err_with_com, abs(pred_with - mj_grav_torque))
                max_err_without_com = max(max_err_without_com, abs(pred_without - mj_grav_torque))

    # Without body_com_offset, error is ~20 N*m; with body_com_offset, it matches MuJoCo RNE to < 5e-5 N*m
    assert max_err_without_com > 10.0
    assert max_err_with_com < 5e-5
