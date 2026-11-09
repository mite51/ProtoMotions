"""Check frame conventions in the imported FBX orientation controls."""
import numpy as np
import pytest
from data.scripts.retarget_fbx_to_smpl import _orientation_correction, RIG_PROFILES


def test_auto_orientation_and_explicit_world_rotation_compose():
    first = _orientation_correction(np.eye(3), rotation_euler=(35., -20., 75.))
    np.testing.assert_allclose(_orientation_correction(first), np.eye(3))
    np.testing.assert_allclose(_orientation_correction(first, True) @ first, np.eye(3), atol=1e-12)
    corrected = _orientation_correction(first, True, (0., 0., 90.)) @ first
    np.testing.assert_allclose(corrected @ [1., 0., 0.], [0., 1., 0.], atol=1e-12)
    np.testing.assert_allclose(corrected @ [0., 0., 1.], [0., 0., 1.], atol=1e-12)
    with pytest.raises(ValueError):
        _orientation_correction(first, rotation_euler=(0., float('nan'), 0.))


def test_biped_maps_left_and_right_limbs_without_swapping():
    profile = RIG_PROFILES['biped']
    assert profile['L_Hip'] == 'Bip001 L Thigh'
    assert profile['R_Knee'] == 'Bip001 R Calf'
    assert profile['L_Shoulder'] == 'Bip001 L UpperArm'
    assert profile['R_Elbow'] == 'Bip001 R Forearm'


def test_height_anchor_clears_collision_geometry_on_getup_and_global_clips():
    import torch
    from data.scripts.retarget_fbx_to_smpl import (
        _compute_height_anchor_lift,
        compute_collision_min_z,
        parse_mjcf_collision_geoms,
    )

    geoms = parse_mjcf_collision_geoms()
    assert len(geoms) == 24

    # Synthetic 20-frame clip where early frames penetrate z=0 by -0.12m and the standing tail is near ground.
    T, B = 20, 24
    body_pos = torch.zeros(T, B, 3)
    body_pos[:, :, 2] = 0.50
    # Frame 0..4: L_Ankle (idx 3, box bottom is -0.0703m below joint) sits at z = -0.05m (so geom is -0.1203m)
    body_pos[:5, 3, 2] = -0.05
    # Frame 5..9: standing, L_Ankle at z = +0.0723m (so geom bottom is +0.0020m) and L_Toe at +0.020m
    body_pos[5:, 3, 2] = 0.0723
    body_pos[5:, 4, 2] = 0.0200

    body_rot = torch.zeros(T, B, 4)
    body_rot[..., 3] = 1.0  # identity xyzw

    for mode in ("last_frame", "global", "per_frame"):
        lift = _compute_height_anchor_lift(
            body_pos=body_pos,
            body_rot=body_rot,
            mode=mode,
            foot_offset=0.015,
            collision_geoms=geoms,
            geom_clearance=0.002,
        )
        shifted_pos = body_pos.clone()
        shifted_pos[:, :, 2] += lift.unsqueeze(1)
        min_geom = compute_collision_min_z(shifted_pos, body_rot, geoms).min(dim=1).values
        assert torch.all(min_geom >= 0.002 - 1e-5), f"mode={mode} left penetration: {min_geom}"
        # Final standing frame should remain anchored near +0.002m clearance, not lifted into the air
        assert float(min_geom[-1].item()) == pytest.approx(0.002, abs=1e-3)

