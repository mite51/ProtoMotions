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
