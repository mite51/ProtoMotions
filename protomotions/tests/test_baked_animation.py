"""The Unity export must retain source velocities, not differentiate them again."""
import copy
import json

import pytest
import torch

from protomotions.simulator.base_simulator.simulator_state import RobotState, StateConversion
from protomotions.utils.baked_animation import (
    FIELDS, motion_to_baked, baked_to_motion, complete_smpl_motion,
)


def sample():
    state = RobotState(state_conversion=StateConversion.COMMON, fps=24.0)
    for index, (_, (name, shape)) in enumerate(FIELDS.items()):
        setattr(state, name, torch.arange(3 * int(torch.tensor(shape).prod()), dtype=torch.float32).reshape(3, *shape) / 100 + index)
    state.rigid_body_rot.zero_()
    state.rigid_body_rot[..., 3] = 1
    return state


def test_export_preserves_authored_state_and_height():
    state = sample()
    original = copy.deepcopy(state)
    data = motion_to_baked(state, 'test')
    assert data['metadata']['sample_rate'] == 24
    assert data['frames'][-1]['time'] == pytest.approx(2 / 24, abs=1e-6)
    for key, (name, _) in FIELDS.items():
        actual = torch.tensor([f[key] for f in data['frames']])
        expected = getattr(original, name).clone()
        if key == 'positions':
            expected[..., :2] -= original.rigid_body_pos[0, 0, :2]
        assert torch.equal(actual, expected), key
        assert torch.equal(getattr(state, name), getattr(original, name)), name


@pytest.mark.parametrize('field', ['dof_vel', 'dof_pos', 'rigid_body_ang_vel'])
def test_export_rejects_missing_state(field):
    state = sample()
    setattr(state, field, None)
    with pytest.raises(ValueError, match=field):
        motion_to_baked(state, 'incomplete')


def test_export_rejects_nonfinite_and_wrong_shapes():
    state = sample()
    state.dof_vel[0, 0] = float('nan')
    with pytest.raises(ValueError, match='dof_vel'):
        motion_to_baked(state, 'bad')
    state.dof_vel = torch.zeros(3, 68)
    with pytest.raises(ValueError, match='dof_vel'):
        motion_to_baked(state, 'bad')


def test_pose_only_completion_uses_forward_velocity_and_preserves_authored_values():
    state = sample()
    # Root rotates 0, 0.1, 0.4 radians around Z. All bodies share that rotation.
    angle = torch.tensor([0., .1, .4])
    state.rigid_body_rot.zero_()
    state.rigid_body_rot[..., 2] = torch.sin(angle[:, None] / 2)
    state.rigid_body_rot[..., 3] = torch.cos(angle[:, None] / 2)
    state.rigid_body_ang_vel = None
    state.dof_pos = None
    preserved = state.dof_vel.clone()
    complete_smpl_motion(state)
    assert torch.equal(state.dof_vel, preserved)
    torch.testing.assert_close(state.rigid_body_ang_vel[:, 0, 2], torch.tensor([2.4, 7.2, 7.2]), atol=1e-4, rtol=0)


def test_round_trip_preserves_all_six_arrays():
    state = sample()
    restored = baked_to_motion(motion_to_baked(state, 'test', center_xy=False))
    for _, (name, _) in FIELDS.items():
        assert torch.equal(getattr(restored, name), getattr(state, name)), name


def test_cli_directory_and_overwrite(tmp_path):
    from data.scripts.convert_motion import main
    source = tmp_path / 'source/nested'
    source.mkdir(parents=True)
    torch.save(sample().to_dict(), source / 'kick.motion')
    output = tmp_path / 'bakes'
    args = ['motion-to-baked', str(source.parent), str(output)]
    main(args)
    bake = output / 'nested/kick_baked.json'
    assert json.loads(bake.read_text())['metadata']['clip_name'] == 'kick'
    with pytest.raises(SystemExit):
        main(args)
    main(args + ['--overwrite'])


def test_blender_numpy2_extraction_loads_in_training_environment():
    import io
    import pickle
    import numpy as np
    from data.scripts.retarget_fbx_to_smpl import _ExtractionUnpickler
    values = np.array([[1., 2., 3.]], dtype=np.float32)
    # Protocol 2 stores GLOBAL module names as lines (no frame lengths to rewrite).
    raw = pickle.dumps({'world_pos': values}, protocol=2)
    numpy2_raw = raw.replace(b'numpy.core', b'numpy._core')
    restored = _ExtractionUnpickler(io.BytesIO(numpy2_raw)).load()
    np.testing.assert_array_equal(restored['world_pos'], values)
