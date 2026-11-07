# SPDX-License-Identifier: Apache-2.0
"""SMPL Unity bake interchange using ProtoMotions' native motion state.

Bakes use COMMON coordinates (Z up, XYZW), not Unity transform coordinates.
Export preserves authored velocities, including their sample alignment. Missing
motion state is an error; pose-only imports can explicitly use complete_smpl_motion
to calculate it with the same pose_lib functions as FBX retargeting.
"""
import json
import math
from pathlib import Path

import torch

from data.smpl.smpl_joint_names import SMPL_MUJOCO_NAMES
from protomotions.simulator.base_simulator.simulator_state import RobotState, StateConversion

FIELDS = {
    'positions': ('rigid_body_pos', (24, 3)),
    'rotations': ('rigid_body_rot', (24, 4)),
    'velocities': ('rigid_body_vel', (24, 3)),
    'angular_velocities': ('rigid_body_ang_vel', (24, 3)),
    'dof_pos': ('dof_pos', (69,)),
    'dof_vel': ('dof_vel', (69,)),
}


def _validate(state, required=tuple(FIELDS)):
    if state.state_conversion != StateConversion.COMMON:
        raise ValueError('Expected a motion in COMMON coordinates')
    fps = float(state.fps or 0)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Motion fps must be finite and positive')
    pos = state.rigid_body_pos
    if pos is None or pos.ndim != 3 or len(pos) == 0:
        raise ValueError('Expected nonempty SMPL positions [T,24,3]')
    count = len(pos)
    for key, (attr, shape) in FIELDS.items():
        value = getattr(state, attr)
        if value is None:
            if key in required:
                raise ValueError(f'Motion is missing {attr}; regenerate it with ProtoMotions retargeting')
            continue
        if tuple(value.shape) != (count, *shape) or not torch.isfinite(value).all():
            raise ValueError(f'Expected finite {attr} [{count},{",".join(map(str, shape))}]')
    if torch.any(state.rigid_body_rot.norm(dim=-1) < 1e-6):
        raise ValueError('Motion contains zero quaternions')
    return count, fps


def motion_to_baked(state, clip_name, *, center_xy=True):
    """Serialize all six state arrays without resampling or recomputing velocities."""
    count, fps = _validate(state)
    arrays = {}
    for key, (attr, _) in FIELDS.items():
        values = getattr(state, attr).detach().cpu()
        if key == 'positions' and center_xy:
            origin = values[0, 0].clone()
            origin[2] = 0
            values = values - origin
        arrays[key] = values.tolist()
    return {
        'metadata': {
            'clip_name': clip_name,
            'sample_rate': fps,
            'frame_count': count,
            'motion_length': round((count - 1) / fps, 6),
            'bone_names': list(SMPL_MUJOCO_NAMES),
            'bone_mapping': {},
            'joint_state_source': 'ProtoMotions motion state; authored DOF positions and velocities preserved',
        },
        'frames': [dict(time=round(i / fps, 6), **{k: v[i] for k, v in arrays.items()})
                   for i in range(count)],
    }


def complete_smpl_motion(state):
    """Fill missing state using pose_lib; never overwrite supplied velocities/DOFs."""
    _validate(state, required=('positions', 'rotations'))
    from protomotions.components.pose_lib import (
        extract_kinematic_info, compute_joint_rot_mats_from_global_mats,
        extract_qpos_from_transforms, compute_angular_velocity, compute_cartesian_velocity,
    )
    from protomotions.utils.rotations import quaternion_to_matrix, matrix_to_quaternion
    mjcf = Path(__file__).resolve().parents[1] / 'data/assets/mjcf/smpl_humanoid.xml'
    info = extract_kinematic_info(str(mjcf))
    world = quaternion_to_matrix(state.rigid_body_rot, w_last=True)
    local = compute_joint_rot_mats_from_global_mats(info, world)
    if state.local_rigid_body_rot is None:
        state.local_rigid_body_rot = matrix_to_quaternion(local, w_last=True)
    if state.dof_pos is None:
        state.dof_pos = extract_qpos_from_transforms(
            info, state.rigid_body_pos[:, 0], local,
            multi_dof_decomposition_method='exp_map')[:, 7:]
    if state.dof_vel is None:
        state.dof_vel = compute_angular_velocity(local[:, 1:], state.fps).reshape(-1, 69)
    if state.rigid_body_vel is None:
        state.rigid_body_vel = compute_cartesian_velocity(state.rigid_body_pos, state.fps)
    if state.rigid_body_ang_vel is None:
        state.rigid_body_ang_vel = compute_angular_velocity(world, state.fps)
    if state.rigid_body_contacts is None:
        from data.scripts.contact_detection import compute_contact_labels_from_pos_and_vel
        state.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
            positions=state.rigid_body_pos, velocity=state.rigid_body_vel,
            vel_thres=0.15, height_thresh=0.1).to(torch.bool)
    _validate(state)
    return state


def baked_to_motion(data):
    """Read a bake, preserving explicit state; complete pose-only imports via pose_lib."""
    meta, frames = data['metadata'], data['frames']
    if meta['bone_names'] != list(SMPL_MUJOCO_NAMES):
        raise ValueError('Bake must use the 24-body SMPL MuJoCo order')
    if not frames or meta['frame_count'] != len(frames):
        raise ValueError('Invalid baked frame count')
    state = RobotState(state_conversion=StateConversion.COMMON, fps=float(meta['sample_rate']))
    if not math.isfinite(state.fps) or state.fps <= 0:
        raise ValueError('Bake sample rate must be finite and positive')
    for i, frame in enumerate(frames):
        if not math.isfinite(frame['time']) or abs(frame['time'] - i / state.fps) > 1e-5:
            raise ValueError('Bake timestamps must follow its sample rate')
    for key, (attr, _) in FIELDS.items():
        present = [f.get(key) is not None for f in frames]
        if any(present) and not all(present):
            raise ValueError(f'Incomplete baked {key}')
        if all(present):
            setattr(state, attr, torch.tensor([f[key] for f in frames], dtype=torch.float32))
    return complete_smpl_motion(state)


def write_baked(data, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
