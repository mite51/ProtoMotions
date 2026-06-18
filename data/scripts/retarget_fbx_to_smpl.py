# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Retarget Mixamo / Maya HIK -style humanoid FBX animations to the ProtoMotions SMPL rig.

Pipeline overview:
  1. Stage 1 (Blender): :file:`_fbx_extract_blender.py` is invoked through
     ``blender --background`` to load the FBX, sample bone world transforms
     for every animation take, and dump them to a pickle.
  2. Stage 2 (this script): we read the pickle, compute a frame-alignment
     rotation between the source rig and the SMPL MJCF rig, then for each
     SMPL joint with a mapped source bone we apply the rest-pose retargeting
     formula::

         R_smpl(t) = R_align @ R_src(t) @ inv(R_src_rest) @ inv(R_align)

     Unmapped SMPL joints (e.g. middle spine, terminal hand) inherit their
     parent's world rotation, i.e. local rotation = identity. The resulting
     SMPL world rotations are converted to local hinge rotations and fed
     through ``fk_from_transforms_with_velocities`` followed by contact
     detection and height fixing — the same final stages used by
     ``convert_amass_to_proto.py``.

Usage::

    python data/scripts/retarget_fbx_to_smpl.py \
        c:/path/to/KB_Movement.fbx \
        --output-dir data/yaml_files/FightingAnimsetPro/

By default the script will call Blender automatically; pass
``--skip-extract --extracted-pkl path/to.pkl`` to reuse a previous extraction.
"""
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import typer

# Ensure relative-to-script imports work when launched from any CWD.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
for _p in (_PROJECT_ROOT, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from contact_detection import compute_contact_labels_from_pos_and_vel  # noqa: E402

from data.smpl.smpl_joint_names import SMPL_MUJOCO_NAMES  # noqa: E402
from protomotions.components.pose_lib import (  # noqa: E402
    extract_kinematic_info,
    fk_from_transforms_with_velocities,
    extract_qpos_from_transforms,
    compute_angular_velocity,
    compute_forward_kinematics_from_transforms,
    compute_joint_rot_mats_from_global_mats,
)


app = typer.Typer(pretty_exceptions_enable=False)


# ---- Bone mapping --------------------------------------------------------- #

# A "rig profile" maps each SMPL MJCF bone name to the corresponding source
# bone in a particular humanoid naming convention. Different DCC tools/asset
# packs use *similar but conflicting* names (e.g. "LeftLeg" means upper leg in
# Unity-humanoid rigs but the knee in Mixamo), so we register profiles and
# auto-pick the best match per FBX.
RigProfile = Dict[str, str]


# Mixamo / Maya HIK / Kubold animset (2 spine joints).
RIG_PROFILE_MIXAMO: RigProfile = {
    "Pelvis": "Hips",
    "L_Hip": "LeftUpLeg",
    "L_Knee": "LeftLeg",
    "L_Ankle": "LeftFoot",
    "L_Toe": "LeftToeBase",
    "R_Hip": "RightUpLeg",
    "R_Knee": "RightLeg",
    "R_Ankle": "RightFoot",
    "R_Toe": "RightToeBase",
    "Torso": "Spine",
    # No middle-spine counterpart on a 2-spine source rig; left unmapped.
    "Chest": "Spine1",
    "Neck": "Neck",
    "Head": "Head",
    "L_Thorax": "LeftShoulder",
    "L_Shoulder": "LeftArm",
    "L_Elbow": "LeftForeArm",
    "L_Wrist": "LeftHand",
    "R_Thorax": "RightShoulder",
    "R_Shoulder": "RightArm",
    "R_Elbow": "RightForeArm",
    "R_Wrist": "RightHand",
}


# Kimodo rigs: "Spine1, Spine2, Chest" spine (3 joints), 2 neck joints,
# legs use Leg/Shin/Foot rather than UpLeg/Leg/Foot.
RIG_PROFILE_KIMODO: RigProfile = {
    "Pelvis": "Hips",
    "L_Hip": "LeftLeg",
    "L_Knee": "LeftShin",
    "L_Ankle": "LeftFoot",
    "L_Toe": "LeftToeBase",
    "R_Hip": "RightLeg",
    "R_Knee": "RightShin",
    "R_Ankle": "RightFoot",
    "R_Toe": "RightToeBase",
    "Torso": "Spine1",
    "Spine": "Spine2",
    "Chest": "Chest",
    "Neck": "Neck1",
    "Head": "Head",
    "L_Thorax": "LeftShoulder",
    "L_Shoulder": "LeftArm",
    "L_Elbow": "LeftForeArm",
    "L_Wrist": "LeftHand",
    "R_Thorax": "RightShoulder",
    "R_Shoulder": "RightArm",
    "R_Elbow": "RightForeArm",
    "R_Wrist": "RightHand",
}


RIG_PROFILES: Dict[str, RigProfile] = {
    "mixamo": RIG_PROFILE_MIXAMO,
    "kimodo": RIG_PROFILE_KIMODO,
}


# Common Mixamo-style fallback name prefixes (e.g. "mixamorig:LeftArm").
MIXAMO_PREFIX_VARIANTS = ("", "mixamorig:", "mixamorig1:", "mixamorig2:")


def _profile_score(
    profile: RigProfile, source_bone_names_set: set
) -> Tuple[int, RigProfile]:
    """Return (number of bones resolved, resolved profile with prefix-applied names)."""
    resolved: RigProfile = {}
    n = 0
    for smpl_name, src_name in profile.items():
        for prefix in MIXAMO_PREFIX_VARIANTS:
            candidate = prefix + src_name
            if candidate in source_bone_names_set:
                resolved[smpl_name] = candidate
                n += 1
                break
    return n, resolved


def select_rig_profile(
    source_bone_names, profile_override: Optional[str] = None
) -> Tuple[str, RigProfile]:
    """Auto-detect the best matching rig profile, or honour ``profile_override``.

    Returns ``(profile_name, resolved_mapping)`` where ``resolved_mapping``
    contains the actual bone names (with prefixes applied) found in the rig.
    Bones missing from the rig are simply omitted from the resolved mapping.
    """
    name_set = set(source_bone_names)
    if profile_override is not None:
        if profile_override not in RIG_PROFILES:
            raise typer.BadParameter(
                f"Unknown rig profile '{profile_override}'. "
                f"Available: {list(RIG_PROFILES.keys())}"
            )
        score, resolved = _profile_score(RIG_PROFILES[profile_override], name_set)
        return profile_override, resolved

    best_name, best_score, best_resolved = None, -1, {}
    for name, profile in RIG_PROFILES.items():
        score, resolved = _profile_score(profile, name_set)
        if score > best_score:
            best_name, best_score, best_resolved = name, score, resolved
    if best_name is None or best_score == 0:
        raise RuntimeError(
            "No rig profile matched any bones in this FBX. "
            f"Source bones (first 20): {list(source_bone_names)[:20]}"
        )
    return best_name, best_resolved


# ---- Helpers -------------------------------------------------------------- #


def _ortho_normalize(R: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix to the closest rotation matrix (SVD)."""
    U, _, Vt = np.linalg.svd(R)
    R_clean = U @ Vt
    if np.linalg.det(R_clean) < 0:
        # Flip one column to ensure right-handed
        U[:, -1] *= -1
        R_clean = U @ Vt
    return R_clean


def _ortho_normalize_batch(R: np.ndarray) -> np.ndarray:
    """Strip non-rotation components (scale/skew) from a stack of 3x3 matrices.

    Blender's FBX import bakes the armature unit-scale into pose-bone matrices,
    so ``M[:3, :3]`` is typically ``s * R`` for some small ``s``. We just need
    the pure rotation, so SVD-project each matrix onto SO(3).
    """
    if R.ndim == 2:
        return _ortho_normalize(R)
    flat = R.reshape(-1, 3, 3)
    # Numpy batch SVD: returns U (..., 3, 3), S (..., 3), Vh (..., 3, 3)
    U, _, Vt = np.linalg.svd(flat)
    out = U @ Vt
    dets = np.linalg.det(out)
    flip = dets < 0
    if flip.any():
        U_f = U[flip].copy()
        U_f[:, :, -1] *= -1
        out[flip] = U_f @ Vt[flip]
    return out.reshape(R.shape)


def _compute_align_rotation(
    source_rest_pos: np.ndarray,
    source_name_to_idx: Dict[str, int],
    resolved_mapping: RigProfile,
) -> np.ndarray:
    """Compute the 3x3 rotation that maps the source-rig world frame onto
    the SMPL MJCF world frame (X-forward, Y-left, Z-up).

    Uses the rest-pose positions of Pelvis, Chest, L_Hip, R_Hip.
    ``resolved_mapping`` must already have the actual rig-specific bone names
    (with any prefixes applied) — see :func:`select_rig_profile`.
    """

    def _resolve(smpl_name: str) -> int:
        src = resolved_mapping.get(smpl_name)
        if src is None:
            raise KeyError(
                f"Resolved rig mapping is missing '{smpl_name}'. "
                "Cannot compute frame alignment without it."
            )
        if src not in source_name_to_idx:
            raise KeyError(f"Source bone '{src}' not found in FBX rig")
        return source_name_to_idx[src]

    hips = source_rest_pos[_resolve("Pelvis")]
    spine1 = source_rest_pos[_resolve("Chest")]
    l_hip = source_rest_pos[_resolve("L_Hip")]
    r_hip = source_rest_pos[_resolve("R_Hip")]

    # Up vector = direction from Hips to Chest (vertical), normalised.
    up = spine1 - hips
    if np.linalg.norm(up) < 1e-6:
        up = np.array([0.0, 0.0, 1.0])
    up /= np.linalg.norm(up)

    # Left vector = LeftUpLeg - RightUpLeg, then made orthogonal to up.
    left = l_hip - r_hip
    left = left - up * float(np.dot(left, up))
    if np.linalg.norm(left) < 1e-6:
        raise RuntimeError("Could not determine left direction from hip joints.")
    left /= np.linalg.norm(left)

    # Forward = left x up (right-handed convention; verified empirically).
    forward = np.cross(left, up)
    forward /= np.linalg.norm(forward)

    # Recompute left to guarantee orthonormality.
    left = np.cross(up, forward)

    # Each row of R_align is one source-frame basis vector expressed in
    # source-world coords; multiplying by R_align maps source-world vectors
    # into MJCF coords where forward=(1,0,0), left=(0,1,0), up=(0,0,1).
    R_align = np.stack([forward, left, up], axis=0)
    return _ortho_normalize(R_align)


def _compute_smpl_rest_world(
    kinematic_info, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run FK with identity local rotations to get SMPL rest world transforms."""
    Nb = kinematic_info.num_bodies
    root_pos = torch.zeros(1, 3, device=device, dtype=dtype)
    eye = torch.eye(3, device=device, dtype=dtype)
    joint_rot_mats = eye[None, None].expand(1, Nb, 3, 3).contiguous()
    world_pos, world_rot = compute_forward_kinematics_from_transforms(
        kinematic_info, root_pos, joint_rot_mats
    )
    return world_pos[0], world_rot[0]


def _build_source_index_map(
    smpl_body_names,
    source_bone_names,
    resolved_mapping: RigProfile,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each SMPL body, return (source_idx, has_source_flag).

    ``resolved_mapping`` is the output of :func:`select_rig_profile` and only
    contains entries whose rig-specific name actually exists in the FBX.
    """
    name_to_idx = {n: i for i, n in enumerate(source_bone_names)}
    src_idx = np.full(len(smpl_body_names), -1, dtype=np.int64)
    for j, smpl_name in enumerate(smpl_body_names):
        src_name = resolved_mapping.get(smpl_name)
        if src_name is None or src_name not in name_to_idx:
            continue
        src_idx[j] = name_to_idx[src_name]
    has_source = src_idx >= 0
    return src_idx, has_source


# ---- Static-frame trimming ------------------------------------------------ #

# Why this exists:
# Many FBX exports (Mixamo, Kubold/FightingAnimsetPro, Maya HIK) ship each
# action with leading/trailing "padding" keyframes that hold the rest pose.
# Common causes:
#   * ``bake_anim_force_startend_keying=True`` during FBX export, which always
#     writes a key at the scene start/end of every action's f-curve — even
#     when the take's actual motion is shorter than the scene timeline.
#   * Multi-take FBX where every take's f-curves are padded to a common range.
#   * Mocap pipelines that record a few seconds of T-pose before/after the
#     real motion for retargeting reference.
# Blender's ``action.frame_range`` is computed from the f-curve keyframes, so
# it includes that padding and we end up with several "no-motion" frames at
# the head and tail of every clip. We strip them here.


def _trim_static_frames(
    src_world_rot: np.ndarray,
    src_root_pos: np.ndarray,
    rot_threshold_rad: float,
    pos_threshold_m: float,
    pad_frames: int = 1,
    min_keep_frames: int = 4,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Strip leading/trailing frames where there is no significant body motion.

    Detection: per frame transition we measure (a) the average per-bone
    rotation delta (Frobenius norm scaled to ~radians for small angles) and
    (b) the root-position delta. A transition is considered "moving" if
    *either* exceeds its threshold. We trim everything outside the first/last
    moving transition (with optional ``pad_frames`` on each side).

    Returns ``(trimmed_world_rot, trimmed_root_pos, first_kept, last_kept)``.
    Falls back to the original arrays if the clip is shorter than
    ``min_keep_frames * 2`` or appears static throughout.
    """
    T, J = src_world_rot.shape[0], src_world_rot.shape[1]
    if T < min_keep_frames * 2:
        return src_world_rot, src_root_pos, 0, T - 1

    # Per-transition rotation magnitude. ||R_t - R_{t-1}||_F ≈ sqrt(2) * theta
    # for a single bone with rotation angle theta, so dividing by sqrt(2*J)
    # gives an "average per-bone radian" estimate.
    rot_diff = src_world_rot[1:] - src_world_rot[:-1]
    rot_motion = np.linalg.norm(rot_diff.reshape(T - 1, -1), axis=1) / np.sqrt(
        2.0 * J
    )

    pos_diff = src_root_pos[1:] - src_root_pos[:-1]
    pos_motion = np.linalg.norm(pos_diff, axis=1)

    is_moving = (rot_motion > rot_threshold_rad) | (pos_motion > pos_threshold_m)
    moving_idx = np.where(is_moving)[0]
    if moving_idx.size == 0:
        return src_world_rot, src_root_pos, 0, T - 1

    first_kept = max(0, int(moving_idx[0]) - pad_frames)
    last_kept = min(T - 1, int(moving_idx[-1]) + 1 + pad_frames)

    if (last_kept - first_kept + 1) < min_keep_frames:
        return src_world_rot, src_root_pos, 0, T - 1
    if first_kept == 0 and last_kept == T - 1:
        return src_world_rot, src_root_pos, 0, T - 1

    return (
        src_world_rot[first_kept : last_kept + 1],
        src_root_pos[first_kept : last_kept + 1],
        first_kept,
        last_kept,
    )


# ---- Height-anchoring strategies ----------------------------------------- #

# Why this is its own concept (vs. just using ``RobotState.fix_height``):
# ``fix_height`` shifts so that the **global** minimum body z over all frames
# equals ``foot_offset``. That works for animations where the character is
# always upright (every frame's lowest body == feet on the floor). For
# animations like ``getup_facedown``/``getup_faceup`` the character lies on
# the ground for many frames, so the global minimum is body parts touching
# the floor *during the lying phase*, not the feet during the standing phase.
# That makes the standing feet end up several centimetres above the floor
# after the global shift.
HEIGHT_MODES = ("global", "last_frame", "first_frame", "per_frame", "disable")


def _apply_height_anchor(motion, mode: str, foot_offset: float) -> None:
    """In-place vertical anchoring of ``motion`` according to ``mode``.

    Modes:
      * ``global``: classic ``fix_height`` — global min lands at ``foot_offset``.
      * ``last_frame`` / ``first_frame``: anchor to one frame's lowest body.
        Ideal for getup/laydown clips that end (or start) in stable standing.
      * ``per_frame``: each frame's lowest body lands at ``foot_offset`` (only
        lifts frames below ground, drops by at most 0.02 m).
      * ``disable``: leave the FK output untouched.
    """
    if mode == "disable":
        return
    if mode == "global":
        motion.fix_height(height_offset=foot_offset)
        return
    if mode == "per_frame":
        motion.fix_height_per_frame(height_offset=foot_offset)
        return
    if mode in {"last_frame", "first_frame"}:
        idx = -1 if mode == "last_frame" else 0
        ref_min = motion.rigid_body_pos[idx, :, 2].min().item()
        shift_z = float(-ref_min + foot_offset)
        shift_vec = torch.zeros(3, device=motion.rigid_body_pos.device)
        shift_vec[2] = shift_z
        motion.translate(shift_vec)
        return
    raise ValueError(f"Unknown height_mode '{mode}'. Choose one of {HEIGHT_MODES}.")


# ---- Retargeting core ----------------------------------------------------- #


def retarget_action(
    source_world_rot: np.ndarray,  # (T, J_src, 3, 3) source rest+anim rotations
    source_root_pos: np.ndarray,  # (T, 3) source Hips world position
    source_rest_world_rot: np.ndarray,  # (J_src, 3, 3)
    source_rest_world_pos: np.ndarray,  # (J_src, 3) - for height ratio
    R_align: np.ndarray,  # (3, 3) src-world -> MJCF-world
    smpl_rest_world_pos: torch.Tensor,  # (Nb, 3)
    src_idx: np.ndarray,  # (Nb,) source idx per SMPL body, -1 if unmapped
    parent_indices: torch.Tensor,  # (Nb,) SMPL parent indices (long)
    kinematic_info,
    fps: int,
    device: torch.device,
    dtype: torch.dtype,
    foot_offset: float = 0.015,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (root_pos, smpl_world_rots, smpl_local_rots) tensors at MJCF frame.

    Returns torch tensors ready for ``fk_from_transforms_with_velocities``.
    """
    T = source_world_rot.shape[0]
    Nb = src_idx.shape[0]

    R_align_t = torch.from_numpy(R_align).to(device=device, dtype=dtype)
    R_align_inv_t = R_align_t.transpose(-1, -2).contiguous()

    # Per-SMPL-body retarget delta: D_j = inv(R_src_rest_aligned_j) @ I  (target rest = I)
    # then R_smpl_j(t) = R_align @ R_src_j(t) @ R_src_rest_j^{-1} @ R_align^T
    # We precompute the per-body constant: K_j = R_src_rest_j^{-1} @ R_align^T
    src_rest_inv = np.transpose(source_rest_world_rot, (0, 2, 1))  # (J_src, 3, 3)
    K = np.einsum("jab,bc->jac", src_rest_inv, R_align.T)  # (J_src, 3, 3)
    K_t = torch.from_numpy(K).to(device=device, dtype=dtype)

    # Build (T, Nb, 3, 3) world rotation tensor
    src_rot_t = torch.from_numpy(source_world_rot).to(device=device, dtype=dtype)  # (T, J_src, 3, 3)

    smpl_world_rot = torch.zeros(T, Nb, 3, 3, device=device, dtype=dtype)
    has_source_t = torch.from_numpy(src_idx >= 0).to(device=device)
    src_idx_clamped = np.where(src_idx >= 0, src_idx, 0)
    src_idx_t = torch.from_numpy(src_idx_clamped).to(device=device, dtype=torch.long)

    # Mapped joints: R_smpl(t) = R_align @ R_src(t) @ K_j  (where K_j depends on j)
    gathered_src = src_rot_t[:, src_idx_t, :, :]  # (T, Nb, 3, 3)
    K_gathered = K_t[src_idx_t, :, :]  # (Nb, 3, 3)
    R_align_b = R_align_t[None, None].expand(T, Nb, 3, 3)
    mapped = torch.matmul(
        R_align_b, torch.matmul(gathered_src, K_gathered[None].expand(T, Nb, 3, 3))
    )  # (T, Nb, 3, 3)

    # Apply for mapped bones; we'll fill unmapped sequentially below.
    smpl_world_rot[:, has_source_t, :, :] = mapped[:, has_source_t, :, :]

    # Sequential pass to fill unmapped joints from their parents.
    # SMPL MJCF body order is parents-before-children, so this single pass works.
    parents = parent_indices.cpu().numpy()
    has_src_np = src_idx >= 0
    for j in range(Nb):
        if has_src_np[j]:
            continue
        if parents[j] < 0:  # safety net for the root
            smpl_world_rot[:, j, :, :] = (
                torch.eye(3, device=device, dtype=dtype)[None].expand(T, 3, 3)
            )
        else:
            smpl_world_rot[:, j, :, :] = smpl_world_rot[:, int(parents[j]), :, :]

    # Root translation in MJCF frame.
    src_root_t = torch.from_numpy(source_root_pos).to(device=device, dtype=dtype)
    root_mjcf = torch.matmul(src_root_t, R_align_t.T)  # (T, 3)

    # Translate so that the rest-pose Hips would land at the SMPL rest Pelvis.
    # In MJCF rest, Pelvis is at the origin (we used root_pos=0 for rest FK),
    # so we just remove the source rest-pose Hips offset (after alignment).
    src_hips_rest_aligned = R_align @ source_rest_world_pos[src_idx[0]]
    src_hips_rest_aligned_t = torch.from_numpy(src_hips_rest_aligned).to(
        device=device, dtype=dtype
    )
    # Keep planar XY but anchor character height by our own height fix later.
    root_mjcf = root_mjcf - src_hips_rest_aligned_t  # zero-out at rest

    # Compute SMPL local hinge rotations from world rotations.
    smpl_local = compute_joint_rot_mats_from_global_mats(
        kinematic_info=kinematic_info, global_rot_mats=smpl_world_rot
    )

    return root_mjcf, smpl_world_rot, smpl_local


# ---- Main pipeline -------------------------------------------------------- #


def _process_action(
    name: str,
    action: dict,
    source_data: dict,
    R_align: np.ndarray,
    smpl_rest_world_pos: torch.Tensor,
    src_idx: np.ndarray,
    parent_indices: torch.Tensor,
    kinematic_info,
    output_fps: int,
    device: torch.device,
    dtype: torch.dtype,
    foot_offset: float,
    output_dir: Path,
    height_mode: str = "global",
    trim_static: bool = True,
    trim_rot_threshold: float = 0.005,
    trim_pos_threshold: float = 0.0008,
    trim_pad_frames: int = 1,
):
    src_world_rot = _ortho_normalize_batch(action["world_rot"])  # strip Blender scale
    src_world_pos = action["world_pos"]  # (T, J, 3)
    pelvis_src_idx = src_idx[0]
    src_root_pos = src_world_pos[:, pelvis_src_idx, :]  # (T, 3)

    source_fps = float(source_data["source_fps"])
    if source_fps <= 0:
        source_fps = 60.0

    # Downsample to closest divisor >= output_fps to preserve integer dt.
    fps_int = int(round(source_fps))
    if fps_int % output_fps == 0:
        ds = fps_int // output_fps
        current_fps = output_fps
    else:
        # Fallback: keep source fps if no clean integer divisor.
        ds = 1
        current_fps = fps_int
        print(f"  [warn] {name}: source fps {fps_int} not divisible by {output_fps}, keeping {fps_int}")
    if ds > 1:
        src_world_rot = src_world_rot[::ds]
        src_root_pos = src_root_pos[::ds]
    if src_world_rot.shape[0] < 2:
        print(f"  [skip] {name}: only {src_world_rot.shape[0]} frame(s) after downsample")
        return

    # Trim leading/trailing static "padding" frames produced by FBX exporters.
    # We do this *after* downsampling so the thresholds correspond to the
    # output frame rate (i.e. radians-per-output-frame, metres-per-output-frame).
    if trim_static:
        T_before = src_world_rot.shape[0]
        src_world_rot, src_root_pos, first_kept, last_kept = _trim_static_frames(
            src_world_rot=src_world_rot,
            src_root_pos=src_root_pos,
            rot_threshold_rad=trim_rot_threshold,
            pos_threshold_m=trim_pos_threshold,
            pad_frames=trim_pad_frames,
        )
        T_after = src_world_rot.shape[0]
        if T_after < T_before:
            trimmed_head = first_kept
            trimmed_tail = (T_before - 1) - last_kept
            print(
                f"  trimmed static padding: kept frames [{first_kept}..{last_kept}] "
                f"({T_after}/{T_before}, removed {trimmed_head} head + {trimmed_tail} tail)"
            )
        if src_world_rot.shape[0] < 2:
            print(f"  [skip] {name}: only {src_world_rot.shape[0]} frame(s) after trim")
            return

    root_pos, _, smpl_local = retarget_action(
        source_world_rot=src_world_rot,
        source_root_pos=src_root_pos,
        source_rest_world_rot=_ortho_normalize_batch(source_data["rest_world_rot"]),
        source_rest_world_pos=source_data["rest_world_pos"],
        R_align=R_align,
        smpl_rest_world_pos=smpl_rest_world_pos,
        src_idx=src_idx,
        parent_indices=parent_indices,
        kinematic_info=kinematic_info,
        fps=current_fps,
        device=device,
        dtype=dtype,
        foot_offset=foot_offset,
    )

    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=root_pos,
        joint_rot_mats=smpl_local,
        fps=current_fps,
        compute_velocities=True,
        velocity_max_horizon=3,
    )

    # Cache SMPL local quaternions for downstream tools (matches convert_amass_to_proto).
    from protomotions.utils.rotations import matrix_to_quaternion
    motion.local_rigid_body_rot = matrix_to_quaternion(smpl_local, w_last=True).clone()

    # qpos/dof
    qpos = extract_qpos_from_transforms(
        kinematic_info=kinematic_info,
        root_pos=root_pos,
        joint_rot_mats=smpl_local,
        multi_dof_decomposition_method="exp_map",
    )
    motion.dof_pos = qpos[:, 7:]
    n_j = kinematic_info.num_bodies - 1
    local_angular_vels = compute_angular_velocity(
        batched_robot_rot_mats=smpl_local[:, 1:, :, :],
        fps=current_fps,
    )
    motion.dof_vel = local_angular_vels.reshape(-1, n_j * 3)

    _apply_height_anchor(motion, height_mode, foot_offset)

    motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
        positions=motion.rigid_body_pos,
        velocity=motion.rigid_body_vel,
        vel_thres=0.15,
        height_thresh=0.1,
    ).to(torch.bool)

    out_path = output_dir / f"{name}.motion"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> {out_path}  T={motion.rigid_body_pos.shape[0]} fps={current_fps}")
    torch.save(motion.to_dict(), str(out_path))


def _ensure_extracted_pkl(
    fbx_path: Path,
    pkl_path: Path,
    blender_exe: Optional[str],
    rest_action: str,
) -> None:
    if pkl_path.exists():
        print(f"Reusing extracted pickle: {pkl_path}")
        return

    if blender_exe is None:
        # Try the default Windows install location, then PATH lookup.
        candidate = Path(r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe")
        if candidate.is_file():
            blender_exe = str(candidate)
        else:
            blender_exe = shutil.which("blender")
    if blender_exe is None:
        raise RuntimeError(
            "Could not find Blender. Pass --blender-exe with the full path to blender.exe."
        )

    extractor = _SCRIPT_DIR / "_fbx_extract_blender.py"
    if not extractor.is_file():
        raise FileNotFoundError(f"Missing extractor: {extractor}")

    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        blender_exe,
        "--background",
        "--python",
        str(extractor),
        "--",
        str(fbx_path),
        str(pkl_path),
        rest_action,
    ]
    print("Running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0 or not pkl_path.exists():
        raise RuntimeError("Blender extraction failed.")


def _ensure_rest_pkl(
    rest_fbx: Path,
    pkl_path: Path,
    blender_exe: Optional[str],
) -> None:
    """Extract just the rest pose (bind pose) from a separate FBX file."""
    if pkl_path.exists():
        print(f"Reusing rest-pose pickle: {pkl_path}")
        return

    if blender_exe is None:
        candidate = Path(r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe")
        if candidate.is_file():
            blender_exe = str(candidate)
        else:
            blender_exe = shutil.which("blender")
    if blender_exe is None:
        raise RuntimeError(
            "Could not find Blender. Pass --blender-exe with the full path to blender.exe."
        )

    extractor = _SCRIPT_DIR / "_fbx_extract_blender.py"
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        blender_exe, "--background", "--python", str(extractor), "--",
        str(rest_fbx), str(pkl_path), "tpose", "--rest-only",
    ]
    print("Running (rest-only):", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0 or not pkl_path.exists():
        raise RuntimeError("Blender rest-pose extraction failed.")


def _replace_rest_pose(source_data: dict, rest_data: dict) -> None:
    """Overlay rest_data's rest pose onto source_data, matching by bone name.

    Bones present in source_data but missing from rest_data keep their
    original (animation-FBX) rest values.
    """
    src_names = source_data["bone_names"]
    rest_names = rest_data["bone_names"]
    rest_name_to_idx = {n: i for i, n in enumerate(rest_names)}

    new_pos = source_data["rest_world_pos"].copy()
    new_rot = source_data["rest_world_rot"].copy()
    matched, missing = 0, []
    for j, n in enumerate(src_names):
        if n in rest_name_to_idx:
            ri = rest_name_to_idx[n]
            new_pos[j] = rest_data["rest_world_pos"][ri]
            new_rot[j] = rest_data["rest_world_rot"][ri]
            matched += 1
        else:
            missing.append(n)
    source_data["rest_world_pos"] = new_pos
    source_data["rest_world_rot"] = new_rot
    print(f"Replaced rest pose: {matched}/{len(src_names)} bones matched from rest FBX.")
    if missing:
        print(f"  [warn] bones missing from rest FBX (kept old rest): {missing[:8]}{'…' if len(missing) > 8 else ''}")


def _detect_scale(
    source_rest_pos: np.ndarray,
    source_name_to_idx: Dict[str, int],
    resolved_mapping: RigProfile,
    R_align: np.ndarray,
    target_hip_height_m: float = 0.95,
) -> float:
    """Estimate cm/m scale factor by measuring rest-pose Hip height in MJCF frame.

    After alignment, the rest pose should be upright (hips above ground), so the
    aligned z-coordinate of the hip joint approximates the human's hip height.
    """
    pelvis_name = resolved_mapping.get("Pelvis")
    if pelvis_name is None or pelvis_name not in source_name_to_idx:
        return 1.0
    hips_src = source_rest_pos[source_name_to_idx[pelvis_name]]
    hips_aligned = R_align @ hips_src

    # Use the foot-rooted form: hip z above the lowest mapped joint.
    foot_candidates = ["L_Ankle", "R_Ankle", "L_Toe", "R_Toe"]
    foot_zs = []
    for sname in foot_candidates:
        rname = resolved_mapping.get(sname)
        if rname and rname in source_name_to_idx:
            foot_aligned = R_align @ source_rest_pos[source_name_to_idx[rname]]
            foot_zs.append(float(foot_aligned[2]))
    if foot_zs:
        height_units = float(hips_aligned[2]) - min(foot_zs)
    else:
        height_units = float(hips_aligned[2])

    if height_units < 1e-3:
        return 1.0
    return target_hip_height_m / height_units


# ---- CLI ------------------------------------------------------------------ #


@app.command()
def main(
    fbx_path: Path = typer.Argument(..., help="Input FBX file."),
    output_dir: Path = typer.Option(
        Path("data/yaml_files/retargeted_fbx"),
        help="Where to write per-action .motion files.",
    ),
    output_fps: int = typer.Option(30, help="Target output FPS for .motion files."),
    foot_offset: float = typer.Option(0.015, help="Height-fix foot offset (smpl=0.015, smplx=0.017)."),
    extracted_pkl: Optional[Path] = typer.Option(
        None,
        help="Reuse a previously extracted pickle (skip Blender invocation).",
    ),
    blender_exe: Optional[str] = typer.Option(
        None, help="Path to blender.exe; defaults to Blender 4.5 Foundation install."
    ),
    rest_action: str = typer.Option(
        "tpose", help="Substring of the FBX action used as the rest pose."
    ),
    only_action: Optional[str] = typer.Option(
        None, help="If set, only process actions whose name contains this substring."
    ),
    smpl_xml: str = typer.Option(
        "protomotions/data/assets/mjcf/smpl_humanoid.xml",
        help="Path to the SMPL MJCF used for kinematic info.",
    ),
    rig_profile: Optional[str] = typer.Option(
        None,
        help=(
            "Override rig auto-detection. Available profiles: "
            f"{', '.join(RIG_PROFILES.keys())}."
        ),
    ),
    rest_fbx: Optional[Path] = typer.Option(
        None,
        help=(
            "Optional separate FBX file whose bind pose (or 'tpose' action) is "
            "used as the source rest pose. Required for rigs whose animation "
            "FBX has a non-T-pose bind (e.g. kimodo getup_facedown)."
        ),
    ),
    scale: Optional[float] = typer.Option(
        None,
        help=(
            "Multiplier applied to all source positions (rest + per-frame). "
            "If omitted, auto-detected from the rest pose (assumes ~0.95 m hip "
            "height). Pass 0.01 for cm rigs, 1.0 to disable."
        ),
    ),
    height_mode: str = typer.Option(
        "global",
        help=(
            "Vertical anchoring strategy. 'global' (default) pins the global "
            "minimum body z to foot_offset (best for upright clips like walks). "
            "'last_frame' / 'first_frame' pin one specific frame's lowest body "
            "to foot_offset (best for getup/laydown clips that end / start in "
            "stable standing). 'per_frame' clamps each frame individually. "
            "'disable' leaves the FK output untouched."
        ),
    ),
    trim_static: bool = typer.Option(
        True,
        help=(
            "Strip leading/trailing frames that hold the rest pose (FBX "
            "padding from force-start/end keying or multi-take exports). "
            "Pass --no-trim-static to keep the original frame count, e.g. "
            "for clips that are *intentionally* a static pose."
        ),
    ),
    trim_rot_threshold: float = typer.Option(
        0.005,
        help=(
            "Per-frame avg-bone rotation delta (radians) below which a frame "
            "is treated as 'no motion' for trimming. ~0.005 rad ≈ 0.3°/frame. "
            "Increase to be more aggressive, decrease to preserve subtle motion."
        ),
    ),
    trim_pos_threshold: float = typer.Option(
        0.0008,
        help=(
            "Per-frame root translation delta (metres) below which a frame "
            "is treated as 'no motion' for trimming."
        ),
    ),
    trim_pad_frames: int = typer.Option(
        1,
        help=(
            "Extra frames to keep on each side of the detected motion span. "
            "Helps avoid clipping the very first/last bit of motion."
        ),
    ),
):
    if height_mode not in HEIGHT_MODES:
        raise typer.BadParameter(
            f"Unknown --height-mode '{height_mode}'. "
            f"Available: {', '.join(HEIGHT_MODES)}"
        )
    fbx_path = fbx_path.resolve()
    if not fbx_path.is_file():
        raise typer.BadParameter(f"FBX not found: {fbx_path}")

    if extracted_pkl is None:
        extracted_pkl = output_dir / "_extracted" / (fbx_path.stem + ".pkl")
    extracted_pkl = extracted_pkl.resolve()

    _ensure_extracted_pkl(fbx_path, extracted_pkl, blender_exe, rest_action)

    with open(extracted_pkl, "rb") as f:
        source_data = pickle.load(f)

    print(
        f"Loaded {len(source_data['actions'])} actions, "
        f"{len(source_data['bone_names'])} source bones, "
        f"fps={source_data['source_fps']}"
    )

    if rest_fbx is not None:
        rest_fbx = rest_fbx.resolve()
        if not rest_fbx.is_file():
            raise typer.BadParameter(f"Rest FBX not found: {rest_fbx}")
        rest_pkl = extracted_pkl.parent / (rest_fbx.stem + "__rest.pkl")
        _ensure_rest_pkl(rest_fbx, rest_pkl, blender_exe)
        with open(rest_pkl, "rb") as f:
            rest_data = pickle.load(f)
        _replace_rest_pose(source_data, rest_data)

    device = torch.device("cpu")
    dtype = torch.float32

    kinematic_info = extract_kinematic_info(smpl_xml)
    smpl_body_names = kinematic_info.body_names
    print(f"SMPL MJCF bodies ({len(smpl_body_names)}): {smpl_body_names[:5]}…")

    # Sanity: confirm the body_names match SMPL_MUJOCO_NAMES (24 joints).
    if list(smpl_body_names) != list(SMPL_MUJOCO_NAMES):
        print(
            "  [warn] kinematic_info body_names differ from SMPL_MUJOCO_NAMES; "
            "retargeting still uses kinematic_info ordering."
        )

    parent_indices = torch.tensor(kinematic_info.parent_indices, dtype=torch.long)

    # Compute SMPL rest world transforms (used only for height ratio).
    smpl_rest_pos, _ = _compute_smpl_rest_world(kinematic_info, device, dtype)

    # Auto-detect (or honour override of) the source rig naming convention,
    # then build a resolved bone map (rig-specific names with prefixes applied).
    profile_name, resolved_mapping = select_rig_profile(
        source_data["bone_names"], profile_override=rig_profile
    )
    print(
        f"Rig profile: '{profile_name}' "
        f"({len(resolved_mapping)}/{len(RIG_PROFILES[profile_name])} bones resolved)"
    )

    name_to_idx = {n: i for i, n in enumerate(source_data["bone_names"])}
    R_align = _compute_align_rotation(
        source_data["rest_world_pos"],
        name_to_idx,
        resolved_mapping,
    )
    print("Source -> MJCF alignment rotation:")
    with np.printoptions(precision=3, suppress=True):
        print(R_align)

    if scale is None:
        scale = _detect_scale(
            source_data["rest_world_pos"], name_to_idx, resolved_mapping, R_align
        )
        print(f"Auto-detected source scale: {scale:.5f}  (1.0=meters, 0.01=cm)")
    else:
        print(f"Using explicit source scale: {scale:.5f}")
    if abs(scale - 1.0) > 1e-6:
        source_data["rest_world_pos"] = source_data["rest_world_pos"] * float(scale)
        for act in source_data["actions"].values():
            act["world_pos"] = act["world_pos"] * float(scale)

    src_idx, _ = _build_source_index_map(
        smpl_body_names,
        source_data["bone_names"],
        resolved_mapping,
    )
    n_mapped = int((src_idx >= 0).sum())
    print(f"Mapped {n_mapped}/{len(smpl_body_names)} SMPL bodies.")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    actions = source_data["actions"]
    keys = sorted(actions.keys())
    if only_action is not None:
        keys = [k for k in keys if only_action in k]
        print(f"Filtered to {len(keys)} actions matching '{only_action}'")

    for name in keys:
        print(f"\n[Action] {name}")
        try:
            _process_action(
                name=name,
                action=actions[name],
                source_data=source_data,
                R_align=R_align,
                smpl_rest_world_pos=smpl_rest_pos,
                src_idx=src_idx,
                parent_indices=parent_indices,
                kinematic_info=kinematic_info,
                output_fps=output_fps,
                device=device,
                dtype=dtype,
                foot_offset=foot_offset,
                output_dir=output_dir,
                height_mode=height_mode,
                trim_static=trim_static,
                trim_rot_threshold=trim_rot_threshold,
                trim_pos_threshold=trim_pos_threshold,
                trim_pad_frames=trim_pad_frames,
            )
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"  [error] failed to process {name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    with torch.no_grad():
        app()
