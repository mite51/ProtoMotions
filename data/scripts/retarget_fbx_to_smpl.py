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

# Default mapping: SMPL MJCF bone name -> source FBX bone name.
# Covers Mixamo, Maya HIK, Kubold animset and most "standard humanoid" rigs.
DEFAULT_BONE_MAPPING: Dict[str, str] = {
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
    # SMPL "Spine" (middle vertebra) has no direct counterpart on a 2-spine
    # source rig, so we let it inherit the parent (Torso).
    "Chest": "Spine1",
    "Neck": "Neck",
    "Head": "Head",
    "L_Thorax": "LeftShoulder",
    "L_Shoulder": "LeftArm",
    "L_Elbow": "LeftForeArm",
    "L_Wrist": "LeftHand",
    # SMPL "L_Hand" is a terminal joint past the wrist — keep at rest.
    "R_Thorax": "RightShoulder",
    "R_Shoulder": "RightArm",
    "R_Elbow": "RightForeArm",
    "R_Wrist": "RightHand",
}


# Common Mixamo-style fallback names with explicit "mixamorig:" prefix.
MIXAMO_PREFIX_VARIANTS = ("", "mixamorig:", "mixamorig1:", "mixamorig2:")


# ---- Helpers -------------------------------------------------------------- #


def _resolve_source_bone(source_bone_names, source_name: str) -> Optional[str]:
    """Return the actual bone name from the rig, trying common Mixamo prefixes."""
    name_set = set(source_bone_names)
    for prefix in MIXAMO_PREFIX_VARIANTS:
        candidate = prefix + source_name
        if candidate in name_set:
            return candidate
    return None


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
    bone_map: Dict[str, str],
) -> np.ndarray:
    """Compute the 3x3 rotation that maps the source-rig world frame onto
    the SMPL MJCF world frame (X-forward, Y-left, Z-up).

    Uses the rest-pose positions of Hips, Spine1, LeftUpLeg, RightUpLeg.
    """

    def _resolve(smpl_name: str) -> int:
        src = bone_map.get(smpl_name)
        if src is None:
            raise KeyError(f"No source mapping for SMPL '{smpl_name}'")
        actual = _resolve_source_bone(list(source_name_to_idx.keys()), src)
        if actual is None:
            raise KeyError(f"Source bone '{src}' not found in FBX rig")
        return source_name_to_idx[actual]

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
    bone_map: Dict[str, str],
) -> Tuple[np.ndarray, np.ndarray]:
    """For each SMPL body, return (source_idx, has_source_flag)."""
    name_to_idx = {n: i for i, n in enumerate(source_bone_names)}
    src_idx = np.full(len(smpl_body_names), -1, dtype=np.int64)
    for j, smpl_name in enumerate(smpl_body_names):
        src_name = bone_map.get(smpl_name)
        if src_name is None:
            continue
        actual = _resolve_source_bone(source_bone_names, src_name)
        if actual is None:
            print(f"  [warn] source bone '{src_name}' not found, leaving '{smpl_name}' free")
            continue
        src_idx[j] = name_to_idx[actual]
    has_source = src_idx >= 0
    return src_idx, has_source


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

    motion.fix_height(height_offset=foot_offset)

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
):
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

    # Map source bones.
    name_to_idx = {n: i for i, n in enumerate(source_data["bone_names"])}
    R_align = _compute_align_rotation(
        source_data["rest_world_pos"],
        name_to_idx,
        DEFAULT_BONE_MAPPING,
    )
    print("Source -> MJCF alignment rotation:")
    with np.printoptions(precision=3, suppress=True):
        print(R_align)

    src_idx, _ = _build_source_index_map(
        smpl_body_names,
        source_data["bone_names"],
        DEFAULT_BONE_MAPPING,
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
            )
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"  [error] failed to process {name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    with torch.no_grad():
        app()
