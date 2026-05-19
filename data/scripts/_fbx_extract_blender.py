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
"""Stage 1: extract bone hierarchy + per-frame world transforms from an FBX.

Must be run inside Blender's bundled Python (which has ``bpy``):

    blender --background --python data/scripts/_fbx_extract_blender.py -- \
        <input.fbx> <output.pkl> [rest_action_substring]

The output pickle contains:
  - bone_names: list[str]            # in armature traversal order
  - parent_indices: list[int]        # -1 for root
  - rest_world_pos: (J, 3) float32   # bone head position in armature world frame
  - rest_world_rot: (J, 3, 3) float32  # bone orientation (Y axis = bone direction)
  - source_fps: float                # scene fps used during sampling
  - actions: dict[clean_name -> {
        "n_frames": int,
        "world_pos": (T, J, 3) float32,
        "world_rot": (T, J, 3, 3) float32,
    }]
"""
import os
import pickle
import re
import sys

import numpy as np


def _parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Missing '--' separator in argv.")
    rest = argv[argv.index("--") + 1 :]
    if len(rest) < 2:
        raise SystemExit(
            "Usage: blender --background --python _fbx_extract_blender.py -- "
            "<input.fbx> <output.pkl> [rest_action_substring]"
        )
    fbx_path = rest[0]
    out_path = rest[1]
    rest_substr = rest[2] if len(rest) > 2 else "tpose"
    return fbx_path, out_path, rest_substr


def _ensure_armature(bpy_module):
    armatures = [o for o in bpy_module.data.objects if o.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("No armature found in the imported scene.")
    if len(armatures) > 1:
        print(f"WARNING: multiple armatures found, using '{armatures[0].name}'")
    return armatures[0]


_FCURVE_PATH_RE = re.compile(r'^pose\.bones\["([^"]+)"\]\.(location|rotation_quaternion|rotation_euler|scale)$')


def _build_action_fcurve_index(action):
    """Return ``{(bone_name, prop): {array_index: fcurve}}`` for fast frame eval."""
    index = {}
    for fc in action.fcurves:
        m = _FCURVE_PATH_RE.match(fc.data_path)
        if not m:
            continue
        bone_name, prop = m.group(1), m.group(2)
        index.setdefault((bone_name, prop), {})[fc.array_index] = fc
    return index


def _build_bone_basis_matrices(bpy_module, fcurve_index, bone_names, frame, rotation_modes):
    """Evaluate per-bone ``matrix_basis`` (the pose-space deviation from rest) at ``frame``.

    Returns a dict {bone_name: 4x4 numpy matrix}.
    """
    Matrix = bpy_module.types.bpy_struct  # placeholder for type hints
    from mathutils import Matrix, Quaternion, Euler  # noqa: E402

    out = {}
    for name in bone_names:
        loc = [0.0, 0.0, 0.0]
        scl = [1.0, 1.0, 1.0]
        if (name, "location") in fcurve_index:
            for ai, fc in fcurve_index[(name, "location")].items():
                loc[ai] = fc.evaluate(frame)
        if (name, "scale") in fcurve_index:
            for ai, fc in fcurve_index[(name, "scale")].items():
                scl[ai] = fc.evaluate(frame)

        rot_mode = rotation_modes.get(name, "QUATERNION")
        if rot_mode == "QUATERNION":
            quat = [1.0, 0.0, 0.0, 0.0]
            if (name, "rotation_quaternion") in fcurve_index:
                for ai, fc in fcurve_index[(name, "rotation_quaternion")].items():
                    quat[ai] = fc.evaluate(frame)
            R = Quaternion(quat).to_matrix().to_4x4()
        elif rot_mode in {"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"}:
            eul = [0.0, 0.0, 0.0]
            if (name, "rotation_euler") in fcurve_index:
                for ai, fc in fcurve_index[(name, "rotation_euler")].items():
                    eul[ai] = fc.evaluate(frame)
            R = Euler(eul, rot_mode).to_matrix().to_4x4()
        else:
            R = Matrix.Identity(4)

        T = Matrix.Translation(loc)
        S = Matrix.Diagonal((scl[0], scl[1], scl[2], 1.0))
        out[name] = np.array(T @ R @ S, dtype=np.float64)
    return out


def _build_armature_pose_matrices(bone_chain, basis_matrices, parent_relative_rest, root_rest, world):
    """Compose pose-bone armature-local matrices from per-bone basis matrices.

    For each bone with a parent::
        pose_armature[b] = pose_armature[parent] @ parent_relative_rest[b] @ basis[b]

    For the root bone(s)::
        pose_armature[b] = root_rest[b] @ basis[b]

    Then world = armature.matrix_world @ pose_armature[b].
    Returns ``{name: 4x4 numpy world matrix}``.
    """
    pose_local = {}
    for name, parent_name in bone_chain:
        if parent_name is None:
            pose_local[name] = root_rest[name] @ basis_matrices[name]
        else:
            pose_local[name] = (
                pose_local[parent_name] @ parent_relative_rest[name] @ basis_matrices[name]
            )
    return {name: world @ pose_local[name] for name in pose_local}


def _sample_pose_from_fcurves(bpy_module, armature, bone_chain, basis_at_frame, parent_rel, root_rest):
    Mw = np.array(armature.matrix_world, dtype=np.float64)
    world = _build_armature_pose_matrices(bone_chain, basis_at_frame, parent_rel, root_rest, Mw)
    n = len(bone_chain)
    pos = np.zeros((n, 3), dtype=np.float64)
    rot = np.zeros((n, 3, 3), dtype=np.float64)
    for i, (name, _parent) in enumerate(bone_chain):
        M = world[name]
        pos[i] = M[:3, 3]
        rot[i] = M[:3, :3]
    return pos, rot


def _clear_nla_tracks(armature):
    ad = armature.animation_data
    if ad is None:
        return
    while ad.nla_tracks:
        ad.nla_tracks.remove(ad.nla_tracks[0])


def _bind_action(armature, action):
    ad = armature.animation_data
    if ad is None:
        ad = armature.animation_data_create()
    ad.action = action
    if hasattr(ad, "action_blend_type"):
        ad.action_blend_type = "REPLACE"
    if hasattr(ad, "action_extrapolation"):
        ad.action_extrapolation = "HOLD"


def _find_action(bpy_module, substr):
    substr_lower = substr.lower()
    for action in bpy_module.data.actions:
        if substr_lower in action.name.lower():
            return action
    return None


def _clean_action_name(action_name: str) -> str:
    """Map e.g. 'Armature|KB_Idle_1|KB_Idle_1:BaseAnimation' -> 'KB_Idle_1'."""
    if "|" in action_name:
        parts = action_name.split("|")
        if len(parts) >= 2:
            mid = parts[1]
            return mid.split(":")[0]
    return action_name


def main():
    fbx_path, out_path, rest_substr = _parse_args()

    if not os.path.isfile(fbx_path):
        raise SystemExit(f"FBX not found: {fbx_path}")

    import bpy  # imported lazily so the module loads outside Blender for type checks

    bpy.ops.wm.read_factory_settings(use_empty=True)

    print(f"Importing FBX: {fbx_path}")
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    armature = _ensure_armature(bpy)
    if armature.animation_data is None:
        armature.animation_data_create()
    _clear_nla_tracks(armature)

    bones = list(armature.data.bones)
    bone_names = [b.name for b in bones]
    name_to_idx = {n: i for i, n in enumerate(bone_names)}
    parent_indices = [
        name_to_idx[b.parent.name] if b.parent is not None else -1
        for b in bones
    ]
    rotation_modes = {pb.name: pb.rotation_mode for pb in armature.pose.bones}

    # bone_chain: ordered list of (name, parent_name) traversed parents-first.
    bone_chain = [
        (b.name, b.parent.name if b.parent is not None else None) for b in bones
    ]

    # Per-bone "parent-relative rest": bone.parent.matrix_local^-1 @ bone.matrix_local.
    # For the root we use bone.matrix_local directly.
    parent_rel = {}
    root_rest = {}
    for b in bones:
        Mb = np.array(b.matrix_local, dtype=np.float64)
        if b.parent is None:
            root_rest[b.name] = Mb
        else:
            Mp_inv = np.linalg.inv(np.array(b.parent.matrix_local, dtype=np.float64))
            parent_rel[b.name] = Mp_inv @ Mb

    fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
    print(f"Scene fps: {fps}, bones: {len(bones)}")

    rest_action = _find_action(bpy, rest_substr)
    if rest_action is not None:
        idx = _build_action_fcurve_index(rest_action)
        f0 = int(round(rest_action.frame_range[0]))
        basis = _build_bone_basis_matrices(bpy, idx, bone_names, f0, rotation_modes)
        rest_pos, rest_rot = _sample_pose_from_fcurves(
            bpy, armature, bone_chain, basis, parent_rel, root_rest
        )
        print(f"Rest pose sampled from action: {rest_action.name} (frame {f0})")
    else:
        # Fallback to bind pose (matrix_local)
        Mw = np.array(armature.matrix_world, dtype=np.float64)
        rest_pos = np.zeros((len(bone_names), 3), dtype=np.float64)
        rest_rot = np.zeros((len(bone_names), 3, 3), dtype=np.float64)
        for i, b in enumerate(bones):
            Mw_b = Mw @ np.array(b.matrix_local, dtype=np.float64)
            rest_pos[i] = Mw_b[:3, 3]
            rest_rot[i] = Mw_b[:3, :3]
        print("Rest pose sampled from bind pose (no rest action found)")

    actions_data = {}
    skip_substrs = ("tpose", "bindpose", "bind_pose", "t_pose", "rest")
    for action in bpy.data.actions:
        lower = action.name.lower()
        if any(s in lower for s in skip_substrs):
            print(f"Skipping rest-style action: {action.name}")
            continue

        clean = _clean_action_name(action.name)
        f_start = int(round(action.frame_range[0]))
        f_end = int(round(action.frame_range[1]))
        if f_end < f_start:
            print(f"Skipping {clean}: invalid frame range {f_start}..{f_end}")
            continue
        n_frames = f_end - f_start + 1

        idx = _build_action_fcurve_index(action)
        pos = np.zeros((n_frames, len(bone_names), 3), dtype=np.float32)
        rot = np.zeros((n_frames, len(bone_names), 3, 3), dtype=np.float32)
        for i, f in enumerate(range(f_start, f_end + 1)):
            basis = _build_bone_basis_matrices(bpy, idx, bone_names, f, rotation_modes)
            p, r = _sample_pose_from_fcurves(
                bpy, armature, bone_chain, basis, parent_rel, root_rest
            )
            pos[i] = p.astype(np.float32)
            rot[i] = r.astype(np.float32)

        actions_data[clean] = {
            "n_frames": n_frames,
            "world_pos": pos,
            "world_rot": rot,
        }
        print(f"  Sampled {clean}: {n_frames} frames")

    payload = {
        "bone_names": bone_names,
        "parent_indices": parent_indices,
        "rest_world_pos": rest_pos.astype(np.float32),
        "rest_world_rot": rest_rot.astype(np.float32),
        "source_fps": float(fps),
        "actions": actions_data,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved {len(actions_data)} actions to {out_path}")


if __name__ == "__main__":
    main()
