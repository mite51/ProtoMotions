#!/usr/bin/env python3
"""Read-only checks for an SMPL packaged motion library before the A/B runs.

Run from the checkout with its Python interpreter: python -m scripts.check_fighting_motion_data FILE --output REPORT.json
"""
import argparse
import json
from pathlib import Path

import torch

from protomotions.components.pose_lib import (
    compute_forward_kinematics_from_transforms,
    extract_transforms_from_qpos_non_root,
)
from protomotions.robot_configs.factory import robot_config
from protomotions.utils.rotations import quaternion_to_matrix


def check(path):
    torch.set_num_threads(4)
    data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    robot = robot_config("smpl")
    frames = len(data["gts"])
    motions = len(data["motion_num_frames"])
    expected = {"gts": (frames, 24, 3), "grs": (frames, 24, 4),
                "gvs": (frames, 24, 3), "gavs": (frames, 24, 3),
                "dps": (frames, 69), "dvs": (frames, 69), "contacts": (frames, 24)}
    errors = []
    fields = {}
    for key, shape in expected.items():
        value = data.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            errors.append(f"{key}: expected {shape}, got {getattr(value, 'shape', None)}")
            continue
        finite = all(torch.isfinite(chunk).all().item() for chunk in value.split(65536))
        fields[key] = {"shape": list(value.shape), "finite": finite}
        if not finite:
            errors.append(f"{key}: non-finite values")
    lengths = data["motion_num_frames"]
    starts = lengths.cumsum(0) - lengths
    if not torch.equal(starts, data["length_starts"]) or lengths.sum().item() != frames:
        errors.append("Motion frame boundaries do not cover the packaged tensors exactly")
    if (lengths < 2).any() or (data["motion_dt"] <= 0).any():
        errors.append("Motion durations/frame intervals are invalid")
    if not data["contacts"].any():
        errors.append("Contact labels are empty; main's contact reward requires them")
    files = data["motion_files"]
    custom = [name for name in files if any(part in name.replace('\\', '/').split('/') for part in ('Fighting', 'Kimodo', 'TEST'))]
    sample = torch.linspace(0, frames - 1, min(2048, frames)).long()
    fk_errors = {}
    for mode in (False, True):
        local = extract_transforms_from_qpos_non_root(robot.kinematic_info, data["dps"][sample], qpos_is_exp_map_on_3dof_joints=mode)
        local[:, 0] = quaternion_to_matrix(data["grs"][sample, 0], w_last=True)
        positions, _ = compute_forward_kinematics_from_transforms(robot.kinematic_info, data["gts"][sample, 0], local)
        error = (positions - data["gts"][sample]).norm(dim=-1)
        fk_errors["exp_map" if mode else "hinge_angles"] = {"mean_m": error.mean().item(), "max_m": error.max().item()}
    report = {"file": str(Path(path).resolve()), "bytes": Path(path).stat().st_size,
              "motions": motions, "frames": frames, "custom_motion_count": len(custom),
              "windows_path_metadata": sum('\\' in name for name in files),
              "fields": fields, "sampled_fk_error": fk_errors, "errors": errors}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check(args.motion_file)
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    raise SystemExit(bool(report["errors"]))


if __name__ == "__main__":
    main()
