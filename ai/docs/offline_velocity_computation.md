# Offline Computation of Rigid Body Linear and Angular Velocities

This document describes how to compute per-body linear and angular velocities from a motion clip consisting of world-space rigid body positions and rotations sampled at a fixed frame rate.

These velocities are computed **offline** (at motion preprocessing time) and stored alongside each frame of the motion data.

---

## Inputs

| Name | Shape | Description |
|---|---|---|
| `positions` | `(T, B, 3)` | World-space positions for each body at each frame. `T` = number of frames, `B` = number of rigid bodies. |
| `rotations` | `(T, B, 3, 3)` | World-space rotation matrices for each body at each frame. |
| `fps` | scalar | Frames per second of the motion clip (e.g. 30). |

The time step between frames is:

```
dt = 1.0 / fps
```

---

## Linear Velocity

Linear velocity is computed via **central finite differences** on the world-space positions, using `torch.gradient` along the time axis.

### Algorithm

```python
# positions: (T, B, 3)
# fps: int

pos_grad = torch.gradient(positions, dim=0)[0]   # (T, B, 3)
linear_velocity = pos_grad * fps                  # (T, B, 3)
```

### What `torch.gradient` does

`torch.gradient(input, dim=0)` computes a **central difference** along axis 0:

- For interior frames `t` (where `1 <= t <= T-2`):

```
grad[t] = (positions[t+1] - positions[t-1]) / 2.0
```

- For boundary frames:
  - Frame 0 (first): `grad[0] = positions[1] - positions[0]` (forward difference)
  - Frame T-1 (last): `grad[T-1] = positions[T-1] - positions[T-2]` (backward difference)

The gradient is then divided by `dt` (equivalently multiplied by `fps`) to convert from displacement-per-frame to displacement-per-second:

```
linear_velocity[t] = grad[t] / dt = grad[t] * fps
```

### Result

| Name | Shape | Units |
|---|---|---|
| `linear_velocity` | `(T, B, 3)` | meters/second (world frame) |

### Edge case

If `T < 2`, return a zero tensor of shape `(T, B, 3)`.

---

## Angular Velocity

Angular velocity is computed via **quaternion finite differences**. The rotation change between consecutive frames is expressed as a difference quaternion, then converted to axis-angle form and divided by `dt`.

### Quaternion Convention

All quaternions use **XYZW** format (scalar-last):

```
q = [x, y, z, w]
```

where `w` is the real/scalar part and `(x, y, z)` is the imaginary/vector part.

### Algorithm

#### Step 1: Convert rotation matrices to quaternions

Convert each `(3, 3)` rotation matrix to a unit quaternion in XYZW format.

```
quats: (T, B, 4)   # XYZW
```

Any standard matrix-to-quaternion conversion works (e.g. Shepperd's method). The resulting quaternion must be unit-length.

#### Step 2: Compute difference quaternions between consecutive frames

The difference quaternion represents the rotation from frame `t` to frame `t+1`:

```
q_diff[t] = q[t+1] * conjugate(q[t])
```

where:
- **Quaternion conjugate** (XYZW): `conjugate([x, y, z, w]) = [-x, -y, -z, w]`
- **Quaternion multiply** `a * b` (XYZW): standard Hamilton product

After multiplication, **normalize** the result to unit length, ensuring the scalar part `w` is non-negative (flip the entire quaternion if `w < 0`; this selects the shorter rotation path).

This yields `T-1` difference quaternions for frames `0..T-2`.

#### Step 3: Pad to match original frame count

Prepend an identity quaternion `[0, 0, 0, 1]` for frame 0 (zero angular velocity at the first frame):

```
diff_quats_padded: (T, B, 4)
diff_quats_padded[0]   = [0, 0, 0, 1]   # identity
diff_quats_padded[1:]  = diff_quats[0:]  # from step 2
```

#### Step 4: Extract axis and angle from difference quaternions

For each difference quaternion `q = [x, y, z, w]`:

1. If `w < 0`, flip the quaternion: `q = -q` (ensures angle is in `[0, pi]`).
2. Compute the vector part norm: `norm_xyz = sqrt(x^2 + y^2 + z^2)`
3. Compute the angle: `angle = 2 * atan2(norm_xyz, w)`
4. Compute the normalized axis: `axis = [x, y, z] / max(norm_xyz, 1e-9)`

#### Step 5: Compute angular velocity

```
angular_velocity = axis * angle * fps
```

Or equivalently:

```
angular_velocity = axis * angle / dt
```

### Result

| Name | Shape | Units |
|---|---|---|
| `angular_velocity` | `(T, B, 3)` | radians/second (world frame) |

### Edge case

If `T < 2`, return a zero tensor of shape `(T, B, 3)`.

### Note on first-frame padding

Because the difference quaternion at frame 0 is the identity, angular velocity at frame 0 is always zero. This is a **backward-looking** finite difference padded at the start, unlike the linear velocity which uses central/boundary differences via `torch.gradient`. The angular velocity at frame `t` (for `t >= 1`) represents the rotation that occurred from frame `t-1` to frame `t`.

---

## Runtime Interpolation Between Frames

When querying velocities at arbitrary times (not exactly on a frame boundary), the stored per-frame velocities are **linearly interpolated** between the two bracketing frames.

Given a query time that falls between frame `f0` and frame `f1 = f0 + 1` with a blend factor `alpha` in `[0, 1]`:

```
velocity_at_time = (1 - alpha) * velocity[f0] + alpha * velocity[f1]
```

This applies identically to both linear and angular velocity.

---

## Complete Reference Implementation (PyTorch)

```python
import torch

def compute_linear_velocity(positions: torch.Tensor, fps: int) -> torch.Tensor:
    """
    Compute per-body linear velocity from world positions.

    Args:
        positions: (T, B, 3) world-space body positions.
        fps: frames per second.

    Returns:
        (T, B, 3) linear velocities in meters/second.
    """
    if positions.shape[0] < 2:
        return torch.zeros_like(positions)

    pos_grad = torch.gradient(positions, dim=0)[0]
    return pos_grad * fps


def compute_angular_velocity(rotation_matrices: torch.Tensor, fps: int) -> torch.Tensor:
    """
    Compute per-body angular velocity from world rotation matrices.

    Args:
        rotation_matrices: (T, B, 3, 3) world-space rotation matrices.
        fps: frames per second.

    Returns:
        (T, B, 3) angular velocities in radians/second.
    """
    T = rotation_matrices.shape[0]
    if T < 2:
        return torch.zeros(
            rotation_matrices.shape[:-2] + (3,),
            device=rotation_matrices.device,
            dtype=rotation_matrices.dtype,
        )

    # 1. Convert rotation matrices to XYZW quaternions
    quats = matrix_to_quat_xyzw(rotation_matrices)  # (T, B, 4)

    # 2. Difference quaternion: q_diff = q_{t+1} * conj(q_t)
    q_t = quats[:-1]          # (T-1, B, 4)
    q_next = quats[1:]        # (T-1, B, 4)
    q_t_inv = quat_conjugate(q_t)
    diff_q = quat_multiply(q_next, q_t_inv)
    diff_q = quat_normalize_positive_w(diff_q)

    # 3. Pad frame 0 with identity quaternion [0, 0, 0, 1]
    identity = torch.zeros_like(diff_q[:1])
    identity[..., 3] = 1.0
    diff_q = torch.cat([identity, diff_q], dim=0)  # (T, B, 4)

    # 4. Extract axis-angle
    angle, axis = quat_to_angle_axis(diff_q)  # angle: (T, B), axis: (T, B, 3)

    # 5. angular_velocity = axis * angle / dt
    ang_vel = axis * angle.unsqueeze(-1) * fps

    return ang_vel


# --- Quaternion helpers (XYZW convention) ---

def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of XYZW quaternion: negate xyz, keep w."""
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two XYZW quaternions."""
    x1, y1, z1, w1 = a.unbind(-1)
    x2, y2, z2, w2 = b.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([x, y, z, w], dim=-1)


def quat_normalize_positive_w(q: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion to unit length with non-negative w."""
    # Flip so w >= 0
    flip = (q[..., 3:] < 0).float()
    q = (1 - 2 * flip) * q
    # Normalize to unit length
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def quat_to_angle_axis(q: torch.Tensor):
    """
    Convert XYZW quaternion to (angle, axis).
    Angle in [0, pi]. Axis is unit-length.

    Args:
        q: (..., 4) XYZW quaternions.

    Returns:
        angle: (...) rotation angle in radians.
        axis: (..., 3) unit rotation axis.
    """
    # Ensure w >= 0 for shortest path
    flip = (q[..., 3] < 0).unsqueeze(-1)
    q = torch.where(flip, -q, q)

    xyz = q[..., :3]
    w = q[..., 3]
    norm_xyz = xyz.norm(dim=-1)

    angle = 2.0 * torch.atan2(norm_xyz, w)
    axis = xyz / norm_xyz.unsqueeze(-1).clamp(min=1e-9)

    return angle, axis
```

---

## Summary Table

| Quantity | Method | Boundary Handling | Frame 0 Value |
|---|---|---|---|
| Linear velocity | Central finite difference on positions (`torch.gradient`) | Forward diff at frame 0, backward diff at frame T-1 | Non-zero (forward difference) |
| Angular velocity | Quaternion difference between consecutive frames, converted to axis-angle | Identity quaternion prepended for frame 0 | Always zero |

---

## Source Code References

All paths are relative to the ProtoMotions repository root.

### Velocity computation (offline, at motion preprocessing time)

| File | Function / Line | Description |
|---|---|---|
| `protomotions/components/pose_lib.py` | `compute_cartesian_velocity()` (line 1004) | Computes linear velocity via `torch.gradient` on world positions. |
| `protomotions/components/pose_lib.py` | `compute_angular_velocity()` (line 1029) | Computes angular velocity via quaternion finite differences on world rotation matrices. |
| `protomotions/components/pose_lib.py` | `compute_kinematics_velocities()` (line 1079) | Wrapper that calls both of the above and returns `(lin_vel, ang_vel)`. |
| `protomotions/components/pose_lib.py` | `fk_from_transforms_with_velocities()` (line 1129) | Performs FK from root pos + joint rotations, then calls `compute_kinematics_velocities` to populate `rigid_body_vel` and `rigid_body_ang_vel` on the result. |

### Motion conversion pipeline (where velocities are first created)

| File | Function / Line | Description |
|---|---|---|
| `data/scripts/convert_amass_to_proto.py` | `convert_amass_to_motion()` (line 131) | Entry point for converting AMASS `.npz` data to `.motion` files. Calls `fk_from_transforms_with_velocities` (line 228) with `compute_velocities=True` to produce `rigid_body_vel` and `rigid_body_ang_vel`. Also calls `compute_angular_velocity` (line 249) on local joint rotations to produce `dof_vel`. |

### Motion storage and retrieval (MotionLib)

| File | Function / Line | Description |
|---|---|---|
| `protomotions/components/motion_lib.py` | `_load_motions()` (line 404) | Loads `.motion` files and concatenates per-frame tensors. Velocities are stored as `gvs` (linear) and `gavs` (angular) via the field mapping at line 57. |
| `protomotions/components/motion_lib.py` | `get_motion_state_exact_frame()` (line 366) | Retrieves stored velocities at exact frame indices by indexing into `gvs` / `gavs`. |
| `protomotions/components/motion_lib.py` | `get_motion_state()` (line 295) | Queries motion at arbitrary times. Interpolates `rigid_body_vel` and `rigid_body_ang_vel` linearly between bracketing frames (lines 310–322). |
| `protomotions/utils/motion_interpolation_utils.py` | `interpolate_pos()` (line 26) | The linear interpolation function used for blending velocity (and position) values between frames. |

### Quaternion primitives

| File | Function / Line | Description |
|---|---|---|
| `protomotions/utils/rotations.py` | `quat_mul()` (line 59) | Hamilton product of two quaternions (supports both WXYZ and XYZW via `w_last` flag). |
| `protomotions/utils/rotations.py` | `quat_conjugate()` (line 111) | Quaternion conjugate (negates the vector part). |
| `protomotions/utils/rotations.py` | `quat_mul_norm()` (line 209) | Quaternion multiply followed by normalize-with-positive-scalar. Used in angular velocity computation. |
| `protomotions/utils/rotations.py` | `quat_angle_axis()` (line 218) | Extracts `(angle, axis)` from a quaternion. Uses `atan2` for numerical stability. Angle is in `[0, pi]`. |
| `protomotions/utils/rotations.py` | `quat_normalize()` (line 200) | Normalizes quaternion to unit length with positive scalar part. |
| `protomotions/utils/rotations.py` | `quat_identity_like()` (line 103) | Creates an identity quaternion matching the shape/device of the input. |

### Observation encoding (how velocities are consumed at runtime)

| File | Function / Line | Description |
|---|---|---|
| `protomotions/envs/obs/mimic_obs.py` | `_get_future_ref_states()` (line 146) | Fetches future reference states (including velocities) from MotionLib at time offsets. |
| `protomotions/envs/obs/mimic_obs.py` | `_build_target_poses()` (line 178) | Passes current and reference velocities into the target pose builder. |
| `protomotions/envs/utils/target_poses.py` | `build_max_coords_target_poses()` (line 164) | Encodes velocity as a **delta** (target vel − current simulated vel) rotated into heading-aligned frame (lines 286–328). |
| `protomotions/envs/utils/target_poses.py` | `build_max_coords_target_poses_simple()` (line 334) | Encodes velocity as **absolute** reference vel rotated into heading-aligned frame (lines 426–454). |
