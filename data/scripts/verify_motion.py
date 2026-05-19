# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Sanity-check a retargeted .motion file and render a stick-figure preview.

Usage:
    python data/scripts/verify_motion.py <path/to.motion> [--save-gif out.gif]
"""
import sys
from pathlib import Path

import numpy as np
import torch
import typer

# Ensure project root is on sys.path so namespace package
# protomotions.simulator can be unpickled by torch.load.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import protomotions.simulator.base_simulator.simulator_state  # noqa: E402,F401


app = typer.Typer(pretty_exceptions_enable=False)


SMPL_BONES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest",
    "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]

SMPL_PARENTS = [
    -1, 0, 1, 2, 3,        # Pelvis, L leg
    0, 5, 6, 7,            # R leg
    0, 9, 10,              # Torso/Spine/Chest
    11, 12,                # Neck/Head
    11, 14, 15, 16, 17,    # L arm
    11, 19, 20, 21, 22,    # R arm
]


@app.command()
def main(
    motion_path: Path = typer.Argument(...),
    save_gif: Path = typer.Option(None, help="Optional gif output."),
    fps_override: int = typer.Option(None, help="Override fps if motion.fps missing."),
    n_preview: int = typer.Option(8, help="Number of preview frames in static plot."),
):
    motion = torch.load(str(motion_path), weights_only=False)

    print(f"\n=== {motion_path} ===")
    print(f"Type: {type(motion).__name__}")

    if isinstance(motion, dict):
        keys = list(motion.keys())
    else:
        keys = [k for k in dir(motion) if not k.startswith("_")]
    print(f"Keys/attrs: {keys[:25]}{'…' if len(keys) > 25 else ''}")

    def get(name):
        if isinstance(motion, dict):
            return motion.get(name)
        return getattr(motion, name, None)

    rb_pos = get("rigid_body_pos")
    rb_rot = get("rigid_body_rot")
    dof_pos = get("dof_pos")
    dof_vel = get("dof_vel")
    fps = get("fps") or fps_override or 30

    print("\nShapes:")
    for n, t in [("rigid_body_pos", rb_pos), ("rigid_body_rot", rb_rot),
                 ("dof_pos", dof_pos), ("dof_vel", dof_vel)]:
        if isinstance(t, torch.Tensor):
            print(f"  {n:18s} {tuple(t.shape)}  dtype={t.dtype}")
        else:
            print(f"  {n:18s} {t}")
    print(f"  fps               {fps}")

    if rb_pos is None:
        print("ERROR: no rigid_body_pos in motion file")
        sys.exit(1)

    rb_pos_np = rb_pos.cpu().numpy() if isinstance(rb_pos, torch.Tensor) else np.asarray(rb_pos)

    print("\nNumeric sanity:")
    print(f"  Any NaN in rb_pos? {np.isnan(rb_pos_np).any()}")
    print(f"  Any Inf in rb_pos? {np.isinf(rb_pos_np).any()}")
    print(f"  rb_pos X range: [{rb_pos_np[..., 0].min():.3f}, {rb_pos_np[..., 0].max():.3f}]")
    print(f"  rb_pos Y range: [{rb_pos_np[..., 1].min():.3f}, {rb_pos_np[..., 1].max():.3f}]")
    print(f"  rb_pos Z range: [{rb_pos_np[..., 2].min():.3f}, {rb_pos_np[..., 2].max():.3f}]")
    if dof_pos is not None and isinstance(dof_pos, torch.Tensor):
        d = dof_pos.cpu().numpy()
        print(f"  dof_pos abs max:  {np.abs(d).max():.3f}  (expect ~< pi for hinges)")
        print(f"  dof_pos NaN:      {np.isnan(d).any()}")

    T, Nb, _ = rb_pos_np.shape
    print(f"\nFrames: {T}, Bodies: {Nb}")

    # Print head/foot trajectory
    head_idx = SMPL_BONES.index("Head") if Nb >= len(SMPL_BONES) else -1
    print("\nFirst & last frame head height:")
    if head_idx >= 0:
        print(f"  frame 0 head pos: {rb_pos_np[0, head_idx]}")
        print(f"  frame {T-1} head pos: {rb_pos_np[-1, head_idx]}")

    # Stick figure plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("\n[matplotlib not installed - skipping plot]")
        return

    print("\nRendering stick-figure preview (plotly = static png)…")
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.cm.viridis
    n_show = min(n_preview, T)
    indices = np.linspace(0, T - 1, n_show).astype(int)
    for k, idx in enumerate(indices):
        color = cmap(k / max(1, n_show - 1))
        pts = rb_pos_np[idx]
        for child, parent in enumerate(SMPL_PARENTS[:Nb]):
            if parent < 0:
                continue
            xs = [pts[parent, 0], pts[child, 0]]
            ys = [pts[parent, 1], pts[child, 1]]
            zs = [pts[parent, 2], pts[child, 2]]
            ax.plot(xs, ys, zs, color=color, linewidth=1.5, alpha=0.8)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=8)

    ax.set_xlabel("X (forward)")
    ax.set_ylabel("Y (left)")
    ax.set_zlabel("Z (up)")
    ax.set_title(f"{motion_path.name} - {n_show} frames spanning {T/fps:.2f}s")

    # Equal aspect
    rng = np.array([
        rb_pos_np[..., 0].max() - rb_pos_np[..., 0].min(),
        rb_pos_np[..., 1].max() - rb_pos_np[..., 1].min(),
        rb_pos_np[..., 2].max() - rb_pos_np[..., 2].min(),
    ]).max()
    cx = (rb_pos_np[..., 0].max() + rb_pos_np[..., 0].min()) * 0.5
    cy = (rb_pos_np[..., 1].max() + rb_pos_np[..., 1].min()) * 0.5
    cz = (rb_pos_np[..., 2].max() + rb_pos_np[..., 2].min()) * 0.5
    rng = max(rng, 1.0) * 0.6
    ax.set_xlim(cx - rng, cx + rng)
    ax.set_ylim(cy - rng, cy + rng)
    ax.set_zlim(cz - rng, cz + rng)

    out_png = motion_path.with_suffix(".preview.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"  Saved {out_png}")

    if save_gif is not None:
        try:
            import imageio
        except ImportError:
            print("  imageio not installed - skipping gif")
            return
        print(f"  Building gif at {save_gif}…")
        frames = []
        for t in range(T):
            fig2 = plt.figure(figsize=(6, 6))
            ax2 = fig2.add_subplot(111, projection="3d")
            pts = rb_pos_np[t]
            for child, parent in enumerate(SMPL_PARENTS[:Nb]):
                if parent < 0:
                    continue
                ax2.plot(
                    [pts[parent, 0], pts[child, 0]],
                    [pts[parent, 1], pts[child, 1]],
                    [pts[parent, 2], pts[child, 2]],
                    color="C0",
                    linewidth=2,
                )
            ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="C1", s=12)
            ax2.set_xlim(cx - rng, cx + rng)
            ax2.set_ylim(cy - rng, cy + rng)
            ax2.set_zlim(cz - rng, cz + rng)
            ax2.set_title(f"t={t}/{T-1}")
            ax2.view_init(elev=15, azim=-60)
            fig2.canvas.draw()
            img = np.asarray(fig2.canvas.buffer_rgba())[..., :3]
            frames.append(img)
            plt.close(fig2)
        imageio.mimsave(str(save_gif), frames, fps=fps)
        print(f"  Saved {save_gif}")


if __name__ == "__main__":
    app()
