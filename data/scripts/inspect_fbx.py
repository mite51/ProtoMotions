# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Quick FBX inspector. Run via Blender's bundled Python:

    blender --background \
        --python data/scripts/inspect_fbx.py -- <path/to.fbx>
"""
import sys
import bpy

argv = sys.argv
argv = argv[argv.index("--") + 1 :] if "--" in argv else []
if not argv:
    print("Usage: blender --background --python inspect_fbx.py -- <path/to.fbx>")
    sys.exit(1)
fbx_path = argv[0]

bpy.ops.wm.read_factory_settings(use_empty=True)
print(f"\n=== Importing FBX: {fbx_path} ===")
bpy.ops.import_scene.fbx(filepath=fbx_path)

print("\n=== Scene objects ===")
for obj in bpy.data.objects:
    print(f"  {obj.type:10s} {obj.name}")

print("\n=== Armatures ===")
for arm in bpy.data.armatures:
    print(f"  Armature: {arm.name}, {len(arm.bones)} bones")
    for b in arm.bones:
        parent = b.parent.name if b.parent else "(root)"
        print(f"    {b.name:40s} parent={parent}")

print("\n=== Actions (animations / takes) ===")
for act in bpy.data.actions:
    fr = act.frame_range
    print(f"  Action: {act.name}, frames=[{fr[0]:.1f}, {fr[1]:.1f}], "
          f"fcurves={len(act.fcurves)}")

print("\n=== Scene framerate ===")
print(f"  fps={bpy.context.scene.render.fps}, "
      f"fps_base={bpy.context.scene.render.fps_base}")
