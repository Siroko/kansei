"""Inside Blender, headless: every `.fbx` in a folder to a `.glb` (skeleton and animation, sampled
every frame), for `mixamo_to_gltf.py`.

    Blender -b --factory-startup -P fbx_to_glb.py -- <fbx folder> <glb folder>

Blender's own FBX importer and glTF exporter do the axis and unit conversion: the glTF is Y up,
the character facing +Z, its armature node carrying the FBX's centimetre scale. Bone orientations
are kept as authored (no automatic bone orientation), so a joint's rest rotation is the FBX's.
"""

import os
import sys

import bpy


def convert(fbx, glb):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx, automatic_bone_orientation=False, ignore_leaf_bones=False)
    scene = bpy.context.scene
    actions = [a for a in bpy.data.actions]
    if actions:
        start, end = actions[0].frame_range
        scene.frame_start, scene.frame_end = int(round(start)), int(round(end))
    bpy.ops.export_scene.gltf(
        filepath=glb,
        export_format="GLB",
        export_yup=True,
        export_animations=True,
        export_force_sampling=True,
        export_frame_step=1,
        export_anim_single_armature=True,
        export_materials="NONE",
    )


def main():
    args = sys.argv[sys.argv.index("--") + 1 :]
    source, out = args[0], args[1]
    os.makedirs(out, exist_ok=True)
    for name in sorted(os.listdir(source)):
        if not name.lower().endswith(".fbx"):
            continue
        glb = os.path.join(out, os.path.splitext(name)[0] + ".glb")
        convert(os.path.join(source, name), glb)
        print("kansei fbx_to_glb:", name, "->", glb)


main()
