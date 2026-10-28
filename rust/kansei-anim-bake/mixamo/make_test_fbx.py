"""Inside Blender, headless: a synthetic Mixamo-like FBX for `tests.py` (no Mixamo data).

    Blender -b --factory-startup -P make_test_fbx.py -- <clip.fbx> <tpose.fbx>

A `mixamorig:` skeleton in a T-pose, facing -Y in Blender (front view), as Mixamo's imports do.
Over 31 frames at 30 fps its hips move 2 m forward and 0.3 m up while the body turns 90° to its
left, and the left arm swings down. The clip is written on its last frame, so (with no skin, no
bind pose) its file's rest is that pose, not the T-pose: as a Without Skin download's may be. The
second file is the T-pose alone, standing in for the With Skin T-pose download.
"""

import math
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

P = "mixamorig:"


def joints():
    """(name, parent, head) of a Mixamo-like T-pose, metres, Z up, facing -Y, its left +X."""
    out = [("Hips", None, (0, 0, 1.0)), ("Spine", "Hips", (0, 0, 1.1)), ("Spine1", "Spine", (0, 0, 1.22)), ("Spine2", "Spine1", (0, 0, 1.35)),
           ("Neck", "Spine2", (0, 0, 1.5)), ("Head", "Neck", (0, 0, 1.6)), ("HeadTop_End", "Head", (0, 0, 1.8))]
    for side, sx in (("Left", 1), ("Right", -1)):
        out += [(f"{side}Shoulder", "Spine2", (0.05 * sx, 0, 1.45)), (f"{side}Arm", f"{side}Shoulder", (0.18 * sx, 0, 1.45)),
                (f"{side}ForeArm", f"{side}Arm", (0.45 * sx, 0, 1.45)), (f"{side}Hand", f"{side}ForeArm", (0.7 * sx, 0, 1.45))]
        for k, (finger, dy) in enumerate((("Thumb", -0.04), ("Index", -0.02), ("Middle", 0.0), ("Ring", 0.02), ("Pinky", 0.04))):
            parent = f"{side}Hand"
            for i in range(1, 5):
                name = f"{side}Hand{finger}{i}"
                out.append((name, parent, ((0.75 + 0.03 * i) * sx, dy, 1.45)))
                parent = name
        out += [(f"{side}UpLeg", "Hips", (0.1 * sx, 0, 0.95)), (f"{side}Leg", f"{side}UpLeg", (0.1 * sx, 0, 0.5)),
                (f"{side}Foot", f"{side}Leg", (0.1 * sx, 0, 0.08)), (f"{side}ToeBase", f"{side}Foot", (0.1 * sx, -0.12, 0.02)),
                (f"{side}Toe_End", f"{side}ToeBase", (0.1 * sx, -0.2, 0.02))]
    return out


def main():
    out, tpose = sys.argv[sys.argv.index("--") + 1 :][:2]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 30
    data = bpy.data.armatures.new("Armature")
    arm = bpy.data.objects.new("Armature", data)
    scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    js = joints()
    children = {}
    for name, parent, _ in js:
        children.setdefault(parent, []).append(name)
    heads = {name: Vector(h) for name, _, h in js}
    for name, parent, head in js:
        b = data.edit_bones.new(P + name)
        b.head = Vector(head)
        kids = children.get(name)
        tail = heads[kids[0]] if kids else Vector(head) + (Vector(head) - heads[parent]).normalized() * 0.05
        if (tail - b.head).length < 1e-4:
            tail = b.head + Vector((0, 0, 0.05))
        b.tail = tail
        if parent:
            b.parent = data.edit_bones[P + parent]
    bpy.ops.object.mode_set(mode="POSE")
    hips = arm.pose.bones[P + "Hips"]
    arm_l = arm.pose.bones[P + "LeftArm"]
    rest = hips.bone.matrix_local.copy()
    for f in range(31):
        u = f / 30.0
        scene.frame_set(f + 1)
        turn = Quaternion((0, 0, 1), math.radians(90) * u)
        world = Matrix.Translation(Vector((0, -2.0 * u, 0.3 * u))) @ Matrix.Translation(rest.translation) @ turn.to_matrix().to_4x4() @ Matrix.Translation(-rest.translation) @ rest
        hips.matrix = world
        hips.keyframe_insert("location", frame=f + 1)
        hips.keyframe_insert("rotation_quaternion", frame=f + 1)
        # down by 70° about the armature's +Y (in the bone's own frame)
        q = arm_l.bone.matrix_local.to_quaternion()
        arm_l.rotation_quaternion = q.inverted() @ Quaternion((0, 1, 0), math.radians(70) * u) @ q
        arm_l.keyframe_insert("rotation_quaternion", frame=f + 1)
        bpy.context.view_layer.update()
    scene.frame_start, scene.frame_end = 1, 31
    bpy.ops.object.mode_set(mode="OBJECT")
    scene.frame_set(31)
    bpy.ops.export_scene.fbx(filepath=out, object_types={"ARMATURE"}, add_leaf_bones=False, bake_anim=True, bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False)
    # the T-pose: no animation, every bone at rest
    arm.animation_data_clear()
    bpy.ops.object.mode_set(mode="POSE")
    for pb in arm.pose.bones:
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()
    bpy.ops.export_scene.fbx(filepath=tpose, object_types={"ARMATURE"}, add_leaf_bones=False, bake_anim=False)
    print("kansei make_test_fbx:", out, tpose)


main()
