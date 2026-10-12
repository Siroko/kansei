# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.23"]
# ///
"""`uv run tests.py <rig mesh.glb>`: the Mixamo route's conventions on a synthetic clip.

Needs Blender (`BLENDER`, default /Applications/Blender.app/Contents/MacOS/Blender): it builds a
Mixamo-like FBX whose stored rest is a posed frame, plus a T-pose FBX (`make_test_fbx.py`: hips 2 m
forward and 0.3 m up while turning 90° left, the left arm swinging down), converts them (`fbx_to_glb.py`) and retargets it onto the rig. No Mixamo
data is used.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "genanim"))
sys.path.insert(0, str(HERE))
import gltf_io  # noqa: E402
import quat  # noqa: E402
from mixamo_to_gltf import FPS, Clip, source_skeleton, tracks  # noqa: E402
from retarget import Retarget  # noqa: E402

import json  # noqa: E402

BLENDER = os.environ.get("BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender")
RIG = sys.argv.pop(1) if len(sys.argv) > 1 else None


def blender(script, *args):
    subprocess.run([BLENDER, "-b", "--factory-startup", "-P", str(HERE / script), "--", *map(str, args)], check=True, capture_output=True)


class Route(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        (tmp / "fbx").mkdir()
        blender("make_test_fbx.py", tmp / "fbx" / "Test.fbx", tmp / "fbx" / "TPose.fbx")
        blender("fbx_to_glb.py", tmp / "fbx", tmp / "glb")
        cls.clip = Clip(tmp / "glb" / "Test.glb")
        cls.tpose = Clip(tmp / "glb" / "TPose.glb")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_metres_y_up_facing_z(self):
        local_t, local_r = self.clip.folded(self.tpose.rest_t, self.tpose.rest_r)
        g, _ = quat.forward_kinematics(self.clip.parents, local_t, local_r)
        n = self.clip.names
        self.assertAlmostEqual(g[n.index("Hips"), 1], 1.0, delta=0.02)
        self.assertAlmostEqual(g[n.index("HeadTop_End"), 1], 1.8, delta=0.02)
        # its left +X, the toes ahead (+Z)
        self.assertGreater(g[n.index("LeftHand"), 0], 0.6)
        self.assertGreater(g[n.index("LeftToe_End"), 2], 0.15)

    def test_root_follows_the_hips_on_the_floor_and_turns_with_the_body(self):
        rotations, root_t, hips_local, global_t = tracks(self.clip, 0.1)
        self.assertTrue(np.allclose(root_t[:, 1], 0.0))
        self.assertAlmostEqual(root_t[-1, 2] - root_t[0, 2], 2.0, delta=0.15)
        self.assertAlmostEqual(abs(root_t[-1, 0] - root_t[0, 0]), 0.0, delta=0.05)
        heading = lambda q: 2 * np.arctan2(q[1], q[3])
        self.assertAlmostEqual(heading(rotations[0, 0]), 0.0, delta=0.05)
        self.assertAlmostEqual(heading(rotations[-1, 0]), np.pi / 2, delta=0.12)
        # the hips under the root: over it, at their height
        self.assertAlmostEqual(hips_local[0, 1], 1.0, delta=0.02)
        self.assertAlmostEqual(hips_local[-1, 1], 1.3, delta=0.02)
        self.assertLess(np.abs(hips_local[:, [0, 2]]).max(), 0.1)
        self.assertEqual(len(rotations), 31)

    @unittest.skipUnless(RIG, "pass the rig's mesh.glb to test the retarget")
    def test_retarget_onto_the_rig(self):
        target = gltf_io.read_skeleton(RIG)
        mapping = json.loads((HERE / "maps" / "mixamo_to_ue5_mannequin.json").read_text())
        source = source_skeleton(self.clip, self.tpose)
        r = Retarget(source, target, mapping, root=("root", "root"), hips=("Hips", "pelvis"))
        rotations, root_t, hips_local, _ = tracks(self.clip, 0.1)
        rot, trans = r.apply(rotations, root_t, hips_local)
        local_t = np.broadcast_to(target.translations, (len(rot),) + target.translations.shape).copy()
        for j, values in trans.items():
            local_t[:, j] = values
        g, _ = quat.forward_kinematics(target.parents, local_t, rot)
        t = target.names
        rest_g, _ = quat.forward_kinematics(target.parents, target.translations, target.rotations)
        pelvis_rest = rest_g[t.index("pelvis"), 1]
        self.assertAlmostEqual(g[0, t.index("pelvis"), 1], pelvis_rest, delta=0.03)
        # a T-pose: the left hand out to the left (+X) at the shoulders' height
        hand, shoulder = g[0, t.index("hand_l")], g[0, t.index("upperarm_l")]
        self.assertGreater(hand[0] - shoulder[0], 0.4)
        self.assertAlmostEqual(hand[1], shoulder[1], delta=0.08)
        # the arm swung down by the end, as the source's did; the root went forward, scaled to the rig
        _, _, _, src = tracks(self.clip, 0.1)
        n = self.clip.names
        drop = (src[-1, n.index("LeftArm"), 1] - src[-1, n.index("LeftHand"), 1]) / np.linalg.norm(src[-1, n.index("LeftHand")] - src[-1, n.index("LeftArm")])
        rig_drop = (g[-1, t.index("upperarm_l"), 1] - g[-1, t.index("hand_l"), 1]) / np.linalg.norm(g[-1, t.index("hand_l")] - g[-1, t.index("upperarm_l")])
        self.assertGreater(drop, 0.8)
        self.assertAlmostEqual(rig_drop, drop, delta=0.1)
        self.assertAlmostEqual(root_t[-1, 2] * r.scale, g[-1, t.index("root"), 2], delta=0.01)
        print(f"\n  rig pelvis {pelvis_rest:.3f} m, scale {r.scale:.3f}, frames {len(rot)} at {FPS} fps")


if __name__ == "__main__":
    unittest.main()
