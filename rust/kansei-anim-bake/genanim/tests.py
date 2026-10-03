# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.23"]
# ///
"""`uv run tests.py`: the path, quaternion, loop and retargeting conventions the converter relies on."""

import math
import unittest

import numpy as np

import paths
import quat
from gltf_io import Skeleton
from kimodo_to_gltf import close_loop
from retarget import Retarget


class Paths(unittest.TestCase):
    def test_a_leg_reaches_its_pace_and_heading(self):
        pos, face, speed = paths.sample([dict(time=1.0, speed=[2.0, 2.0]), dict(time=1.0, dir=90, facing=90)])
        self.assertEqual(len(pos), 60)
        self.assertAlmostEqual(speed[-1], 2.0)
        # 90 degrees turns from +Z to +X, the character's left
        step = pos[-1] - pos[-2]
        self.assertGreater(step[0], 0.06)
        self.assertAlmostEqual(step[1], 0.0, places=3)
        self.assertAlmostEqual(face[-1], math.pi / 2)

    def test_root2d_heading_is_cos_sin(self):
        c = paths.root2d_constraint([dict(time=1.0, facing=90)], 30)
        self.assertEqual(c["frame_indices"][-1], 29)
        cos, sin = c["global_root_heading"][-1]
        self.assertAlmostEqual(cos, 0.0, places=6)
        self.assertAlmostEqual(sin, 1.0, places=6)


class Quaternions(unittest.TestCase):
    def test_yaw_turns_forward_to_its_heading(self):
        v = quat.rotate(quat.yaw(math.pi / 2), np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(v, [1.0, 0.0, 0.0], atol=1e-9)

    def test_matrix_round_trip(self):
        q = quat.normalize(np.array([[0.3, -0.2, 0.5, 0.8], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]]))
        back = quat.from_matrix(quat.to_matrix(q))
        np.testing.assert_allclose(np.abs(np.sum(back * q, -1)), 1.0, atol=1e-9)


class Loops(unittest.TestCase):
    def test_a_closed_loop_ends_on_its_first_pose(self):
        n, joints = 20, 3
        angles = np.linspace(0, 0.4, n)
        rotations = np.zeros((n, joints, 4))
        rotations[..., 3] = 1
        rotations[:, 1] = quat.yaw(angles)
        hips = np.stack([np.zeros(n), 0.9 + 0.01 * np.arange(n), np.zeros(n)], -1)
        out, _, hips_out, bent = close_loop(rotations, np.zeros((n, 3)), hips)
        np.testing.assert_allclose(out[-1], out[0])
        np.testing.assert_allclose(hips_out[-1], hips_out[0])
        self.assertAlmostEqual(bent, math.degrees(0.4), places=4)


class Retargeting(unittest.TestCase):
    def test_a_t_pose_arm_drives_an_a_pose_arm(self):
        # root > hips > arm > hand: the source's arm points +X (T-pose), the target's down 45 degrees
        source = Skeleton(["root", "Hips", "Arm", "Hand"], [-1, 0, 1, 2], [[0, 0, 0], [0, 1, 0], [0.2, 0.4, 0], [0.5, 0, 0]])
        down = np.array([math.cos(math.radians(45)), -math.sin(math.radians(45)), 0]) * 0.5
        target = Skeleton(["root", "pelvis", "arm", "hand"], [-1, 0, 1, 2], [[0, 0, 0], [0, 0.9, 0], [0.2, 0.4, 0], down])
        mapping = {"joints": {"root": "root", "Hips": "pelvis", "Arm": "arm", "Hand": "hand"}, "aim": {"Arm": ["Hand", "hand"]}}
        r = Retarget(source, target, mapping)
        identity = np.tile([0.0, 0, 0, 1], (1, 4, 1))
        # the source in its T-pose puts the target in a T-pose: its arm level
        local, translations = r.apply(identity, np.zeros((1, 3)), np.array([[0, 1, 0]]))
        t, _ = quat.forward_kinematics(target.parents, target.translations, local[0])
        np.testing.assert_allclose((t[3] - t[2]) / 0.5, [1.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(translations[1][0], [0, 0.9, 0])
        # the source's arm lowered 45 degrees (about -Z) is the target's rest pose
        lowered = identity.copy()
        lowered[0, 2] = [0, 0, -math.sin(math.radians(22.5)), math.cos(math.radians(22.5))]
        local, _ = r.apply(lowered, np.zeros((1, 3)), np.array([[0, 1, 0]]))
        np.testing.assert_allclose(np.abs(np.sum(local[0] * target.rotations, -1)), 1.0, atol=1e-9)

if __name__ == "__main__":
    unittest.main()
