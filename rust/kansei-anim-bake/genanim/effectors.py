"""End-effector constraints for Kimodo from a clip's `effectors` in a prompt set: hands on a bar or
a ledge, feet on a wall, placed by geometry so the generated motion touches what the level holds.

Kimodo pins an end effector at a frame to a whole pose: the joint chain's positions and its base
joint's rotation, and with them the root's height, ground position and heading. So each
constraint here is a small pose built on the model's own skeleton (rest pose turned by a few joint
rotations, e.g. arms overhead), placed so its hands or feet land on the target: no captured pose
goes in (never Epic GASP data: see README.md).

Kimodo's space (`paths.py`): Y up, the character facing +Z, its left +X, the first frame's root at
(0, 0) on the ground. Kinds:

- `bar_hang`: both hands on a bar `height` m up along X, `grip` m apart, the body hanging under it.
- `bar_swing`: the same, the body swinging about the bar, `amplitude` radians, `period` seconds.
  Either holds only the first `until` seconds when given (a pull-up from the hang).
- `shimmy`: hanging from a ledge at `height` (its face at z = 0), the hands stepping `step` m to
  the character's left (`direction` 1) or right (-1) every `step_time` seconds.
- `ledge_hang`: hanging still from a ledge at `height` (its face at z = 0), hands on its edge,
  the feet braced on the face (`wall`, the feet's z), the hips under the hands.
- `wall_run`: running along +Z at `speed`, a wall on the right (`side` -1) or left (1) at
  `distance` m, the body leaning `lean` radians away from it; from `start` s for `time` s the feet
  step on the wall every `step_time` s, the root rising by `rise` m and back.
"""

import math

import torch

from kimodo.constraints import EndEffectorConstraintSet


def rotation(axis, angle):
    """A rotation matrix about a unit axis (x, y, z)."""
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return torch.tensor(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=torch.float32,
    )


class Poser:
    """Poses on the model's skeleton: local rotations by joint name, forward kinematics."""

    def __init__(self, skeleton):
        self.skeleton = skeleton
        self.names = list(skeleton.bone_order_names)
        # (the model's skeleton lives on its device)
        buffers = list(skeleton.buffers()) if hasattr(skeleton, "buffers") else []
        self.device = buffers[0].device if buffers else torch.device("cpu")

    def local(self, turns):
        """Local rotation matrices [J, 3, 3]: identity (the rest pose) but for `turns`, a dict of
        joint name to a list of (axis, angle) applied in order."""
        rots = torch.eye(3).repeat(len(self.names), 1, 1)
        for name, steps in turns.items():
            r = torch.eye(3)
            for axis, angle in steps:
                r = r @ rotation(axis, angle)
            rots[self.names.index(name)] = r
        return rots.to(self.device)

    def fk(self, local, root):
        """Global rotations [J, 3, 3] and positions [J, 3] with the root at `root`."""
        g, p, _ = self.skeleton.fk(local[None], torch.tensor([root], dtype=torch.float32, device=self.device))
        return g[0], p[0]

    def index(self, name):
        return self.names.index(name)


def arms_up(forward=0.25):
    """Both arms overhead, a little forward (the rest pose is a T-pose facing +Z, left +X)."""
    return {
        "LeftArm": [((0.0, 0.0, 1.0), math.radians(80)), ((1.0, 0.0, 0.0), -forward)],
        "RightArm": [((0.0, 0.0, 1.0), -math.radians(80)), ((1.0, 0.0, 0.0), -forward)],
        "LeftForeArm": [((0.0, 0.0, 1.0), math.radians(8))],
        "RightForeArm": [((0.0, 0.0, 1.0), -math.radians(8))],
    }


def placed(poser, turns, joint_targets):
    """The pose `turns` with its root moved so the mean of `joint_targets` (name -> target) lands
    on the mean target: (global rotations, positions)."""
    local = poser.local(turns)
    _, p = poser.fk(local, [0.0, 0.0, 0.0])
    have = torch.stack([p[poser.index(n)] for n in joint_targets]).mean(0)
    want = torch.tensor([list(t) for t in joint_targets.values()], dtype=torch.float32, device=p.device).mean(0)
    return poser.fk(local, (want - have).tolist())


def constraint(skeleton, frames, poses, joint):
    """An end-effector constraint on `joint` (LeftHand, RightHand, LeftFoot, RightFoot) at
    `frames`, from `poses` (global rotations, positions) one a frame."""
    g = torch.stack([q[0] for q in poses])
    p = torch.stack([q[1] for q in poses])
    # (frame and joint indices on the CPU, poses on the model's device, as `from_dict` makes them)
    return EndEffectorConstraintSet(skeleton, torch.tensor(frames), p, g, None, joint_names=[joint])


def build(skeleton, spec, frames, fps):
    """The constraints for a clip of `frames` frames from its `effectors` spec."""
    poser = Poser(skeleton)
    kind = spec["kind"]
    every = spec.get("every", 3)
    out = []
    if kind in ("bar_hang", "bar_swing"):
        height, grip = spec.get("height", 2.3), spec.get("grip", 0.5)
        amplitude, period = spec.get("amplitude", 0.0 if kind == "bar_hang" else 0.6), spec.get("period", 2.2)
        # (`until`: only the first seconds held, e.g. a pull-up that leaves the bar's hang)
        keys = list(range(0, min(frames, int(spec.get("until", frames / fps) * fps)), every))
        poses = []
        for f in keys:
            theta = amplitude * math.sin(2.0 * math.pi * f / fps / period)
            turns = arms_up()
            # the whole body about the bar (X): forward of it (+Z) when theta > 0
            turns["Hips"] = [((1.0, 0.0, 0.0), -theta)]
            targets = {"LeftHand": (grip / 2, height, 0.0), "RightHand": (-grip / 2, height, 0.0)}
            poses.append(placed(poser, turns, targets))
        for joint in ("LeftHand", "RightHand", "Hips"):
            out.append(constraint(skeleton, keys, poses, joint))
    elif kind == "shimmy":
        height, grip = spec.get("height", 2.3), spec.get("grip", 0.5)
        step, step_time, direction = spec.get("step", 0.3), spec.get("step_time", 0.9), spec.get("direction", 1.0)
        start = spec.get("start", 0.5)
        # each hand planted between its steps; the leading hand steps first, the other after half
        # a step time; the root follows their midpoint
        lead, trail = ("LeftHand", "RightHand") if direction > 0 else ("RightHand", "LeftHand")
        x0 = {"LeftHand": grip / 2, "RightHand": -grip / 2}

        def hand_x(hand, t):
            delay = 0.0 if hand == lead else step_time * 0.5
            k = max(0.0, (t - start - delay) / step_time)
            n, frac = math.floor(k), k - math.floor(k)
            # a step takes the first 40% of each step time
            moving = min(1.0, frac / 0.4)
            moving = moving * moving * (3 - 2 * moving)
            return x0[hand] + direction * step * (n + moving)

        for hand in (lead, trail):
            keys, poses = [], []
            for f in range(0, frames, every):
                t = f / fps
                delay = 0.0 if hand == lead else step_time * 0.5
                k = max(0.0, (t - start - delay) / step_time)
                if k - math.floor(k) < 0.45 and t > start:
                    continue  # stepping: free
                targets = {"LeftHand": (hand_x("LeftHand", t), height, 0.05), "RightHand": (hand_x("RightHand", t), height, 0.05)}
                keys.append(f)
                poses.append(placed(poser, arms_up(0.45), targets))
            out.append(constraint(skeleton, keys, poses, hand))
        # the hips hanging under the hands' midpoint on every key (no floor to stand on)
        keys = list(range(0, frames, every))
        poses = []
        for f in keys:
            t = f / fps
            targets = {"LeftHand": (hand_x("LeftHand", t), height, 0.05), "RightHand": (hand_x("RightHand", t), height, 0.05)}
            poses.append(placed(poser, arms_up(0.45), targets))
        out.append(constraint(skeleton, keys, poses, "Hips"))
    elif kind == "wall_run":
        speed, side, lean = spec.get("speed", 5.0), spec.get("side", -1.0), spec.get("lean", 0.8)
        start, time, step_time, rise = spec.get("start", 0.6), spec.get("time", 1.0), spec.get("step_time", 0.2), spec.get("rise", 0.9)
        # the body leaning away from the wall (about +Z): up tilts toward -side
        turns = {"Hips": [((0.0, 0.0, 1.0), lean * side)], **arms_down()}
        local = poser.local(turns)
        _, rest = poser.fk(local, [0.0, 0.0, 0.0])
        feet = {"LeftFoot": rest[poser.index("LeftFoot")], "RightFoot": rest[poser.index("RightFoot")]}
        # the wall where the outer foot is
        wall_x = spec.get("distance", None)
        step = 0
        t = start
        while t < start + time - 1e-6:
            foot = "RightFoot" if (step % 2 == 0) == (side < 0) else "LeftFoot"
            u = (t - start) / time
            y = rise * 4.0 * u * (1.0 - u)
            z = speed * t
            f = int(round(t * fps))
            if f < frames:
                # the root so this foot is on the wall at this height and distance along
                fx = feet[foot][0].item()
                x = (wall_x if wall_x is not None else side * 0.8) - fx
                # the hips at standing height, raised by the run's rise
                pose = poser.fk(local, [x, 1.0 + y, z])
                out.append(constraint(skeleton, [f], [pose], foot))
            step += 1
            t += step_time
        # the hips leaning away from the wall all along the run (every key)
        keys, poses = [], []
        for f in range(int(round(start * fps)), min(frames, int(round((start + time) * fps)) + 1), every):
            t = f / fps
            u = (t - start) / time
            y = rise * 4.0 * u * (1.0 - u)
            fx = (feet["RightFoot"][0].item() + feet["LeftFoot"][0].item()) * 0.5
            x = (wall_x if wall_x is not None else side * 0.8) - fx
            keys.append(f)
            poses.append(poser.fk(local, [x, 1.0 + y, speed * t]))
        out.append(constraint(skeleton, keys, poses, "Hips"))
    elif kind == "ledge_hang":
        # hanging still from a ledge, the feet braced on the face under it (z = 0 the face, the
        # body out toward -z): hands on the top's edge, the hips under them, the feet on the wall
        height, grip, wall = spec.get("height", 2.3), spec.get("grip", 0.5), spec.get("wall", -0.08)
        hands = {"LeftHand": (grip / 2, height, 0.04), "RightHand": (-grip / 2, height, 0.04)}

        def braced(flex):
            turns = arms_up(0.35)
            # the body leaning in from the hips, the hips out from the face under the hands
            turns["Hips"] = [((1.0, 0.0, 0.0), spec.get("lean", 0.36))]
            for side in ("Left", "Right"):
                turns[f"{side}Leg"] = [((1.0, 0.0, 0.0), -flex)]
                turns[f"{side}Shin"] = [((1.0, 0.0, 0.0), flex * 1.2)]
            return placed(poser, turns, hands)

        # the hip flex that puts the feet on the wall
        best = min((abs(braced(f)[1][poser.index("LeftFoot")][2].item() - wall), f) for f in [k * 0.05 for k in range(4, 30)])[1]
        pose = braced(best)
        keys = list(range(0, frames, every))
        for joint in ("LeftHand", "RightHand", "LeftFoot", "RightFoot", "Hips"):
            out.append(constraint(skeleton, keys, [pose] * len(keys), joint))
    else:
        raise ValueError(f"unknown effectors kind {kind}")
    return out


def arms_down():
    """Arms down at the sides from the T-pose, for running."""
    return {
        "LeftArm": [((0.0, 0.0, 1.0), -math.radians(70))],
        "RightArm": [((0.0, 0.0, 1.0), math.radians(70))],
        "LeftForeArm": [((1.0, 0.0, 0.0), -math.radians(60))],
        "RightForeArm": [((1.0, 0.0, 0.0), -math.radians(60))],
    }
