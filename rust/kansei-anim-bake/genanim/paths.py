"""Root paths for Kimodo's `root2d` constraint, from a clip's `legs` in a prompt set.

A clip's ground path is a list of legs, each `time` seconds long. Over a leg the character eases
(smoothstep) from where the last leg left it to the leg's targets:

- `speed`: metres per second along the ground;
- `dir`: the direction it travels, in degrees (0 is +Z, 90 is +X: the character's left when it
  faces +Z, so positive angles turn left);
- `facing`: the direction its body faces, in the same degrees.

Omitted values keep the last leg's (the first leg starts from speed 0, dir 0, facing 0). `speed`
may be `[from, to]` to jump to `from` at the leg's start (a loop at its pace from the first frame).
`"ease": "linear"` changes at a constant rate instead (a circle). A start is a leg at speed 0 then
one easing up to the pace; a strafe travels at 90 while facing 0; a turn in place changes `facing`
at speed 0.

Kimodo's space: Y up, +Z forward, the smoothed root at (0, 0) on the first frame, and a heading
angle h whose forward is (sin h, cos h) on (x, z), stored as (cos h, sin h). Only numpy is needed,
so this runs on the generator and on the Mac alike.
"""

import math

import numpy as np

FPS = 30


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def sample(legs, fps=FPS):
    """Per frame: (positions [T, 2] as (x, z), facing [T] in radians, speed [T])."""
    speed, direction, facing = 0.0, 0.0, 0.0
    positions, facings, speeds = [], [], []
    p = np.zeros(2)
    first = True
    for leg in legs:
        n = max(1, round(leg["time"] * fps))
        s1 = leg.get("speed", speed)
        s0, s1 = (s1[0], s1[1]) if isinstance(s1, list) else (speed, s1)
        d0, d1 = direction, leg.get("dir", direction)
        f0, f1 = facing, leg.get("facing", facing)
        for i in range(n):
            if first:
                # the first frame sits at the origin
                first = False
                positions.append(p.copy())
                facings.append(math.radians(f0))
                speeds.append(s0)
                continue
            t = (i + 1) / n
            k = t if leg.get("ease") == "linear" else smoothstep(t)
            v = s0 + (s1 - s0) * k
            d = math.radians(d0 + (d1 - d0) * k)
            p = p + v / fps * np.array([math.sin(d), math.cos(d)])
            positions.append(p.copy())
            facings.append(math.radians(f0 + (f1 - f0) * k))
            speeds.append(v)
        speed, direction, facing = s1, d1, f1
    return np.array(positions), np.array(facings), np.array(speeds)


def duration(legs):
    return sum(leg["time"] for leg in legs)


def root2d_constraint(legs, frames, every=1, heading=True, fps=FPS):
    """Kimodo's `root2d` constraint dict for the path, on every `every`-th frame (and the last).

    `frames` is the clip's frame count: the path is cut or held to it.
    """
    pos, face, _ = sample(legs, fps)
    if len(pos) < frames:
        pad = frames - len(pos)
        pos = np.concatenate([pos, np.repeat(pos[-1:], pad, 0)])
        face = np.concatenate([face, np.repeat(face[-1:], pad)])
    indices = list(range(0, frames, every))
    if indices[-1] != frames - 1:
        indices.append(frames - 1)
    constraint = {
        "type": "root2d",
        "frame_indices": indices,
        "smooth_root_2d": [[float(pos[i, 0]), float(pos[i, 1])] for i in indices],
    }
    if heading:
        constraint["global_root_heading"] = [[math.cos(face[i]), math.sin(face[i])] for i in indices]
    return constraint
