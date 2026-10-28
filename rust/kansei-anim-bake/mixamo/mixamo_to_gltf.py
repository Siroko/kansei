# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.23"]
# ///
"""Mixamo clips (as `.glb`, from `fbx_to_glb.py`) onto another skeleton, with a ground root.

    uv run mixamo_to_gltf.py --target <rig mesh.glb> --map maps/mixamo_to_ue5_mannequin.json \\
        [--rest YBot_TPose.glb] [--prefix Mixamo/] [--loop 'Hang_Idle_*,Shimmy_*'] <out> <glb folder>

Mixamo's skeleton has no root joint: `mixamorig:Hips` is the top joint and carries the travel,
under an armature node holding the FBX's axes and centimetre scale. Here:

- the armature node is folded into the hips (its scale into every bone's translation, its turn
  into the hips' rotation), so the skeleton is in metres, Y up, facing +Z;
- a ground joint `root` goes above the hips: the hips' path on the floor (y = 0), smoothed over
  `--smooth` seconds, turned to the body's heading (the hips' left-to-right axis, from the
  thighs), so it carries the root motion and faces +Z as the bake wants;
- the clip is resampled at 30 fps and retargeted onto `--target` with `genanim/retarget.py`
  (rest-pose aware: the T-pose source drives the target's own rest pose).

`--rest` takes the rest pose from a file other than each clip's (a With Skin T-pose export, whose
rest is the bind pose). The joint names' `mixamorig:` (or `mixamorig1:`…) prefix is dropped.

Mixamo's terms forbid AI/ML use of its content: these clips go into motion-matching packs only,
never into a generator (Kimodo).
"""

import argparse
import fnmatch
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "genanim"))
import gltf_io  # noqa: E402
import quat  # noqa: E402
from retarget import Retarget  # noqa: E402

FPS = 30
COMPONENTS = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}


def accessor(doc, blob, index):
    a = doc["accessors"][index]
    view = doc["bufferViews"][a["bufferView"]]
    n = COMPONENTS[a["type"]]
    start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = view.get("byteStride", 4 * n)
    raw = np.frombuffer(blob, dtype=np.uint8, count=stride * (a["count"] - 1) + 4 * n, offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, shape=(a["count"], 4 * n), strides=(stride, 1))
    return np.ascontiguousarray(rows).view(np.float32).reshape(a["count"], n).astype(np.float64)


def short(name):
    return name.split(":")[-1]


def trs(node):
    t = np.array(node.get("translation", [0.0, 0.0, 0.0]))
    r = np.array(node.get("rotation", [0.0, 0.0, 0.0, 1.0]))
    s = np.array(node.get("scale", [1.0, 1.0, 1.0]))
    return t, r, s


class Clip:
    """A Mixamo glb: the hips' subtree as joints (parents first), the static transform above the
    hips, the rest pose and the animation sampled at FPS."""

    def __init__(self, path):
        doc, blob = gltf_io.read(path)
        nodes = doc["nodes"]
        self.parent_of = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
        hips = next(i for i, n in enumerate(nodes) if short(n.get("name", "")) == "Hips")
        order = []

        def visit(i):
            order.append(i)
            for c in nodes[i].get("children", []):
                visit(c)

        visit(hips)
        self.order = order
        index = {node: k for k, node in enumerate(order)}
        self.names = [short(nodes[i].get("name", f"node{i}")) for i in order]
        self.parents = [index.get(self.parent_of.get(i), -1) if i != hips else -1 for i in order]
        rest = [trs(nodes[i]) for i in order]
        self.rest_t = np.array([t for t, _, _ in rest])
        self.rest_r = np.array([r for _, r, _ in rest])
        # the nodes above the hips (the armature): one static rotation, uniform scale, translation
        above_t, above_r, above_s = np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), 1.0
        p = self.parent_of.get(hips)
        chain = []
        while p is not None:
            chain.append(nodes[p])
            p = self.parent_of.get(p)
        for node in reversed(chain):
            t, r, s = trs(node)
            above_t = above_t + quat.rotate(above_r, t * above_s)
            above_r = quat.mul(above_r, r)
            above_s = above_s * float(np.mean(s))
        self.above = (above_t, above_r, above_s)
        # the animation, sampled at FPS
        anim = (doc.get("animations") or [{}])[0]
        tracks = {}
        start, end = np.inf, 0.0
        for ch in anim.get("channels", []):
            node, path_ = ch["target"].get("node"), ch["target"]["path"]
            if node not in index or path_ not in ("rotation", "translation"):
                continue
            s = anim["samplers"][ch["sampler"]]
            times = accessor(doc, blob, s["input"])[:, 0]
            values = accessor(doc, blob, s["output"])
            tracks[(index[node], path_)] = (times, values)
            start, end = min(start, float(times[0])), max(end, float(times[-1]))
        # (from the first sample: Blender's importer may start an action a frame in)
        start = 0.0 if not np.isfinite(start) else start
        frames = max(2, int(round((end - start) * FPS)) + 1)
        self.times = start + np.arange(frames) / FPS
        self.local_r = np.broadcast_to(self.rest_r, (frames,) + self.rest_r.shape).copy()
        self.local_t = np.broadcast_to(self.rest_t, (frames,) + self.rest_t.shape).copy()
        for (j, path_), (times, values) in tracks.items():
            if path_ == "translation":
                for c in range(3):
                    self.local_t[:, j, c] = np.interp(self.times, times, values[:, c])
            else:
                values = quat.continuous(values)
                k = np.clip(np.searchsorted(times, self.times, side="right") - 1, 0, len(times) - 1)
                k1 = np.minimum(k + 1, len(times) - 1)
                span = np.where(times[k1] > times[k], times[k1] - times[k], 1.0)
                u = np.clip((self.times - times[k]) / span, 0.0, 1.0)[:, None]
                self.local_r[:, j] = quat.slerp(values[k], values[k1], u)

    def folded(self, local_t, local_r):
        """Local translations and rotations with the armature folded into the hips: metres, Y up."""
        t, r, s = self.above
        local_t = local_t * s
        local_r = local_r.copy()
        local_t[..., 0, :] = t + quat.rotate(r, local_t[..., 0, :])
        local_r[..., 0, :] = quat.mul(r, local_r[..., 0, :])
        return local_t, local_r


def gaussian(x, sigma):
    """x [T, ...] smoothed along axis 0; the ends padded by odd reflection, so a straight path stays
    straight to its ends."""
    if sigma <= 0 or len(x) < 3:
        return x
    radius = min(int(3 * sigma) + 1, len(x) - 1)
    w = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    w /= w.sum()
    head = 2 * x[:1] - x[radius:0:-1]
    tail = 2 * x[-1:] - x[-2 : -radius - 2 : -1]
    padded = np.concatenate([head, x, tail])
    return np.stack([np.tensordot(w, padded[k : k + 2 * radius + 1], axes=(0, 0)) for k in range(len(x))])


def ground_root(names, global_t, smooth):
    """The root's path on the floor and its heading (radians about +Y, 0 facing +Z)."""
    hips = global_t[:, names.index("Hips")]
    left = global_t[:, names.index("LeftUpLeg")] - global_t[:, names.index("RightUpLeg")]
    # facing = left x up
    heading = np.unwrap(np.arctan2(-left[:, 2], left[:, 0]))
    sigma = smooth * FPS
    root_t = np.stack([hips[:, 0], np.zeros(len(hips)), hips[:, 2]], -1)
    return gaussian(root_t, sigma), gaussian(heading, sigma)


def source_skeleton(clip, rest):
    """The source skeleton for the retarget: a ground `root`, then the hips' subtree in metres."""
    rest_t, rest_r = clip.folded(rest.rest_t, rest.rest_r)
    names = ["root"] + clip.names
    parents = [-1] + [p + 1 for p in clip.parents]
    return gltf_io.Skeleton(names, parents, np.vstack([np.zeros(3), rest_t]), np.vstack([[0.0, 0.0, 0.0, 1.0], rest_r]))


def tracks(clip, smooth):
    """Source local rotations [T, J+1, 4] (root first), the root's path and the hips' translation
    under it."""
    local_t, local_r = clip.folded(clip.local_t, clip.local_r)
    global_t, global_r = quat.forward_kinematics(clip.parents, local_t, local_r)
    root_t, heading = ground_root(clip.names, global_t, smooth)
    root_r = quat.yaw(heading)
    inv_root = quat.inv(root_r)
    hips = clip.names.index("Hips")
    rotations = np.concatenate([root_r[:, None], quat.mul(inv_root, global_r[:, hips])[:, None], local_r[:, 1:]], 1)
    hips_local = quat.rotate(inv_root, global_t[:, hips] - root_t)
    return quat.continuous(rotations), root_t, hips_local, global_t


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target", required=True, help="a glTF whose skeleton to retarget onto (the rig's mesh.glb)")
    parser.add_argument("--map", required=True, help="the joint map (maps/*.json)")
    parser.add_argument("--rest", help="a glb whose rest pose to use (the With Skin T-pose); default: each clip's own")
    parser.add_argument("--prefix", default="Mixamo/", help="clip path prefix (default Mixamo/)")
    parser.add_argument("--loop", default="Hang_Idle_*,Shimmy_*", help="comma-separated name patterns of looping clips")
    parser.add_argument("--smooth", type=float, default=0.1, help="the root path's smoothing, seconds (default 0.1)")
    parser.add_argument("out")
    parser.add_argument("glbs")
    args = parser.parse_args()

    target = gltf_io.read_skeleton(args.target)
    mapping = json.loads(Path(args.map).read_text())
    rest = Clip(args.rest) if args.rest else None
    loops = [p.strip() for p in args.loop.split(",") if p.strip()]
    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    manifest, report = [], []
    for path in sorted(Path(args.glbs).glob("*.glb")):
        if args.rest and path.resolve() == Path(args.rest).resolve():
            continue
        clip = Clip(path)
        if len(clip.times) < 3:
            print(f"{path.name}: no animation, skipped")
            continue
        source = source_skeleton(clip, rest or clip)
        retarget = Retarget(source, target, mapping, root=("root", "root"), hips=("Hips", mapping.get("hips", "pelvis")))
        rotations, root_t, hips_local, global_t = tracks(clip, args.smooth)
        target_rotations, translations = retarget.apply(rotations, root_t, hips_local)
        name = args.prefix + path.stem
        clip_file = Path("clips") / f"{name}.glb"
        (out / clip_file).parent.mkdir(parents=True, exist_ok=True)
        gltf_io.write_clip(out / clip_file, target, FPS, target_rotations, translations, name=name)
        looping = any(fnmatch.fnmatch(path.stem, p) for p in loops)
        frames = len(rotations)
        manifest.append({"file": clip_file.as_posix(), "looping": looping, "frames": frames, "length": (frames - 1) / FPS})
        hips = global_t[:, clip.names.index("Hips")]
        hands = [global_t[:, clip.names.index(h)] for h in ("LeftHand", "RightHand") if h in clip.names]
        info = {
            "clip": name,
            "frames": frames,
            "looping": looping,
            "travel_m": [round(float(v), 2) for v in (root_t[-1] - root_t[0])],
            "hips_y": [round(float(hips[:, 1].min()), 2), round(float(hips[:, 1].max()), 2)],
            "hands_y": [round(float(min(h[:, 1].min() for h in hands)), 2), round(float(max(h[:, 1].max() for h in hands)), 2)] if hands else None,
            "scale": round(float(retarget.scale), 3),
        }
        report.append(info)
        print(f"{name}: {frames} frames, travel {info['travel_m']} m, hips y {info['hips_y']}, hands y {info['hands_y']}" + (", loop" if looping else ""))
    (out / "manifest.json").write_text(json.dumps({"clips": manifest}, indent=1))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{len(manifest)} clips in {out}")


if __name__ == "__main__":
    main()
