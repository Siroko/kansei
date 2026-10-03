# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.23"]
# ///
"""Kimodo clips (NPZ) to glTF on the SOMA skeleton, ready for `kansei-anim-bake` (route A).

    uv run kimodo_to_gltf.py --kimodo path/to/kimodo out/export gen/dance_card [gen/custom ...]

Each `<clip name>_<sample>.npz` written by `kimodo_batch.py` (with its `<clip name>.json`) becomes
`out/export/clips/<clip name>_<sample>.glb`, listed with its loop flag in `manifest.json`, the
layout `kansei-anim-bake` reads. `mesh.glb` is SOMA's skinned body from the Kimodo checkout
(`kimodo/assets/skeletons/somaskel77/skin_standard.npz`, Apache-2.0).

What changes on the way:
- a ground joint `root` goes above SOMA's `Hips`: Kimodo's smoothed root path at floor height,
  turned to Kimodo's heading (+Z forward at 0, as `kansei-anim-bake` wants);
- loops (`"loop": true` in the prompt set) are cut where the pose comes back closest to an earlier
  one (at least `--loop-min` apart, 3 s for a standing loop, or the clip's `loop_min`) and the
  remaining difference is spread over the cycle, so the last frame is the first;
- other clips can lose frames at either end (`--trim`).

With `--target <mesh.glb> --map <map.json>` the clips are retargeted onto that file's skeleton
instead (route B, `retarget.py`) and no `mesh.glb` is written: bake them with that skeleton's
mesh. The target file is only read here, on this machine; it never goes to the generator.

`report.json` lists each clip's frames, loop cut and how far its loop had to be bent.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gltf_io  # noqa: E402
import quat  # noqa: E402
from retarget import Retarget  # noqa: E402

FPS = 30
ROOT = "root"
# joints that decide whether two poses match for a loop cut (fingers, eyes and ends left out)
BODY = ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
        "RightLeg", "RightShin", "RightFoot", "RightToeBase"]


def soma_skin(kimodo):
    path = Path(kimodo) / "kimodo/assets/skeletons/somaskel77/skin_standard.npz"
    return np.load(path)


def soma_parents(names, connections):
    parents = [-1] * len(names)
    for parent, child in connections:
        parents[int(child)] = int(parent)
    return parents


def rest_offsets(npz, parents):
    """SOMA's rest offsets (each joint from its parent, in the parent's frame at rest: the T-pose
    has no rotation) read back from a clip's joint positions and rotations."""
    pos, rot = npz["posed_joints"], npz["global_rot_mats"]
    offsets = np.zeros((len(parents), 3))
    for j, p in enumerate(parents):
        if p >= 0:
            offsets[j] = np.einsum("tji,tj->ti", rot[:, p], pos[:, j] - pos[:, p]).mean(0)
    return offsets


def skeleton(names, parents, offsets):
    """root (ground) > SOMA joints; Hips at its rest height, the toes' ends on the ground."""
    tpose_t, _ = quat.forward_kinematics(parents, offsets, np.tile([0.0, 0, 0, 1], (len(names), 1)))
    hips_height = -min(tpose_t[names.index(n), 1] for n in ("LeftToeEnd", "RightToeEnd"))
    translations = np.vstack([[0.0, 0.0, 0.0], offsets])
    translations[1] = [0.0, hips_height, 0.0]
    return gltf_io.Skeleton([ROOT] + names, [-1] + [p + 1 if p >= 0 else 0 for p in parents], translations)


def clip_tracks(npz):
    """Local rotations [T, 78, 4] and the root's and hips' translations of a Kimodo clip."""
    local = quat.from_matrix(npz["local_rot_mats"])
    hips_r = quat.from_matrix(npz["global_rot_mats"][:, 0])
    hips_t = npz["posed_joints"][:, 0].astype(np.float64)
    smooth = npz["smooth_root_pos"].astype(np.float64)
    heading = np.unwrap(np.arctan2(npz["global_root_heading"][:, 1], npz["global_root_heading"][:, 0]))
    root_t = np.stack([smooth[:, 0], np.zeros(len(smooth)), smooth[:, 2]], -1)
    root_r = quat.yaw(heading)
    inv_root = quat.inv(root_r)
    rotations = np.concatenate([root_r[:, None], quat.mul(inv_root, hips_r)[:, None], local[:, 1:]], 1)
    hips_local = quat.rotate(inv_root, hips_t - root_t)
    return quat.continuous(rotations), root_t, hips_local


def pose_cost(rotations, hips, body, a, b):
    """How far frame b is from frame a (joint angles, hips height), over a's and b's neighbours."""
    cost = 0.0
    for d in (-1, 0, 1):
        cost += np.sum(quat.angle(rotations[a + d, body], rotations[b + d, body]) ** 2)
        cost += 10.0 * (hips[a + d, 1] - hips[b + d, 1]) ** 2
    return cost


def find_loop(rotations, hips, body, min_frames, skip):
    """(first, last) frames of the longest cut whose ends match about as well as the best one."""
    n = len(rotations)
    costs = {}
    for a in range(max(1, skip), n - min_frames - 1):
        for b in range(a + min_frames, n - 1):
            costs[(a, b)] = pose_cost(rotations, hips, body, a, b)
    best = min(costs.values())
    good = [ab for ab, c in costs.items() if c <= 1.5 * best + 1e-4]
    return max(good, key=lambda ab: (ab[1] - ab[0], -costs[ab])), best


def close_loop(rotations, root_t, hips):
    """Spread the difference between the last and first pose over the clip (root path kept)."""
    n = len(rotations)
    t = (np.arange(n) / (n - 1))[:, None, None]
    err = quat.mul(rotations[0, 1:], quat.inv(rotations[-1, 1:]))
    ident = np.tile([0.0, 0, 0, 1], err.shape[:-1] + (1,))
    fix = quat.slerp(np.broadcast_to(ident, (n,) + ident.shape), np.broadcast_to(err, (n,) + err.shape), t)
    rotations = rotations.copy()
    rotations[:, 1:] = quat.mul(fix, rotations[:, 1:])
    rotations[-1, 1:] = rotations[0, 1:]
    hips = hips + (hips[0] - hips[-1]) * t[:, :, 0]
    hips[-1] = hips[0]
    return quat.continuous(rotations), root_t, hips, float(np.degrees(quat.angle(err, ident).max()))


def write_mesh(path, skin, skel):
    positions = skin["bind_vertices"].astype(np.float64)
    faces = skin["faces"].astype(np.int64)
    # four strongest influences per vertex, as Kansei skins
    order = np.argsort(-skin["lbs_weights"], axis=1)[:, :4]
    weights = np.take_along_axis(skin["lbs_weights"], order, 1)
    joints = np.take_along_axis(skin["lbs_indices"], order, 1)
    weights = weights / weights.sum(1, keepdims=True)
    normals = np.zeros_like(positions)
    tri = positions[faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(normals, faces[:, k], face_n)
    if np.sum(normals * (positions - positions.mean(0))) < 0:
        # wound clockwise: turn the faces round for counter-clockwise fronts
        faces, normals = faces[:, ::-1], -normals
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    # the skin lists every joint, the ground root first (Kansei's skeleton is the skin's joints):
    # SOMA joint k is skin joint k + 1
    inverse_bind = np.concatenate([np.eye(4)[None], np.linalg.inv(skin["bind_rig_transform"].astype(np.float64))])
    gltf_io.write_skinned_mesh(path, skel, positions, normals, faces, joints + 1, weights, inverse_bind, list(range(len(skel))), name="SOMA")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kimodo", required=True, help="a Kimodo checkout (for SOMA's skinned body)")
    parser.add_argument("--samples", default=None, help="comma-separated samples to keep (default: all), e.g. 00,01")
    parser.add_argument("--trim", type=int, nargs=2, default=[0, 0], metavar=("START", "END"), help="frames to drop from non-loop clips")
    parser.add_argument("--loop-min", type=float, default=0.6, help="shortest loop cycle, in seconds")
    parser.add_argument("--prefix", default="", help="put before each clip's file name (Walk/<prefix>Walk_Loop_F_00), to tell clips apart in a mixed pack")
    parser.add_argument("--target", help="a glTF whose skeleton to retarget onto (route B)")
    parser.add_argument("--map", help="the joint map for --target (maps/*.json)")
    parser.add_argument("out")
    parser.add_argument("generated", nargs="+")
    args = parser.parse_args()

    skin = soma_skin(args.kimodo)
    names = [str(n) for n in skin["rig_joint_names"]]
    parents = soma_parents(names, skin["rig_joint_connections"])
    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    keep = set(args.samples.split(",")) if args.samples else None

    target = gltf_io.read_skeleton(args.target) if args.target else None
    mapping = json.loads(Path(args.map).read_text()) if args.target else None
    skel, retarget, manifest, report = None, None, [], []
    body = [names.index(n) + 1 for n in BODY]
    for folder in map(Path, args.generated):
        for record_path in sorted(folder.rglob("*.json")):
            if record_path.name == "provenance.json" or record_path.parent == folder and record_path.stem == folder.name:
                continue
            record = json.loads(record_path.read_text())
            if "files" not in record:
                continue
            for file in record["files"]:
                sample = Path(file).stem.rsplit("_", 1)[1]
                if keep and sample not in keep:
                    continue
                npz = np.load(record_path.parent / file)
                if skel is None:
                    skel = skeleton(names, parents, rest_offsets(npz, parents))
                    if target is None:
                        write_mesh(out / "mesh.glb", skin, skel)
                    else:
                        retarget = Retarget(skel, target, mapping, hips=("Hips", mapping.get("hips", "pelvis")))
                rotations, root_t, hips = clip_tracks(npz)
                info = {"clip": f"{record['name']}_{sample}", "prompts": record["prompts"], "frames_generated": len(rotations)}
                looping = bool(record.get("loop"))
                if looping:
                    # a standing loop (an idle) wants seconds of it, not one sway
                    standing = np.linalg.norm(np.diff(root_t, axis=0), axis=1).mean() * FPS < 0.2
                    shortest = record.get("loop_min", 3.0 if standing else args.loop_min)
                    (a, b), cost = find_loop(rotations, hips, body, min(int(shortest * FPS), len(rotations) - FPS), skip=FPS // 2)
                    rotations, root_t, hips = rotations[a : b + 1], root_t[a : b + 1], hips[a : b + 1]
                    rotations, root_t, hips, bent = close_loop(rotations, root_t, hips)
                    info.update(loop_cut=[a, b], loop_match_cost=round(float(cost), 4), loop_bent_degrees=round(bent, 1))
                else:
                    s, e = args.trim
                    rotations, root_t, hips = rotations[s : len(rotations) - e], root_t[s : len(root_t) - e], hips[s : len(hips) - e]
                speed = np.linalg.norm(np.diff(root_t, axis=0), axis=1).mean() * FPS
                info.update(frames=len(rotations), mean_speed=round(float(speed), 2))
                folder, _, leaf = record["name"].rpartition("/")
                clip_name = "/".join(filter(None, [folder, f"{args.prefix}{leaf}_{sample}"]))
                info["clip"] = clip_name
                clip_file = Path("clips") / f"{clip_name}.glb"
                (out / clip_file).parent.mkdir(parents=True, exist_ok=True)
                if retarget is None:
                    gltf_io.write_clip(out / clip_file, skel, FPS, rotations, {0: root_t, 1: hips}, name=info["clip"])
                else:
                    target_rotations, translations = retarget.apply(rotations, root_t, hips)
                    gltf_io.write_clip(out / clip_file, target, FPS, target_rotations, translations, name=info["clip"])
                manifest.append({"file": clip_file.as_posix(), "looping": looping, "frames": len(rotations), "length": (len(rotations) - 1) / FPS})
                report.append(info)
                print(f"{info['clip']}: {info['frames']} frames, {info['mean_speed']} m/s" + (f", loop {info['loop_cut']} bent {info['loop_bent_degrees']} deg" if looping else ""))

    (out / "manifest.json").write_text(json.dumps({"clips": manifest}, indent=1))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{len(manifest)} clips in {out}")


if __name__ == "__main__":
    main()
