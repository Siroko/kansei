"""Measure how a parkour set's clips meet their effectors: hands on the bar or ledge, feet on the
wall, from Kimodo's NPZ (posed joints, SOMA 77). Runs where Kimodo is (the Docker image):

    python spike_check.py /out/parkour   (the set's prompt file is /out/parkour/parkour.json)
"""

import json
import sys
from pathlib import Path

import numpy as np

from kimodo.skeleton.definitions import SOMASkeleton77


def main():
    out = Path(sys.argv[1])
    names = SOMASkeleton77().bone_order_names
    j = {n: names.index(n) for n in ["Hips", "LeftHand", "RightHand", "LeftFoot", "RightFoot", "Head"]}
    spec = json.loads((out / f"{out.name}.json").read_text())
    for clip in spec["clips"]:
        e = clip["effectors"]
        for npz in sorted(out.glob(clip["name"] + "_*.npz")):
            p = np.load(npz)["posed_joints"]
            fps = 30.0
            hands = p[:, [j["LeftHand"], j["RightHand"]]]
            hips = p[:, j["Hips"]]
            line = []
            if e["kind"] in ("bar_hang", "bar_swing", "shimmy"):
                h = e.get("height", 2.3)
                z = 0.05 if e["kind"] == "shimmy" else 0.0
                # distance of each hand from the bar's line (y, z); hand speed
                off = np.sqrt((hands[..., 1] - h) ** 2 + (hands[..., 2] - z) ** 2)
                speed = np.linalg.norm(np.diff(hands, axis=0), axis=-1) * fps
                line.append(f"hands off the line: mean {100*off.mean():.1f} cm, 95% {100*np.percentile(off,95):.1f}, max {100*off.max():.1f}")
                line.append(f"hand speed along it: median {np.median(speed):.2f} m/s, 95% {np.percentile(speed,95):.2f}")
                line.append(f"hips y {hips[:,1].min():.2f}..{hips[:,1].max():.2f}, z {hips[:,2].min():.2f}..{hips[:,2].max():.2f}, x {hips[0,0]:.2f}->{hips[-1,0]:.2f}")
                feet = p[:, [j["LeftFoot"], j["RightFoot"]]]
                line.append(f"feet lowest {feet[...,1].min():.2f} m")
            if e["kind"] == "wall_run":
                wall = e.get("distance") or 0.8 * e.get("side", -1.0)
                feet = p[:, [j["LeftFoot"], j["RightFoot"]]]
                start, end = int(e["start"] * fps), int((e["start"] + e["time"]) * fps)
                win = feet[start:end]
                gap = np.abs(win[..., 0] - wall)
                line.append(f"feet in the wall window: nearest the wall {100*gap.min(axis=1).mean():.1f} cm on average, feet height {win[...,1].min():.2f}..{win[...,1].max():.2f}")
                line.append(f"hips x {hips[start:end,0].min():.2f}..{hips[start:end,0].max():.2f}, y {hips[start:end,1].min():.2f}..{hips[start:end,1].max():.2f}; z {hips[0,2]:.1f}->{hips[-1,2]:.1f}")
                beyond = (feet[..., 0] * np.sign(wall) > abs(wall) + 0.05).sum()
                line.append(f"foot frames through the wall (> 5 cm): {beyond}")
            print(npz.name)
            for l in line:
                print("   ", l)


if __name__ == "__main__":
    main()
