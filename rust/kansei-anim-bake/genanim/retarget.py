"""Rest-pose-aware retargeting of SOMA clips onto another humanoid skeleton (route B), numpy only.

A map names, for each source joint, the target joint that takes its rotation (`joints`), and,
for joints whose bone direction tells the two rest poses apart, the child each one aims at
(`aim`: source child, target child). A target joint copies its source joint's rotation in world
space, after the source is first bent into the target's rest pose: each source bone is swung
(the shortest rotation) from its rest direction onto the target bone's rest direction, which takes
SOMA's T-pose to, say, an A-pose. Target joints without a source keep their rest rotation under
their parent. The ground root moves as the source's; the hips' translation and the root's path
are scaled by the two skeletons' hip heights.

This is plain geometry, not a learned model: it may put generated clips on a licensed rig (the
private GASP pack) without feeding that rig to any generator.
"""

import numpy as np

import quat


def swing(a, b):
    """The shortest rotation taking direction a to direction b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    c = np.cross(a, b)
    d = float(np.dot(a, b))
    if d < -0.999999:
        axis = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        return np.append(axis / np.linalg.norm(axis), 0.0)
    return quat.normalize(np.append(c, 1.0 + d))


class Retarget:
    def __init__(self, source, target, mapping, root=("root", "root"), hips=("Hips", "pelvis")):
        self.source, self.target = source, target
        self.pairs = [(source.index(s), target.index(t)) for s, t in mapping["joints"].items()]
        self.root = (source.index(root[0]), target.index(root[1]))
        self.hips = (source.index(hips[0]), target.index(hips[1]))
        src_t, src_r = quat.forward_kinematics(source.parents, source.translations, source.rotations)
        tgt_t, tgt_r = quat.forward_kinematics(target.parents, target.translations, target.rotations)
        self.target_rest = tgt_r
        self.scale = tgt_t[self.hips[1], 1] / src_t[self.hips[0], 1]
        # the source's rest bent into the target's rest, joint by joint (parents first)
        aims = {source.index(s): (source.index(sc), target.index(tc)) for s, (sc, tc) in mapping.get("aim", {}).items()}
        to_target = dict(self.pairs)
        aligned = {}
        for s in range(len(source)):
            if s in aims:
                sc, tc = aims[s]
                t = to_target[s]
                bend = swing(src_t[sc] - src_t[s], tgt_t[tc] - tgt_t[t])
                aligned[s] = quat.mul(bend, src_r[s])
            else:
                p = source.parents[s]
                # no bone to aim: bent as its parent is
                aligned[s] = quat.mul(quat.mul(aligned[p], quat.inv(src_r[p])), src_r[s]) if p >= 0 else src_r[s]
        # target global = source global * offset, offset = aligned source rest^-1 * target rest
        self.offsets = {t: quat.mul(quat.inv(aligned[s]), tgt_r[t]) for s, t in self.pairs}

    def apply(self, rotations, root_t, hips_t):
        """Source local rotations [T, Js, 4], root and hips translations [T, 3] to the target's local
        rotations [T, Jt, 4] and its {root index: [T, 3], hips index: [T, 3]} translations."""
        frames = rotations.shape[0]
        src_t = np.broadcast_to(self.source.translations, (frames,) + self.source.translations.shape).copy()
        _, src_g = quat.forward_kinematics(self.source.parents, src_t, rotations)
        tgt = self.target
        mapped = {t: s for s, t in self.pairs}
        glob = np.empty((frames, len(tgt), 4))
        local = np.empty((frames, len(tgt), 4))
        for j, p in enumerate(tgt.parents):
            if j in mapped:
                glob[:, j] = quat.mul(src_g[:, mapped[j]], self.offsets[j])
                local[:, j] = glob[:, j] if p < 0 else quat.mul(quat.inv(glob[:, p]), glob[:, j])
            else:
                local[:, j] = tgt.rotations[j]
                glob[:, j] = local[:, j] if p < 0 else quat.mul(glob[:, p], local[:, j])
        return quat.continuous(quat.normalize(local)), {self.root[1]: root_t * self.scale, self.hips[1]: hips_t * self.scale}
