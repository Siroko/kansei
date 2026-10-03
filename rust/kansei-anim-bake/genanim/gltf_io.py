"""A small glTF 2.0 (GLB) writer and reader, numpy only: skeletons as node trees, skinned meshes and
animation clips, as `kansei-anim-bake` reads them.

Rotations are quaternions (x, y, z, w), as glTF stores them.
"""

import json
import struct

import numpy as np

FLOAT, UINT16, UINT32 = 5126, 5123, 5125


class Skeleton:
    """Joints in parent-before-child order: names, parent indices (-1 for a top joint) and the rest
    pose's local translations [J, 3] and rotations [J, 4]."""

    def __init__(self, names, parents, translations, rotations=None):
        self.names = list(names)
        self.parents = list(parents)
        self.translations = np.asarray(translations, dtype=np.float64)
        self.rotations = np.tile([0.0, 0.0, 0.0, 1.0], (len(names), 1)) if rotations is None else np.asarray(rotations, dtype=np.float64)

    def __len__(self):
        return len(self.names)

    def index(self, name):
        return self.names.index(name)


class _Builder:
    def __init__(self):
        self.doc = {"asset": {"version": "2.0", "generator": "kansei-anim-bake genanim"}, "buffers": [], "bufferViews": [], "accessors": []}
        self.blob = bytearray()

    def accessor(self, array, component, kind, target=None, minmax=False):
        array = np.ascontiguousarray(array)
        while len(self.blob) % 4:
            self.blob.append(0)
        view = {"buffer": 0, "byteOffset": len(self.blob), "byteLength": array.nbytes}
        if target:
            view["target"] = target
        self.blob += array.tobytes()
        self.doc["bufferViews"].append(view)
        acc = {"bufferView": len(self.doc["bufferViews"]) - 1, "componentType": component, "count": int(array.shape[0]), "type": kind}
        if minmax:
            flat = array.reshape(array.shape[0], -1)
            acc["min"], acc["max"] = flat.min(0).tolist(), flat.max(0).tolist()
        self.doc["accessors"].append(acc)
        return len(self.doc["accessors"]) - 1

    def nodes(self, skeleton):
        nodes = []
        for j, name in enumerate(skeleton.names):
            node = {"name": name}
            t, r = skeleton.translations[j], skeleton.rotations[j]
            if np.any(t != 0):
                node["translation"] = [float(v) for v in t]
            if np.any(r[:3] != 0):
                node["rotation"] = [float(v) for v in r]
            children = [c for c, p in enumerate(skeleton.parents) if p == j]
            if children:
                node["children"] = children
            nodes.append(node)
        self.doc["nodes"] = nodes
        tops = [j for j, p in enumerate(skeleton.parents) if p < 0]
        self.doc["scenes"] = [{"nodes": tops}]
        self.doc["scene"] = 0

    def write(self, path):
        while len(self.blob) % 4:
            self.blob.append(0)
        self.doc["buffers"] = [{"byteLength": len(self.blob)}]
        text = json.dumps(self.doc, separators=(",", ":")).encode()
        text += b" " * (-len(text) % 4)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(text) + 8 + len(self.blob)))
            f.write(struct.pack("<II", len(text), 0x4E4F534A) + text)
            f.write(struct.pack("<II", len(self.blob), 0x004E4942) + bytes(self.blob))


def write_clip(path, skeleton, fps, rotations, translations, name="clip"):
    """A clip file: the skeleton's nodes and one animation.

    rotations: [T, J, 4] local quaternions of every joint; translations: {joint index: [T, 3]} for
    the joints whose translation moves (the others keep their rest translation).
    """
    b = _Builder()
    b.nodes(skeleton)
    frames = rotations.shape[0]
    times = b.accessor(np.arange(frames, dtype=np.float32) / fps, FLOAT, "SCALAR", minmax=True)
    samplers, channels = [], []
    for j in range(len(skeleton)):
        samplers.append({"input": times, "output": b.accessor(rotations[:, j].astype(np.float32), FLOAT, "VEC4"), "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": j, "path": "rotation"}})
    for j, values in sorted(translations.items()):
        samplers.append({"input": times, "output": b.accessor(np.asarray(values, dtype=np.float32), FLOAT, "VEC3"), "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": j, "path": "translation"}})
    b.doc["animations"] = [{"name": name, "samplers": samplers, "channels": channels}]
    b.write(path)


def write_skinned_mesh(path, skeleton, positions, normals, indices, joints, weights, inverse_bind, skin_joints, name="body"):
    """A mesh file: the skeleton's nodes and one skinned mesh.

    joints/weights: [V, 4] indices into `skin_joints` (skeleton joint indices) and their weights;
    inverse_bind: [len(skin_joints), 4, 4] row-major matrices.
    """
    b = _Builder()
    b.nodes(skeleton)
    attributes = {
        "POSITION": b.accessor(positions.astype(np.float32), FLOAT, "VEC3", 34962, minmax=True),
        "NORMAL": b.accessor(normals.astype(np.float32), FLOAT, "VEC3", 34962),
        "JOINTS_0": b.accessor(joints.astype(np.uint16), UINT16, "VEC4", 34962),
        "WEIGHTS_0": b.accessor(weights.astype(np.float32), FLOAT, "VEC4", 34962),
    }
    index = b.accessor(indices.reshape(-1).astype(np.uint32), UINT32, "SCALAR", 34963)
    b.doc["meshes"] = [{"name": name, "primitives": [{"attributes": attributes, "indices": index}]}]
    # glTF matrices are column-major
    ibm = b.accessor(np.transpose(np.asarray(inverse_bind, dtype=np.float32), (0, 2, 1)).reshape(-1, 16), FLOAT, "MAT4")
    b.doc["skins"] = [{"joints": list(skin_joints), "inverseBindMatrices": ibm}]
    b.doc["nodes"].append({"name": name, "mesh": 0, "skin": 0})
    b.doc["scenes"][0]["nodes"].append(len(b.doc["nodes"]) - 1)
    b.write(path)


def read(path):
    """(json document, binary chunk) of a GLB file."""
    data = open(path, "rb").read()
    magic, _, _ = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67, f"{path} is not a GLB file"
    length, _ = struct.unpack_from("<II", data, 12)
    doc = json.loads(data[20 : 20 + length])
    rest = 20 + length
    blob = data[rest + 8 :] if rest < len(data) else b""
    return doc, blob


def read_skeleton(path):
    """The skeleton of a GLB file's first skin (or of all its nodes): names, parents and rest pose,
    joints ordered parent before child."""
    doc, _ = read(path)
    nodes = doc["nodes"]
    joints = doc["skins"][0]["joints"] if doc.get("skins") else [i for i, n in enumerate(nodes) if "mesh" not in n]
    parent_of = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    keep = set(joints)
    order = []

    def visit(i):
        if i in keep:
            order.append(i)
        for c in nodes[i].get("children", []):
            visit(c)

    for top in (i for i in range(len(nodes)) if i not in parent_of):
        visit(top)
    index = {node: k for k, node in enumerate(order)}
    names, parents, ts, rs = [], [], [], []
    for node in order:
        n = nodes[node]
        p = parent_of.get(node)
        while p is not None and p not in keep:
            p = parent_of.get(p)
        names.append(n.get("name", f"node{node}"))
        parents.append(index[p] if p is not None else -1)
        ts.append(n.get("translation", [0.0, 0.0, 0.0]))
        rs.append(n.get("rotation", [0.0, 0.0, 0.0, 1.0]))
    return Skeleton(names, parents, ts, rs)
