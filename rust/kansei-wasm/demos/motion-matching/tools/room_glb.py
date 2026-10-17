#!/usr/bin/env python3
"""Pack the room exported from Blender for the page: www/assets/room/room.glb.

The room is authored in a .blend kept outside this repository (see the README's "The room"):
Blender exports it as a .glb with JPEG/PNG textures, and this script rewrites that file as the
page loads it:

- each material's occlusion and metallic-roughness maps as one ORM image (occlusion, roughness,
  metallic in R, G, B: a grayscale roughness map gets metallic 0, a missing occlusion 1), set as
  both its occlusion and metallic-roughness texture;
- every image encoded to KTX2 with rust/tools/ktx2 (`kansei-ktx2`): colour as ETC1S, normals and
  ORM as UASTC, at most `--max` texels a side (`--max-data` for normals and ORM; models' 1k maps
  `--max-small`), referenced through KHR_texture_basisu;
- everything else (nodes, extras, lights, cameras, meshes) as it was.

Needs Pillow, `basisu` on PATH and the tool built (`cargo build --release -p kansei-ktx2-tool` in
rust/). Usage:

    python3 tools/room_glb.py <exported.glb> [--out www/assets/room/room.glb]
"""

import argparse
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "www", "assets", "room", "room.glb")
TOOL = os.path.join(HERE, "..", "..", "..", "..", "target", "release", "kansei-ktx2")


def read_glb(path):
    data = open(path, "rb").read()
    assert data[:4] == b"glTF", "not a glb"
    offset, gltf, binary = 12, None, b""
    while offset < len(data):
        length, kind = struct.unpack("<I4s", data[offset:offset + 8])
        chunk = data[offset + 8:offset + 8 + length]
        if kind == b"JSON":
            gltf = json.loads(chunk)
        elif kind == b"BIN\0":
            binary = chunk
        offset += 8 + length
    return gltf, binary


def image_bytes(gltf, binary, index):
    view = gltf["bufferViews"][gltf["images"][index]["bufferView"]]
    start = view.get("byteOffset", 0)
    return binary[start:start + view["byteLength"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("glb")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--max", type=int, default=2048)
    parser.add_argument("--max-data", type=int, default=1024)
    parser.add_argument("--max-small", type=int, default=512, help="normal and ORM maps of the models (images named *_1k)")
    parser.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "kansei-room-ktx2"), help="encoded images, by content (a re-export encodes only what changed)")
    args = parser.parse_args()
    if not os.path.exists(TOOL):
        sys.exit(f"no {TOOL}: cargo build --release -p kansei-ktx2-tool in rust/")
    gltf, binary = read_glb(args.glb)
    textures = gltf.get("textures", [])
    source = lambda t: textures[t["index"]]["source"]
    decoded = {}

    def load(index):
        if index not in decoded:
            decoded[index] = Image.open(io.BytesIO(image_bytes(gltf, binary, index)))
        return decoded[index]

    # the images to write: (kind, name, PIL image), and each material's textures pointed at them
    out_images, keys = [], {}

    def add(key, kind, name, make):
        if key not in keys:
            keys[key] = len(out_images)
            out_images.append((kind, name, make))
        return keys[key]

    new_textures = []

    def texture(image):
        new_textures.append({"extensions": {"KHR_texture_basisu": {"source": image}}})
        return len(new_textures) - 1

    for m in gltf.get("materials", []):
        pbr = m.setdefault("pbrMetallicRoughness", {})
        if "baseColorTexture" in pbr:
            i = source(pbr["baseColorTexture"])
            img = add(("color", i), "color", gltf["images"][i].get("name", f"image{i}"), lambda i=i: load(i))
            pbr["baseColorTexture"] = {"index": texture(img)}
        if "normalTexture" in m:
            i = source(m["normalTexture"])
            img = add(("normal", i), "normal", gltf["images"][i].get("name", f"image{i}"), lambda i=i: load(i).convert("RGB"))
            m["normalTexture"] = {"index": texture(img), **({"scale": m["normalTexture"]["scale"]} if "scale" in m["normalTexture"] else {})}
        occ = source(m["occlusionTexture"]) if "occlusionTexture" in m else None
        mr = source(pbr["metallicRoughnessTexture"]) if "metallicRoughnessTexture" in pbr else None
        if occ is not None or mr is not None:

            def orm(occ=occ, mr=mr):
                ref = load(mr if mr is not None else occ)
                size = ref.size
                o = load(occ).convert("RGB").split()[0].resize(size) if occ is not None else Image.new("L", size, 255)
                if mr is None:
                    r, b = Image.new("L", size, 255), Image.new("L", size, 0)
                elif load(mr).mode in ("L", "LA", "I", "I;16"):
                    r, b = load(mr).convert("L"), Image.new("L", size, 0)
                else:
                    _, r, b = load(mr).convert("RGB").split()
                return Image.merge("RGB", (o, r, b))

            name = gltf["images"][mr if mr is not None else occ].get("name", "orm")
            img = add(("orm", occ, mr), "data", name, orm)
            t = texture(img)
            pbr["metallicRoughnessTexture"] = {"index": t}
            m["occlusionTexture"] = {"index": t}

    # the binary chunk: every buffer view but the old images', then the KTX2 images
    image_views = {img["bufferView"] for img in gltf.get("images", []) if "bufferView" in img}
    remap, views, body = {}, [], bytearray()
    for k, view in enumerate(gltf["bufferViews"]):
        if k in image_views:
            continue
        start = view.get("byteOffset", 0)
        body.extend(b"\0" * (-len(body) % 4))
        remap[k] = len(views)
        views.append({**view, "byteOffset": len(body)})
        body.extend(binary[start:start + view["byteLength"]])
    for accessor in gltf.get("accessors", []):
        if "bufferView" in accessor:
            accessor["bufferView"] = remap[accessor["bufferView"]]
    images, total = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for k, (kind, name, make) in enumerate(out_images):
            im = make()
            small = name.endswith("_1k")
            cap = args.max if kind == "color" else (args.max_small if small else args.max_data)
            side = min(cap, max(im.size))
            if max(im.size) > side:
                im = im.resize((side * im.size[0] // max(im.size), side * im.size[1] // max(im.size)), Image.LANCZOS)
            mode = "RGBA" if kind == "color" and im.mode in ("RGBA", "LA", "P") else "RGB"
            png = os.path.join(tmp, f"{k}.png")
            im.convert(mode).save(png)
            digest = hashlib.sha1(kind.encode() + open(png, "rb").read()).hexdigest()
            os.makedirs(args.cache, exist_ok=True)
            ktx2 = os.path.join(args.cache, f"{digest}.ktx2")
            if not os.path.exists(ktx2):
                subprocess.run([TOOL, "--kind", kind, "--out", ktx2 + ".part", png], check=True, stdout=subprocess.DEVNULL)
                os.replace(ktx2 + ".part", ktx2)
            data = open(ktx2, "rb").read()
            total += len(data)
            body.extend(b"\0" * (-len(body) % 4))
            views.append({"buffer": 0, "byteOffset": len(body), "byteLength": len(data)})
            body.extend(data)
            images.append({"name": name, "bufferView": len(views) - 1, "mimeType": "image/ktx2"})
            print(f"{kind:6s} {im.size[0]:5d} {len(data) // 1024:6d} KiB  {name}")
    body.extend(b"\0" * (-len(body) % 4))
    gltf["bufferViews"], gltf["images"], gltf["textures"] = views, images, new_textures
    gltf.pop("samplers", None)
    gltf["buffers"] = [{"byteLength": len(body)}]
    for key in ("extensionsUsed", "extensionsRequired"):
        gltf[key] = sorted(set(gltf.get(key, [])) | {"KHR_texture_basisu"})
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    glb = b"glTF" + struct.pack("<II", 2, 12 + 8 + len(js) + 8 + len(body))
    glb += struct.pack("<I", len(js)) + b"JSON" + js + struct.pack("<I", len(body)) + b"BIN\0" + bytes(body)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "wb").write(glb)
    print(f"{args.out}: {len(glb) / 1e6:.1f} MB ({len(images)} images, {total / 1e6:.1f} MB of KTX2)")


if __name__ == "__main__":
    main()
