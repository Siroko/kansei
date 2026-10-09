#!/usr/bin/env python3
"""Fetch the room page's CC0 assets and encode them as the page loads them (www/assets/room/).

- Surfaces (ambientCG, CC0): colour, normal and occlusion/roughness/metallic maps, each encoded to
  KTX2 with rust/tools/ktx2 (`kansei-ktx2`): colour as ETC1S, normals and ORM as UASTC.
- Models (Poly Haven, CC0): the 1k glTF, its textures encoded the same way and the whole written
  as one .glb whose textures use KHR_texture_basisu (the KTX2 images in its binary chunk).

Needs Pillow, `basisu` on PATH and the tool built (`cargo build --release -p kansei-ktx2-tool` in
rust/). Downloads are cached in --cache. Usage:

    python3 tools/room_assets.py [--cache DIR] [--only NAME ...]
"""

import argparse
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "www", "assets", "room")
TOOL = os.path.join(HERE, "..", "..", "..", "..", "target", "release", "kansei-ktx2")

# name: (ambientCG id, colour size, normal and ORM size)
SURFACES = {
    "marble": ("Marble012", 2048, 1024),
    "wood": ("WoodFloor016", 2048, 1024),
    "plaster": ("Plaster001", 1024, 512),
    "concrete": ("Concrete031", 1024, 512),
    "ceiling": ("Concrete046", 1024, 512),
    "oak": ("Wood049", 1024, 512),
    "rug": ("Carpet012", 512, 512),
    "blackmarble": ("Marble016", 1024, 512),
    "gravel": ("Gravel043", 1024, 512),
    "metal": ("Metal032", 512, 512),
}

# Poly Haven model: (colour size, normal and ORM size)
MODELS = {
    "sofa_02": (1024, 512),
    "mid_century_lounge_chair": (1024, 512),
    "modern_arm_chair_01": (1024, 512),
    "Ottoman_01": (512, 512),
    "modern_coffee_table_01": (1024, 512),
    "modern_wooden_cabinet": (1024, 512),
    "wooden_display_shelves_01": (512, 512),
    "steel_frame_shelves_03": (512, 512),
    "round_wooden_table_01": (1024, 512),
    "dining_chair_02": (512, 512),
    "potted_plant_02": (1024, 512),
    "potted_plant_04": (512, 512),
    "ceramic_vase_01": (512, 256),
    "ceramic_vase_03": (512, 256),
    "throw_pillows_01": (512, 256),
    "hanging_picture_frame_02": (1024, 256),
    "book_encyclopedia_set_01": (512, 256),
    "modern_ceiling_lamp_01": (512, 256),
}


def fetch(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "kansei-room-assets"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.replace(path + ".part", path)
    return path


def encode(png, kind, out):
    subprocess.run([TOOL, "--kind", kind, "--out", out, png], check=True, stdout=subprocess.DEVNULL)


def resized(image, size, mode):
    image = image.convert(mode)
    return image if image.size == (size, size) else image.resize((size, size), Image.LANCZOS)


def surface(name, cache):
    asset, color_size, data_size = SURFACES[name]
    res = "2K" if color_size > 1024 else "1K"
    zpath = fetch(f"https://ambientcg.com/get?file={asset}_{res}-JPG.zip", os.path.join(cache, "ambientcg", f"{asset}_{res}-JPG.zip"))
    maps = {}
    with zipfile.ZipFile(zpath) as z:
        for entry in z.namelist():
            for key in ("Color", "NormalGL", "Roughness", "AmbientOcclusion", "Metalness"):
                if entry.endswith(f"_{key}.jpg"):
                    maps[key] = Image.open(io.BytesIO(z.read(entry)))
    with tempfile.TemporaryDirectory() as tmp:
        color = os.path.join(tmp, "color.png")
        resized(maps["Color"], color_size, "RGB").save(color)
        normal = os.path.join(tmp, "normal.png")
        resized(maps["NormalGL"], data_size, "RGB").save(normal)
        # occlusion, roughness, metallic in R, G, B
        one = lambda key, fill: resized(maps[key], data_size, "L") if key in maps else Image.new("L", (data_size, data_size), fill)
        orm = os.path.join(tmp, "orm.png")
        Image.merge("RGB", (one("AmbientOcclusion", 255), one("Roughness", 160), one("Metalness", 0))).save(orm)
        for path, kind, suffix in ((color, "color", "color"), (normal, "normal", "normal"), (orm, "data", "orm")):
            encode(path, kind, os.path.join(OUT, f"{name}_{suffix}.ktx2"))
    print(f"surface {name} ({asset})")


def model(name, cache):
    color_size, data_size = MODELS[name]
    req = urllib.request.Request(f"https://api.polyhaven.com/files/{name}", headers={"User-Agent": "kansei-room-assets"})
    files = json.load(urllib.request.urlopen(req, timeout=60))
    entry = files["gltf"]["1k"]["gltf"]
    base = os.path.join(cache, "polyhaven", name)
    gltf = json.load(open(fetch(entry["url"], os.path.join(base, os.path.basename(entry["url"])))))
    for rel, inc in entry["include"].items():
        fetch(inc["url"], os.path.join(base, rel))
    assert len(gltf["buffers"]) == 1, name
    bin_data = bytearray(open(os.path.join(base, gltf["buffers"][0]["uri"]), "rb").read())
    images = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, image in enumerate(gltf["images"]):
            uri = image["uri"]
            lower = uri.lower()
            kind = "normal" if "nor" in lower else "data" if any(k in lower for k in ("arm", "rough", "metal", "_ao")) else "color"
            size = color_size if kind == "color" else data_size
            src = Image.open(os.path.join(base, uri))
            mode = "RGBA" if kind == "color" and src.mode in ("RGBA", "LA", "P") else "RGB"
            png = os.path.join(tmp, f"{i}.png")
            resized(src, size, mode).save(png)
            ktx2 = os.path.join(tmp, f"{i}.ktx2")
            encode(png, kind, ktx2)
            data = open(ktx2, "rb").read()
            bin_data.extend(b"\0" * (-len(bin_data) % 4))
            gltf["bufferViews"].append({"buffer": 0, "byteOffset": len(bin_data), "byteLength": len(data)})
            bin_data.extend(data)
            images.append({"name": os.path.splitext(os.path.basename(uri))[0], "bufferView": len(gltf["bufferViews"]) - 1, "mimeType": "image/ktx2"})
    bin_data.extend(b"\0" * (-len(bin_data) % 4))
    gltf["images"] = images
    for texture in gltf.get("textures", []):
        source = texture.pop("source")
        texture.setdefault("extensions", {})["KHR_texture_basisu"] = {"source": source}
    for key in ("extensionsUsed", "extensionsRequired"):
        gltf[key] = sorted(set(gltf.get(key, [])) | {"KHR_texture_basisu"})
    gltf["buffers"] = [{"byteLength": len(bin_data)}]
    body = json.dumps(gltf, separators=(",", ":")).encode()
    body += b" " * (-len(body) % 4)
    glb = b"glTF" + struct.pack("<II", 2, 12 + 8 + len(body) + 8 + len(bin_data))
    glb += struct.pack("<I", len(body)) + b"JSON" + body + struct.pack("<I", len(bin_data)) + b"BIN\0" + bytes(bin_data)
    open(os.path.join(OUT, f"{name}.glb"), "wb").write(glb)
    print(f"model {name}: {len(glb) / 1e6:.2f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "kansei-room-assets"))
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    if not os.path.exists(TOOL):
        sys.exit(f"no {TOOL}: cargo build --release -p kansei-ktx2-tool in rust/")
    os.makedirs(OUT, exist_ok=True)
    for name in SURFACES:
        if not args.only or name in args.only:
            surface(name, args.cache)
    for name in MODELS:
        if not args.only or name in args.only:
            model(name, args.cache)
    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print(f"www/assets/room: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
