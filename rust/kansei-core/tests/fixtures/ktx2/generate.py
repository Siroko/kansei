#!/usr/bin/env python3
"""Regenerate the KTX2 fixtures and their golden transcodes (tests/ktx2_transcode.rs).

Needs the official Basis Universal CLI (`basisu`, e.g. `brew install basis_universal`), clang++,
Python 3 with Pillow, and network access to fetch the official transcoder source.

For each fixture it
  1. encodes a tiny source PNG with `basisu` (ETC1S, UASTC, UASTC HDR; mips; sRGB or linear),
  2. records `basisu -unpack`'s output for every target the engine uses ("cli/<TARGET>"): the
     KTX1 files' levels, and the RGBA32 PNGs; and its CPU decode of each compressed target's
     blocks ("decoded/<TARGET>", RGBA8), which the GPU test compares hardware sampling with,
  3. builds the official transcoder at the same tag with strict IEEE float (-ffp-contract=off:
     no fused multiply-adds, as in its x86 and emscripten/WASM builds) and records its output
     for the same targets ("oracle/<TARGET>").

Why both: on arm64, clang fuses multiply-adds by default, and the Homebrew `basisu` build picks
different BC7 p-bits for a few UASTC blocks with alpha. The `basisu` crate matches the strict
build. The test checks the crate against both goldens wherever they agree, and against the
oracle where they differ, and allows a difference only for BC7.

Golden file: b"KGOLDEN1", u32 entry count; per entry a u8 name length, the name, a u32 level
count, then per level a u32 byte length and the bytes. All integers little-endian.
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.request

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
BASIS_TAG = "v2_50"  # matches basisu --version 2.50.0
RAW = f"https://raw.githubusercontent.com/BinomialLLC/basis_universal/{BASIS_TAG}"

# transcoder_texture_format codes and the names -unpack gives them
LDR_TARGETS = {
    "ETC1_RGB": 0, "ETC2_RGBA": 1, "BC1_RGB": 2, "BC4_R": 4, "BC5_RG": 5, "BC7_RGBA": 6,
    "ASTC_LDR_4X4_RGBA": 10, "RGBA32": 13, "ETC2_EAC_R11": 20, "ETC2_EAC_RG11": 21,
}
HDR_TARGETS = {"BC6H": 22, "ASTC_HDR_4X4_RGBA": 23, "RGBA_HALF": 25}

# name: (source pngs, basisu arguments, targets); several sources make a 2D array
FIXTURES = {
    "etc1s_rgb": (["rgb.png"], ["-etc1s", "-srgb"], LDR_TARGETS),
    "etc1s_rgba": (["rgba.png"], ["-etc1s", "-srgb"], LDR_TARGETS),
    "uastc_rgba": (["rgba.png"], ["-uastc", "-srgb", "-uastc_rdo_l", "1.0"], LDR_TARGETS),
    "uastc_normal": (["normal.png"], ["-uastc", "-linear", "-normal_map", "-uastc_rdo_l", "0.5"], LDR_TARGETS),
    "uastc_hdr": (["rgb.png"], ["-hdr"], HDR_TARGETS),
    "etc1s_array": (["rgba.png", "rgb.png"], ["-etc1s", "-srgb", "-tex_array"], LDR_TARGETS),
    "uastc_array": (["rgba.png", "rgb.png"], ["-uastc", "-srgb", "-tex_array"], LDR_TARGETS),
}

ORACLE_CPP = r"""
#include "transcoder/basisu_transcoder.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace basist;
// oracle <in.ktx2> <format> <out.bin>: every level, each holding every array layer in turn
int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "rb"); std::vector<uint8_t> d; int c;
  while ((c = fgetc(f)) != EOF) d.push_back((uint8_t)c);
  fclose(f);
  basisu_transcoder_init();
  ktx2_transcoder t;
  if (!t.init(d.data(), (uint32_t)d.size()) || !t.start_transcoding()) return 1;
  auto fmt = (transcoder_texture_format)atoi(argv[2]);
  FILE* o = fopen(argv[3], "wb");
  uint32_t layers = t.get_layers() ? t.get_layers() : 1;
  for (uint32_t l = 0; l < t.get_levels(); l++) {
    std::vector<uint8_t> level;
    for (uint32_t layer = 0; layer < layers; layer++) {
      ktx2_image_level_info li; t.get_image_level_info(li, l, layer, 0);
      uint32_t n = basis_transcoder_format_is_uncompressed(fmt) ? li.m_orig_width * li.m_orig_height : li.m_total_blocks;
      std::vector<uint8_t> out(n * basis_get_bytes_per_block_or_pixel(fmt));
      if (!t.transcode_image_level(l, layer, 0, out.data(), n, fmt, 0)) return 2;
      level.insert(level.end(), out.begin(), out.end());
    }
    uint32_t len = (uint32_t)level.size();
    fwrite(&len, 4, 1, o); fwrite(level.data(), 1, level.size(), o);
  }
  fclose(o);
  return 0;
}
"""


def make_sources():
    """The source images: 20x12 (mips 10x6, 5x3, 2x1, 1x1 cross block edges), RGB and RGBA,
    and an 8x8 normal map."""
    w, h = 20, 12
    rgba = Image.new("RGBA", (w, h))
    for y in range(h):
        for x in range(w):
            rgba.putpixel((x, y), (255 * x // (w - 1), 255 * y // (h - 1), (x * 37 + y * 11) % 256,
                                   255 if (x + y) % 5 else 128))
    rgba.save(os.path.join(HERE, "rgba.png"))
    rgba.convert("RGB").save(os.path.join(HERE, "rgb.png"))
    normal = Image.new("RGB", (8, 8))
    for y in range(8):
        for x in range(8):
            nx, ny = (x - 3.5) / 5, (y - 3.5) / 5
            nz = max(0.0, 1 - nx * nx - ny * ny) ** 0.5
            normal.putpixel((x, y), tuple(int((c * 0.5 + 0.5) * 255 + 0.5) for c in (nx, ny, nz)))
    normal.save(os.path.join(HERE, "normal.png"))


def build_oracle(work):
    src = os.path.join(work, "basis")
    os.makedirs(os.path.join(src, "transcoder"))
    os.makedirs(os.path.join(src, "zstd"))
    listing = urllib.request.urlopen(
        f"https://api.github.com/repos/BinomialLLC/basis_universal/contents/transcoder?ref={BASIS_TAG}").read()
    names = sorted({e["name"] for e in json.loads(listing)})
    for name in names:
        urllib.request.urlretrieve(f"{RAW}/transcoder/{name}", os.path.join(src, "transcoder", name))
    for name in ["zstddeclib.c", "zstd.h", "zstd_errors.h"]:
        urllib.request.urlretrieve(f"{RAW}/zstd/{name}", os.path.join(src, "zstd", name))
    with open(os.path.join(work, "oracle.cpp"), "w") as f:
        f.write(ORACLE_CPP)
    exe = os.path.join(work, "oracle")
    zstd = os.path.join(work, "zstd.o")
    subprocess.run(["clang", "-O1", "-w", "-c", os.path.join(src, "zstd", "zstddeclib.c"), "-o", zstd], check=True)
    subprocess.run(["clang++", "-std=c++17", "-O1", "-ffp-contract=off", "-fno-strict-aliasing", "-w",
                    "-I", src, "-o", exe, os.path.join(work, "oracle.cpp"),
                    os.path.join(src, "transcoder", "basisu_transcoder.cpp"), zstd], check=True)
    return exe


def ktx1_levels(path):
    b = open(path, "rb").read()
    levels, kv = struct.unpack_from("<II", b, 12 + 4 * 11)
    p, out = 64 + kv, []
    for _ in range(max(levels, 1)):
        (n,) = struct.unpack_from("<I", b, p)
        out.append(b[p + 4:p + 4 + n])
        p = (p + 4 + n + 3) & ~3
    return out


def cli_ktx1(unpacked, stem, target, layers):
    """-unpack's KTX1 files for `target` (one per layer), as levels of every layer in turn."""
    paths = [os.path.join(unpacked, f"{stem}_transcoded_{target}_layer_{layer:04d}.ktx") for layer in range(layers)]
    if not all(os.path.exists(p) for p in paths):
        return None
    per_layer = [ktx1_levels(p) for p in paths]
    return [b"".join(levels) for levels in zip(*per_layer)]


def cli_rgba32(unpacked, stem, levels, layers):
    out = []
    for level in range(levels):
        data = b""
        for layer in range(layers):
            rgb = Image.open(os.path.join(unpacked, f"{stem}_unpacked_rgb_RGBA32_level_{level}_face_0_layer{layer:04d}.png")).convert("RGB")
            a = Image.open(os.path.join(unpacked, f"{stem}_unpacked_a_RGBA32_{level}_0_{layer:04d}.png")).convert("L")
            data += Image.merge("RGBA", (*rgb.split(), a)).tobytes()
        out.append(data)
    return out


def cli_decoded(unpacked, stem, target, levels, layers):
    """-unpack's own CPU decode of `target`'s blocks, as RGBA8 (rgb PNG, plus the alpha PNG
    where the target has alpha), or None when it wrote none."""
    out = []
    for level in range(levels):
        data = b""
        for layer in range(layers):
            rgb = os.path.join(unpacked, f"{stem}_unpacked_rgb_{target}_level_{level}_face_0_layer_{layer:04d}.png")
            if not os.path.exists(rgb):
                return None
            a = os.path.join(unpacked, f"{stem}_unpacked_a_{target}_level_{level}_face_0_layer_{layer:04d}.png")
            rgb = Image.open(rgb).convert("RGB")
            alpha = Image.open(a).convert("L") if os.path.exists(a) else Image.new("L", rgb.size, 255)
            data += Image.merge("RGBA", (*rgb.split(), alpha)).tobytes()
        out.append(data)
    return out


def oracle_levels(oracle, ktx2, code, work):
    out_path = os.path.join(work, "oracle.bin")
    subprocess.run([oracle, ktx2, str(code), out_path], check=True)
    b, p, out = open(out_path, "rb").read(), 0, []
    while p < len(b):
        (n,) = struct.unpack_from("<I", b, p)
        out.append(b[p + 4:p + 4 + n])
        p += 4 + n
    return out


def write_golden(path, entries):
    with open(path, "wb") as f:
        f.write(b"KGOLDEN1" + struct.pack("<I", len(entries)))
        for name, levels in entries:
            f.write(struct.pack("<B", len(name)) + name.encode() + struct.pack("<I", len(levels)))
            for level in levels:
                f.write(struct.pack("<I", len(level)) + level)


def main():
    if not shutil.which("basisu"):
        sys.exit("basisu not found (brew install basis_universal)")
    version = subprocess.run(["basisu", "-version"], capture_output=True, text=True).stdout
    print(version.splitlines()[0])
    if not all(os.path.exists(os.path.join(HERE, n)) for n in ["rgb.png", "rgba.png", "normal.png"]):
        make_sources()
    work = tempfile.mkdtemp(prefix="kansei-ktx2-")
    try:
        oracle = build_oracle(work)
        for name, (sources, args, targets) in FIXTURES.items():
            ktx2 = os.path.join(HERE, f"{name}.ktx2")
            layers = len(sources)
            subprocess.run(["basisu", "-quiet", "-mipmap", *args, "-output_file", ktx2, *(os.path.join(HERE, s) for s in sources)],
                           check=True, stdout=subprocess.DEVNULL)
            unpacked = os.path.join(work, name)
            os.makedirs(unpacked)
            subprocess.run(["basisu", "-unpack", ktx2], cwd=unpacked, check=True, stdout=subprocess.DEVNULL)
            entries, report = [], []
            for target, code in targets.items():
                oracle_out = oracle_levels(oracle, ktx2, code, work)
                if target == "RGBA32":
                    cli = cli_rgba32(unpacked, name, len(oracle_out), layers)
                else:
                    cli = cli_ktx1(unpacked, name, target, layers)  # None for RGBA_HALF: -unpack writes EXR
                if cli is not None:
                    entries.append((f"cli/{target}", cli))
                decoded = None if target == "RGBA32" else cli_decoded(unpacked, name, target, len(oracle_out), layers)
                if decoded is not None:
                    entries.append((f"decoded/{target}", decoded))
                entries.append((f"oracle/{target}", oracle_out))
                report.append(f"{target}:{'-' if cli is None else ('=' if cli == oracle_out else 'DIFFERS')}")
            write_golden(os.path.join(HERE, f"{name}.golden"), entries)
            print(f"{name}: {os.path.getsize(ktx2)} bytes; cli vs oracle: {' '.join(report)}")
    finally:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
