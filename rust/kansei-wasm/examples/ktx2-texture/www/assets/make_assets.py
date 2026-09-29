#!/usr/bin/env python3
"""Regenerate this example's textures: three procedural 256x256 images, encoded with
`kansei-ktx2` (rust/tools/ktx2; needs basisu on PATH). Run from this directory."""

import colorsys
import math
import os
import subprocess
import tempfile

from PIL import Image, ImageDraw

S = 256


def card():
    """Bulk colour: hue sweep, rings and a fine grid (ETC1S)."""
    im = Image.new("RGB", (S, S))
    for y in range(S):
        for x in range(S):
            r = math.hypot(x - S / 2, y - S / 2)
            h = (x / S + 0.15 * math.sin(r / 9)) % 1.0
            v = 0.55 + 0.45 * (y / S)
            im.putpixel((x, y), tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, 0.75, v)))
    d = ImageDraw.Draw(im)
    for i in range(0, S, 16):
        d.line([(i, 0), (i, S)], fill=(20, 20, 30))
        d.line([(0, i), (S, i)], fill=(20, 20, 30))
    d.text((12, 12), "ETC1S", fill=(255, 255, 255))
    return im


def badge():
    """Colour with alpha: a round badge with a soft edge and a star cut-out (UASTC)."""
    im = Image.new("RGBA", (S, S))
    for y in range(S):
        for x in range(S):
            r = math.hypot(x - S / 2 + 0.5, y - S / 2 + 0.5) / (S / 2)
            a = max(0.0, min(1.0, (0.95 - r) * 20))
            t = y / S
            im.putpixel((x, y), (int(240 - 120 * t), int(90 + 110 * t), int(60 + 180 * r), int(a * 255)))
    d = ImageDraw.Draw(im)
    pts = [(S / 2 + math.cos(math.pi / 2 + i * math.pi / 5) * (70 if i % 2 == 0 else 30),
            S / 2 - math.sin(math.pi / 2 + i * math.pi / 5) * (70 if i % 2 == 0 else 30)) for i in range(10)]
    d.polygon(pts, fill=(0, 0, 0, 0))
    d.text((100, 200), "UASTC", fill=(255, 255, 255, 255))
    return im


def bumps():
    """Tangent-space normal map (+x right, +y up, +z out): a grid of domes and a groove ring."""
    def height(x, y):
        cx, cy = (x % 64) - 32, (y % 64) - 32
        dome = max(0.0, 1 - (cx * cx + cy * cy) / 26.0 ** 2) ** 0.5 * 12
        r = math.hypot(x - S / 2, y - S / 2)
        return dome - 6 * math.exp(-((r - 100) / 5) ** 2)

    im = Image.new("RGB", (S, S))
    for y in range(S):
        for x in range(S):
            dx = height(x + 1, y) - height(x - 1, y)
            drow = height(x, y + 1) - height(x, y - 1)
            n = (-dx / 2, drow / 2, 1.0)  # image rows run down, tangent +y up
            l = math.sqrt(sum(c * c for c in n))
            im.putpixel((x, y), tuple(int((c / l * 0.5 + 0.5) * 255 + 0.5) for c in n))
    return im


def main():
    tool = ["cargo", "run", "--quiet", "--release", "--manifest-path",
            os.path.join(os.path.dirname(__file__), "../../../../../Cargo.toml"), "-p", "kansei-ktx2-tool", "--"]
    with tempfile.TemporaryDirectory() as tmp:
        for name, image, kind in [("card_etc1s", card(), "color"), ("badge_uastc", badge(), "hero"), ("bumps_normal", bumps(), "normal")]:
            png = os.path.join(tmp, name + ".png")
            image.save(png)
            subprocess.run(tool + ["--kind", kind, "--out", name + ".ktx2", png], check=True)


if __name__ == "__main__":
    main()
