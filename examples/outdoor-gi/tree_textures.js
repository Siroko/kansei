// Procedural tree textures, painted on the CPU at load (no image files ship): a port of the Rust
// example's tree_textures.rs, itself the Raggare intro's.
//
// Foliage atlas (1024², RGBA: sRGB albedo + alpha, and a matching tangent-space normal map):
//   +--------------------+--------------------+
//   | A spruce spray     | B spruce curtain   |   A: a branch seen from above, base at the bottom
//   |   (fishbone)       |   (hanging comb)   |      edge, tip at the top (v = 1 -> 0 along it)
//   +--------------------+--------------------+   B: pendulous branchlets hanging from the top edge
//   | C birch, weeping   | D birch, spray     |   C: leafy twigs hanging from the top edge
//   |   twigs            |   (fan of twigs)   |   D: a fan of leafy twigs from the bottom centre
//   +--------------------+--------------------+
// Bark: one tileable 256x512 set per species (u around the trunk, v along it).
import { Canvas, Rng, add2, sub2, mul2, len2, norm2, lerp3, mul3 } from './canvas.js'

export const ATLAS = 1024
const R = 512 // region size

/** UV rectangle [u0, v0, u1, v1] of an atlas region. */
export class Region {
  constructor(r) {
    this.r = r
  }
  /** Map region-local (s, t) in 0..1 to atlas UV, inset half a texel against bleeding. */
  uv(s, t) {
    const [u0, v0, u1, v1] = this.r
    const inset = 2 / ATLAS
    return [u0 + inset + (u1 - u0 - 2 * inset) * s, v0 + inset + (v1 - v0 - 2 * inset) * t]
  }
}

export const SPRUCE_SPRAY = new Region([0, 0, 0.5, 0.5])
export const SPRUCE_CURTAIN = new Region([0.5, 0, 1, 0.5])
export const BIRCH_HANGING = new Region([0, 0.5, 0.5, 1])
export const BIRCH_SPRAY = new Region([0.5, 0.5, 1, 1])

export const BARK_W = 256
export const BARK_H = 512

const rad = (d) => (d * Math.PI) / 180
const rot2 = (d, a) => {
  const s = Math.sin(a), c = Math.cos(a)
  return [d[0] * c - d[1] * s, d[0] * s + d[1] * c]
}

// ── foliage ─────────────────────────────────────────────────────────────────────────────

const TWIG = [0.045, 0.032, 0.02]

function needleColour(rng, fresh) {
  // dark blue-green Norway spruce needles, lighter and yellower at the new growth
  const old = mul3([0.022, 0.05, 0.03], rng.range(0.75, 1.25))
  const nu = mul3([0.06, 0.1, 0.035], rng.range(0.85, 1.15))
  return lerp3(old, nu, Math.min(Math.max(fresh, 0), 1))
}

/** Needles along a polyline, alternating sides, angled forward along it. */
function needlesAlong(c, rng, pts, len, spacing, spread, freshFrom, droop) {
  let total = 0
  for (let i = 1; i < pts.length; i++) total += len2(sub2(pts[i], pts[i - 1]))
  let travelled = 0
  for (let k = 1; k < pts.length; k++) {
    const a = pts[k - 1], b = pts[k]
    const seg = len2(sub2(b, a))
    const dir = mul2(sub2(b, a), 1 / Math.max(seg, 1e-4))
    let s = 0
    while (s < seg) {
      const p = add2(a, mul2(dir, s))
      const t = (travelled + s) / Math.max(total, 1e-4)
      for (const side of [-1, 1]) {
        const ang = rad(rng.range(spread[0], spread[1])) * side
        let d = rot2(dir, ang)
        d = norm2(add2(d, mul2(droop, rng.f())))
        const l = len * rng.range(0.75, 1.15) * (1 - 0.35 * t)
        const fresh = Math.max((t - freshFrom) / (1 - freshFrom), 0)
        c.stroke(p, add2(p, mul2(d, l)), 1.8, 0.7, needleColour(rng, fresh), 2 + rng.f())
      }
      s += spacing * rng.range(0.8, 1.2)
    }
    travelled += seg
  }
}

/** A gently curving polyline from `a` in direction `dir`, `n` segments of `step` pixels. */
function curve(rng, a, dir, step, n, bend, pull) {
  const pts = [a]
  let d = norm2(dir)
  let p = a
  for (let i = 0; i < n; i++) {
    d = rot2(d, rng.range(-bend, bend))
    d = norm2(add2(d, pull))
    p = add2(p, mul2(d, step))
    pts.push(p)
  }
  return pts
}

function drawPolyline(c, pts, w0, w1, colour, height) {
  const n = Math.max(pts.length - 1, 1)
  for (let i = 0; i + 1 < pts.length; i++) {
    const ta = i / n, tb = (i + 1) / n
    c.stroke(pts[i], pts[i + 1], w0 + (w1 - w0) * ta, w0 + (w1 - w0) * tb, colour, height)
  }
}

function spruceSpray(c, o, rng) {
  const s = R
  // main axis: base at the bottom edge, tip near the top
  const axis = curve(rng, add2(o, [s * 0.5, s - 6]), [0, -1], 24, 20, 0.02, [0, -0.05])
  drawPolyline(c, axis, 5, 2, TWIG, 1)
  needlesAlong(c, rng, axis, 20, 2, [35, 75], 0.7, [0, 0])
  // side twigs, alternating, shorter toward the tip; second-order twigs on the long ones
  const n = axis.length
  for (let i = 1; i < 1 + (n - 3); i++) {
    const p = axis[i]
    const t = i / n
    for (const side of [-1, 1]) {
      if (rng.f() < 0.12) continue
      const ang = rad(rng.range(52, 72)) * side
      const d = [Math.sin(ang), -Math.cos(ang)]
      const len = s * 0.44 * (1 - 0.7 * t) * rng.range(0.8, 1.1)
      const steps = Math.trunc(Math.max(len / 12, 2))
      const twig = curve(rng, p, d, len / steps, steps, 0.08, [0, -0.04])
      drawPolyline(c, twig, 2.5, 1, TWIG, 1)
      needlesAlong(c, rng, twig, 16, 2, [40, 80], 0.6, [0, 0])
      if (len > s * 0.2) {
        // every second point from the third, three at most
        for (let k = 2, taken = 0; k < twig.length && taken < 3; k += 2, taken++) {
          const q = twig[k]
          const a2 = ang + rng.range(0.5, 0.8) * side
          const d2 = [Math.sin(a2), -Math.cos(a2)]
          const sub = curve(rng, q, d2, 9, 4, 0.1, [0, 0])
          drawPolyline(c, sub, 1.6, 0.8, TWIG, 1)
          needlesAlong(c, rng, sub, 13, 2.2, [40, 80], 0.5, [0, 0])
        }
      }
    }
  }
}

function spruceCurtain(c, o, rng) {
  const s = R
  const count = 16
  for (let i = 0; i < count; i++) {
    const x = s * (0.07 + (0.86 * (i + rng.range(-0.3, 0.3))) / (count - 1))
    const len = s * rng.range(0.55, 0.95)
    const steps = 14
    const lean = rng.range(-0.25, 0.25)
    const hang = curve(rng, add2(o, [x, 4]), [lean, 1], len / steps, steps, 0.07, [0, 0.05])
    drawPolyline(c, hang, 2.4, 1, TWIG, 1)
    needlesAlong(c, rng, hang, 17, 2, [35, 75], 0.75, [0, 0.3])
  }
}

function leafColour(rng) {
  const yellow = rng.f()
  return mul3([0.04 + 0.03 * yellow, 0.085 + 0.02 * yellow, 0.025], rng.range(0.7, 1.3))
}

function leafyTwig(c, rng, pts, leaf, every) {
  drawPolyline(c, pts, 1.6, 0.8, [0.06, 0.04, 0.03], 1)
  let acc = 0
  let side = 1
  for (let k = 1; k < pts.length; k++) {
    const a = pts[k - 1], b = pts[k]
    const seg = len2(sub2(b, a))
    const dir = mul2(sub2(b, a), 1 / Math.max(seg, 1e-4))
    let s = acc
    while (s < seg) {
      const p = add2(a, mul2(dir, s))
      const ang = rad(rng.range(35, 65)) * side
      const d = rot2(dir, ang)
      const r = leaf * rng.range(0.8, 1.2)
      const stalk = add2(p, mul2(d, r * 0.5))
      c.stroke(p, stalk, 1, 0.8, [0.06, 0.05, 0.02], 1.5)
      c.ellipse(add2(stalk, mul2(d, r)), d, r, r * rng.range(0.6, 0.75), leafColour(rng), 2 + rng.f())
      side = -side
      s += every * rng.range(0.7, 1.3)
    }
    acc = s - seg
  }
}

function birchHanging(c, o, rng) {
  const s = R
  for (let i = 0; i < 15; i++) {
    const x = s * (0.06 + (0.88 * (i + rng.range(-0.4, 0.4))) / 14)
    const len = s * rng.range(0.6, 0.95)
    const lean = rng.range(-0.3, 0.3)
    const pts = curve(rng, add2(o, [x, 4]), [lean, 1], len / 16, 16, 0.08, [0, 0.06])
    leafyTwig(c, rng, pts, 7, 8.5)
  }
}

function birchSpray(c, o, rng) {
  const s = R
  const base = add2(o, [s * 0.5, s - 6])
  for (let i = 0; i < 13; i++) {
    const ang = rad(-75 + (150 * i) / 12 + rng.range(-8, 8))
    const d = [Math.sin(ang), -Math.cos(ang)]
    const len = s * rng.range(0.55, 0.85)
    const pts = curve(rng, base, d, len / 14, 14, 0.07, [0, 0.03])
    leafyTwig(c, rng, pts, 7, 8)
  }
}

function foliageAtlas() {
  const c = new Canvas(ATLAS, ATLAS)
  const regions = [
    [spruceSpray, [0, 0], 11],
    [spruceCurtain, [R, 0], 12],
    [birchHanging, [0, R], 13],
    [birchSpray, [R, R], 14],
  ]
  for (const [paint, o, seed] of regions) {
    // keep each painting inside its region, with a small gutter against mip bleeding
    c.clip = [o[0] + 2, o[1] + 2, o[0] + R - 2, o[1] + R - 2]
    paint(c, o, new Rng(seed))
  }
  c.clip = [0, 0, ATLAS, ATLAS]
  c.bleed(6)
  return c.finish(0.9, false)
}

// ── bark ────────────────────────────────────────────────────────────────────────────────

/** Wrapping cellular noise: [F1, F2] distances to the nearest seeds, in cell units. */
function cells(px, py, gx, gy, seeds) {
  const cx0 = Math.floor(px), cy0 = Math.floor(py)
  let f1 = Infinity, f2 = Infinity
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const cx = cx0 + dx, cy = cy0 + dy
      const k = (((cy % gy) + gy) % gy) * gx + (((cx % gx) + gx) % gx)
      const d = Math.hypot(cx + seeds[2 * k] - px, cy + seeds[2 * k + 1] - py)
      if (d < f1) {
        f2 = f1
        f1 = d
      } else if (d < f2) f2 = d
    }
  }
  return [f1, f2]
}

function spruceBark() {
  // reddish grey-brown bark in thin, rounded scales
  const c = new Canvas(BARK_W, BARK_H)
  const rng = new Rng(21)
  const gx = 18, gy = 44
  const seeds = new Float32Array(gx * gy * 2)
  for (let i = 0; i < gx * gy; i++) {
    seeds[2 * i] = rng.range(0.1, 0.9)
    seeds[2 * i + 1] = rng.range(0.1, 0.9)
  }
  const tones = Array.from({ length: gx * gy }, () => rng.range(0.7, 1.3))
  for (let y = 0; y < BARK_H; y++) {
    for (let x = 0; x < BARK_W; x++) {
      const px = (x / BARK_W) * gx, py = (y / BARK_H) * gy
      const [f1, f2] = cells(px, py, gx, gy, seeds)
      const edge = Math.min((f2 - f1) * 5, 1)
      const tone = tones[(Math.floor(py) % gy) * gx + (Math.floor(px) % gx)]
      const base = mul3([0.13, 0.095, 0.075], (0.85 + 0.15 * tone) * (0.9 + 0.2 * Math.abs(Math.sin(py * 0.7 + px * 1.3))))
      const crack = [0.05, 0.04, 0.034]
      const i = y * BARK_W + x
      const col = lerp3(crack, base, Math.pow(edge, 0.4))
      c.color.set(col, 3 * i)
      c.height[i] = Math.sqrt(edge) * 2 - f1 * 0.6
      c.alpha[i] = 1
    }
  }
  return c.finish(1.2, true)
}

function birchBark() {
  // silver birch: chalky white with dark horizontal lenticels and the odd black scar
  const c = new Canvas(BARK_W, BARK_H)
  const rng = new Rng(22)
  for (let y = 0; y < BARK_H; y++) {
    const band = 0.94 + 0.06 * (Math.sin(y * 0.09) * Math.cos(y * 0.023))
    for (let x = 0; x < BARK_W; x++) {
      const i = y * BARK_W + x
      c.color.set(mul3([0.62, 0.61, 0.56], band * rng.range(0.96, 1.04)), 3 * i)
      c.height[i] = 0
      c.alpha[i] = 1
    }
  }
  const w = BARK_W
  const mark = (a, b, width, colour) => {
    // wrap around the trunk
    for (const off of [-w, 0, w]) c.stroke([a[0] + off, a[1]], [b[0] + off, b[1]], width, width * 0.6, colour, 0.5)
  }
  for (let k = 0; k < 140; k++) {
    const p = [rng.range(0, w), rng.range(0, BARK_H)]
    const len = rng.range(8, 38)
    const q = [p[0] + len, p[1] + rng.range(-1.5, 1.5)]
    mark(p, q, rng.range(1.5, 3), [0.07, 0.06, 0.05])
  }
  for (let k = 0; k < 8; k++) {
    const p = [rng.range(0, w), rng.range(0, BARK_H)]
    const len = rng.range(10, 22)
    const q = [p[0] + rng.range(-4, 4), p[1] + len]
    mark(p, q, rng.range(6, 12), [0.035, 0.03, 0.028])
  }
  return c.finish(0.8, true)
}

/** { foliage, spruceBark, birchBark }: each [albedo RGBA8 sRGB, normal RGBA8]. */
export function generate() {
  return { foliage: foliageAtlas(), spruceBark: spruceBark(), birchBark: birchBark() }
}
