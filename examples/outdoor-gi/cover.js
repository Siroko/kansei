// The ground cover: grass tufts, flower clumps and shrubs as alpha-tested cards on a CPU-painted
// atlas. A port of the Rust example's cover_meshes.rs and cover_textures.rs, themselves the
// Raggare intro's.
//
// The atlas (1024², sRGB albedo + alpha and a tangent-space normal map):
//   +--------------------+--------------------+
//   | GRASS  meadow tuft | SEEDS  grass with  |   all four are side views of a plant standing
//   |                    |   flowering heads  |   on the bottom edge (v = 1 at the ground)
//   +--------------------+--------------------+
//   | FLOWERS  daisies,  | SHRUB  leafy twigs |
//   |  buttercups, campion|                   |
//   +--------------------+--------------------+
import { Canvas, Rng, add2, mul2, lerp2, norm2, add3, mul3, cross3, norm3, lerp3 } from './canvas.js'
import { Region } from './tree_textures.js'
import { Mesh } from './tree_meshes.js'

export const ATLAS = 1024
const R = 512

export const GRASS = new Region([0, 0, 0.5, 0.5])
export const SEEDS = new Region([0.5, 0, 1, 0.5])
export const FLOWERS = new Region([0, 0.5, 0.5, 1])
export const SHRUB = new Region([0.5, 0.5, 1, 1])

// ── meshes: in metres at instance scale 1, standing on the origin; region-local UVs (v = 1 at the
// ground), position.w the ambient occlusion (dark at the roots), normals leaning up ─────────────

function pushCard(m, centre, across, up, width, height, flip, aoBase) {
  const base = m.vertexCount
  const n = norm3(cross3(across, up))
  // mostly upward, a little of the card's own facing
  const normal = norm3(add3([0, 0.75, 0], mul3(n, 0.25)))
  const [u0, u1] = flip ? [1, 0] : [0, 1]
  for (const [s, t, u, v] of [[-0.5, 0, u0, 1], [0.5, 0, u1, 1], [-0.5, 1, u0, 0], [0.5, 1, u1, 0]]) {
    const p = add3(add3(centre, mul3(across, s * width)), mul3(up, t * height))
    m.vertex(p, aoBase + (1 - aoBase) * t, normal, [u, v])
  }
  m.indices.push(base, base + 1, base + 2, base + 2, base + 1, base + 3)
}

/** Crossed vertical cards through the centre, fanned around Y, each leaning a little outward. */
function crossed(cards, width, height, lean, seed) {
  const rng = new Rng(seed)
  const m = new Mesh()
  for (let i = 0; i < cards; i++) {
    const a = (i / cards) * Math.PI + rng.range(-0.15, 0.15)
    const across = [Math.cos(a), 0, Math.sin(a)]
    const out = mul3([-Math.sin(a), 0, Math.cos(a)], rng.range(-lean, lean))
    const up = norm3(add3([0, 1, 0], out))
    pushCard(m, [0, 0, 0], across, up, width, height * rng.range(0.9, 1.1), i % 2 === 1, 0.35)
  }
  return m
}

/** A low shrub: leafy cards round an ellipsoid, facing out and up. */
function shrubMesh(cards, radius, height, seed) {
  const rng = new Rng(seed)
  const m = new Mesh()
  for (let i = 0; i < cards; i++) {
    const f = (i + 0.5) / cards
    const a = i * 2.3999632
    const y = height * (0.15 + 0.6 * f)
    const r = radius * Math.min(Math.max(1 - Math.abs(f - 0.35), 0.4), 1) * rng.range(0.7, 1)
    const dir = [Math.cos(a), 0, Math.sin(a)]
    const centre = add3(mul3(dir, r * 0.4), [0, y * 0.6, 0])
    const up = norm3(add3([0, 0.8, 0], mul3(dir, 0.6)))
    const across = norm3(cross3([0, 1, 0], dir))
    pushCard(m, centre, across, up, radius * 1.1, height * 0.75, i % 2 === 0, 0.5)
  }
  return m
}

/** Grass tuft, flowers and shrub meshes: [near, far] each. */
export function meshes() {
  return {
    // the Unreal verge's clumps: wide, dense and up to about 0.9 m tall
    grass: [crossed(8, 1.35, 0.9, 0.25, 1), crossed(4, 1.35, 0.86, 0.15, 2)],
    flowers: [crossed(4, 0.95, 0.62, 0.15, 3), crossed(3, 0.95, 0.6, 0.1, 4)],
    shrub: [shrubMesh(18, 0.75, 1, 5), shrubMesh(7, 0.8, 1, 6)],
  }
}

// ── the atlas ───────────────────────────────────────────────────────────────────────────────

function grassColour(rng) {
  // the verge's grass: a lighter, yellower green than the forest's, through to the first straw
  const green = [0.07, 0.13, 0.035]
  const straw = [0.17, 0.155, 0.07]
  return mul3(lerp3(green, straw, rng.f() ** 3), rng.range(0.7, 1.25))
}

/** A curved, tapering blade from `base` rising by `height` pixels, leaning by `lean`. */
function blade(c, rng, base, height, lean, width, colour) {
  const steps = 10
  let prev = base
  const droop = rng.range(0, 0.5)
  for (let i = 1; i <= steps; i++) {
    const t = i / steps
    // leaning more toward the tip, and the tallest blades bowing over
    const x = base[0] + lean * t * t + droop * Math.sign(lean) * height * 0.25 * t ** 4
    const y = base[1] - height * t + droop * height * 0.15 * t ** 3
    const p = [x, y]
    const w0 = width * (1 - (t - 1 / steps) * 0.9)
    const w1 = width * (1 - t * 0.9)
    c.stroke(prev, p, Math.max(w0, 0.6), Math.max(w1, 0.5), mul3(colour, 0.8 + 0.3 * t), 1 + t)
    prev = p
  }
  return prev
}

function tuft(c, o, rng, blades, seedHeads) {
  const s = R
  for (let k = 0; k < blades; k++) {
    const base = add2(o, [s * 0.5 + rng.range(-0.12, 0.12) * s, s - 3])
    const h = s * rng.range(0.45, 0.96)
    const lean = rng.range(-0.38, 0.38) * s * (h / s)
    const colour = grassColour(rng)
    const width = rng.range(3.5, 7)
    blade(c, rng, base, h, lean, width, colour)
  }
  if (seedHeads) {
    // flowering stalks (timothy, cocksfoot): thin stems with a dense head near the top
    for (let k = 0; k < 9; k++) {
      const base = add2(o, [s * 0.5 + rng.range(-0.1, 0.1) * s, s - 3])
      const h = s * rng.range(0.7, 0.97)
      const lean = rng.range(-0.2, 0.2) * s
      const top = blade(c, rng, base, h, lean, 2.2, [0.07, 0.085, 0.035])
      const from = add2(base, [lean * 0.8, -h * 0.8])
      let dir = norm2([top[0] - from[0], top[1] - from[1]])
      if (dir[0] === 0 && dir[1] === 0) dir = [0, -1]
      const head = mul3([0.11, 0.1, 0.06], rng.range(0.8, 1.2))
      for (let i = 0; i < 14; i++) {
        const p = add2(top, mul2(dir, -i * 3.5))
        c.ellipse(p, dir, 3.2, 2.2, mul3(head, rng.range(0.85, 1.15)), 3)
      }
    }
  }
}

function flowerHead(c, rng, p, kind) {
  if (kind === 0) {
    // ox-eye daisy: white rays round a yellow disc
    const petals = 14
    for (let k = 0; k < petals; k++) {
      const a = (k / petals) * Math.PI * 2 + rng.range(-0.1, 0.1)
      const d = [Math.cos(a), Math.sin(a) * 0.55]
      c.ellipse(add2(p, mul2(d, 9)), d, 7, 2.4, mul3([0.72, 0.72, 0.66], rng.range(0.9, 1.05)), 3)
    }
    c.ellipse(p, [1, 0], 4.5, 3.2, [0.55, 0.38, 0.03], 4)
  } else if (kind === 1) {
    // buttercup: five glossy yellow petals
    for (let k = 0; k < 5; k++) {
      const a = (k / 5) * Math.PI * 2
      const d = [Math.cos(a), Math.sin(a) * 0.7]
      c.ellipse(add2(p, mul2(d, 4)), d, 4.5, 3.5, [0.62, 0.45, 0.02], 3)
    }
  } else {
    // red campion: five notched pink petals
    for (let k = 0; k < 5; k++) {
      const a = (k / 5) * Math.PI * 2 + 0.3
      const d = [Math.cos(a), Math.sin(a) * 0.7]
      c.ellipse(add2(p, mul2(d, 5)), d, 5.5, 3, [0.5, 0.08, 0.17], 3)
    }
    c.ellipse(p, [1, 0], 2, 2, [0.3, 0.05, 0.1], 4)
  }
}

function flowers(c, o, rng) {
  const s = R
  // a few grass blades behind, so the clump sits in the verge
  for (let k = 0; k < 40; k++) {
    const base = add2(o, [s * 0.5 + rng.range(-0.3, 0.3) * s, s - 3])
    const colour = grassColour(rng)
    const h = s * rng.range(0.25, 0.55), lean = rng.range(-0.2, 0.2) * s * 0.4, width = rng.range(3, 5)
    blade(c, rng, base, h, lean, width, colour)
  }
  for (let i = 0; i < 16; i++) {
    const kind = [0, 0, 0, 1, 1, 2][i % 6]
    const base = add2(o, [s * 0.5 + rng.range(-0.33, 0.33) * s, s - 3])
    const h = s * rng.range(0.4, 0.9)
    const lean = rng.range(-0.12, 0.12) * s
    const top = blade(c, rng, base, h, lean, 2.6, [0.05, 0.08, 0.025])
    // a leaf or two up the stem
    for (let k = 0; k < 2; k++) {
      const t = rng.range(0.2, 0.6)
      const at = lerp2(base, top, t)
      const side = rng.f() < 0.5 ? -1 : 1
      c.ellipse(add2(at, [side * 9, -4]), [side, -0.6], 11, 3, [0.05, 0.09, 0.03], 2)
    }
    flowerHead(c, rng, top, kind)
  }
}

function shrub(c, o, rng) {
  const s = R
  const twig = [0.05, 0.035, 0.025]
  for (let k = 0; k < 16; k++) {
    const base = add2(o, [s * 0.5 + rng.range(-0.08, 0.08) * s, s - 3])
    const a = rng.range(-1.1, 1.1)
    const len = s * rng.range(0.45, 0.9)
    const end = add2(base, mul2([Math.sin(a), -Math.cos(a)], len))
    c.stroke(base, end, 4, 1.2, twig, 1)
    const n = Math.trunc(len / 13)
    for (let j = 2; j < n; j++) {
      const p = lerp2(base, end, j / n)
      const side = j % 2 === 0 ? -1 : 1
      const d = norm2([side * 0.8, -0.6])
      const leaf = mul3([0.045, 0.08, 0.03], rng.range(0.7, 1.3))
      c.ellipse(add2(p, mul2(d, 9)), d, 10, 6.5, leaf, 2 + rng.f())
    }
  }
}

/** The atlas: [albedo RGBA8 sRGB, normal RGBA8]. */
export function atlas() {
  const c = new Canvas(ATLAS, ATLAS)
  const regions = [
    [(c, o, rng) => tuft(c, o, rng, 150, false), [0, 0], 31],
    [(c, o, rng) => tuft(c, o, rng, 110, true), [R, 0], 32],
    [flowers, [0, R], 33],
    [shrub, [R, R], 34],
  ]
  for (const [paint, o, seed] of regions) {
    c.clip = [o[0] + 2, o[1] + 2, o[0] + R - 2, o[1] + R - 2]
    paint(c, o, new Rng(seed))
  }
  c.clip = [0, 0, ATLAS, ATLAS]
  c.bleed(6)
  return c.finish(0.9, false)
}
