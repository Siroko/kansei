// Procedural tree meshes, 1 m tall (instances scale them), with the trunk along +Y from the
// origin: a port of the Rust example's tree_meshes.rs, itself the Raggare intro's. Each tree is two
// meshes: bark (opaque) and foliage (alpha-tested cards on the atlas of tree_textures.js).
// `position.w` carries an ambient-occlusion factor (1 = open, lower = inside the crown); foliage
// normals are bent toward the crown's outward direction for soft shading.
//
// Norway spruce (Picea abies), forest-grown: a straight tapering trunk, bare with dead stubs for
// the lower third, then a narrow conical crown of whorled branches that rise near the top, run
// level in the middle and droop lower down, carrying flat sprays with pendulous branchlets hanging
// beneath ("comb" spruce), and a vertical leader at the tip. Silver birch (Betula pendula): a
// slender, slightly wavering white trunk, steep ascending limbs from about a third of the height,
// weeping leafy twigs hanging from them and a few leafy fans at the limb ends and the top.
import { Rng, add3, sub3, mul3, cross3, dot3, norm3, len3, lerp3 } from './canvas.js'
import { SPRUCE_SPRAY, SPRUCE_CURTAIN, BIRCH_HANGING, BIRCH_SPRAY } from './tree_textures.js'

const Y = [0, 1, 0]

/** Vertices as interleaved floats (position xyzw, normal xyz, uv: 9 a vertex) and indices. */
export class Mesh {
  constructor() {
    this.v = []
    this.indices = []
  }
  get vertexCount() {
    return this.v.length / 9
  }
  triangles() {
    return this.indices.length / 3
  }
  vertex(p, ao, n, uv) {
    this.v.push(p[0], p[1], p[2], ao, n[0], n[1], n[2], uv[0], uv[1])
    return this.vertexCount - 1
  }

  /** A card along a polyline `spine` (base first), `halfWidths` across it along `across` (one per
   * spine point), textured with `region` (s across, t from `t0` at the base to `t1` at the end of
   * the spine). Normals blend the card normal with `outward(p)`. */
  ribbon(spine, across, halfWidths, region, t0, t1, ao, outward, bend) {
    const base = this.vertexCount
    const n = spine.length
    for (let i = 0; i < n; i++) {
      const p = spine[i], hw = halfWidths[i]
      const f = i / (n - 1)
      const along = i + 1 < n ? sub3(spine[i + 1], p) : sub3(p, spine[i - 1])
      let cardN = norm3(cross3(along, across))
      if (dot3(cardN, outward(p)) < 0) cardN = mul3(cardN, -1)
      const normal = norm3(lerp3(cardN, outward(p), bend))
      const t = t0 + (t1 - t0) * f
      this.vertex(sub3(p, mul3(across, hw)), ao(f), normal, region.uv(0, t))
      this.vertex(add3(p, mul3(across, hw)), ao(f), normal, region.uv(1, t))
    }
    for (let i = 0; i < n - 1; i++) {
      const a = base + 2 * i, b = a + 1, c = a + 2, d = a + 3
      this.indices.push(a, b, c, c, b, d)
    }
  }

  /** A tube along `spine` with radii `radii`, `sides` around; UV u around (`uOffset` + 0..1),
   * v = `vScale` times the distance along the spine (tiled bark). */
  tube(spine, radii, sides, vScale, uOffset, ao, capEnd) {
    const base = this.vertexCount
    const n = spine.length
    let dist = 0
    for (let i = 0; i < n; i++) {
      const p = spine[i]
      if (i > 0) dist += len3(sub3(p, spine[i - 1]))
      const dir = norm3(i + 1 < n ? sub3(spine[i + 1], p) : sub3(p, spine[i - 1]))
      const side = norm3(cross3(Math.abs(dir[1]) > 0.95 ? [1, 0, 0] : Y, dir))
      const up = cross3(dir, side)
      for (let k = 0; k <= sides; k++) {
        const a = (k / sides) * Math.PI * 2
        const radial = add3(mul3(side, Math.cos(a)), mul3(up, Math.sin(a)))
        this.vertex(add3(p, mul3(radial, radii[i])), ao(i / (n - 1)), radial, [uOffset + k / sides, dist * vScale])
      }
    }
    const ring = sides + 1
    for (let i = 0; i < n - 1; i++) {
      for (let k = 0; k < sides; k++) {
        const a = base + i * ring + k
        const b = a + ring
        // winds outward (counter-clockwise seen from outside)
        this.indices.push(a, a + 1, b, a + 1, b + 1, b)
      }
    }
    if (capEnd) {
      const last = spine[n - 1]
      const tip = this.vertex(last, ao(1), norm3(sub3(last, spine[n - 2])), [uOffset + 0.5, dist * vScale])
      const r0 = base + (n - 1) * ring
      for (let k = 0; k < sides; k++) this.indices.push(r0 + k, r0 + k + 1, tip)
    }
  }
}

const SPRUCE_LODS = [
  { whorls: 30, perWhorl: 6, segments: 3, curtains: true, trunkSides: 8, stubs: true, cardScale: 1.1 },
  { whorls: 15, perWhorl: 5, segments: 2, curtains: true, trunkSides: 6, stubs: false, cardScale: 1.35 },
  { whorls: 11, perWhorl: 5, segments: 1, curtains: false, trunkSides: 4, stubs: false, cardScale: 1.6 },
]

/** Crown shapes (fractions of the tree's height): interior trees (slim and full, mixed at
 * random) and forest-edge trees, which keep live branches almost to the ground. */
export const SPRUCE_STYLES = [
  { crownBase: 0.3, crownRadius: 0.14, droop: 0.2, seed: 7 },
  { crownBase: 0.22, crownRadius: 0.165, droop: 0.25, seed: 17 },
  { crownBase: 0.07, crownRadius: 0.19, droop: 0.32, seed: 27 },
]
export const SPRUCE_EDGE_STYLE = 2

function spruceTrunkRadius(y) {
  const flare = 1 + 0.7 * Math.exp(-y * 45)
  return (0.0105 * Math.pow(1 - y, 0.9) + 0.0012) * flare
}

const GOLDEN = 2.3999632

/** { bark, foliage } of a spruce at `lod` (0-2) in `style`. */
export function spruce(lod, style) {
  const l = SPRUCE_LODS[Math.min(lod, 2)]
  const rng = new Rng(style.seed)
  const bark = new Mesh()
  const foliage = new Mesh()
  const cb = style.crownBase

  // trunk: denser rings near the flared base
  const rings = lod === 0 ? 24 : lod === 1 ? 10 : 4
  const top = lod === 2 ? cb + 0.15 : 0.985
  const ys = Array.from({ length: rings + 1 }, (_, i) => top * Math.pow(i / rings, 1.3))
  bark.tube(ys.map((y) => [0, y, 0]), ys.map(spruceTrunkRadius), l.trunkSides, 12, 0, (f) => (f * top < cb ? 1 : 0.55), lod < 2)

  // dead stubs on the bare bole
  if (l.stubs) {
    for (let k = 0; k < 14; k++) {
      const y = rng.range(0.05, Math.max(cb, 0.1))
      const a = rng.range(0, Math.PI * 2)
      const dir = norm3([Math.cos(a), rng.range(-0.5, 0.1), Math.sin(a)])
      const r = spruceTrunkRadius(y)
      const base = add3([0, y, 0], mul3(dir, r * 0.8))
      // snapped short: 10-30 cm on a 20 m tree
      const len = rng.range(0.005, 0.015)
      bark.tube([base, add3(base, mul3(dir, len))], [0.0012, 0.0005], 3, 12, 0, () => 0.8, false)
    }
  }

  const crownCentre = [0, cb + (1 - cb) * 0.3, 0]
  const outward = (p) => {
    const d = sub3(p, crownCentre)
    return norm3([d[0], d[1] * 0.35 + 0.15, d[2]])
  }

  // whorls of branches: rising near the top, level mid-crown, drooping low down
  let azimuth = rng.range(0, Math.PI * 2)
  for (let w = 0; w < l.whorls; w++) {
    const f = (w + rng.range(0, 0.6)) / l.whorls
    const y = cb + (0.965 - cb) * f
    const t = (1 - y) / (1 - cb) // 0 at the top, 1 at the crown base
    const reach = style.crownRadius * Math.pow(t, 0.8) + 0.02
    for (let b = 0; b < l.perWhorl; b++) {
      azimuth += GOLDEN + rng.range(-0.3, 0.3)
      const len = reach * rng.range(0.8, 1.15)
      const h = [Math.cos(azimuth), 0, Math.sin(azimuth)]
      const rise = Math.tan(((25 - 35 * t + rng.range(-6, 6)) * Math.PI) / 180)
      const droop = 0.06 + style.droop * t
      const start = add3([0, y, 0], mul3(h, spruceTrunkRadius(y) * 0.8))
      const segs = l.segments
      const spine = Array.from({ length: segs + 1 }, (_, i) => {
        const s = i / segs
        const p = add3(add3(start, mul3(h, len * s)), mul3(Y, len * (rise * s - droop * s * s)))
        return [p[0], Math.max(p[1], 0.03), p[2]] // low edge branches rest just above the ground
      })
      // the spray lies roughly flat, rolled a little around the branch
      const roll = rng.range(-0.35, 0.35) + (b % 2 === 0 ? 0.1 : -0.1)
      const lateral = norm3(cross3(Y, h))
      const across = norm3(add3(mul3(lateral, Math.cos(roll)), mul3(Y, Math.sin(roll))))
      // a floor on the width keeps the short top branches from leaving gaps between whorls
      const hw = Array.from({ length: segs + 1 }, (_, i) => Math.max(len * 0.5, 0.022) * l.cardScale * (1 - (0.25 * i) / segs))
      const crownAo = 0.55 + 0.45 * (1 - t * 0.5)
      const ao = (s) => (0.45 + 0.55 * s) * crownAo
      foliage.ribbon(spine, across, hw, SPRUCE_SPRAY, 1, 0.03, ao, outward, 0.55)
      // a second spray rolled steeply about the branch: a real branch's sprays fan in 3D
      const hw2 = hw.map((x) => x * 0.8)
      const clearOfGround = spine.every((p, i) => p[1] > hw2[i] + 0.005)
      if (lod < 2 && clearOfGround) {
        const steep = norm3(add3(mul3(lateral, Math.cos(roll + 0.95)), mul3(Y, Math.sin(roll + 0.95))))
        foliage.ribbon(spine, steep, hw2, SPRUCE_SPRAY, 1, 0.03, ao, outward, 0.55)
      }

      // pendulous branchlets hanging from the outer part of the lower and middle branches (short,
      // and on every other branch)
      if (l.curtains && t > 0.3 && b % 2 === 0) {
        const from = Math.min(1, segs)
        const hang = len * (0.1 + 0.15 * t) * l.cardScale
        const topLine = spine.slice(from)
        if (topLine.length >= 2) {
          // a vertical curtain: the ribbon runs down from the branch
          const mid = topLine[Math.floor(topLine.length / 2)]
          const n = topLine.length
          const baseIdx = foliage.vertexCount
          const face = norm3(cross3(h, Y))
          const normal = norm3(lerp3(face, outward(mid), 0.7))
          for (let i = 0; i < n; i++) {
            const p = topLine[i]
            const q = [p[0], Math.max(p[1] - hang, 0.005), p[2]]
            const s = i / (n - 1)
            foliage.vertex(p, crownAo * 0.8, normal, SPRUCE_CURTAIN.uv(0.08 + 0.84 * s, 0))
            foliage.vertex(q, crownAo * 0.55, normal, SPRUCE_CURTAIN.uv(0.08 + 0.84 * s, 1))
          }
          for (let i = 0; i < n - 1; i++) {
            const a = baseIdx + 2 * i, b2 = a + 1, c = a + 2, d = a + 3
            foliage.indices.push(a, b2, c, c, b2, d)
          }
        }
      }
    }
  }

  // the leader: two crossed vertical sprays at the tip
  const leaderBase = 0.93
  for (let k = 0; k < 2; k++) {
    const a = azimuth + (k * Math.PI) / 2
    const across = [Math.cos(a), 0, Math.sin(a)]
    foliage.ribbon([[0, leaderBase, 0], [0, 1, 0]], across, [0.02 * l.cardScale, 0.012 * l.cardScale], SPRUCE_SPRAY, 1, 0.03, () => 1, outward, 0.5)
  }
  return { bark, foliage }
}

const BIRCH_LODS = [
  { limbs: 16, clusters: 9, trunkSides: 8, limbSides: 4, cardScale: 1.15 },
  { limbs: 10, clusters: 4, trunkSides: 6, limbSides: 3, cardScale: 1.4 },
  { limbs: 5, clusters: 1, trunkSides: 4, limbSides: 0, cardScale: 1.7 },
]

/** { bark, foliage } of a birch at `lod` (0-2). */
export function birch(lod, seed) {
  const l = BIRCH_LODS[Math.min(lod, 2)]
  const rng = new Rng(seed)
  const bark = new Mesh()
  const foliage = new Mesh()
  const cb = 0.34
  const phase = rng.range(0, 6)
  const axis = (y) => [0.018 * Math.sin(y * 2.7 + phase) * y, y, 0.012 * Math.cos(y * 3.3 + phase) * y]
  const radius = (y) => (0.0115 * Math.pow(1 - y, 1.1) + 0.0012) * (1 + 0.5 * Math.exp(-y * 40))

  const rings = lod === 0 ? 16 : lod === 1 ? 8 : 4
  const ys = Array.from({ length: rings + 1 }, (_, i) => (0.93 * i) / rings)
  bark.tube(ys.map(axis), ys.map(radius), l.trunkSides, 12, 0, (f) => (f < 0.4 ? 1 : 0.7), lod < 2)

  const crownCentre = [0, cb + (1 - cb) * 0.45, 0]
  const outward = (p) => {
    const d = sub3(p, crownCentre)
    return norm3([d[0], d[1] * 0.6 + 0.2, d[2]])
  }

  let azimuth = rng.range(0, Math.PI * 2)
  for (let i = 0; i < l.limbs; i++) {
    azimuth += GOLDEN + rng.range(-0.25, 0.25)
    const f = (i + rng.range(0, 0.8)) / l.limbs
    const y = cb + (0.86 - cb) * f
    const len = 0.24 * (1 - 0.45 * f) * rng.range(0.8, 1.15)
    const h = [Math.cos(azimuth), 0, Math.sin(azimuth)]
    const elev = ((58 - 20 * (1 - f) + rng.range(-8, 8)) * Math.PI) / 180
    const start = axis(y)
    // the limb arcs upward and out, then bends back toward the horizontal at its end
    const limb = Array.from({ length: 5 }, (_, k) => {
      const s = k / 4
      return add3(add3(start, mul3(h, len * s * Math.cos(elev))), mul3(Y, len * s * Math.sin(elev) * (1 - 0.45 * s)))
    })
    if (l.limbSides > 0) {
      const r0 = radius(y) * 0.45
      // limbs: u in 1..2 marks them for the shader's dark twig bark
      bark.tube(limb, Array.from({ length: 5 }, (_, k) => r0 * (1 - (0.8 * k) / 4)), l.limbSides, 12, 1, () => 0.75, false)
    }
    // along the limb: short weeping twigs alternating with leafy fans; a fan at its end
    for (let c = 0; c < l.clusters; c++) {
      const s = 0.35 + (0.6 * (c + 0.5)) / l.clusters
      const k = Math.min(s * 4, 3.999)
      const ki = Math.floor(k)
      const p = lerp3(limb[ki], limb[ki + 1], k - ki)
      const face = rng.range(0, Math.PI * 2)
      const across = [Math.cos(face), 0, Math.sin(face)]
      // mostly leafy fans (a birch's crown is a broadleaf mass), a hanging twig now and then
      if (c % 5 === 0) {
        const hang = 0.1 * rng.range(0.8, 1.2) * l.cardScale
        const width = 0.06 * l.cardScale
        const spine = [p, add3(sub3(p, mul3(Y, hang * 0.5)), mul3(h, 0.01)), add3(sub3(p, mul3(Y, hang)), mul3(h, 0.02))]
        foliage.ribbon(spine, across, [width, width * 1.1, width * 1.2], BIRCH_HANGING, 0, 1, (f) => 0.95 - 0.35 * f, outward, 0.6)
      } else {
        const out = norm3(add3(add3(mul3(h, 0.7), mul3(across, rng.range(-0.5, 0.5))), mul3(Y, rng.range(0.1, 0.5))))
        const side = norm3(cross3(out, Y))
        const fan = 0.13 * rng.range(0.85, 1.2) * l.cardScale
        foliage.ribbon([p, add3(p, mul3(out, fan))], side, [fan * 0.45, fan * 0.6], BIRCH_SPRAY, 1, 0, () => 0.9, outward, 0.6)
      }
    }
    const tip = limb[limb.length - 1]
    const across = norm3(cross3(Y, h))
    const fanLen = 0.14 * l.cardScale
    foliage.ribbon([sub3(tip, mul3(h, fanLen * 0.2)), add3(add3(tip, mul3(h, fanLen * 0.8)), [0, 0.02, 0])], across, [fanLen * 0.45, fanLen * 0.6], BIRCH_SPRAY, 1, 0, () => 1, outward, 0.6)
  }
  // a couple of fans at the top
  for (let k = 0; k < 2; k++) {
    const a = azimuth + (k * Math.PI) / 2
    const across = [Math.cos(a), 0, Math.sin(a)]
    const top = axis(0.86)
    foliage.ribbon([top, add3(top, mul3(Y, 0.1 * l.cardScale))], across, [0.06, 0.08], BIRCH_SPRAY, 1, 0, () => 1, outward, 0.5)
  }
  return { bark, foliage }
}
