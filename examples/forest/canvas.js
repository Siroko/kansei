// A tiny CPU painter for the procedural tree and ground-cover textures: antialiased tapered
// strokes and ellipses that carry colour, coverage and a height (for the normal map), composited
// front-to-back by height. Everything is deterministic (seeded). A port of the Rust example's
// canvas.rs (rust/kansei-wasm/examples/outdoor-gi/src), itself the Raggare intro's.

// 2D and 3D vectors as plain arrays.
export const v2 = (x, y) => [x, y]
export const add2 = (a, b) => [a[0] + b[0], a[1] + b[1]]
export const sub2 = (a, b) => [a[0] - b[0], a[1] - b[1]]
export const mul2 = (a, s) => [a[0] * s, a[1] * s]
export const len2 = (a) => Math.hypot(a[0], a[1])
export const dot2 = (a, b) => a[0] * b[0] + a[1] * b[1]
export const norm2 = (a) => {
  const l = len2(a)
  return l > 0 ? [a[0] / l, a[1] / l] : [0, 0]
}
export const lerp2 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
export const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
export const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
export const mul3 = (a, s) => [a[0] * s, a[1] * s, a[2] * s]
export const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
export const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
export const len3 = (a) => Math.hypot(a[0], a[1], a[2])
export const norm3 = (a) => {
  const l = len3(a)
  return l > 0 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 0, 0]
}
export const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]

/** Small deterministic PRNG (xorshift32), the Rust one's sequence. */
export class Rng {
  constructor(seed) {
    this.s = (Math.imul(seed, 0x9e3779b9) | 1) >>> 0
  }
  nextU32() {
    let x = this.s
    x = (x ^ (x << 13)) >>> 0
    x = (x ^ (x >>> 17)) >>> 0
    x = (x ^ (x << 5)) >>> 0
    this.s = x
    return x
  }
  /** Uniform in [0, 1). */
  f() {
    return (this.nextU32() >>> 8) / (1 << 24)
  }
  /** Uniform in [a, b). */
  range(a, b) {
    return a + (b - a) * this.f()
  }
}

export class Canvas {
  constructor(w, h) {
    this.w = w
    this.h = h
    this.color = new Float32Array(w * h * 3)
    this.alpha = new Float32Array(w * h)
    this.height = new Float32Array(w * h)
    /** Painting is clipped to this pixel rectangle [x0, y0, x1, y1), exclusive of x1/y1. */
    this.clip = [0, 0, w, h]
  }

  plot(x, y, cov, color, height) {
    const [cx0, cy0, cx1, cy1] = this.clip
    if (x < cx0 || y < cy0 || x >= cx1 || y >= cy1) return
    const i = y * this.w + x
    // front-to-back by height: a higher element paints over, a lower one only fills gaps
    if (height >= this.height[i] || this.alpha[i] < 0.5) {
      const c = this.color
      c[3 * i] += (color[0] - c[3 * i]) * cov
      c[3 * i + 1] += (color[1] - c[3 * i + 1]) * cov
      c[3 * i + 2] += (color[2] - c[3 * i + 2]) * cov
      if (cov > 0.5) this.height[i] = height
    }
    this.alpha[i] = Math.max(this.alpha[i], cov)
  }

  /** A tapered capsule from `a` (width `wa`) to `b` (width `wb`), in pixels, with a rounded
   * height profile peaking at `height`. */
  stroke(a, b, wa, wb, color, height) {
    const rMax = Math.max(wa, wb) * 0.5 + 1
    const x0 = Math.max(Math.floor(Math.min(a[0], b[0]) - rMax), 0)
    const x1 = Math.min(Math.ceil(Math.max(a[0], b[0]) + rMax), this.w - 1)
    const y0 = Math.max(Math.floor(Math.min(a[1], b[1]) - rMax), 0)
    const y1 = Math.min(Math.ceil(Math.max(a[1], b[1]) + rMax), this.h - 1)
    if (x0 > x1 || y0 > y1) return
    const abx = b[0] - a[0], aby = b[1] - a[1]
    const lenSq = Math.max(abx * abx + aby * aby, 1e-6)
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const px = x + 0.5, py = y + 0.5
        const t = Math.min(Math.max(((px - a[0]) * abx + (py - a[1]) * aby) / lenSq, 0), 1)
        const d = Math.hypot(px - (a[0] + abx * t), py - (a[1] + aby * t))
        const r = (wa + (wb - wa) * t) * 0.5
        const cov = Math.min(Math.max(r - d + 0.5, 0), 1)
        if (cov <= 0) continue
        const profile = Math.sqrt(1 - Math.min(d / Math.max(r, 0.5), 1) ** 2)
        this.plot(x, y, cov, mul3(color, 0.75 + 0.25 * profile), height + profile * Math.min(r, 2) * 0.5)
      }
    }
  }

  /** A filled ellipse centred at `c` with half-axes `ra` along `dir` and `rb` across it. */
  ellipse(c, dir, ra, rb, color, height) {
    const r = Math.max(ra, rb) + 1
    const x0 = Math.max(Math.floor(c[0] - r), 0), x1 = Math.min(Math.ceil(c[0] + r), this.w - 1)
    const y0 = Math.max(Math.floor(c[1] - r), 0), y1 = Math.min(Math.ceil(c[1] + r), this.h - 1)
    if (x0 > x1 || y0 > y1) return
    const d = norm2(dir)
    const n = [-d[1], d[0]]
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const px = x + 0.5 - c[0], py = y + 0.5 - c[1]
        const u = (px * d[0] + py * d[1]) / ra
        const v = (px * n[0] + py * n[1]) / rb
        const q = u * u + v * v
        const edge = (1 - Math.sqrt(q)) * Math.min(ra, rb)
        const cov = Math.min(Math.max(edge + 0.5, 0), 1)
        if (cov <= 0) continue
        // a leaf: domed, with a faint midrib
        const dome = Math.sqrt(Math.max(1 - q, 0))
        const rib = (1 - Math.min(Math.abs(v) * rb, 1)) * 0.3
        this.plot(x, y, cov, mul3(color, 0.8 + 0.2 * dome - rib * 0.3), height + dome * 1.5 - rib)
      }
    }
  }

  /** RGBA8 colour (linear albedo, sRGB-encoded: upload as rgba8unorm-srgb) and a tangent-space
   * normal map (x along +u, y along +v, z out of the card), both row-major. */
  finish(normalStrength, wrap) {
    const { w, h } = this
    const rgba = new Uint8Array(w * h * 4)
    const nrm = new Uint8Array(w * h * 4)
    const at = (x, y) => {
      if (wrap) {
        x = ((x % w) + w) % w
        y = ((y % h) + h) % h
      } else {
        x = Math.min(Math.max(x, 0), w - 1)
        y = Math.min(Math.max(y, 0), h - 1)
      }
      return this.height[y * w + x]
    }
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x
        rgba[i * 4] = srgb8(this.color[3 * i])
        rgba[i * 4 + 1] = srgb8(this.color[3 * i + 1])
        rgba[i * 4 + 2] = srgb8(this.color[3 * i + 2])
        rgba[i * 4 + 3] = Math.round(Math.min(Math.max(this.alpha[i], 0), 1) * 255)
        const dx = (at(x + 1, y) - at(x - 1, y)) * 0.5
        const dy = (at(x, y + 1) - at(x, y - 1)) * 0.5
        const n = norm3([-dx * normalStrength, -dy * normalStrength, 1])
        nrm[i * 4] = Math.round((n[0] * 0.5 + 0.5) * 255)
        nrm[i * 4 + 1] = Math.round((n[1] * 0.5 + 0.5) * 255)
        nrm[i * 4 + 2] = Math.round((n[2] * 0.5 + 0.5) * 255)
        nrm[i * 4 + 3] = 255
      }
    }
    return [rgba, nrm]
  }

  /** Dilate colour and height into transparent pixels (a few passes), so bilinear filtering and
   * mips at card edges pick up foliage colour instead of black. */
  bleed(passes) {
    const { w, h } = this
    for (let p = 0; p < passes; p++) {
      const c0 = this.color.slice(), h0 = this.height.slice(), a0 = this.alpha.slice()
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const i = y * w + x
          if (a0[i] > 0) continue
          let sr = 0, sg = 0, sb = 0, hs = 0, n = 0
          for (const [dx, dy] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
            const xx = x + dx, yy = y + dy
            if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue
            const j = yy * w + xx
            if (a0[j] > 0 || c0[3 * j] !== 0 || c0[3 * j + 1] !== 0 || c0[3 * j + 2] !== 0) {
              sr += c0[3 * j]
              sg += c0[3 * j + 1]
              sb += c0[3 * j + 2]
              hs += h0[j]
              n++
            }
          }
          if (n > 0) {
            this.color[3 * i] = sr / n
            this.color[3 * i + 1] = sg / n
            this.color[3 * i + 2] = sb / n
            this.height[i] = hs / n
          }
        }
      }
    }
  }
}

function srgb8(linear) {
  const c = Math.min(Math.max(linear, 0), 1)
  const s = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055
  return Math.round(s * 255)
}

/** Box-filtered mip chain of an RGBA8 image: [width, height, data] per level. With `alphaTest`
 * (a threshold), each level's alpha is rescaled so the share of texels passing the alpha test
 * matches level 0's (keeps alpha-tested foliage from thinning out with distance). */
export function mipChain(w, h, level0, alphaTest) {
  const coverage = (data, scale, t) => {
    let pass = 0
    for (let i = 3; i < data.length; i += 4) if ((data[i] / 255) * scale >= t) pass++
    return pass / (data.length / 4)
  }
  const target = alphaTest !== undefined ? coverage(level0, 1, alphaTest) : undefined
  const levels = [[w, h, level0]]
  for (;;) {
    const [pw, ph, prev] = levels[levels.length - 1]
    if (pw <= 1 && ph <= 1) break
    const nw = Math.max(pw >> 1, 1), nh = Math.max(ph >> 1, 1)
    const next = new Uint8Array(nw * nh * 4)
    for (let y = 0; y < nh; y++) {
      for (let x = 0; x < nw; x++) {
        for (let c = 0; c < 4; c++) {
          let s = 0
          for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
            const sx = Math.min(x * 2 + dx, pw - 1), sy = Math.min(y * 2 + dy, ph - 1)
            s += prev[(sy * pw + sx) * 4 + c]
          }
          next[(y * nw + x) * 4 + c] = (s + 2) >> 2
        }
      }
    }
    if (alphaTest !== undefined) {
      // binary search the alpha scale that restores the coverage
      let lo = 0.5, hi = 8
      for (let k = 0; k < 16; k++) {
        const mid = 0.5 * (lo + hi)
        if (coverage(next, mid, alphaTest) < target) lo = mid
        else hi = mid
      }
      for (let i = 3; i < next.length; i += 4) next[i] = Math.min(next[i] * hi, 255)
    }
    levels.push([nw, nh, next])
  }
  return levels
}
