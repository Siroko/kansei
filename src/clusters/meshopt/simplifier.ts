/**
 * Mesh simplification by quadric-error edge collapses: a TypeScript port of the edge path of
 * `optimesh` 1.1's `simplifier::simplify_with_attributes` (meshoptimizer v1.1, MIT, after Garland
 * and Heckbert, "Surface Simplification Using Quadric Error Metrics", 1997, with Hoppe's
 * attribute quadrics). Ported for the options the cluster graph builder uses: `SIMPLIFY_SPARSE`,
 * `SIMPLIFY_ERROR_ABSOLUTE`, `SIMPLIFY_LOCK_BORDER` and per-vertex `SIMPLIFY_VERTEX_LOCK` /
 * `SIMPLIFY_VERTEX_PRIORITY`; the other options throw. Every float operation is rounded to f32
 * (`Math.fround`), as Rust computes it, so the result is Rust's.
 */

const f = Math.fround;
const F32_MAX = 3.4028234663852886e38;
const NONE = 0xffffffff;

/** Do not move vertices on the mesh's border. */
export const SIMPLIFY_LOCK_BORDER = 1 << 0;
/** Indices reference a small subset of a large vertex buffer: work on that subset only. */
export const SIMPLIFY_SPARSE = 1 << 1;
/** `targetError` (and the error returned) is in the positions' units, not relative to the mesh's extent. */
export const SIMPLIFY_ERROR_ABSOLUTE = 1 << 2;
/** The options this port does not carry (they throw). */
const UNSUPPORTED = (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6);

/** Per-vertex flag: the vertex cannot move. */
export const SIMPLIFY_VERTEX_LOCK = 1 << 0;
/** Per-vertex flag: the vertex prefers to stay (a stronger regularizing quadric). */
export const SIMPLIFY_VERTEX_PRIORITY = 1 << 2;

const KIND_MANIFOLD = 0; // interior vertex, no attribute seam or boundary
const KIND_BORDER = 1; // on a boundary, with exactly two open edges
const KIND_SEAM = 2; // on an attribute seam, with exactly two seam edges
const KIND_COMPLEX = 3; // movable only if all its wedges move to the target
const KIND_LOCKED = 4; // cannot move

/** Whether kind A may collapse onto kind B (rows A, columns B). */
const CAN_COLLAPSE = [
    [1, 1, 1, 1, 1],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0],
];
/** Whether an edge between kinds A and B has a guaranteed opposite half-edge. */
const HAS_OPPOSITE = [
    [1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0],
];

/** Positions of a vertex buffer: vertex `i`'s at `positions[i * stride]` (floats), `count` vertices. */
export interface VertexData {
    positions: Float32Array;
    count: number;
    stride: number;
}

/** Vertex attributes (strided like positions, in floats) with one weight per channel; 0 ignores a channel. */
export interface Attributes {
    data: Float32Array;
    stride: number;
    weights: ArrayLike<number>;
}

/** When simplification stops: at most `targetIndexCount` indices or `targetError` of error; `options`: `SIMPLIFY_*`. */
export interface SimplifyTarget {
    targetIndexCount: number;
    targetError: number;
    options: number;
}

// quadric fields, 11 floats each
const A00 = 0, A11 = 1, A22 = 2, A10 = 3, A20 = 4, A21 = 5, B0 = 6, B1 = 7, B2 = 8, C = 9, W = 10;
const Q = 11;

/** Per-vertex half-edges (the other two corners of each triangle), in compressed-row form. */
class EdgeAdjacency {
    readonly offsets: Uint32Array;
    readonly next: Uint32Array;
    readonly prev: Uint32Array;

    constructor(indexCount: number, vertexCount: number) {
        this.offsets = new Uint32Array(vertexCount + 1);
        this.next = new Uint32Array(indexCount);
        this.prev = new Uint32Array(indexCount);
    }

    /** Rebuild for `indices[0..count]`, welded through `remap` when given. */
    update(indices: Uint32Array, count: number, vertexCount: number, remap: Uint32Array | null): void {
        const o = this.offsets;
        o.fill(0, 1, vertexCount + 1);
        for (let i = 0; i < count; i++) o[1 + (remap ? remap[indices[i]] : indices[i])]++;
        let offset = 0;
        for (let v = 1; v <= vertexCount; v++) {
            const c = o[v];
            o[v] = offset;
            offset += c;
        }
        // cursors live in offsets[1..]: they advance as edges are scattered
        for (let t = 0; t < count; t += 3) {
            const a = remap ? remap[indices[t]] : indices[t];
            const b = remap ? remap[indices[t + 1]] : indices[t + 1];
            const c = remap ? remap[indices[t + 2]] : indices[t + 2];
            let k = o[1 + a]++;
            this.next[k] = b;
            this.prev[k] = c;
            k = o[1 + b]++;
            this.next[k] = c;
            this.prev[k] = a;
            k = o[1 + c]++;
            this.next[k] = a;
            this.prev[k] = b;
        }
        o[0] = 0;
    }

    /** Whether a half-edge a -> b exists. */
    hasEdge(a: number, b: number): boolean {
        for (let k = this.offsets[a]; k < this.offsets[a + 1]; k++) if (this.next[k] === b) return true;
        return false;
    }
}

/** A power-of-two table size with about 25% headroom for `count` entries. */
function hashBuckets(count: number): number {
    let buckets = 1;
    while (buckets < count + (count >> 2)) buckets *= 2;
    return buckets;
}

/** The slot for `key` in `table` (its own when absent), by quadratic probing. */
function hashLookup(table: Uint32Array, hash: (key: number) => number, equal: (item: number, key: number) => boolean, key: number): number {
    const hashmod = table.length - 1;
    let bucket = hash(key) & hashmod;
    for (let probe = 0; probe <= hashmod; probe++) {
        const item = table[bucket];
        if (item === NONE || equal(item, key)) return bucket;
        bucket = (bucket + probe + 1) & hashmod;
    }
    throw new Error('simplifier: hash table is full');
}

/** Rewrites `indices` to a dense numbering of the vertices they reference; returns the dense-to-original map. */
function buildSparseRemap(indices: Uint32Array, vertexCount: number): Uint32Array {
    const filter = new Uint8Array(Math.ceil(vertexCount / 8));
    let unique = 0;
    for (const index of indices) {
        const bit = 1 << (index & 7);
        unique += (filter[index >>> 3] & bit) === 0 ? 1 : 0;
        filter[index >>> 3] |= bit;
    }
    const remap = new Uint32Array(unique);
    let offset = 0;
    const revremap = new Uint32Array(hashBuckets(unique)).fill(NONE);
    const hash = (id: number) => Math.imul(id, 0x5bd1e995) >>> 0;
    const equal = (item: number, key: number) => remap[item] === key;
    for (let i = 0; i < indices.length; i++) {
        const index = indices[i];
        const bucket = hashLookup(revremap, hash, equal, index);
        if (revremap[bucket] === NONE) {
            remap[offset] = index;
            revremap[bucket] = offset++;
        }
        indices[i] = revremap[bucket];
    }
    return remap;
}

/**
 * Welds each vertex to a canonical one at the same position (`remap`), and links co-located
 * vertices into cycles (`wedge`).
 */
function buildPositionRemap(remap: Uint32Array, wedge: Uint32Array, positions: Float32Array, stride: number, sparse: Uint32Array | null): void {
    const count = remap.length;
    const bits = new Uint32Array(positions.buffer, positions.byteOffset, positions.length);
    const base = (i: number) => (sparse ? sparse[i] : i) * stride;
    const hash = (i: number) => {
        const o = base(i);
        let x = bits[o], y = bits[o + 1], z = bits[o + 2];
        // negative zero hashes as zero
        if (x === 0x80000000) x = 0;
        if (y === 0x80000000) y = 0;
        if (z === 0x80000000) z = 0;
        x ^= x >>> 17;
        y ^= y >>> 17;
        z ^= z >>> 17;
        return (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) >>> 0;
    };
    const equal = (a: number, b: number) => {
        const l = base(a), r = base(b);
        return positions[l] === positions[r] && positions[l + 1] === positions[r + 1] && positions[l + 2] === positions[r + 2];
    };
    const table = new Uint32Array(hashBuckets(count)).fill(NONE);
    for (let i = 0; i < count; i++) {
        const slot = hashLookup(table, hash, equal, i);
        if (table[slot] === NONE) table[slot] = i;
        remap[i] = table[slot];
    }
    for (let i = 0; i < count; i++) wedge[i] = i;
    for (let i = 0; i < count; i++) {
        const r = remap[i];
        if (r !== i) {
            wedge[i] = wedge[r];
            wedge[r] = i;
        }
    }
}

/** Each vertex's kind (`KIND_*`) from its open edges and seams, and its border/seam loops. */
function classifyVertices(kind: Uint8Array, loops: Uint32Array, loopback: Uint32Array, adjacency: EdgeAdjacency, remap: Uint32Array, wedge: Uint32Array, lock: Uint8Array | null, sparse: Uint32Array | null, options: number): void {
    const vertexCount = kind.length;
    loops.fill(NONE);
    loopback.fill(NONE);
    // incoming / outgoing open half-edge per vertex, or the vertex itself once it has more than one
    const openinc = loopback, openout = loops;
    for (let vertex = 0; vertex < vertexCount; vertex++) {
        for (let j = adjacency.offsets[vertex]; j < adjacency.offsets[vertex + 1]; j++) {
            const target = adjacency.next[j];
            if (target === vertex) {
                openinc[vertex] = vertex;
                openout[vertex] = vertex;
            } else if (!adjacency.hasEdge(target, vertex)) {
                openinc[target] = openinc[target] === NONE ? vertex : target;
                openout[vertex] = openout[vertex] === NONE ? target : vertex;
            }
        }
    }
    for (let i = 0; i < vertexCount; i++) {
        if (remap[i] === i) {
            if (wedge[i] === i) {
                const openi = openinc[i], openo = openout[i];
                if (openi === NONE && openo === NONE) kind[i] = KIND_MANIFOLD;
                else if (openi !== NONE && openo !== NONE && remap[openi] === remap[openo] && openi !== i) kind[i] = KIND_SEAM;
                else if (openi !== i && openo !== i) kind[i] = KIND_BORDER;
                else kind[i] = KIND_LOCKED;
            } else if (wedge[wedge[i]] === i) {
                const w = wedge[i];
                const openiv = openinc[i], openov = openout[i], openiw = openinc[w], openow = openout[w];
                if (openiv !== NONE && openiv !== i && openov !== NONE && openov !== i && openiw !== NONE && openiw !== w && openow !== NONE && openow !== w) {
                    kind[i] = remap[openiv] === remap[openow] && remap[openov] === remap[openiw] && remap[openiv] !== remap[openov] ? KIND_SEAM : KIND_LOCKED;
                } else {
                    kind[i] = KIND_LOCKED;
                }
            } else {
                kind[i] = KIND_LOCKED;
            }
        } else {
            kind[i] = kind[remap[i]];
        }
    }
    if (lock) {
        for (let i = 0; i < vertexCount; i++) {
            if (lock[sparse ? sparse[i] : i] & SIMPLIFY_VERTEX_LOCK) kind[remap[i]] = KIND_LOCKED;
        }
        for (let i = 0; i < vertexCount; i++) if (kind[remap[i]] === KIND_LOCKED) kind[i] = KIND_LOCKED;
    }
    if (options & SIMPLIFY_LOCK_BORDER) {
        for (let i = 0; i < vertexCount; i++) if (kind[i] === KIND_BORDER) kind[i] = KIND_LOCKED;
    }
}

/** Copies the positions, normalized into [0, 1] by the largest extent, into `result`; returns the extent. */
function rescalePositions(result: Float32Array, positions: Float32Array, vertexCount: number, stride: number, sparse: Uint32Array | null): number {
    const minv = [F32_MAX, F32_MAX, F32_MAX], maxv = [-F32_MAX, -F32_MAX, -F32_MAX];
    for (let i = 0; i < vertexCount; i++) {
        const o = (sparse ? sparse[i] : i) * stride;
        for (let j = 0; j < 3; j++) {
            const v = positions[o + j];
            result[i * 3 + j] = v;
            minv[j] = minv[j] > v ? v : minv[j];
            maxv[j] = maxv[j] < v ? v : maxv[j];
        }
    }
    let extent = 0;
    for (let j = 0; j < 3; j++) {
        const e = f(maxv[j] - minv[j]);
        extent = e < extent ? extent : e;
    }
    const scale = extent === 0 ? 0 : f(1 / extent);
    for (let i = 0; i < vertexCount; i++) {
        for (let j = 0; j < 3; j++) result[i * 3 + j] = f(f(result[i * 3 + j] - minv[j]) * scale);
    }
    return extent;
}

/** Adds quadric `r[ro]` into `q[qo]`. */
function quadricAdd(q: Float32Array, qo: number, r: Float32Array, ro: number): void {
    for (let k = 0; k < Q; k++) q[qo + k] = f(q[qo + k] + r[ro + k]);
}

/** Adds `count` gradients (4 floats each) of `r[ro]` into `g[go]`. */
function gradAddRow(g: Float32Array, go: number, r: Float32Array, ro: number, count: number): void {
    for (let k = 0; k < count * 4; k++) g[go + k] = f(g[go + k] + r[ro + k]);
}

/** The quadric form `q[o]` at point (x, y, z). */
function quadricEval(q: Float32Array, o: number, x: number, y: number, z: number): number {
    let rx = f(q[o + B0] + f(q[o + A10] * y));
    let ry = f(q[o + B1] + f(q[o + A21] * z));
    let rz = f(q[o + B2] + f(q[o + A20] * x));
    rx = f(rx * 2);
    ry = f(ry * 2);
    rz = f(rz * 2);
    rx = f(rx + f(q[o + A00] * x));
    ry = f(ry + f(q[o + A11] * y));
    rz = f(rz + f(q[o + A22] * z));
    let r = q[o + C];
    r = f(r + f(rx * x));
    r = f(r + f(ry * y));
    r = f(r + f(rz * z));
    return r;
}

/** The normalized positional error of quadric `q[o]` at (x, y, z). */
function quadricError(q: Float32Array, o: number, x: number, y: number, z: number): number {
    const r = quadricEval(q, o, x, y, z);
    const s = q[o + W] === 0 ? 0 : f(1 / q[o + W]);
    return f(Math.abs(r) * s);
}

/** The positional and attribute error of attribute quadric `q[o]` and gradients `g[go]` at (x, y, z) with attributes `va[vo]`. */
function quadricErrorAttr(q: Float32Array, o: number, g: Float32Array, go: number, count: number, x: number, y: number, z: number, va: Float32Array, vo: number): number {
    let r = quadricEval(q, o, x, y, z);
    for (let k = 0; k < count; k++) {
        const a = va[vo + k];
        const gk = go + k * 4;
        const grad = f(f(f(f(x * g[gk]) + f(y * g[gk + 1])) + f(z * g[gk + 2])) + g[gk + 3]);
        r = f(r + f(a * f(f(a * q[o + W]) - f(2 * grad))));
    }
    return Math.abs(r);
}

/** Sets `q[o]` to the quadric of plane (a, b, c, d) weighted `w`. */
function quadricFromPlane(q: Float32Array, o: number, a: number, b: number, c: number, d: number, w: number): void {
    const aw = f(a * w), bw = f(b * w), cw = f(c * w), dw = f(d * w);
    q[o + A00] = f(a * aw);
    q[o + A11] = f(b * bw);
    q[o + A22] = f(c * cw);
    q[o + A10] = f(a * bw);
    q[o + A20] = f(a * cw);
    q[o + A21] = f(b * cw);
    q[o + B0] = f(a * dw);
    q[o + B1] = f(b * dw);
    q[o + B2] = f(c * dw);
    q[o + C] = f(d * dw);
    q[o + W] = w;
}

/** Normalizes vector `v` in place, returning its length. */
function normalize(v: number[]): number {
    const length = f(Math.sqrt(f(f(f(v[0] * v[0]) + f(v[1] * v[1])) + f(v[2] * v[2]))));
    if (length > 0) {
        v[0] = f(v[0] / length);
        v[1] = f(v[1] / length);
        v[2] = f(v[2] / length);
    }
    return length;
}

/** `p1 - p0` and `p2 - p0` of positions `p` at vertices `i0`, `i1`, `i2`. */
function edges(p: Float32Array, i0: number, i1: number, i2: number): [number[], number[]] {
    const a = i0 * 3, b = i1 * 3, c = i2 * 3;
    return [
        [f(p[b] - p[a]), f(p[b + 1] - p[a + 1]), f(p[b + 2] - p[a + 2])],
        [f(p[c] - p[a]), f(p[c + 1] - p[a + 1]), f(p[c + 2] - p[a + 2])],
    ];
}

function cross(u: number[], v: number[]): number[] {
    return [f(f(u[1] * v[2]) - f(u[2] * v[1])), f(f(u[2] * v[0]) - f(u[0] * v[2])), f(f(u[0] * v[1]) - f(u[1] * v[0]))];
}

function dot(u: number[], v: number[]): number {
    return f(f(f(u[0] * v[0]) + f(u[1] * v[1])) + f(u[2] * v[2]));
}

/** Sets `q[o]` to triangle (i0, i1, i2)'s plane quadric, weighted by the square root of its area times `weight`. */
function quadricFromTriangle(q: Float32Array, o: number, p: Float32Array, i0: number, i1: number, i2: number, weight: number): void {
    const [p10, p20] = edges(p, i0, i1, i2);
    const normal = cross(p10, p20);
    const area = normalize(normal);
    const distance = f(f(f(normal[0] * p[i0 * 3]) + f(normal[1] * p[i0 * 3 + 1])) + f(normal[2] * p[i0 * 3 + 2]));
    quadricFromPlane(q, o, normal[0], normal[1], normal[2], -distance, f(f(Math.sqrt(area)) * weight));
}

/** Sets `q[o]` to the quadric of the plane through edge (i0, i1) perpendicular to triangle (i0, i1, i2). */
function quadricFromTriangleEdge(q: Float32Array, o: number, p: Float32Array, i0: number, i1: number, i2: number, weight: number): void {
    const [p10, p20] = edges(p, i0, i1, i2);
    const lengthsq = dot(p10, p10);
    const length = f(Math.sqrt(lengthsq));
    const p20p = f(f(f(p20[0] * p10[0]) + f(p20[1] * p10[1])) + f(p20[2] * p10[2]));
    const perp = [0, 1, 2].map((k) => f(f(p20[k] * lengthsq) - f(p10[k] * p20p)));
    normalize(perp);
    const distance = f(f(f(perp[0] * p[i0 * 3]) + f(perp[1] * p[i0 * 3 + 1])) + f(perp[2] * p[i0 * 3 + 2]));
    quadricFromPlane(q, o, perp[0], perp[1], perp[2], -distance, f(length * weight));
}

/**
 * Sets `q[o]` and the gradients `g[0..count * 4]` to triangle (i0, i1, i2)'s attribute quadric:
 * the squared difference of its attributes `va` (`count` each) from the plane they lie on.
 */
function quadricFromAttributes(q: Float32Array, o: number, g: Float32Array, p: Float32Array, i0: number, i1: number, i2: number, va: Float32Array, count: number): void {
    const [v0, v1] = edges(p, i0, i1, i2);
    const normal = cross(v0, v1);
    const w = f(f(Math.sqrt(dot(normal, normal))) * 0.5);
    const d00 = dot(v0, v0);
    const d01 = f(f(f(v0[0] * v1[0]) + f(v0[1] * v1[1])) + f(v0[2] * v1[2]));
    const d11 = dot(v1, v1);
    const denom = f(f(d00 * d11) - f(d01 * d01));
    const denomr = denom === 0 ? 0 : f(1 / denom);
    const g1 = [0, 1, 2].map((k) => f(f(f(d11 * v0[k]) - f(d01 * v1[k])) * denomr));
    const g2 = [0, 1, 2].map((k) => f(f(f(d00 * v1[k]) - f(d01 * v0[k])) * denomr));
    q.fill(0, o, o + Q);
    q[o + W] = w;
    const px = p[i0 * 3], py = p[i0 * 3 + 1], pz = p[i0 * 3 + 2];
    for (let k = 0; k < count; k++) {
        const a0 = va[i0 * count + k], a1 = va[i1 * count + k], a2 = va[i2 * count + k];
        const d1 = f(a1 - a0), d2 = f(a2 - a0);
        const gx = f(f(g1[0] * d1) + f(g2[0] * d2));
        const gy = f(f(g1[1] * d1) + f(g2[1] * d2));
        const gz = f(f(g1[2] * d1) + f(g2[2] * d2));
        const gw = f(f(f(a0 - f(px * gx)) - f(py * gy)) - f(pz * gz));
        q[o + A00] = f(q[o + A00] + f(w * f(gx * gx)));
        q[o + A11] = f(q[o + A11] + f(w * f(gy * gy)));
        q[o + A22] = f(q[o + A22] + f(w * f(gz * gz)));
        q[o + A10] = f(q[o + A10] + f(w * f(gy * gx)));
        q[o + A20] = f(q[o + A20] + f(w * f(gz * gx)));
        q[o + A21] = f(q[o + A21] + f(w * f(gz * gy)));
        q[o + B0] = f(q[o + B0] + f(w * f(gx * gw)));
        q[o + B1] = f(q[o + B1] + f(w * f(gy * gw)));
        q[o + B2] = f(q[o + B2] + f(w * f(gz * gw)));
        q[o + C] = f(q[o + C] + f(w * f(gw * gw)));
        g[k * 4] = f(w * gx);
        g[k * 4 + 1] = f(w * gy);
        g[k * 4 + 2] = f(w * gz);
        g[k * 4 + 3] = f(w * gw);
    }
}

/** Whether triangle ABC (positions `p` at vertices a, b, c) flips when C moves to D. */
function hasTriangleFlip(p: Float32Array, a: number, b: number, c: number, d: number): boolean {
    const [eb, ec] = edges(p, a, b, c);
    const ed = [f(p[d * 3] - p[a * 3]), f(p[d * 3 + 1] - p[a * 3 + 1]), f(p[d * 3 + 2] - p[a * 3 + 2])];
    const nbc = cross(eb, ec), nbd = cross(eb, ed);
    const ndp = dot(nbc, nbd);
    const abc = dot(nbc, nbc), abd = dot(nbd, nbd);
    return ndp <= f(f(0.25) * f(Math.sqrt(f(abc * abd))));
}

/** Whether collapsing (welded) i0 onto i1 flips a triangle around i0. */
function hasTriangleFlipsCollapse(adjacency: EdgeAdjacency, p: Float32Array, collapseRemap: Uint32Array, i0: number, i1: number): boolean {
    for (let k = adjacency.offsets[i0]; k < adjacency.offsets[i0 + 1]; k++) {
        const a = collapseRemap[adjacency.next[k]], b = collapseRemap[adjacency.prev[k]];
        if (a === i1 || b === i1 || a === b) continue;
        if (hasTriangleFlip(p, a, b, i0, i1)) return true;
    }
    return false;
}

/** A complex vertex's wedge `v` resolved to the wedge of `target` its loops lead to. */
function complexTarget(v: number, target: number, remap: Uint32Array, loops: Uint32Array, loopback: Uint32Array): number {
    const r = remap[target];
    if (loops[v] !== NONE && remap[loops[v]] === r) return loops[v];
    if (loopback[v] !== NONE && remap[loopback[v]] === r) return loopback[v];
    return target;
}

/**
 * Simplifies a mesh while preserving weighted vertex attributes: `indices` down toward
 * `target.targetIndexCount`, within `target.targetError`, vertices flagged
 * `SIMPLIFY_VERTEX_LOCK` in `vertexLock` (by original index) never moving. Returns the
 * simplified indices (into the same vertices) and the linear error. Rust:
 * `simplifier::simplify_with_attributes`.
 */
export function simplifyWithAttributes(indices: Uint32Array, vertices: VertexData, attributes: Attributes | null, vertexLock: Uint8Array | null, target: SimplifyTarget): { indices: Uint32Array; error: number } {
    const options = target.options;
    if (options & UNSUPPORTED) throw new Error(`simplifyWithAttributes: options ${options} are not ported (only LOCK_BORDER, SPARSE, ERROR_ABSOLUTE)`);
    const indexCount = indices.length;
    const result = indices.slice();
    let vertexCount = vertices.count;
    const sparse = options & SIMPLIFY_SPARSE ? buildSparseRemap(result, vertexCount) : null;
    if (sparse) vertexCount = sparse.length;

    const adjacency = new EdgeAdjacency(indexCount, vertexCount);
    adjacency.update(result, indexCount, vertexCount, null);
    const remap = new Uint32Array(vertexCount);
    const wedge = new Uint32Array(vertexCount);
    buildPositionRemap(remap, wedge, vertices.positions, vertices.stride, sparse);

    const kind = new Uint8Array(vertexCount);
    const loops = new Uint32Array(vertexCount);
    const loopback = new Uint32Array(vertexCount);
    classifyVertices(kind, loops, loopback, adjacency, remap, wedge, vertexLock, sparse, options);

    const positions = new Float32Array(vertexCount * 3);
    const vertexScale = rescalePositions(positions, vertices.positions, vertexCount, vertices.stride, sparse);

    // the used attribute channels, weighted
    const attributeRemap: number[] = [];
    if (attributes) {
        for (let i = 0; i < attributes.weights.length; i++) if (f(attributes.weights[i]) > 0) attributeRemap.push(i);
    }
    const attributeCount = attributeRemap.length;
    const vertexAttributes = new Float32Array(vertexCount * attributeCount);
    if (attributes) {
        for (let i = 0; i < vertexCount; i++) {
            const o = (sparse ? sparse[i] : i) * attributes.stride;
            attributeRemap.forEach((rk, k) => {
                vertexAttributes[i * attributeCount + k] = f(attributes.data[o + rk] * f(attributes.weights[rk]));
            });
        }
    }

    const vertexQuadrics = new Float32Array(vertexCount * Q);
    const attributeQuadrics = new Float32Array(attributeCount ? vertexCount * Q : 0);
    const attributeGradients = new Float32Array(vertexCount * attributeCount * 4);
    const scratch = new Float32Array(Q), scratch2 = new Float32Array(Q);

    // the triangles' plane quadrics
    for (let t = 0; t < indexCount; t += 3) {
        const i0 = result[t], i1 = result[t + 1], i2 = result[t + 2];
        quadricFromTriangle(scratch, 0, positions, i0, i1, i2, 1);
        quadricAdd(vertexQuadrics, remap[i0] * Q, scratch, 0);
        quadricAdd(vertexQuadrics, remap[i1] * Q, scratch, 0);
        quadricAdd(vertexQuadrics, remap[i2] * Q, scratch, 0);
    }
    // a small regularizing point quadric on each primary vertex
    for (let i = 0; i < vertexCount; i++) {
        if (remap[i] !== i) continue;
        const priority = vertexLock !== null && (vertexLock[sparse ? sparse[i] : i] & SIMPLIFY_VERTEX_PRIORITY) !== 0;
        const w = f(vertexQuadrics[i * Q + W] * (priority ? 1 : f(1e-7)));
        const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
        scratch[A00] = w;
        scratch[A11] = w;
        scratch[A22] = w;
        scratch[A10] = 0;
        scratch[A20] = 0;
        scratch[A21] = 0;
        scratch[B0] = f(-x * w);
        scratch[B1] = f(-y * w);
        scratch[B2] = f(-z * w);
        scratch[C] = f(f(f(f(x * x) + f(y * y)) + f(z * z)) * w);
        scratch[W] = w;
        quadricAdd(vertexQuadrics, i * Q, scratch, 0);
    }
    // border and seam edges resist moving
    const NEXT = [1, 2, 0, 1];
    for (let t = 0; t < indexCount; t += 3) {
        for (let e = 0; e < 3; e++) {
            const i0 = result[t + e], i1 = result[t + NEXT[e]];
            const k0 = kind[i0], k1 = kind[i1];
            const open0 = k0 === KIND_BORDER || k0 === KIND_SEAM, open1 = k1 === KIND_BORDER || k1 === KIND_SEAM;
            if (!open0 && !open1) continue;
            if (open0 && loops[i0] !== i1) continue;
            if (open1 && loopback[i1] !== i0) continue;
            const i2 = result[t + NEXT[e + 1]];
            // a seam edge is visited from both of its half-edges (0.5 each); borders are pinned harder
            const edgeWeight = k0 === KIND_BORDER || k1 === KIND_BORDER ? 10 : 0.5;
            quadricFromTriangleEdge(scratch, 0, positions, i0, i1, i2, edgeWeight);
            quadricFromTriangle(scratch2, 0, positions, i0, i1, i2, edgeWeight);
            // the triangle's terms with no normalization weight
            scratch2[W] = 0;
            quadricAdd(scratch, 0, scratch2, 0);
            quadricAdd(vertexQuadrics, remap[i0] * Q, scratch, 0);
            quadricAdd(vertexQuadrics, remap[i1] * Q, scratch, 0);
        }
    }
    // the triangles' attribute quadrics
    if (attributeCount) {
        const g = new Float32Array(attributeCount * 4);
        for (let t = 0; t < indexCount; t += 3) {
            const i0 = result[t], i1 = result[t + 1], i2 = result[t + 2];
            quadricFromAttributes(scratch, 0, g, positions, i0, i1, i2, vertexAttributes, attributeCount);
            for (const i of [i0, i1, i2]) {
                quadricAdd(attributeQuadrics, i * Q, scratch, 0);
                gradAddRow(attributeGradients, i * attributeCount * 4, g, 0, attributeCount);
            }
        }
    }

    // room for every candidate collapse of a pass
    let dualCount = 0;
    for (let i = 0; i < vertexCount; i++) {
        if (kind[i] === KIND_MANIFOLD || kind[i] === KIND_SEAM) dualCount += adjacency.offsets[i + 1] - adjacency.offsets[i];
    }
    const capacity = indexCount - (dualCount >> 1) + 3;
    const cv0 = new Uint32Array(capacity), cv1 = new Uint32Array(capacity);
    const payload = new Uint32Array(capacity);
    const cerror = new Float32Array(payload.buffer);
    const order = new Uint32Array(capacity);
    const collapseRemap = new Uint32Array(vertexCount);
    const locked = new Uint8Array(vertexCount);
    const histogram = new Uint32Array(2048 + 512);

    let resultCount = indexCount;
    let resultError = 0;
    let vertexError = 0;
    const errorScale = options & SIMPLIFY_ERROR_ABSOLUTE ? vertexScale : 1;
    const targetError = f(target.targetError);
    const errorLimit = f(f(targetError * targetError) / f(errorScale * errorScale));

    const attrErr = (v: number, t: number) => quadricErrorAttr(attributeQuadrics, v * Q, attributeGradients, v * attributeCount * 4, attributeCount,
        positions[t * 3], positions[t * 3 + 1], positions[t * 3 + 2], vertexAttributes, t * attributeCount);
    // a seam vertex's other wedge and where it goes
    const seamPair = (i0: number, i1: number): [number, number] => {
        const s0 = wedge[i0];
        const s1 = loops[i0] === i1 ? loopback[s0] : loops[s0];
        return [s0, s1 !== NONE ? s1 : wedge[i1]];
    };

    while (resultCount > target.targetIndexCount) {
        adjacency.update(result, resultCount, vertexCount, remap);

        // the collapsible edges, each one-way or either way
        let count = 0;
        for (let t = 0; t < resultCount; t += 3) {
            if (count + 3 > capacity) break;
            for (let e = 0; e < 3; e++) {
                const i0 = result[t + e], i1 = result[t + NEXT[e]];
                if (remap[i0] === remap[i1]) continue;
                const k0 = kind[i0], k1 = kind[i1];
                if ((CAN_COLLAPSE[k0][k1] | CAN_COLLAPSE[k1][k0]) === 0) continue;
                if (HAS_OPPOSITE[k0][k1] !== 0 && remap[i1] > remap[i0]) continue;
                if ((k0 === KIND_BORDER || k0 === KIND_SEAM) && k1 !== KIND_MANIFOLD && loops[i0] !== i1) continue;
                if ((k1 === KIND_BORDER || k1 === KIND_SEAM) && k0 !== KIND_MANIFOLD && loopback[i1] !== i0) continue;
                if ((CAN_COLLAPSE[k0][k1] & CAN_COLLAPSE[k1][k0]) !== 0) {
                    cv0[count] = i0;
                    cv1[count] = i1;
                    payload[count] = 1;
                } else {
                    const forward = CAN_COLLAPSE[k0][k1] !== 0;
                    cv0[count] = forward ? i0 : i1;
                    cv1[count] = forward ? i1 : i0;
                    payload[count] = 0;
                }
                count++;
            }
        }
        if (count === 0) break;

        // each collapse's error, in its cheaper direction
        for (let c = 0; c < count; c++) {
            const i0 = cv0[c], i1 = cv1[c];
            const bidi = payload[c] !== 0;
            let ei = quadricError(vertexQuadrics, remap[i0] * Q, positions[i1 * 3], positions[i1 * 3 + 1], positions[i1 * 3 + 2]);
            let ej = bidi ? quadricError(vertexQuadrics, remap[i1] * Q, positions[i0 * 3], positions[i0 * 3 + 1], positions[i0 * 3 + 2]) : F32_MAX;
            if (attributeCount) {
                ei = f(ei + attrErr(i0, i1));
                if (bidi) ej = f(ej + attrErr(i1, i0));
                if (kind[i0] === KIND_SEAM) {
                    const [s0, s1] = seamPair(i0, i1);
                    ei = f(ei + attrErr(s0, s1));
                    if (bidi) ej = f(ej + attrErr(s1, s0));
                } else {
                    if (kind[i0] === KIND_COMPLEX) {
                        for (let v = wedge[i0]; v !== i0; v = wedge[v]) ei = f(ei + attrErr(v, complexTarget(v, i1, remap, loops, loopback)));
                    }
                    if (kind[i1] === KIND_COMPLEX && bidi) {
                        for (let v = wedge[i1]; v !== i1; v = wedge[v]) ej = f(ej + attrErr(v, complexTarget(v, i0, remap, loops, loopback)));
                    }
                }
            }
            const rev = bidi && ej < ei;
            cv0[c] = rev ? i1 : i0;
            cv1[c] = rev ? i0 : i1;
            cerror[c] = ej < ei ? ej : ei;
        }

        // counting sort by the top 12 bits of the errors (8 exponent, 4 mantissa)
        histogram.fill(0);
        const bin = (bits: number) => Math.min((bits << 1) >>> 20, histogram.length - 1);
        for (let c = 0; c < count; c++) histogram[bin(payload[c])]++;
        let sum = 0;
        for (let h = 0; h < histogram.length; h++) {
            const n = histogram[h];
            histogram[h] = sum;
            sum += n;
        }
        for (let c = 0; c < count; c++) order[histogram[bin(payload[c])]++] = c;

        // collapse in error order, within the budget, without flipping or touching a locked vertex
        const triangleGoal = Math.floor((resultCount - target.targetIndexCount) / 3);
        for (let i = 0; i < vertexCount; i++) collapseRemap[i] = i;
        locked.fill(0);
        let edgeCollapses = 0, triangleCollapses = 0;
        let edgeCollapseGoal = triangleGoal >> 1;
        for (let k = 0; k < count; k++) {
            const c = order[k];
            const error = cerror[c];
            if (error > errorLimit) break;
            if (triangleCollapses >= triangleGoal) break;
            const errorGoal = edgeCollapseGoal < count ? f(f(1.5) * cerror[order[edgeCollapseGoal]]) : F32_MAX;
            if (error > errorGoal && error > resultError && triangleCollapses > Math.floor(triangleGoal / 6)) break;
            const i0 = cv0[c], i1 = cv1[c];
            const r0 = remap[i0], r1 = remap[i1];
            const k0 = kind[i0];
            if ((locked[r0] | locked[r1]) !== 0) continue;
            if (hasTriangleFlipsCollapse(adjacency, positions, collapseRemap, r0, r1)) {
                edgeCollapseGoal++;
                continue;
            }
            if (k0 === KIND_COMPLEX) {
                let v = i0;
                do {
                    collapseRemap[v] = complexTarget(v, i1, remap, loops, loopback);
                    v = wedge[v];
                } while (v !== i0);
            } else if (k0 === KIND_SEAM) {
                const [s0, s1] = seamPair(i0, i1);
                collapseRemap[i0] = i1;
                collapseRemap[s0] = s1;
            } else {
                collapseRemap[i0] = i1;
            }
            locked[r0] = 1;
            locked[r1] = 1;
            triangleCollapses += k0 === KIND_BORDER ? 1 : 2;
            edgeCollapses++;
            resultError = resultError < error ? error : resultError;
        }
        if (edgeCollapses === 0) break;

        // the collapsed vertices' quadrics into their targets
        for (let i0 = 0; i0 < vertexCount; i0++) {
            const i1 = collapseRemap[i0];
            if (i1 === i0) continue;
            const r0 = remap[i0], r1 = remap[i1];
            if (i0 === r0) quadricAdd(vertexQuadrics, r1 * Q, vertexQuadrics, r0 * Q);
            if (attributeCount) {
                quadricAdd(attributeQuadrics, i1 * Q, attributeQuadrics, i0 * Q);
                gradAddRow(attributeGradients, i1 * attributeCount * 4, attributeGradients, i0 * attributeCount * 4, attributeCount);
                if (i0 === r0) {
                    const derr = quadricError(vertexQuadrics, r0 * Q, positions[r1 * 3], positions[r1 * 3 + 1], positions[r1 * 3 + 2]);
                    vertexError = vertexError < derr ? derr : vertexError;
                }
            }
        }
        vertexError = attributeCount === 0 ? resultError : vertexError;

        for (const l of [loops, loopback]) {
            for (let i = 0; i < l.length; i++) {
                if (l[i] === NONE) continue;
                const r = collapseRemap[l[i]];
                l[i] = i === r ? (l[l[i]] !== NONE ? collapseRemap[l[l[i]]] : NONE) : r;
            }
        }

        // the index buffer through the collapses, degenerate triangles dropped
        let write = 0;
        for (let t = 0; t < resultCount; t += 3) {
            const v0 = collapseRemap[result[t]], v1 = collapseRemap[result[t + 1]], v2 = collapseRemap[result[t + 2]];
            const r0 = remap[v0], r1 = remap[v1], r2 = remap[v2];
            if (r0 !== r1 && r0 !== r2 && r1 !== r2) {
                result[write] = v0;
                result[write + 1] = v1;
                result[write + 2] = v2;
                write += 3;
            }
        }
        resultCount = write;
    }

    const out = result.slice(0, resultCount);
    if (sparse) for (let i = 0; i < resultCount; i++) out[i] = sparse[out[i]];
    return { indices: out, error: f(f(Math.sqrt(resultError)) * errorScale) };
}
