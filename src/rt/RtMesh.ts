import { Geometry } from '../buffers/Geometry';

type Vec3 = [number, number, number];

/** Floats of a geometry's interleaved vertex: position (vec4), normal (vec3), uv (vec2). */
const VERTEX_FLOATS = 9;
/** Words before an `RtMesh`'s vertices: where its vertices and indices start, how many vertices and triangles it has. */
const HEADER_WORDS = 4;

/**
 * A mesh for `RtGrid.gather`: its vertices' positions and uvs, its triangles, and its bounds.
 * Rust: `rt::RtMesh`.
 */
export class RtMesh {
    constructor(
        /** x, y, z a vertex. */
        public readonly positions: Float32Array,
        /** u, v a vertex. */
        public readonly uvs: Float32Array,
        /** Three vertex indices a triangle. */
        public readonly indices: Uint32Array,
        public readonly min: Vec3,
        public readonly max: Vec3,
    ) { }

    /** `geometry`'s CPU vertices and indices (empty when it keeps none). */
    static fromGeometry(geometry: Geometry): RtMesh {
        const v = geometry.vertices ?? new Float32Array(0);
        const count = Math.floor(v.length / VERTEX_FLOATS);
        const positions = new Float32Array(count * 3);
        const uvs = new Float32Array(count * 2);
        const min: Vec3 = [Infinity, Infinity, Infinity];
        const max: Vec3 = [-Infinity, -Infinity, -Infinity];
        for (let i = 0; i < count; i++) {
            const at = i * VERTEX_FLOATS;
            for (let c = 0; c < 3; c++) {
                const x = v[at + c];
                positions[i * 3 + c] = x;
                min[c] = Math.min(min[c], x);
                max[c] = Math.max(max[c], x);
            }
            uvs[i * 2] = v[at + 7];
            uvs[i * 2 + 1] = v[at + 8];
        }
        if (count === 0) {
            min.fill(0);
            max.fill(0);
        }
        return new RtMesh(positions, uvs, Uint32Array.from(geometry.indices ?? []), min, max);
    }

    get vertexCount(): number {
        return this.positions.length / 3;
    }

    get triangleCount(): number {
        return Math.floor(this.indices.length / 3);
    }

    /**
     * The mesh as the gather reads it: a header (where the vertices and indices start, the vertex
     * and triangle counts), the vertices (x, y, z and the uv as two f16), the indices.
     */
    gpuWords(): Uint32Array {
        const n = this.vertexCount;
        const vertices = HEADER_WORDS;
        const indices = vertices + n * 4;
        const words = new Uint32Array(indices + this.indices.length);
        const floats = new Float32Array(words.buffer);
        words.set([vertices, indices, n, this.triangleCount]);
        for (let i = 0; i < n; i++) {
            const at = vertices + i * 4;
            floats[at] = this.positions[i * 3];
            floats[at + 1] = this.positions[i * 3 + 1];
            floats[at + 2] = this.positions[i * 3 + 2];
            words[at + 3] = packHalf2(this.uvs[i * 2], this.uvs[i * 2 + 1]);
        }
        words.set(this.indices, indices);
        return words;
    }

    /** A STORAGE buffer of `gpuWords`. */
    createBuffer(device: GPUDevice): GPUBuffer {
        const words = this.gpuWords();
        const buffer = device.createBuffer({ label: 'RtMesh', size: Math.max(words.byteLength, 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(buffer, 0, words);
        return buffer;
    }
}

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

/**
 * An f32 as IEEE half bits, rounded to nearest even (overflow to infinity, small values to
 * subnormals or zero), as WGSL's `pack2x16float` packs it.
 */
export function halfBits(x: number): number {
    f32Scratch[0] = x;
    const b = u32Scratch[0];
    const sign = (b >>> 16) & 0x8000;
    const abs = b & 0x7fffffff;
    if (abs >= 0x7f800000) {
        // infinity, NaN
        return sign | 0x7c00 | (abs > 0x7f800000 ? 0x200 : 0);
    }
    const exp = (abs >>> 23) - 127 + 15;
    if (exp >= 31) return sign | 0x7c00;
    if (exp <= 0) {
        if (exp < -10) return sign;
        // a subnormal half
        const mant = (abs & 0x7fffff) | 0x800000;
        const shift = 14 - exp;
        const half = mant >>> shift;
        const rest = mant & ((1 << shift) - 1);
        const mid = 1 << (shift - 1);
        const round = rest > mid || (rest === mid && (half & 1) === 1) ? 1 : 0;
        return sign | (half + round);
    }
    const mant = abs & 0x7fffff;
    const half = (exp << 10) | (mant >>> 13);
    const rest = mant & 0x1fff;
    const round = rest > 0x1000 || (rest === 0x1000 && (half & 1) === 1) ? 1 : 0;
    return sign | (half + round);
}

/** Two f32 as WGSL's `pack2x16float` packs them. */
export function packHalf2(a: number, b: number): number {
    return (halfBits(a) | (halfBits(b) << 16)) >>> 0;
}

/**
 * miaumiau.cat/?p=1457's split, at load: every triangle with an edge longer than `maxEdge` cut
 * into four at its edges' midpoints, again until none is (midpoints shared, so no cracks). A
 * grid's build then scatters each triangle into a few cells, at the cost of more triangles.
 * Rust: `rt::split_large_triangles`.
 */
export function splitLargeTriangles(geometry: Geometry, maxEdge: number): Geometry {
    const vertices: number[] = Array.from(geometry.vertices ?? []);
    const midpoints = new Map<string, number>();
    const mid = (a: number, b: number): number => {
        const key = `${Math.min(a, b)},${Math.max(a, b)}`;
        let m = midpoints.get(key);
        if (m === undefined) {
            m = vertices.length / VERTEX_FLOATS;
            for (let k = 0; k < VERTEX_FLOATS; k++) vertices.push((vertices[a * VERTEX_FLOATS + k] + vertices[b * VERTEX_FLOATS + k]) * 0.5);
            midpoints.set(key, m);
        }
        return m;
    };
    const dist = (a: number, b: number) => Math.hypot(
        vertices[a * VERTEX_FLOATS] - vertices[b * VERTEX_FLOATS],
        vertices[a * VERTEX_FLOATS + 1] - vertices[b * VERTEX_FLOATS + 1],
        vertices[a * VERTEX_FLOATS + 2] - vertices[b * VERTEX_FLOATS + 2],
    );
    const source = geometry.indices ?? [];
    const pending: [number, number, number][] = [];
    for (let t = source.length - 3; t >= 0; t -= 3) pending.push([source[t], source[t + 1], source[t + 2]]);
    const indices: number[] = [];
    while (pending.length > 0) {
        const [a, b, c] = pending.pop()!;
        if (Math.max(dist(a, b), dist(b, c), dist(c, a)) <= maxEdge) {
            indices.push(a, b, c);
            continue;
        }
        const ab = mid(a, b);
        const bc = mid(b, c);
        const ca = mid(c, a);
        pending.push([ab, bc, ca], [ca, bc, c], [ab, b, bc], [a, ab, ca]);
    }
    return Geometry.fromArrays(geometry.label, new Float32Array(vertices), new Uint32Array(indices));
}

/** The world box of the box `min..max` under the column-major matrix `m`. Rust: `rt::transform_box`. */
export function transformBox(m: ArrayLike<number>, min: Vec3, max: Vec3): [Vec3, Vec3] {
    const c = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, (min[2] + max[2]) * 0.5];
    const h = [(max[0] - min[0]) * 0.5, (max[1] - min[1]) * 0.5, (max[2] - min[2]) * 0.5];
    const lo: Vec3 = [0, 0, 0];
    const hi: Vec3 = [0, 0, 0];
    for (let r = 0; r < 3; r++) {
        const centre = m[r] * c[0] + m[4 + r] * c[1] + m[8 + r] * c[2] + m[12 + r];
        const reach = Math.abs(m[r]) * h[0] + Math.abs(m[4 + r]) * h[1] + Math.abs(m[8 + r]) * h[2];
        lo[r] = centre - reach;
        hi[r] = centre + reach;
    }
    return [lo, hi];
}

/** Whether the boxes `a` and `b` meet, `b` widened by `eps`. */
export function boxesMeet(aMin: Vec3, aMax: Vec3, bMin: Vec3, bMax: Vec3, eps: number): boolean {
    for (let k = 0; k < 3; k++) {
        if (aMin[k] > bMax[k] + eps || aMax[k] < bMin[k] - eps) return false;
    }
    return true;
}
