// The test meshes and checks of rust/kansei-core/src/clusters/{tests,card_tests}.rs, ported.
import { Geometry } from "../../src/buffers/Geometry";
import { ClusterMesh, LodView, VERTEX_WORDS, Vec3 } from "../../src/clusters/ClusterMesh";

type Vertex = { position: Vec3; normal: Vec3; uv: [number, number] };

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
export const dist = (a: Vec3, b: Vec3) => len(sub(a, b));

/** A geometry of `vertices` and `indices`. */
export function geometry(label: string, vertices: Vertex[], indices: number[]): Geometry {
    const data = new Float32Array(vertices.length * VERTEX_WORDS);
    vertices.forEach((v, i) => data.set([...v.position, 1, ...v.normal, ...v.uv], i * VERTEX_WORDS));
    return Geometry.fromArrays(label, data, Uint32Array.from(indices));
}

/** The vertices of `g`. */
export function verticesOf(g: Geometry): Vertex[] {
    const d = g.vertices!;
    return Array.from({ length: d.length / VERTEX_WORDS }, (_, i) => {
        const o = i * VERTEX_WORDS;
        return { position: [d[o], d[o + 1], d[o + 2]], normal: [d[o + 4], d[o + 5], d[o + 6]], uv: [d[o + 7], d[o + 8]] };
    });
}

/**
 * A noisy icosphere (a rock) with `subdivisions`. With `seam`, the triangles on the x < 0 side get
 * their own vertices with other uvs, so a uv seam runs round the rock.
 */
export function rock(subdivisions: number, seam: boolean): Geometry {
    // in f32, as the Rust test computes it
    const f = Math.fround;
    const norm = (a: number[]): Vec3 => {
        const l = f(Math.sqrt(f(f(f(a[0] * a[0]) + f(a[1] * a[1])) + f(a[2] * a[2]))));
        return [f(a[0] / l), f(a[1] / l), f(a[2] / l)];
    };
    const t = f(f(1 + f(Math.sqrt(5))) / 2);
    const p: Vec3[] = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]].map(norm);
    let faces: number[][] = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
    for (let s = 0; s < subdivisions; s++) {
        const mid = new Map<string, number>();
        const m = (x: number, y: number) => {
            const key = `${Math.min(x, y)},${Math.max(x, y)}`;
            let v = mid.get(key);
            if (v === undefined) {
                p.push(norm([0, 1, 2].map((k) => f(f(p[x][k] + p[y][k]) * 0.5))));
                mid.set(key, v = p.length - 1);
            }
            return v;
        };
        faces = faces.flatMap(([a, b, c]) => {
            const [ab, bc, ca] = [m(a, b), m(b, c), m(c, a)];
            return [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]];
        });
    }
    const bump = (v: Vec3): Vec3 => {
        const h = f(f(1 + f(f(f(0.08) * f(Math.sin(f(7 * v[0])))) * f(Math.cos(f(5 * v[1]))))) + f(f(0.05) * f(Math.sin(f(11 * v[2])))));
        return [f(v[0] * h), f(v[1] * h), f(v[2] * h)];
    };
    const vertex = (v: Vec3, u: number): Vertex => ({ position: bump(v), normal: v, uv: [f(f(v[0] * 0.5) + u), f(f(v[1] * 0.5) + 0.5)] });
    const vertices = p.map((v) => vertex(v, 0.5));
    if (seam) {
        // the x < 0 side's own copies, with uvs a whole unit over
        const copies = p.map((v) => {
            vertices.push(vertex(v, 1.5));
            return vertices.length - 1;
        });
        for (const tri of faces) {
            if (f(f(p[tri[0]][0] + p[tri[1]][0]) + p[tri[2]][0]) < 0) for (let k = 0; k < 3; k++) tri[k] = copies[tri[k]];
        }
    }
    return geometry("rock", vertices, faces.flat());
}

/** A seeded sequence in [0, 1). */
export function random(seed: { value: number }): number {
    seed.value = (Math.imul(seed.value, 1664525) + 1013904223) >>> 0;
    return (seed.value >>> 8) / (1 << 24);
}

/** `count` eyes round the origin, in random directions, from `near` to `far` (log-uniform). */
export function eyes(count: number, near: number, far: number, seed: { value: number }): Vec3[] {
    return Array.from({ length: count }, () => {
        const z = random(seed) * 2 - 1, a = random(seed) * 2 * Math.PI;
        const r = Math.sqrt(1 - z * z);
        const s = near * (far / near) ** random(seed);
        return [r * Math.cos(a) * s, z * s, r * Math.sin(a) * s] as Vec3;
    });
}

export function view(eye: Vec3, threshold: number): LodView {
    return { eye, pixelsPerRadian: 1080 / 0.8, near: 0.1, threshold, orthographic: false };
}

/** Position `v` of `mesh`. */
export function position(mesh: ClusterMesh, v: number): Vec3 {
    const o = v * VERTEX_WORDS;
    return [mesh.vertices[o], mesh.vertices[o + 1], mesh.vertices[o + 2]];
}

/** Cluster `c`'s triangles, 3 vertices each. */
export function triangles(mesh: ClusterMesh, c: number): [number, number, number][] {
    const t = mesh.triangles(c);
    return Array.from({ length: t.length / 3 }, (_, k) => [t[k * 3], t[k * 3 + 1], t[k * 3 + 2]]);
}

/** One id per point of `mesh`'s vertices: positions within `tolerance` share one. */
export function positionKeys(mesh: ClusterMesh, tolerance: number): number[] {
    const cells = new Map<string, number[]>();
    const points: Vec3[] = [];
    return Array.from({ length: mesh.vertexCount }, (_, v) => {
        const p = position(mesh, v);
        const c = p.map((x) => Math.floor(x / tolerance));
        for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
                for (let dz = -1; dz <= 1; dz++) {
                    const near = cells.get(`${c[0] + dx},${c[1] + dy},${c[2] + dz}`)?.find((k) => dist(points[k], p) <= tolerance);
                    if (near !== undefined) return near;
                }
            }
        }
        points.push(p);
        const key = `${c[0]},${c[1]},${c[2]}`;
        cells.set(key, [...(cells.get(key) ?? []), points.length - 1]);
        return points.length - 1;
    });
}

/**
 * Edges of `clusters`' triangles that betray a hole or an overlap: used an odd number of times,
 * or shared by other than exactly two clusters once each; zero-area triangles are skipped.
 */
export function badEdges(mesh: ClusterMesh, keys: number[], clusters: number[]): number {
    const uses = new Map<string, number[]>();
    for (const c of clusters) {
        for (const t of triangles(mesh, c)) {
            const k = t.map((v) => keys[v]);
            if (k[0] === k[1] || k[1] === k[2] || k[2] === k[0]) continue;
            for (const [a, b] of [[k[0], k[1]], [k[1], k[2]], [k[2], k[0]]]) {
                const key = `${Math.min(a, b)},${Math.max(a, b)}`;
                uses.set(key, [...(uses.get(key) ?? []), c]);
            }
        }
    }
    return [...uses.values()].filter((u) => u.length % 2 === 1 || (u.some((c) => c !== u[0]) && u.length !== 2)).length;
}

/** The total area of `clusters`' triangles. */
export function area(mesh: ClusterMesh, clusters: number[]): number {
    let sum = 0;
    for (const c of clusters) {
        for (const [a, b, d] of triangles(mesh, c)) sum += len(cross(sub(position(mesh, b), position(mesh, a)), sub(position(mesh, d), position(mesh, a)))) * 0.5;
    }
    return sum;
}

/** The triangles of `clusters`. */
export function triangleCount(mesh: ClusterMesh, clusters: number[]): number {
    return clusters.reduce((n, c) => n + mesh.clusters[c].triangleCount, 0);
}

/** The level-0 clusters. */
export function levelZero(mesh: ClusterMesh): number[] {
    return mesh.clusters.flatMap((c, i) => c.level === 0 ? [i] : []);
}

/** Throws unless every cut of `mesh` seen from `eyes` at `budgets` is non-empty, covers about the mesh's area, and has no hole or overlap. */
export function assertCutsClosed(name: string, mesh: ClusterMesh, eyes: Vec3[], budgets: number[]): void {
    const whole = area(mesh, levelZero(mesh));
    const keys = positionKeys(mesh, 1e-5);
    for (const eye of eyes) {
        for (const threshold of budgets) {
            const cut = mesh.select(view(eye, threshold));
            if (cut.length === 0) throw new Error(`${name}: eye ${eye}, budget ${threshold}: an empty cut`);
            const covered = area(mesh, cut) / whole;
            if (!(covered >= 0.8 && covered < 1.2)) throw new Error(`${name}: eye ${eye}, budget ${threshold}: the cut covers ${covered} of the mesh`);
            const bad = badEdges(mesh, keys, cut);
            if (bad !== 0) throw new Error(`${name}: eye ${eye}, budget ${threshold}: ${bad} edges open or overlapping in a cut of ${cut.length} clusters`);
        }
    }
}

// glam's f32 Vec3 operations, for meshes made as the Rust tests make them
const f = Math.fround;
const fadd = (a: Vec3, b: Vec3): Vec3 => [f(a[0] + b[0]), f(a[1] + b[1]), f(a[2] + b[2])];
const fscale = (a: Vec3, s: number): Vec3 => [f(a[0] * s), f(a[1] * s), f(a[2] * s)];
const fcross = (a: Vec3, b: Vec3): Vec3 => [f(f(a[1] * b[2]) - f(b[1] * a[2])), f(f(a[2] * b[0]) - f(b[2] * a[0])), f(f(a[0] * b[1]) - f(b[0] * a[1]))];
const fnormalize = (a: Vec3): Vec3 => fscale(a, f(1 / f(Math.sqrt(f(f(f(a[0] * a[0]) + f(a[1] * a[1])) + f(a[2] * a[2]))))));

/** A quad (two triangles, its own four vertices) centred at `c`, in the plane of `u` and `v` (half extents), normal u x v. */
export function quad(vertices: Vertex[], indices: number[], c: Vec3, u: Vec3, v: Vec3): void {
    const base = vertices.length;
    const n = fnormalize(fcross(u, v));
    [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(([su, sv], i) => {
        vertices.push({ position: fadd(fadd(c, fscale(u, su)), fscale(v, sv)), normal: n, uv: [i === 1 || i === 2 ? 1 : 0, i >= 2 ? 1 : 0] });
    });
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
}

/**
 * A tree's crown of `count` cards: quads over a cone 10 m tall and 3 m wide at its base, each
 * tilted at random (deterministic), in f32 as the Rust test makes it.
 */
export function crown(count: number): Geometry {
    const vertices: Vertex[] = [], indices: number[] = [];
    const seed = { value: 11 };
    const r = () => f(random(seed));
    for (let k = 0; k < count; k++) {
        const h = r(), a = f(r() * f(2 * Math.PI));
        const radius = f(f(3 * f(1 - h)) * f(f(0.6) + f(f(0.4) * r())));
        const cos = f(Math.cos(a)), sin = f(Math.sin(a));
        const c: Vec3 = [f(radius * cos), f(10 * h), f(radius * sin)];
        const out = fnormalize([cos, f(0.3), sin]);
        const side = fnormalize(fcross([0, 1, 0], out));
        const tilt = f(r() - 0.5);
        const v = fscale(fnormalize(fadd(fscale(out, tilt), fscale([0, 1, 0], f(1 - Math.abs(tilt))))), 0.25);
        quad(vertices, indices, c, fscale(side, f(0.35)), v);
    }
    return geometry("crown", vertices, indices);
}
