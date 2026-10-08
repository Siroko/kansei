/**
 * The cluster graph builder: a mesh split into clusters, then repeatedly grouped, simplified and
 * re-split into coarser levels (Nanite's DAG). Port of the Rust engine's `clusters/build.rs`, over
 * the TS port of optimesh (`./meshopt`), every float operation rounded to f32 as Rust computes it,
 * so a mesh builds the graph Rust builds, word for word (`ClusterMesh.gpuWords`).
 */

import type { Geometry } from '../buffers/Geometry';
import { Cluster, ClusterMesh, ClusterOptions, DEFAULT_CLUSTER_OPTIONS, Sphere, Vec3, VERTEX_WORDS, enclosingSphere } from './ClusterMesh';
import { Card, Placed, findCards, levelZero, pruneRound } from './cards';
import { buildMeshlets } from './meshopt/clusterizer';
import { computeClusterBounds } from './meshopt/clusterBounds';
import { partitionClusters } from './meshopt/partition';
import { SIMPLIFY_ERROR_ABSOLUTE, SIMPLIFY_SPARSE, SIMPLIFY_VERTEX_LOCK, simplifyWithAttributes } from './meshopt/simplifier';

const f = Math.fround;
const F32_MAX = 3.4028234663852886e38;
const NONE = 0xffffffff;
/** Positions closer than this share of the mesh's extent are one position (`weld`). */
const WELD = f(1e-6);
/** f32's smallest positive normal. */
const F32_MIN_POSITIVE = 1.1754943508222875e-38;

/** A cluster still without a parent: its index, and its triangles as indices into the vertices. */
type Pending = [number, number[]];

/** Positions, 3 floats a vertex, appended to as card rounds add scaled copies. */
export class GrowingPositions {
    private data: Float32Array;
    private length: number;

    constructor(initial: Float32Array) {
        this.data = Float32Array.from(initial);
        this.length = initial.length;
    }

    /** The positions so far (a view: valid until the next `push`). */
    get array(): Float32Array {
        return this.data.subarray(0, this.length);
    }

    push(p: Vec3): void {
        if (this.length + 3 > this.data.length) {
            const grown = new Float32Array(Math.max(this.data.length * 2, 48));
            grown.set(this.data);
            this.data = grown;
        }
        this.data.set(p, this.length);
        this.length += 3;
    }
}

/** A `ClusterMesh` while it is built: vertices and clusters still growing. */
export class ClusterBuild {
    /** 9 floats a vertex (the mesh's vertices, then cards' scaled copies). */
    readonly vertices: number[];
    readonly clusters: Cluster[] = [];
    readonly clusterVertices: number[] = [];
    readonly clusterTriangles: number[] = [];

    constructor(vertices: ArrayLike<number>) {
        this.vertices = Array.from(vertices);
    }

    get vertexCount(): number {
        return this.vertices.length / VERTEX_WORDS;
    }

    /** Vertex `v`'s position. */
    position(v: number): Vec3 {
        return [this.vertices[v * VERTEX_WORDS], this.vertices[v * VERTEX_WORDS + 1], this.vertices[v * VERTEX_WORDS + 2]];
    }

    /** A copy of vertex `v` at `p`, appended. */
    pushVertex(v: number, p: Vec3): void {
        const o = v * VERTEX_WORDS;
        this.vertices.push(p[0], p[1], p[2], ...this.vertices.slice(o + 3, o + VERTEX_WORDS));
    }

    /**
     * A cluster of `vertices` (into the mesh's) and `triangles` (3 indices into those each), with
     * its bounds and cone from `positions`, carrying `error` and `lodBounds` (its own bounds when
     * null); its index.
     */
    pushCluster(vertices: number[], triangles: number[], positions: Float32Array, error: number, lodBounds: Sphere | null, level: number, card: boolean): number {
        const global = Uint32Array.from(triangles, (t) => vertices[t]);
        const b = computeClusterBounds(global, positions, 3);
        const bounds: Sphere = { center: b.center, radius: b.radius };
        const lod = lodBounds ?? bounds;
        this.clusters.push({
            vertexOffset: this.clusterVertices.length,
            vertexCount: vertices.length,
            triangleOffset: this.clusterTriangles.length / 3,
            triangleCount: triangles.length / 3,
            bounds,
            coneApex: b.coneApex,
            coneAxis: b.coneAxis,
            coneCutoff: b.coneCutoff,
            error,
            lodBounds: lod,
            parentError: Infinity,
            parentBounds: lod,
            level,
            card,
        });
        this.clusterVertices.push(...vertices);
        this.clusterTriangles.push(...triangles);
        return this.clusters.length - 1;
    }

    /**
     * Clusters of `indices` (into the vertices), carrying `error` and `lodBounds` (their own
     * bounds when null); per new cluster, its index and its triangles as indices into the vertices.
     */
    split(indices: ArrayLike<number>, positions: Float32Array, error: number, lodBounds: Sphere | null, level: number, options: ClusterOptions): Pending[] {
        if (indices.length === 0) return [];
        const m = buildMeshlets(Uint32Array.from(indices), { data: positions, count: positions.length / 3, stride: 3 }, options.maxVertices, options.maxTriangles, options.coneWeight);
        return m.meshlets.map((meshlet) => {
            const local = Array.from(m.vertices.subarray(meshlet.vertexOffset, meshlet.vertexOffset + meshlet.vertexCount));
            // (meshlet triangle offsets count indices, 3 per triangle)
            const triangles = Array.from(m.triangles.subarray(meshlet.triangleOffset, meshlet.triangleOffset + meshlet.triangleCount * 3));
            const index = this.pushCluster(local, triangles, positions, error, lodBounds, level, false);
            return [index, triangles.map((t) => local[t])];
        });
    }

    /** The finished mesh. */
    finish(): ClusterMesh {
        return new ClusterMesh(Float32Array.from(this.vertices), this.clusters, Uint32Array.from(this.clusterVertices), Uint8Array.from(this.clusterTriangles));
    }
}

/** The diagonal of the bounding box of `positions`. */
function extent(positions: Float32Array): number {
    const lo = [F32_MAX, F32_MAX, F32_MAX], hi = [-F32_MAX, -F32_MAX, -F32_MAX];
    for (let i = 0; i < positions.length; i += 3) {
        for (let k = 0; k < 3; k++) {
            lo[k] = Math.min(lo[k], positions[i + k]);
            hi[k] = Math.max(hi[k], positions[i + k]);
        }
    }
    if (lo[0] > hi[0]) return 0;
    const d = [0, 1, 2].map((k) => f(lo[k] - hi[k]));
    return f(Math.sqrt(f(f(f(d[0] * d[0]) + f(d[1] * d[1])) + f(d[2] * d[2]))));
}

/**
 * Each position within `tolerance` of an earlier one takes that one's value, bit for bit (and
 * -0 is 0): a seam's copies that differ by rounding or by the sign of zero become one position to
 * the simplifier and the locks, which match positions exactly.
 */
function weld(positions: Float32Array, tolerance: number): void {
    tolerance = Math.max(tolerance, F32_MIN_POSITIVE);
    const count = positions.length / 3;
    const cell = new Float64Array(positions.length);
    let small = true;
    for (let k = 0; k < positions.length; k++) {
        cell[k] = Math.floor(f((positions[k] + 0) / tolerance));
        small &&= Math.abs(cell[k]) < 2 ** 24;
    }
    // the cells' lists: by (x, y) as one exact number, then by z; by string past 2^24 cells out
    const byXy = new Map<number, Map<number, number[]>>();
    const byString = new Map<string, number[]>();
    const get = (x: number, y: number, z: number) => small ? byXy.get((x + 2 ** 25) * 2 ** 26 + (y + 2 ** 25))?.get(z) : byString.get(`${x},${y},${z}`);
    const add = (x: number, y: number, z: number, i: number) => {
        if (!small) {
            const key = `${x},${y},${z}`;
            const list = byString.get(key);
            if (list) list.push(i);
            else byString.set(key, [i]);
            return;
        }
        const xy = (x + 2 ** 25) * 2 ** 26 + (y + 2 ** 25);
        let column = byXy.get(xy);
        if (!column) byXy.set(xy, column = new Map());
        const list = column.get(z);
        if (list) list.push(i);
        else column.set(z, [i]);
    };
    for (let i = 0; i < count; i++) {
        const px = positions[i * 3] + 0, py = positions[i * 3 + 1] + 0, pz = positions[i * 3 + 2] + 0;
        const cx = cell[i * 3], cy = cell[i * 3 + 1], cz = cell[i * 3 + 2];
        let near = -1;
        search: for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
                for (let dz = -1; dz <= 1; dz++) {
                    const list = get(cx + dx, cy + dy, cz + dz);
                    if (!list) continue;
                    for (const j of list) {
                        const ex = f(positions[j * 3] - px), ey = f(positions[j * 3 + 1] - py), ez = f(positions[j * 3 + 2] - pz);
                        if (f(Math.sqrt(f(f(f(ex * ex) + f(ey * ey)) + f(ez * ez)))) <= tolerance) {
                            near = j;
                            break search;
                        }
                    }
                }
            }
        }
        if (near >= 0) {
            positions.copyWithin(i * 3, near * 3, near * 3 + 3);
        } else {
            positions[i * 3] = px;
            positions[i * 3 + 1] = py;
            positions[i * 3 + 2] = pz;
            add(cx, cy, cz, i);
        }
    }
}

/** One id per distinct position (seams split vertices, not positions). */
export function positionIds(positions: Float32Array): Uint32Array {
    const bits = new Uint32Array(positions.buffer, positions.byteOffset, positions.length);
    const ids = new Map<string, number>();
    const out = new Uint32Array(positions.length / 3);
    for (let v = 0; v < out.length; v++) {
        const key = `${bits[v * 3]},${bits[v * 3 + 1]},${bits[v * 3 + 2]}`;
        let id = ids.get(key);
        if (id === undefined) ids.set(key, id = ids.size);
        out[v] = id;
    }
    return out;
}

/**
 * Groups of about `size` neighbouring clusters (indices into `pending`); `canonical` maps each
 * vertex to one vertex per position, so a seam doesn't part neighbours.
 */
function partition(pending: Pending[], positions: Float32Array, canonical: Uint32Array, size: number): number[][] {
    let total = 0;
    for (const [, t] of pending) total += t.length;
    const indices = new Uint32Array(total);
    let at = 0;
    for (const [, t] of pending) for (const v of t) indices[at++] = canonical[v];
    const counts = Uint32Array.from(pending, ([, t]) => t.length);
    const { partitions, count } = partitionClusters(indices, counts, positions, positions.length / 3, 3, size);
    const out: number[][] = Array.from({ length: count }, () => []);
    partitions.forEach((g, i) => out[g].push(i));
    return out;
}

/**
 * `SIMPLIFY_VERTEX_LOCK` on every vertex whose position more than one group uses: groups meet at
 * the same vertices whatever level each is drawn at.
 */
function sharedVertexLocks(groups: number[][], pending: Pending[], ids: Uint32Array): Uint8Array {
    const SHARED = NONE - 1;
    const owner = new Uint32Array(ids.length).fill(NONE);
    groups.forEach((group, g) => {
        for (const i of group) {
            for (const v of pending[i][1]) {
                const p = ids[v];
                owner[p] = owner[p] === NONE ? g : owner[p] === g ? g : SHARED;
            }
        }
    });
    return Uint8Array.from(ids, (p) => owner[p] === SHARED ? SIMPLIFY_VERTEX_LOCK : 0);
}

/**
 * Split `geometry` (the standard 9-float vertices, triangle indices) into clusters and build the
 * graph of coarser versions over them (Rust `ClusterMesh::build`).
 */
export function buildClusterMesh(geometry: Geometry, overrides: Partial<ClusterOptions> = {}): ClusterMesh {
    const options: ClusterOptions = { ...DEFAULT_CLUSTER_OPTIONS, ...overrides };
    const source = geometry.vertices;
    const sourceIndices = geometry.indices;
    if (!source || !sourceIndices) throw new Error('buildClusterMesh: the geometry has no CPU vertices or indices');
    const original = source.length / VERTEX_WORDS;
    const welded = new Float32Array(original * 3);
    const attributes = new Float32Array(original * 5);
    for (let v = 0; v < original; v++) {
        welded.set(source.subarray(v * VERTEX_WORDS, v * VERTEX_WORDS + 3), v * 3);
        attributes.set(source.subarray(v * VERTEX_WORDS + 4, v * VERTEX_WORDS + 9), v * 5);
    }
    weld(welded, f(WELD * extent(welded)));
    const weights = [options.normalWeight, options.normalWeight, options.normalWeight, options.uvWeight, options.uvWeight];
    const ids = positionIds(welded);
    // one vertex per position, so clusters on either side of a seam are neighbours
    const first = new Uint32Array(ids.length).fill(NONE);
    const canonical = Uint32Array.from(ids, (p, v) => {
        if (first[p] === NONE) first[p] = v;
        return first[p];
    });
    const vertices = Float32Array.from(source);
    for (let v = 0; v < original; v++) vertices.set(welded.subarray(v * 3, v * 3 + 3), v * VERTEX_WORDS);
    const mesh = new ClusterBuild(vertices);

    // cards (with `options.cards`) are pruned level by level, the rest simplified; the card rounds
    // append their scaled copies to the vertices (and `positions`), which the simplifier never sees
    const positions = new GrowingPositions(welded);
    let cardList: Card[] = [];
    let solid: ArrayLike<number> = sourceIndices;
    if (options.cards) ({ cards: cardList, rest: solid } = findCards(sourceIndices, welded, ids, options));
    let pending = mesh.split(solid, welded, 0, null, 0, options);
    let pendingCards: [number, Placed[]][] = levelZero(mesh, cardList, welded, options);
    const simplifyRatio = f(options.simplifyRatio), stallRatio = f(options.stallRatio);
    let level = 0;
    while (pending.length > 1 || pendingCards.length > 1) {
        level++;
        let progress = false;
        if (pendingCards.length > 1) {
            const round = pruneRound(mesh, cardList, pendingCards, positions, level, options);
            pendingCards = round.next;
            progress ||= round.pruned;
        }
        if (pending.length <= 1) {
            if (!progress) break;
            continue;
        }
        const current = positions.array;
        const groups = partition(pending, current, canonical, options.groupSize);
        const lock = sharedVertexLocks(groups, pending, ids);
        const next: Pending[] = [];
        for (const group of groups) {
            const merged = Uint32Array.from(group.flatMap((i) => pending[i][1]));
            const target = Math.floor(Math.trunc(f(f(merged.length) * simplifyRatio)) / 3) * 3;
            const s = simplifyWithAttributes(merged, { positions: welded, count: original, stride: 3 }, { data: attributes, stride: 5, weights }, lock,
                { targetIndexCount: target, targetError: F32_MAX, options: SIMPLIFY_SPARSE | SIMPLIFY_ERROR_ABSOLUTE });
            if (s.indices.length > f(f(merged.length) * stallRatio)) {
                // too little came off: next round, grouped with other neighbours
                next.push(...group.map((i) => pending[i]));
                continue;
            }
            progress = true;
            // never less than a child's error, from a sphere round all of theirs
            let error = s.error;
            for (const i of group) error = Math.max(error, mesh.clusters[pending[i][0]].error);
            const bounds = enclosingSphere(group.map((i) => mesh.clusters[pending[i][0]].lodBounds));
            for (const i of group) {
                const child = mesh.clusters[pending[i][0]];
                child.parentError = error;
                child.parentBounds = bounds;
            }
            next.push(...mesh.split(s.indices, current, error, bounds, level, options));
        }
        pending = next;
        if (!progress) break;
    }
    return mesh.finish();
}
