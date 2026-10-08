import { Geometry } from "../buffers/Geometry";
import { buildClusterMesh } from "./build";

/**
 * Cluster LOD (meshlets, as Unreal's Nanite): a mesh split into clusters of about 124
 * triangles, with a graph of coarser versions over them, so a view can draw each part of the
 * mesh at the coarsest version whose error it doesn't see. Port of the Rust engine's
 * `clusters` module (`rust/kansei-core/src/clusters/mod.rs`); see
 * `docs/plans/2026-09-28-cluster-lod-design.md`.
 */

export type Vec3 = [number, number, number];

/** A sphere; clusters' bounds and the spheres their errors are measured from. */
export interface Sphere {
    center: Vec3;
    radius: number;
}

const f = Math.fround;

/** The length of `a`, in f32 as glam's `Vec3::length`. */
function length(a: Vec3): number {
    return f(Math.sqrt(f(f(f(a[0] * a[0]) + f(a[1] * a[1])) + f(a[2] * a[2]))));
}

/** The distance between `a` and `b`, in f32 as glam's `Vec3::distance`. */
function distance(a: Vec3, b: Vec3): number {
    const dx = f(a[0] - b[0]), dy = f(a[1] - b[1]), dz = f(a[2] - b[2]);
    return f(Math.sqrt(f(f(f(dx * dx) + f(dy * dy)) + f(dz * dz))));
}

/** A sphere enclosing all of `spheres` (grown from the first, not the smallest), in f32 as Rust's `Sphere::enclosing`. */
export function enclosingSphere(spheres: Iterable<Sphere>): Sphere {
    let s: Sphere | null = null;
    for (const o of spheres) {
        if (!s) {
            s = { center: [...o.center], radius: o.radius };
            continue;
        }
        const d = distance(o.center, s.center);
        if (f(d + o.radius) <= s.radius) continue;
        if (f(d + s.radius) <= o.radius) {
            s = { center: [...o.center], radius: o.radius };
            continue;
        }
        const radius: number = f(f(f(d + s.radius) + o.radius) * 0.5);
        const t: number = f(f(radius - s.radius) / Math.max(d, f(1e-12)));
        const c: Vec3 = s.center;
        s = { center: [f(c[0] + f(f(o.center[0] - c[0]) * t)), f(c[1] + f(f(o.center[1] - c[1]) * t)), f(c[2] + f(f(o.center[2] - c[2]) * t))], radius };
    }
    if (!s) throw new Error("enclosingSphere: at least one sphere");
    return s;
}

/** Whether `inner` lies inside `outer` (to float precision). */
export function sphereContains(outer: Sphere, inner: Sphere): boolean {
    return distance(outer.center, inner.center) + inner.radius <= outer.radius * (1 + 1e-5) + 1e-6;
}

/** How `ClusterMesh` builds split and simplify (Rust `ClusterOptions`). */
export interface ClusterOptions {
    /** Vertices a cluster may reference (at most 256: its triangles' indices are bytes). 128
     *  lets clusters fill up to `maxTriangles` (a cluster is drawn as `maxTriangles`). */
    maxVertices: number;
    maxTriangles: number;
    /** Clusters per group simplified together (about). */
    groupSize: number;
    /** Triangles a group keeps when simplified. */
    simplifyRatio: number;
    /** A group keeping more than this share of its triangles has stalled: it isn't simplified. */
    stallRatio: number;
    /** Weight of normal-cone tightness against compactness when splitting (backface culling). */
    coneWeight: number;
    /** Weights of the normals and uvs in the simplifier's error. */
    normalWeight: number;
    uvWeight: number;
    /** Treat small, open, flat-ish components as cards (foliage), whose coarser levels are
     *  pruned rather than simplified. Off by default. */
    cards: boolean;
    /** The most triangles a card has. */
    cardMaxTriangles: number;
    /** The share of a card's area its area-weighted normal keeps (1: flat). */
    cardFlatness: number;
    /** Scales the part of pruned levels' error that is a crown thinning out. */
    cardErrorScale: number;
    /** The most a pruned card is scaled up to cover the cards it stands for. */
    cardMaxScale: number;
}

export const DEFAULT_CLUSTER_OPTIONS: Readonly<ClusterOptions> = Object.freeze({
    maxVertices: 128,
    maxTriangles: 124,
    groupSize: 8,
    simplifyRatio: 0.5,
    stallRatio: 0.85,
    coneWeight: 0.25,
    normalWeight: 0.5,
    uvWeight: 0.1,
    cards: false,
    cardMaxTriangles: 64,
    cardFlatness: 0.5,
    cardErrorScale: 1.0,
    cardMaxScale: 4.0,
});

/**
 * One cluster: a run of `clusterVertices` and `clusterTriangles`, its bounds, and the errors the
 * cut rule compares (see `ClusterMesh.select`).
 */
export interface Cluster {
    vertexOffset: number;
    vertexCount: number;
    /** In triangles (3 local indices each). */
    triangleOffset: number;
    triangleCount: number;
    /** Of its triangles, for culling. */
    bounds: Sphere;
    /** Its triangles' normal cone, for backface culling (`clusterBackfacing`), as meshoptimizer
     *  gives it: the apex, the axis, and the sine of the cone's half-angle. A cone too wide to
     *  cull has a zero axis and a cutoff of 1. */
    coneApex: Vec3;
    coneAxis: Vec3;
    coneCutoff: number;
    /** The error of the simplification that made it (0 at level 0), measured from `lodBounds`. */
    error: number;
    lodBounds: Sphere;
    /** The error of the group simplified from it and its siblings (Infinity when none was), from
     *  `parentBounds`. */
    parentError: number;
    parentBounds: Sphere;
    level: number;
    /** One of pruned cards (foliage), not of simplified triangles. */
    card: boolean;
}

/** Whether every one of `c`'s triangles faces away from `eye` (the mesh's own space). */
export function clusterBackfacing(c: Cluster, eye: Vec3): boolean {
    // glam's `normalize_or_zero`, then the dot
    const d: Vec3 = [f(c.coneApex[0] - eye[0]), f(c.coneApex[1] - eye[1]), f(c.coneApex[2] - eye[2])];
    const l = length(d);
    const n: Vec3 = l > 0 && Number.isFinite(f(1 / l)) ? [f(d[0] * f(1 / l)), f(d[1] * f(1 / l)), f(d[2] * f(1 / l))] : [0, 0, 0];
    return f(f(f(n[0] * c.coneAxis[0]) + f(n[1] * c.coneAxis[1])) + f(n[2] * c.coneAxis[2])) >= c.coneCutoff;
}

/** Where a view sees a mesh from, in the mesh's own space, and how much error it tolerates. */
export interface LodView {
    eye: Vec3;
    /** Pixels per radian at the view's centre: the viewport's height / (2 tan(fovY / 2)). */
    pixelsPerRadian: number;
    /** Distances are clamped to this (an eye inside a sphere). */
    near: number;
    /** The error budget, pixels. */
    threshold: number;
    /** An orthographic view: `pixelsPerRadian` is pixels per metre, at any distance. */
    orthographic: boolean;
}

/**
 * `error` (metres) seen from the view as pixels: over the distance to the nearest point of
 * `sphere`. A parent's sphere contains its children's and its error is at least theirs, so its
 * projected error is at least theirs from any eye.
 */
export function projectedError(error: number, sphere: Sphere, view: LodView): number {
    return projectedErrorAt(error, f(distance(sphere.center, view.eye) - sphere.radius), view);
}

/** `error` seen from `distance` away (clamped to the view's `near`), in pixels. */
export function projectedErrorAt(error: number, distance: number, view: LodView): number {
    if (error === 0) return 0;
    if (!Number.isFinite(error)) return Infinity;
    if (view.orthographic) return f(error * f(view.pixelsPerRadian));
    return f(f(error / Math.max(distance, f(view.near))) * f(view.pixelsPerRadian));
}

/** A relative margin `levelMayDraw` gives the budget, so rounding never skips a level the cut rule would draw from. */
export const WINDOW_SLACK = 1e-3;

/**
 * What one build round's clusters (`Cluster.level`) span, to skip a whole level: `levelMayDraw`
 * is false only when none of them can pass the cut rule.
 */
export interface LevelBounds {
    /** Its clusters: `first..first + count` of `ClusterMesh.clusters`. */
    first: number;
    count: number;
    /** The smallest of its clusters' errors. */
    minError: number;
    /** The largest of their parents' errors (Infinity when one has none). */
    maxParentError: number;
    /** The largest `|lodBounds.center| - lodBounds.radius`. */
    nearReach: number;
    /** The largest `|parentBounds.center| + parentBounds.radius` (of the finite parents). */
    farReach: number;
}

/** Whether some cluster of `level` may pass the cut rule for `view`. */
export function levelMayDraw(level: LevelBounds, view: LodView): boolean {
    const d = length(view.eye);
    const threshold = f(view.threshold), slack = f(WINDOW_SLACK);
    const fineEnough = projectedErrorAt(level.minError, f(d + level.nearReach), view) <= f(threshold * f(1 + slack));
    const parentOver = !Number.isFinite(level.maxParentError)
        || projectedErrorAt(level.maxParentError, f(d - level.farReach), view) > f(threshold * f(1 - slack));
    return fineEnough && parentOver;
}

/** The packed mesh's layout (`ClusterMesh.gpuWords`, read by cluster_mesh.wgsl). */
export const HEADER_WORDS = 8;
export const VERTEX_WORDS = 9;
export const CLUSTER_WORDS = 28;
export const LEVEL_WORDS = 8;
/** The error of a missing parent (Infinity): shaders never see infinities. */
export const NO_PARENT = -1;

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
function bits(x: number): number {
    f32[0] = x;
    return u32[0];
}
function float(word: number): number {
    u32[0] = word;
    return f32[0];
}


/** A mesh as a graph of clusters over its own vertices (Rust `ClusterMesh`). */
export class ClusterMesh {
    constructor(
        /** The geometry's vertices (position vec4, normal vec3, uv vec2: 9 floats each), shared by every level. */
        public readonly vertices: Float32Array,
        public readonly clusters: Cluster[],
        /** Per cluster, its vertices as indices into `vertices`. */
        public readonly clusterVertices: Uint32Array,
        /** Per cluster, 3 indices into its own vertices per triangle. */
        public readonly clusterTriangles: Uint8Array,
    ) { }

    /**
     * Split `geometry` (the standard vertices, triangle indices) into clusters and build the
     * graph of coarser versions over them: Rust's `ClusterMesh::build`, word for word
     * (`buildClusterMesh`).
     */
    static build(geometry: Geometry, options: Partial<ClusterOptions> = {}): ClusterMesh {
        return buildClusterMesh(geometry, options);
    }

    /** Vertices of the mesh. */
    get vertexCount(): number {
        return this.vertices.length / VERTEX_WORDS;
    }

    /** Cluster `cluster`'s triangles, as indices into `vertices` (3 per triangle). */
    triangles(cluster: number): Uint32Array {
        const c = this.clusters[cluster];
        const out = new Uint32Array(c.triangleCount * 3);
        for (let k = 0; k < out.length; k++) {
            out[k] = this.clusterVertices[c.vertexOffset + this.clusterTriangles[c.triangleOffset * 3 + k]];
        }
        return out;
    }

    /** The most triangles in one cluster: what every cluster is drawn as. */
    maxTriangles(): number {
        let most = 0;
        for (const c of this.clusters) most = Math.max(most, c.triangleCount);
        return most;
    }

    /** Each build round's clusters and what they span (clusters are stored by round). */
    levels(): LevelBounds[] {
        const levels: LevelBounds[] = [];
        this.clusters.forEach((c, i) => {
            while (levels.length <= c.level) {
                levels.push({ first: i, count: 0, minError: Infinity, maxParentError: 0, nearReach: -Infinity, farReach: -Infinity });
            }
            const level = levels[c.level];
            if (level.first + level.count !== i) throw new Error("ClusterMesh: clusters are stored by level");
            level.count++;
            level.minError = Math.min(level.minError, c.error);
            level.maxParentError = Math.max(level.maxParentError, c.parentError);
            const l = c.lodBounds, p = c.parentBounds;
            level.nearReach = Math.max(level.nearReach, f(length(l.center) - l.radius));
            if (Number.isFinite(c.parentError)) {
                level.farReach = Math.max(level.farReach, f(length(p.center) + p.radius));
            }
        });
        return levels;
    }

    /**
     * The clusters `view` draws: each whose error is within the budget and whose parent's is over
     * it. Every point of the mesh is in exactly one (no holes, no overlaps).
     */
    select(view: LodView): number[] {
        const out: number[] = [];
        const threshold = Math.fround(view.threshold);
        this.clusters.forEach((c, i) => {
            if (projectedError(c.error, c.lodBounds, view) <= threshold && projectedError(c.parentError, c.parentBounds, view) > threshold) out.push(i);
        });
        return out;
    }

    /**
     * The cut `view` selects as an ordinary mesh: a discrete LOD baked from the graph (for LOD
     * bands, impostor bakes, or comparing against the per-cluster path).
     */
    cutGeometry(label: string, view: LodView): Geometry {
        const selected = this.select(view);
        let count = 0;
        for (const c of selected) count += this.clusters[c].triangleCount * 3;
        const indices = new Uint32Array(count);
        let at = 0;
        for (const c of selected) {
            const t = this.triangles(c);
            indices.set(t, at);
            at += t.length;
        }
        return Geometry.fromArrays(label, this.vertices.slice(), indices);
    }

    /**
     * The mesh as the GPU reads it, in one buffer. A header of where each section starts (in
     * words), the cluster and level counts, and `maxTriangles`; then the vertices (as is), the
     * clusters' vertices, their triangles (3 local indices in a word's low 3 bytes), the cluster
     * records and the level records (see cluster_mesh.wgsl).
     */
    gpuWords(): Uint32Array {
        const levels = this.levels();
        const vertices = HEADER_WORDS;
        const clusterVertices = vertices + this.vertices.length;
        const triangles = clusterVertices + this.clusterVertices.length;
        const clusters = triangles + this.clusterTriangles.length / 3;
        const levelRecords = clusters + this.clusters.length * CLUSTER_WORDS;
        const words = new Uint32Array(levelRecords + levels.length * LEVEL_WORDS);
        words.set([vertices, clusterVertices, triangles, clusters, levelRecords, this.clusters.length, levels.length, this.maxTriangles()]);
        words.set(new Uint32Array(this.vertices.buffer, this.vertices.byteOffset, this.vertices.length), vertices);
        words.set(this.clusterVertices, clusterVertices);
        const t = this.clusterTriangles;
        for (let k = 0; k < t.length / 3; k++) words[triangles + k] = t[3 * k] | (t[3 * k + 1] << 8) | (t[3 * k + 2] << 16);
        const parent = (e: number) => Number.isFinite(e) ? e : NO_PARENT;
        let at = clusters;
        const put = (...values: number[]) => {
            for (const v of values) words[at++] = v;
        };
        const sphere = (s: Sphere) => put(bits(s.center[0]), bits(s.center[1]), bits(s.center[2]), bits(s.radius));
        for (const c of this.clusters) {
            put(c.vertexOffset, c.triangleOffset, c.triangleCount, c.level);
            sphere(c.bounds);
            put(bits(c.coneApex[0]), bits(c.coneApex[1]), bits(c.coneApex[2]), bits(c.coneCutoff));
            put(bits(c.coneAxis[0]), bits(c.coneAxis[1]), bits(c.coneAxis[2]), bits(c.error));
            sphere(c.lodBounds);
            sphere(c.parentBounds);
            put(bits(parent(c.parentError)), 0, 0, 0);
        }
        const finite = (x: number) => Number.isFinite(x) ? x : 0;
        for (const l of levels) {
            put(l.first, l.count, bits(l.minError), bits(parent(l.maxParentError)), bits(finite(l.nearReach)), bits(finite(l.farReach)), 0, 0);
        }
        return words;
    }

    /**
     * A mesh from its `gpuWords` (the Rust engine's `ClusterMesh::gpu_words`, say, built offline
     * and fetched): everything but the clusters' `card` flags, which the GPU doesn't read.
     */
    static fromGpuWords(words: Uint32Array): ClusterMesh {
        const [vertices, clusterVertices, triangles, clusters, , clusterCount] = words;
        const vertexFloats = new Float32Array(clusterVertices - vertices);
        vertexFloats.set(new Float32Array(words.buffer, words.byteOffset + vertices * 4, clusterVertices - vertices));
        const list: Cluster[] = [];
        const sphere = (w: number): Sphere => ({ center: [float(words[w]), float(words[w + 1]), float(words[w + 2])], radius: float(words[w + 3]) });
        for (let i = 0; i < clusterCount; i++) {
            const w = clusters + i * CLUSTER_WORDS;
            const parentError = float(words[w + 24]);
            list.push({
                vertexOffset: words[w],
                vertexCount: 0,
                triangleOffset: words[w + 1],
                triangleCount: words[w + 2],
                level: words[w + 3],
                bounds: sphere(w + 4),
                coneApex: [float(words[w + 8]), float(words[w + 9]), float(words[w + 10])],
                coneCutoff: float(words[w + 11]),
                coneAxis: [float(words[w + 12]), float(words[w + 13]), float(words[w + 14])],
                error: float(words[w + 15]),
                lodBounds: sphere(w + 16),
                parentBounds: sphere(w + 20),
                parentError: parentError < 0 ? Infinity : parentError,
                card: false,
            });
        }
        const clusterTriangles = new Uint8Array((clusters - triangles) * 3);
        for (let k = 0; k < clusters - triangles; k++) {
            const packed = words[triangles + k];
            clusterTriangles[3 * k] = packed & 0xff;
            clusterTriangles[3 * k + 1] = (packed >> 8) & 0xff;
            clusterTriangles[3 * k + 2] = (packed >> 16) & 0xff;
        }
        // (a cluster's vertices are those its triangles index)
        for (const c of list) {
            let most = -1;
            for (let k = c.triangleOffset * 3; k < (c.triangleOffset + c.triangleCount) * 3; k++) most = Math.max(most, clusterTriangles[k]);
            c.vertexCount = most + 1;
        }
        return new ClusterMesh(vertexFloats, list, words.slice(clusterVertices, triangles), clusterTriangles);
    }
}
