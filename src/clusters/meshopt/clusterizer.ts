/**
 * Meshlets (small vertex/triangle clusters) built by greedy connectivity-and-cone flow: a
 * TypeScript port of the greedy builder of `optimesh` 1.1 (`clusterizer.rs`), the pure-Rust,
 * bit-exact port of meshoptimizer v1.1 (MIT, Arseny Kapoulkine) the Rust engine builds cluster
 * graphs with. Each meshlet grows by the adjacent triangle that scores best on connectivity and
 * cone tightness, falling back to a kd-tree nearest-neighbour search. Based on Wihlidal,
 * "Optimizing the Graphics Pipeline with Compute" (2016).
 *
 * Float arithmetic is rounded to f32 after every operation (`Math.fround`), which reproduces
 * single-precision IEEE results exactly for +, -, *, / and sqrt: the meshlets match Rust's.
 */

const f = Math.fround;
const F32_MAX = 3.4028234663852886e38;
const NONE = 0xffffffff;

/** A meshlet: a range of the shared vertex and triangle index arrays. */
export interface Meshlet {
    /** Start of its vertices in the vertex array. */
    vertexOffset: number;
    /** Start of its triangle indices in the triangle array (3 per triangle). */
    triangleOffset: number;
    vertexCount: number;
    triangleCount: number;
}

/** Meshlets and their data: per meshlet, its vertices (global indices) and local triangle indices. */
export interface Meshlets {
    meshlets: Meshlet[];
    vertices: Uint32Array;
    triangles: Uint8Array;
}

/** Positions: vertex `i`'s at `data[i * stride]` (in floats), `count` vertices. */
export interface Positions {
    data: Float32Array;
    count: number;
    /** Floats between consecutive vertices. */
    stride: number;
}

/** Cap on the seed-triangle pool carried across meshlets. */
const MESHLET_MAX_SEEDS = 256;
/** Seed triangles collected from each finished meshlet's boundary. */
const MESHLET_ADD_SEEDS = 4;
/** Cap on tree depth to bound recursion on malformed inputs. */
const MESHLET_MAX_TREE_DEPTH = 50;

/** The most meshlets `buildMeshlets` makes of `indexCount` indices. Rust: `build_meshlets_bound`. */
export function buildMeshletsBound(indexCount: number, maxVertices: number, maxTriangles: number): number {
    // worst case leaves 2 vertices unpacked per meshlet
    const limitVertices = Math.ceil(indexCount / (maxVertices - 2));
    const limitTriangles = Math.ceil(indexCount / 3 / maxTriangles);
    return Math.max(limitVertices, limitTriangles);
}

/** Vertex-to-triangle adjacency in compressed-row form; `counts` doubles as the live triangle count. */
class TriangleAdjacency {
    readonly counts: Uint32Array;
    readonly offsets: Uint32Array;
    readonly data: Uint32Array;

    constructor(indices: Uint32Array, vertexCount: number) {
        const counts = new Uint32Array(vertexCount);
        for (let i = 0; i < indices.length; i++) counts[indices[i]]++;
        const offsets = new Uint32Array(vertexCount);
        let offset = 0;
        for (let v = 0; v < vertexCount; v++) {
            offsets[v] = offset;
            offset += counts[v];
        }
        const data = new Uint32Array(indices.length);
        const cursor = offsets.slice();
        for (let t = 0; t < indices.length / 3; t++) {
            for (let k = 0; k < 3; k++) data[cursor[indices[t * 3 + k]]++] = t;
        }
        this.counts = counts;
        this.offsets = offsets;
        this.data = data;
    }
}

/** Scores a candidate triangle by distance and cone spread; lower is better. */
function meshletScore(distance: number, spread: number, coneWeight: number, expectedRadius: number): number {
    const cone = f(1 - f(spread * coneWeight));
    const clamped = cone < 1e-3 ? f(1e-3) : cone;
    return f(f(1 + f(f(distance / expectedRadius) * f(1 - coneWeight))) * clamped);
}

/**
 * Each triangle's centroid and unit normal (6 floats: px py pz nx ny nz), and the summed
 * cross-product magnitude (twice the total area).
 */
function computeTriangleCones(cones: Float32Array, indices: Uint32Array, p: Float32Array, stride: number): number {
    let meshArea = 0;
    for (let t = 0; t < indices.length / 3; t++) {
        const a = indices[t * 3] * stride, b = indices[t * 3 + 1] * stride, c = indices[t * 3 + 2] * stride;
        const e1x = f(p[b] - p[a]), e1y = f(p[b + 1] - p[a + 1]), e1z = f(p[b + 2] - p[a + 2]);
        const e2x = f(p[c] - p[a]), e2y = f(p[c + 1] - p[a + 1]), e2z = f(p[c + 2] - p[a + 2]);
        const nx = f(f(e1y * e2z) - f(e1z * e2y));
        const ny = f(f(e1z * e2x) - f(e1x * e2z));
        const nz = f(f(e1x * e2y) - f(e1y * e2x));
        const area = f(Math.sqrt(f(f(f(nx * nx) + f(ny * ny)) + f(nz * nz))));
        const inv = area === 0 ? 0 : f(1 / area);
        const o = t * 6;
        cones[o] = f(f(f(p[a] + p[b]) + p[c]) / 3);
        cones[o + 1] = f(f(f(p[a + 1] + p[b + 1]) + p[c + 1]) / 3);
        cones[o + 2] = f(f(f(p[a + 2] + p[b + 2]) + p[c + 2]) / 3);
        cones[o + 3] = f(nx * inv);
        cones[o + 4] = f(ny * inv);
        cones[o + 5] = f(nz * inv);
        meshArea = f(meshArea + area);
    }
    return meshArea;
}

/** The distance between centroid `t` of `cones` and point (x, y, z). */
function centroidDistance(cones: Float32Array, t: number, x: number, y: number, z: number): number {
    const dx = f(cones[t * 6] - x), dy = f(cones[t * 6 + 1] - y), dz = f(cones[t * 6 + 2] - z);
    return f(Math.sqrt(f(f(f(dx * dx) + f(dy * dy)) + f(dz * dz))));
}

/** A kd-tree over the triangle centroids: per node a value (a point, or a split's f32 bits) and axis | children << 2. */
class KdTree {
    readonly value: Uint32Array;
    readonly split: Float32Array;
    readonly axisChildren: Uint32Array;

    constructor(nodes: number) {
        this.value = new Uint32Array(nodes);
        this.split = new Float32Array(this.value.buffer);
        this.axisChildren = new Uint32Array(nodes);
    }

    axis(n: number): number { return this.axisChildren[n] & 3; }
    children(n: number): number { return this.axisChildren[n] >>> 2; }
    setAxis(n: number, axis: number): void { this.axisChildren[n] = ((this.axisChildren[n] & ~3) | (axis & 3)) >>> 0; }
    setChildren(n: number, children: number): void { this.axisChildren[n] = ((this.axisChildren[n] & 3) | (children << 2)) >>> 0; }

    private buildLeaf(offset: number, indices: Uint32Array, first: number, count: number): number {
        this.value[offset] = indices[first];
        this.setAxis(offset, 3);
        this.setChildren(offset, count);
        for (let i = 1; i < count; i++) {
            this.value[offset + i] = indices[first + i];
            this.setAxis(offset + i, 3);
            this.setChildren(offset + i, NONE >>> 2);
        }
        return offset + count;
    }

    /** Recursively builds over `indices[first..first + count]`, returning the next free node. */
    build(offset: number, cones: Float32Array, indices: Uint32Array, first: number, count: number, leafSize: number, depth: number): number {
        if (count <= leafSize) return this.buildLeaf(offset, indices, first, count);
        // Welford's running mean and variance over the centroids; the widest-variance axis is split
        const mean = [0, 0, 0], vars = [0, 0, 0];
        let runCount = 1, runScale = 1;
        for (let i = first; i < first + count; i++) {
            const o = indices[i] * 6;
            for (let k = 0; k < 3; k++) {
                const point = cones[o + k];
                const delta = f(point - mean[k]);
                mean[k] = f(mean[k] + f(delta * runScale));
                vars[k] = f(vars[k] + f(delta * f(point - mean[k])));
            }
            runCount = f(runCount + 1);
            runScale = f(1 / runCount);
        }
        const axis = vars[0] >= vars[1] && vars[0] >= vars[2] ? 0 : vars[1] >= vars[2] ? 1 : 2;
        const split = mean[axis];
        // partition around the split
        let middle = 0;
        for (let i = 0; i < count; i++) {
            const v = cones[indices[first + i] * 6 + axis];
            const t = indices[first + middle];
            indices[first + middle] = indices[first + i];
            indices[first + i] = t;
            middle += v < split ? 1 : 0;
        }
        if (middle <= leafSize >> 1 || middle >= count - (leafSize >> 1) || depth >= MESHLET_MAX_TREE_DEPTH) {
            return this.buildLeaf(offset, indices, first, count);
        }
        this.split[offset] = split;
        this.setAxis(offset, axis);
        const next = this.build(offset + 1, cones, indices, first, middle, leafSize, depth + 1);
        this.setChildren(offset, next - offset - 1);
        return this.build(next, cones, indices, first + middle, count - middle, leafSize, depth + 1);
    }

    /**
     * The nearest not-yet-emitted point to (x, y, z) below `root`, into `best` ([index, distance]),
     * deactivating exhausted subtrees as it goes.
     */
    nearest(root: number, cones: Float32Array, emitted: Uint8Array, x: number, y: number, z: number, best: [number, number]): void {
        if (this.children(root) === 0) return;
        if (this.axis(root) === 3) {
            let inactive = true;
            for (let i = 0; i < this.children(root); i++) {
                const index = this.value[root + i];
                if (emitted[index] !== 0) continue;
                inactive = false;
                const distance = centroidDistance(cones, index, x, y, z);
                if (distance < best[1]) {
                    best[0] = index;
                    best[1] = distance;
                }
            }
            if (inactive) this.setChildren(root, 0);
        } else {
            const axis = this.axis(root);
            const delta = f((axis === 0 ? x : axis === 1 ? y : z) - this.split[root]);
            const children = this.children(root);
            const first = delta <= 0 ? 0 : children;
            const second = first ^ children;
            if ((this.children(root + 1 + first) | this.children(root + 1 + second)) === 0) this.setChildren(root, 0);
            this.nearest(root + 1 + first, cones, emitted, x, y, z, best);
            if (Math.abs(delta) <= best[1]) this.nearest(root + 1 + second, cones, emitted, x, y, z, best);
        }
    }
}

/**
 * Builds meshlets of at most `maxVertices` vertices (3..256) and `maxTriangles` triangles
 * (1..512) by greedy connectivity-and-cone flow, `coneWeight` (0..1) weighting cone tightness
 * against spatial distance. Rust: `clusterizer::build_meshlets` (`build_meshlets_flex` with
 * `min_triangles == max_triangles` and no splits).
 */
export function buildMeshlets(indices: Uint32Array, positions: Positions, maxVertices: number, maxTriangles: number, coneWeight: number): Meshlets {
    const coneW = f(coneWeight);
    const bound = buildMeshletsBound(indices.length, maxVertices, maxTriangles);
    const out: Meshlets = { meshlets: [], vertices: new Uint32Array(bound * maxVertices), triangles: new Uint8Array(bound * maxTriangles * 3) };
    if (indices.length === 0) return { meshlets: [], vertices: new Uint32Array(0), triangles: new Uint8Array(0) };

    const vertexCount = positions.count;
    const faceCount = indices.length / 3;
    const adjacency = new TriangleAdjacency(indices, vertexCount);
    const live = adjacency.counts;
    const emitted = new Uint8Array(faceCount);
    const cones = new Float32Array(faceCount * 6);
    const meshArea = computeTriangleCones(cones, indices, positions.data, positions.stride);
    const triangleAreaAvg = f(f(meshArea / f(faceCount)) * 0.5);
    const expectedRadius = f(f(Math.sqrt(f(triangleAreaAvg * f(maxTriangles)))) * 0.5);

    const kdIndices = new Uint32Array(faceCount);
    for (let i = 0; i < faceCount; i++) kdIndices[i] = i;
    const tree = new KdTree(faceCount * 2);
    tree.build(0, cones, kdIndices, 0, faceCount, 8, 0);

    const corner = [F32_MAX, F32_MAX, F32_MAX];
    for (let t = 0; t < faceCount; t++) {
        for (let k = 0; k < 3; k++) corner[k] = Math.min(corner[k], cones[t * 6 + k]);
    }

    // per vertex, its index in the current meshlet (-1: none)
    const used = new Int16Array(vertexCount);
    if (vertexCount <= indices.length) used.fill(-1);
    else for (let i = 0; i < indices.length; i++) used[indices[i]] = -1;

    let initialSeed = NONE, initialScore = F32_MAX;
    for (let t = 0; t < faceCount; t++) {
        const score = centroidDistance(cones, t, corner[0], corner[1], corner[2]);
        if (initialSeed === NONE || score < initialScore) {
            initialSeed = t;
            initialScore = score;
        }
    }

    const seeds = new Uint32Array(MESHLET_MAX_SEEDS);
    let seedCount = 0;
    const meshlet: Meshlet = { vertexOffset: 0, triangleOffset: 0, vertexCount: 0, triangleCount: 0 };
    const meshletVertices = out.vertices;
    // the cone accumulated over the meshlet's triangles
    const acc = new Float32Array(6);
    const cone = new Float32Array(6);
    const best: [number, number] = [NONE, F32_MAX];

    const extraOf = (t: number) => (used[indices[t * 3]] >>> 31) + (used[indices[t * 3 + 1]] >>> 31) + (used[indices[t * 3 + 2]] >>> 31);
    const liveOf = (t: number) => live[indices[t * 3]] + live[indices[t * 3 + 1]] + live[indices[t * 3 + 2]];

    for (;;) {
        // the meshlet's cone: averaged centroid, unit normal
        const centerScale = meshlet.triangleCount === 0 ? 0 : f(1 / f(meshlet.triangleCount));
        for (let k = 0; k < 3; k++) cone[k] = f(acc[k] * centerScale);
        const axisLength = f(f(f(acc[3] * acc[3]) + f(acc[4] * acc[4])) + f(acc[5] * acc[5]));
        const axisScale = axisLength === 0 ? 0 : f(1 / f(Math.sqrt(axisLength)));
        for (let k = 3; k < 6; k++) cone[k] = f(acc[k] * axisScale);

        let bestTriangle = NONE;
        if (out.meshlets.length === 0 && meshlet.triangleCount === 0) {
            bestTriangle = initialSeed;
        } else {
            // the best adjacent triangle: by a topology priority, then distance and cone spread
            let bestPriority = 5, bestScore = F32_MAX;
            for (let i = 0; i < meshlet.vertexCount; i++) {
                const index = meshletVertices[meshlet.vertexOffset + i];
                const begin = adjacency.offsets[index], end = begin + live[index];
                for (let j = begin; j < end; j++) {
                    const triangle = adjacency.data[j];
                    const a = indices[triangle * 3], b = indices[triangle * 3 + 1], c = indices[triangle * 3 + 2];
                    const extra = extraOf(triangle);
                    // lower is better: 0 adds no new vertex, and triangles that use up a vertex's
                    // last live triangles rank ahead
                    const priority = extra === 0 ? 0
                        : live[a] === 1 || live[b] === 1 || live[c] === 1 ? 1
                            : (live[a] === 2 ? 1 : 0) + (live[b] === 2 ? 1 : 0) + (live[c] === 2 ? 1 : 0) >= 2 ? 1 + extra
                                : 2 + extra;
                    if (priority > bestPriority) continue;
                    const o = triangle * 6;
                    const distance = centroidDistance(cones, triangle, cone[0], cone[1], cone[2]);
                    const spread = f(f(f(cones[o + 3] * cone[3]) + f(cones[o + 4] * cone[4])) + f(cones[o + 5] * cone[5]));
                    const score = meshletScore(distance, spread, coneW, expectedRadius);
                    if (priority < bestPriority || score < bestScore) {
                        bestTriangle = triangle;
                        bestPriority = priority;
                        bestScore = score;
                    }
                }
            }
        }

        if (bestTriangle === NONE) {
            best[0] = NONE;
            best[1] = F32_MAX;
            tree.nearest(0, cones, emitted, cone[0], cone[1], cone[2], best);
            bestTriangle = best[0];
        }
        if (bestTriangle === NONE) break;

        if (meshlet.vertexCount + extraOf(bestTriangle) > maxVertices || meshlet.triangleCount >= maxTriangles) {
            // prune emitted seeds, keeping order
            let kept = 0;
            for (let i = 0; i < seedCount; i++) {
                seeds[kept] = seeds[i];
                kept += emitted[seeds[i]] === 0 ? 1 : 0;
            }
            seedCount = kept + MESHLET_ADD_SEEDS <= MESHLET_MAX_SEEDS ? kept : MESHLET_MAX_SEEDS - MESHLET_ADD_SEEDS;
            // up to MESHLET_ADD_SEEDS seeds from the meshlet's boundary: low live counts near the corner
            const bestSeeds = [NONE, NONE, NONE, NONE];
            const bestLive = [NONE, NONE, NONE, NONE];
            const bestScores = [F32_MAX, F32_MAX, F32_MAX, F32_MAX];
            for (let i = 0; i < meshlet.vertexCount; i++) {
                const index = meshletVertices[meshlet.vertexOffset + i];
                const begin = adjacency.offsets[index], end = begin + live[index];
                let neighbor = NONE, neighborLive = NONE;
                for (let j = begin; j < end; j++) {
                    const l = liveOf(adjacency.data[j]);
                    if (l < neighborLive) {
                        neighbor = adjacency.data[j];
                        neighborLive = l;
                    }
                }
                if (neighbor === NONE) continue;
                const neighborScore = centroidDistance(cones, neighbor, corner[0], corner[1], corner[2]);
                for (let j = 0; j < MESHLET_ADD_SEEDS; j++) {
                    // non-strict comparison reduces duplicate seeds
                    if (neighborLive < bestLive[j] || (neighborLive === bestLive[j] && neighborScore <= bestScores[j])) {
                        bestSeeds[j] = neighbor;
                        bestLive[j] = neighborLive;
                        bestScores[j] = neighborScore;
                        break;
                    }
                }
            }
            for (const seed of bestSeeds) if (seed !== NONE) seeds[seedCount++] = seed;
            // the seed with the lowest live count nearest the corner
            let bestSeed = NONE, bestSeedLive = NONE, bestSeedScore = F32_MAX;
            for (let i = 0; i < seedCount; i++) {
                const index = seeds[i];
                const l = liveOf(index);
                const score = centroidDistance(cones, index, corner[0], corner[1], corner[2]);
                if (l < bestSeedLive || (l === bestSeedLive && score < bestSeedScore)) {
                    bestSeed = index;
                    bestSeedLive = l;
                    bestSeedScore = score;
                }
            }
            if (bestSeed !== NONE) bestTriangle = bestSeed;
        }

        const a = indices[bestTriangle * 3], b = indices[bestTriangle * 3 + 1], c = indices[bestTriangle * 3 + 2];
        // flush the meshlet first when the triangle would overflow it
        const usedExtra = extraOf(bestTriangle);
        if (meshlet.vertexCount + usedExtra > maxVertices || meshlet.triangleCount >= maxTriangles) {
            out.meshlets.push({ ...meshlet });
            for (let j = 0; j < meshlet.vertexCount; j++) used[meshletVertices[meshlet.vertexOffset + j]] = -1;
            meshlet.vertexOffset += meshlet.vertexCount;
            meshlet.triangleOffset += meshlet.triangleCount * 3;
            meshlet.vertexCount = 0;
            meshlet.triangleCount = 0;
            acc.fill(0);
        }
        // (each slot re-read after the earlier corners: a repeated vertex aliases to one index)
        const local = (v: number) => {
            if (used[v] < 0) {
                used[v] = meshlet.vertexCount;
                meshletVertices[meshlet.vertexOffset + meshlet.vertexCount++] = v;
            }
            return used[v];
        };
        const av = local(a), bv = local(b), cv = local(c);
        const base = meshlet.triangleOffset + meshlet.triangleCount * 3;
        out.triangles[base] = av;
        out.triangles[base + 1] = bv;
        out.triangles[base + 2] = cv;
        meshlet.triangleCount++;

        // drop the emitted triangle from each vertex's adjacency list (and its live count)
        for (let k = 0; k < 3; k++) {
            const index = indices[bestTriangle * 3 + k];
            const begin = adjacency.offsets[index], size = live[index];
            for (let i = 0; i < size; i++) {
                if (adjacency.data[begin + i] === bestTriangle) {
                    adjacency.data[begin + i] = adjacency.data[begin + size - 1];
                    live[index]--;
                    break;
                }
            }
        }
        for (let k = 0; k < 6; k++) acc[k] = f(acc[k] + cones[bestTriangle * 6 + k]);
        emitted[bestTriangle] = 1;
    }

    if (meshlet.triangleCount !== 0) out.meshlets.push({ ...meshlet });
    return out;
}
