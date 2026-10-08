/**
 * Agglomerative partitioning of clusters into balanced groups, preferring clusters that share
 * vertices or lie close together: a TypeScript port of `optimesh` 1.1's
 * `partition::partition_clusters` (meshoptimizer v1.1, MIT). Clusters start as singleton groups
 * in a min-heap; the smallest group repeatedly merges with the neighbour it shares the most
 * boundary with (and, with positions, that is nearest) until groups reach the target size. Based
 * on Kurita, "An efficient agglomerative clustering algorithm using a heap" (1991). Every float
 * operation is rounded to f32, as Rust computes it.
 */

const f = Math.fround;

/** Recursion depth at which the spatial merge falls back to median bisection. */
const MERGE_DEPTH_CUTOFF = 40;

/** Groups during merging (struct of arrays), each linked into a chain via `next` from its root (whose `size` is nonzero). */
class Groups {
    readonly group: Int32Array;
    readonly next: Int32Array;
    readonly size: Uint32Array;
    readonly vertices: Uint32Array;
    readonly center: Float32Array;
    readonly radius: Float32Array;

    constructor(count: number) {
        this.group = new Int32Array(count);
        this.next = new Int32Array(count);
        this.size = new Uint32Array(count);
        this.vertices = new Uint32Array(count);
        this.center = new Float32Array(count * 3);
        this.radius = new Float32Array(count);
    }

    /** The distance between groups `a`'s and `b`'s centres, and the offset from `a` to `b`. */
    private offset(a: number, b: number): [number, number, number, number] {
        const c = this.center;
        const dx = f(c[b * 3] - c[a * 3]), dy = f(c[b * 3 + 1] - c[a * 3 + 1]), dz = f(c[b * 3 + 2] - c[a * 3 + 2]);
        return [dx, dy, dz, f(Math.sqrt(f(f(f(dx * dx) + f(dy * dy)) + f(dz * dz))))];
    }

    /** Grows `target`'s sphere to also enclose `source`'s. */
    mergeBounds(target: number, source: number): void {
        const r1 = this.radius[target], r2 = this.radius[source];
        const [dx, dy, dz, d] = this.offset(target, source);
        if (f(d + r1) < r2) {
            this.center.copyWithin(target * 3, source * 3, source * 3 + 3);
            this.radius[target] = r2;
            return;
        }
        if (f(d + r2) > r1) {
            const k = d > 0 ? f(f(f(d + r2) - r1) / f(2 * d)) : 0;
            const c = this.center;
            c[target * 3] = f(c[target * 3] + f(dx * k));
            c[target * 3 + 1] = f(c[target * 3 + 1] + f(dy * k));
            c[target * 3 + 2] = f(c[target * 3 + 2] + f(dz * k));
            this.radius[target] = f(f(f(d + r2) + r1) / 2);
        }
    }

    /** How little merging `source` into `target` grows `target`'s sphere; 1 when `source` is already inside. */
    boundsScore(target: number, source: number): number {
        const r1 = this.radius[target], r2 = this.radius[source];
        const d = this.offset(target, source)[3];
        const merged = f(d + r1) < r2 ? r2 : f(d + r2) < r1 ? r1 : f(f(f(d + r2) + r1) / 2);
        return merged > 0 ? f(r1 / merged) : 0;
    }
}

/** A min-heap of (id, order), ordered by order. */
class Heap {
    readonly id: Uint32Array;
    readonly order: Int32Array;
    size = 0;

    constructor(capacity: number) {
        this.id = new Uint32Array(capacity);
        this.order = new Int32Array(capacity);
    }

    private swap(a: number, b: number): void {
        const id = this.id[a], order = this.order[a];
        this.id[a] = this.id[b];
        this.order[a] = this.order[b];
        this.id[b] = id;
        this.order[b] = order;
    }

    push(id: number, order: number): void {
        let i = this.size++;
        this.id[i] = id;
        this.order[i] = order;
        while (i > 0 && this.order[i] < this.order[(i - 1) >> 1]) {
            const parent = (i - 1) >> 1;
            this.swap(i, parent);
            i = parent;
        }
    }

    /** Removes the smallest item, returning its id. */
    pop(): number {
        const top = this.id[0];
        const size = --this.size;
        this.id[0] = this.id[size];
        this.order[0] = this.order[size];
        let i = 0;
        while (i * 2 + 1 < size) {
            let child = i * 2 + 1;
            if (child + 1 < size && this.order[child + 1] < this.order[child]) child++;
            if (this.order[child] >= this.order[i]) break;
            this.swap(i, child);
            i = child;
        }
        return top;
    }
}

/** Cluster-to-cluster adjacency in compressed-row form, with a shared-vertex count per edge. */
interface ClusterAdjacency {
    offsets: Uint32Array;
    clusters: Uint32Array;
    shared: Uint32Array;
}

/** Adjacency with shared-vertex edge weights, from each cluster's (deduplicated) vertices. */
function buildClusterAdjacency(indices: Uint32Array, clusterOffsets: Uint32Array, clusterCount: number, vertexCount: number): ClusterAdjacency {
    const refOffsets = new Uint32Array(vertexCount + 1);
    for (let i = 0; i < clusterCount; i++) {
        for (let j = clusterOffsets[i]; j < clusterOffsets[i + 1]; j++) refOffsets[indices[j]]++;
    }
    // worst case: every shared vertex implies a distinct neighbour, but no more than every other cluster
    let totalAdjacency = 0;
    for (let i = 0; i < clusterCount; i++) {
        let count = 0;
        for (let j = clusterOffsets[i]; j < clusterOffsets[i + 1]; j++) count += refOffsets[indices[j]] - 1;
        totalAdjacency += Math.min(count, clusterCount - 1);
    }
    const clusters = new Uint32Array(totalAdjacency);
    const shared = new Uint32Array(totalAdjacency);
    const offsets = new Uint32Array(clusterCount + 1);

    let totalRefs = 0;
    for (let v = 0; v < vertexCount; v++) {
        const count = refOffsets[v];
        refOffsets[v] = totalRefs;
        totalRefs += count;
    }
    const refData = new Uint32Array(totalRefs);
    for (let i = 0; i < clusterCount; i++) {
        for (let j = clusterOffsets[i]; j < clusterOffsets[i + 1]; j++) refData[refOffsets[indices[j]]++] = i;
    }
    // the fill left each vertex's end: shift to get the starts
    for (let v = vertexCount - 1; v >= 0; v--) refOffsets[v + 1] = refOffsets[v];
    refOffsets[0] = 0;

    for (let i = 0; i < clusterCount; i++) {
        const base = offsets[i];
        let count = 0;
        for (let j = clusterOffsets[i]; j < clusterOffsets[i + 1]; j++) {
            const v = indices[j];
            for (let k = refOffsets[v]; k < refOffsets[v + 1]; k++) {
                const c = refData[k];
                if (c === i) continue;
                let found = false;
                for (let l = 0; l < count; l++) {
                    if (clusters[base + l] === c) {
                        found = true;
                        shared[base + l]++;
                        break;
                    }
                }
                if (!found) {
                    clusters[base + count] = c;
                    shared[base + count] = 1;
                    count++;
                }
            }
        }
        offsets[i + 1] = offsets[i] + count;
    }
    return { offsets, clusters, shared };
}

/**
 * The best neighbouring group to merge `id` into, and the vertices they share ([-1, 0] for none).
 * `sharedAcc` is scratch, all zeros on entry and left zeroed.
 */
function pickGroupToMerge(groups: Groups, id: number, adjacency: ClusterAdjacency, maxPartitionSize: number, useBounds: boolean, sharedAcc: Uint32Array): [number, number] {
    const groupRsqrt = f(1 / f(Math.sqrt(f(groups.vertices[id] | 0))));
    let bestGroup = -1, bestScore = 0, bestShared = 0;
    // shared vertex counts per adjacent group, across the whole chain
    for (let ci = id; ci >= 0; ci = groups.next[ci]) {
        for (let adj = adjacency.offsets[ci]; adj < adjacency.offsets[ci + 1]; adj++) {
            const other = groups.group[adjacency.clusters[adj]];
            if (other >= 0) sharedAcc[other] += adjacency.shared[adj];
        }
    }
    // score each adjacent group, clearing sharedAcc as it goes
    for (let ci = id; ci >= 0; ci = groups.next[ci]) {
        for (let adj = adjacency.offsets[ci]; adj < adjacency.offsets[ci + 1]; adj++) {
            const other = groups.group[adjacency.clusters[adj]];
            if (other < 0 || sharedAcc[other] === 0) continue;
            const shared = sharedAcc[other];
            sharedAcc[other] = 0;
            if (groups.size[id] + groups.size[other] > maxPartitionSize) continue;
            const otherRsqrt = f(1 / f(Math.sqrt(f(groups.vertices[other] | 0))));
            // the shared count normalized by each group's expected boundary
            let score = f(f(shared | 0) * f(groupRsqrt + otherRsqrt));
            if (useBounds) score = f(score * f(1 + f(f(0.4) * groups.boundsScore(id, other))));
            if (score > bestScore) {
                bestGroup = other;
                bestScore = score;
                bestShared = shared;
            }
        }
    }
    return [bestGroup, bestShared];
}

/** Appends `source`'s chain to `target`'s. */
function link(groups: Groups, target: number, source: number): void {
    let tail = target;
    while (groups.next[tail] >= 0) tail = groups.next[tail];
    groups.next[tail] = source;
}

/** Merges the remaining small groups of `order[first..first + count]` by proximity alone. */
function mergeLeaf(groups: Groups, order: Uint32Array, first: number, count: number, targetPartitionSize: number, maxPartitionSize: number): void {
    for (let i = first; i < first + count; i++) {
        const id = order[i];
        if (groups.size[id] === 0 || groups.size[id] >= targetPartitionSize) continue;
        let bestScore = -1, bestGroup = -1;
        for (let j = first; j < first + count; j++) {
            const other = order[j];
            if (id === other || groups.size[other] === 0) continue;
            if (groups.size[id] + groups.size[other] > maxPartitionSize) continue;
            const score = groups.boundsScore(id, other);
            if (score > bestScore) {
                bestScore = score;
                bestGroup = other;
            }
        }
        // merge id into bestGroup, so more groups can accrete onto the same root
        if (bestGroup !== -1) {
            link(groups, bestGroup, id);
            groups.size[bestGroup] += groups.size[id];
            groups.size[id] = 0;
            groups.mergeBounds(bestGroup, id);
            groups.radius[id] = 0;
        }
    }
}

/** Splits `order[first..first + count]` along its widest axis, merging leaves by proximity once small enough. */
function mergeSpatial(groups: Groups, order: Uint32Array, first: number, count: number, targetPartitionSize: number, maxPartitionSize: number, leafSize: number, depth: number): void {
    let total = 0;
    for (let i = first; i < first + count; i++) total += groups.size[order[i]];
    if (total <= maxPartitionSize || count <= leafSize) {
        mergeLeaf(groups, order, first, count, targetPartitionSize, maxPartitionSize);
        return;
    }
    // Welford's running mean and variance over the centres
    const mean = [0, 0, 0], vars = [0, 0, 0];
    let runCount = 1, runScale = 1;
    for (let i = first; i < first + count; i++) {
        const o = order[i] * 3;
        for (let k = 0; k < 3; k++) {
            const point = groups.center[o + k];
            const delta = f(point - mean[k]);
            mean[k] = f(mean[k] + f(delta * runScale));
            vars[k] = f(vars[k] + f(delta * f(point - mean[k])));
        }
        runCount = f(runCount + 1);
        runScale = f(1 / runCount);
    }
    const axis = vars[0] >= vars[1] && vars[0] >= vars[2] ? 0 : vars[1] >= vars[2] ? 1 : 2;
    const split = mean[axis];
    let middle = 0;
    for (let i = 0; i < count; i++) {
        const v = groups.center[order[first + i] * 3 + axis];
        const t = order[first + middle];
        order[first + middle] = order[first + i];
        order[first + i] = t;
        middle += v < split ? 1 : 0;
    }
    // keep the split balanced and the recursion bounded
    if (middle <= leafSize >> 1 || count - middle <= leafSize >> 1 || depth >= MERGE_DEPTH_CUTOFF) middle = count >> 1;
    mergeSpatial(groups, order, first, middle, targetPartitionSize, maxPartitionSize, leafSize, depth + 1);
    mergeSpatial(groups, order, first + middle, count - middle, targetPartitionSize, maxPartitionSize, leafSize, depth + 1);
}

/**
 * Partitions clusters into groups of about `targetPartitionSize` clusters (at most a third
 * more): returns each cluster's partition id and the number of partitions. `clusterIndices`
 * holds every cluster's vertex indices back to back, `clusterIndexCounts[i]` of them in cluster
 * `i`; `positions` (`stride` floats apart), when given, enables spatial grouping. Rust:
 * `partition::partition_clusters`.
 */
export function partitionClusters(
    clusterIndices: Uint32Array,
    clusterIndexCounts: Uint32Array,
    positions: Float32Array | null,
    vertexCount: number,
    stride: number,
    targetPartitionSize: number,
): { partitions: Uint32Array; count: number } {
    const clusterCount = clusterIndexCounts.length;
    const maxPartitionSize = targetPartitionSize + Math.floor(targetPartitionSize / 3);

    // each cluster's indices with duplicates removed
    const used = new Uint8Array(vertexCount);
    let total = 0;
    for (const c of clusterIndexCounts) total += c;
    const indices = new Uint32Array(total);
    const offsets = new Uint32Array(clusterCount + 1);
    let start = 0, write = 0;
    for (let i = 0; i < clusterCount; i++) {
        offsets[i] = write;
        for (let j = 0; j < clusterIndexCounts[i]; j++) {
            const v = clusterIndices[start + j];
            indices[write] = v;
            write += 1 - used[v];
            used[v] = 1;
        }
        for (let j = offsets[i]; j < write; j++) used[indices[j]] = 0;
        start += clusterIndexCounts[i];
    }
    offsets[clusterCount] = write;

    const adjacency = buildClusterAdjacency(indices, offsets, clusterCount, vertexCount);
    const groups = new Groups(clusterCount);
    const heap = new Heap(clusterCount);
    const sharedAcc = new Uint32Array(clusterCount);
    const useBounds = positions !== null;

    for (let i = 0; i < clusterCount; i++) {
        groups.group[i] = i;
        groups.next[i] = -1;
        groups.size[i] = 1;
        groups.vertices[i] = offsets[i + 1] - offsets[i];
        if (positions) {
            // the average of its vertices, and the farthest of them
            let cx = 0, cy = 0, cz = 0;
            for (let j = offsets[i]; j < offsets[i + 1]; j++) {
                const o = indices[j] * stride;
                cx = f(cx + positions[o]);
                cy = f(cy + positions[o + 1]);
                cz = f(cz + positions[o + 2]);
            }
            const n = f(offsets[i + 1] - offsets[i]);
            if (n > 0) {
                cx = f(cx / n);
                cy = f(cy / n);
                cz = f(cz / n);
            }
            let radius2 = 0;
            for (let j = offsets[i]; j < offsets[i + 1]; j++) {
                const o = indices[j] * stride;
                const dx = f(positions[o] - cx), dy = f(positions[o + 1] - cy), dz = f(positions[o + 2] - cz);
                const d2 = f(f(f(dx * dx) + f(dy * dy)) + f(dz * dz));
                if (radius2 < d2) radius2 = d2;
            }
            groups.center[i * 3] = cx;
            groups.center[i * 3 + 1] = cy;
            groups.center[i * 3 + 2] = cz;
            groups.radius[i] = f(Math.sqrt(radius2));
        }
        heap.push(i, groups.vertices[i] | 0);
    }

    while (heap.size > 0) {
        const top = heap.pop();
        if (groups.size[top] === 0) continue;
        // freeze the chain's clusters, so other groups cannot merge into it
        for (let node = top; node >= 0; node = groups.next[node]) groups.group[node] = -1;
        if (groups.size[top] >= targetPartitionSize) continue;
        const [best, bestShared] = pickGroupToMerge(groups, top, adjacency, maxPartitionSize, useBounds, sharedAcc);
        if (best === -1) continue;
        link(groups, top, best);
        groups.size[top] += groups.size[best];
        groups.vertices[top] += groups.vertices[best];
        groups.vertices[top] = groups.vertices[top] > bestShared ? groups.vertices[top] - bestShared : 1;
        groups.size[best] = 0;
        groups.vertices[best] = 0;
        if (useBounds) {
            groups.mergeBounds(top, best);
            groups.radius[best] = 0;
        }
        // re-associate the merged chain with its root and reinsert it
        for (let node = top; node >= 0; node = groups.next[node]) groups.group[node] = top;
        heap.push(top, groups.vertices[top] | 0);
    }

    if (useBounds) {
        const mergeOrder: number[] = [];
        for (let i = 0; i < clusterCount; i++) if (groups.size[i] !== 0) mergeOrder.push(i);
        mergeSpatial(groups, Uint32Array.from(mergeOrder), 0, mergeOrder.length, targetPartitionSize, maxPartitionSize, 8, 0);
    }

    const partitions = new Uint32Array(clusterCount);
    let next = 0;
    for (let i = 0; i < clusterCount; i++) {
        if (groups.size[i] === 0) continue;
        for (let node = i; node >= 0; node = groups.next[node]) partitions[node] = next;
        next++;
    }
    return { partitions, count: next };
}
