/**
 * Cards: foliage's small, open, flat-ish pieces (sprays, leaves, ribbons), which edge collapse
 * can't reduce (every vertex is on a border). Their coarser levels are pruned instead: see
 * `buildClusterMesh` and the card rounds below. Port of the Rust engine's `clusters/cards.rs`,
 * every float operation rounded to f32 as Rust computes it.
 */

import type { ClusterBuild, GrowingPositions } from './build';
import { ClusterOptions, Sphere, Vec3, enclosingSphere } from './ClusterMesh';

const f = Math.fround;
const F32_MAX = 3.4028234663852886e38;
const NONE = 0xffffffff;

/** A card: its vertices (into the mesh's), its triangles (into those), its area-weighted centroid, its area, and its radius from the centroid. */
export interface Card {
    vertices: number[];
    triangles: [number, number, number][];
    centroid: Vec3;
    area: number;
    radius: number;
}

// f32 vector helpers (glam's Vec3)
const sub = (a: Vec3, b: Vec3): Vec3 => [f(a[0] - b[0]), f(a[1] - b[1]), f(a[2] - b[2])];
const add = (a: Vec3, b: Vec3): Vec3 => [f(a[0] + b[0]), f(a[1] + b[1]), f(a[2] + b[2])];
const scale = (a: Vec3, s: number): Vec3 => [f(a[0] * s), f(a[1] * s), f(a[2] * s)];
const divide = (a: Vec3, s: number): Vec3 => [f(a[0] / s), f(a[1] / s), f(a[2] / s)];
const dot = (a: Vec3, b: Vec3) => f(f(f(a[0] * b[0]) + f(a[1] * b[1])) + f(a[2] * b[2]));
const length = (a: Vec3) => f(Math.sqrt(dot(a, a)));
const cross = (a: Vec3, b: Vec3): Vec3 => [f(f(a[1] * b[2]) - f(b[1] * a[2])), f(f(a[2] * b[0]) - f(b[2] * a[0])), f(f(a[0] * b[1]) - f(b[0] * a[1]))];
const vmin = (a: Vec3, b: Vec3): Vec3 => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.min(a[2], b[2])];
const vmax = (a: Vec3, b: Vec3): Vec3 => [Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.max(a[2], b[2])];

/** Position `v` of `positions` (3 floats a vertex). */
function at(positions: ArrayLike<number>, v: number): Vec3 {
    return [positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]];
}

/**
 * The triangles of `indices` split into cards and the rest (as indices), by connected component
 * over `positionIds` (welded positions): a card has at most `cardMaxTriangles` triangles and
 * `maxVertices` vertices, is open (an edge used once), and keeps `cardFlatness` of its area in
 * its area-weighted normal.
 */
export function findCards(indices: ArrayLike<number>, positions: ArrayLike<number>, positionIds: Uint32Array, options: ClusterOptions): { cards: Card[]; rest: number[] } {
    // components: union-find over positions
    const parent = Uint32Array.from({ length: positionIds.length }, (_, i) => i);
    const root = (x: number) => {
        while (parent[x] !== x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    for (let t = 0; t < indices.length; t += 3) {
        const a = root(positionIds[indices[t]]);
        for (let k = 1; k < 3; k++) {
            const b = root(positionIds[indices[t + k]]);
            parent[b] = a;
        }
    }
    const components = new Map<number, number[]>();
    for (let i = 0; i < indices.length / 3; i++) {
        const r = root(positionIds[indices[i * 3]]);
        let list = components.get(r);
        if (!list) components.set(r, list = []);
        list.push(i);
    }
    const roots = [...components.keys()].sort((a, b) => a - b);
    const cards: Card[] = [];
    const rest: number[] = [];
    for (const r of roots) {
        const triangles = components.get(r)!;
        const tri = (i: number): [number, number, number] => [indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]];
        const vertices = [...new Set(triangles.flatMap(tri))].sort((a, b) => a - b);
        let area = 0;
        let normal: Vec3 = [0, 0, 0], centroid: Vec3 = [0, 0, 0];
        const edges = new Map<string, number>();
        for (const i of triangles) {
            const [a, b, c] = tri(i);
            const pa = at(positions, a), pb = at(positions, b), pc = at(positions, c);
            const n = cross(sub(pb, pa), sub(pc, pa));
            const ta = f(0.5 * length(n));
            area = f(area + ta);
            normal = add(normal, scale(n, 0.5));
            centroid = add(centroid, scale(divide(add(add(pa, pb), pc), 3), ta));
            for (const [x0, y0] of [[a, b], [b, c], [c, a]]) {
                const x = positionIds[x0], y = positionIds[y0];
                const key = `${Math.min(x, y)},${Math.max(x, y)}`;
                edges.set(key, (edges.get(key) ?? 0) + 1);
            }
        }
        const open = [...edges.values()].some((n) => n === 1);
        const isCard = triangles.length <= options.cardMaxTriangles && vertices.length <= options.maxVertices && open && area > 0
            && length(normal) >= f(f(options.cardFlatness) * area);
        if (!isCard) {
            for (const i of triangles) rest.push(...tri(i));
            continue;
        }
        const center = divide(centroid, area);
        const local = new Map(vertices.map((v, j) => [v, j]));
        let radius = 0;
        for (const v of vertices) radius = Math.max(radius, length(sub(at(positions, v), center)));
        cards.push({ triangles: triangles.map((i) => tri(i).map((v) => local.get(v)!) as [number, number, number]), vertices, centroid: center, area, radius });
    }
    return { cards, rest };
}

/** An axis-aligned box round cards. */
interface CardBox {
    lo: Vec3;
    hi: Vec3;
}

function boxOf(points: Vec3[]): CardBox {
    let lo: Vec3 = [F32_MAX, F32_MAX, F32_MAX], hi: Vec3 = [-F32_MAX, -F32_MAX, -F32_MAX];
    for (const p of points) {
        lo = vmin(lo, p);
        hi = vmax(hi, p);
    }
    return { lo, hi };
}

/** How far `p` lies outside `box` (0 inside). */
function boxDistance(box: CardBox, p: Vec3): number {
    return length(vmax(vmax(sub(box.lo, p), sub(p, box.hi)), [0, 0, 0]));
}

/**
 * A card at some level: which card, where its vertices are (`NONE`: the card's own; else the
 * first of its scaled copy's, in the card's vertex order), how many level-0 cards it stands for,
 * the area it covers, and where (their area-weighted centre, where it is drawn), and the box round
 * those level-0 cards (the outline grown cards should keep within).
 */
export interface Placed {
    card: number;
    firstVertex: number;
    represents: number;
    area: number;
    center: Vec3;
    region: CardBox;
}

function placedVertex(p: Placed, card: Card, j: number): number {
    return p.firstVertex === NONE ? card.vertices[j] : p.firstVertex + j;
}

/** A 30-bit Morton code of `p` in `(lo, hi)`. */
export function morton(p: Vec3, lo: Vec3, hi: Vec3): number {
    const q = [0, 1, 2].map((k) => Math.trunc(Math.min(Math.max(f(f(f(p[k] - lo[k]) / Math.max(f(hi[k] - lo[k]), f(1e-9))) * 1023), 0), 1023)));
    const spread = (x: number) => {
        x = (x | (x << 16)) & 0x030000ff;
        x = (x | (x << 8)) & 0x0300f00f;
        x = (x | (x << 4)) & 0x030c30c3;
        return (x | (x << 2)) & 0x09249249;
    };
    return (spread(q[0]) | (spread(q[1]) << 1) | (spread(q[2]) << 2)) >>> 0;
}

function hash(a: number, b: number): number {
    const x = (Math.imul(a, 0x9e3779b1) ^ Math.imul(b, 0x85ebca77)) >>> 0;
    return Math.imul(x ^ (x >>> 15), 0x2c1b3c6d) >>> 0;
}

/** The bounding box of `positions` (3 floats a vertex). */
function boundsOf(positions: ArrayLike<number>): [Vec3, Vec3] {
    let lo: Vec3 = [F32_MAX, F32_MAX, F32_MAX], hi: Vec3 = [-F32_MAX, -F32_MAX, -F32_MAX];
    for (let v = 0; v < positions.length / 3; v++) {
        lo = vmin(lo, at(positions, v));
        hi = vmax(hi, at(positions, v));
    }
    return [lo, hi];
}

/** Sorts by a key, stably (Rust's `sort_by_key`). */
function sortByKey<T>(items: T[], key: (item: T) => number): T[] {
    const keyed = items.map((item, i) => ({ item, key: key(item), i }));
    keyed.sort((a, b) => a.key - b.key || a.i - b.i);
    return keyed.map((k) => k.item);
}

/**
 * Whole cards packed into clusters, in the order given (neighbours: Morton order), each up to the
 * options' vertex and triangle limits, with `error`, `lodBounds` (their own when null) and
 * `level`; per new cluster, its index and its cards.
 */
export function pack(mesh: ClusterBuild, cards: Card[], placed: Placed[], positions: Float32Array, error: number, lodBounds: Sphere | null, level: number, options: ClusterOptions): [number, Placed[]][] {
    const out: [number, Placed[]][] = [];
    let start = 0;
    while (start < placed.length) {
        let v = 0, t = 0, end = start;
        while (end < placed.length) {
            const card = cards[placed[end].card];
            if (end > start && (v + card.vertices.length > options.maxVertices || t + card.triangles.length > options.maxTriangles)) break;
            v += card.vertices.length;
            t += card.triangles.length;
            end++;
        }
        const vertices: number[] = [], triangles: number[] = [];
        for (const p of placed.slice(start, end)) {
            const card = cards[p.card];
            const base = vertices.length;
            for (let j = 0; j < card.vertices.length; j++) vertices.push(placedVertex(p, card, j));
            for (const tri of card.triangles) triangles.push(base + tri[0], base + tri[1], base + tri[2]);
        }
        const index = mesh.pushCluster(vertices, triangles, positions, error, lodBounds, level, true);
        out.push([index, placed.slice(start, end)]);
        start = end;
    }
    return out;
}

/** Neighbouring pairs of `placed` (in Morton order): each card with the nearest still free among the next few; an odd one alone. */
function pairs(placed: Placed[]): [Placed, Placed | null][] {
    const WINDOW = 8;
    const taken = new Array<boolean>(placed.length).fill(false);
    const out: [Placed, Placed | null][] = [];
    const d2 = (a: Vec3, b: Vec3) => {
        const d = sub(a, b);
        return dot(d, d);
    };
    for (let i = 0; i < placed.length; i++) {
        if (taken[i]) continue;
        taken[i] = true;
        // (the first of equals, as Rust's `min_by`)
        let nearest = -1;
        for (let j = i + 1; j < Math.min(placed.length, i + 1 + WINDOW); j++) {
            if (taken[j]) continue;
            if (nearest < 0 || d2(placed[i].center, placed[j].center) < d2(placed[i].center, placed[nearest].center)) nearest = j;
        }
        if (nearest >= 0) {
            taken[nearest] = true;
            out.push([placed[i], placed[nearest]]);
        } else {
            out.push([placed[i], null]);
        }
    }
    return out;
}

/**
 * One round over the card clusters still without a parent (`pending`): grouped in Morton order,
 * each group's cards pruned to one of each neighbouring pair (the kept one scaled to cover its
 * pair's area: new vertices, appended to the mesh and to `positions`), the children given their
 * parent's error and sphere, and the kept cards packed into this level's clusters. Returns the
 * next pending set and whether any group was pruned.
 */
export function pruneRound(mesh: ClusterBuild, cards: Card[], pending: [number, Placed[]][], positions: GrowingPositions, level: number, options: ClusterOptions): { next: [number, Placed[]][]; pruned: boolean } {
    const [lo, hi] = boundsOf(positions.array);
    const sorted = sortByKey(pending, ([c]) => morton(mesh.clusters[c].bounds.center, lo, hi));
    const next: [number, Placed[]][] = [];
    let progress = false;
    const size = Math.max(options.groupSize, 2);
    for (let g = 0; g < sorted.length; g += size) {
        const group = sorted.slice(g, g + size);
        let placed = group.flatMap(([, p]) => p);
        if (placed.length < 2) {
            next.push(...group);
            continue;
        }
        placed = sortByKey(placed, (p) => morton(p.center, lo, hi));
        let area = 0, n0 = 0;
        for (const p of placed) {
            area = f(area + p.area);
            n0 += p.represents;
        }
        const kept: Placed[] = [];
        // how far the grown cards reach past the level-0 cards they stand for
        let protrusion = 0;
        for (let [a, b] of pairs(placed)) {
            if (!b) {
                kept.push(a);
                continue;
            }
            // the one kept: either, by a hash (deterministic, uncorrelated with place)
            if (hash(a.card, level) > hash(b.card, level)) [a, b] = [b, a];
            const card = cards[a.card];
            const represents = a.represents + b.represents, covered = f(a.area + b.area);
            const grow = f(Math.sqrt(f(covered / card.area)));
            if (grow > f(options.cardMaxScale)) {
                // as far as this card goes: both stay
                kept.push(a, b);
                continue;
            }
            // drawn at the centre of the area it stands for, scaled to cover it
            const center = divide(add(scale(a.center, a.area), scale(b.center, b.area)), covered);
            const region = { lo: vmin(a.region.lo, b.region.lo), hi: vmax(a.region.hi, b.region.hi) };
            const firstVertex = mesh.vertexCount;
            for (const v of card.vertices) {
                const p = add(center, scale(sub(mesh.position(v), card.centroid), grow));
                protrusion = Math.max(protrusion, boxDistance(region, p));
                mesh.pushVertex(v, p);
                positions.push(p);
            }
            kept.push({ card: a.card, firstVertex, represents, area: covered, center, region });
        }
        if (kept.length === placed.length) {
            // nothing came off: next round, grouped with other neighbours
            next.push(...group);
            continue;
        }
        progress = true;
        const k = f(kept.length);
        // the crown thinned (scaled: how soon a crown may thin is a choice), or its outline moved
        // (in full: a spire rounded off is seen as it is)
        const thinned = f(f(options.cardErrorScale) * f(f(Math.sqrt(f(area / k))) - f(Math.sqrt(f(area / f(n0))))));
        let error = Math.max(Math.max(thinned, protrusion), 0);
        for (const [c] of group) error = Math.max(error, mesh.clusters[c].error);
        const bounds = enclosingSphere(group.map(([c]) => mesh.clusters[c].lodBounds));
        for (const [c] of group) {
            mesh.clusters[c].parentError = error;
            mesh.clusters[c].parentBounds = bounds;
        }
        next.push(...pack(mesh, cards, kept, positions.array, error, bounds, level, options));
    }
    return { next, pruned: progress };
}

/** The cards at level 0: each on its own, in Morton order of their centroids, packed into clusters. */
export function levelZero(mesh: ClusterBuild, cards: Card[], positions: Float32Array, options: ClusterOptions): [number, Placed[]][] {
    if (cards.length === 0) return [];
    const [lo, hi] = boundsOf(positions);
    const order = sortByKey(cards.map((_, c) => c), (c) => morton(cards[c].centroid, lo, hi));
    const placed: Placed[] = order.map((c) => ({
        card: c,
        firstVertex: NONE,
        represents: 1,
        area: cards[c].area,
        center: cards[c].centroid,
        region: boxOf(cards[c].vertices.map((v) => at(positions, v))),
    }));
    return pack(mesh, cards, placed, positions, 0, null, 0, options);
}
