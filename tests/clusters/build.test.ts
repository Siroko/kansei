// `buildClusterMesh` against Rust's `ClusterMesh::build` on the same meshes: a rock, the rock with
// a uv seam, and cards with a rock, a tube and a grid (`tests/fixtures/graph-*.json`, written by
// `rust/tools/meshopt-fixture graph`): the packed graphs (`gpuWords`) must match word for word.
// GRAPH_FIXTURE=<path>[,<path>...] checks other fixtures of the same form (larger meshes).
import { readFileSync } from "node:fs";
import process from "node:process";
import { assert, assertEq, test } from "../harness";
import { Geometry } from "../../src/buffers/Geometry";
import { buildClusterMesh } from "../../src/clusters/build";
import { ClusterMesh, DEFAULT_CLUSTER_OPTIONS, LodView, Vec3, clusterBackfacing, levelMayDraw, sphereContains } from "../../src/clusters/ClusterMesh";
import { SphereGeometry } from "../../src/geometries/SphereGeometry";
import { area, assertCutsClosed, badEdges, dist, eyes, geometry, levelZero, position, positionKeys, rock, triangleCount, triangles, verticesOf, view } from "./meshes";

interface GraphFixture {
    vertices: number[];
    indices: number[];
    cards: boolean;
    words: number[];
}

const fixtures = ["graph-rock3", "graph-rock3-seam", "graph-mixed"].map((n) => `tests/fixtures/${n}.json`)
    .concat(process.env.GRAPH_FIXTURE ? process.env.GRAPH_FIXTURE.split(",") : []);

for (const path of fixtures) {
    test(`graphs are Rust's (${path.split("/").pop()})`, () => {
        const fx: GraphFixture = JSON.parse(readFileSync(path, "utf8"));
        const geometry = Geometry.fromArrays("fixture", new Float32Array(Uint32Array.from(fx.vertices).buffer), Uint32Array.from(fx.indices));
        const words = buildClusterMesh(geometry, { cards: fx.cards }).gpuWords();
        assertEq(words.length, fx.words.length, "words");
        const first = words.findIndex((w, i) => w !== fx.words[i]);
        assert(first < 0, `word ${first} differs: ${words[first]} != ${fx.words[first]} (header ${Array.from(words.subarray(0, 8))})`);
    });
}

// rust/kansei-core/src/clusters/tests.rs, ported

const build = (g: Geometry, options = {}) => ClusterMesh.build(g, options);

test("level zero is the mesh in clusters", () => {
    const r = rock(4, false);
    const mesh = build(r);
    const level0 = levelZero(mesh);
    assertEq(triangleCount(mesh, level0), r.indices!.length / 3);
    for (const i of level0) {
        const c = mesh.clusters[i];
        assert(c.triangleCount <= DEFAULT_CLUSTER_OPTIONS.maxTriangles && c.vertexCount <= DEFAULT_CLUSTER_OPTIONS.maxVertices);
        assertEq(c.error, 0);
        for (const t of triangles(mesh, i)) for (const v of t) assert(dist(c.bounds.center, position(mesh, v)) <= c.bounds.radius * 1.0001, `cluster ${i}'s sphere misses vertex ${v}`);
    }
});

test("tiny and empty meshes", () => {
    assertEq(build(geometry("empty", [], [])).clusters.length, 0);
    const v = (x: number, y: number) => ({ position: [x, y, 0] as Vec3, normal: [0, 0, 1] as Vec3, uv: [x, y] as [number, number] });
    const one = build(geometry("one", [v(0, 0), v(1, 0), v(0, 1)], [0, 1, 2]));
    assertEq(one.clusters.length, 1);
    assertEq(triangles(one, 0), [[0, 1, 2]]);
});

test("levels shrink to a root, and errors and bounds nest", () => {
    const mesh = build(rock(5, false));
    const levels = Math.max(...mesh.clusters.map((c) => c.level));
    const perLevel = Array.from({ length: levels + 1 }, (_, l) => mesh.clusters.filter((c) => c.level === l).reduce((n, c) => n + c.triangleCount, 0));
    assert(levels >= 4, `${perLevel}`);
    assert(perLevel.every((n, l) => l === 0 || n < perLevel[l - 1]), `${perLevel}`);
    const root = mesh.clusters.filter((c) => c.parentError === Infinity).reduce((n, c) => n + c.triangleCount, 0);
    assert(root <= 4 * 124, `${root} root triangles`);
    for (const c of mesh.clusters) {
        assert(c.parentError >= c.error);
        assert(sphereContains(c.parentBounds, c.lodBounds));
    }
});

test("open and disjoint meshes terminate", () => {
    // a field of disjoint quads: nothing to collapse, and the build still ends
    const vertices = [], indices: number[] = [];
    for (let i = 0; i < 2000; i++) {
        const x = i % 50, z = Math.floor(i / 50), base = vertices.length;
        for (const [dx, dy] of [[0, 0], [0.8, 0], [0.8, 0.8], [0, 0.8]]) vertices.push({ position: [x + dx, dy, z] as Vec3, normal: [0, 0, 1] as Vec3, uv: [dx, dy] as [number, number] });
        indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
    assert(build(geometry("cards", vertices, indices)).clusters.length > 0);
    // an open grid reduces: its outline isn't locked
    const n = 120, gv = [], gi: number[] = [];
    for (let z = 0; z <= n; z++) {
        for (let x = 0; x <= n; x++) {
            const fx = x / n, fz = z / n;
            gv.push({ position: [fx, 0.05 * Math.sin(fx * 9) * Math.cos(fz * 7), fz] as Vec3, normal: [0, 1, 0] as Vec3, uv: [fx, fz] as [number, number] });
        }
    }
    for (let z = 0; z < n; z++) {
        for (let x = 0; x < n; x++) {
            const i = z * (n + 1) + x;
            gi.push(i, i + n + 1, i + 1, i + 1, i + n + 1, i + n + 2);
        }
    }
    assert(build(geometry("grid", gv, gi)).clusters.some((c) => c.level >= 3));
});

test("flat-shaded meshes keep one level", () => {
    // every triangle its own vertices with its face normal: every edge a seam
    const smooth = rock(3, false), sv = verticesOf(smooth), idx = smooth.indices!;
    const vertices = [], indices: number[] = [];
    for (let t = 0; t < idx.length; t += 3) {
        const p = [0, 1, 2].map((k) => sv[idx[t + k]].position);
        const e1 = p[1].map((x, k) => x - p[0][k]), e2 = p[2].map((x, k) => x - p[0][k]);
        const c = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        const l = Math.hypot(c[0], c[1], c[2]);
        for (const q of p) {
            indices.push(vertices.length);
            vertices.push({ position: q, normal: c.map((x) => x / l) as Vec3, uv: [0, 0] as [number, number] });
        }
    }
    const mesh = build(geometry("flat", vertices, indices));
    assert(mesh.clusters.every((c) => c.level === 0 && c.parentError === Infinity));
});

test("every cut is closed", () => {
    for (const seam of [false, true]) {
        const mesh = build(rock(5, seam));
        const cuts: number[] = [];
        for (const eye of [[0, 0, 3], [2, 1, 1.5], [0, 0, 40], [-300, 20, 0]] as Vec3[]) {
            for (const threshold of [0, 0.5, 1, 4, 1e9]) {
                assertCutsClosed(`seam ${seam}`, mesh, [eye], [threshold]);
                cuts.push(triangleCount(mesh, mesh.select(view(eye, threshold))));
            }
        }
        // at a pixel's budget, the cut from 300 m is far coarser than the one from 3 m
        assert(cuts[17] > 0 && cuts[17] * 20 < cuts[2], `seam ${seam}: ${cuts}`);
        // and from anywhere, 1.3 m to 1.1 km
        assertCutsClosed(`seam ${seam}`, mesh, eyes(30, 1.3, 1100, { value: 7 }), [0.5, 2]);
    }
});

test("an eye inside the mesh gets the finest cut", () => {
    const mesh = build(rock(4, false));
    const cut = mesh.select(view([0, 0, 0], 1));
    assert(cut.every((i) => mesh.clusters[i].level === 0));
    assertEq(badEdges(mesh, positionKeys(mesh, 1e-5), cut), 0);
});

test("degenerate triangles never break a cut", () => {
    const r = rock(4, false);
    const vertices = verticesOf(r), indices = Array.from(r.indices!);
    // zero-area triangles: repeated vertices, and three vertices on one line
    indices.push(0, 0, 1, 5, 5, 5);
    const a = vertices.length;
    for (let k = 0; k < 3; k++) vertices.push({ ...vertices[0], position: [vertices[0].position[0] + 0.001 * k, vertices[0].position[1], vertices[0].position[2]] });
    indices.push(a, a + 1, a + 2);
    const mesh = build(geometry("degenerate", vertices, indices));
    for (const threshold of [0, 1, 1e9]) assert(mesh.select(view([0, 0, 30], threshold)).length > 0);
});

/** The rock cut into `islands` uv islands by longitude: each island's triangles get their own vertices (uvs a unit over per island). */
function rockIslands(subdivisions: number, islands: number): Geometry {
    const r = rock(subdivisions, false), rv = verticesOf(r), ri = r.indices!;
    const copies = new Map<string, number>();
    const vertices: ReturnType<typeof verticesOf> = [], indices: number[] = [];
    for (let t = 0; t < ri.length; t += 3) {
        const c = [0, 1, 2].map((k) => rv[ri[t]].position[k] + rv[ri[t + 1]].position[k] + rv[ri[t + 2]].position[k]);
        const longitude = Math.atan2(c[2], c[0]) + Math.PI;
        const island = Math.min(Math.floor(longitude / (2 * Math.PI) * islands), islands - 1);
        for (let k = 0; k < 3; k++) {
            const i = ri[t + k], key = `${i},${island}`;
            let v = copies.get(key);
            if (v === undefined) {
                vertices.push({ ...rv[i], uv: [rv[i].uv[0] + island, rv[i].uv[1]] });
                copies.set(key, v = vertices.length - 1);
            }
            indices.push(v);
        }
    }
    return geometry("rock islands", vertices, indices);
}

test("seams that differ by rounding or the sign of zero stay closed", () => {
    const eyeList: Vec3[] = [[0, 0, 3], [0, 40, 0], ...eyes(10, 1.5, 500, { value: 11 })];
    // the engine's sphere (Rust's test takes its UV sphere, whose last column repeats the first,
    // off by rounding, and whose poles' copies differ in the sign of zero; the TS sphere is built
    // otherwise)
    const sphere = build(new SphereGeometry(1, 64));
    assert(sphere.clusters.filter((c) => c.parentError === Infinity).reduce((n, c) => n + c.triangleCount, 0) <= 4 * 124);
    assertCutsClosed("sphere", sphere, eyeList, [0.5, 1, 1e9]);
    // the seamed rock, its seam copies' zeros written as -0
    const r = rock(5, true), rv = verticesOf(r);
    for (const v of rv.slice(rv.length / 2)) v.position = v.position.map((x) => x === 0 ? -0 : x) as Vec3;
    assertCutsClosed("rock with -0", build(geometry("rock -0", rv, Array.from(r.indices!))), eyeList, [0.5, 1, 1e9]);
});

test("uv islands reduce like one piece", () => {
    const far = view([0, 0, 40], 1);
    const whole = build(rockIslands(5, 1)), islands = build(rockIslands(5, 16));
    const one = triangleCount(whole, whole.select(far)), sixteen = triangleCount(islands, islands.select(far));
    assert(sixteen <= 2 * one, `from 40 m: ${sixteen} triangles in 16 islands, ${one} in one piece`);
    const many = build(rockIslands(5, 64));
    const root = many.clusters.filter((c) => c.parentError === Infinity).reduce((n, c) => n + c.triangleCount, 0);
    assert(root <= 1600, `${root} root triangles in 64 islands`);
    assertCutsClosed("islands", islands, eyes(8, 1.5, 500, { value: 3 }), [1]);
});

test("the cone test never culls a cluster facing the eye", () => {
    const mesh = build(rock(5, false));
    let culled = 0;
    for (const eye of eyes(40, 1.2, 200, { value: 5 })) {
        mesh.clusters.forEach((c, i) => {
            if (!clusterBackfacing(c, eye)) return;
            culled++;
            for (const [a, b, d] of triangles(mesh, i)) {
                const pa = position(mesh, a), pb = position(mesh, b), pd = position(mesh, d);
                const e1 = pb.map((x, k) => x - pa[k]), e2 = pd.map((x, k) => x - pa[k]), to = eye.map((x, k) => x - pa[k]);
                const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
                assert(n[0] * to[0] + n[1] * to[1] + n[2] * to[2] <= 1e-5 * Math.hypot(n[0], n[1], n[2]) * Math.hypot(to[0], to[1], to[2]), `cluster ${i} culled from ${eye} with a triangle facing it`);
            }
        });
    }
    assert(culled > 0, "nothing culled");
});

test("clusters are well filled", () => {
    const mesh = build(rock(5, false));
    const fill = mesh.clusters.reduce((n, c) => n + c.triangleCount, 0) / (mesh.clusters.length * DEFAULT_CLUSTER_OPTIONS.maxTriangles);
    assert(fill >= 0.9, `clusters ${fill.toFixed(2)} full`);
});

test("levels partition the clusters in order", () => {
    const mesh = build(rock(4, false));
    const levels = mesh.levels();
    assert(levels.length > 3, `${levels.length} levels`);
    let next = 0;
    levels.forEach((level, l) => {
        assertEq(level.first, next, `level ${l} starts where the one before ends`);
        assert(level.count > 0, `level ${l} is empty`);
        for (const c of mesh.clusters.slice(level.first, level.first + level.count)) {
            assertEq(c.level, l);
            assert(c.error >= level.minError);
            assert(!Number.isFinite(c.parentError) || c.parentError <= level.maxParentError);
            assert(Math.hypot(...c.lodBounds.center) - c.lodBounds.radius <= level.nearReach + 1e-6);
        }
        next += level.count;
    });
    assertEq(next, mesh.clusters.length);
    assert(!Number.isFinite(levels[levels.length - 1].maxParentError), "the root level has no parent");
});

test("the level window keeps every drawn cluster", () => {
    const mesh = build(rock(4, false));
    const levels = mesh.levels();
    let checked = 0;
    for (const eye of [...eyes(40, 0.5, 600, { value: 7 }), [0, 0, 0] as Vec3, [0, 0, 1.02] as Vec3]) {
        for (const threshold of [0, 0.25, 1, 4, 32]) {
            const v = view(eye, threshold);
            for (const i of mesh.select(v)) {
                assert(levelMayDraw(levels[mesh.clusters[i].level], v), `cluster ${i} drawn from ${eye} at ${threshold} px, its level skipped`);
                checked++;
            }
        }
    }
    assert(checked > 1000, `${checked}`);
});

test("the level window skips the levels a view cannot reach", () => {
    const mesh = build(rock(4, false));
    const levels = mesh.levels();
    const skipped = (v: LodView) => levels.map((l) => !levelMayDraw(l, v));
    const far = skipped(view([0, 0, 400], 1));
    assert(far[0] && far[1], `far: ${far}`);
    const near = skipped(view([0, 0, 1.02], 0.25));
    assert(!near[0] && near[near.length - 1], `near: ${near}`);
});

test("orthographic cuts are closed and ignore distance", () => {
    const mesh = build(rock(5, false));
    const whole = area(mesh, levelZero(mesh));
    const keys = positionKeys(mesh, 1e-5);
    const ortho = (eye: Vec3, pixelsPerMetre: number, threshold: number): LodView => ({ eye, pixelsPerRadian: pixelsPerMetre, near: 0.1, threshold, orthographic: true });
    const counts: number[] = [];
    for (const ppm of [2, 20, 200]) {
        for (const threshold of [0.5, 1, 4]) {
            const near = mesh.select(ortho([0, 10, 0], ppm, threshold)), far = mesh.select(ortho([0, 100, 0], ppm, threshold));
            assertEq(near, far, `${ppm} px/m, budget ${threshold}: the eye's distance changed the cut`);
            const covered = area(mesh, near) / whole;
            assert(covered >= 0.8 && covered < 1.2, `${ppm} px/m, budget ${threshold}: the cut covers ${covered}`);
            assertEq(badEdges(mesh, keys, near), 0, `${ppm} px/m, budget ${threshold}: an open cut`);
            counts.push(triangleCount(mesh, near));
        }
    }
    assert(counts[1] < counts[4] && counts[4] < counts[7], `${counts}`);
});
