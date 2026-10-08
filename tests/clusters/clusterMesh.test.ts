// The cluster graph's data (rust/kansei-core/src/clusters/{tests,gpu_tests}.rs, ported where they
// need no GPU): spheres, the packed words, the cut rule and level window on a hand-made graph.
import { assert, assertEq, test } from "../harness";
import {
    CLUSTER_WORDS, Cluster, ClusterMesh, LEVEL_WORDS, LodView, NO_PARENT, Sphere, VERTEX_WORDS,
    enclosingSphere, levelMayDraw, projectedError, sphereContains,
} from "../../src/clusters/ClusterMesh";

const sphere = (x: number, y: number, z: number, radius: number): Sphere => ({ center: [x, y, z], radius });

test("an enclosing sphere contains every sphere", () => {
    const spheres = [sphere(0, 0, 0, 1), sphere(3, 0, 0, 0.5), sphere(0, -2, 1, 2), sphere(0.5, 0, 0, 0.1)];
    const s = enclosingSphere(spheres);
    for (const o of spheres) assert(sphereContains(s, o), `${JSON.stringify(s)} misses ${JSON.stringify(o)}`);
    // one inside another: the outer one
    const outer = sphere(0.1, 0, 0, 2);
    assertEq(enclosingSphere([sphere(0, 0, 0, 0.5), outer]), outer);
});

/**
 * A graph by hand: a unit quad in the z = 0 plane as two one-triangle clusters (level 0, error
 * 0), whose group was simplified into one cluster of the same two triangles (level 1, error 0.01),
 * the root (no parent).
 */
function quad(): ClusterMesh {
    const vertices = new Float32Array(4 * VERTEX_WORDS);
    [[0, 0], [1, 0], [1, 1], [0, 1]].forEach(([x, y], i) => vertices.set([x, y, 0, 1, 0, 0, 1, x, y], i * VERTEX_WORDS));
    const bounds = sphere(0.5, 0.5, 0, 0.75);
    const base = { coneApex: [0.5, 0.5, 0] as [number, number, number], coneAxis: [0, 0, -1] as [number, number, number], coneCutoff: 0.5, card: false };
    const clusters: Cluster[] = [
        { ...base, vertexOffset: 0, vertexCount: 3, triangleOffset: 0, triangleCount: 1, bounds: sphere(0.6, 0.3, 0, 0.6), error: 0, lodBounds: bounds, parentError: 0.01, parentBounds: bounds, level: 0 },
        { ...base, vertexOffset: 3, vertexCount: 3, triangleOffset: 1, triangleCount: 1, bounds: sphere(0.3, 0.6, 0, 0.6), error: 0, lodBounds: bounds, parentError: 0.01, parentBounds: bounds, level: 0 },
        { ...base, vertexOffset: 6, vertexCount: 4, triangleOffset: 2, triangleCount: 2, bounds, error: 0.01, lodBounds: bounds, parentError: Infinity, parentBounds: bounds, level: 1 },
    ];
    const clusterVertices = new Uint32Array([0, 1, 2, 0, 2, 3, 0, 1, 2, 3]);
    const clusterTriangles = new Uint8Array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2, 3]);
    return new ClusterMesh(vertices, clusters, clusterVertices, clusterTriangles);
}

const view = (z: number, threshold = 1): LodView => ({ eye: [0.5, 0.5, z], pixelsPerRadian: 1000, near: 0.01, threshold, orthographic: false });

test("the gpu words hold the clusters and levels", () => {
    const mesh = quad();
    const words = mesh.gpuWords();
    const section = (k: number) => words[k];
    const levels = mesh.levels();
    assertEq([section(5), section(6), words[7]], [mesh.clusters.length, levels.length, mesh.maxTriangles()]);
    assertEq(words.length, section(4) + levels.length * LEVEL_WORDS);
    const f = new Float32Array(words.buffer);
    assertEq(f[section(0) + 2 * VERTEX_WORDS + 7], 1, "vertex 2's u");
    mesh.clusters.forEach((c, i) => {
        const record = words.subarray(section(3) + i * CLUSTER_WORDS, section(3) + (i + 1) * CLUSTER_WORDS);
        const decoded: number[] = [];
        for (let t = 0; t < record[2]; t++) {
            const packed = words[section(2) + record[1] + t];
            for (let k = 0; k < 3; k++) decoded.push(words[section(1) + record[0] + ((packed >> (8 * k)) & 0xff)]);
        }
        assertEq(decoded, Array.from(mesh.triangles(i)), `cluster ${i}`);
        assertEq(new Float32Array(record.slice(15, 16).buffer)[0], Math.fround(c.error));
        assertEq(new Float32Array(record.slice(24, 25).buffer)[0], Number.isFinite(c.parentError) ? Math.fround(c.parentError) : NO_PARENT);
    });
    const root = words.subarray(section(4) + (levels.length - 1) * LEVEL_WORDS);
    assertEq([root[0], root[1], new Float32Array(root.slice(3, 4).buffer)[0]], [levels[1].first, levels[1].count, NO_PARENT]);
    // no infinity or NaN among the records' floats
    assert(Array.from(words.subarray(section(3))).every((w) => ((w >>> 23) & 0xff) !== 0xff), "an infinity or NaN reaches the GPU");
});

test("a mesh read back from its gpu words packs the same words", () => {
    const words = quad().gpuWords();
    const back = ClusterMesh.fromGpuWords(words);
    assertEq(Array.from(back.gpuWords()), Array.from(words));
    assertEq(back.clusters.map((c) => c.vertexCount), [3, 3, 4]);
    assertEq(back.clusters[2].parentError, Infinity);
});

test("each point is drawn by exactly one cluster of a cut", () => {
    const mesh = quad();
    // near: the two fine clusters; far: the coarse one
    assertEq(mesh.select(view(2)), [0, 1]);
    assertEq(mesh.select(view(100)), [2]);
    // zero budget: the full mesh
    assertEq(mesh.select(view(100, 0)), [0, 1]);
    assertEq(Array.from(mesh.cutGeometry("cut", view(100)).indices!), [0, 1, 2, 0, 2, 3]);
    assertEq(projectedError(0.01, mesh.clusters[2].lodBounds, view(10.75)), 1);
});

test("the level window keeps every drawn cluster", () => {
    const mesh = quad();
    const levels = mesh.levels();
    for (const z of [0.2, 1, 2, 5, 9.9, 10.75, 11, 30, 1000]) {
        for (const threshold of [0.5, 1, 4]) {
            const v = view(z, threshold);
            for (const c of mesh.select(v)) {
                assert(levelMayDraw(levels[mesh.clusters[c].level], v), `z ${z} tau ${threshold}: cluster ${c}'s level skipped`);
            }
        }
    }
});
