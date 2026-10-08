// The TS port of optimesh's meshlets, cluster bounds, partition and simplifier, against the
// Rust crate's output on the same rock, with and without a uv seam
// (`tests/fixtures/meshopt-rock3*.json`, written by `rust/tools/meshopt-fixture`, which calls
// optimesh 1.1 as `ClusterMesh::build` does): every index and every float's bits must match.
// MESHOPT_FIXTURE=<path> checks another fixture of the same form (a larger rock, say).
import { readFileSync } from "node:fs";
import process from "node:process";
import { assertEq, test } from "../harness";
import { buildMeshlets } from "../../src/clusters/meshopt/clusterizer";
import { computeClusterBounds } from "../../src/clusters/meshopt/clusterBounds";
import { partitionClusters } from "../../src/clusters/meshopt/partition";
import { SIMPLIFY_ERROR_ABSOLUTE, SIMPLIFY_SPARSE, simplifyWithAttributes } from "../../src/clusters/meshopt/simplifier";

interface Fixture {
    positions: number[];
    attributes: number[];
    indices: number[];
    meshlets: [number, number, number, number][];
    meshletVertices: number[];
    meshletTriangles: number[];
    bounds: number[][];
    partitions: number[];
    partitionCount: number;
    lock: number[];
    merged: number[];
    target: number;
    simplified: number[];
    simplifyError: number;
}

const fixtures = ["tests/fixtures/meshopt-rock3.json", "tests/fixtures/meshopt-rock3-seam.json", ...(process.env.MESHOPT_FIXTURE ? [process.env.MESHOPT_FIXTURE] : [])];
const floats = (bits: number[]) => new Float32Array(Uint32Array.from(bits).buffer);
const bitsOf = (values: number[]) => Array.from(new Uint32Array(Float32Array.from(values).buffer));

for (const path of fixtures) {
    const fx: Fixture = JSON.parse(readFileSync(path, "utf8"));
    const positions = floats(fx.positions);
    const indices = Uint32Array.from(fx.indices);
    const vertexCount = positions.length / 3;
    const name = path.split("/").pop();

    test(`meshlets are Rust's (${name})`, () => {
        const m = buildMeshlets(indices, { data: positions, count: vertexCount, stride: 3 }, 128, 124, 0.25);
        assertEq(m.meshlets.map((x) => [x.vertexOffset, x.triangleOffset, x.vertexCount, x.triangleCount]), fx.meshlets);
        const last = m.meshlets[m.meshlets.length - 1];
        assertEq(Array.from(m.vertices.subarray(0, last.vertexOffset + last.vertexCount)), fx.meshletVertices);
        assertEq(Array.from(m.triangles.subarray(0, last.triangleOffset + last.triangleCount * 3)), fx.meshletTriangles);
    });

    // each meshlet's triangles as vertex indices
    const globals = fx.meshlets.map(([vo, to, , tc]) => Uint32Array.from({ length: tc * 3 }, (_, k) => fx.meshletVertices[vo + fx.meshletTriangles[to + k]]));

    test(`cluster bounds are Rust's (${name})`, () => {
        globals.forEach((g, i) => {
            const b = computeClusterBounds(g, positions, 3);
            assertEq(bitsOf([...b.center, b.radius, ...b.coneApex, ...b.coneAxis, b.coneCutoff]), fx.bounds[i], `meshlet ${i}`);
        });
    });

    test(`partitions are Rust's (${name})`, () => {
        const flat = new Uint32Array(globals.reduce((n, g) => n + g.length, 0));
        let at = 0;
        for (const g of globals) {
            flat.set(g, at);
            at += g.length;
        }
        const p = partitionClusters(flat, Uint32Array.from(globals.map((g) => g.length)), positions, vertexCount, 3, 4);
        assertEq(p.count, fx.partitionCount);
        assertEq(Array.from(p.partitions), fx.partitions);
    });

    test(`simplification is Rust's (${name})`, () => {
        const s = simplifyWithAttributes(Uint32Array.from(fx.merged), { positions, count: vertexCount, stride: 3 },
            { data: floats(fx.attributes), stride: 5, weights: [0.5, 0.5, 0.5, 0.1, 0.1] }, Uint8Array.from(fx.lock),
            { targetIndexCount: fx.target, targetError: 3.4028234663852886e38, options: SIMPLIFY_SPARSE | SIMPLIFY_ERROR_ABSOLUTE });
        assertEq(Array.from(s.indices), fx.simplified);
        assertEq(bitsOf([s.error])[0], fx.simplifyError);
    });
}
