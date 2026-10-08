// rust/kansei-core/src/clusters/card_tests.rs, ported
import { assert, assertEq, test } from "../harness";
import { Geometry } from "../../src/buffers/Geometry";
import { ClusterMesh, ClusterOptions, DEFAULT_CLUSTER_OPTIONS, Vec3, sphereContains } from "../../src/clusters/ClusterMesh";
import { positionIds } from "../../src/clusters/build";
import { findCards } from "../../src/clusters/cards";
import { badEdges, crown, eyes, geometry, position, positionKeys, rock, triangleCount, triangles, verticesOf, view } from "./meshes";

const cardOptions = (more: Partial<ClusterOptions> = {}): Partial<ClusterOptions> => ({ cards: true, ...more });

/** `g` with `other`'s vertices and triangles appended, its positions moved by `offset`. */
function append(g: Geometry, other: Geometry, offset: Vec3 = [0, 0, 0]): Geometry {
    const base = verticesOf(g).length;
    const vertices = [...verticesOf(g), ...verticesOf(other).map((v) => ({ ...v, position: v.position.map((x, k) => x + offset[k]) as Vec3 }))];
    return geometry(g.label, vertices, [...Array.from(g.indices!), ...Array.from(other.indices!, (i) => i + base)]);
}

/** Positions and position ids of `g` (unwelded). */
function welded(g: Geometry): [Float32Array, Uint32Array] {
    const positions = Float32Array.from(verticesOf(g).flatMap((v) => v.position));
    return [positions, positionIds(positions)];
}

test("small open flat components are cards", () => {
    // 50 cards, a closed rock, an open tube (24 triangles: its normals cancel) and a flat grid too large to be a card
    const r = rock(2, false);
    let g = append(crown(50), r);
    const vertices = verticesOf(g), indices = Array.from(g.indices!);
    let base = vertices.length;
    for (let k = 0; k <= 12; k++) {
        const a = k / 12 * 2 * Math.PI;
        for (const y of [0, 2]) vertices.push({ position: [20 + Math.cos(a), y, Math.sin(a)], normal: [Math.cos(a), 0, Math.sin(a)], uv: [0, 0] });
    }
    for (let k = 0; k < 12; k++) {
        const [a, b, c, d] = [base + 2 * k, base + 2 * k + 1, base + 2 * k + 2, base + 2 * k + 3];
        indices.push(a, c, b, b, c, d);
    }
    base = vertices.length;
    for (let j = 0; j <= 10; j++) for (let i = 0; i <= 10; i++) vertices.push({ position: [-20 + i, 0, j], normal: [0, 1, 0], uv: [0, 0] });
    for (let j = 0; j < 10; j++) {
        for (let i = 0; i < 10; i++) {
            const [a, b, c, d] = [base + j * 11 + i, base + j * 11 + i + 1, base + (j + 1) * 11 + i, base + (j + 1) * 11 + i + 1];
            indices.push(a, c, b, b, c, d);
        }
    }
    g = geometry("mixed", vertices, indices);
    const options = { ...DEFAULT_CLUSTER_OPTIONS, cards: true };
    const [positions, ids] = welded(g);
    const { cards, rest } = findCards(g.indices!, positions, ids, options);
    assertEq(cards.length, 50);
    assert(cards.every((c) => c.triangles.length === 2 && c.vertices.length === 4 && Math.abs(c.area - 0.35) < 1e-3 && c.radius > 0.3), JSON.stringify(cards.map((c) => [c.triangles.length, c.area])));
    assertEq(rest.length / 3, r.indices!.length / 3 + 24 + 200);
    // a zero-area triangle doesn't break a card
    const three = crown(3);
    const withDegenerate = geometry("crown", verticesOf(three), [...Array.from(three.indices!), 0, 0, 1]);
    const [p3, ids3] = welded(withDegenerate);
    assertEq(findCards(withDegenerate.indices!, p3, ids3, options).cards.length, 3);
});

/** The drawn cards' area per cell of a `cells`³ grid over `bounds`, and in all. */
function areaPerCell(mesh: ClusterMesh, cut: number[], [lo, hi]: [Vec3, Vec3], cells: number): [number[], number] {
    const per = new Array<number>(cells ** 3).fill(0);
    let total = 0;
    for (const c of cut) {
        for (const [a, b, d] of triangles(mesh, c)) {
            const pa = position(mesh, a), pb = position(mesh, b), pd = position(mesh, d);
            const e1 = pb.map((x, k) => x - pa[k]), e2 = pd.map((x, k) => x - pa[k]);
            const area = 0.5 * Math.hypot(e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]);
            const cell = [0, 1, 2].map((k) => Math.trunc(Math.min(Math.max((pa[k] + pb[k] + pd[k]) / 3 - lo[k], 0) / (hi[k] - lo[k]) * cells, cells - 1)));
            per[(cell[2] * cells + cell[1]) * cells + cell[0]] += area;
            total += area;
        }
    }
    return [per, total];
}

test("pruned levels hold the cards' area", () => {
    const mesh = ClusterMesh.build(crown(800), cardOptions());
    assert(mesh.clusters.every((c) => c.card));
    assert(mesh.levels().length >= 4, `${mesh.levels().length} levels`);
    const bounds: [Vec3, Vec3] = [[-3.5, -0.5, -3.5], [3.5, 10.5, 3.5]];
    const level0 = mesh.clusters.flatMap((c, i) => c.level === 0 ? [i] : []);
    const [cells0, total0] = areaPerCell(mesh, level0, bounds, 3);
    const counts: number[] = [];
    for (const eye of [...eyes(30, 5, 3000, { value: 5 }), [0, 5, 40] as Vec3, [0, 5, 4000] as Vec3]) {
        for (const threshold of [0.5, 1, 4]) {
            const cut = mesh.select(view(eye, threshold));
            const [cells, total] = areaPerCell(mesh, cut, bounds, 3);
            assert(Math.abs(total / total0 - 1) < 0.1, `eye ${eye}, ${threshold} px: ${total} of ${total0}`);
            // (a cell is held to it where it keeps enough cards for the share to mean something)
            const kept = triangleCount(mesh, cut) / 1600;
            cells.forEach((a, k) => {
                if (cells0[k] / 0.35 * kept >= 8) assert(Math.abs(a / cells0[k] - 1) < 0.35, `eye ${eye}, ${threshold} px: cell ${k} holds ${a} of ${cells0[k]}`);
            });
            counts.push(triangleCount(mesh, cut));
        }
    }
    // from 4 km at 1 px, a small share of the cards
    const far = counts[counts.length - 2];
    assert(far * 8 < 1600, `${far} triangles from 4 km`);
    // and there each part of the crown keeps about its area
    const [cells] = areaPerCell(mesh, mesh.select(view([0, 5, 4000], 1)), bounds, 3);
    cells.forEach((a, k) => {
        if (cells0[k] > 0.05 * total0) assert(Math.abs(a / cells0[k] - 1) <= 0.2, `from 4 km, cell ${k} holds ${a} of ${cells0[k]}`);
    });
});

test("levels of cards nest like simplified levels", () => {
    const mesh = ClusterMesh.build(crown(800), cardOptions());
    for (const c of mesh.clusters) {
        if (!Number.isFinite(c.parentError)) continue;
        assert(c.parentError >= c.error, `${c.parentError} < ${c.error}`);
        assert(sphereContains(c.parentBounds, c.lodBounds));
    }
    const roots = mesh.clusters.filter((c) => !Number.isFinite(c.parentError)).length;
    assert(roots <= 4, `${roots} roots`);
});

test("a mixed mesh prunes its cards and simplifies the rest", () => {
    // a rock under a crown of cards, as one mesh
    const g = append(crown(400), rock(4, false), [0, -2, 0]);
    const mesh = ClusterMesh.build(g, cardOptions());
    assert(mesh.clusters.some((c) => c.card) && mesh.clusters.some((c) => !c.card));
    const keys = positionKeys(mesh, 1e-5);
    for (const eye of eyes(20, 4, 1000, { value: 9 })) {
        for (const threshold of [0.5, 2]) {
            const solid = mesh.select(view(eye, threshold)).filter((c) => !mesh.clusters[c].card);
            assertEq(badEdges(mesh, keys, solid), 0, `eye ${eye}: the rock's cut is open`);
        }
    }
    // without `cards`, the same mesh is all solid
    assert(ClusterMesh.build(g).clusters.every((c) => !c.card));
});

test("the card error scale moves the switch nearer", () => {
    const count = (scale: number) => {
        const mesh = ClusterMesh.build(crown(800), cardOptions({ cardErrorScale: scale }));
        return triangleCount(mesh, mesh.select(view([0, 5, 250], 1)));
    };
    const full = count(1), quarter = count(0.25);
    assert(quarter < full, `${quarter} with a quarter of the error, ${full} with all of it`);
});

/** How far past the crown (its cone widened by `margin`) the area `cut` draws reaches, at most. */
function reachPastTheCrown(mesh: ClusterMesh, cut: number[], margin: number): number {
    let worst = 0;
    for (const c of cut) {
        for (const t of triangles(mesh, c)) {
            for (const v of t) {
                const q = position(mesh, v);
                const radius = 3 * Math.max(1 - q[1] / 10, 0) + margin;
                worst = Math.max(worst, Math.hypot(q[0], q[2]) - radius, q[1] - 10 - margin);
            }
        }
    }
    return worst;
}

test("the silhouette stays within the budget at any error scale", () => {
    for (const ces of [1, 0.25, 0.1]) {
        const mesh = ClusterMesh.build(crown(800), cardOptions({ cardErrorScale: ces }));
        for (const d of [30, 60, 120, 250, 500, 1000, 2000, 4000]) {
            const v = view([d, 5, 0], 1);
            // (level 0's cards reach 0.45 m past the cone their centres fill)
            const pixels = reachPastTheCrown(mesh, mesh.select(v), 0.45) / (d - 3) * v.pixelsPerRadian;
            assert(pixels <= 1, `cardErrorScale ${ces}, ${d} m away: the crown reaches ${pixels.toFixed(2)} px past its own`);
        }
    }
    const count = (ces: number) => {
        const mesh = ClusterMesh.build(crown(800), cardOptions({ cardErrorScale: ces }));
        return triangleCount(mesh, mesh.select(view([250, 5, 0], 1)));
    };
    assert(count(0.25) < count(1), `at 250 m: ${count(0.25)} triangles at 0.25, ${count(1)} at 1`);
});
