// rust/kansei-core/src/collision/tests.rs, ported.
import { quat, vec3 } from "gl-matrix";
import { assert, assertClose, assertEq, test } from "../harness";
import { CollisionWorld, Obb, TriangleMesh, Triangle, closestPointTriangle, rayTriangle } from "../../src/collision/CollisionWorld";

function hash(i: number): number {
    const x = ((Math.imul(i, 747796405) + 2891336453) >>> 0) ^ (Math.imul(i >>> 7, 277803737) >>> 0);
    return ((x >>> 0) % 100003) / 100003;
}

function randomUnit(i: number): vec3 {
    const v = vec3.fromValues(hash(i) - 0.5, hash(i + 1) - 0.5, hash(i + 2) - 0.5);
    return vec3.length(v) > 0 ? vec3.normalize(v, v) : vec3.fromValues(1, 0, 0);
}

/** glam's `Quat::from_euler(EulerRot::YXZ, y, x, z)`. */
export function eulerYXZ(y: number, x: number, z: number): quat {
    const q = quat.setAxisAngle(quat.create(), [0, 1, 0], y);
    quat.multiply(q, q, quat.setAxisAngle(quat.create(), [1, 0, 0], x));
    return quat.multiply(q, q, quat.setAxisAngle(quat.create(), [0, 0, 1], z));
}

const v = (x: number, y: number, z: number) => vec3.fromValues(x, y, z);
const unit = (a: vec3) => vec3.normalize(vec3.create(), a);
const along = (o: vec3, d: vec3, t: number) => vec3.scaleAndAdd(vec3.create(), o, d, t);

/** The box as 12 triangles. */
function boxMesh(b: Obb): TriangleMesh {
    const corner = (i: number) => {
        const local = v(b.halfExtents[0] * (i & 1 ? 1 : -1), b.halfExtents[1] * (i & 2 ? 1 : -1), b.halfExtents[2] * (i & 4 ? 1 : -1));
        return vec3.add(local, b.center, vec3.transformQuat(local, local, b.rotation));
    };
    const quads = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]];
    return new TriangleMesh(quads.flatMap((q): Triangle[] => [[corner(q[0]), corner(q[1]), corner(q[2])], [corner(q[0]), corner(q[2]), corner(q[3])]]));
}

function tiltedBox(): Obb {
    return new Obb(v(0.3, 0.8, -0.2), v(1, 0.5, 0.7), eulerYXZ(0.7, 0.3, -0.2));
}

test("rays hit boxes where the faces are", () => {
    let b = Obb.fromMinMax(v(-1, 0, -1), v(1, 2, 1));
    let [t, n] = b.raycast(v(-5, 0.5, 0), v(1, 0, 0), 10)!;
    assert(Math.abs(t - 4) < 1e-6 && vec3.equals(n, v(-1, 0, 0)), `${t} ${n}`);
    assert(b.raycast(v(-5, 2.5, 0), v(1, 0, 0), 10) === undefined);
    assert(b.raycast(v(-5, 0.5, 0), v(1, 0, 0), 3) === undefined);
    assertEq(b.raycast(v(0, 0.5, 0), v(1, 0, 0), 3)![0], 0);
    [t, n] = b.raycast(v(0.2, 5, 0.3), v(0, -1, 0), 10)!;
    assert(Math.abs(t - 3) < 1e-6 && vec3.equals(n, v(0, 1, 0)));
    // a turned box agrees with its triangles
    b = tiltedBox();
    const mesh = boxMesh(b);
    let hits = 0;
    for (let i = 0; i < 300; i++) {
        const origin = along(b.center, randomUnit(i * 7), 4);
        const direction = unit(vec3.subtract(vec3.create(), along(b.center, randomUnit(i * 7 + 3), 0.8), origin));
        const x = b.raycast(origin, direction, 10), y = mesh.raycast(origin, direction, 10);
        assertEq(x !== undefined, y !== undefined, `ray ${i}`);
        if (x && y) {
            assert(Math.abs(x[0] - y[0]) < 1e-4 && vec3.dot(x[1], y[1]) > 0.999, `ray ${i}: ${x} ${y}`);
            hits++;
        }
    }
    assert(hits > 100, `${hits} hits`);
});

/** The first distance at which a sphere moving along a ray touches the box, by small steps. */
function sampledSphereCast(b: Obb, o: vec3, r: number, d: vec3, max: number): number | undefined {
    const step = 1e-3;
    for (let k = 0; k <= Math.floor(max / step); k++) {
        const p = along(o, d, k * step);
        if (vec3.distance(b.closestPoint(p), p) <= r) return k * step;
    }
    return undefined;
}

test("swept spheres touch boxes at the first contact", () => {
    let b = Obb.fromMinMax(v(-1, 0, -1), v(1, 2, 1));
    let [t, n] = b.sphereCast(v(-5, 1, 0), 0.5, v(1, 0, 0), 10)!;
    assert(Math.abs(t - 3.5) < 1e-5, `${t}`);
    assertClose(n, [-1, 0, 0], 1e-5);
    // grazing a vertical edge: the rounded corner, not the grown box's corner
    [t, n] = b.sphereCast(v(-5, 1, -1.4), 0.5, v(1, 0, 0), 10)!;
    const expected = 4 - Math.sqrt(0.25 - 0.16);
    assert(Math.abs(t - expected) < 1e-4, `${t} vs ${expected}`);
    assert(n[0] < 0 && n[2] < 0);
    // starting in contact
    assertEq(b.sphereCast(v(-1.2, 1, 0), 0.5, v(1, 0, 0), 10)![0], 0);
    // random sweeps against a turned box, against small steps and against its triangles
    b = tiltedBox();
    const mesh = boxMesh(b);
    for (let i = 0; i < 120; i++) {
        const origin = along(b.center, randomUnit(i * 11), 3.5);
        const direction = unit(vec3.subtract(vec3.create(), along(b.center, randomUnit(i * 11 + 5), 1.2), origin));
        const r = 0.1 + hash(i * 3) * 0.5;
        const cast = b.sphereCast(origin, r, direction, 8)?.[0];
        const sampled = sampledSphereCast(b, origin, r, direction, 8);
        assertEq(cast !== undefined, sampled !== undefined, `sweep ${i}: ${cast} vs ${sampled}`);
        if (cast !== undefined && sampled !== undefined) assert(Math.abs(cast - sampled) < 3e-3, `sweep ${i}: ${cast} vs ${sampled}`);
        const m = mesh.sphereCast(origin, r, direction, 8)?.[0];
        assert((cast === undefined) === (m === undefined) && (cast === undefined || Math.abs(cast - m!) < 1e-3), `sweep ${i}: box ${cast} mesh ${m}`);
    }
});

test("triangles' closest points are the nearest samples", () => {
    const t: Triangle = [v(0, 0, 0), v(2, 0.3, 0), v(0.5, 0.1, 1.5)];
    const e1 = vec3.subtract(vec3.create(), t[1], t[0]), e2 = vec3.subtract(vec3.create(), t[2], t[0]);
    for (let i = 0; i < 200; i++) {
        const p = vec3.scale(vec3.create(), randomUnit(i * 5), 3);
        const [q] = closestPointTriangle(p, t);
        let best = Infinity;
        for (let a = 0; a <= 60; a++) {
            for (let b = 0; b <= 60 - a; b++) {
                const s = vec3.scaleAndAdd(vec3.create(), vec3.scaleAndAdd(vec3.create(), t[0], e1, a / 60), e2, b / 60);
                best = Math.min(best, vec3.distance(s, p));
            }
        }
        const d = vec3.distance(q, p);
        assert(d <= best + 1e-5 && d > best - 0.03, `${p}: ${d} vs ${best}`);
    }
    assert(rayTriangle(v(0.5, 5, 0.3), v(0, -1, 0), t) !== undefined);
    assert(rayTriangle(v(3, 5, 3), v(0, -1, 0), t) === undefined);
});

test("spheres are pushed out to touching", () => {
    const b = tiltedBox();
    for (let i = 0; i < 100; i++) {
        const center = along(b.center, randomUnit(i * 13), 0.2 + hash(i) * 1.5);
        const r = 0.3;
        const push = b.spherePenetration(center, r);
        const moved = vec3.add(vec3.create(), center, push);
        const gap = vec3.distance(b.closestPoint(moved), moved);
        if (vec3.length(push) > 0) assert(Math.abs(gap - r) < 1e-4, `${i}: ${gap}`);
        else assert(vec3.distance(b.closestPoint(center), center) >= r - 1e-6);
    }
});

test("a world answers casts, overlaps, ground and capsules", () => {
    const world = new CollisionWorld();
    // a floor slab, a 1 m box and a 0.2 m curb
    world.addBox(Obb.fromMinMax(v(-50, -1, -50), v(50, 0, 50)));
    const crate = world.addBox(Obb.fromMinMax(v(2, 0, -0.5), v(3, 1, 0.5)));
    world.addBox(Obb.fromMinMax(v(-3, 0, -1), v(-2, 0.2, 1)));
    const tri = world.add(new TriangleMesh([[v(10, 0, -1), v(10, 3, 0), v(10, 0, 1)]]), 2);
    const ALL = 0xffffffff;

    let hit = world.raycast(v(0, 0.5, 0), v(1, 0, 0), 20, ALL)!;
    assertEq(hit.collider, crate);
    assertClose(hit.point, [2, 0.5, 0], 1e-5);
    assert(vec3.equals(hit.normal, v(-1, 0, 0)));
    // the mesh sits on layer 2: a layer-1 ray above the crate misses it
    assert(world.raycast(v(0, 1.5, 0), v(1, 0, 0), 20, 1) === undefined);
    assertEq(world.raycast(v(0, 1.5, 0), v(1, 0, 0), 20, 2)!.collider, tri);
    hit = world.sphereCast(v(0, 0.5, 0), 0.3, v(1, 0, 0), 20, ALL)!;
    assert(Math.abs(hit.distance - 1.7) < 1e-5);
    assertClose(hit.point, [1.7, 0.5, 0], 1e-5);

    assertEq(world.groundHeight(v(2.5, 0, 0), 2, 2, ALL), 1);
    assertEq(world.groundHeight(v(0, 0, 0), 2, 2, ALL), 0);
    assert(world.overlapSphere(v(1.9, 0.5, 0), 0.2, ALL));
    assert(!world.overlapCapsule(v(0, 0.5, 0), v(0, 1.5, 0), 0.3, ALL));

    // a capsule half into the crate's side moves out sideways; the curb is under the step
    const out = world.resolveCapsule(v(1.9, 0, 0.1), 1.8, 0.3, 0.3, ALL);
    assert(Math.abs(out[0] - 1.7) < 1e-3 && Math.abs(out[2] - 0.1) < 1e-3 && out[1] === 0, `${out}`);
    const onCurb = world.resolveCapsule(v(-2.5, 0, 0), 1.8, 0.3, 0.3, ALL);
    assertClose(onCurb, [-2.5, 0, 0], 0);
});
