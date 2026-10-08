import { mat4, quat, vec3 } from "gl-matrix";

/**
 * A minimal CPU collision world: static boxes and triangle meshes, with ray and sphere casts,
 * overlap tests and capsule push-out. Rust: `collision`.
 *
 * Enough for a character on a course of obstacles: ledge detection casts rays and spheres at
 * them, a character capsule is kept out of them, and ground height is a ray down. The shapes and
 * queries are the textbook ones (Ericson, "Real-Time Collision Detection", 2005: slab tests,
 * Möller–Trumbore, closest points on boxes and triangles, rays against capsules for swept
 * spheres). Meshes are tested triangle by triangle behind their bounds; a BVH can come when
 * scenes need it.
 *
 * Casts return `[distance, normal]` pairs on the shapes and `Hit`s on the world; a miss is
 * `undefined`. Directions are unit vectors.
 */

/** A cast's distance and the surface normal (toward the caster). */
type CastResult = [number, vec3];

const v3 = (x: number, y: number, z: number): vec3 => vec3.fromValues(x, y, z);
const sub = (a: vec3, b: vec3): vec3 => vec3.subtract(vec3.create(), a, b);
const add = (a: vec3, b: vec3): vec3 => vec3.add(vec3.create(), a, b);
const scale = (a: vec3, s: number): vec3 => vec3.scale(vec3.create(), a, s);
const madd = (a: vec3, b: vec3, s: number): vec3 => vec3.scaleAndAdd(vec3.create(), a, b, s);
const rotate = (q: quat, v: vec3): vec3 => vec3.transformQuat(vec3.create(), v, q);
const clampBox = (p: vec3, e: vec3): vec3 => v3(
    Math.min(Math.max(p[0], -e[0]), e[0]),
    Math.min(Math.max(p[1], -e[1]), e[1]),
    Math.min(Math.max(p[2], -e[2]), e[2]),
);
const normalizeOrZero = (v: vec3): vec3 => {
    const l = vec3.length(v);
    return l > 0 && Number.isFinite(1 / l) ? scale(v, 1 / l) : vec3.create();
};

/**
 * `out` = (x, y, z) turned by `q`, or by its inverse: gl-matrix's `transformQuat` on scalars,
 * so the hot queries (pushing a capsule out, casting at every box) allocate nothing.
 */
function rotateXYZ(out: vec3, q: quat, x: number, y: number, z: number, inverse: boolean): vec3 {
    const s = inverse ? -1 : 1;
    const qx = s * q[0], qy = s * q[1], qz = s * q[2], w2 = q[3] * 2;
    const uvx = qy * z - qz * y, uvy = qz * x - qx * z, uvz = qx * y - qy * x;
    const uuvx = qy * uvz - qz * uvy, uuvy = qz * uvx - qx * uvz, uuvz = qx * uvy - qy * uvx;
    out[0] = x + uvx * w2 + uuvx * 2;
    out[1] = y + uvy * w2 + uuvy * 2;
    out[2] = z + uvz * w2 + uuvz * 2;
    return out;
}

// scratch for the queries below (never handed out)
const LOCAL = vec3.create();
const LOCAL_DIRECTION = vec3.create();
const AXIS = vec3.create();
const PUSH = vec3.create();

/** An oriented box: centre, half extents along its axes, and the rotation of its axes. */
class Obb {
    constructor(
        public center: vec3,
        public halfExtents: vec3,
        public rotation: quat = quat.create(),
    ) { }

    /** An axis-aligned box from its corners. */
    public static fromMinMax(min: vec3, max: vec3): Obb {
        return new Obb(scale(add(min, max), 0.5), scale(sub(max, min), 0.5), quat.create());
    }

    /** `p` in the box's frame, into `out`. */
    private localPoint(p: vec3, out: vec3 = vec3.create()): vec3 {
        const c = this.center;
        return rotateXYZ(out, this.rotation, p[0] - c[0], p[1] - c[1], p[2] - c[2], true);
    }

    private localVector(v: vec3, out: vec3 = vec3.create()): vec3 {
        return rotateXYZ(out, this.rotation, v[0], v[1], v[2], true);
    }

    /** The point of the box nearest `p` (`p` itself inside). */
    public closestPoint(p: vec3): vec3 {
        const e = this.halfExtents;
        const l = this.localPoint(p, LOCAL);
        const out = rotateXYZ(vec3.create(), this.rotation,
            Math.min(Math.max(l[0], -e[0]), e[0]), Math.min(Math.max(l[1], -e[1]), e[1]), Math.min(Math.max(l[2], -e[2]), e[2]), false);
        return vec3.add(out, out, this.center);
    }

    /** Squared distance from `p` to the box (0 inside). */
    public distanceSquared(p: vec3): number {
        const e = this.halfExtents;
        const l = this.localPoint(p, LOCAL);
        let d2 = 0;
        for (let i = 0; i < 3; i++) {
            const g = l[i] - Math.min(Math.max(l[i], -e[i]), e[i]);
            d2 += g * g;
        }
        return d2;
    }

    public contains(p: vec3): boolean {
        const l = this.localPoint(p, LOCAL);
        return Math.abs(l[0]) <= this.halfExtents[0] && Math.abs(l[1]) <= this.halfExtents[1] && Math.abs(l[2]) <= this.halfExtents[2];
    }

    /** World-space bounds: `[min, max]`. */
    public bounds(): [vec3, vec3] {
        const e = this.halfExtents;
        const reach = vec3.create();
        for (let i = 0; i < 3; i++) {
            const axis = rotate(this.rotation, v3(i === 0 ? 1 : 0, i === 1 ? 1 : 0, i === 2 ? 1 : 0));
            for (let k = 0; k < 3; k++) reach[k] += Math.abs(axis[k]) * e[i];
        }
        return [sub(this.center, reach), add(this.center, reach)];
    }

    /** First hit of a ray (unit `direction`) within `max`: distance and outward normal. */
    public raycast(origin: vec3, direction: vec3, max: number): CastResult | undefined {
        const t = slabInto(this.localPoint(origin, LOCAL), this.localVector(direction, LOCAL_DIRECTION), this.halfExtents, max, AXIS);
        return t === undefined ? undefined : [t, rotateXYZ(vec3.create(), this.rotation, AXIS[0], AXIS[1], AXIS[2], false)];
    }

    /**
     * First contact of a sphere of `radius` moving from `origin` along unit `direction` within
     * `max`: distance and the normal at the contact (from the box toward the sphere). A sphere
     * that starts overlapping hits at 0.
     */
    public sphereCast(origin: vec3, radius: number, direction: vec3, max: number): CastResult | undefined {
        const e = this.halfExtents;
        if (this.distanceSquared(origin) <= radius * radius) {
            return [0, rotate(this.rotation, pushNormal(this.localPoint(origin), e))];
        }
        const o = this.localPoint(origin, LOCAL);
        const d = this.localVector(direction, LOCAL_DIRECTION);
        // the box grown by the radius: its faces are exact, its edges and corners are rounded
        PUSH[0] = e[0] + radius;
        PUSH[1] = e[1] + radius;
        PUSH[2] = e[2] + radius;
        const t = slabInto(o, d, PUSH, max, AXIS);
        if (t === undefined) return undefined;
        let outside = 0;
        for (let i = 0; i < 3; i++) if (Math.abs(o[i] + d[i] * t) - e[i] > 1e-5) outside++;
        if (outside <= 1) return [t, rotateXYZ(vec3.create(), this.rotation, AXIS[0], AXIS[1], AXIS[2], false)];
        // an edge or corner region: the ray against the capsules round the 12 edges
        const corner = (i: number) => v3(i & 1 ? e[0] : -e[0], i & 2 ? e[1] : -e[1], i & 4 ? e[2] : -e[2]);
        let best: number | undefined;
        for (let a = 0; a < 8; a++) {
            for (const bit of [1, 2, 4]) {
                if ((a & bit) !== 0) continue;
                const t = rayCapsule(o, d, corner(a), corner(a | bit), radius);
                if (t !== undefined && t <= max && (best === undefined || t < best)) best = t;
            }
        }
        if (best === undefined) return undefined;
        const c = madd(o, d, best);
        return [best, rotate(this.rotation, normalizeOrZero(sub(c, clampBox(c, e))))];
    }

    /** Push that moves a sphere at `center` out of the box (zero when apart). */
    public spherePenetration(center: vec3, radius: number): vec3 {
        return this.penetrationInto(center, radius, vec3.create());
    }

    /** `spherePenetration` into `out`. */
    public penetrationInto(center: vec3, radius: number, out: vec3): vec3 {
        const o = this.localPoint(center, LOCAL);
        const e = this.halfExtents;
        const gx = o[0] - Math.min(Math.max(o[0], -e[0]), e[0]);
        const gy = o[1] - Math.min(Math.max(o[1], -e[1]), e[1]);
        const gz = o[2] - Math.min(Math.max(o[2], -e[2]), e[2]);
        const distance = Math.sqrt(gx * gx + gy * gy + gz * gz);
        if (distance > radius) return vec3.zero(out);
        if (distance > 1e-6) {
            const k = (radius - distance) / distance;
            return rotateXYZ(out, this.rotation, gx * k, gy * k, gz * k, false);
        }
        // the centre is inside: out through the nearest face
        const depth = v3(e[0] - Math.abs(o[0]), e[1] - Math.abs(o[1]), e[2] - Math.abs(o[2]));
        const axis = minAxis(depth);
        const push = vec3.create();
        push[axis] = o[axis] < 0 ? -(depth[axis] + radius) : depth[axis] + radius;
        return rotateXYZ(out, this.rotation, push[0], push[1], push[2], false);
    }
}

/** Index of the smallest component. */
function minAxis(v: vec3): number {
    if (v[0] <= v[1] && v[0] <= v[2]) return 0;
    return v[1] <= v[2] ? 1 : 2;
}

/** The outward face normal (box space) of the face nearest a point inside or on the box. */
function pushNormal(o: vec3, e: vec3): vec3 {
    const gap = sub(o, clampBox(o, e));
    if (vec3.squaredLength(gap) > 1e-12) return vec3.normalize(gap, gap);
    const depth = v3(e[0] - Math.abs(o[0]), e[1] - Math.abs(o[1]), e[2] - Math.abs(o[2]));
    const axis = minAxis(depth);
    const n = vec3.create();
    n[axis] = o[axis] < 0 ? -1 : 1;
    return n;
}

/**
 * A ray against the axis-aligned box [-e, e]: entry distance within `max` and the entry face's
 * normal; a ray starting inside enters at 0 (normal zero).
 */
function slab(o: vec3, d: vec3, e: vec3, max: number): CastResult | undefined {
    const axis = vec3.create();
    const t = slabInto(o, d, e, max, axis);
    return t === undefined ? undefined : [t, axis];
}

/** `slab`'s distance (`undefined` for a miss), its normal into `axis`. */
function slabInto(o: vec3, d: vec3, e: vec3, max: number, axis: vec3): number | undefined {
    let t0 = -Infinity;
    let t1 = Infinity;
    let entry = -1;
    for (let i = 0; i < 3; i++) {
        if (Math.abs(d[i]) < 1e-12) {
            if (Math.abs(o[i]) > e[i]) return undefined;
            continue;
        }
        const a = (-e[i] - o[i]) / d[i];
        const b = (e[i] - o[i]) / d[i];
        const near = Math.min(a, b);
        const far = Math.max(a, b);
        if (near > t0) {
            t0 = near;
            entry = i;
        }
        t1 = Math.min(t1, far);
        if (t0 > t1) return undefined;
    }
    if (t1 < 0 || t0 > max) return undefined;
    vec3.zero(axis);
    if (t0 < 0) return 0;
    if (entry >= 0) axis[entry] = -Math.sign(d[entry]);
    return t0;
}

/** First distance along a ray (unit `d`) at which it is within `r` of the segment `a`-`b`. */
function rayCapsule(o: vec3, d: vec3, a: vec3, b: vec3, r: number): number | undefined {
    const ab = sub(b, a);
    const ao = sub(o, a);
    const abab = vec3.dot(ab, ab), abd = vec3.dot(ab, d), abao = vec3.dot(ab, ao);
    let best: number | undefined;
    // the cylinder's side
    const qa = abab - abd * abd;
    const qb = abab * vec3.dot(ao, d) - abao * abd;
    const qc = abab * vec3.dot(ao, ao) - abao * abao - r * r * abab;
    if (Math.abs(qa) > 1e-12) {
        const disc = qb * qb - qa * qc;
        if (disc >= 0) {
            const t = (-qb - Math.sqrt(disc)) / qa;
            const s = abao + t * abd;
            if (t >= 0 && s >= 0 && s <= abab) best = t;
        }
    }
    // the end caps
    for (const c of [a, b]) {
        const t = raySphere(o, d, c, r);
        if (t !== undefined && (best === undefined || t < best)) best = t;
    }
    return best;
}

/** First distance along a ray (unit `d`) at which it enters the sphere (`undefined` behind or missed). */
function raySphere(o: vec3, d: vec3, c: vec3, r: number): number | undefined {
    const m = sub(o, c);
    const b = vec3.dot(m, d);
    const cc = vec3.dot(m, m) - r * r;
    if (cc > 0 && b > 0) return undefined;
    const disc = b * b - cc;
    if (disc < 0) return undefined;
    return Math.max(-b - Math.sqrt(disc), 0);
}

/** A triangle: three corners. */
type Triangle = [vec3, vec3, vec3];

/** A triangle soup with its bounds. */
class TriangleMesh {
    private min: vec3;
    private max: vec3;

    constructor(public triangles: Triangle[]) {
        this.min = v3(Infinity, Infinity, Infinity);
        this.max = v3(-Infinity, -Infinity, -Infinity);
        for (const t of triangles) {
            for (const p of t) {
                vec3.min(this.min, this.min, p);
                vec3.max(this.max, this.max, p);
            }
        }
    }

    /** From indexed positions (3 floats each, or `vec3`s), transformed by `transform`. */
    public static fromIndexed(positions: ArrayLike<number> | vec3[], indices: ArrayLike<number>, transform: mat4 = mat4.create()): TriangleMesh {
        const flat = positions.length > 0 && typeof positions[0] !== "number";
        const at = (i: number): vec3 => {
            const p = flat ? vec3.clone((positions as vec3[])[i]) : v3((positions as ArrayLike<number>)[3 * i], (positions as ArrayLike<number>)[3 * i + 1], (positions as ArrayLike<number>)[3 * i + 2]);
            return vec3.transformMat4(p, p, transform);
        };
        const triangles: Triangle[] = [];
        for (let k = 0; k + 2 < indices.length; k += 3) triangles.push([at(indices[k]), at(indices[k + 1]), at(indices[k + 2])]);
        return new TriangleMesh(triangles);
    }

    public bounds(): [vec3, vec3] {
        return [vec3.clone(this.min), vec3.clone(this.max)];
    }

    /** Whether a ray (or a sphere of `pad` radius along it) could touch the bounds within `max`. */
    private mayHit(o: vec3, d: vec3, max: number, pad: number): boolean {
        const e = madd(v3(pad, pad, pad), sub(this.max, this.min), 0.5);
        return slab(sub(o, scale(add(this.min, this.max), 0.5)), d, e, max) !== undefined;
    }

    public raycast(o: vec3, d: vec3, max: number): CastResult | undefined {
        if (!this.mayHit(o, d, max, 0)) return undefined;
        let best: CastResult | undefined;
        for (const t of this.triangles) {
            const dist = rayTriangle(o, d, t);
            if (dist !== undefined && dist <= max && (best === undefined || dist < best[0])) {
                const n = normalizeOrZero(vec3.cross(vec3.create(), sub(t[1], t[0]), sub(t[2], t[0])));
                if (vec3.dot(n, d) > 0) vec3.negate(n, n);
                best = [dist, n];
            }
        }
        return best;
    }

    public sphereCast(o: vec3, r: number, d: vec3, max: number): CastResult | undefined {
        if (!this.mayHit(o, d, max, r)) return undefined;
        let best: number | undefined;
        for (const t of this.triangles) {
            const [closest] = closestPointTriangle(o, t);
            if (vec3.squaredDistance(closest, o) <= r * r) {
                best = 0;
                break;
            }
            let hit: number | undefined;
            // the face, offset toward the sphere
            const n = normalizeOrZero(vec3.cross(vec3.create(), sub(t[1], t[0]), sub(t[2], t[0])));
            if (vec3.dot(n, sub(o, t[0])) < 0) vec3.negate(n, n);
            const denom = vec3.dot(n, d);
            if (denom < -1e-9) {
                const dist = (r - vec3.dot(n, sub(o, t[0]))) / denom;
                if (dist >= 0) {
                    const p = madd(madd(o, d, dist), n, -r);
                    if (vec3.squaredDistance(closestPointTriangle(p, t)[0], p) < 1e-10) hit = dist;
                }
            }
            // the edges (and so the corners)
            for (const [a, b] of [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]]) {
                const dist = rayCapsule(o, d, a, b, r);
                if (dist !== undefined && (hit === undefined || dist < hit)) hit = dist;
            }
            if (hit !== undefined && hit <= max && (best === undefined || hit < best)) best = hit;
        }
        if (best === undefined) return undefined;
        const c = madd(o, d, best);
        let nearest = c;
        let nearestDistance = Infinity;
        for (const t of this.triangles) {
            const [q] = closestPointTriangle(c, t);
            const dd = vec3.squaredDistance(q, c);
            if (dd < nearestDistance) {
                nearestDistance = dd;
                nearest = q;
            }
        }
        return [best, normalizeOrZero(sub(c, nearest))];
    }

    /** `spherePenetration` into `out`. */
    public penetrationInto(center: vec3, radius: number, out: vec3): vec3 {
        return vec3.copy(out, this.spherePenetration(center, radius));
    }

    public spherePenetration(center: vec3, radius: number): vec3 {
        const push = vec3.create();
        for (const t of this.triangles) {
            const at = add(center, push);
            const [q] = closestPointTriangle(at, t);
            const gap = sub(at, q);
            const distance = vec3.length(gap);
            if (distance < radius && distance > 1e-6) vec3.scaleAndAdd(push, push, gap, (radius - distance) / distance);
        }
        return push;
    }
}

/** Distance along a ray (unit `d`) to a triangle (Möller–Trumbore), either side. */
function rayTriangle(o: vec3, d: vec3, t: Triangle): number | undefined {
    const e1 = sub(t[1], t[0]);
    const e2 = sub(t[2], t[0]);
    const p = vec3.cross(vec3.create(), d, e2);
    const det = vec3.dot(e1, p);
    if (Math.abs(det) < 1e-12) return undefined;
    const inv = 1 / det;
    const s = sub(o, t[0]);
    const u = vec3.dot(s, p) * inv;
    if (!(u >= 0 && u <= 1)) return undefined;
    const q = vec3.cross(vec3.create(), s, e1);
    const v = vec3.dot(d, q) * inv;
    if (v < 0 || u + v > 1) return undefined;
    const dist = vec3.dot(e2, q) * inv;
    return dist >= 0 ? dist : undefined;
}

/** The point of a triangle nearest `p`, and whether it is inside the face (not on an edge). */
function closestPointTriangle(p: vec3, t: Triangle): [vec3, boolean] {
    const [a, b, c] = t;
    const ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    const d1 = vec3.dot(ab, ap), d2 = vec3.dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return [vec3.clone(a), false];
    const bp = sub(p, b);
    const d3 = vec3.dot(ab, bp), d4 = vec3.dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return [vec3.clone(b), false];
    const vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) return [madd(a, ab, d1 / (d1 - d3)), false];
    const cp = sub(p, c);
    const d5 = vec3.dot(ab, cp), d6 = vec3.dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return [vec3.clone(c), false];
    const vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) return [madd(a, ac, d2 / (d2 - d6)), false];
    const va = d3 * d6 - d5 * d4;
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) return [madd(b, sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))), false];
    const denom = 1 / (va + vb + vc);
    return [madd(madd(a, ab, vb * denom), ac, vc * denom), true];
}

/** A collider's shape. */
type Shape = Obb | TriangleMesh;

/** A static shape on some layers (a bit mask queries filter on). */
interface Collider {
    shape: Shape;
    layers: number;
}

/** The first thing a cast touched. */
interface Hit {
    distance: number;
    /** Where the ray is, or the sphere's centre, at contact. */
    point: vec3;
    /** The surface normal at the contact, toward the caster. */
    normal: vec3;
    collider: number;
}

/** Every layer, for queries. */
const ALL_LAYERS = 0xffffffff;

/** Static colliders and the queries on them. */
class CollisionWorld {
    private items: Collider[] = [];

    /** Add a collider on `layers`; returns its index. */
    public add(shape: Shape, layers: number = 1): number {
        this.items.push({ shape, layers });
        return this.items.length - 1;
    }

    public addBox(obb: Obb): number {
        return this.add(obb, 1);
    }

    public get colliders(): readonly Collider[] {
        return this.items;
    }

    private nearest(layers: number, cast: (s: Shape) => CastResult | undefined): Hit | undefined {
        let best: Hit | undefined;
        this.items.forEach((c, i) => {
            if ((c.layers & layers) === 0) return;
            const hit = cast(c.shape);
            if (hit && (best === undefined || hit[0] < best.distance)) best = { distance: hit[0], point: vec3.create(), normal: hit[1], collider: i };
        });
        return best;
    }

    /** The first surface a ray (unit `direction`) meets within `max`. */
    public raycast(origin: vec3, direction: vec3, max: number, layers: number = ALL_LAYERS): Hit | undefined {
        const hit = this.nearest(layers, (s) => s.raycast(origin, direction, max));
        if (hit) hit.point = madd(origin, direction, hit.distance);
        return hit;
    }

    /** The first contact of a sphere swept along unit `direction` within `max`. */
    public sphereCast(origin: vec3, radius: number, direction: vec3, max: number, layers: number = ALL_LAYERS): Hit | undefined {
        const hit = this.nearest(layers, (s) => s.sphereCast(origin, radius, direction, max));
        if (hit) hit.point = madd(origin, direction, hit.distance);
        return hit;
    }

    /** Whether a sphere touches anything. */
    public overlapSphere(center: vec3, radius: number, layers: number = ALL_LAYERS): boolean {
        const r2 = radius * radius;
        return this.items.some((c) => (c.layers & layers) !== 0 && (c.shape instanceof Obb
            ? c.shape.distanceSquared(center) < r2
            : c.shape.triangles.some((t) => vec3.squaredDistance(closestPointTriangle(center, t)[0], center) < r2)));
    }

    /**
     * Whether a capsule (segment `a`-`b`, `radius`) touches anything: spheres along it, spaced
     * half a radius apart.
     */
    public overlapCapsule(a: vec3, b: vec3, radius: number, layers: number = ALL_LAYERS): boolean {
        const steps = Math.max(Math.ceil(vec3.distance(a, b) / (radius * 0.5)), 1);
        const center = vec3.create();
        for (let k = 0; k <= steps; k++) {
            if (this.overlapSphere(vec3.lerp(center, a, b, k / steps), radius, layers)) return true;
        }
        return false;
    }

    /**
     * Where a vertical capsule standing at `feet` (`height` tall, `radius` wide) must move to
     * stop overlapping the world, sideways only; its lowest `step` metres are free (steps and
     * curbs don't block). A few iterations of pushing its spheres out.
     */
    public resolveCapsule(feet: vec3, height: number, radius: number, step: number, layers: number = ALL_LAYERS): vec3 {
        const position = vec3.clone(feet);
        const bottom = step + radius;
        const top = Math.max(height - radius, bottom);
        const steps = Math.max(Math.ceil((top - bottom) / (radius * 0.5)), 1);
        const center = vec3.create(), p = vec3.create();
        for (let iteration = 0; iteration < 4; iteration++) {
            let pushX = 0, pushZ = 0, push2 = 0;
            for (let k = 0; k <= steps; k++) {
                vec3.set(center, position[0], position[1] + bottom + (top - bottom) * k / steps, position[2]);
                for (const c of this.items) {
                    if ((c.layers & layers) === 0) continue;
                    c.shape.penetrationInto(center, radius, p);
                    const l2 = p[0] * p[0] + p[2] * p[2];
                    if (l2 > push2) {
                        pushX = p[0];
                        pushZ = p[2];
                        push2 = l2;
                    }
                }
            }
            if (push2 < 1e-10) break;
            position[0] += pushX;
            position[2] += pushZ;
        }
        return position;
    }

    /**
     * The height of the first surface below `position`, looking from `above` metres over it down
     * to `below` metres under it.
     */
    public groundHeight(position: vec3, above: number, below: number, layers: number = ALL_LAYERS): number | undefined {
        const hit = this.raycast(v3(position[0], position[1] + above, position[2]), v3(0, -1, 0), above + below, layers);
        return hit?.point[1];
    }
}

export { Obb, TriangleMesh, CollisionWorld, ALL_LAYERS, rayCapsule, raySphere, rayTriangle, closestPointTriangle };
export type { Triangle, Shape, Collider, Hit, CastResult };
