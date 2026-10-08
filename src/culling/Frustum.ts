import { Matrix4 } from '../math/Matrix4';

type Vec3 = [number, number, number];
/** A plane `(nx, ny, nz, d)`: `n · p + d >= 0` on its inner side, `n` of unit length. */
type Plane = [number, number, number, number];

/**
 * The six planes of the frustum of `viewProj` (projection times view, column-major, depth in
 * `[0, 1]`), facing in: left, right, bottom, top, near, far (Gribb and Hartmann).
 * Port of the Rust engine's `culling::frustum_planes`.
 */
function frustumPlanes(viewProj: Matrix4 | ArrayLike<number>): Plane[] {
    const m = viewProj instanceof Matrix4 ? (viewProj.internalMat4 as ArrayLike<number>) : viewProj;
    const row = (i: number): Plane => [m[i], m[4 + i], m[8 + i], m[12 + i]];
    const [r0, r1, r2, r3] = [row(0), row(1), row(2), row(3)];
    const add = (a: Plane, b: Plane, s: number): Plane => [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2], a[3] + s * b[3]];
    return [add(r3, r0, 1), add(r3, r0, -1), add(r3, r1, 1), add(r3, r1, -1), r2, add(r3, r2, -1)].map((p) => {
        const l = Math.max(Math.hypot(p[0], p[1], p[2]), 1e-12);
        return [p[0] / l, p[1] / l, p[2] / l, p[3] / l];
    });
}

/**
 * Whether the box `[min, max]` is at least partly inside the frustum `planes` (`frustumPlanes`):
 * false only when it lies wholly outside one plane, so a box near a corner of the frustum may be
 * kept though nothing of it shows (conservative, as culling must be).
 * Port of the Rust engine's `culling::aabb_in_frustum`.
 */
function aabbInFrustum(planes: Plane[], min: Vec3, max: Vec3): boolean {
    return planes.every((p) => {
        // the box's corner furthest along the plane's normal
        const x = p[0] >= 0 ? max[0] : min[0];
        const y = p[1] >= 0 ? max[1] : min[1];
        const z = p[2] >= 0 ? max[2] : min[2];
        return p[0] * x + p[1] * y + p[2] * z + p[3] >= 0;
    });
}

export { frustumPlanes, aabbInFrustum };
export type { Plane };
