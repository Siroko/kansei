import { quat, vec3 } from "gl-matrix";

/**
 * Headings on the ground: a character's facing as a yaw about +Y. Rust: `motion_matching`'s
 * `FORWARD`, `yaw_of`, `yaw_rotation` and `wrap_angle` (in `database.rs`).
 */

/** The character's forward axis in its root's space (glTF: models face +Z). */
const FORWARD: Readonly<vec3> = vec3.fromValues(0, 0, 1);

/** The yaw of a rotation: the heading its forward axis points to, about +Y. */
function yawOf(rotation: quat): number {
    const f = vec3.transformQuat(vec3.create(), FORWARD, rotation);
    return Math.atan2(f[0], f[2]);
}

/** A rotation of `yaw` radians about +Y, into `out`. */
function yawRotation(yaw: number, out: quat = quat.create()): quat {
    return quat.set(out, 0, Math.sin(yaw * 0.5), 0, Math.cos(yaw * 0.5));
}

/** `x` modulo `m` (positive `m`), never negative: Rust's `rem_euclid`. */
function remEuclid(x: number, m: number): number {
    const r = x % m;
    return r < 0 ? r + m : r;
}

/** Wrap an angle to (-pi, pi]. */
function wrapAngle(a: number): number {
    const w = remEuclid(a + Math.PI, 2 * Math.PI) - Math.PI;
    return w <= -Math.PI ? w + 2 * Math.PI : w;
}

export { FORWARD, yawOf, yawRotation, wrapAngle, remEuclid };
