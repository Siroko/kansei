import { quat, vec3 } from "gl-matrix";

/**
 * Critically damped springs, in closed form (exact for any time step). Rust:
 * `animation::springs`.
 *
 * After Daniel Holden's "Spring-It-On: The Game Developer's Spring-Roll-Call"
 * (<https://theorangeduck.com/page/spring-roll-call>) and his MIT-licensed Motion-Matching
 * reference code (<https://github.com/orangeduck/Motion-Matching>). A spring's stiffness is given
 * as a half-life: the time to cover half the distance to its goal.
 *
 * The springs advance their state in place: `x`, `v` (and `a`, `q`, `w`) are rewritten.
 */

const LN2 = Math.LN2;

/**
 * `exp(-x)`. (Holden's rational approximation is close for a frame's step but several times
 * too large a second ahead, where trajectory prediction evaluates the springs.)
 */
function negexp(x: number): number {
    return Math.exp(-x);
}

/** Damping of a critically damped spring with this half-life. */
function halflifeToDamping(halflife: number): number {
    return (4 * LN2) / (halflife + 1e-5);
}

/** `x` moved toward `goal`, covering half the distance every `halflife` seconds (no velocity), into `out`. */
function damperExact(x: vec3, goal: vec3, halflife: number, dt: number, out: vec3 = vec3.create()): vec3 {
    return vec3.lerp(out, x, goal, 1 - negexp((LN2 * dt) / (halflife + 1e-5)));
}

/** A critically damped spring from `x` (velocity `v`) toward `goal`, advanced by `dt`. */
function springDamperExact(x: vec3, v: vec3, goal: vec3, halflife: number, dt: number): void {
    const y = halflifeToDamping(halflife) / 2;
    const eydt = negexp(y * dt);
    for (let i = 0; i < 3; i++) {
        const j0 = x[i] - goal[i];
        const j1 = v[i] + j0 * y;
        x[i] = eydt * (j0 + j1 * dt) + goal[i];
        v[i] = eydt * (v[i] - j1 * y * dt);
    }
}

/** `springDamperExact` toward zero: an offset (and its velocity) fading away. */
function decaySpringDamperExact(x: vec3, v: vec3, halflife: number, dt: number): void {
    const y = halflifeToDamping(halflife) / 2;
    const eydt = negexp(y * dt);
    for (let i = 0; i < 3; i++) {
        const j1 = v[i] + x[i] * y;
        x[i] = eydt * (x[i] + j1 * dt);
        v[i] = eydt * (v[i] - j1 * y * dt);
    }
}

/**
 * `q` (unit, w on the non-negative side) as axis times angle, into `out`: `quatToScaledAngleAxis`
 * on scalars (the springs below run per joint per frame, so they allocate nothing).
 */
function scaledAngleAxis(x: number, y: number, z: number, w: number, out: vec3): vec3 {
    if (w < 0) {
        x = -x; y = -y; z = -z; w = -w;
    }
    const length = Math.sqrt(x * x + y * y + z * z);
    const k = length < 1e-8 ? 2 : 2 * Math.atan2(length, Math.min(Math.max(w, -1), 1)) / length;
    return vec3.set(out, x * k, y * k, z * k);
}

/** The rotation of axis times angle (`vx`, `vy`, `vz`), into `out`: `quatFromScaledAngleAxis` on scalars. */
function fromScaledAngleAxis(vx: number, vy: number, vz: number, out: quat): quat {
    const hx = vx * 0.5, hy = vy * 0.5, hz = vz * 0.5;
    const half = Math.sqrt(hx * hx + hy * hy + hz * hz);
    if (half < 1e-8) return quat.normalize(out, quat.set(out, hx, hy, hz, 1));
    const s = Math.sin(half) / half;
    return quat.set(out, hx * s, hy * s, hz * s, Math.cos(half));
}

const J0 = vec3.create();
const OFFSET = quat.create();
const TURN = quat.create();

/** `springDamperExact` for a rotation `q` with angular velocity `w` (radians per second). */
function springDamperExactQuat(q: quat, w: vec3, goal: quat, halflife: number, dt: number): void {
    const y = halflifeToDamping(halflife) / 2;
    // q times the goal's conjugate, as axis times angle along the shorter arc
    const gx = -goal[0], gy = -goal[1], gz = -goal[2], gw = goal[3];
    quat.set(OFFSET,
        q[3] * gx + q[0] * gw + q[1] * gz - q[2] * gy,
        q[3] * gy - q[0] * gz + q[1] * gw + q[2] * gx,
        q[3] * gz + q[0] * gy - q[1] * gx + q[2] * gw,
        q[3] * gw - q[0] * gx - q[1] * gy - q[2] * gz);
    const j0 = scaledAngleAxis(OFFSET[0], OFFSET[1], OFFSET[2], OFFSET[3], J0);
    const eydt = negexp(y * dt);
    let tx = 0, ty = 0, tz = 0;
    for (let i = 0; i < 3; i++) {
        const j1 = w[i] + j0[i] * y;
        const turn = eydt * (j0[i] + j1 * dt);
        if (i === 0) tx = turn; else if (i === 1) ty = turn; else tz = turn;
        w[i] = eydt * (w[i] - j1 * y * dt);
    }
    quat.multiply(q, fromScaledAngleAxis(tx, ty, tz, TURN), goal);
    quat.normalize(q, q);
}

/** `decaySpringDamperExact` for a rotation offset: `q` fades to the identity. */
function decaySpringDamperExactQuat(q: quat, w: vec3, halflife: number, dt: number): void {
    const y = halflifeToDamping(halflife) / 2;
    const j0 = scaledAngleAxis(q[0], q[1], q[2], q[3], J0);
    const eydt = negexp(y * dt);
    let tx = 0, ty = 0, tz = 0;
    for (let i = 0; i < 3; i++) {
        const j1 = w[i] + j0[i] * y;
        const turn = eydt * (j0[i] + j1 * dt);
        if (i === 0) tx = turn; else if (i === 1) ty = turn; else tz = turn;
        w[i] = eydt * (w[i] - j1 * y * dt);
    }
    quat.normalize(q, fromScaledAngleAxis(tx, ty, tz, q));
}

/**
 * A character's velocity as a critically damped spring toward `goalVelocity`, integrated into
 * its position: `x` (position), `v` (velocity), `a` (acceleration) after `dt`.
 */
function springCharacterUpdate(x: vec3, v: vec3, a: vec3, goalVelocity: vec3, halflife: number, dt: number): void {
    const y = halflifeToDamping(halflife) / 2;
    const eydt = negexp(y * dt);
    for (let i = 0; i < 3; i++) {
        const j0 = v[i] - goalVelocity[i];
        const j1 = a[i] + j0 * y;
        x[i] = eydt * ((-j1 / (y * y)) + ((-j0 - j1 * dt) / y)) + (j1 / (y * y)) + j0 / y + goalVelocity[i] * dt + x[i];
        v[i] = eydt * (j0 + j1 * dt) + goalVelocity[i];
        a[i] = eydt * (a[i] - j1 * y * dt);
    }
}

export {
    negexp, halflifeToDamping, damperExact, springDamperExact, decaySpringDamperExact,
    springDamperExactQuat, decaySpringDamperExactQuat, springCharacterUpdate,
};
