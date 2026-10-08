import { quat, vec3 } from "gl-matrix";
import { decaySpringDamperExact } from "./Springs";
import { Pose } from "./Pose";
import { Skeleton } from "./Skeleton";
import { Transform } from "./Transform";

/**
 * Two-joint IK and foot locking. Rust: `animation::ik`.
 *
 * The foot lock follows Daniel Holden's "Inverse Kinematics & Foot Locking"
 * (<https://theorangeduck.com/page/inverse-kinematics-foot-locking>): while the animation says a
 * foot is planted, pin it where it touched down; release it when the contact ends or the
 * animated foot strays past a radius, and let the difference fade with a spring.
 */

/** A unit vector perpendicular to `v` (glam's `any_orthonormal_vector`). */
function anyOrthonormal(v: vec3): vec3 {
    const sign = v[2] < 0 || Object.is(v[2], -0) ? -1 : 1;
    const a = -1 / (sign + v[2]);
    const b = v[0] * v[1] * a;
    return vec3.fromValues(b, sign + v[1] * v[1] * a, -v[1]);
}

/**
 * Bend a three-joint chain (`upper` → `middle` → `end`, e.g. thigh, knee, foot) so `end` reaches
 * `target` (model space), keeping the chain's plane when it can and `end`'s model rotation.
 * Rewrites `pose`'s local rotations of `upper` and `middle`; `model` must be `pose`'s model
 * transforms and is updated for the three joints.
 */
function twoJointIK(skeleton: Skeleton, pose: Pose, model: Transform[], upper: number, middle: number, end: number, target: vec3): void {
    const a = model[upper].translation, b = model[middle].translation, c = model[end].translation;
    const endRotation = quat.clone(model[end].rotation);
    const eps = 1e-5;
    const lab = vec3.distance(b, a);
    const lcb = vec3.distance(c, b);
    const lat = Math.min(Math.max(vec3.distance(target, a), eps), (lab + lcb) * 0.9999);
    if (lab < eps || lcb < eps) return;
    const sub = (x: vec3, y: vec3) => vec3.subtract(vec3.create(), x, y);
    const unit = (x: vec3) => {
        const l = vec3.length(x);
        return l > 0 ? vec3.scale(vec3.create(), x, 1 / l) : vec3.create();
    };
    const angle = (x: vec3, y: vec3) => Math.acos(Math.min(Math.max(vec3.dot(unit(x), unit(y)), -1), 1));
    const ca = sub(c, a), ba = sub(b, a), ta = sub(target, a);
    // the current angles at the upper and middle joints, and the ones that reach `target`
    const acAb0 = angle(ca, ba);
    const baBc0 = angle(sub(a, b), sub(c, b));
    const acAt0 = angle(ca, ta);
    const acAb1 = Math.acos(Math.min(Math.max((lcb * lcb - lab * lab - lat * lat) / (-2 * lab * lat), -1), 1));
    const baBc1 = Math.acos(Math.min(Math.max((lat * lat - lab * lab - lcb * lcb) / (-2 * lab * lcb), -1), 1));
    // bend about the chain's normal; a straight chain bends about any perpendicular
    let bendAxis = vec3.cross(vec3.create(), ca, ba);
    if (vec3.squaredLength(bendAxis) < 1e-10) bendAxis = anyOrthonormal(ca);
    vec3.normalize(bendAxis, bendAxis);
    const swingAxis = vec3.cross(vec3.create(), ca, ta);
    const inverseUpper = quat.conjugate(quat.create(), model[upper].rotation);
    const inverseMiddle = quat.conjugate(quat.create(), model[middle].rotation);
    const axisIn = (q: quat, axis: vec3) => vec3.transformQuat(vec3.create(), axis, q);
    const r0 = quat.setAxisAngle(quat.create(), axisIn(inverseUpper, bendAxis), acAb1 - acAb0);
    const r1 = quat.setAxisAngle(quat.create(), axisIn(inverseMiddle, bendAxis), baBc1 - baBc0);
    const r2 = vec3.squaredLength(swingAxis) > 1e-10
        ? quat.setAxisAngle(quat.create(), axisIn(inverseUpper, vec3.normalize(swingAxis, swingAxis)), acAt0)
        : quat.create();
    // bend first, then swing onto the target: in the joint's frame the swing comes last
    const upperRotation = pose.local[upper].rotation;
    quat.multiply(upperRotation, upperRotation, quat.multiply(quat.create(), r2, r0));
    quat.normalize(upperRotation, upperRotation);
    const middleRotation = pose.local[middle].rotation;
    quat.multiply(middleRotation, middleRotation, r1);
    quat.normalize(middleRotation, middleRotation);
    // the chain's model transforms again, with the end keeping its model rotation
    const parent = (j: number): Transform => {
        const p = skeleton.parents[j];
        return p === null ? Transform.identity() : model[p];
    };
    model[upper] = parent(upper).mul(pose.local[upper]);
    model[middle] = parent(middle).mul(pose.local[middle]);
    const endParent = parent(end);
    const endLocal = pose.local[end].rotation;
    quat.multiply(endLocal, quat.conjugate(quat.create(), endParent.rotation), endRotation);
    quat.normalize(endLocal, endLocal);
    model[end] = endParent.mul(pose.local[end]);
}

/** A foot pinned to where it touched down while it is planted. */
class FootLock {
    private locked = false;
    /** Where the foot is pinned while locked. */
    private position = vec3.create();
    /** Last frame's contact, to lock on a new one only. */
    private contact = false;
    /** Added to the output, fading: it keeps the output continuous when the lock lets go. */
    private offset = vec3.create();
    private offsetVelocity = vec3.create();

    /**
     * The foot's target this frame (world space) from where the animation puts it and whether
     * it is planted. A new contact pins the foot there; the pin lets go when the contact ends or
     * the animated foot strays past `unlockRadius`, and the output then fades back onto the
     * animation with `halflife`.
     */
    public update(animated: vec3, contact: boolean, unlockRadius: number, halflife: number, dt: number): vec3 {
        if (this.locked && (!contact || vec3.distance(animated, this.position) > unlockRadius)) {
            this.locked = false;
            // the output was the pin: carry the difference over, to fade
            vec3.add(this.offset, this.offset, vec3.subtract(vec3.create(), this.position, animated));
        }
        if (!this.locked && contact && !this.contact) {
            this.locked = true;
            vec3.copy(this.position, animated);
        }
        this.contact = contact;
        decaySpringDamperExact(this.offset, this.offsetVelocity, halflife, dt);
        return vec3.add(vec3.create(), this.locked ? this.position : animated, this.offset);
    }

    public isLocked(): boolean {
        return this.locked;
    }

    /** Release the foot and forget its history (after a teleport). */
    public reset(): void {
        this.locked = false;
        this.contact = false;
        vec3.zero(this.position);
        vec3.zero(this.offset);
        vec3.zero(this.offsetVelocity);
    }
}

export { twoJointIK, FootLock, anyOrthonormal };
