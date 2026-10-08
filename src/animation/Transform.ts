import { mat4, quat, vec3 } from "gl-matrix";

/**
 * A joint transform: scale, then rotation, then translation (glTF's TRS). Rust:
 * `animation::Transform`.
 *
 * Composition (`mul`) is exact for uniform scales, which skeletons use; a non-uniform scale
 * under a rotated child would shear, which a TRS cannot hold. Rotations stay quaternions end to
 * end (gl-matrix `quat`, x y z w).
 */
class Transform {
    constructor(
        public translation: vec3 = vec3.create(),
        public rotation: quat = quat.create(),
        public scale: vec3 = vec3.fromValues(1, 1, 1),
    ) { }

    /** The identity: no translation or rotation, unit scale. */
    public static identity(): Transform {
        return new Transform();
    }

    public static fromTranslationRotation(translation: vec3, rotation: quat): Transform {
        return new Transform(vec3.clone(translation), quat.clone(rotation));
    }

    /** The transform of a column-major matrix without shear. */
    public static fromMat4(m: mat4): Transform {
        const translation = mat4.getTranslation(vec3.create(), m);
        const scale = mat4.getScaling(vec3.create(), m);
        const rotation = quat.normalize(quat.create(), mat4.getRotation(quat.create(), m));
        return new Transform(translation, rotation, scale);
    }

    public clone(): Transform {
        return new Transform(vec3.clone(this.translation), quat.clone(this.rotation), vec3.clone(this.scale));
    }

    /** The column-major matrix of this transform, into `out` (a new one by default). */
    public toMat4(out: mat4 = mat4.create()): mat4 {
        return mat4.fromRotationTranslationScale(out, this.rotation, this.translation, this.scale);
    }

    /** `this` applied after `child`: a child's local transform into its parent's space. */
    public mul(child: Transform): Transform {
        const rotation = quat.multiply(quat.create(), this.rotation, child.rotation);
        return new Transform(
            this.transformPoint(child.translation),
            quat.normalize(rotation, rotation),
            vec3.multiply(vec3.create(), this.scale, child.scale),
        );
    }

    public inverse(): Transform {
        const rotation = quat.conjugate(quat.create(), this.rotation);
        const scale = vec3.inverse(vec3.create(), this.scale);
        const translation = vec3.transformQuat(vec3.create(), this.translation, rotation);
        vec3.multiply(translation, scale, translation);
        vec3.negate(translation, translation);
        return new Transform(translation, rotation, scale);
    }

    public transformPoint(p: vec3, out: vec3 = vec3.create()): vec3 {
        this.transformVector(p, out);
        return vec3.add(out, out, this.translation);
    }

    public transformVector(v: vec3, out: vec3 = vec3.create()): vec3 {
        vec3.multiply(out, this.scale, v);
        return vec3.transformQuat(out, out, this.rotation);
    }

    /**
     * Linear interpolation of translation and scale, normalized lerp of rotation along the
     * shorter arc.
     */
    public lerp(other: Transform, t: number): Transform {
        return new Transform(
            vec3.lerp(vec3.create(), this.translation, other.translation, t),
            nlerp(this.rotation, other.rotation, t),
            vec3.lerp(vec3.create(), this.scale, other.scale, t),
        );
    }

    /** Whether every component is within `eps` of `other`'s. */
    public equals(other: Transform, eps: number = 0): boolean {
        const close = (a: ArrayLike<number>, b: ArrayLike<number>) => {
            for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) > eps) return false;
            return true;
        };
        return close(this.translation, other.translation) && close(this.rotation, other.rotation) && close(this.scale, other.scale);
    }
}

/**
 * Normalized lerp between two rotations along the shorter arc, into `out`. Close to slerp for
 * the small steps between animation frames, and cheaper.
 */
function nlerp(a: quat, b: quat, t: number, out: quat = quat.create()): quat {
    const sign = quat.dot(a, b) < 0 ? -1 : 1;
    for (let i = 0; i < 4; i++) out[i] = a[i] * (1 - t) + sign * b[i] * t;
    return quat.normalize(out, out);
}

/**
 * `q` or `-q`, whichever has a non-negative w: the same rotation, on the hemisphere where
 * `quatLog` gives the shorter rotation vector.
 */
function quatAbs(q: quat): quat {
    return q[3] < 0 ? quat.fromValues(-q[0], -q[1], -q[2], -q[3]) : quat.clone(q);
}

/** The rotation vector (axis times half angle) of a unit quaternion: the inverse of `quatExp`. */
function quatLog(q: quat): vec3 {
    const v = vec3.fromValues(q[0], q[1], q[2]);
    const length = vec3.length(v);
    if (length < 1e-8) return v;
    const halfAngle = Math.atan2(length, Math.min(Math.max(q[3], -1), 1));
    return vec3.scale(v, v, halfAngle / length);
}

/** The unit quaternion of a rotation vector (axis times half angle). */
function quatExp(v: vec3): quat {
    const halfAngle = vec3.length(v);
    if (halfAngle < 1e-8) {
        const q = quat.fromValues(v[0], v[1], v[2], 1);
        return quat.normalize(q, q);
    }
    const s = Math.sin(halfAngle) / halfAngle;
    return quat.fromValues(v[0] * s, v[1] * s, v[2] * s, Math.cos(halfAngle));
}

/** Axis times angle of a rotation, along the shorter arc. */
function quatToScaledAngleAxis(q: quat): vec3 {
    const v = quatLog(quatAbs(q));
    return vec3.scale(v, v, 2);
}

/** The rotation of axis times angle `v`. */
function quatFromScaledAngleAxis(v: vec3): quat {
    return quatExp(vec3.scale(vec3.create(), v, 0.5));
}

/** Angular velocity (radians per second, world axes) that turns `from` into `to` in `dt`. */
function angularVelocity(from: quat, to: quat, dt: number): vec3 {
    const delta = quat.multiply(quat.create(), to, quat.conjugate(quat.create(), from));
    const v = quatToScaledAngleAxis(delta);
    return vec3.scale(v, v, 1 / dt);
}

export { Transform, nlerp, quatAbs, quatLog, quatExp, quatToScaledAngleAxis, quatFromScaledAngleAxis, angularVelocity };
