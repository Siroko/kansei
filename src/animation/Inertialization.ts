import { quat, vec3 } from "gl-matrix";
import { decaySpringDamperExact, decaySpringDamperExactQuat } from "./Springs";
import { Pose } from "./Pose";
import { Transform, angularVelocity, quatAbs } from "./Transform";

/**
 * Inertialization: switch animations instantly, and hide the jump by adding the difference
 * between the old and the new pose as an offset that decays to nothing with a spring. Rust:
 * `animation::inertialization`.
 *
 * After David Bollo, "Inertialization: High-Performance Animation Transitions in Gears of War"
 * (GDC 2018), in the spring form of Daniel Holden's "Spring-It-On" and his MIT-licensed
 * Motion-Matching code. Only the new animation is evaluated each frame.
 */

/** Per-joint offsets (and their velocities) between what was showing and what now plays. */
class Inertializer {
    private position: vec3[] = [];
    private velocity: vec3[] = [];
    private rotation: quat[] = [];
    private angular: vec3[] = [];

    constructor(private joints: number) {
        this.reset();
    }

    /**
     * Switch from the source animation (its local pose and joint velocities now) to the
     * destination: the offsets become what is showing (source plus the offsets still decaying)
     * minus the destination, so the output does not jump.
     */
    public transition(source: Pose, sourceLinear: vec3[], sourceAngular: vec3[], destination: Pose, destinationLinear: vec3[], destinationAngular: vec3[]): void {
        const conjugate = quat.create();
        for (let j = 0; j < this.joints; j++) {
            const s = source.local[j], d = destination.local[j];
            const p = this.position[j], v = this.velocity[j], w = this.angular[j];
            for (let i = 0; i < 3; i++) {
                p[i] = (s.translation[i] + p[i]) - d.translation[i];
                v[i] = (sourceLinear[j][i] + v[i]) - destinationLinear[j][i];
                w[i] = (sourceAngular[j][i] + w[i]) - destinationAngular[j][i];
            }
            const r = this.rotation[j];
            quat.multiply(r, r, s.rotation);
            quat.multiply(r, r, quat.conjugate(conjugate, d.rotation));
            quat.copy(r, quatAbs(r));
        }
    }

    /**
     * Decay the offsets by `dt` (half gone every `halflife` seconds) and apply them to the pose
     * that plays, `pose`, in place.
     */
    public update(pose: Pose, halflife: number, dt: number): void {
        for (let j = 0; j < this.joints; j++) {
            decaySpringDamperExact(this.position[j], this.velocity[j], halflife, dt);
            decaySpringDamperExactQuat(this.rotation[j], this.angular[j], halflife, dt);
            const t = pose.local[j];
            vec3.add(t.translation, t.translation, this.position[j]);
            quat.multiply(t.rotation, this.rotation[j], t.rotation);
            quat.normalize(t.rotation, t.rotation);
        }
    }

    /** Forget the offsets (after a teleport). */
    public reset(): void {
        const n = this.joints;
        this.position = Array.from({ length: n }, () => vec3.create());
        this.velocity = Array.from({ length: n }, () => vec3.create());
        this.rotation = Array.from({ length: n }, () => quat.create());
        this.angular = Array.from({ length: n }, () => vec3.create());
    }

    /** The largest rotation offset left, in radians (for debugging and tests). */
    public largestAngle(): number {
        return this.rotation.reduce((m, q) => Math.max(m, 2 * Math.acos(Math.min(Math.max(quatAbs(q)[3], -1), 1))), 0);
    }
}

/** Joint velocities between two poses `dt` apart: linear and angular per joint. */
function poseVelocities(from: Transform[], to: Transform[], dt: number): { linear: vec3[], angular: vec3[] } {
    const linear: vec3[] = [];
    const angular: vec3[] = [];
    from.forEach((a, i) => {
        const b = to[i];
        linear.push(vec3.scale(vec3.create(), vec3.subtract(vec3.create(), b.translation, a.translation), 1 / dt));
        angular.push(angularVelocity(a.rotation, b.rotation, dt));
    });
    return { linear, angular };
}

export { Inertializer, poseVelocities };
