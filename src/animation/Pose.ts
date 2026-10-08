import { Skeleton } from "./Skeleton";
import { Transform } from "./Transform";

/**
 * A skeleton's joints in local space (relative to their parents), as sampled from a clip. Rust:
 * `animation::Pose`.
 */
class Pose {
    constructor(public local: Transform[] = []) { }

    /** The skeleton's rest pose. */
    public static rest(skeleton: Skeleton): Pose {
        return new Pose(skeleton.rest.map((t) => t.clone()));
    }

    public get length(): number {
        return this.local.length;
    }

    public isEmpty(): boolean {
        return this.local.length === 0;
    }

    /**
     * Forward kinematics: each joint's transform in model space (the skeleton's root space),
     * into `model` (resized to fit).
     */
    public toModel(skeleton: Skeleton, model: Transform[]): Transform[] {
        model.length = 0;
        this.local.forEach((local, i) => {
            const p = skeleton.parents[i];
            model.push(p === null ? local.clone() : model[p].mul(local));
        });
        return model;
    }

    /** `toModel` into a new array. */
    public model(skeleton: Skeleton): Transform[] {
        return this.toModel(skeleton, []);
    }

    /** Blend towards `other` by `t` (0: this, 1: other), joint by joint. */
    public blend(other: Pose, t: number): void {
        for (let i = 0; i < Math.min(this.local.length, other.local.length); i++) {
            this.local[i] = this.local[i].lerp(other.local[i], t);
        }
    }
}

export { Pose };
