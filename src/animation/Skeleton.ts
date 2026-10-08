import { Transform } from "./Transform";

/**
 * A joint hierarchy: names, parents and the rest (bind) pose, in local space. Rust:
 * `animation::Skeleton`.
 *
 * Joints are ordered parents first (`parents[i] < i`), so a single forward pass computes model
 * space (`Pose.toModel`).
 */
class Skeleton {
    /**
     * A skeleton from joints listed parents first. Throws if a parent comes after its child.
     *
     * @param names - Each joint's name.
     * @param parents - Parent of each joint, `null` for a root.
     * @param rest - Each joint's local transform at rest.
     */
    constructor(
        public readonly names: string[],
        public readonly parents: (number | null)[],
        public readonly rest: Transform[],
    ) {
        if (names.length !== parents.length || names.length !== rest.length) {
            throw new Error(`Skeleton: ${names.length} names, ${parents.length} parents and ${rest.length} rest transforms`);
        }
        parents.forEach((p, i) => {
            if (p !== null && p >= i) throw new Error(`Skeleton: joint ${i} (${names[i]}) comes before its parent ${p}`);
        });
    }

    public get length(): number {
        return this.names.length;
    }

    public isEmpty(): boolean {
        return this.names.length === 0;
    }

    /** Index of the joint called `name`, or `undefined`. */
    public find(name: string): number | undefined {
        const i = this.names.indexOf(name);
        return i < 0 ? undefined : i;
    }

    /** The rest pose in model space. */
    public restModel(): Transform[] {
        const model: Transform[] = [];
        this.rest.forEach((local, i) => {
            const p = this.parents[i];
            model.push(p === null ? local.clone() : model[p].mul(local));
        });
        return model;
    }

    /** Whether `joint` is `ancestor` or below it. */
    public isDescendant(joint: number, ancestor: number): boolean {
        let j: number | null = joint;
        while (j !== null) {
            if (j === ancestor) return true;
            j = this.parents[j];
        }
        return false;
    }

    /** For each of `other`'s joints, the index of the joint of this skeleton with its name. */
    public mapNames(other: Skeleton): (number | undefined)[] {
        return other.names.map((n) => this.find(n));
    }
}

export { Skeleton };
