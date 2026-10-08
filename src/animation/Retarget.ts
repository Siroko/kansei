import { quat, vec3 } from "gl-matrix";
import { anyOrthonormal } from "./IK";
import { Pose } from "./Pose";
import { Skeleton } from "./Skeleton";
import { Transform } from "./Transform";

/**
 * Retargeting a pose between skeletons that share joint names and joint axes but not
 * proportions (a stylised character rigged to an animation set's skeleton). Rust:
 * `animation::retarget`.
 *
 * Rotations carry over as they are (the axes agree). Translations, which hold the bone lengths,
 * are chosen per joint:
 * - `Animation`: the animation's own (the root, attachment and IK helper joints, whose
 *   positions mean something in the animation's space);
 * - `Skeleton`: the target's rest translation (fixed bone lengths);
 * - `OrientAndScale` (the default): the animation's translation, turned from the source rest
 *   direction onto the target's and scaled by the ratio of their lengths. A joint whose
 *   translation never moves gets the target's bone exactly; one that moves (the hips going up
 *   and down) moves in proportion to the target's size.
 */

/** How one joint's translation is retargeted. */
enum TranslationMode {
    Animation = 'animation',
    Skeleton = 'skeleton',
    OrientAndScale = 'orientAndScale',
}

interface RetargetJoint {
    source: number;
    mode: TranslationMode;
    /** OrientAndScale: from the source rest direction onto the target's, and the length ratio. */
    turn: quat;
    scale: number;
}

/** Whether `name` matches `pattern` (a trailing `*` matches any rest). */
function matches(pattern: string, name: string): boolean {
    return pattern.endsWith('*') ? name.startsWith(pattern.slice(0, -1)) : pattern === name;
}

/** The rotation taking unit `from` onto unit `to` along the shorter arc (glam's `from_rotation_arc`). */
function rotationArc(from: vec3, to: vec3): quat {
    const ONE_MINUS_EPS = 1 - 2 * 1.1920929e-7;
    const dot = vec3.dot(from, to);
    if (dot > ONE_MINUS_EPS) return quat.create();
    if (dot < -ONE_MINUS_EPS) {
        return quat.setAxisAngle(quat.create(), anyOrthonormal(from), Math.PI);
    }
    const c = vec3.cross(vec3.create(), from, to);
    const q = quat.fromValues(c[0], c[1], c[2], 1 + dot);
    return quat.normalize(q, q);
}

/** Maps poses of one skeleton onto another with the same joint names. */
class Retarget {
    /**
     * The joints Unreal-style skeletons keep in animation space: the root, `attach`, the IK
     * targets, prop and virtual bones.
     */
    public static readonly UNREAL_KEEP: readonly string[] = ["root", "attach", "ik_*", "props_root", "prop_*", "VB *"];

    /** Per target joint, its source joint (`undefined`: not in the source; keeps the target's rest). */
    private joints: (RetargetJoint | undefined)[];
    private targetRest: Transform[];

    /**
     * Retarget from `source` onto `target`, joints paired by name. Joints matching any of
     * `keepAnimation` (exact names, or prefixes ending in `*`) keep the animation's
     * translation; every other joint orients and scales it.
     */
    constructor(source: Skeleton, target: Skeleton, keepAnimation: readonly string[] = []) {
        this.joints = target.names.map((name, j) => {
            const s = source.find(name);
            if (s === undefined) return undefined;
            const from = source.rest[s].translation, to = target.rest[j].translation;
            const mode = keepAnimation.some((p) => matches(p, name))
                ? TranslationMode.Animation
                : vec3.length(from) < 1e-5 || vec3.length(to) < 1e-5 ? TranslationMode.Skeleton : TranslationMode.OrientAndScale;
            const orient = mode === TranslationMode.OrientAndScale;
            const turn = orient ? rotationArc(vec3.normalize(vec3.create(), from), vec3.normalize(vec3.create(), to)) : quat.create();
            const scale = orient ? vec3.length(to) / vec3.length(from) : 1;
            return { source: s, mode, turn, scale };
        });
        this.targetRest = target.rest.map((t) => t.clone());
    }

    /** The translation mode chosen for each target joint (`undefined`: not in the source). */
    public modes(): (TranslationMode | undefined)[] {
        return this.joints.map((j) => j?.mode);
    }

    /** `source` (a pose of the source skeleton) as a pose of the target, into `out`. */
    public apply(source: Pose, out: Pose): Pose {
        if (out.local.length !== this.joints.length) {
            out.local = this.targetRest.map(() => Transform.identity());
        }
        this.joints.forEach((joint, j) => {
            const rest = this.targetRest[j];
            const o = out.local[j];
            if (joint === undefined) {
                vec3.copy(o.translation, rest.translation);
                quat.copy(o.rotation, rest.rotation);
                vec3.copy(o.scale, rest.scale);
                return;
            }
            const s = source.local[joint.source];
            switch (joint.mode) {
                case TranslationMode.Animation:
                    vec3.copy(o.translation, s.translation);
                    break;
                case TranslationMode.Skeleton:
                    vec3.copy(o.translation, rest.translation);
                    break;
                case TranslationMode.OrientAndScale:
                    vec3.transformQuat(o.translation, s.translation, joint.turn);
                    vec3.scale(o.translation, o.translation, joint.scale);
                    break;
            }
            quat.copy(o.rotation, s.rotation);
            vec3.copy(o.scale, rest.scale);
        });
        return out;
    }
}

export { Retarget, TranslationMode, rotationArc };
