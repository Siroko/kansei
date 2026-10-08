import { quat, vec3 } from "gl-matrix";
import { Pose } from "./Pose";
import { Skeleton } from "./Skeleton";
import { Transform, nlerp } from "./Transform";

/**
 * An animation sampled at a fixed rate: every joint's local transform at every frame. Rust:
 * `animation::Clip`.
 *
 * Frame `i` is at time `i / sampleRate`; the clip lasts `(frames - 1) / sampleRate`, so a
 * looping clip's last frame is the pose it wraps back to (its first), as exported by DCC tools.
 */
class Clip {
    /** Sampling past the end wraps around instead of holding the last frame. */
    public looping: boolean = false;

    private constructor(
        public name: string,
        /** Frames per second. */
        public sampleRate: number,
        private joints: number,
        /** Frame-major (`frame * joints + joint`), 4 floats each (x y z w). */
        private rotations: Float32Array,
        /** 3 floats each. */
        private translations: Float32Array,
        /** 3 floats each; `null` when every joint keeps unit scale. */
        private scales: Float32Array | null,
    ) { }

    /** A clip from its frames' local poses (each with the same joint count). */
    public static fromPoses(name: string, sampleRate: number, poses: Pose[]): Clip {
        if (poses.length === 0) throw new Error('Clip: a clip needs at least one frame');
        if (!(sampleRate > 0)) throw new Error('Clip: the sample rate must be positive');
        const joints = poses[0].length;
        const count = poses.length * joints;
        const rotations = new Float32Array(count * 4);
        const translations = new Float32Array(count * 3);
        const scales = new Float32Array(count * 3);
        let unitScale = true;
        let k = 0;
        for (const pose of poses) {
            if (pose.length !== joints) throw new Error('Clip: every frame has the same joints');
            for (const t of pose.local) {
                rotations.set(t.rotation, k * 4);
                translations.set(t.translation, k * 3);
                scales.set(t.scale, k * 3);
                for (let i = 0; i < 3; i++) if (Math.abs(t.scale[i] - 1) > 1e-5) unitScale = false;
                k++;
            }
        }
        return new Clip(name, sampleRate, joints, rotations, translations, unitScale ? null : scales);
    }

    public get frameCount(): number {
        return this.rotations.length / 4 / Math.max(this.joints, 1);
    }

    public get jointCount(): number {
        return this.joints;
    }

    /** Seconds from the first frame to the last. */
    public get duration(): number {
        return (this.frameCount - 1) / this.sampleRate;
    }

    /** Joint `joint`'s local transform at frame `frame`. */
    public transform(frame: number, joint: number): Transform {
        const k = frame * this.joints + joint;
        return new Transform(
            vec3.fromValues(this.translations[3 * k], this.translations[3 * k + 1], this.translations[3 * k + 2]),
            quat.fromValues(this.rotations[4 * k], this.rotations[4 * k + 1], this.rotations[4 * k + 2], this.rotations[4 * k + 3]),
            this.scales ? vec3.fromValues(this.scales[3 * k], this.scales[3 * k + 1], this.scales[3 * k + 2]) : vec3.fromValues(1, 1, 1),
        );
    }

    /** Frame `frame`'s pose into `out`. */
    public framePose(frame: number, out: Pose): Pose {
        out.local = [];
        for (let j = 0; j < this.joints; j++) out.local.push(this.transform(frame, j));
        return out;
    }

    /**
     * The frame pair and blend weight at `time` seconds: clamped to the clip, or wrapped if it
     * loops.
     */
    public framesAt(time: number): [number, number, number] {
        const last = this.frameCount - 1;
        if (last === 0) return [0, 0, 0];
        let f = time * this.sampleRate;
        if (this.looping) f = ((f % last) + last) % last;
        f = Math.min(Math.max(f, 0), last);
        const a = Math.min(Math.floor(f), last);
        const b = Math.min(a + 1, last);
        return [a, b, f - a];
    }

    /**
     * The pose at `time` seconds, blending the two nearest frames, into `out` (its transforms
     * are reused when it already has one per joint).
     */
    public sample(time: number, out: Pose): Pose {
        const [a, b, t] = this.framesAt(time);
        if (out.local.length !== this.joints) {
            out.local = Array.from({ length: this.joints }, () => Transform.identity());
        }
        const ra = quat.create(), rb = quat.create();
        for (let j = 0; j < this.joints; j++) {
            const ka = a * this.joints + j, kb = b * this.joints + j;
            const o = out.local[j];
            for (let i = 0; i < 3; i++) {
                const ta = this.translations[3 * ka + i];
                o.translation[i] = ta + (this.translations[3 * kb + i] - ta) * t;
                if (this.scales) {
                    const sa = this.scales[3 * ka + i];
                    o.scale[i] = sa + (this.scales[3 * kb + i] - sa) * t;
                } else {
                    o.scale[i] = 1;
                }
            }
            for (let i = 0; i < 4; i++) {
                ra[i] = this.rotations[4 * ka + i];
                rb[i] = this.rotations[4 * kb + i];
            }
            nlerp(ra, rb, t, o.rotation);
        }
        return out;
    }

    /**
     * This clip, authored on `from`, for skeleton `to`: joints matched by name, and `to`'s rest
     * transform for the joints `from` lacks.
     */
    public retargetByName(from: Skeleton, to: Skeleton): Clip {
        if (from.length !== this.joints) throw new Error('Clip.retargetByName: `from` is not the clip\'s skeleton');
        const source = from.mapNames(to);
        const poses: Pose[] = [];
        for (let f = 0; f < this.frameCount; f++) {
            poses.push(new Pose(source.map((s, j) => s === undefined ? to.rest[j].clone() : this.transform(f, s))));
        }
        const clip = Clip.fromPoses(this.name, this.sampleRate, poses);
        clip.looping = this.looping;
        return clip;
    }

    /** A joint's local transform at every frame: for a root joint, its motion in model space. */
    public track(joint: number): Transform[] {
        return Array.from({ length: this.frameCount }, (_, f) => this.transform(f, joint));
    }
}

export { Clip };
