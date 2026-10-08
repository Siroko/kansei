import { quat, vec3 } from "gl-matrix";
import { Clip } from "../Clip";
import { Pose } from "../Pose";
import { Skeleton } from "../Skeleton";
import { Transform, angularVelocity, nlerp } from "../Transform";
import { FORWARD, wrapAngle, yawOf, yawRotation } from "./Heading";

/** Features per frame (see `FeatureWeights` for the layout). */
const FEATURES = 27;
/** Floats per frame in the feature table: `FEATURES` padded to whole vec4s. */
const STRIDE = 28;
/** Frames per small and large bounding box of the search's acceleration structure. */
const BOUND_SMALL = 16;
const BOUND_LARGE = 64;
/** Seconds ahead of each trajectory sample. */
const TRAJECTORY_TIMES: readonly number[] = [1 / 3, 2 / 3, 1];

/**
 * The joints the database reads: the root (whose motion moves the character, and whose facing
 * the features are relative to), the hips and the feet (left, right).
 */
interface JointRoles {
    root: number;
    hips: number;
    feet: [number, number];
}

/** The joints with these names; the root must be a top joint (no parent). Throws otherwise. */
function findJointRoles(skeleton: Skeleton, root: string, hips: string, leftFoot: string, rightFoot: string): JointRoles {
    const find = (name: string) => {
        const j = skeleton.find(name);
        if (j === undefined) throw new Error(`the skeleton has no joint '${name}'`);
        return j;
    };
    const roles: JointRoles = { root: find(root), hips: find(hips), feet: [find(leftFoot), find(rightFoot)] };
    if (skeleton.parents[roles.root] !== null) throw new Error(`the root joint '${root}' has a parent: the character root must be a top joint`);
    return roles;
}

/**
 * Weights of the feature groups; the layout of a frame's features is:
 *
 * | floats | feature (in the root's frame, at the character's facing) |
 * |---|---|
 * | 0-5 | left and right foot positions |
 * | 6-11 | left and right foot velocities |
 * | 12-14 | hips velocity |
 * | 15-20 | the root's future positions (x, z) at `TRAJECTORY_TIMES` |
 * | 21-26 | the root's future facing directions (x, z) at `TRAJECTORY_TIMES` |
 *
 * Each group is normalized by its spread over the database, then scaled by its weight: a larger
 * weight makes that group matter more in the search.
 */
interface FeatureWeights {
    footPosition: number;
    footVelocity: number;
    hipsVelocity: number;
    trajectoryPosition: number;
    trajectoryDirection: number;
}

function defaultFeatureWeights(): FeatureWeights {
    return { footPosition: 0.75, footVelocity: 1, hipsVelocity: 1, trajectoryPosition: 1, trajectoryDirection: 1.5 };
}

/** (first float, floats, weight) of each group. */
function featureGroups(w: FeatureWeights): [number, number, number][] {
    return [[0, 6, w.footPosition], [6, 6, w.footVelocity], [12, 3, w.hipsVelocity], [15, 6, w.trajectoryPosition], [21, 6, w.trajectoryDirection]];
}

/**
 * When a foot is planted: its joint below `height` metres above the ground and slower than
 * `speed` metres per second, for at least `minFrames` frames in a row.
 */
interface ContactThresholds {
    height: number;
    speed: number;
    minFrames: number;
}

function defaultContactThresholds(): ContactThresholds {
    return { height: 0.15, speed: 0.25, minFrames: 3 };
}

/**
 * Tag bit (in `ClipInfo.tags`) of clips only played on command (traversals, falls, landings):
 * searches leave them out unless their filter asks for this bit.
 */
const ACTION_TAG = 0x80000000;

/** One clip's frames in the database. */
class ClipInfo {
    constructor(
        public name: string,
        /** First frame (a database index)... */
        public start: number,
        /** ...and frame count. */
        public frames: number,
        /** Plays around: the last frame is the first again, and play wraps from it to frame 1. */
        public looping: boolean,
        /** A bit mask the search can filter clips with (e.g. one bit per gait). */
        public tags: number,
    ) { }

    /** Frames the playhead can be on: a loop's last frame is its first. */
    public get playable(): number {
        return this.looping ? this.frames - 1 : this.frames;
    }

    public contains(frame: number): boolean {
        return frame >= this.start && frame < this.start + this.frames;
    }
}

/** Rust's `f32::round` (half away from zero). */
function roundAway(x: number): number {
    return Math.sign(x) * Math.round(Math.abs(x));
}

/**
 * A vec3 per joint per frame: stored once for joints that never change, else quantized to 16
 * bits per axis over the joint's range.
 */
class Vec3Tracks {
    constructor(
        public joints: number,
        /** Per joint: 1 when constant (its value in `constants`), 0 when animated. */
        public isConstant: Uint8Array,
        /** 3 floats per joint. */
        public constants: Float32Array,
        /** Per animated joint, its slot in each frame's row of `animated` (-1 for constant joints). */
        public slot: Int32Array,
        public animatedJoints: number,
        /** Per slot: the middle of the joint's range and half its extent (3 floats each). */
        public center: Float32Array,
        public extent: Float32Array,
        /** `(frame * animatedJoints + slot) * 3`: (value - center) / extent in signed 16 bits. */
        public animated: Int16Array,
    ) { }

    /**
     * Tracks of `values` (frame-major, `(frame * joints + joint) * 3`), keeping joints that vary
     * by more than `tolerance` animated.
     */
    public static build(values: Float32Array, frames: number, joints: number, tolerance: number): Vec3Tracks {
        const isConstant = new Uint8Array(joints);
        const constants = new Float32Array(joints * 3);
        const slot = new Int32Array(joints).fill(-1);
        const center: number[] = [];
        const extent: number[] = [];
        for (let j = 0; j < joints; j++) {
            const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
            for (let f = 0; f < frames; f++) {
                for (let i = 0; i < 3; i++) {
                    const x = values[(f * joints + j) * 3 + i];
                    lo[i] = Math.min(lo[i], x);
                    hi[i] = Math.max(hi[i], x);
                }
            }
            const spread = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
            if (frames === 0 || spread <= 2 * tolerance) {
                isConstant[j] = 1;
                for (let i = 0; i < 3; i++) constants[3 * j + i] = frames === 0 ? 0 : (lo[i] + hi[i]) * 0.5;
            } else {
                slot[j] = center.length / 3;
                for (let i = 0; i < 3; i++) {
                    center.push((lo[i] + hi[i]) * 0.5);
                    extent.push(Math.max((hi[i] - lo[i]) * 0.5, 1e-9));
                }
            }
        }
        const animatedJoints = center.length / 3;
        const centers = new Float32Array(center), extents = new Float32Array(extent);
        const animated = new Int16Array(frames * animatedJoints * 3);
        let k = 0;
        for (let f = 0; f < frames; f++) {
            for (let j = 0; j < joints; j++) {
                if (isConstant[j]) continue;
                const s = slot[j];
                for (let i = 0; i < 3; i++) {
                    const q = (values[(f * joints + j) * 3 + i] - centers[3 * s + i]) / extents[3 * s + i];
                    animated[k++] = roundAway(Math.min(Math.max(q, -1), 1) * 32767);
                }
            }
        }
        return new Vec3Tracks(joints, isConstant, constants, slot, animatedJoints, centers, extents, animated);
    }

    /** Joint `joint`'s value at `frame`, into `out` at `offset`. */
    public get(frame: number, joint: number, out: Float32Array | number[], offset: number = 0): void {
        if (this.isConstant[joint]) {
            out[offset] = this.constants[3 * joint];
            out[offset + 1] = this.constants[3 * joint + 1];
            out[offset + 2] = this.constants[3 * joint + 2];
            return;
        }
        const s = this.slot[joint];
        const q = (frame * this.animatedJoints + s) * 3;
        for (let i = 0; i < 3; i++) out[offset + i] = this.center[3 * s + i] + this.extent[3 * s + i] * this.animated[q + i] / 32767;
    }
}

/** A quaternion quantized to four signed 16-bit components (w kept non-negative), into `out` at `offset`. */
function quantize(q: quat, out: Int16Array, offset: number): void {
    const s = q[3] < 0 ? -1 : 1;
    for (let i = 0; i < 4; i++) out[offset + i] = roundAway(Math.min(Math.max(s * q[i], -1), 1) * 32767);
}

/** What a search may return. */
interface SearchFilter {
    /**
     * Frames at the end of a clip that doesn't loop that the search never lands on, so playback
     * has time to search again before the clip runs out.
     */
    ignoreEnd: number;
    /** Only clips with one of these tag bits (every clip but actions by default: `~ACTION_TAG`). */
    tags: number;
    /**
     * The frame playing now: frames of its clip within `ignoreNear` of it are skipped (landing
     * next to the playhead only restarts what is already playing).
     */
    current?: number;
    ignoreNear: number;
}

function defaultSearchFilter(): SearchFilter {
    return { ignoreEnd: 10, tags: ~ACTION_TAG >>> 0, current: undefined, ignoreNear: 3 };
}

/**
 * Whether a clip with these tags may be found: one of the filter's bits, and action clips
 * only when the filter asks for `ACTION_TAG`.
 */
function filterAllows(filter: SearchFilter, tags: number): boolean {
    return (tags & filter.tags) !== 0 && ((tags & ACTION_TAG) === 0 || (filter.tags & ACTION_TAG) !== 0);
}

/** The best match found: a database frame and its squared feature distance. */
interface Match {
    frame: number;
    cost: number;
}

const f32 = Math.fround;

/**
 * Squared distance between feature rows (`a` at `ao`, `b` at `bo`), stopping once it passes
 * `limit`. In f32, summed as the Rust engine's wasm build sums it (glam's simd128 `Vec4::dot`:
 * (x² + z²) + (y² + w²), then the running total), so equal costs (mirrored frames of a cycle)
 * break the same way.
 */
function distance(a: Float32Array, ao: number, b: Float32Array, bo: number, limit: number): number {
    let total = 0;
    for (let k = 0; k < STRIDE; k += 4) {
        const d0 = f32(a[ao + k] - b[bo + k]);
        const d1 = f32(a[ao + k + 1] - b[bo + k + 1]);
        const d2 = f32(a[ao + k + 2] - b[bo + k + 2]);
        const d3 = f32(a[ao + k + 3] - b[bo + k + 3]);
        total = f32(total + f32(f32(f32(d0 * d0) + f32(d2 * d2)) + f32(f32(d1 * d1) + f32(d3 * d3))));
        if (total >= limit) break;
    }
    return total;
}

/**
 * Squared distance from `q` to box `index` of `bounds` (lo row then hi row), a lower bound of
 * its distance to every row inside; in f32 as `distance`.
 */
function boxDistance(q: Float32Array, bounds: Float32Array, index: number, limit: number): number {
    const lo = index * 2 * STRIDE;
    const hi = lo + STRIDE;
    let total = 0;
    for (let k = 0; k < STRIDE; k += 4) {
        const v0 = q[k], v1 = q[k + 1], v2 = q[k + 2], v3 = q[k + 3];
        const d0 = f32(v0 - Math.min(Math.max(v0, bounds[lo + k]), bounds[hi + k]));
        const d1 = f32(v1 - Math.min(Math.max(v1, bounds[lo + k + 1]), bounds[hi + k + 1]));
        const d2 = f32(v2 - Math.min(Math.max(v2, bounds[lo + k + 2]), bounds[hi + k + 2]));
        const d3 = f32(v3 - Math.min(Math.max(v3, bounds[lo + k + 3]), bounds[hi + k + 3]));
        total = f32(total + f32(f32(f32(d0 * d0) + f32(d2 * d2)) + f32(f32(d1 * d1) + f32(d3 * d3))));
        if (total >= limit) break;
    }
    return total;
}

/** A root on the ground: its position and heading (radians about +Y). */
type RootSample = [vec3, number];

/**
 * Animation frames ready for motion matching: every clip's poses (quantized rotations), the
 * character root's motion, foot contacts and the normalized search features, with bounding
 * boxes over runs of frames for the search to skip. Rust: `motion_matching::Database`.
 */
class Database {
    /** The character root per frame, in its clip's space: positions (3 floats), rotations (4) and headings. */
    public readonly rootTranslations: Float32Array;
    public readonly rootRotations: Float32Array;
    private readonly rootYaws: Float32Array;
    /** (lo row, hi row) per `BOUND_SMALL` and `BOUND_LARGE` frames. */
    private boundsSmall: Float32Array = new Float32Array(0);
    private boundsLarge: Float32Array = new Float32Array(0);
    private scratchA = new Float32Array(3);
    private scratchB = new Float32Array(3);

    constructor(
        public skeleton: Skeleton,
        public roles: JointRoles,
        public sampleRate: number,
        public weights: FeatureWeights,
        public clips: ClipInfo[],
        /** `(frame * joints + joint) * 4`: local, the root joint relative to the character root. */
        public readonly rotations: Int16Array,
        public readonly translations: Vec3Tracks,
        public readonly scales: Vec3Tracks,
        roots: { translations: Float32Array, rotations: Float32Array },
        /** Per frame: bit 0 left foot planted, bit 1 right. */
        public readonly contactBits: Uint8Array,
        /** Per feature, subtracted then divided to normalize. */
        public featureOffset: Float32Array,
        public featureScale: Float32Array,
        /** `frame * STRIDE`, normalized, the padding zero. */
        public readonly featureRows: Float32Array,
    ) {
        this.rootTranslations = roots.translations;
        this.rootRotations = roots.rotations;
        const frames = this.rootTranslations.length / 3;
        this.rootYaws = new Float32Array(frames);
        for (let f = 0; f < frames; f++) {
            const x = this.rootRotations[4 * f], y = this.rootRotations[4 * f + 1], z = this.rootRotations[4 * f + 2], w = this.rootRotations[4 * f + 3];
            // the forward axis (+Z) rotated: atan2 of its x and z
            this.rootYaws[f] = Math.atan2(2 * (x * z + w * y), 1 - 2 * (x * x + y * y));
        }
        this.buildBounds();
    }

    public get frameCount(): number {
        return this.rootYaws.length;
    }

    public get jointCount(): number {
        return this.skeleton.length;
    }

    /** The clip holding database frame `frame`. */
    public clipOf(frame: number): number {
        let lo = 0, hi = this.clips.length;
        while (lo < hi) {
            const mid = (lo + hi) >>> 1;
            if (this.clips[mid].start <= frame) lo = mid + 1;
            else hi = mid;
        }
        return lo - 1;
    }

    /** A frame's normalized features (a view of `featureRows`). */
    public features(frame: number): Float32Array {
        return this.featureRows.subarray(frame * STRIDE, (frame + 1) * STRIDE);
    }

    /** Whether each foot is planted at `frame`. */
    public contacts(frame: number): [boolean, boolean] {
        const c = this.contactBits[frame];
        return [(c & 1) !== 0, (c & 2) !== 0];
    }

    /** The character root at `frame`, in its clip's space. */
    public root(frame: number): Transform {
        return Transform.fromTranslationRotation(
            this.rootTranslations.subarray(3 * frame, 3 * frame + 3) as vec3,
            this.rootRotations.subarray(4 * frame, 4 * frame + 4) as quat,
        );
    }

    /** A joint's local transform at a frame. */
    public transform(frame: number, joint: number): Transform {
        const t = Transform.identity();
        this.translations.get(frame, joint, t.translation);
        this.scales.get(frame, joint, t.scale);
        const k = (frame * this.jointCount + joint) * 4;
        for (let i = 0; i < 4; i++) t.rotation[i] = this.rotations[k + i] / 32767;
        quat.normalize(t.rotation, t.rotation);
        return t;
    }

    /** The pose between frames `a` and `b` (`t` of the way), into `out` (its transforms reused). */
    public pose(a: number, b: number, t: number, out: Pose): Pose {
        const joints = this.jointCount;
        if (out.local.length !== joints) out.local = Array.from({ length: joints }, () => Transform.identity());
        const x = this.scratchA, y = this.scratchB;
        const ra = quat.create(), rb = quat.create();
        for (let j = 0; j < joints; j++) {
            const o = out.local[j];
            this.translations.get(a, j, x);
            this.translations.get(b, j, y);
            vec3.lerp(o.translation, x as vec3, y as vec3, t);
            this.scales.get(a, j, x);
            this.scales.get(b, j, y);
            vec3.lerp(o.scale, x as vec3, y as vec3, t);
            const ka = (a * joints + j) * 4, kb = (b * joints + j) * 4;
            for (let i = 0; i < 4; i++) {
                ra[i] = this.rotations[ka + i] / 32767;
                rb[i] = this.rotations[kb + i] / 32767;
            }
            quat.normalize(ra, ra);
            quat.normalize(rb, rb);
            nlerp(ra, rb, t, o.rotation);
        }
        return out;
    }

    /**
     * Each joint's local linear and angular velocity at `frame` (forward difference, backward at
     * a clip's last frame), into `linear` and `angular` (resized to fit).
     */
    public velocities(frame: number, linear: vec3[], angular: vec3[]): void {
        const clip = this.clips[this.clipOf(frame)];
        const [a, b] = frame + 1 < clip.start + clip.frames ? [frame, frame + 1] : [frame - 1, frame];
        const joints = this.jointCount;
        linear.length = joints;
        angular.length = joints;
        for (let j = 0; j < joints; j++) {
            const x = this.transform(a, j), y = this.transform(b, j);
            linear[j] = vec3.scale(vec3.create(), vec3.subtract(vec3.create(), y.translation, x.translation), this.sampleRate);
            angular[j] = angularVelocity(x.rotation, y.rotation, 1 / this.sampleRate);
        }
    }

    /**
     * The character root (position and heading) at fractional frame `frame` of clip `clip`, in
     * the clip's space.
     */
    public rootAt(clip: number, frame: number): RootSample {
        return this.rootIn(this.clips[clip], frame);
    }

    private rootIn(clip: ClipInfo, f: number): RootSample {
        const a = Math.min(Math.max(Math.floor(f), 0), clip.frames - 1);
        const b = Math.min(a + 1, clip.frames - 1);
        const t = f - a;
        const ka = clip.start + a, kb = clip.start + b;
        const p = vec3.create();
        for (let i = 0; i < 3; i++) {
            const x = this.rootTranslations[3 * ka + i];
            p[i] = x + (this.rootTranslations[3 * kb + i] - x) * t;
        }
        const ya = this.rootYaws[ka];
        return [p, ya + wrapAngle(this.rootYaws[kb] - ya) * t];
    }

    /**
     * How the character root moves while clip `clip` plays from fractional frame `from` to `to`
     * (clip frames; `to` may pass a loop's end): the displacement in the root's frame at `from`,
     * and the turn in radians about +Y.
     */
    public rootMotion(clip: number, from: number, to: number): [vec3, number] {
        const info = this.clips[clip];
        if (info.looping && to > info.frames - 1) {
            // to the loop's end, then on from its start
            const end = info.frames - 1;
            const [d0, y0] = this.rootMotion(clip, from, end);
            const [d1, y1] = this.rootMotion(clip, 0, to - end);
            return [vec3.add(d0, d0, vec3.transformQuat(d1, d1, yawRotation(y0))), y0 + y1];
        }
        const [pa, ya] = this.rootIn(info, from);
        const [pb, yb] = this.rootIn(info, Math.min(to, info.frames - 1));
        const d = vec3.subtract(pb, pb, pa);
        return [vec3.transformQuat(d, d, yawRotation(-ya)), wrapAngle(yb - ya)];
    }

    /** Normalized features of raw ones, into `out` (padding zero). */
    public normalizeQuery(raw: ArrayLike<number>, out: Float32Array = new Float32Array(STRIDE)): Float32Array {
        for (let i = 0; i < FEATURES; i++) out[i] = (raw[i] - this.featureOffset[i]) / this.featureScale[i];
        out[FEATURES] = 0;
        return out;
    }

    /** Raw features of normalized ones. */
    public denormalize(features: ArrayLike<number>, out: Float32Array = new Float32Array(FEATURES)): Float32Array {
        for (let i = 0; i < FEATURES; i++) out[i] = features[i] * this.featureScale[i] + this.featureOffset[i];
        return out;
    }

    /**
     * The frame whose features are nearest `query` (normalized, see `normalizeQuery`), if any is
     * nearer than `bestCost`. Brute force over every allowed frame, skipping runs of frames whose
     * bounding box is already too far (Holden et al., "Learned Motion Matching", 2020).
     */
    public search(query: Float32Array, filter: SearchFilter, bestCost: number = Infinity): Match | undefined {
        let best: Match | undefined;
        let limit = bestCost;
        const rows = this.featureRows, small = this.boundsSmall, large = this.boundsLarge;
        const currentClip = filter.current === undefined ? -1 : this.clipOf(filter.current);
        for (let c = 0; c < this.clips.length; c++) {
            const clip = this.clips[c];
            if (!filterAllows(filter, clip.tags)) continue;
            const end = clip.start + (clip.looping ? clip.playable : Math.max(clip.frames - filter.ignoreEnd, 1));
            // the frames near the playhead, in its clip
            let skipFrom = 0, skipTo = 0;
            if (c === currentClip) {
                skipFrom = Math.max(filter.current! - filter.ignoreNear, 0);
                skipTo = filter.current! + filter.ignoreNear + 1;
            }
            let i = clip.start;
            while (i < end) {
                const l = (i / BOUND_LARGE) | 0;
                const largeEnd = Math.min((l + 1) * BOUND_LARGE, end);
                if (boxDistance(query, large, l, limit) >= limit) {
                    i = largeEnd;
                    continue;
                }
                while (i < largeEnd) {
                    const s = (i / BOUND_SMALL) | 0;
                    const smallEnd = Math.min((s + 1) * BOUND_SMALL, largeEnd);
                    if (boxDistance(query, small, s, limit) >= limit) {
                        i = smallEnd;
                        continue;
                    }
                    for (; i < smallEnd; i++) {
                        if (i >= skipFrom && i < skipTo) continue;
                        const cost = distance(query, 0, rows, i * STRIDE, limit);
                        if (cost < limit) {
                            limit = cost;
                            best = { frame: i, cost };
                        }
                    }
                }
            }
        }
        return best;
    }

    /**
     * `search` without the bounding boxes: every allowed frame's full distance. The reference the
     * accelerated search must agree with.
     */
    public searchBruteForce(query: Float32Array, filter: SearchFilter, bestCost: number = Infinity): Match | undefined {
        let best: Match | undefined;
        let limit = bestCost;
        const currentClip = filter.current === undefined ? -1 : this.clipOf(filter.current);
        this.clips.forEach((clip, c) => {
            if (!filterAllows(filter, clip.tags)) return;
            const end = clip.start + (clip.looping ? clip.playable : Math.max(clip.frames - filter.ignoreEnd, 1));
            for (let i = clip.start; i < end; i++) {
                if (c === currentClip && i + filter.ignoreNear >= filter.current! && i <= filter.current! + filter.ignoreNear) continue;
                const cost = distance(query, 0, this.featureRows, i * STRIDE, Infinity);
                if (cost < limit) {
                    limit = cost;
                    best = { frame: i, cost };
                }
            }
        });
        return best;
    }

    /** Per-feature mean; per-group spread (the root mean square deviation over the group), divided by the group's weight. */
    public static normalization(raw: Float32Array, frames: number, weights: FeatureWeights): { offset: Float32Array, scale: Float32Array } {
        const offset = new Float32Array(FEATURES), scale = new Float32Array(FEATURES).fill(1);
        const n = Math.max(frames, 1);
        for (let i = 0; i < FEATURES; i++) {
            let sum = 0;
            for (let f = 0; f < frames; f++) sum += raw[f * FEATURES + i];
            offset[i] = sum / n;
        }
        for (const [first, count, weight] of featureGroups(weights)) {
            let sum = 0;
            for (let f = 0; f < frames; f++) {
                for (let i = first; i < first + count; i++) sum += (raw[f * FEATURES + i] - offset[i]) ** 2;
            }
            const spread = Math.max(Math.sqrt(sum / (n * count)), 1e-6);
            for (let i = first; i < first + count; i++) scale[i] = spread / Math.max(weight, 1e-6);
        }
        return { offset, scale };
    }

    private buildBounds(): void {
        const frames = this.frameCount;
        const bounds = (size: number) => {
            const count = Math.ceil(frames / size);
            const out = new Float32Array(count * 2 * STRIDE);
            for (let b = 0; b < count; b++) {
                const lo = b * 2 * STRIDE, hi = lo + STRIDE;
                out.fill(Infinity, lo, lo + STRIDE);
                out.fill(-Infinity, hi, hi + STRIDE);
                for (let f = b * size; f < Math.min((b + 1) * size, frames); f++) {
                    for (let i = 0; i < STRIDE; i++) {
                        const v = this.featureRows[f * STRIDE + i];
                        if (v < out[lo + i]) out[lo + i] = v;
                        if (v > out[hi + i]) out[hi + i] = v;
                    }
                }
            }
            return out;
        };
        this.boundsSmall = bounds(BOUND_SMALL);
        this.boundsLarge = bounds(BOUND_LARGE);
    }
}

/** Collects clips on one skeleton and builds their `Database`. Rust: `motion_matching::DatabaseBuilder`. */
class DatabaseBuilder {
    public weights: FeatureWeights = defaultFeatureWeights();
    public contacts: ContactThresholds = defaultContactThresholds();
    private clips: { clip: Clip, looping: boolean, tags: number }[] = [];

    constructor(public skeleton: Skeleton, public roles: JointRoles, public sampleRate: number) { }

    public withWeights(weights: FeatureWeights): this {
        this.weights = weights;
        return this;
    }

    public withContacts(contacts: ContactThresholds): this {
        this.contacts = contacts;
        return this;
    }

    /** Add a clip on the builder's skeleton at its sample rate; throws when it doesn't fit. */
    public addClip(clip: Clip, looping: boolean, tags: number): void {
        if (clip.jointCount !== this.skeleton.length) throw new Error(`clip '${clip.name}' has ${clip.jointCount} joints, the skeleton ${this.skeleton.length}`);
        if (Math.abs(clip.sampleRate - this.sampleRate) > 1e-3) throw new Error(`clip '${clip.name}' is at ${clip.sampleRate} fps, the database at ${this.sampleRate}`);
        if (clip.frameCount < 2 || (looping && clip.frameCount < 3)) throw new Error(`clip '${clip.name}' is too short (${clip.frameCount} frames)`);
        this.clips.push({ clip, looping, tags });
    }

    public build(): Database {
        const skeleton = this.skeleton, roles = this.roles, rate = this.sampleRate;
        const joints = skeleton.length;
        const total = this.clips.reduce((s, c) => s + c.clip.frameCount, 0);
        const clips: ClipInfo[] = [];
        const rotations = new Int16Array(total * joints * 4);
        const translations = new Float32Array(total * joints * 3);
        const scales = new Float32Array(total * joints * 3);
        const rootTranslations = new Float32Array(total * 3);
        const rootRotations = new Float32Array(total * 4);
        const contacts = new Uint8Array(total);
        const raw = new Float32Array(total * FEATURES);
        const restModel = skeleton.restModel();
        const restFeet = [restModel[roles.feet[0]].translation[1], restModel[roles.feet[1]].translation[1]];
        let at = 0;

        for (const { clip, looping, tags } of this.clips) {
            const n = clip.frameCount;
            const start = at;
            clips.push(new ClipInfo(clip.name, start, n, looping, tags));

            // the character root per frame: the root joint's position and heading; the pose keeps
            // what is left of the root joint under it (nothing, for a root on the ground)
            const pose = new Pose([]);
            const model: Transform[] = [];
            const root: Transform[] = [];
            const feet: [vec3, vec3][] = [];
            const hips: vec3[] = [];
            for (let f = 0; f < n; f++) {
                clip.framePose(f, pose);
                pose.toModel(skeleton, model);
                const r = model[roles.root];
                const character = Transform.fromTranslationRotation(r.translation, yawRotation(yawOf(r.rotation)));
                root.push(character);
                pose.local[roles.root] = character.inverse().mul(pose.local[roles.root]);
                pose.local.forEach((t, j) => {
                    const k = (start + f) * joints + j;
                    quantize(t.rotation, rotations, 4 * k);
                    translations.set(t.translation, 3 * k);
                    scales.set(t.scale, 3 * k);
                });
                feet.push([vec3.clone(model[roles.feet[0]].translation), vec3.clone(model[roles.feet[1]].translation)]);
                hips.push(vec3.clone(model[roles.hips].translation));
            }

            // velocities by central differences: one-sided at a clip's ends, around the seam of a
            // loop (the frame before its first is its second-to-last a cycle back, the frame after
            // its last is its second a cycle on)
            const period = looping ? n - 1 : n;
            const cycle = root[n - 1].mul(root[0].inverse());
            const cycleInverse = cycle.inverse();
            const velocity = (f: number, positions: (k: number) => vec3): vec3 => {
                let before: vec3, after: vec3, span: number;
                if (looping && f === 0) [before, after, span] = [cycleInverse.transformPoint(positions(n - 2)), positions(1), 2];
                else if (looping && f === n - 1) [before, after, span] = [positions(n - 2), cycle.transformPoint(positions(1)), 2];
                else if (!looping && f === 0) [before, after, span] = [positions(0), positions(1), 1];
                else if (!looping && f === n - 1) [before, after, span] = [positions(n - 2), positions(n - 1), 1];
                else [before, after, span] = [positions(f - 1), positions(f + 1), 2];
                return vec3.scale(vec3.create(), vec3.subtract(vec3.create(), after, before), rate / span);
            };

            // contacts: low and slow, in runs of at least minFrames
            const planted = Array.from({ length: n }, () => [false, false]);
            for (let side = 0; side < 2; side++) {
                for (let f = 0; f < n; f++) {
                    const v = velocity(f, (k) => feet[k][side]);
                    const height = feet[f][side][1] - root[f].translation[1];
                    planted[f][side] = height < restFeet[side] + this.contacts.height && vec3.length(v) < this.contacts.speed;
                }
                let f = 0;
                while (f < n) {
                    if (planted[f][side]) {
                        let run = 0;
                        while (f + run < n && planted[f + run][side]) run++;
                        if (run < this.contacts.minFrames) for (let k = f; k < f + run; k++) planted[k][side] = false;
                        f += run;
                    } else {
                        f++;
                    }
                }
            }
            planted.forEach((p, f) => contacts[start + f] = (p[0] ? 1 : 0) | (p[1] ? 2 : 0));

            // the root at any frame ahead: loops continue cycle after cycle, other clips go on
            // at their last frame's velocity and turn rate
            const lastVelocity = vec3.scale(vec3.create(), root[n - 2].inverse().transformPoint(root[n - 1].translation), rate);
            const lastYaw = yawOf(root[n - 1].rotation);
            const lastTurn = wrapAngle(lastYaw - yawOf(root[n - 2].rotation)) * rate;
            const rootAhead = (f: number): Transform => {
                if (f < n) return root[f];
                if (looping) {
                    const cycles = Math.floor(f / period), rest = f % period;
                    let t = root[rest];
                    for (let c = 0; c < cycles; c++) t = cycle.mul(t);
                    return t;
                }
                const dt = (f - (n - 1)) / rate;
                const heading = yawRotation(lastYaw + 0.5 * lastTurn * dt);
                const p = vec3.transformQuat(vec3.create(), lastVelocity, heading);
                vec3.scaleAndAdd(p, root[n - 1].translation, p, dt);
                return Transform.fromTranslationRotation(p, yawRotation(lastYaw + lastTurn * dt));
            };

            const toLocal = quat.create();
            const forward = vec3.create();
            for (let f = 0; f < n; f++) {
                const here = root[f];
                quat.conjugate(toLocal, here.rotation);
                const x = raw.subarray((start + f) * FEATURES, (start + f + 1) * FEATURES);
                const local = (v: vec3) => vec3.transformQuat(v, v, toLocal);
                for (let side = 0; side < 2; side++) {
                    x.set(local(vec3.subtract(vec3.create(), feet[f][side], here.translation)), 3 * side);
                    x.set(local(velocity(f, (k) => feet[k][side])), 6 + 3 * side);
                }
                x.set(local(velocity(f, (k) => hips[k])), 12);
                TRAJECTORY_TIMES.forEach((t, k) => {
                    const ahead = rootAhead(f + Math.round(t * rate));
                    const p = local(vec3.subtract(vec3.create(), ahead.translation, here.translation));
                    const d = local(vec3.transformQuat(forward, FORWARD, ahead.rotation));
                    x[15 + 2 * k] = p[0];
                    x[16 + 2 * k] = p[2];
                    x[21 + 2 * k] = d[0];
                    x[22 + 2 * k] = d[2];
                });
            }
            root.forEach((r, f) => {
                rootTranslations.set(r.translation, 3 * (start + f));
                rootRotations.set(r.rotation, 4 * (start + f));
            });
            at += n;
        }

        const { offset, scale } = Database.normalization(raw, total, this.weights);
        const rows = new Float32Array(total * STRIDE);
        for (let f = 0; f < total; f++) {
            for (let i = 0; i < FEATURES; i++) rows[f * STRIDE + i] = (raw[f * FEATURES + i] - offset[i]) / scale[i];
        }
        return new Database(
            skeleton, roles, rate, this.weights, clips, rotations,
            Vec3Tracks.build(translations, total, joints, 1e-5),
            Vec3Tracks.build(scales, total, joints, 1e-5),
            { translations: rootTranslations, rotations: rootRotations },
            contacts, offset, scale, rows,
        );
    }
}

export {
    FEATURES, STRIDE, BOUND_SMALL, BOUND_LARGE, TRAJECTORY_TIMES, ACTION_TAG,
    findJointRoles, defaultFeatureWeights, defaultContactThresholds, defaultSearchFilter, filterAllows, distance,
    ClipInfo, Vec3Tracks, Database, DatabaseBuilder,
};
export type { JointRoles, FeatureWeights, ContactThresholds, SearchFilter, Match, RootSample };
