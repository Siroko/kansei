import { quat, vec3 } from "gl-matrix";
import { FootLock, twoJointIK } from "../IK";
import type { FootPose } from "../IK";
import { Inertializer } from "../Inertialization";
import { Pose } from "../Pose";
import { Retarget } from "../Retarget";
import { Skeleton } from "../Skeleton";
import { damperExact, negexp, springCharacterUpdate, springDamperExactQuat } from "../Springs";
import { Transform, quatAbs, quatFromScaledAngleAxis, quatToScaledAngleAxis } from "../Transform";
import { RootWarp } from "../Warping";
import { Database, FEATURES, STRIDE, SearchFilter, TRAJECTORY_TIMES, defaultSearchFilter, distance } from "./Database";
import { FORWARD, remEuclid, wrapAngle, yawOf, yawRotation } from "./Heading";

/** Tuning of a `MotionMatcher`. The defaults follow Holden's reference controller at 30 Hz data. */
interface MotionMatchingSettings {
    /** Seconds between searches (a change of input searches at once). */
    searchInterval: number;
    /**
     * A search switches only to a frame that costs less than the frame playing minus
     * `continuingBias` (in squared normalized feature units). Databases cut from the same takes
     * hold many copies of the same frames (a loop's cycle at the start of a stop, a turn, a
     * pivot); without a bias the search hops between them.
     */
    continuingBias: number;
    /**
     * Search now when the desired velocity changed by this much (m/s) or the desired facing by
     * this much (radians) since the last search.
     */
    forceSearchVelocity: number;
    forceSearchTurn: number;
    filter: SearchFilter;
    /** Half-life of the transition offsets (seconds). */
    inertializationHalflife: number;
    /** Half-lives of the simulated character's velocity and facing springs (seconds). */
    velocityHalflife: number;
    rotationHalflife: number;
    /**
     * How the animated character is pulled toward the simulation: a damper of this half-life,
     * limited (when `adjustByVelocity`) to `maxAdjustmentRatio` of the character's own speed,
     * so a standing character does not slide; then clamped to `clampDistance` metres and
     * `clampAngle` radians.
     */
    adjustmentHalflife: number;
    adjustByVelocity: boolean;
    maxAdjustmentRatio: number;
    clampDistance: number;
    clampAngle: number;
    /**
     * Foot locking: pin planted feet with two-joint IK. A pin lets go when the contact ends or
     * the animated foot strays `footUnlockRadius` metres from it; the foot then fades back onto
     * the animation with `footLockHalflife`, and is pinned again there while the contact lasts.
     */
    footLock: boolean;
    footUnlockRadius: number;
    footLockHalflife: number;
}

function defaultMotionMatchingSettings(): MotionMatchingSettings {
    return {
        searchInterval: 0.1,
        continuingBias: 0.01,
        forceSearchVelocity: 0.5,
        forceSearchTurn: 0.35,
        filter: defaultSearchFilter(),
        inertializationHalflife: 0.1,
        velocityHalflife: 0.27,
        rotationHalflife: 0.27,
        adjustmentHalflife: 0.1,
        adjustByVelocity: true,
        maxAdjustmentRatio: 0.5,
        clampDistance: 0.15,
        clampAngle: Math.PI / 2,
        footLock: true,
        footUnlockRadius: 0.3,
        footLockHalflife: 0.1,
    };
}

/** What the player asks for this frame. */
interface MotionInput {
    /** Desired velocity in world space (m/s; y ignored). */
    velocity: vec3;
    /** Desired facing (yaw about +Y, radians) for strafing; unset faces the way it moves. */
    facing?: number;
}

/** The simulated character: where the input says it should be, as springs on velocity and facing. */
interface Simulation {
    position: vec3;
    velocity: vec3;
    acceleration: vec3;
    rotation: quat;
    angularVelocity: vec3;
}

/** The last search: whether one ran, whether it switched, the frame it chose and its cost. */
interface SearchInfo {
    searched: boolean;
    switched: boolean;
    frame: number;
    cost: number;
}

/** How the character moves while an action plays. */
type RootPath =
    /** The clip's root motion placed in the world by a warp (traversal, landing). */
    | { kind: 'warp', warp: RootWarp }
    /** Momentum and gravity (jumping, falling); the clip only animates the pose. */
    | { kind: 'ballistic', velocity: vec3, gravity: number };

/** A clip played on command instead of found by the search: a traversal, a fall, a landing. */
interface Action {
    /** Database clip, and the clip frame it starts at. */
    clip: number;
    start: number;
    /**
     * Clip frame at which motion matching takes over again; unset plays until
     * `MotionMatcher.stopAction` (a looping clip goes round).
     */
    exit?: number;
    path: RootPath;
    /**
     * Whether the path is kept out of the world like the simulation (`updateConstrained`):
     * yes for jumps, falls and landings; not for traversals, which go over what they cross.
     */
    collides: boolean;
    /** For the caller: what the action is (e.g. a traversal kind). */
    tag: number;
}

/** Where a move from `from` toward `to` may end (the world's collision). */
type Constrain = (from: vec3, to: vec3) => vec3;

/** A character with its own proportions shown instead of the database's skeleton (see `MotionMatcher.setDisplay`). */
interface Display {
    skeleton: Skeleton;
    retarget: Retarget;
    legs: [Leg, Leg];
}

/**
 * A leg for foot locking: its (upper, middle, foot) joints, the foot's ball (its first child, or
 * the foot itself), and the heights of the foot and the ball at rest.
 */
interface Leg {
    joints: [number, number, number];
    ball: number;
    rest: [number, number];
}

const flat = (v: vec3): vec3 => vec3.fromValues(v[0], 0, v[2]);

function normalizeOr(v: vec3, fallback: vec3): vec3 {
    const l = vec3.length(v);
    return l > 0 && Number.isFinite(1 / l) ? vec3.scale(vec3.create(), v, 1 / l) : fallback;
}

/** `v` shortened to at most `max` long, in place. */
function clampLengthMax(v: vec3, max: number): vec3 {
    const l2 = vec3.squaredLength(v);
    if (l2 > max * max) vec3.scale(v, v, max / Math.sqrt(l2));
    return v;
}

/**
 * The horizontal direction a move from `from` toward `wanted` was blocked in, when it ended at
 * `allowed`: what was taken off the move, less any part of it along the move that was allowed
 * (a slide along a wall, stopped a hair short of it, loses a little of both).
 */
function blockedDirection(from: vec3, wanted: vec3, allowed: vec3): vec3 {
    const removed = flat(vec3.subtract(vec3.create(), wanted, allowed));
    const along = normalizeOr(flat(vec3.subtract(vec3.create(), allowed, from)), vec3.create());
    const across = vec3.scaleAndAdd(vec3.create(), removed, along, -vec3.dot(removed, along));
    return normalizeOr(across, normalizeOr(removed, vec3.create()));
}

/** The leg ending at `foot`. */
function leg(skeleton: Skeleton, foot: number): Leg {
    const middle = skeleton.parents[foot] ?? foot;
    const child = skeleton.parents.indexOf(foot);
    const ball = child < 0 ? foot : child;
    const rest = skeleton.restModel();
    return { joints: [skeleton.parents[middle] ?? middle, middle, foot], ball, rest: [rest[foot].translation[1], rest[ball].translation[1]] };
}

/**
 * A character driven by motion matching: every few frames it searches the database for the
 * frame whose pose and future trajectory best match its current pose and the trajectory the
 * input predicts, switches there with inertialization, and plays on, moved by the clips' root
 * motion, pulled toward a spring-simulated position, its planted feet pinned by IK. Rust:
 * `motion_matching::MotionMatcher`.
 */
class MotionMatcher {
    public settings: MotionMatchingSettings;
    private clip = 0;
    /** Playhead in frames of `clip`. */
    private frame = 0;
    /**
     * The database's pose at the playhead, the pose shown (inertialized, on the database's
     * skeleton), and the output: that pose on the display skeleton if any, feet locked.
     */
    private sampled: Pose;
    private matched: Pose;
    private output: Pose;
    private outputModel: Transform[] = [];
    private display?: Display;
    /** The character root in the world: position and heading. */
    private root: Transform;
    private sim: Simulation;
    private desiredYaw: number;
    private predicted: [Transform, Transform, Transform];
    private inertializer: Inertializer;
    private searchTimer = 0;
    private searchedInput?: [vec3, number];
    private feet: [FootLock, FootLock] = [new FootLock(), new FootLock()];
    /** (upper, middle, foot) joints of each leg. */
    private legs: [Leg, Leg];
    private searchInfo: SearchInfo = { searched: false, switched: false, frame: 0, cost: 0 };
    /** The root motion's speed (m/s) and turn rate (rad/s) this frame. */
    private rootSpeed: [number, number] = [0, 0];
    /** The clip playing on command, if any, instead of searching. */
    private playingAction?: Action;
    /** Height of the ground the character stands on while matching. */
    private groundHeight: number;
    private scratch = { sourceLinear: [] as vec3[], sourceAngular: [] as vec3[], destinationLinear: [] as vec3[], destinationAngular: [] as vec3[], destination: new Pose([]), raw: new Float32Array(FEATURES), query: new Float32Array(STRIDE) };

    /** A character standing at `position` facing `yaw`, on the database's first frame until the first update searches. */
    constructor(db: Database, settings: MotionMatchingSettings = defaultMotionMatchingSettings(), position: vec3 = vec3.create(), yaw: number = 0) {
        this.settings = settings;
        this.sampled = db.pose(0, 0, 0, new Pose([]));
        this.output = this.sampled.clone();
        this.matched = this.sampled.clone();
        this.root = Transform.fromTranslationRotation(position, yawRotation(yaw));
        this.sim = { position: vec3.clone(position), velocity: vec3.create(), acceleration: vec3.create(), rotation: quat.clone(this.root.rotation), angularVelocity: vec3.create() };
        this.desiredYaw = yaw;
        this.predicted = [this.root.clone(), this.root.clone(), this.root.clone()];
        this.inertializer = new Inertializer(db.jointCount);
        this.legs = [leg(db.skeleton, db.roles.feet[0]), leg(db.skeleton, db.roles.feet[1])];
        this.groundHeight = position[1];
        this.output.toModel(db.skeleton, this.outputModel);
    }

    /**
     * Show the animation on another skeleton with the same joint names and axes but its own
     * proportions (`retarget` from the database's skeleton to `skeleton`); unset shows the
     * database's. Foot locking then works on that skeleton's legs.
     */
    public setDisplay(db: Database, display?: { skeleton: Skeleton, retarget: Retarget }): void {
        if (display) {
            const foot = (j: number) => display.skeleton.find(db.skeleton.names[j]) ?? 0;
            this.display = { ...display, legs: [leg(display.skeleton, foot(db.roles.feet[0])), leg(display.skeleton, foot(db.roles.feet[1]))] };
        } else {
            this.display = undefined;
        }
        this.feet.forEach((f) => f.reset());
        this.updateOutput(db);
    }

    /** The skeleton the output pose is on. */
    public outputSkeleton(db: Database): Skeleton {
        return this.display?.skeleton ?? db.skeleton;
    }

    /** The output pose (local, on `outputSkeleton`; the root joint relative to `character`). */
    public get pose(): Pose {
        return this.output;
    }

    /** The output pose in model space (relative to `character`, on `outputSkeleton`). */
    public get model(): Transform[] {
        return this.outputModel;
    }

    /** The character root in the world (position and heading): the mesh's placement. */
    public get character(): Transform {
        return this.root;
    }

    public get simulation(): Simulation {
        return this.sim;
    }

    /** The simulation's predicted root at `TRAJECTORY_TIMES` ahead (world). */
    public get trajectory(): readonly Transform[] {
        return this.predicted;
    }

    /** The clip playing and the playhead in its frames. */
    public playing(): [number, number] {
        return [this.clip, this.frame];
    }

    /** The database frame at the playhead. */
    public currentFrame(db: Database): number {
        const info = db.clips[this.clip];
        return info.start + Math.min(Math.round(this.frame), info.frames - 1);
    }

    public get lastSearch(): SearchInfo {
        return this.searchInfo;
    }

    /** Whether each foot is pinned. */
    public feetLocked(): [boolean, boolean] {
        return [this.feet[0].isLocked(), this.feet[1].isLocked()];
    }

    /** The action playing, if any (change its path or exit as it plays: a jump leaving the ground). */
    public get action(): Action | undefined {
        return this.playingAction;
    }

    /** Play `action` now instead of searching, inertializing from what shows. */
    public startAction(db: Database, action: Action): void {
        const from = this.currentFrame(db);
        const to = db.clips[action.clip].start + Math.min(Math.round(action.start), db.clips[action.clip].frames - 1);
        this.transition(db, from, to);
        this.frame = action.start;
        this.feet.forEach((f) => f.reset());
        this.playingAction = action;
    }

    /** End the action: motion matching searches again at the next update. */
    public stopAction(): void {
        if (this.playingAction === undefined) return;
        this.playingAction = undefined;
        this.searchedInput = undefined;
        vec3.copy(this.sim.position, this.root.translation);
        quat.copy(this.sim.rotation, this.root.rotation);
        vec3.zero(this.sim.angularVelocity);
        this.desiredYaw = yawOf(this.root.rotation);
    }

    /** Put the character here (keeping its pose and what plays), its simulation with it. */
    public place(character: Transform): void {
        this.root = character.clone();
        vec3.copy(this.sim.position, character.translation);
    }

    /**
     * The height of the ground under the character, which it stands on while matching (an
     * action's root path decides its height itself).
     */
    public setGround(height: number): void {
        this.groundHeight = height;
    }

    public get ground(): number {
        return this.groundHeight;
    }

    /** Move the character (and its simulation) without blending, facing `yaw`. */
    public teleport(position: vec3, yaw: number): void {
        this.root = Transform.fromTranslationRotation(position, yawRotation(yaw));
        this.sim = { position: vec3.clone(position), velocity: vec3.create(), acceleration: vec3.create(), rotation: quat.clone(this.root.rotation), angularVelocity: vec3.create() };
        this.desiredYaw = yaw;
        this.inertializer.reset();
        this.feet.forEach((f) => f.reset());
        this.searchedInput = undefined;
        this.playingAction = undefined;
        this.groundHeight = position[1];
    }

    /** Advance by `dt` seconds under `input`. */
    public update(db: Database, input: MotionInput, dt: number): void {
        this.updateConstrained(db, input, dt, (_, to) => to);
    }

    /**
     * `update`, with `constrain(from, to)` saying where a move from `from` toward `to` may end
     * (the world's collision): it shapes the simulated position, its predicted trajectory (so
     * the search sees the character stopping at a wall), the character, and an action's path
     * if it `collides`.
     */
    public updateConstrained(db: Database, input: MotionInput, dt: number, constrain: Constrain): void {
        dt = Math.max(dt, 1e-4);
        this.simulate(input, dt, constrain);
        if (this.playingAction) {
            this.playAction(db, dt, constrain);
        } else {
            this.searchIfDue(db, input, dt);
            this.play(db, dt);
        }
        this.inertializer.update(this.matched, this.settings.inertializationHalflife, dt);
        if (this.playingAction) {
            // the simulation waits where the action takes the character
            const s = this.sim;
            vec3.copy(s.position, this.root.translation);
            quat.copy(s.rotation, this.root.rotation);
            vec3.zero(s.acceleration);
            vec3.zero(s.angularVelocity);
        } else {
            const before = vec3.clone(this.root.translation);
            this.synchronize(dt);
            this.root.translation = vec3.clone(constrain(before, this.root.translation));
            // stand on the ground
            const t = this.root.translation;
            const g = this.groundHeight;
            t[1] = Math.abs(t[1] - g) > 0.5 ? g : t[1] + (g - t[1]) * (1 - negexp(dt / 0.05));
            this.sim.position[1] = g;
        }
        this.updateOutput(db);
        if (this.settings.footLock && !this.playingAction) this.lockFeet(db, dt);
    }

    /** The output pose from the matched one: retargeted onto the display skeleton if any. */
    private updateOutput(db: Database): void {
        if (this.display) {
            this.display.retarget.apply(this.matched, this.output);
            this.output.toModel(this.display.skeleton, this.outputModel);
        } else {
            this.output.copy(this.matched);
            this.output.toModel(db.skeleton, this.outputModel);
        }
    }

    /** The simulated character follows the input with springs; predict its trajectory. */
    private simulate(input: MotionInput, dt: number, constrain: Constrain): void {
        const goal = flat(input.velocity);
        if (input.facing !== undefined) this.desiredYaw = input.facing;
        else if (vec3.length(goal) > 0.1) this.desiredYaw = Math.atan2(goal[0], goal[2]);
        const goalRotation = yawRotation(this.desiredYaw);
        const s = this.sim;
        const before = vec3.clone(s.position);
        springCharacterUpdate(s.position, s.velocity, s.acceleration, goal, this.settings.velocityHalflife, dt);
        const allowed = constrain(before, s.position);
        if (vec3.distance(allowed, s.position) > 1e-5) {
            // blocked: drop the velocity into what blocked it (pushing out of an overlap is a
            // correction of the position, not speed)
            const blocked = blockedDirection(before, s.position, allowed);
            vec3.scaleAndAdd(s.velocity, s.velocity, blocked, -Math.max(vec3.dot(s.velocity, blocked), 0));
            s.position = vec3.clone(allowed);
        }
        springDamperExactQuat(s.rotation, s.angularVelocity, goalRotation, this.settings.rotationHalflife, dt);
        TRAJECTORY_TIMES.forEach((t, k) => {
            const p = vec3.clone(s.position), v = vec3.clone(s.velocity), a = vec3.clone(s.acceleration);
            springCharacterUpdate(p, v, a, goal, this.settings.velocityHalflife, t);
            const r = quat.clone(s.rotation), w = vec3.clone(s.angularVelocity);
            springDamperExactQuat(r, w, goalRotation, this.settings.rotationHalflife, t);
            const from = k === 0 ? s.position : this.predicted[k - 1].translation;
            this.predicted[k] = Transform.fromTranslationRotation(constrain(from, p), r);
        });
    }

    /** The query: the playing frame's pose features, and the predicted trajectory relative to the character. */
    private query(db: Database, frame: number): Float32Array {
        const raw = db.denormalize(db.features(frame), this.scratch.raw);
        const toLocal = quat.conjugate(quat.create(), this.root.rotation);
        const p = vec3.create(), d = vec3.create();
        this.predicted.forEach((t, k) => {
            vec3.transformQuat(p, vec3.subtract(p, t.translation, this.root.translation), toLocal);
            vec3.transformQuat(d, vec3.transformQuat(d, FORWARD, t.rotation), toLocal);
            raw[15 + 2 * k] = p[0];
            raw[16 + 2 * k] = p[2];
            raw[21 + 2 * k] = d[0];
            raw[22 + 2 * k] = d[2];
        });
        return db.normalizeQuery(raw, this.scratch.query);
    }

    private searchIfDue(db: Database, input: MotionInput, dt: number): void {
        this.searchTimer -= dt;
        this.searchInfo.searched = false;
        this.searchInfo.switched = false;
        const info = db.clips[this.clip];
        const atEnd = !info.looping && this.frame >= info.frames - 1 - 1e-3;
        const changed = this.searchedInput === undefined
            || vec3.distance(this.searchedInput[0], input.velocity) > this.settings.forceSearchVelocity
            || Math.abs(wrapAngle(this.searchedInput[1] - this.desiredYaw)) > this.settings.forceSearchTurn;
        // (a hair under zero counts: float steps of the frame time must not skip a frame)
        if (this.searchTimer > 1e-4 && !changed && !atEnd) return;
        this.searchTimer = this.settings.searchInterval;
        this.searchedInput = [vec3.clone(input.velocity), this.desiredYaw];
        const current = this.currentFrame(db);
        const query = this.query(db, current);
        // staying costs what the playing frame does, unless its clip has run out
        const stay = atEnd ? Infinity : distance(query, 0, db.featureRows, current * STRIDE, Infinity);
        const filter: SearchFilter = { ...this.settings.filter, current };
        const limit = atEnd ? Infinity : stay - this.settings.continuingBias;
        this.searchInfo = { searched: true, switched: false, frame: current, cost: stay };
        const found = limit > 0 ? db.search(query, filter, limit) : undefined;
        if (found) {
            this.transition(db, current, found.frame);
            this.searchInfo = { searched: true, switched: true, frame: found.frame, cost: found.cost };
        }
    }

    /** Switch playback to database frame `to`, inertializing from what plays now. */
    private transition(db: Database, from: number, to: number): void {
        const s = this.scratch;
        db.velocities(from, s.sourceLinear, s.sourceAngular);
        db.velocities(to, s.destinationLinear, s.destinationAngular);
        db.pose(to, to, 0, s.destination);
        this.inertializer.transition(this.sampled, s.sourceLinear, s.sourceAngular, s.destination, s.destinationLinear, s.destinationAngular);
        this.clip = db.clipOf(to);
        this.frame = to - db.clips[this.clip].start;
    }

    /** Advance the playhead, moving the character by the root motion, and sample the pose. */
    private play(db: Database, dt: number): void {
        const info = db.clips[this.clip];
        const next = this.frame + dt * db.sampleRate;
        const [moved, turned] = db.rootMotion(this.clip, this.frame, next);
        const step = vec3.transformQuat(vec3.create(), moved, this.root.rotation);
        vec3.add(this.root.translation, this.root.translation, step);
        quat.multiply(this.root.rotation, this.root.rotation, yawRotation(turned));
        quat.normalize(this.root.rotation, this.root.rotation);
        this.frame = info.looping ? remEuclid(next, info.playable) : Math.min(next, info.frames - 1);
        this.samplePose(db);
        this.rootSpeed = [vec3.length(moved) / dt, Math.abs(turned) / dt];
    }

    /** The database's pose at the playhead into `sampled` and `matched`. */
    private samplePose(db: Database): void {
        const info = db.clips[this.clip];
        const a = Math.floor(this.frame);
        const b = Math.min(a + 1, info.frames - 1);
        db.pose(info.start + a, info.start + b, this.frame - a, this.sampled);
        this.matched.copy(this.sampled);
    }

    /** Play the action: its clip on, the character moved by its root path; hand back to the search at its exit frame. */
    private playAction(db: Database, dt: number, constrain: Constrain): void {
        const action = this.playingAction;
        if (!action) return;
        const info = db.clips[action.clip];
        const before = this.root.clone();
        let next = this.frame + dt * db.sampleRate;
        const exit = action.exit === undefined ? undefined : Math.min(action.exit, info.frames - 1);
        if (info.looping && exit === undefined) next = remEuclid(next, info.playable);
        const done = exit !== undefined && next >= exit;
        this.frame = Math.min(next, info.frames - 1);
        const path = action.path;
        if (path.kind === 'warp') {
            const [p, yaw] = path.warp.root(this.frame, db.rootAt(action.clip, this.frame));
            const at = action.collides ? constrain(before.translation, p) : p;
            this.root = Transform.fromTranslationRotation(at, yawRotation(yaw));
        } else {
            const from = this.root.translation;
            const to = vec3.scaleAndAdd(vec3.create(), from, path.velocity, dt);
            to[1] -= 0.5 * path.gravity * dt * dt;
            const allowed = action.collides ? constrain(from, to) : to;
            // blocked: drop the velocity into what blocked it
            if (vec3.distance(allowed, to) > 1e-5) {
                const blocked = blockedDirection(from, to, allowed);
                vec3.scaleAndAdd(path.velocity, path.velocity, blocked, -Math.max(vec3.dot(path.velocity, blocked), 0));
            }
            path.velocity[1] -= path.gravity * dt;
            this.root.translation = vec3.clone(allowed);
        }
        this.samplePose(db);
        const moved = flat(vec3.subtract(vec3.create(), this.root.translation, before.translation));
        this.rootSpeed = [vec3.length(moved) / dt, Math.abs(wrapAngle(yawOf(this.root.rotation) - yawOf(before.rotation))) / dt];
        vec3.scale(this.sim.velocity, moved, 1 / dt);
        if (done) {
            this.groundHeight = this.root.translation[1];
            this.stopAction();
        }
    }

    /** Pull the animated character toward the simulation, then clamp it within reach of it. */
    private synchronize(dt: number): void {
        const s = this.settings;
        const [speed, turnRate] = this.rootSpeed;
        const difference = vec3.subtract(vec3.create(), this.sim.position, this.root.translation);
        const adjustment = damperExact(vec3.create(), difference, s.adjustmentHalflife, dt);
        if (s.adjustByVelocity) clampLengthMax(adjustment, s.maxAdjustmentRatio * speed * dt);
        this.root.translation[0] += adjustment[0];
        this.root.translation[2] += adjustment[2];
        const offset = quat.multiply(quat.create(), this.sim.rotation, quat.conjugate(quat.create(), this.root.rotation));
        const rotationAdjustment = quatToScaledAngleAxis(quatAbs(offset));
        vec3.scale(rotationAdjustment, rotationAdjustment, 1 - negexp((Math.LN2 * dt) / (s.adjustmentHalflife + 1e-5)));
        if (s.adjustByVelocity) clampLengthMax(rotationAdjustment, s.maxAdjustmentRatio * turnRate * dt);
        const turned = quat.multiply(quat.create(), quatFromScaledAngleAxis(rotationAdjustment), this.root.rotation);
        this.root.rotation = yawRotation(yawOf(turned));
        // never farther than the clamps from the simulation
        const away = vec3.subtract(vec3.create(), this.root.translation, this.sim.position);
        const flatAway = flat(away);
        const length = vec3.length(flatAway);
        if (length > s.clampDistance) {
            const t = vec3.scaleAndAdd(vec3.create(), this.sim.position, flatAway, s.clampDistance / length);
            t[1] += away[1];
            this.root.translation = t;
        }
        const simYaw = yawOf(this.sim.rotation);
        const yawGap = wrapAngle(yawOf(this.root.rotation) - simYaw);
        if (Math.abs(yawGap) > s.clampAngle) this.root.rotation = yawRotation(simYaw + Math.sign(yawGap) * s.clampAngle);
    }

    /** Pin planted feet where they touched down (two-joint IK on each leg): the ankle while the heel is down, the ball once it lifts. */
    private lockFeet(db: Database, dt: number): void {
        const contacts = db.contacts(this.currentFrame(db));
        const s = this.settings;
        let moved = false;
        const skeleton = this.display?.skeleton ?? db.skeleton;
        const legs = this.display?.legs ?? this.legs;
        let inverse: Transform | undefined;
        for (let side = 0; side < 2; side++) {
            const leg = legs[side];
            const [upper, middle, foot] = leg.joints;
            if (upper === middle || middle === foot) continue;
            // the model is relative to the character, which stands on the ground
            const ankle = this.outputModel[foot].translation, ball = this.outputModel[leg.ball].translation;
            const animated: FootPose = { ankle: this.root.transformPoint(ankle), ball: this.root.transformPoint(ball), ankleLift: ankle[1] - leg.rest[0], ballLift: ball[1] - leg.rest[1] };
            const target = this.feet[side].updateFoot(animated, contacts[side], s.footUnlockRadius, s.footLockHalflife, dt);
            if (vec3.distance(target, animated.ankle) > 1e-5) {
                inverse ??= this.root.inverse();
                twoJointIK(skeleton, this.output, this.outputModel, upper, middle, foot, inverse.transformPoint(target));
                moved = true;
            }
        }
        // the joints below the feet follow them
        if (moved) this.output.toModel(skeleton, this.outputModel);
    }
}

export { MotionMatcher, defaultMotionMatchingSettings, blockedDirection };
export type { MotionMatchingSettings, MotionInput, Simulation, SearchInfo, RootPath, Action, Constrain };
