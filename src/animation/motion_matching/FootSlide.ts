import { vec3 } from "gl-matrix";
import { Skeleton } from "../Skeleton";
import { Database } from "./Database";
import { wrapAngle, yawOf } from "./Heading";
import { MotionMatcher } from "./MotionMatcher";

/**
 * Foot slide: how far planted feet travel across the ground, and a scripted course of moves to
 * measure it on (straight walks and runs, circles both ways, starts and stops, 180° turns).
 * Rust: `motion_matching::foot_slide`.
 *
 * A foot counts as planted while the frame playing says so (`Database.contacts` of a reference
 * database: the one played, or the same frames with their contacts found by other thresholds, so
 * tuning the contacts foot locking pins on does not move the yardstick). Each frame its slide is
 * whichever of the ankle and the toe moved less across the ground (a foot rolling onto its toes
 * moves its ankle, not its toe), counted only while it was planted the frame before too. The
 * report gives it per second planted, and per plant; and how far the character (the mesh) faces
 * from the simulation (the capsule).
 */

/** What a `FootSlide` measured. */
interface FootSlideReport {
    /** Planted-foot slide (cm per second planted, both feet together). */
    cmPerSecond: number;
    /** Slide per plant (cm). */
    cmPerPlant: number;
    /** Share of the time each foot was planted, on average. */
    planted: number;
    /** Mean and largest angle between the character's facing and the simulation's (degrees). */
    yawGap: number;
    yawGapMax: number;
}

const flatLength = (v: vec3): number => Math.hypot(v[0], v[2]);

/** Collects foot slide while a `MotionMatcher` plays (`sample` after each update). */
class FootSlide {
    /** Ankle and toe joints of each foot on the output skeleton (left, right). */
    private joints: [number, number | undefined][];
    /** Where each planted foot's ankle and toe were last frame (world). */
    private last: ([vec3, vec3] | undefined)[] = [undefined, undefined];
    private plantedTime = [0, 0];
    private slide = [0, 0];
    private plants = [0, 0];
    private time = 0;
    private yawGapSum = 0;
    private yawGapMax = 0;

    /** A meter for the feet named `feet` (left, right) on `skeleton`. */
    constructor(skeleton: Skeleton, feet: [string, string]) {
        this.joints = feet.map((name) => {
            const foot = skeleton.find(name) ?? 0;
            const toe = skeleton.parents.indexOf(foot);
            return [foot, toe < 0 ? undefined : toe];
        });
    }

    /** A meter for `matcher`'s output skeleton: the database's feet, found there by name, and each foot's first child as its toe. */
    public static of(matcher: MotionMatcher, db: Database): FootSlide {
        return new FootSlide(matcher.outputSkeleton(db), [db.skeleton.names[db.roles.feet[0]], db.skeleton.names[db.roles.feet[1]]]);
    }

    /** Account for the frame `matcher` just played, `dt` seconds long, with whether each foot is `planted`. */
    public sample(matcher: MotionMatcher, planted: [boolean, boolean], dt: number): void {
        const character = matcher.character, model = matcher.model;
        this.time += dt;
        const gap = Math.abs(wrapAngle(yawOf(character.rotation) - yawOf(matcher.simulation.rotation))) * 180 / Math.PI;
        this.yawGapSum += gap * dt;
        this.yawGapMax = Math.max(this.yawGapMax, gap);
        for (let side = 0; side < 2; side++) {
            if (!planted[side]) {
                this.last[side] = undefined;
                continue;
            }
            const [ankleJoint, toeJoint] = this.joints[side];
            const ankle = character.transformPoint(model[ankleJoint].translation);
            const toe = toeJoint === undefined ? ankle : character.transformPoint(model[toeJoint].translation);
            const last = this.last[side];
            if (last) {
                this.slide[side] += Math.min(flatLength(vec3.subtract(vec3.create(), ankle, last[0])), flatLength(vec3.subtract(vec3.create(), toe, last[1])));
                this.plantedTime[side] += dt;
            } else {
                this.plants[side]++;
            }
            this.last[side] = [ankle, toe];
        }
    }

    public report(): FootSlideReport {
        const planted = this.plantedTime[0] + this.plantedTime[1];
        const slide = this.slide[0] + this.slide[1];
        const plants = this.plants[0] + this.plants[1];
        return {
            cmPerSecond: planted > 0 ? 100 * slide / planted : 0,
            cmPerPlant: plants > 0 ? 100 * slide / plants : 0,
            planted: this.time > 0 ? planted / (2 * this.time) : 0,
            yawGap: this.time > 0 ? this.yawGapSum / this.time : 0,
            yawGapMax: this.yawGapMax,
        };
    }
}

/** A move of the foot-slide course: the input, by the seconds since it started, for a character starting out facing +Z. */
type Move =
    /** Straight ahead (+Z) at `speed` m/s. */
    | { kind: "straight", speed: number }
    /** Round a circle of `radius` m at `speed` m/s; `left` turns left (counter-clockwise from above). */
    | { kind: "circle", radius: number, speed: number, left: boolean }
    /** Stand `still` seconds, go ahead at `speed` for `go` seconds, again and again. */
    | { kind: "startStop", speed: number, go: number, still: number }
    /** Go ahead at `speed` for `leg` seconds, then back the way it came, again and again. */
    | { kind: "reverse", speed: number, leg: number };

/** The desired velocity `t` seconds into `move`. */
function moveVelocity(move: Move, t: number): vec3 {
    switch (move.kind) {
        case "straight": return vec3.fromValues(0, 0, move.speed);
        case "circle": {
            // the heading turns at speed / radius (yaw grows to the left: +Z toward +X)
            const yaw = move.speed / move.radius * t * (move.left ? 1 : -1);
            return vec3.fromValues(Math.sin(yaw) * move.speed, 0, Math.cos(yaw) * move.speed);
        }
        case "startStop": return vec3.fromValues(0, 0, t % (move.go + move.still) < move.still ? 0 : move.speed);
        case "reverse": return vec3.fromValues(0, 0, Math.floor(t / move.leg) % 2 === 0 ? move.speed : -move.speed);
    }
}

/** One run of the course: a move, at a gait (`run` picks the run tags), measured for `seconds` after `settle` seconds of it. */
interface Scenario {
    name: string;
    move: Move;
    run: boolean;
    settle: number;
    seconds: number;
}

/**
 * The course, at walk and run paces of 2 and 5 m/s: straight walk and run, run circles of 2,
 * 3.5 and 5 m and walk circles of 1.5 and 3 m both ways, starts and stops at a run, and 180°
 * turns at a run.
 */
function footSlideCourse(walk: number, run: number): Scenario[] {
    const circle = (name: string, radius: number, speed: number, left: boolean, run: boolean): Scenario => ({ name, move: { kind: "circle", radius, speed, left }, run, settle: 3, seconds: 12 });
    return [
        { name: "walk straight", move: { kind: "straight", speed: walk }, run: false, settle: 2, seconds: 8 },
        { name: "run straight", move: { kind: "straight", speed: run }, run: true, settle: 2, seconds: 8 },
        circle("run circle 2 m left", 2, run, true, true),
        circle("run circle 2 m right", 2, run, false, true),
        circle("run circle 3.5 m left", 3.5, run, true, true),
        circle("run circle 3.5 m right", 3.5, run, false, true),
        circle("run circle 5 m left", 5, run, true, true),
        circle("run circle 5 m right", 5, run, false, true),
        circle("walk circle 1.5 m left", 1.5, walk, true, false),
        circle("walk circle 1.5 m right", 1.5, walk, false, false),
        circle("walk circle 3 m left", 3, walk, true, false),
        circle("walk circle 3 m right", 3, walk, false, false),
        { name: "start and stop", move: { kind: "startStop", speed: run, go: 3, still: 2.5 }, run: true, settle: 1, seconds: 22 },
        { name: "180 turn", move: { kind: "reverse", speed: run, leg: 3 }, run: true, settle: 3, seconds: 24 },
    ];
}

/**
 * Play `scenario` on `matcher` (standing at the origin facing +Z, its filter set for the
 * scenario's gait) at a fixed 60 Hz step, and measure it, the feet planted where `reference`
 * (`db`'s frames) has its contacts. Within a few seconds a scenario settles into a cycle of the
 * same frames, whatever it started from: a change of tuning that moves one scenario a lot may
 * only have sent it round another cycle, so judge one over the whole course.
 */
function measureFootSlide(db: Database, reference: Database, matcher: MotionMatcher, scenario: Scenario): FootSlideReport {
    const dt = 1 / 60;
    const meter = FootSlide.of(matcher, db);
    const frames = Math.round((scenario.settle + scenario.seconds) / dt);
    for (let f = 0; f < frames; f++) {
        const t = f * dt;
        matcher.update(db, { velocity: moveVelocity(scenario.move, t) }, dt);
        if (t >= scenario.settle) meter.sample(matcher, reference.contacts(matcher.currentFrame(db)), dt);
    }
    return meter.report();
}

export { FootSlide, moveVelocity, footSlideCourse, measureFootSlide };
export type { FootSlideReport, Move, Scenario };
