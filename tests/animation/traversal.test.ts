// rust/kansei-core/src/animation/motion_matching/traversal/tests.rs, ported.
import { quat, vec3 } from "gl-matrix";
import { assert, assertEq, close, test } from "../harness";
import { CollisionWorld, Obb } from "../../src/collision/CollisionWorld";
import { Clip } from "../../src/animation/Clip";
import { Pose } from "../../src/animation/Pose";
import { Skeleton } from "../../src/animation/Skeleton";
import { Transform } from "../../src/animation/Transform";
import {
    ACTION_TAG, ActionClip, ActionKind, Database, DatabaseBuilder, MotionMatcher, crosses, defaultMotionMatchingSettings, findJointRoles, yawOf,
} from "../../src/animation/motion_matching/index";
import {
    CharacterController, CharacterState, Refusal, defaultDetectionSettings, defaultTraversalRules, detectObstacle, isRefusal, planTraversal, standsAt, traversalKind,
} from "../../src/animation/motion_matching/Traversal";

const RATE = 30;
const HANDS: [number, number] = [2, 9];
const ALL = 0xffffffff;
const v = (x: number, y: number, z: number) => vec3.fromValues(x, y, z);
const Z = v(0, 0, 1);

/** A root, hips 1 m up, a hand each side, two legs. */
function skeleton(): Skeleton {
    const t = (x: number, y: number) => Transform.fromTranslationRotation(v(x, y, 0), quat.create());
    return new Skeleton(
        ["root", "hips", "hand_l", "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r", "hand_r"],
        [null, 0, 1, 1, 3, 4, 1, 6, 7, 1],
        [t(0, 0), t(0, 1), t(0.25, 0), t(0.1, 0), t(0, -0.45), t(0, -0.45), t(-0.1, 0), t(0, -0.45), t(0, -0.45), t(-0.25, 0)],
    );
}

/** A clip from per-frame root (position), hips height and optional world hand positions. */
function clip(name: string, frames: number, root: (f: number) => vec3, hips: (f: number) => number, hands: (f: number) => [vec3, vec3] | undefined, swing: number): Clip {
    const s = skeleton();
    const poses: Pose[] = [];
    for (let f = 0; f < frames; f++) {
        const pose = Pose.rest(s);
        const r = root(f);
        const h = hips(f);
        pose.local[0] = Transform.fromTranslationRotation(r, quat.create());
        pose.local[1].translation = v(0, h, 0);
        const hand = hands(f);
        if (hand) {
            pose.local[2].translation = v(hand[0][0] - r[0], hand[0][1] - r[1] - h, hand[0][2] - r[2]);
            pose.local[9].translation = v(hand[1][0] - r[0], hand[1][1] - r[1] - h, hand[1][2] - r[2]);
        }
        const a = Math.sin(2 * Math.PI * f / RATE) * swing;
        pose.local[3].rotation = quat.setAxisAngle(quat.create(), [1, 0, 0], -a);
        pose.local[6].rotation = quat.setAxisAngle(quat.create(), [1, 0, 0], a);
        poses.push(pose);
    }
    return Clip.fromPoses(name, RATE, poses);
}

function smooth(x: number): number {
    const t = Math.min(Math.max(x, 0), 1);
    return t * t * (3 - 2 * t);
}

/**
 * Walking into a 1 m block at 2 m/s: hands on its edge (z = 0) from frame 22, up from frame 26 to
 * 32, over the edge, standing again by frame 40 and walking on at 1 m/s.
 */
function mantle(): Clip {
    return mantleStandingUpAt("mantle", 32);
}

/** `mantle`, the hips back up to standing from frame `up` (the later, the farther it walks first). */
function mantleStandingUpAt(name: string, up: number): Clip {
    const z = (f: number) => f <= 25 ? -2 + f * 2 / 30 : f <= 35 ? -2 + 25 * 2 / 30 + (f - 25) * (0.2 + 2 - 25 * 2 / 30) / 10 : 0.2 + (f - 35) / 30;
    const y = (f: number) => smooth((f - 26) / 6);
    const hips = (f: number) => f < 26 ? 1 : f < up ? 0.7 : 0.7 + 0.3 * smooth((f - up) / 8);
    return clip(name, 71, (f) => v(0, y(f), z(f)), hips, (f) => f >= 22 && f <= 34 ? [v(0.2, 1.03, 0.05), v(-0.2, 1.03, 0.05)] : undefined, 0.3);
}

/** Running at 5 m/s over a 1 m hurdle: the root up from frame 17 to 21, on top to 24, down by 28. */
function hurdle(name: string = "hurdle"): Clip {
    const y = (f: number) => smooth((f - 17) / 4) - smooth((f - 24) / 4);
    return clip(name, 61, (f) => v(0, y(f), -3.6 + f / 6), () => 1, () => undefined, 0.4);
}

/** Dropping 3 m (landing at frame 15), then walking. */
function land(): Clip {
    return clip("land", 50, (f) => v(0, Math.max(3 - f * 0.2, 0), f * 0.05), () => 1, () => undefined, 0.2);
}

/** Dropping `drop` metres (landing at frame `drop / 0.2`), then walking. */
function landFrom(name: string, drop: number): Clip {
    return clip(name, Math.trunc(Math.fround(drop / 0.2)) + 35, (f) => v(0, Math.max(drop - f * 0.2, 0), f * 0.05), () => 1, () => undefined, 0.2);
}

/** A run-up at `speed` m/s, the take-off at frame 10, a 1 m apex, then falling on past the take-off height. */
function jumpClip(name: string, speed: number): Clip {
    const up = Math.sqrt(2 * 9.81);
    const y = (f: number) => {
        const t = (f - 10) / RATE;
        return t <= 0 ? 0 : up * t - 0.5 * 9.81 * t * t;
    };
    return clip(name, 61, (f) => v(0, y(f), (f - 10) * speed / RATE), (f) => f < 10 ? 0.9 : 1, () => undefined, 0.1 + 0.2 * Math.min(speed, 1));
}

const fall = () => clip("fall", 31, () => v(0, 0, 0), () => 1, () => undefined, 0.6);
const idle = () => clip("idle", 61, () => v(0, 0, 0), () => 1, () => undefined, 0);
const walk = () => clip("walk", 61, (f) => v(0, 0, f * 1.5 / 30), () => 1, () => undefined, 0.4);

function database(): [Database, ActionClip[]] {
    return databaseWith([[mantle(), ActionKind.Mantle], [hurdle(), ActionKind.Hurdle], [land(), ActionKind.Land], [fall(), ActionKind.Fall]]);
}

/** Idle and walk loops, then `actions` (their table in the same order). */
function databaseWith(actions: [Clip, ActionKind][]): [Database, ActionClip[]] {
    const s = skeleton();
    const b = new DatabaseBuilder(s, findJointRoles(s, "root", "hips", "foot_l", "foot_r"), RATE);
    b.addClip(idle(), true, 1);
    b.addClip(walk(), true, 2);
    for (const [c] of actions) b.addClip(c, c.name === "fall", ACTION_TAG);
    const db = b.build();
    return [db, actions.map(([, kind], i) => ActionClip.analyze(db, 2 + i, kind, HANDS)!)];
}

test("analysis reads heights, ledges and phases from the animation", () => {
    const [db, table] = database();
    const m = table[0];
    assert(Math.abs(m.height - 1) < 1e-3, `${m.height}`);
    assertEq([m.rise, m.onTop, m.anchor], [26, 32, 22]);
    // the planted hands at z = 0.05 put the ledge just in front of them, on the top
    assert(close(m.ledge, [0, 1, 0], 0.02), `${m.ledge}`);
    assert(close(m.forward, Z, 1e-5));
    // hands back to the search once the hips stand again (0.9 m from frame 37)
    assert(m.exit >= 37 && m.exit <= 41, `${m.exit}`);
    assert(Math.abs(m.speedAt(db, 5) - 2) < 0.05);
    assert(Math.abs(m.distanceAt(db, 0) - 2) < 0.02);

    const h = table[1];
    assert(Math.abs(h.height - 1) < 1e-3 && crosses(h.kind));
    assert(h.rise <= 17 && h.onTop >= 20 && h.offTop >= 24 && h.down >= 27, JSON.stringify(h));
    assert(h.span > 0.3 && h.span < 1, `${h.span}`);
    assert(h.exit > h.down);

    const l = table[2];
    assertEq(l.anchor, 15);
    assert(Math.abs(l.height - 3) < 1e-3);
    // clips that never leave the ground are no traversal
    assert(ActionClip.analyze(db, 1, ActionKind.Mantle, HANDS) === undefined);
});

/** A box from its corners. */
const box = (min: [number, number, number], max: [number, number, number]) => Obb.fromMinMax(v(...min), v(...max));

function floor(): CollisionWorld {
    const w = new CollisionWorld();
    w.addBox(box([-100, -1, -100], [100, 0, 100]));
    return w;
}

/**
 * A floor, and boxes: a thin one (0.3 deep, 0.8 high), a deep one (2 m, 1.3 high), a wall
 * (2.4 m), a narrow post and a thin box with a wall right behind it.
 */
function course(): CollisionWorld {
    const w = floor();
    w.addBox(box([-1, 0, 0], [1, 0.8, 0.3]));
    w.addBox(box([9, 0, 0], [11, 1.3, 2]));
    w.addBox(box([19, 0, 0], [21, 2.4, 3]));
    w.addBox(box([29.9, 0, 0], [30.1, 1, 0.3]));
    w.addBox(box([39, 0, 0], [41, 0.8, 0.3]));
    w.addBox(box([39, 0, 0.5], [41, 3, 0.8]));
    return w;
}

test("obstacles are measured from the geometry", () => {
    const w = course();
    const s = defaultDetectionSettings();
    const thin = detectObstacle(w, v(0, 0, -3), Z, 5, s)!;
    assert(close(thin.ledge, [0, 0.8, 0], 0.02), `${thin.ledge}`);
    assert(close(thin.normal, [0, 0, -1], 1e-4));
    assert(Math.abs(thin.height - 0.8) < 1e-4 && Math.abs(thin.distance - 3) < 0.02);
    assert(Math.abs(thin.depth! - 0.3) <= 0.06, `${thin.depth}`);
    assertEq(thin.backFloor, 0);
    assert(thin.halfWidth >= 0.9, `${thin.halfWidth}`);

    const deep = detectObstacle(w, v(10, 0, -2), Z, 5, s)!;
    assert(Math.abs(deep.height - 1.3) < 1e-4 && deep.depth === undefined && deep.backFloor === undefined);
    const wall = detectObstacle(w, v(20, 0, -2), Z, 5, s)!;
    assert(Math.abs(wall.height - 2.4) < 1e-4);
    // at an angle: the face's normal, not the approach
    const angled = detectObstacle(w, v(9, 0, -2), v(0.4, 0, 1), 5, s)!;
    assert(close(angled.normal, [0, 0, -1], 1e-4) && Math.abs(angled.distance - 2) < 0.02);
    assert(detectObstacle(w, v(0, 0, -3), v(0, 0, -1), 5, s) === undefined, "nothing behind");
    assert(detectObstacle(w, v(0, 0, -9), Z, 5, s) === undefined, "out of reach");

    const rules = defaultTraversalRules();
    const kind = (feet: vec3) => traversalKind(w, detectObstacle(w, feet, Z, 5, s)!, feet, rules, ALL);
    assertEq(kind(v(0, 0, -3)), ActionKind.Hurdle);
    assertEq(kind(v(10, 0, -2)), ActionKind.Mantle);
    assertEq(kind(v(20, 0, -2)), ActionKind.Climb);
    assertEq(kind(v(30, 0, -2)), Refusal.TooNarrow);
    assertEq(kind(v(40, 0, -2)), Refusal.NoRoom);
});

function controller(db: Database, table: ActionClip[], at: vec3): CharacterController {
    const matcher = new MotionMatcher(db, defaultMotionMatchingSettings(), at, 0);
    matcher.settings.footLock = false;
    return new CharacterController(matcher, table);
}

/** Rust's `(seconds / dt) as usize` frames, in f32. */
const framesOf = (seconds: number) => Math.trunc(Math.fround(Math.fround(seconds) / Math.fround(1 / 60)));

function run(c: CharacterController, db: Database, w: CollisionWorld, velocity: vec3, seconds: number): void {
    for (let i = 0; i < framesOf(seconds); i++) c.update(db, w, { velocity }, 1 / 60);
}

const kindOf = (s: CharacterState) => s.kind;
const position = (c: CharacterController) => c.matcher.character.translation;

test("walls stop the character and its trajectory", () => {
    const [db, table] = database();
    const w = floor();
    w.addBox(box([-2, 0, 0], [2, 2, 1]));
    const c = controller(db, table, v(0, 0, -3));
    run(c, db, w, v(0, 0, 1.5), 4);
    const z = position(c)[2];
    assert(z < -0.29 && z > -0.6, `stopped at the wall: ${z}`);
    assert(c.matcher.trajectory.every((t) => t.translation[2] < -0.29));
    assertEq(kindOf(c.state), "grounded");
});

test("the character mantles up a block and stands on it", () => {
    const [db, table] = database();
    const w = floor();
    w.addBox(box([-2, 0, 0], [2, 1.3, 3]));
    const c = controller(db, table, v(0.3, 0, -2.5));
    run(c, db, w, v(0, 0, 1.5), 0.4);
    assertEq(c.traverse(db, w), ActionKind.Mantle);
    assertEq(c.state, { kind: "traversing", action: ActionKind.Mantle });
    // mid-way: the warp puts the clip's ledge on the real one at the anchor frame
    const path = c.matcher.action!.path;
    assert(path.kind === "warp");
    const warp = path.warp;
    const m = c.actions.find((a) => a.kind === ActionKind.Mantle)!;
    const placed = warp.to.apply(v(m.ledge[0], 0, m.ledge[2]));
    const ledge = c.lastObstacle!.ledge;
    assert(Math.abs(placed[0] - ledge[0]) < 1e-4 && Math.abs(placed[2] - ledge[2]) < 1e-4, `${placed} vs ${ledge}`);
    const [, yaw] = warp.root(m.onTop, db.rootAt(m.clip, m.onTop));
    assert(Math.abs(yaw) < 1e-3, `faces into the block: ${yaw}`);
    const [top] = warp.root(m.onTop + 2, db.rootAt(m.clip, m.onTop + 2));
    assert(Math.abs(top[1] - 1.3) < 1e-3, `lifted to the real top: ${top}`);
    run(c, db, w, v(0, 0, 1.5), 3);
    const p = position(c);
    assertEq(kindOf(c.state), "grounded");
    assert(Math.abs(p[1] - 1.3) < 0.02 && p[2] > 0.2, `on the block: ${p}`);
});

test("the character hurdles a thin box and runs on", () => {
    const [db, table] = database();
    const w = floor();
    w.addBox(box([-2, 0, 0], [2, 0.7, 0.3]));
    const c = controller(db, table, v(0, 0, -2.4));
    run(c, db, w, v(0, 0, 1.5), 0.5);
    assertEq(c.traverse(db, w), ActionKind.Hurdle);
    let highest = 0;
    for (let i = 0; i < 120; i++) {
        c.update(db, w, { velocity: v(0, 0, 1.5) }, 1 / 60);
        const p = position(c);
        highest = Math.max(highest, p[1]);
        // over the box, never through it
        if (p[2] > 0 && p[2] < 0.3) assert(p[1] > 0.6, `over the box at ${p}`);
    }
    const p = position(c);
    assert(Math.abs(highest - 0.7) < 0.05, `up to the real top: ${highest}`);
    assert(p[2] > 0.6 && Math.abs(p[1]) < 0.02, `beyond it on the floor: ${p}`);
    assertEq(kindOf(c.state), "grounded");
});

test("walking off a ledge falls and lands", () => {
    const [db, table] = database();
    const w = floor();
    w.addBox(box([-2, 0, -4], [2, 1.5, 0]));
    const c = controller(db, table, v(0, 1.5, -1.5));
    c.matcher.setGround(1.5);
    const states: string[] = [];
    for (let i = 0; i < 240; i++) {
        c.update(db, w, { velocity: v(0, 0, 1.5) }, 1 / 60);
        if (states[states.length - 1] !== c.state.kind) states.push(c.state.kind);
    }
    const p = position(c);
    assert(Math.abs(p[1]) < 0.02 && p[2] > 0.3, `down on the floor: ${p}`);
    assertEq(states, ["grounded", "falling", "landing", "grounded"]);
});

test("a request made early traverses once in reach", () => {
    const [db, table] = database();
    const w = floor();
    w.addBox(box([-2, 0, 0], [2, 1.3, 3]));
    const c = controller(db, table, v(0, 0, -6));
    run(c, db, w, v(0, 0, 1.5), 0.3);
    // far out of reach: refused now, but kept trying
    assert(isRefusal(c.requestTraverse(db, w, 3)));
    let started = false;
    for (let i = 0; i < 180; i++) {
        c.update(db, w, { velocity: v(0, 0, 1.5) }, 1 / 60);
        started ||= c.state.kind === "traversing" && c.state.action === ActionKind.Mantle;
    }
    assert(started, `mantled once in reach: ${c.lastResult}`);
});

test("a mantle is chosen by where it leaves the character", () => {
    // one mantle stands up by the edge, the other walks on a metre before handing over
    const [db, table] = databaseWith([[mantle(), ActionKind.Mantle], [mantleStandingUpAt("mantle_on", 50), ActionKind.Mantle]]);
    const [near, far] = table;
    const pastLedge = (c: ActionClip) => vec3.dot(vec3.subtract(vec3.create(), db.rootAt(c.clip, c.exit)[0], c.ledge), c.forward);
    assert(pastLedge(near) < 0.5 && pastLedge(far) > 0.8, `${pastLedge(near)} ${pastLedge(far)}`);
    // a block with a wall on it 1.1 m back from its edge: room to stand by the edge, not beyond
    const w = floor();
    w.addBox(box([-2, 0, 0], [2, 1.3, 3]));
    w.addBox(box([-2, 1.3, 1.1], [2, 3.3, 3]));
    const rules = defaultTraversalRules();
    const feet = v(0, 0, -2);
    const obstacle = detectObstacle(w, feet, Z, 5, defaultDetectionSettings())!;
    assertEq(traversalKind(w, obstacle, feet, rules, ALL), ActionKind.Mantle);
    const plan = (t: ActionClip[], world: CollisionWorld) => planTraversal(db, t, ActionKind.Mantle, obstacle, feet, 0, 0, db.clips[0].start, rules, (f) => standsAt(world, f, rules, ALL));
    assertEq(plan([far], w), Refusal.NoRoom);
    const chosen = plan(table, w);
    assert(!isRefusal(chosen) && chosen.clip === near.clip, `${JSON.stringify(chosen)}`);
    // without the wall, either fits
    assert(!isRefusal(plan([far], course())));
});

test("pushing the character out of a wall gives it no speed", () => {
    const [db, table] = database();
    const w = floor();
    w.addBox(box([-2, 0, 0], [2, 2, 1]));
    // standing 0.2 m into the wall (as an action can leave it)
    const c = controller(db, table, v(0, 0, -0.1));
    run(c, db, w, v(0, 0, 0), 1);
    const z = position(c)[2];
    assert(z < -0.29 && z > -0.45, `out of the wall, and no farther: ${z}`);
    assert(vec3.length(c.matcher.simulation.velocity) < 0.1, `${c.matcher.simulation.velocity}`);
});

/** Standing and walking jumps (and a running one if `withRun`), light and heavy landings, the fall loop and a hurdle. */
function jumpDatabase(withRun: boolean): [Database, ActionClip[]] {
    const actions: [Clip, ActionKind][] = [
        [jumpClip("jump_stand", 0), ActionKind.Jump],
        [jumpClip("jump_walk", 1.5), ActionKind.Jump],
        [land(), ActionKind.Land],
        [landFrom("land_heavy", 6), ActionKind.Land],
        [fall(), ActionKind.Fall],
        [hurdle(), ActionKind.Hurdle],
    ];
    if (withRun) actions.push([jumpClip("jump_run", 4), ActionKind.Jump]);
    return databaseWith(actions);
}

/**
 * Runs `c` for `seconds`, walking along +Z, and returns each state and clip it went through (an
 * entry whenever either changes), and the highest point it reached.
 */
function fly(c: CharacterController, db: Database, w: CollisionWorld, seconds: number): [[string, string][], number] {
    const seen: [string, string][] = [];
    let highest = -Infinity;
    for (let i = 0; i < Math.trunc(seconds * 60); i++) {
        c.update(db, w, { velocity: v(0, 0, 1.5) }, 1 / 60);
        highest = Math.max(highest, position(c)[1]);
        const state = c.state.kind;
        const name = db.clips[c.matcher.playing()[0]].name;
        const last = seen[seen.length - 1];
        if (!last || last[0] !== state || last[1] !== name) seen.push([state, name]);
    }
    return [seen, highest];
}

test("analysis reads the take-off and apex of a jump", () => {
    const [db, table] = jumpDatabase(false);
    const w = table.find((a) => db.clips[a.clip].name === "jump_walk")!;
    assertEq(w.kind, ActionKind.Jump);
    assertEq(w.rise, 10);
    assert(Math.abs(w.anchor - 23.5) <= 0.5, `apex at ${w.anchor}`);
    assert(Math.abs(w.height - 1) < 0.01, `${w.height}`);
    assert(Math.abs(w.speedAt(db, 5) - 1.5) < 0.01);
});

test("a jump takes off, flies as high as asked and lands light", () => {
    const [db, table] = jumpDatabase(false);
    const w = floor();
    const c = controller(db, table, v(0, 0, 0));
    c.jumpHeight = 0.8;
    run(c, db, w, v(0, 0, 1.5), 1);
    assertEq(c.jump(db), ActionKind.Jump);
    // the run-up that goes with walking
    assertEq(db.clips[c.matcher.action!.clip].name, "jump_walk");
    const [seen, highest] = fly(c, db, w, 3);
    assert(Math.abs(highest - 0.8) < 0.03, `as high as asked: ${highest}`);
    const states = seen.map(([s]) => s).filter((s, i, all) => i === 0 || all[i - 1] !== s);
    assertEq(states, ["jumping", "falling", "landing", "grounded"]);
    assert(seen.some(([s, n]) => s === "landing" && n === "land"), `a light landing for a 0.8 m fall: ${JSON.stringify(seen)}`);
    const p = position(c);
    assert(Math.abs(p[1]) < 0.02 && p[2] > 1.5, `down and on: ${p}`);
});

test("a running jump lands on top of a box", () => {
    const [db, all] = jumpDatabase(true);
    const table = all.filter((a) => a.kind !== ActionKind.Jump || db.clips[a.clip].name === "jump_run");
    // 0.5 m high (more than a step), its front 2.8 m ahead
    const w = floor();
    w.addBox(box([-2, 0, 2.8], [2, 0.5, 20]));
    const c = controller(db, table, v(0, 0, 0));
    assertEq(c.jump(db), ActionKind.Jump);
    const [seen] = fly(c, db, w, 3);
    assert(seen.some(([s]) => s === "landing"), JSON.stringify(seen));
    const p = position(c);
    assertEq(kindOf(c.state), "grounded");
    assert(Math.abs(p[1] - 0.5) < 0.02 && p[2] > 2.8, `on the box: ${p}`);
});

test("a wall stops a jump", () => {
    const [db, table] = jumpDatabase(false);
    const w = floor();
    w.addBox(box([-2, 0, 1], [2, 3, 2]));
    const c = controller(db, table, v(0, 0, 0));
    run(c, db, w, v(0, 0, 1.5), 0.2);
    assertEq(c.jump(db), ActionKind.Jump);
    for (let i = 0; i < 180; i++) {
        c.update(db, w, { velocity: v(0, 0, 1.5) }, 1 / 60);
        const z = position(c)[2];
        assert(z < 0.72, `kept out of the wall: ${z}`);
    }
    assertEq(kindOf(c.state), "grounded");
    assert(Math.abs(position(c)[1]) < 0.02);
});

test("a jump off a high platform goes on with the fall loop and lands heavy", () => {
    const [db, table] = jumpDatabase(false);
    const w = floor();
    w.addBox(box([-5, 0, -5], [5, 10, 0]));
    const c = controller(db, table, v(0, 10, -2.6));
    c.matcher.setGround(10);
    run(c, db, w, v(0, 0, 1.5), 1);
    assertEq(c.jump(db), ActionKind.Jump);
    const [seen] = fly(c, db, w, 4);
    // off the edge in the air; the jump clip runs out before the ground, the fall loop goes on
    assertEq(seen.slice(0, 4), [["jumping", "jump_walk"], ["falling", "jump_walk"], ["falling", "fall"], ["landing", "land_heavy"]]);
    assertEq(kindOf(c.state), "grounded");
    assert(Math.abs(position(c)[1]) < 0.02);
});

test("space jumps unless there is something to traverse", () => {
    const [db, table] = jumpDatabase(false);
    // open floor: a jump
    let w = floor();
    let c = controller(db, [...table], v(0, 0, 0));
    run(c, db, w, v(0, 0, 1.5), 0.5);
    assertEq(c.requestTraverseOrJump(db, w, 1), ActionKind.Jump);
    // a thin box ahead: a hurdle
    w = floor();
    w.addBox(box([-2, 0, 0], [2, 0.7, 0.3]));
    c = controller(db, [...table], v(0, 0, -2.4));
    run(c, db, w, v(0, 0, 1.5), 0.5);
    assertEq(c.requestTraverseOrJump(db, w, 1), ActionKind.Hurdle);
    // the end of a beam: too narrow to traverse, so a jump
    w = floor();
    w.addBox(box([-0.175, 0, 1], [0.175, 0.6, 8]));
    c = controller(db, [...table], v(0, 0, -0.8));
    run(c, db, w, v(0, 0, 1.5), 0.3);
    assertEq(c.requestTraverseOrJump(db, w, 1), ActionKind.Jump);
    assertEq(c.lastObstacle !== undefined && c.lastObstacle.halfWidth < 0.3, true);
});

/**
 * A hurdle, a vault, a mantle and a climb (the synthetic hurdle and mantle, lifted by the warp),
 * the fall loop, a landing and a jump: every clip's run-up starts well back from its ledge (the
 * mantle 2 m, reaching 0.67 m by its `lastEntry`), like captured ones.
 */
function contactDatabase(): [Database, ActionClip[]] {
    return databaseWith([
        [hurdle(), ActionKind.Hurdle],
        [hurdle("vault"), ActionKind.Vault],
        [mantle(), ActionKind.Mantle],
        [mantleStandingUpAt("climb", 32), ActionKind.Climb],
        [fall(), ActionKind.Fall],
        [land(), ActionKind.Land],
        [jumpClip("jump_walk", 1.5), ActionKind.Jump],
    ]);
}

/** A box 20 m wide whose front face is at z = 0, `height` high and `depth` deep, on the floor. */
function block(height: number, depth: number): CollisionWorld {
    const w = floor();
    w.addBox(box([-10, 0, 0], [10, height, depth]));
    return w;
}

/** The boxes each kind of traversal is for. */
const BLOCKS: [ActionKind, number, number][] = [[ActionKind.Hurdle, 0.8, 0.3], [ActionKind.Vault, 1, 0.9], [ActionKind.Mantle, 1.3, 3], [ActionKind.Climb, 2.4, 3]];

/** Moves `c` under `velocity` until `stop` says so (or `seconds` run out); true if it stopped. */
function stepUntil(c: CharacterController, db: Database, w: CollisionWorld, velocity: vec3, seconds: number, stop: (c: CharacterController) => boolean): boolean {
    for (let i = 0; i < Math.trunc(Math.fround(seconds * 60)); i++) {
        c.update(db, w, { velocity }, 1 / 60);
        if (stop(c)) return true;
    }
    return false;
}

/**
 * Plays out a traversal started just now, the input still pushing along `velocity`: the most
 * it slid backward (away from the face at z = 0) in a frame, and where it was once back on its feet.
 */
function playOut(c: CharacterController, db: Database, w: CollisionWorld, velocity: vec3): [number, vec3] {
    let last = vec3.clone(position(c));
    let back = 0;
    stepUntil(c, db, w, velocity, 4, (c) => {
        const p = vec3.clone(position(c));
        back = Math.max(back, last[2] - p[2]);
        last = p;
        return c.state.kind === "grounded";
    });
    return [back, last];
}

const name = (kind: ActionKind) => ActionKind[kind];

test("pressed against an obstacle of any height it traverses", () => {
    const [db, table] = contactDatabase();
    for (const [kind, height, depth] of BLOCKS) {
        const w = block(height, depth);
        const c = controller(db, [...table], v(0, 0, -3));
        // walked into it and still pushing
        const forward = v(0, 0, 1.5);
        run(c, db, w, forward, 3);
        const z = position(c)[2];
        assert(z > -0.35 && z < -0.25, `${name(kind)}: against it at ${z}`);
        assertEq(c.requestTraverseOrJump(db, w, 1), kind, `${name(kind)}: ${JSON.stringify(c.lastObstacle)}`);
        const [back, end] = playOut(c, db, w, forward);
        // the clip starts late rather than sliding the character back to its run-up
        assert(back < 0.01, `${name(kind)}: slid back ${back} m in a frame`);
        if (crosses(kind)) assert(end[2] > depth && Math.abs(end[1]) < 0.02, `${name(kind)}: over it, on the floor: ${end}`);
        else assert(end[2] > 0.1 && Math.abs(end[1] - height) < 0.02, `${name(kind)}: on top: ${end}`);
    }
});

test("running into an obstacle it traverses the moment it meets it", () => {
    const [db, table] = contactDatabase();
    for (const [kind, height, depth] of BLOCKS) {
        const w = block(height, depth);
        const c = controller(db, [...table], v(0, 0, -6));
        const forward = v(0, 0, 4);
        // Space as it meets the face
        assert(stepUntil(c, db, w, forward, 3, (c) => position(c)[2] > -0.36), `${name(kind)}: never reached it`);
        const result = c.requestTraverseOrJump(db, w, 1);
        const started = result === kind || stepUntil(c, db, w, forward, 0.2, (c) => c.state.kind === "traversing" && c.state.action === kind);
        assert(started, `${name(kind)}: ${result}, ${c.lastResult}`);
    }
});

test("pushed along an obstacle at an angle it traverses rather than jumps", () => {
    const [db, table] = contactDatabase();
    for (const [kind, height, depth] of BLOCKS) {
        for (const degrees of [30, 50]) {
            const w = block(height, depth);
            const c = controller(db, [...table], v(0, 0, -3));
            const a = Math.fround(degrees * Math.fround(Math.PI / 180));
            const heading = v(Math.sin(a) * 1.5, 0, Math.cos(a) * 1.5);
            assert(stepUntil(c, db, w, heading, 4, (c) => position(c)[2] > -0.36));
            run(c, db, w, heading, 0.5);
            // sliding along the face: the wall leaves it no speed into it
            const vel = c.matcher.simulation.velocity;
            assert(Math.abs(vel[2]) < 0.2 && vel[0] > 0.2, `${name(kind)} at ${degrees}: sliding along it ${vel}`);
            assertEq(c.requestTraverseOrJump(db, w, 1), kind, `${name(kind)} at ${degrees}: ${JSON.stringify(c.lastObstacle)}`);
            // squared up to it
            stepUntil(c, db, w, heading, 0.4, () => false);
            const yaw = yawOf(c.matcher.character.rotation);
            assert(Math.abs(yaw) < 0.05, `${name(kind)} at ${degrees}: facing into it, ${yaw}`);
        }
    }
});

test("space pressed as a landing ends runs once it is back on its feet", () => {
    const [db, table] = contactDatabase();
    const w = floor();
    const forward = v(0, 0, 1.5);
    const jumping = () => {
        const c = controller(db, [...table], v(0, 0, 0));
        run(c, db, w, forward, 0.5);
        assertEq(c.jump(db), ActionKind.Jump);
        return c;
    };
    // the frame (after the jump) the landing ends
    const states: string[] = [];
    stepUntil(jumping(), db, w, forward, 4, (c) => {
        states.push(c.state.kind);
        return false;
    });
    const end = states.lastIndexOf("landing");
    assert(end >= 0, "it lands");
    // whether a Space pressed this many frames before then jumps again once it has landed
    const jumpAgain = (early: number) => {
        const c = jumping();
        for (let i = 0; i <= end - early; i++) c.update(db, w, { velocity: forward }, 1 / 60);
        assertEq(kindOf(c.state), "landing");
        assertEq(c.requestTraverseOrJump(db, w, 1), Refusal.Busy);
        return stepUntil(c, db, w, forward, 0.5, (c) => c.state.kind === "jumping");
    };
    assert(jumpAgain(6), "pressed 0.1 s before the landing ends: jumps as it ends");
    assert(!jumpAgain(24), "pressed 0.4 s before: dropped");
});

test("from right against an obstacle a clip starts late and lands its ledge just past the edge", () => {
    const [db, table] = contactDatabase();
    const w = block(1.3, 3);
    const rules = defaultTraversalRules();
    const m = table.find((a) => a.kind === ActionKind.Mantle)!;
    const plan = (feet: vec3): [number, number] => {
        const obstacle = detectObstacle(w, feet, Z, 5, defaultDetectionSettings())!;
        const action = planTraversal(db, [m], ActionKind.Mantle, obstacle, feet, 0, 0, db.clips[0].start, rules, () => true);
        assert(!isRefusal(action) && action.path.kind === "warp", `${action}`);
        if (isRefusal(action) || action.path.kind !== "warp") throw new Error("no plan");
        const placed = action.path.warp.to.apply(v(m.ledge[0], 0, m.ledge[2]));
        return [action.start, placed[2] - obstacle.ledge[2]];
    };
    // from its run-up: within it, the ledge on the edge
    let [start, past] = plan(v(0, 0, -1.5));
    assert(start <= m.lastEntry && Math.abs(past) < 1e-3, `${start} ${past}`);
    // against it (0.3 m, closer than the run-up comes before the hands reach the ledge): after
    // `lastEntry`, no later than the hands reach it, the ledge at most `ledgeTolerance` past
    [start, past] = plan(v(0, 0, -0.3));
    assert(start > m.lastEntry && start <= m.latestEntry(), `${start} (${m.lastEntry} to ${m.latestEntry()})`);
    assert(past >= -1e-3 && past <= rules.ledgeTolerance + 1e-3, `${past}`);
});

test("pushed into a wall at an angle it slides along it", () => {
    const [db, table] = contactDatabase();
    const w = block(2.4, 3);
    const c = controller(db, table, v(0, 0, -3));
    const a = Math.fround(30 * Math.fround(Math.PI / 180));
    const heading = v(Math.sin(a) * 1.5, 0, Math.cos(a) * 1.5);
    assert(stepUntil(c, db, w, heading, 4, (c) => position(c)[2] > -0.36));
    run(c, db, w, heading, 0.5);
    const x = c.matcher.simulation.position[0];
    run(c, db, w, heading, 1);
    // the input's pace along the wall (0.75 m/s), none into it
    const vel = c.matcher.simulation.velocity;
    assert(Math.abs(vel[0] - 0.75) < 0.05 && Math.abs(vel[2]) < 0.05, `${vel}`);
    const moved = c.matcher.simulation.position[0] - x;
    assert(moved > 0.65 && moved < 0.8, `${moved} m along the wall in a second`);
});

test("against a face the ledge is in front of the feet", () => {
    // a box 2 m wide; the feet against its face 0.35 m from one end, probing at 45 degrees toward
    // that end (where the probe meets the face only 0.1 m from the corner)
    const w = floor();
    w.addBox(box([-1, 0, 0], [1, 1, 0.9]));
    const s = defaultDetectionSettings();
    const feet = v(-0.65, 0, -0.3);
    let o = detectObstacle(w, feet, v(-1, 0, 1), 2, s)!;
    assert(close(o.ledge, [-0.65, 1, 0], 0.01), `${o.ledge}`);
    assert(o.halfWidth >= 0.3, `${o.halfWidth}`);
    assertEq(traversalKind(w, o, feet, defaultTraversalRules(), ALL), ActionKind.Vault);
    // from farther away, where the probe meets it
    o = detectObstacle(w, v(0.5, 0, -1.5), v(-1, 0, 1), 3, s)!;
    assert(close(o.ledge, [-0.88, 1, 0], 0.05), `${o.ledge}`);
});
