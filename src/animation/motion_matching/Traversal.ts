import { vec3 } from "gl-matrix";
import { CollisionWorld, ALL_LAYERS } from "../../collision/CollisionWorld";
import { Placement, Ramp, RootWarp } from "../Warping";
import { ActionClip, ActionKind, crosses } from "./ActionClip";
import { Database, STRIDE } from "./Database";
import { FORWARD, yawOf } from "./Heading";
import { Action, MotionInput, MotionMatcher } from "./MotionMatcher";

/**
 * Traversal: hurdles, vaults, mantles and climbs over obstacles found in a collision world, and
 * the falls and landings around them, played as `Action`s with warped root motion between
 * stretches of motion matching. Rust: `motion_matching::traversal`.
 *
 * - `ActionClip.analyze` reads what a traversal clip was captured against from its animation
 *   alone (see `ActionClip`).
 * - `detectObstacle` probes the world in front of the character: casts at several heights for a
 *   face, a ray down for its top, then rays along the top and the edge for its depth, width and
 *   the floor behind.
 * - `traversalKind` picks the kind from the obstacle's shape and the room to stand beyond its
 *   ledge. `planTraversal` picks the clip from the character's speed, the frame to start at from
 *   the distance to the ledge and the pose, and skips clips that would end inside something; it
 *   warps the clip so its ledge lands on the real one, lifted to the real height and stretched to
 *   the real depth. A character already at the obstacle (closer than any clip's run-up) starts a
 *   clip late, up to its take-off, and the clip's ledge may land a little past the real edge.
 * - `CharacterController` keeps a `MotionMatcher` out of the world's colliders, on its ground,
 *   falling off edges and landing, and traverses or jumps on request. A jump plays a jump clip
 *   (`planJump`: the pace and pose that fit) up to its take-off, then flies ballistic under
 *   the controller's gravity; landings are chosen by the height of the fall and the pace.
 */

const UP = vec3.fromValues(0, 1, 0);
const DOWN = vec3.fromValues(0, -1, 0);
const v3 = (x: number, y: number, z: number) => vec3.fromValues(x, y, z);
const flat = (v: vec3) => v3(v[0], 0, v[2]);
const along = (p: vec3, d: vec3, s: number) => vec3.scaleAndAdd(vec3.create(), p, d, s);

function normalizeOr(v: vec3, fallback: Readonly<vec3>): vec3 {
    const l = vec3.length(v);
    return l > 0 && Number.isFinite(1 / l) ? vec3.scale(vec3.create(), v, 1 / l) : vec3.clone(fallback as vec3);
}

/** An obstacle in front of the character. */
interface Obstacle {
    /** On the front edge of the top, where the character meets it. */
    ledge: vec3;
    /** The front face's normal (horizontal, toward the character). */
    normal: vec3;
    /** Top above the character's feet. */
    height: number;
    /** Length of the top away from the character; unset when deeper than probed. */
    depth?: number;
    /** Height (world) of the floor behind a shallow top. */
    backFloor?: number;
    /** Free top along the edge on each side of the approach line, the smaller. */
    halfWidth: number;
    /** From the character's feet to the ledge, along the approach. */
    distance: number;
}

/** Probes for obstacle detection. */
interface DetectionSettings {
    /** Heights above the feet the forward sweeps run at, and their sphere radius. */
    heights: number[];
    radius: number;
    /** Highest top considered, above the feet. */
    maxHeight: number;
    /** How far along the top depth is measured, and the step. */
    maxDepth: number;
    step: number;
    /**
     * Feet this close to the face (m) are against it: the ledge is straight in front of them,
     * not where a probe at an angle meets the face farther along.
     */
    contact: number;
    layers: number;
}

function defaultDetectionSettings(): DetectionSettings {
    return { heights: [0.35, 0.65, 1.0, 1.45, 1.95, 2.45], radius: 0.12, maxHeight: 3, maxDepth: 2, step: 0.1, contact: 0.5, layers: ALL_LAYERS };
}

/** The obstacle ahead of `feet` along `direction` (horizontal) within `reach`, if any. */
function detectObstacle(world: CollisionWorld, feet: vec3, direction: vec3, reach: number, settings: DetectionSettings = defaultDetectionSettings()): Obstacle | undefined {
    const dir = normalizeOr(flat(direction), vec3.create());
    if (vec3.length(dir) === 0) return undefined;
    const layers = settings.layers;
    // the nearest steep face at any probe height
    let hit: ReturnType<CollisionWorld["sphereCast"]>;
    for (const h of settings.heights) {
        const probe = world.sphereCast(v3(feet[0], feet[1] + h, feet[2]), settings.radius, dir, reach, layers);
        if (probe && Math.abs(probe.normal[1]) < 0.3 && probe.distance > 0 && (hit === undefined || probe.distance < hit.distance)) hit = probe;
    }
    if (!hit) return undefined;
    const normal = normalizeOr(flat(hit.normal), vec3.create());
    if (vec3.length(normal) === 0 || vec3.dot(normal, dir) > -0.3) return undefined;
    let face = along(hit.point, normal, -settings.radius);
    // against the face: straight in front of the feet, if the face goes on there
    const gap = vec3.dot(vec3.subtract(vec3.create(), feet, face), normal);
    if (gap <= settings.contact) {
        const front = world.raycast(v3(feet[0], face[1], feet[2]), vec3.negate(vec3.create(), normal), Math.max(gap, 0) + 0.05, layers);
        if (front && vec3.dot(front.normal, normal) > 0.95) face = front.point;
    }
    // its top, just behind the face
    const down = (p: vec3) => world.raycast(v3(p[0], feet[1] + settings.maxHeight + 0.3, p[2]), DOWN, settings.maxHeight + 0.3 - 0.05, layers);
    const topHit = down(along(face, normal, -0.08));
    if (!topHit || topHit.normal[1] < 0.7) return undefined;
    const top = topHit.point[1];
    const height = top - feet[1];
    if (height < 0.25 || height > settings.maxHeight) return undefined;
    const ledge = v3(face[0], top, face[2]);
    const onTop = (p: vec3) => {
        const h = world.raycast(v3(p[0], top + 0.3, p[2]), DOWN, 0.45, layers);
        return h !== undefined && Math.abs(h.point[1] - top) < 0.12;
    };
    // the top along the approach
    let depth: number | undefined;
    const steps = Math.round(settings.maxDepth / settings.step);
    for (let k = 1; k <= steps; k++) {
        const d = k * settings.step;
        if (!onTop(along(ledge, normal, -d))) {
            depth = d - settings.step * 0.5;
            break;
        }
    }
    // the floor behind a shallow top: none when the probe starts inside something (a wall right
    // behind) or finds nothing within 3 m below the feet
    let backFloor: number | undefined;
    if (depth !== undefined) {
        const behind = along(ledge, normal, -(depth + 0.5));
        const floor = world.raycast(v3(behind[0], top + 0.3, behind[2]), DOWN, top + 0.3 - (feet[1] - 3), layers);
        if (floor && floor.distance > 0) backFloor = floor.point[1];
    }
    // the edge to each side
    const side = vec3.normalize(vec3.create(), vec3.cross(vec3.create(), normal, UP));
    const reachSide = (s: number) => {
        let last = 0;
        for (let k = 1; k <= 10; k++) {
            const w = k * 0.1;
            if (!onTop(along(along(ledge, normal, -0.1), side, s * w))) break;
            last = w;
        }
        return last;
    };
    const halfWidth = Math.min(reachSide(1), reachSide(-1));
    const distance = vec3.dot(vec3.subtract(vec3.create(), feet, ledge), normal);
    return { ledge, normal, height, depth, backFloor, halfWidth, distance };
}

/** Why a traversal was not started. */
enum Refusal {
    Busy = "busy",
    NoObstacle = "nothing to traverse ahead",
    TooNarrow = "the ledge is too narrow",
    TooHigh = "too high or too low",
    NoRoom = "no room to land",
    NoClip = "no clip for it",
    OutOfReach = "too far or too close",
}

/** What a traversal request did: the kind started, or why not (the refusal's value says it in words). */
type TraversalResult = ActionKind | Refusal;

function isRefusal(r: unknown): r is Refusal {
    return typeof r === "string";
}

/** Clearances used when choosing a traversal. */
interface TraversalRules {
    /** Deepest top a hurdle, then a vault, goes over. */
    hurdleDepth: number;
    vaultDepth: number;
    /** Height ranges (above the feet) of each kind. */
    crossHeights: [number, number];
    mantleHeights: [number, number];
    climbHeights: [number, number];
    /** Narrowest half-width of ledge to use. */
    minHalfWidth: number;
    /** A standing character's capsule, for landing room. */
    radius: number;
    height: number;
    /**
     * Weights of the entry frame's cost: distance to the ledge (per m²), speed (per (m/s)²) and
     * pose (feature distance).
     */
    distanceWeight: number;
    speedWeight: number;
    poseWeight: number;
    /**
     * Largest distance error a warp may absorb (m), and fastest it may slide the root to do so
     * before the ledge is reached (m/s).
     */
    maxDistanceError: number;
    maxWarpSpeed: number;
    /**
     * Shortest time a warp eases the clip onto the obstacle (s): a traversal started late, from
     * right against it, turns and slides the character over at least this long.
     */
    minWarpTime: number;
    /**
     * Most of a clip's own travel over the warp that a character closer than captured may cut
     * (a fraction): its steps shorten, and it never slides backward.
     */
    maxShortening: number;
    /**
     * How far past the real edge a clip's ledge may land when the character is closer than the
     * warp can slide it back from (m; at most half the top's depth).
     */
    ledgeTolerance: number;
    /** Cost per second a clip starts past its `lastEntry`, skipping the start of its run-up. */
    lateWeight: number;
}

function defaultTraversalRules(): TraversalRules {
    return {
        hurdleDepth: 0.55,
        vaultDepth: 1.3,
        crossHeights: [0.3, 1.3],
        mantleHeights: [0.5, 1.8],
        climbHeights: [1.8, 2.9],
        minHalfWidth: 0.3,
        radius: 0.3,
        height: 1.75,
        distanceWeight: 4,
        speedWeight: 0.5,
        poseWeight: 0.05,
        maxDistanceError: 1.2,
        maxWarpSpeed: 1.5,
        minWarpTime: 0.2,
        maxShortening: 0.5,
        ledgeTolerance: 0.12,
        lateWeight: 1,
    };
}

/** Whether a standing character (the rules' capsule) fits with its feet at `feet`. */
function standsAt(world: CollisionWorld, feet: vec3, rules: TraversalRules, layers: number = ALL_LAYERS): boolean {
    const y = feet[1] + 0.05;
    return !world.overlapCapsule(v3(feet[0], y + rules.radius, feet[2]), v3(feet[0], y + rules.height - rules.radius, feet[2]), rules.radius, layers);
}

/** Which kind of traversal an obstacle calls for, or why none. */
function traversalKind(world: CollisionWorld, obstacle: Obstacle, feet: vec3, rules: TraversalRules, layers: number = ALL_LAYERS): TraversalResult {
    if (obstacle.halfWidth < rules.minHalfWidth) return Refusal.TooNarrow;
    const h = obstacle.height;
    const within = (r: [number, number]) => h >= r[0] && h <= r[1];
    const roomOnTop = () => standsAt(world, along(obstacle.ledge, obstacle.normal, -(rules.radius + 0.35)), rules, layers);
    const deepEnough = obstacle.depth === undefined || obstacle.depth >= 2 * rules.radius + 0.2;
    // shallow tops: over them, onto a floor near the feet behind
    if (obstacle.depth !== undefined && obstacle.depth <= rules.vaultDepth && within(rules.crossHeights)) {
        const depth = obstacle.depth;
        const floor = obstacle.backFloor;
        if (floor !== undefined && Math.abs(floor - feet[1]) < 0.6) {
            const behind = along(obstacle.ledge, obstacle.normal, -(depth + rules.radius + 0.3));
            if (!standsAt(world, v3(behind[0], floor, behind[2]), rules, layers)) return Refusal.NoRoom;
            return depth <= rules.hurdleDepth ? ActionKind.Hurdle : ActionKind.Vault;
        }
        // no floor to land on behind, and too shallow to stand on
        if (!deepEnough) return Refusal.NoRoom;
    }
    if (within(rules.mantleHeights) && deepEnough) return roomOnTop() ? ActionKind.Mantle : Refusal.NoRoom;
    if (within(rules.climbHeights) && deepEnough) return roomOnTop() ? ActionKind.Climb : Refusal.NoRoom;
    return Refusal.TooHigh;
}

/** Squared distance between two frames' pose features (feet and hips, not the trajectory). */
function poseDistance(db: Database, a: number, b: number): number {
    const rows = db.featureRows;
    let sum = 0;
    for (let i = 0; i < 15; i++) {
        const d = rows[a * STRIDE + i] - rows[b * STRIDE + i];
        sum += d * d;
    }
    return sum;
}

/**
 * Frames a warp from clip frame `start` has to ease onto the obstacle: until the ledge is
 * reached, and never less than `minWarpTime`.
 */
function warpFrames(c: ActionClip, start: number, db: Database, rules: TraversalRules): number {
    return Math.max(c.anchor - start, rules.minWarpTime * db.sampleRate, 1);
}

/**
 * The best clip of `kind` and frame to start it at for a character at `feet` with this speed
 * toward the obstacle, playing database frame `current`, and the action that warps it onto the
 * obstacle (or why none). A clip that ends on the top must leave the character where `stands`
 * (feet position) allows: a mantle that walks on before handing over needs more room than one
 * that stands up by the edge.
 *
 * A clip starts where its distance to the ledge is the character's, give or take what the warp
 * can slide before the ledge is reached. Right against the obstacle that is often nowhere in its
 * run-up (captured from half a metre or more): it may then start later, up to its take-off, and
 * land its ledge up to `ledgeTolerance` past the real edge rather than slide the character back.
 */
function planTraversal(
    db: Database, table: readonly ActionClip[], kind: ActionKind, obstacle: Obstacle, feet: vec3, heading: number, speed: number,
    current: number, rules: TraversalRules, stands: (feet: vec3) => boolean,
): Action | Refusal {
    const tolerance = Math.max(Math.min(rules.ledgeTolerance, obstacle.depth === undefined ? Infinity : obstacle.depth * 0.5), 0);
    // each clip's best entry frame and how far past the edge it lands, cheapest first
    const candidates: [number, ActionClip, number, number][] = [];
    for (const c of table) {
        if (c.kind !== kind) continue;
        let best: [number, number, number] | undefined;
        for (let f = 0; f <= c.latestEntry(); f += 1) {
            const d = c.distanceAt(db, f);
            // positive: the clip expects the ledge farther than it is
            const error = d - obstacle.distance;
            const frames = warpFrames(c, f, db, rules);
            const slide = Math.min(rules.maxDistanceError, rules.maxWarpSpeed * frames / db.sampleRate);
            // closer than captured: the clip's approach over the warp shortened, not reversed
            const travel = Math.max(d - c.distanceAt(db, f + frames), 0);
            const back = Math.min(slide, rules.maxShortening * travel);
            if ((error <= 0 && -error <= slide) || (error > 0 && error <= back + tolerance)) {
                const past = Math.max(error - back, 0);
                const s = c.speedAt(db, f) - speed;
                const pose = poseDistance(db, db.clips[c.clip].start + Math.floor(f), current);
                const late = Math.max((f - c.lastEntry) / db.sampleRate, 0);
                const e = error - past;
                const cost = rules.distanceWeight * (e * e + past * past) + rules.speedWeight * s * s + rules.poseWeight * pose + rules.lateWeight * late;
                if (best === undefined || cost < best[0]) best = [cost, f, past];
            }
        }
        if (best) candidates.push([best[0], c, best[1], best[2]]);
    }
    if (candidates.length === 0) return table.some((c) => c.kind === kind) ? Refusal.OutOfReach : Refusal.NoClip;
    candidates.sort((a, b) => a[0] - b[0]);
    for (const [, c, start, past] of candidates) {
        const warp = warpOnto(db, c, start, obstacle, past, feet, heading, rules);
        if (crosses(c.kind) || stands(warp.root(c.exit, db.rootAt(c.clip, c.exit))[0])) {
            return { clip: c.clip, start, exit: c.exit, path: { kind: 'warp', warp }, collides: false, tag: kind };
        }
    }
    return Refusal.NoRoom;
}

/**
 * The jump clip and frame to start it at for a character at this speed, playing database frame
 * `current`: the pace nearest the character's and the pose nearest the one showing, taking off
 * within `maxDelay` seconds.
 */
function planJump(db: Database, table: readonly ActionClip[], speed: number, current: number, rules: TraversalRules, maxDelay: number): [ActionClip, number] | undefined {
    let best: [number, ActionClip, number] | undefined;
    for (const c of table) {
        if (c.kind !== ActionKind.Jump) continue;
        for (let f = Math.floor(Math.max(c.rise - maxDelay * db.sampleRate, 0)); f <= c.lastEntry; f += 1) {
            const s = c.speedAt(db, f) - speed;
            const cost = rules.speedWeight * s * s + rules.poseWeight * poseDistance(db, db.clips[c.clip].start + f, current);
            if (best === undefined || cost < best[0]) best = [cost, c, f];
        }
    }
    return best && [best[1], best[2]];
}

/** `c`'s root motion from `start`, warped from the character at `feet` onto the obstacle, its ledge `past` metres beyond the real edge. */
function warpOnto(db: Database, c: ActionClip, start: number, obstacle: Obstacle, past: number, feet: vec3, heading: number, rules: TraversalRules): RootWarp {
    const clipStart = db.rootAt(c.clip, start);
    const warp = RootWarp.identity(clipStart, [feet, heading]);
    // the clip's ledge onto the real one, facing into it
    const clipHeading = Math.atan2(c.forward[0], c.forward[2]);
    const into = vec3.negate(vec3.create(), obstacle.normal);
    const target = along(obstacle.ledge, into, past);
    warp.to = Placement.between([v3(c.ledge[0], 0, c.ledge[2]), clipHeading], [v3(target[0], 0, target[2]), Math.atan2(into[0], into[2])]);
    warp.window = new Ramp(start, start + warpFrames(c, start, db, rules), 0, 1);
    // up by what the real obstacle has more than the captured one, by the time the ledge is
    // reached; over one, down again to the floor behind
    warp.ground = feet[1] - clipStart[0][1];
    const lift = obstacle.height - c.height;
    const from = Math.max(c.rise, start);
    warp.lift.push(new Ramp(from, Math.max(c.anchor, from + 1), 0, lift));
    if (crosses(c.kind)) {
        const floor = (obstacle.backFloor ?? feet[1]) - feet[1];
        warp.lift.push(new Ramp(c.offTop, Math.max(c.down, c.offTop + 1), 0, floor - lift));
        const extra = Math.min(Math.max((obstacle.depth ?? c.span) - past - c.span, -0.3), 1);
        warp.stretch.push(new Ramp(c.onTop, Math.max(c.offTop, c.onTop + 1), 0, extra));
        warp.stretchDirection = into;
    }
    return warp;
}

/** What a `CharacterController` is doing. */
type CharacterState =
    | { kind: 'grounded' }
    | { kind: 'traversing', action: ActionKind }
    /** Running up to a jump's take-off. */
    | { kind: 'jumping' }
    /** `time`: seconds in the air (after a take-off or off an edge). */
    | { kind: 'falling', time: number }
    | { kind: 'landing' };

/**
 * A motion-matched character in a collision world: kept out of colliders, on its ground,
 * falling off edges and landing, and traversing obstacles or jumping on request. Rust:
 * `motion_matching::traversal::CharacterController`.
 */
class CharacterController {
    public rules: TraversalRules = defaultTraversalRules();
    public detection: DetectionSettings = defaultDetectionSettings();
    /** Capsule: radius, height, and the step it walks up without blocking. */
    public radius = 0.3;
    public height = 1.75;
    public step = 0.35;
    public gravity = 9.81;
    public layers = ALL_LAYERS;
    /**
     * Jumps: how high they go (m above the take-off; unset: as high as the clip jumps), and
     * how soon after the request they may leave the ground (s).
     */
    public jumpHeight?: number;
    public maxJumpDelay = 0.4;
    /** A fall from higher than this (m) lands with the harder landings. */
    public heavyFall = 2;
    /**
     * How long a `requestTraverseOrJump` made while busy (landing, mid-traversal) waits to run
     * once the character is back on its feet (s).
     */
    public requestBuffer = 0.2;
    /** The last obstacle probed and what came of it (for debug views). */
    public lastObstacle?: Obstacle;
    public lastResult?: TraversalResult;
    private current: CharacterState = { kind: 'grounded' };
    /** Seconds a `requestTraverse` keeps trying while the obstacle is still out of reach. */
    private pending = 0;
    /** Whether the last obstacle probed has a kind of traversal (a clip may fit once closer). */
    private traversable = false;
    /**
     * Where the input last asked to go (m/s): traversals look that way, even while a wall the
     * character slides along turns its velocity away.
     */
    private intent = vec3.create();
    /** A request made while busy: seconds it still waits, and its patience. */
    private buffered?: [number, number];
    /** The jump running up to its take-off, and the highest point of this time in the air. */
    private jumpClip?: ActionClip;
    private airTop = 0;

    constructor(public matcher: MotionMatcher, public actions: ActionClip[]) { }

    public get state(): CharacterState {
        return this.current;
    }

    private first(kind: ActionKind): ActionClip | undefined {
        return this.actions.find((c) => c.kind === kind);
    }

    /**
     * Traverse the obstacle ahead as soon as it is in reach, trying for up to `patience` seconds
     * (a button pressed a little early still vaults).
     */
    public requestTraverse(db: Database, world: CollisionWorld, patience: number): TraversalResult {
        const result = this.traverse(db, world);
        // what may change as it comes closer: the obstacle in reach, a clip that fits
        this.pending = result === Refusal.OutOfReach || result === Refusal.NoObstacle || result === Refusal.NoRoom ? patience : 0;
        return result;
    }

    /**
     * Traverse the obstacle ahead as `requestTraverse` does, or jump when there is nothing to
     * traverse: no obstacle, one too high or too narrow, no room on it, no clip for it. Asked while
     * busy (landing, traversing), it runs once the character is back on its feet, if that is
     * within `requestBuffer`.
     */
    public requestTraverseOrJump(db: Database, world: CollisionWorld, patience: number): TraversalResult {
        const result = this.requestTraverse(db, world, patience);
        this.buffered = result === Refusal.Busy ? [this.requestBuffer, patience] : undefined;
        const nothingToTraverse = result === Refusal.NoObstacle || result === Refusal.TooHigh || result === Refusal.TooNarrow || result === Refusal.NoClip
            || (result === Refusal.NoRoom && !this.traversable);
        if (!nothingToTraverse) return result;
        this.pending = 0;
        const jumped = this.jump(db);
        this.lastResult = jumped;
        return jumped;
    }

    /**
     * Jump: the jump clip whose pace and pose fit plays its run-up to the take-off; from there
     * the character flies ballistic, with the run-up's momentum and up at the speed that reaches
     * `jumpHeight` (else the clip's own apex), the clip and then the fall loop animating it
     * until it lands.
     */
    public jump(db: Database): TraversalResult {
        if (this.current.kind !== 'grounded') return Refusal.Busy;
        const velocity = this.matcher.simulation.velocity;
        const speed = Math.hypot(velocity[0], velocity[2]);
        const plan = planJump(db, this.actions, speed, this.matcher.currentFrame(db), this.rules, this.maxJumpDelay);
        if (!plan) return Refusal.NoClip;
        const [clip, start] = plan;
        const character = this.matcher.character;
        const warp = RootWarp.identity(db.rootAt(clip.clip, start), [character.translation, yawOf(character.rotation)]);
        this.matcher.startAction(db, { clip: clip.clip, start, exit: undefined, path: { kind: 'warp', warp }, collides: true, tag: ActionKind.Jump });
        this.jumpClip = clip;
        this.current = { kind: 'jumping' };
        return ActionKind.Jump;
    }

    /** Advance by `dt` under `input`. */
    public update(db: Database, world: CollisionWorld, input: MotionInput, dt: number): void {
        this.intent = flat(input.velocity);
        this.advance(db, world, input, dt);
        if (this.buffered) {
            const [left, patience] = this.buffered;
            this.buffered = undefined;
            if (this.current.kind === 'grounded') this.requestTraverseOrJump(db, world, patience);
            else if (left > dt) this.buffered = [left - dt, patience];
        }
    }

    private advance(db: Database, world: CollisionWorld, input: MotionInput, dt: number): void {
        if (this.pending > 0) {
            this.pending -= dt;
            if (this.current.kind === 'grounded' && !isRefusal(this.traverse(db, world))) this.pending = 0;
        }
        const { radius, height, step, layers } = this;
        const constrain = (from: vec3, to: vec3): vec3 => {
            // sweep at the waist, stop short of what it meets and slide along it, then push out
            const delta = v3(to[0] - from[0], 0, to[2] - from[2]);
            const length = vec3.length(delta);
            let end = v3(to[0], from[1], to[2]);
            if (length > 1e-6) {
                const dir = vec3.scale(vec3.create(), delta, 1 / length);
                const hit = world.sphereCast(v3(from[0], from[1] + height * 0.5, from[2]), radius, dir, length, layers);
                // walls it moves into (not what it moves away from, e.g. overlapped on landing)
                if (hit && Math.abs(hit.normal[1]) < 0.5 && vec3.dot(hit.normal, delta) < 0) {
                    const n = normalizeOr(flat(hit.normal), vec3.create());
                    const stop = along(from, dir, Math.max(hit.distance - 0.01, 0));
                    const rest = vec3.scale(vec3.create(), delta, 1 - hit.distance / length);
                    end = vec3.add(stop, stop, along(rest, n, -vec3.dot(rest, n)));
                }
            }
            const resolved = world.resolveCapsule(end, height, radius, step, layers);
            return v3(resolved[0], to[1], resolved[2]);
        };
        const state = this.current;
        switch (state.kind) {
            case 'grounded': {
                this.matcher.updateConstrained(db, input, dt, constrain);
                const ground = world.groundHeight(this.matcher.character.translation, step + 0.1, 4, layers);
                if (ground !== undefined && ground > this.matcher.ground - 0.4) this.matcher.setGround(ground);
                else this.startFall(db);
                break;
            }
            case 'traversing': {
                this.matcher.update(db, input, dt);
                if (this.matcher.action) break;
                // over an obstacle, land on the floor behind; else stand where it ended
                const c = this.matcher.character.translation;
                const ground = world.groundHeight(c, 0.3, 4, layers);
                if (state.action === ActionKind.Vault && ground !== undefined && Math.abs(ground - c[1]) < 0.3) {
                    this.matcher.setGround(ground);
                    this.land(db, this.lastObstacle?.height ?? 0);
                } else if (ground !== undefined && ground > c[1] - 0.4) {
                    this.matcher.setGround(ground);
                    this.current = { kind: 'grounded' };
                } else {
                    this.current = { kind: 'grounded' };
                    this.startFall(db);
                }
                break;
            }
            case 'jumping': {
                this.matcher.updateConstrained(db, input, dt, constrain);
                const clip = this.jumpClip;
                if (!clip) {
                    this.matcher.stopAction();
                    this.current = { kind: 'grounded' };
                    return;
                }
                if (this.matcher.playing()[1] >= clip.rise) {
                    // leave the ground: the run-up's momentum, and up at the speed that reaches
                    // the jump's height
                    const v = this.matcher.simulation.velocity;
                    const up = Math.sqrt(2 * this.gravity * (this.jumpHeight ?? clip.height));
                    const action = this.matcher.action;
                    if (action) action.path = { kind: 'ballistic', velocity: v3(v[0], up, v[2]), gravity: this.gravity };
                    this.jumpClip = undefined;
                    this.airTop = this.matcher.character.translation[1];
                    this.current = { kind: 'falling', time: 0 };
                }
                break;
            }
            case 'falling': {
                this.matcher.updateConstrained(db, input, dt, constrain);
                // a jump's clip played out: on with the fall loop, at the same momentum
                const action = this.matcher.action;
                const air = action && action.path.kind === 'ballistic' ? { clip: action.clip, velocity: vec3.clone(action.path.velocity) } : undefined;
                const fall = this.first(ActionKind.Fall);
                if (air && fall && air.clip !== fall.clip && this.matcher.playing()[1] >= db.clips[air.clip].frames - 1) this.fallWith(db, air.velocity);
                const c = this.matcher.character.translation;
                this.airTop = Math.max(this.airTop, c[1]);
                // lands only on the way down (not on a step it rises past)
                const rising = air !== undefined && air.velocity[1] > 0;
                const ground = world.groundHeight(c, 1, 50, layers);
                if (ground !== undefined && !rising && c[1] <= ground) {
                    const t = this.matcher.character.clone();
                    t.translation[1] = ground;
                    this.matcher.place(t);
                    this.matcher.setGround(ground);
                    if (state.time > 0.35) {
                        this.land(db, this.airTop - ground);
                    } else {
                        this.matcher.stopAction();
                        this.current = { kind: 'grounded' };
                    }
                } else {
                    this.current = { kind: 'falling', time: state.time + dt };
                }
                break;
            }
            case 'landing': {
                this.matcher.updateConstrained(db, input, dt, constrain);
                if (!this.matcher.action) this.current = { kind: 'grounded' };
                break;
            }
        }
    }

    /** Off an edge: falling with the pace it walked at. */
    private startFall(db: Database): void {
        this.fallWith(db, flat(this.matcher.simulation.velocity));
        this.airTop = this.matcher.character.translation[1];
        this.current = { kind: 'falling', time: 0 };
    }

    /** Fly on with this momentum, animated by the fall loop (without one, by the pose that plays). */
    private fallWith(db: Database, velocity: vec3): void {
        const fall = this.first(ActionKind.Fall);
        const [clip, start] = fall ? [fall.clip, 0] : this.matcher.playing();
        this.matcher.startAction(db, { clip, start, exit: undefined, path: { kind: 'ballistic', velocity, gravity: this.gravity }, collides: true, tag: ActionKind.Fall });
    }

    /**
     * Land from a fall of `drop` metres: among the harder landings (those captured falling
     * farther) past `heavyFall`, else among the lighter ones, the one whose run-out pace is
     * nearest.
     */
    private land(db: Database, drop: number): void {
        const speed = vec3.length(this.matcher.simulation.velocity);
        const character = this.matcher.character;
        const lands = this.actions.filter((c) => c.kind === ActionKind.Land);
        const low = lands.reduce((m, c) => Math.min(m, c.height), Infinity);
        const high = lands.reduce((m, c) => Math.max(m, c.height), -Infinity);
        const heavy = drop > this.heavyFall;
        const pace = (c: ActionClip) => Math.abs(c.speedAt(db, Math.min(c.anchor + 5, c.exit)) - speed);
        let best: ActionClip | undefined;
        for (const c of lands) {
            if (!(high - low < 0.1 || (c.height > (low + high) * 0.5) === heavy)) continue;
            if (best === undefined || pace(c) < pace(best)) best = c;
        }
        if (!best) {
            this.matcher.stopAction();
            this.current = { kind: 'grounded' };
            return;
        }
        const clipRoot = db.rootAt(best.clip, best.anchor);
        // from the impact on, on the ground (whatever height the fall or vault ended at)
        const onGround = v3(character.translation[0], this.matcher.ground, character.translation[2]);
        const warp = RootWarp.identity(clipRoot, [onGround, yawOf(character.rotation)]);
        this.matcher.startAction(db, { clip: best.clip, start: best.anchor, exit: best.exit, path: { kind: 'warp', warp }, collides: true, tag: ActionKind.Land });
        this.current = { kind: 'landing' };
    }

    /**
     * Look for an obstacle ahead (where the input heads, else along the character's movement,
     * else its facing) and traverse it if it can.
     */
    public traverse(db: Database, world: CollisionWorld): TraversalResult {
        const result = this.tryTraverse(db, world);
        this.lastResult = result;
        return result;
    }

    private tryTraverse(db: Database, world: CollisionWorld): TraversalResult {
        if (this.current.kind !== 'grounded') return Refusal.Busy;
        const character = this.matcher.character;
        const moving = flat(this.matcher.simulation.velocity);
        const facing = vec3.transformQuat(vec3.create(), FORWARD, character.rotation);
        // where the player heads: a wall the character is pushed against leaves it no speed into
        // the wall, only along it
        const wanted = vec3.length(this.intent) > 0.1 ? this.intent : vec3.length(moving) > 0.5 ? moving : facing;
        const direction = normalizeOr(flat(wanted), FORWARD);
        // the pace toward the obstacle picks the clip, not a slide along it
        const speed = Math.max(vec3.dot(moving, direction), 0);
        const reach = 1.5 + speed * 1.1;
        const obstacle = detectObstacle(world, character.translation, direction, reach, this.detection);
        this.lastObstacle = obstacle;
        this.traversable = false;
        if (!obstacle) return Refusal.NoObstacle;
        const kind = traversalKind(world, obstacle, character.translation, this.rules, this.layers);
        if (isRefusal(kind)) return kind;
        this.traversable = true;
        const stands = (feet: vec3) => standsAt(world, feet, this.rules, this.layers);
        const action = planTraversal(db, this.actions, kind, obstacle, character.translation, yawOf(character.rotation), speed, this.matcher.currentFrame(db), this.rules, stands);
        if (isRefusal(action)) return action;
        this.matcher.startAction(db, action);
        this.current = { kind: 'traversing', action: kind };
        return kind;
    }
}

export {
    detectObstacle, defaultDetectionSettings, Refusal, isRefusal, defaultTraversalRules, standsAt, traversalKind, planTraversal, planJump,
    CharacterController,
};
export type { Obstacle, DetectionSettings, TraversalResult, TraversalRules, CharacterState };
