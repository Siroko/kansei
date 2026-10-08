import { vec3 } from "gl-matrix";
import { Pose } from "../Pose";
import { Transform } from "../Transform";
import { Database } from "./Database";
import { FORWARD, yawRotation } from "./Heading";

/** What an action clip does (the `.kmm` byte of each kind is its value). */
enum ActionKind {
    /** Over a thin obstacle, landing on the far side. */
    Hurdle = 0,
    /** Over a deeper one, hands on top. */
    Vault = 1,
    /** Up onto a top to stand on. */
    Mantle = 2,
    /** Up a tall wall onto its top. */
    Climb = 3,
    /** Falling (a loop). */
    Fall = 4,
    /** Landing from a fall. */
    Land = 5,
    /** Jumping: the run-up and take-off, then the pose in the air. */
    Jump = 6,
}

const ACTION_KIND_NAMES = ["hurdle", "vault", "mantle", "climb", "fall", "land", "jump"];

function actionKindName(kind: ActionKind): string {
    return ACTION_KIND_NAMES[kind];
}

function actionKindFromName(name: string): ActionKind | undefined {
    const i = ACTION_KIND_NAMES.indexOf(name);
    return i < 0 ? undefined : i as ActionKind;
}

/** Goes over the obstacle (lands behind it) rather than onto it. */
function crosses(kind: ActionKind): boolean {
    return kind === ActionKind.Hurdle || kind === ActionKind.Vault;
}

/** The fields of an `ActionClip`. */
interface ActionClipFields {
    clip: number;
    kind: ActionKind;
    height: number;
    ledge: vec3;
    forward: vec3;
    rise: number;
    anchor: number;
    onTop: number;
    offTop: number;
    down: number;
    exit: number;
    span: number;
    lastEntry: number;
}

/**
 * What an action clip was captured against, and its phases (clip frames). Rust:
 * `motion_matching::traversal::ActionClip`.
 */
class ActionClip implements ActionClipFields {
    public clip: number;
    public kind: ActionKind;
    /**
     * Height of the obstacle above the clip's starting ground (for a landing: the drop, for a
     * jump: the apex above the take-off).
     */
    public height: number;
    /**
     * The obstacle's front ledge (on its top), in the clip's space, and the direction the clip
     * runs into it (horizontal, unit).
     */
    public ledge: vec3;
    public forward: vec3;
    /** The last frame on the starting ground before the lift (for a jump: the take-off). */
    public rise: number;
    /**
     * When the ledge is reached (hands on it, or the root most of the way up; for a landing:
     * the impact, for a jump: the apex).
     */
    public anchor: number;
    /** When the root is up on the top, and (over an obstacle) when it leaves it and is back down. */
    public onTop: number;
    public offTop: number;
    public down: number;
    /** When motion matching can take over. */
    public exit: number;
    /** Distance the root travels on the top (over an obstacle). */
    public span: number;
    /** Latest frame a traversal can start at (time for the warp before the lift). */
    public lastEntry: number;

    constructor(fields: ActionClipFields) {
        this.clip = fields.clip;
        this.kind = fields.kind;
        this.height = fields.height;
        this.ledge = vec3.clone(fields.ledge);
        this.forward = vec3.clone(fields.forward);
        this.rise = fields.rise;
        this.anchor = fields.anchor;
        this.onTop = fields.onTop;
        this.offTop = fields.offTop;
        this.down = fields.down;
        this.exit = fields.exit;
        this.span = fields.span;
        this.lastEntry = fields.lastEntry;
    }

    /**
     * Read `clip`'s phases for `kind` from its animation (`hands`: the hand joints). `undefined`
     * when the clip doesn't show the motion (a traversal that never leaves the ground, a landing
     * with no fall).
     */
    public static analyze(db: Database, clip: number, kind: ActionKind, hands: [number, number]): ActionClip | undefined {
        const n = db.clips[clip].frames;
        const rate = db.sampleRate;
        const t = tracks(db, clip, hands);
        const base = t.root[0][0][1];
        const y = t.root.map(([p]) => p[1] - base);
        const last = n - 1;
        const empty: ActionClipFields = {
            clip, kind, height: 0, ledge: vec3.create(), forward: vec3.clone(FORWARD),
            rise: 0, anchor: 0, onTop: 0, offTop: 0, down: 0, exit: last, span: 0, lastEntry: 0,
        };
        const flatForward = (yaw: number) => {
            const f = vec3.transformQuat(vec3.create(), FORWARD, yawRotation(yaw));
            f[1] = 0;
            return vec3.length(f) > 0 ? vec3.normalize(f, f) : vec3.clone(FORWARD);
        };
        const first = (from: number, to: number, test: (f: number) => boolean) => {
            for (let f = from; f < to; f++) if (test(f)) return f;
            return undefined;
        };
        const lastOf = (from: number, to: number, test: (f: number) => boolean) => {
            for (let f = to - 1; f >= from; f--) if (test(f)) return f;
            return undefined;
        };
        if (kind === ActionKind.Fall) return new ActionClip(empty);
        if (kind === ActionKind.Land) {
            // the impact: the root reaches its final height after being well above it
            const end = y[n - 1];
            const top = y.reduce((a, b) => Math.max(a, b), -Infinity);
            if (top - end < 0.3) return undefined;
            let above = false;
            const impact = first(0, n, (f) => {
                const hit = above && y[f] - end <= 0.02;
                above ||= y[f] - end > 0.3;
                return hit;
            });
            if (impact === undefined) return undefined;
            return new ActionClip({ ...empty, height: top - end, anchor: impact, exit: Math.min(impact + 0.5 * rate, last) });
        }
        if (kind === ActionKind.Jump) {
            // the take-off: the last frame on the ground before the root rises to its apex
            let apex = 0;
            for (let f = 1; f < n; f++) if (y[f] >= y[apex]) apex = f;
            if (y[apex] < 0.2) return undefined;
            const rise = lastOf(0, apex, (f) => y[f] <= 0.02);
            if (rise === undefined) return undefined;
            return new ActionClip({
                ...empty, height: y[apex], forward: flatForward(t.root[rise][1]), rise, anchor: apex, onTop: apex, lastEntry: Math.max(rise - 2, 0),
            });
        }
        const peak = crosses(kind) ? y.reduce((a, b) => Math.max(a, b), -Infinity) : y[n - 1];
        if (peak < 0.2) return undefined;
        const onTop = first(0, n, (f) => y[f] >= 0.98 * peak);
        if (onTop === undefined) return undefined;
        const rise = lastOf(0, onTop, (f) => y[f] <= 0.02) ?? 0;
        const halfway = first(0, n, (f) => y[f] >= 0.9 * peak) ?? onTop;
        const forward = flatForward(t.root[onTop][1]);
        // hands planted on the top near the lift: the ledge is just in front of them
        const topY = base + peak;
        let planted: [number, number] | undefined;
        for (let f = Math.max(rise - 10, 0); f < Math.min(onTop + 10, n - 1); f++) {
            for (let h = 0; h < 2; h++) {
                const p = t.hands[f][h], q = t.hands[f + 1][h];
                if (p[1] > topY - 0.15 && p[1] < topY + 0.25 && vec3.distance(p, q) * rate < 0.6) {
                    const along = vec3.dot(p, forward);
                    if (planted === undefined || f < planted[0] || (f === planted[0] && along < planted[1])) planted = [f, along];
                }
            }
        }
        const [anchor, along] = planted ? [Math.min(planted[0], onTop), planted[1] - 0.05] : [halfway, vec3.dot(t.root[halfway][0], forward)];
        const at = t.root[anchor][0];
        const ledge = vec3.scaleAndAdd(vec3.create(), vec3.fromValues(at[0], topY, at[2]), forward, along - vec3.dot(at, forward));
        let offTop: number, down: number, span: number, exit: number;
        if (crosses(kind)) {
            let off = onTop;
            while (off + 1 < n && y[off + 1] >= 0.9 * peak) off++;
            down = first(off, n, (f) => y[f] <= 0.02) ?? n - 1;
            span = vec3.dot(vec3.subtract(vec3.create(), t.root[off][0], t.root[onTop][0]), forward);
            // a hurdle runs on from the landing; a vault hands over as it reaches the ground
            exit = kind === ActionKind.Hurdle ? Math.min(down + 0.3 * rate, last) : down;
            offTop = off;
        } else {
            // up there once the hips are back to standing height
            const standing = t.hipsHeight[0] * 0.9;
            const up = first(onTop, n, (f) => t.hipsHeight[f] >= standing) ?? n - 1;
            [offTop, down, span, exit] = [last, last, 0, Math.min(up + 0.1 * rate, last)];
        }
        return new ActionClip({
            clip, kind, height: peak, ledge, forward, rise, anchor, onTop, offTop, down, exit, span, lastEntry: Math.max(rise - 0.2 * rate, 0),
        });
    }

    /** Distance from the clip root at `frame` to the ledge, along the approach. */
    public distanceAt(db: Database, frame: number): number {
        const [p] = db.rootAt(this.clip, frame);
        return vec3.dot(vec3.subtract(p, this.ledge, p), this.forward);
    }

    /**
     * Latest frame a traversal can start at from right against the obstacle: its take-off, or
     * the hands reaching the ledge if that comes first (never earlier than `lastEntry`).
     */
    public latestEntry(): number {
        return Math.max(Math.min(this.rise, this.anchor), this.lastEntry);
    }

    /** The clip root's speed (m/s) at `frame`. */
    public speedAt(db: Database, frame: number): number {
        const [a] = db.rootAt(this.clip, frame);
        const [b] = db.rootAt(this.clip, frame + 1);
        return Math.hypot(b[0] - a[0], b[2] - a[2]) * db.sampleRate;
    }
}

/** The analysis' view of a clip: root per frame, hands and hips in the clip's space. */
function tracks(db: Database, clip: number, hands: [number, number]): { root: [vec3, number][], hands: [vec3, vec3][], hipsHeight: number[] } {
    const info = db.clips[clip];
    const pose = new Pose([]);
    const model: Transform[] = [];
    const out = { root: [] as [vec3, number][], hands: [] as [vec3, vec3][], hipsHeight: [] as number[] };
    for (let f = 0; f < info.frames; f++) {
        const frame = info.start + f;
        db.pose(frame, frame, 0, pose);
        pose.toModel(db.skeleton, model);
        const [p, yaw] = db.rootAt(clip, f);
        const r = yawRotation(yaw);
        out.root.push([p, yaw]);
        out.hands.push(hands.map((h) => vec3.add(vec3.create(), p, vec3.transformQuat(vec3.create(), model[h].translation, r))) as [vec3, vec3]);
        out.hipsHeight.push(model[db.roles.hips].translation[1]);
    }
    return out;
}

export { ActionKind, ActionClip, actionKindName, actionKindFromName, crosses };
export type { ActionClipFields };
