import { quat, vec3 } from "gl-matrix";
import { wrapAngle, yawRotation } from "./motion_matching/Heading";

/**
 * Motion warping: bend a clip's root motion so a moment of it lands on a target, such as a hand
 * reaching the ledge the clip was captured against, wherever the real ledge is. Rust:
 * `animation::warping`.
 *
 * The clip's root path is placed in the world through a frame (a heading and a ground-plane
 * offset) that eases from where the character is when the clip starts to where the target puts
 * the clip, over a window of frames; heights and lengths the target changes are eased in the same
 * way with ramps. Before the window the clip plays as captured from the character; after it, it
 * plays in the target's frame. After the idea of motion warping (Witkin and Popović, "Motion
 * Warping", SIGGRAPH 1995) as games use it for traversal.
 */

/** A root on the ground: a position and a heading (radians about +Y). */
type Root = [vec3, number];

/** Smooth step between 0 and 1. */
function ease(x: number): number {
    const t = Math.min(Math.max(x, 0), 1);
    return t * t * (3 - 2 * t);
}

/** A value that eases from `from` to `to` between two clip frames. */
class Ramp {
    constructor(public start: number, public end: number, public from: number, public to: number) { }

    public at(frame: number): number {
        if (this.end <= this.start) return frame < this.start ? this.from : this.to;
        return this.from + (this.to - this.from) * ease((frame - this.start) / (this.end - this.start));
    }
}

/** A rigid placement on the ground: a heading (radians about +Y) and a ground-plane offset (x, z). */
class Placement {
    constructor(public yaw: number = 0, public offset: [number, number] = [0, 0]) { }

    public static identity(): Placement {
        return new Placement();
    }

    /** The placement that takes a clip root (position, heading) to a world one. */
    public static between(clip: Root, world: Root): Placement {
        const yaw = wrapAngle(world[1] - clip[1]);
        const rotated = vec3.transformQuat(vec3.create(), clip[0], yawRotation(yaw));
        return new Placement(yaw, [world[0][0] - rotated[0], world[0][2] - rotated[2]]);
    }

    public apply(p: vec3, out: vec3 = vec3.create()): vec3 {
        vec3.transformQuat(out, p, yawRotation(this.yaw));
        out[0] += this.offset[0];
        out[2] += this.offset[1];
        return out;
    }

    public lerp(other: Placement, t: number): Placement {
        return new Placement(
            this.yaw + wrapAngle(other.yaw - this.yaw) * t,
            [this.offset[0] + (other.offset[0] - this.offset[0]) * t, this.offset[1] + (other.offset[1] - this.offset[1]) * t],
        );
    }

    public clone(): Placement {
        return new Placement(this.yaw, [this.offset[0], this.offset[1]]);
    }
}

/** A clip's root path warped into the world. */
class RootWarp {
    /** Where the clip is placed when it starts (so the character does not jump)... */
    public from: Placement;
    /** ...and where the target puts it, reached over `window` (0 to 1 over its frames). */
    public to: Placement;
    public window: Ramp = new Ramp(0, 0, 0, 1);
    /** World height of the clip's height 0. */
    public ground: number;
    /**
     * Heights added to the clip's root, summed (e.g. up by the difference between the real and
     * the captured obstacle, then down again past it).
     */
    public lift: Ramp[] = [];
    /** Distances added along `stretchDirection` (world), summed (a deeper obstacle). */
    public stretch: Ramp[] = [];
    public stretchDirection: vec3 = vec3.create();

    private constructor(from: Placement, ground: number) {
        this.from = from;
        this.to = from.clone();
        this.ground = ground;
    }

    /**
     * A warp that plays the clip as captured from where the character is: the clip root at
     * `clipStart` (in clip space) maps to the character (world position and heading).
     */
    public static identity(clipStart: Root, character: Root): RootWarp {
        const from = Placement.between(
            [vec3.fromValues(clipStart[0][0], 0, clipStart[0][2]), clipStart[1]],
            [vec3.fromValues(character[0][0], 0, character[0][2]), character[1]],
        );
        return new RootWarp(from, character[0][1] - clipStart[0][1]);
    }

    /**
     * The world root (position and heading) of the clip root (clip-space position and heading)
     * at clip frame `frame`.
     */
    public root(frame: number, clip: Root): Root {
        const placement = this.from.lerp(this.to, this.window.at(frame));
        const p = placement.apply(clip[0]);
        p[1] = this.ground + clip[0][1] + this.lift.reduce((s, r) => s + r.at(frame), 0);
        vec3.scaleAndAdd(p, p, this.stretchDirection, this.stretch.reduce((s, r) => s + r.at(frame), 0));
        return [p, clip[1] + placement.yaw];
    }

    /** `root` with the heading as a rotation. */
    public rootRotation(frame: number, clip: Root): [vec3, quat] {
        const [p, yaw] = this.root(frame, clip);
        return [p, yawRotation(yaw)];
    }
}

export { Ramp, Placement, RootWarp };
export type { Root };
