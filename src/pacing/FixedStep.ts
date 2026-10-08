/** The most real time one frame feeds a `FixedStep`: a 60 Hz frame's, s. */
export const MAX_FRAME_DT = 1 / 60;

/**
 * A fixed-step accumulator for simulations: feed it each frame's time and it says how many steps
 * of `step` seconds to run, so a simulation evolves the same at any frame rate. A port of the
 * Rust engine's `pacing::FixedStep`.
 *
 * A frame feeds it at most `maxFrameDt` of real time (a 60 Hz frame's, `MAX_FRAME_DT`), so below
 * 60 fps the simulation runs slower than real time instead of taking more steps. A frame's time
 * includes the GPU time of the steps before it: where a step costs nearly the real time it stands
 * for (a fluid at 1.9 times real time in steps of 1/60 s: 8.8 ms), each extra step lengthens the
 * next frame by about the time it simulated, so that frame asks for another, and the frames
 * settle at `maxSteps` (4 steps of 8 ms: under 30 fps) while simulating little more than two
 * steps a frame would (the GPU runs steps as fast as it can either way). Time beyond `maxSteps`
 * steps is dropped rather than carried over.
 */
export class FixedStep {
    step: number;
    maxSteps = 4;
    /** Real seconds one frame may feed (see the class docs). */
    maxFrameDt = MAX_FRAME_DT;
    private accumulator = 0;

    /** Steps of `step` seconds, at most 4 a frame, fed at most `MAX_FRAME_DT` a frame. */
    constructor(step: number) {
        this.step = Math.max(step, 1e-6);
    }

    withMaxSteps(maxSteps: number): this {
        this.maxSteps = Math.max(Math.floor(maxSteps), 1);
        return this;
    }

    withMaxFrameDt(maxFrameDt: number): this {
        this.maxFrameDt = Math.max(maxFrameDt, 0);
        return this;
    }

    /** Add a frame of `dt` real seconds and take the whole steps they make up. */
    advance(dt: number): number {
        return this.advanceScaled(dt, 1);
    }

    /**
     * Add a frame of `dt` real seconds simulated `timeScale` times as fast (each step still `step`
     * seconds of the scaled clock), and take the whole steps they make up.
     */
    advanceScaled(dt: number, timeScale: number): number {
        const fed = Math.min(Math.max(dt, 0), this.maxFrameDt) * Math.max(timeScale, 0);
        this.accumulator = Math.min(this.accumulator + fed, this.step * this.maxSteps);
        // (a step given as an f32 1/60 is a hair longer than `MAX_FRAME_DT`: count it whole and
        // carry no debt)
        const steps = Math.floor(this.accumulator / this.step + 1e-6);
        this.accumulator = Math.max(this.accumulator - steps * this.step, 0);
        return steps;
    }

    /** Forget the time carried over (pausing, a reset). */
    reset(): void {
        this.accumulator = 0;
    }
}
