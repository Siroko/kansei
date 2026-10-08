import { FixedStep } from '../../pacing/FixedStep';

type Vec3 = [number, number, number];

/**
 * How a simulation's units relate to the world's: `length` simulation units a metre, and `time`
 * simulated seconds a real second. A fluid tuned at one size (its smoothing radius, its spacing)
 * runs in a world of another this way: positions and lengths go in times `length`, velocities
 * times `length / time`, and each real second steps `time` simulated ones.
 * Port of the Rust engine's `WorldScale` (`rust/kansei-core/src/simulations/fluid/stepper.rs`).
 */
class WorldScale {
    constructor(public length: number = 1, public time: number = 1) {}

    /** The simulation's units are metres and seconds. */
    public static identity(): WorldScale {
        return new WorldScale(1, 1);
    }

    /**
     * `length` simulation units a metre, with simulated time running √`length` times faster than
     * real time: a simulation whose gravity is 9.8 of its units per second² then falls at the
     * real pace.
     */
    public static withRealGravity(length: number): WorldScale {
        return new WorldScale(length, Math.sqrt(length));
    }

    /** A point (metres) in simulation units. */
    public pointToSim(p: Vec3): Vec3 {
        return [p[0] * this.length, p[1] * this.length, p[2] * this.length];
    }

    /** A point in simulation units, in metres. */
    public pointToWorld(p: Vec3): Vec3 {
        return [p[0] / this.length, p[1] / this.length, p[2] / this.length];
    }

    /** A length (metres) in simulation units. */
    public lengthToSim(m: number): number {
        return m * this.length;
    }

    /** A length in simulation units, in metres. */
    public lengthToWorld(l: number): number {
        return l / this.length;
    }

    /** A velocity (m/s) in simulation units per simulated second. */
    public velocityToSim(v: Vec3): Vec3 {
        return [this.speedToSim(v[0]), this.speedToSim(v[1]), this.speedToSim(v[2])];
    }

    /** A speed (m/s) in simulation units per simulated second. */
    public speedToSim(s: number): number {
        return s * this.length / this.time;
    }

    /** A speed in simulation units per simulated second, in m/s. */
    public speedToWorld(s: number): number {
        return s * this.time / this.length;
    }

    /** Real seconds as simulated seconds. */
    public simSeconds(dt: number): number {
        return dt * this.time;
    }
}

/**
 * Fixed steps for a fluid, at a `WorldScale`. Each frame, `advance` says how many steps of
 * `stepDt` simulated seconds to run:
 *
 * ```ts
 * const steps = stepper.advance(dt);
 * if (steps > 0) sim.updateBatched(stepper.stepDt, steps, mouse, dir, strength, passes);
 * ```
 *
 * The Rust `FluidStepper` can also let the fluid rest (asleep or culled); this one always runs.
 */
class FluidStepper {
    private fixed: FixedStep;
    private _scale: WorldScale;

    /**
     * Steps of `step` real seconds, at most `maxSteps` a frame (time beyond is dropped), each
     * `step * scale.time` simulated seconds. A frame feeds at most `MAX_FRAME_DT` of real time
     * (see `FixedStep`).
     */
    constructor(step: number, maxSteps: number, scale: WorldScale = WorldScale.identity()) {
        this.fixed = new FixedStep(step).withMaxSteps(maxSteps);
        this._scale = scale;
    }

    public get scale(): WorldScale {
        return this._scale;
    }

    /** Run simulated time `time` times as fast as real time from now on. */
    public setTimeScale(time: number): void {
        this._scale.time = time;
    }

    /** Real seconds a step stands for. */
    public get step(): number {
        return this.fixed.step;
    }

    /** Simulated seconds a step advances: the `dt` to step the simulation (and emit) with. */
    public get stepDt(): number {
        return this._scale.simSeconds(this.fixed.step);
    }

    /** The steps to run for a frame of `dt` real seconds. */
    public advance(dt: number): number {
        return this.fixed.advance(dt);
    }

    /** Forget the time carried over (a pause, a reset). */
    public reset(): void {
        this.fixed.reset();
    }
}

export { FluidStepper, WorldScale };
