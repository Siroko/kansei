import { FixedStep } from '../../pacing/FixedStep';
import { FluidActivity, FluidSleep, FluidSpeedProbe } from './FluidActivity';
import type { FluidSleepOptions, FluidSpeed } from './FluidActivity';
import type { FluidSimulation } from './FluidSimulation';

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

interface Rest {
    sleep: FluidSleep;
    probe: FluidSpeedProbe;
    enabled: boolean;
}

/**
 * Fixed steps for a fluid, at a `WorldScale`, resting when it may. Port of the Rust engine's
 * `FluidStepper` (`rust/kansei-core/src/simulations/fluid/stepper.rs`).
 *
 * Each frame, `advance` says how many steps of `stepDt` simulated seconds to run. With
 * `withRest`, call `updateRest` first: while the fluid is culled or asleep, `advance` returns 0
 * (and carries no time over), and the caller shows the state through
 * `FluidSurfaceEffect.setActivity`. After the steps, `stepped` measures the speed it settles by.
 *
 * ```ts
 * const [min, max] = sim.bounds();
 * const inView = aabbInFrustum(frustumPlanes(viewProj), min, max);
 * surface.setActivity(stepper.updateRest(dt, inView, disturbed));
 * const steps = stepper.advance(dt);
 * if (steps > 0) sim.updateBatched(stepper.stepDt, steps, mouse, dir, strength, passes);
 * stepper.stepped(sim, steps);
 * ```
 */
class FluidStepper {
    private fixed: FixedStep;
    private _scale: WorldScale;
    private rest: Rest | null = null;
    private _speed: FluidSpeed | null = null;

    /**
     * Steps of `step` real seconds, at most `maxSteps` a frame (time beyond is dropped), each
     * `step * scale.time` simulated seconds. A frame feeds at most `MAX_FRAME_DT` of real time
     * (see `FixedStep`).
     */
    constructor(step: number, maxSteps: number, scale: WorldScale = WorldScale.identity()) {
        this.fixed = new FixedStep(step).withMaxSteps(maxSteps);
        this._scale = scale;
    }

    /**
     * Let the fluid rest by `options`, whose speeds are in m/s (the stepper converts through its
     * scale), measuring `sim`'s particles.
     */
    public withRest(sim: FluidSimulation, options: Partial<FluidSleepOptions> = {}): this {
        const sleep = new FluidSleep(options);
        const probe = new FluidSpeedProbe(sim, this._scale.speedToSim(sleep.options.settleSpeed));
        this.rest = { sleep, probe, enabled: true };
        return this;
    }

    public get scale(): WorldScale {
        return this._scale;
    }

    /**
     * Run simulated time `time` times as fast as real time from now on (the speed probe's
     * threshold follows).
     */
    public setTimeScale(time: number): void {
        this._scale.time = time;
        if (this.rest) this.rest.probe.setThreshold(this._scale.speedToSim(this.rest.sleep.options.settleSpeed));
    }

    /** Real seconds a step stands for. */
    public get step(): number {
        return this.fixed.step;
    }

    /** Simulated seconds a step advances: the `dt` to step the simulation (and emit) with. */
    public get stepDt(): number {
        return this._scale.simSeconds(this.fixed.step);
    }

    /** Whether the fluid may rest (with `withRest`); off, it steps every frame, in view or not. */
    public get restEnabled(): boolean {
        return this.rest?.enabled ?? false;
    }

    public set restEnabled(enabled: boolean) {
        if (this.rest) this.rest.enabled = enabled;
    }

    /**
     * Decide this frame's state from whether the fluid's box is `inView`, whether something is
     * `disturbed`-ing it, and the latest speed read. Always running without `withRest`.
     */
    public updateRest(dt: number, inView: boolean, disturbed: boolean): FluidActivity {
        const rest = this.rest;
        if (!rest) return FluidActivity.Running;
        const read = rest.probe.take();
        let speed: number | null = null;
        if (read) {
            this._speed = { max: this._scale.speedToWorld(read.max), above: read.above };
            speed = this._speed.max;
        }
        const state = rest.sleep.update(dt, inView || !rest.enabled, disturbed || !rest.enabled, speed);
        if (state !== FluidActivity.Running) this.fixed.reset();
        return state;
    }

    /** The steps to run for a frame of `dt` real seconds: none while the fluid rests. */
    public advance(dt: number): number {
        if (this.state !== FluidActivity.Running) return 0;
        return this.fixed.advance(dt);
    }

    /**
     * After the frame's `steps` were submitted: measure the speed they left (with rest, when any
     * ran and no measurement is in flight).
     */
    public stepped(sim: FluidSimulation, steps: number): void {
        if (this.rest && steps > 0) this.rest.probe.measure(sim);
    }

    /** Running, culled or asleep (always running without rest). */
    public get state(): FluidActivity {
        return this.rest?.sleep.state ?? FluidActivity.Running;
    }

    /**
     * The last speed read, in m/s: the fastest particle, and how many moved faster than the
     * settle speed. Null before the first read, or without rest.
     */
    public get speed(): FluidSpeed | null {
        return this._speed;
    }

    /**
     * The fluid changed (a reset, a setting): run it until it settles again, and drop the speed
     * read in flight (taken before the change).
     */
    public wake(): void {
        if (!this.rest) return;
        this.rest.sleep.wake();
        this.rest.probe.forget();
    }

    /** Forget the time carried over (a pause, a reset). */
    public reset(): void {
        this.fixed.reset();
    }
}

export { FluidStepper, WorldScale };
