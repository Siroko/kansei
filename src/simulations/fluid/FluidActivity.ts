/**
 * Letting a fluid rest: stop stepping it (and extracting its surface) while nobody can see it, or
 * while it has settled and nothing is touching it, and resume when that changes. Port of the Rust
 * engine's `simulations/fluid/activity.rs`, whose probe WGSL it imports (Vite `?raw`).
 *
 * - `FluidSpeedProbe` measures the particles' fastest speed (and how many move faster than a
 *   threshold) on the GPU and reads it back through a `ReadbackRing`, a few frames late: the frame
 *   never waits for the GPU.
 * - `FluidSleep` decides from that, from whether the fluid's box is in view (`aabbInFrustum` with
 *   `FluidSimulation.bounds`) and from whether anything is disturbing it, whether to step it:
 *   `FluidActivity`.
 *
 * The caller skips `FluidSimulation.updateBatched` while it is not running, and turns off the
 * surface's extraction (`FluidSurfaceEffect.extract`, which keeps drawing the last surface) or,
 * out of view, the whole effect (`FluidSurfaceEffect.active`): `FluidSurfaceEffect.setActivity`
 * does both. `FluidStepper.withRest` puts the two together with fixed steps; the
 * `index_fluid_lake` example shows it.
 */
import speedProbeWgsl from '../../../rust/kansei-core/src/simulations/fluid/shaders/speed_probe.wgsl?raw';
import { ReadbackRing } from '../../renderers/ReadbackRing';
import { gpuPass } from '../../profiling/Profiler';
import type { FluidSimulation } from './FluidSimulation';

/** Whether a fluid steps this frame, and why not. */
enum FluidActivity {
    /** Stepping, and its surface extracted, every frame. */
    Running = 'running',
    /** Out of view long enough for its last waves to have died out: paused, and not drawn. */
    Culled = 'culled',
    /** In view but settled, with nothing near it: paused, its last surface drawn as it was. */
    Asleep = 'asleep',
}

/** When `FluidSleep` pauses a fluid. */
interface FluidSleepOptions {
    /**
     * Seconds out of view before it is culled: it keeps stepping meanwhile, so waves started in
     * view die out rather than freeze.
     */
    cullAfter: number;
    /**
     * The fastest a particle may move for the fluid to count as settled (the speeds given to
     * `FluidSleep.update`; set it under what shows on screen).
     */
    settleSpeed: number;
    /**
     * Seconds every speed read must stay under `settleSpeed` before it sleeps (and at least two
     * reads: they may arrive seconds apart on a busy GPU).
     */
    settleAfter: number;
}

const DEFAULT_FLUID_SLEEP_OPTIONS: Readonly<FluidSleepOptions> = { cullAfter: 1.5, settleSpeed: 0.05, settleAfter: 1.0 };

/**
 * Decides each frame whether a fluid steps (`FluidActivity`): culled after `cullAfter` seconds
 * out of view; asleep once in view, undisturbed, and every speed read (two at least) over
 * `settleAfter` seconds of stepping has been under `settleSpeed`; else running. A disturbance
 * (something near enough to touch it) or `wake` runs it at once (while in view); coming back into
 * view runs it again, unless it had settled before it was culled.
 */
class FluidSleep {
    public options: FluidSleepOptions;
    private _state: FluidActivity = FluidActivity.Running;
    private outOfView = 0;
    /** Seconds stepped since the speed was last over `settleSpeed` (and since a disturbance) */
    private settled = 0;
    /** Speed reads since then (all under it) */
    private calm = 0;

    constructor(options: Partial<FluidSleepOptions> = {}) {
        this.options = { ...DEFAULT_FLUID_SLEEP_OPTIONS, ...options };
    }

    public get state(): FluidActivity {
        return this._state;
    }

    /** Whether the fluid steps (and its surface is extracted) this frame. */
    public get running(): boolean {
        return this._state === FluidActivity.Running;
    }

    /** The fluid changed (a setting, a reset): run it until it settles again. */
    public wake(): void {
        this.settled = 0;
        this.calm = 0;
        if (this._state === FluidActivity.Asleep) this._state = FluidActivity.Running;
    }

    /**
     * Advance by `dt` seconds: whether the fluid's box is `inView`, whether something is
     * `disturbed`-ing it (near enough to touch it), and the fastest particle's `speed` if a new
     * measurement arrived (taken while it was stepping). Returns this frame's state.
     */
    public update(dt: number, inView: boolean, disturbed: boolean, speed: number | null = null): FluidActivity {
        const o = this.options;
        this.outOfView = inView ? 0 : this.outOfView + dt;
        if (disturbed) {
            this.settled = 0;
            this.calm = 0;
        } else if (this._state === FluidActivity.Running) {
            if (speed !== null) {
                if (speed > o.settleSpeed || !Number.isFinite(speed)) {
                    this.settled = 0;
                    this.calm = 0;
                } else {
                    this.calm++;
                }
            }
            if (this.calm > 0) this.settled += dt;
        }
        this._state = this.outOfView >= o.cullAfter
            ? FluidActivity.Culled
            : this.calm >= 2 && this.settled >= o.settleAfter
                ? FluidActivity.Asleep
                : FluidActivity.Running;
        return this._state;
    }
}

/** One measurement of the particles' speeds (the simulation's units per simulated second). */
interface FluidSpeed {
    /** The fastest particle's speed. */
    max: number;
    /** How many particles moved faster than the probe's threshold. */
    above: number;
}

/** Bytes of the probe's `Params` (`speed_probe.wgsl`): count, threshold and two pads. */
const PROBE_PARAMS_BYTES = 16;

/**
 * Measures a simulation's particle speeds on the GPU (a reduction to two words) and reads them
 * back without stalling, through a `ReadbackRing` of `depth` staging buffers (1, as in Rust, by
 * default): `measure` does nothing while every one is on its way, and `take` collects the newest
 * once mapped (a frame or a few later).
 */
class FluidSpeedProbe {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly params: GPUBuffer;
    private readonly result: GPUBuffer;
    private readonly ring: ReadbackRing;
    private bindGroup: GPUBindGroup | null = null;
    /** The velocities `bindGroup` was made with */
    private bound: GPUBuffer | null = null;
    private count = -1;
    private _threshold: number;

    /** A probe on `sim`'s velocities, counting the particles faster than `threshold`. */
    constructor(sim: FluidSimulation, threshold: number, depth: number = 1) {
        const device = sim.gpuDevice;
        this.device = device;
        this._threshold = threshold;
        this.pipeline = device.createComputePipeline({
            label: 'FluidSpeedProbe',
            layout: 'auto',
            compute: { module: device.createShaderModule({ label: 'FluidSpeedProbe', code: speedProbeWgsl }), entryPoint: 'main' },
        });
        this.params = device.createBuffer({ label: 'FluidSpeedProbe/Params', size: PROBE_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.result = device.createBuffer({ label: 'FluidSpeedProbe/Result', size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.ring = new ReadbackRing(device, 8, depth, 'FluidSpeedProbe/Readback');
    }

    private upload(): void {
        const data = new ArrayBuffer(PROBE_PARAMS_BYTES);
        new Uint32Array(data)[0] = this.count;
        new Float32Array(data)[1] = this._threshold;
        this.device.queue.writeBuffer(this.params, 0, data);
    }

    public get threshold(): number {
        return this._threshold;
    }

    /** Count the particles faster than `threshold` from the next measurement on. */
    public setThreshold(threshold: number): void {
        this._threshold = threshold;
        if (this.count >= 0) this.upload();
    }

    /**
     * Measure the speeds as the GPU has them after the work submitted so far (this frame's
     * steps), unless every readback is still on its way.
     */
    public measure(sim: FluidSimulation): void {
        if (!this.ring.free) return;
        const velocities = sim.velocitiesBufferRef.resource.buffer;
        if (this.bound !== velocities) {
            this.bound = velocities;
            this.bindGroup = this.device.createBindGroup({
                label: 'FluidSpeedProbe',
                layout: this.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: velocities } },
                    { binding: 1, resource: { buffer: this.result } },
                    { binding: 2, resource: { buffer: this.params } },
                ],
            });
        }
        // the live particles (particles emitted since the last measurement included)
        if (this.count !== sim.particleCount) {
            this.count = sim.particleCount;
            this.upload();
        }
        const encoder = this.device.createCommandEncoder({ label: 'FluidSpeedProbe' });
        encoder.clearBuffer(this.result);
        const pass = encoder.beginComputePass({ label: 'FluidSpeedProbe', timestampWrites: gpuPass('FluidSpeedProbe') });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup!);
        pass.dispatchWorkgroups(Math.ceil(this.count / 256));
        pass.end();
        this.ring.copy(encoder, this.result);
        this.device.queue.submit([encoder.finish()]);
        this.ring.submitted();
    }

    /** Drop the measurements on their way (taken before the fluid changed): `take` will not return them. */
    public forget(): void {
        this.ring.forget();
    }

    /** The newest measurement, once it has arrived (each once); null until then. */
    public take(): FluidSpeed | null {
        const words = this.ring.take();
        if (!words) return null;
        return { max: new Float32Array(words.buffer)[0], above: words[1] };
    }

    public destroy(): void {
        this.params.destroy();
        this.result.destroy();
        this.ring.destroy();
    }
}

export { FluidActivity, FluidSleep, FluidSpeedProbe, DEFAULT_FLUID_SLEEP_OPTIONS };
export type { FluidSleepOptions, FluidSpeed };
