/**
 * Moving colliders for the fluid: capsules the app places each frame (a character's legs, an
 * oar, a boat's hull made of a few) that push the particles out of them and carry them along
 * with their own velocity: bow waves, wakes and splashes. Applied after each substep's
 * integration by `FluidColliders` (a `FluidSubstepPass`).
 *
 * The coupling is one way: the fluid does not push back on the colliders (`FluidBody` is the TS
 * engine's two-way 2D body).
 *
 * Port of the Rust engine's `colliders.rs` (`rust/kansei-core/src/simulations/fluid/`), whose
 * WGSL it imports (Vite `?raw`).
 */
import collidersWgsl from '../../../rust/kansei-core/src/simulations/fluid/shaders/colliders.wgsl?raw';
import { ComputeBuffer } from '../../buffers/ComputeBuffer';
import { BufferBase } from '../../buffers/BufferBase';
import { assemble } from '../../materials/shaders/ShaderUtils';
import { simParamsStruct } from './shaders/sim-params.wgsl';
import type { FluidSimulation, FluidSubstepPass } from './FluidSimulation';

type Vec3 = [number, number, number];

/** The colliders' substep pass: `SimParams`, then the colliders (Rust: `COLLIDERS_WGSL`). */
const COLLIDERS_WGSL: string = assemble([simParamsStruct, collidersWgsl]);

/** Floats of one capsule on the GPU (`Capsule` in `colliders.wgsl`: 64 bytes). */
const CAPSULE_FLOATS = 16;
/** Bytes of the colliders' uniform (`Colliders` in `colliders.wgsl`). */
const COLLIDERS_UNIFORM_BYTES = 16;

/**
 * A capsule: the segment `a`-`b` grown by `radius`, and the velocity of each end (per second of
 * simulation time), in the simulation's space. A sphere is a capsule with `a == b`.
 * `expansion` is how fast its surface moves out along its normal (a body displacing the fluid
 * all round, such as a landing's impact), on top of the ends' velocities.
 */
class FluidCapsule {
    constructor(
        public a: Vec3,
        public b: Vec3,
        public radius: number,
        public velocityA: Vec3 = [0, 0, 0],
        public velocityB: Vec3 = [0, 0, 0],
        public expansion: number = 0,
    ) {}

    /**
     * A capsule on a rigid body (a paddle on a turning wheel, a moving hull's part), its ends'
     * velocities from the body's motion: the body moves at `linear` and turns at `angular`
     * (radians per second about that axis) about `center`, so a point `p` moves at
     * `linear + angular × (p - center)`.
     */
    public static rigid(a: Vec3, b: Vec3, radius: number, center: Vec3, linear: Vec3, angular: Vec3): FluidCapsule {
        const at = (p: Vec3): Vec3 => {
            const r = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
            return [
                linear[0] + angular[1] * r[2] - angular[2] * r[1],
                linear[1] + angular[2] * r[0] - angular[0] * r[2],
                linear[2] + angular[0] * r[1] - angular[1] * r[0],
            ];
        };
        return new FluidCapsule([...a], [...b], radius, at(a), at(b));
    }

    /**
     * The same capsule in a space scaled by `s`, with its velocities scaled by `velocityScale`
     * (e.g. `s` over the simulation's time scale).
     */
    public scaled(s: number, velocityScale: number): FluidCapsule {
        const m = (v: Vec3, k: number): Vec3 => [v[0] * k, v[1] * k, v[2] * k];
        return new FluidCapsule(m(this.a, s), m(this.b, s), this.radius * s, m(this.velocityA, velocityScale), m(this.velocityB, velocityScale), this.expansion * velocityScale);
    }

    /** Write it at float `offset` of `out`, in the GPU's layout. */
    public pack(out: Float32Array, offset: number): void {
        out.set([...this.a, this.radius, ...this.b, this.expansion, ...this.velocityA, 0, ...this.velocityB, 0], offset);
    }
}

/** How the colliders treat the particles they touch. */
interface FluidCollidersOptions {
    /** The fraction of the speed into a collider (relative to its surface) that bounces back. */
    restitution: number;
    /**
     * The fraction of the speed along a collider (relative to its surface) lost at each contact:
     * 1 carries the touching fluid along with it, 0 lets it slide.
     */
    drag: number;
}

const DEFAULT_COLLIDERS_OPTIONS: FluidCollidersOptions = { restitution: 0.2, drag: 0.3 };

/**
 * The colliders' pass: up to `capacity` capsules, placed with `set` (once a frame: every substep
 * of the frame sees the same ones).
 */
class FluidColliders implements FluidSubstepPass {
    public options: FluidCollidersOptions;
    private readonly _capacity: number;
    private _count = 0;
    private readonly uniformData = new ArrayBuffer(COLLIDERS_UNIFORM_BYTES);
    private readonly uniform: ComputeBuffer;
    private readonly capsulesF32: Float32Array;
    private readonly capsules: ComputeBuffer;
    private readonly pass: FluidSubstepPass;

    constructor(sim: FluidSimulation, capacity: number, options: Partial<FluidCollidersOptions> = {}) {
        this.options = { ...DEFAULT_COLLIDERS_OPTIONS, ...options };
        this._capacity = Math.max(Math.floor(capacity), 1);
        this.uniform = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_UNIFORM,
            usage: BufferBase.BUFFER_USAGE_UNIFORM | BufferBase.BUFFER_USAGE_COPY_DST,
            buffer: new Float32Array(this.uniformData),
        });
        this.capsulesF32 = new Float32Array(this._capacity * CAPSULE_FLOATS);
        this.capsules = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_READ_ONLY_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE | BufferBase.BUFFER_USAGE_COPY_DST,
            buffer: this.capsulesF32,
        });
        this.pass = sim.substepPass(COLLIDERS_WGSL, [this.uniform, this.capsules]);
        this.uploadUniform();
    }

    /** Place the capsules (the first `capacity` of them) for the next steps. */
    public set(capsules: readonly FluidCapsule[]): void {
        this._count = Math.min(capsules.length, this._capacity);
        for (let k = 0; k < this._count; k++) {
            capsules[k].pack(this.capsulesF32, k * CAPSULE_FLOATS);
        }
        if (this._count > 0) this.capsules.needsUpdate = true;
        this.uploadUniform();
    }

    /** Upload `options` after changing them. */
    public uploadUniform(): void {
        const u = new Uint32Array(this.uniformData);
        const f = new Float32Array(this.uniformData);
        u[0] = this._count;
        f[1] = this.options.restitution;
        f[2] = this.options.drag;
        this.uniform.needsUpdate = true;
    }

    public get capacity(): number {
        return this._capacity;
    }

    /** The capsules placed by the last `set`. */
    public get count(): number {
        return this._count;
    }

    public dispatch(pass: GPUComputePassEncoder, particleCount: number, device: GPUDevice): void {
        if (this._count === 0) return;
        this.pass.dispatch(pass, particleCount, device);
    }
}

export { FluidCapsule, FluidColliders, COLLIDERS_WGSL, CAPSULE_FLOATS, COLLIDERS_UNIFORM_BYTES };
export type { FluidCollidersOptions };
