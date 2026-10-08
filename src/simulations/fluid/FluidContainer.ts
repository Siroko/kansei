/**
 * A container of any outline in plan for the 3D fluid: vertical walls that follow a closed 2D
 * outline on the XZ plane, and a floor whose height varies across it (a lake bed, a pool with a
 * shallow end, a channel). Both are sampled from one grid of (signed distance to the outline,
 * floor height) nodes, built on the CPU by `PlanarContainerShape` and applied to the particles
 * after each substep's integration by `FluidContainer` (a `FluidSubstepPass`).
 *
 * The solver's own box bounds still apply; keep them around the container.
 *
 * Port of the Rust engine's `container.rs` and `fill.rs` (`rust/kansei-core/src/simulations/fluid/`),
 * whose WGSL it imports (Vite `?raw`); `fill.rs`'s `lattice_density` is `latticeDensity` in
 * `FluidSimulationParams.ts`.
 */
import containerWgsl from '../../../rust/kansei-core/src/simulations/fluid/shaders/container.wgsl?raw';
import { ComputeBuffer } from '../../buffers/ComputeBuffer';
import { BufferBase } from '../../buffers/BufferBase';
import { assemble } from '../../materials/shaders/ShaderUtils';
import { simParamsStruct } from './shaders/sim-params.wgsl';
import type { FluidSimulation, FluidSubstepPass } from './FluidSimulation';

type Vec2 = [number, number];

/** The container's substep pass: `SimParams`, then the container (Rust: `CONTAINER_WGSL`). */
const CONTAINER_WGSL: string = assemble([simParamsStruct, containerWgsl]);

/** Signed distance from `p` to the closed polygon `outline` (negative inside, even-odd rule). */
function signedDistance(outline: readonly Vec2[], p: Vec2): number {
    let d2 = Number.MAX_VALUE;
    let inside = false;
    let j = outline.length - 1;
    for (let i = 0; i < outline.length; i++) {
        const a = outline[j], b = outline[i];
        const e = [b[0] - a[0], b[1] - a[1]];
        const w = [p[0] - a[0], p[1] - a[1]];
        const t = Math.min(Math.max((w[0] * e[0] + w[1] * e[1]) / Math.max(e[0] * e[0] + e[1] * e[1], 1e-12), 0), 1);
        const q = [w[0] - e[0] * t, w[1] - e[1] * t];
        d2 = Math.min(d2, q[0] * q[0] + q[1] * q[1]);
        // crossing test on the edge a→b
        if ((a[1] > p[1]) !== (b[1] > p[1]) && p[0] < a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0])) {
            inside = !inside;
        }
        j = i;
    }
    return inside ? -Math.sqrt(d2) : Math.sqrt(d2);
}

/**
 * A closed outline on the XZ plane and a floor under it, sampled on a grid of nodes.
 *
 * `distance` is the signed distance to the outline (negative inside); the container's walls
 * stand `wallOffset` outside the outline, so the fluid can be confined to the outline plus a
 * strip around it (a shore) while the floor is shaped by the distance to the outline itself.
 */
class PlanarContainerShape {
    constructor(
        /** The grid's first node (x, z). */
        public origin: Vec2,
        /** The spacing of the nodes. */
        public cell: number,
        /** Nodes along x and z. */
        public dims: Vec2,
        /** Where the walls stand: this far outside the outline (negative: inside it). */
        public wallOffset: number,
        /** Per node, x fastest: (signed distance to the outline, floor height), interleaved. */
        public nodes: Float32Array,
    ) {}

    /**
     * Sample `outline` (a closed polygon on XZ, either winding, the last point joined to the
     * first) every `cell` over its bounds grown by `wallOffset` and `cell` on each side. The
     * floor at each node is `floor(x, z, distance)`, `distance` the node's signed distance to
     * the outline (negative inside).
     */
    public static fromOutline(outline: readonly Vec2[], cell: number, wallOffset: number, floor: (x: number, z: number, distance: number) => number): PlanarContainerShape {
        if (outline.length < 3) throw new Error('PlanarContainerShape: an outline needs 3 points or more');
        if (!(cell > 0)) throw new Error('PlanarContainerShape: the cell must be positive');
        const min: Vec2 = [Number.MAX_VALUE, Number.MAX_VALUE];
        const max: Vec2 = [-Number.MAX_VALUE, -Number.MAX_VALUE];
        for (const p of outline) {
            for (let k = 0; k < 2; k++) {
                min[k] = Math.min(min[k], p[k]);
                max[k] = Math.max(max[k], p[k]);
            }
        }
        const grow = Math.max(wallOffset, 0) + cell;
        const origin: Vec2 = [Math.fround(min[0] - grow), Math.fround(min[1] - grow)];
        const dims: Vec2 = [0, 1].map(k => Math.ceil((max[k] + grow - origin[k]) / cell) + 1) as Vec2;
        const nodes = new Float32Array(dims[0] * dims[1] * 2);
        for (let j = 0; j < dims[1]; j++) {
            for (let i = 0; i < dims[0]; i++) {
                const p: Vec2 = [origin[0] + i * cell, origin[1] + j * cell];
                const d = signedDistance(outline, p);
                const n = (j * dims[0] + i) * 2;
                nodes[n] = d;
                nodes[n + 1] = floor(p[0], p[1], d);
            }
        }
        return new PlanarContainerShape(origin, cell, dims, wallOffset, nodes);
    }

    /** Bilinear (distance, floor) at `(x, z)`, as the GPU pass samples it (clamped to the grid). */
    public sample(x: number, z: number): Vec2 {
        const fx = Math.min(Math.max((x - this.origin[0]) / this.cell, 0), this.dims[0] - 1);
        const fz = Math.min(Math.max((z - this.origin[1]) / this.cell, 0), this.dims[1] - 1);
        const i0 = Math.floor(fx), j0 = Math.floor(fz);
        const i1 = Math.min(i0 + 1, this.dims[0] - 1), j1 = Math.min(j0 + 1, this.dims[1] - 1);
        const tx = fx - i0, tz = fz - j0;
        const out: Vec2 = [0, 0];
        for (let k = 0; k < 2; k++) {
            const n = (i: number, j: number) => this.nodes[(j * this.dims[0] + i) * 2 + k];
            const a = n(i0, j0) + (n(i1, j0) - n(i0, j0)) * tx;
            const b = n(i0, j1) + (n(i1, j1) - n(i0, j1)) * tx;
            out[k] = a + (b - a) * tz;
        }
        return out;
    }

    /** Signed distance to the outline at `(x, z)` (negative inside). */
    public distance(x: number, z: number): number {
        return this.sample(x, z)[0];
    }

    /** Floor height at `(x, z)`. */
    public floor(x: number, z: number): number {
        return this.sample(x, z)[1];
    }

    /** Whether `(x, z)` is inside the walls. */
    public inside(x: number, z: number): boolean {
        return this.distance(x, z) < this.wallOffset;
    }

    /** The grid's extent: its first and last node (x, z). */
    public bounds(): [Vec2, Vec2] {
        const last: Vec2 = [0, 1].map(k => this.origin[k] + (this.dims[k] - 1) * this.cell) as Vec2;
        return [[...this.origin], last];
    }

    /**
     * The same shape in a space scaled by `s` (distances, floor and grid alike), e.g. a
     * simulation run at a scale of the world.
     */
    public scaled(s: number): PlanarContainerShape {
        return new PlanarContainerShape(
            [this.origin[0] * s, this.origin[1] * s],
            this.cell * s,
            [...this.dims],
            this.wallOffset * s,
            this.nodes.map(v => v * s),
        );
    }

    /**
     * A lattice `spacing` apart in the columns whose signed distance to the outline `inside`
     * accepts, from half a spacing over the floor up to `top`: the water the container holds
     * to that level (4 floats a particle, w = 1). Its length / 4 is also how many particles
     * fill it to there.
     */
    public lattice(spacing: number, top: number, inside: (distance: number) => boolean): Float32Array {
        const [lo, hi] = this.bounds();
        const particles: number[] = [];
        // stepped in f32, as the Rust engine does, so both fill the same columns and layers
        const f = Math.fround;
        const step = f(spacing);
        for (let z = f(lo[1]); z <= hi[1]; z = f(z + step)) {
            for (let x = f(lo[0]); x <= hi[0]; x = f(x + step)) {
                const [d, floor] = this.sample(x, z);
                if (inside(d)) {
                    for (let y = f(floor + step * 0.5); y < top; y = f(y + step)) {
                        particles.push(x, y, z, 1);
                    }
                }
            }
        }
        return new Float32Array(particles);
    }

    /**
     * Blur the floor (a box filter `radius` nodes each way, along x then z, twice). A floor
     * shaped by the distance to the outline creases outside its concave stretches; this smooths
     * the creases out.
     */
    public smoothFloor(radius: number): void {
        const [w, h] = this.dims;
        const r = Math.floor(radius);
        for (let pass = 0; pass < 2; pass++) {
            for (let axis = 0; axis < 2; axis++) {
                const floor = new Float32Array(w * h);
                for (let k = 0; k < w * h; k++) floor[k] = this.nodes[k * 2 + 1];
                for (let j = 0; j < h; j++) {
                    for (let i = 0; i < w; i++) {
                        let sum = 0, n = 0;
                        for (let k = -r; k <= r; k++) {
                            const x = axis === 0 ? i + k : i;
                            const z = axis === 0 ? j : j + k;
                            if (x >= 0 && x < w && z >= 0 && z < h) {
                                sum += floor[z * w + x];
                                n += 1;
                            }
                        }
                        this.nodes[(j * w + i) * 2 + 1] = sum / n;
                    }
                }
            }
        }
    }
}

/**
 * `count` particles (4 floats each, w = 1) on a cubic lattice through the box `lo`..`hi`, its
 * spacing chosen so they fill it, each coordinate moved by up to `jitter` / 2 of a spacing (the
 * same pseudo-random offsets every call) so the lattice does not stay a crystal. When the
 * rounding leaves the lattice short of `count`, further layers fill in half a cell higher.
 *
 * The rows nearest +z come first: a renderer that draws particles in their order, seen from +z,
 * then draws them roughly front to back, and the depth test spares the shading of those behind.
 */
function fillBox(count: number, lo: [number, number, number], hi: [number, number, number], jitter: number): Float32Array {
    const size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    const spacing = Math.cbrt(size[0] * size[1] * size[2] / Math.max(count, 1));
    const cells = size.map(s => Math.max(Math.floor(s / spacing), 1));
    let rng = 12345;
    const offset = () => {
        rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0;
        return ((rng >>> 8) / 16777216 - 0.5) * spacing * jitter;
    };
    const positions = new Float32Array(count * 4);
    let k = 0;
    for (let layer = 0; k < count; layer++) {
        for (let z = cells[2] - 1; z >= 0 && k < count; z--) {
            for (let y = 0; y < cells[1] && k < count; y++) {
                for (let x = 0; x < cells[0] && k < count; x++) {
                    const lift = layer * spacing * 0.5;
                    // the offsets in Rust's order: x, y, z
                    const ox = offset(), oy = offset(), oz = offset();
                    positions.set([
                        lo[0] + (x + 0.5) * spacing + ox,
                        lo[1] + (y + 0.5) * spacing + lift + oy,
                        lo[2] + (z + 0.5) * spacing + oz,
                        1,
                    ], k * 4);
                    k++;
                }
            }
        }
    }
    return positions;
}

/** How the container treats the particles it stops. */
interface FluidContainerOptions {
    /** How far inside the walls and above the floor the particles are kept. */
    margin: number;
    /** The fraction of the speed into a wall or the floor that bounces back. */
    restitution: number;
    /** The fraction of the speed along a wall or the floor lost at each contact. */
    friction: number;
}

const DEFAULT_CONTAINER_OPTIONS: FluidContainerOptions = { margin: 0.05, restitution: 0.1, friction: 0.02 };

/** Bytes of the container's uniform (`Container` in `container.wgsl`). */
const CONTAINER_UNIFORM_BYTES = 48;

/**
 * The container's pass: keeps a `FluidSimulation`'s particles inside a `PlanarContainerShape`
 * (in the simulation's space), after each substep's integration. Pass it to
 * `FluidSimulation.updateBatched`'s `extra` (after any collider passes, so the walls and floor
 * have the last word).
 */
class FluidContainer implements FluidSubstepPass {
    public options: FluidContainerOptions;
    private readonly _shape: PlanarContainerShape;
    private readonly uniformF32 = new Float32Array(CONTAINER_UNIFORM_BYTES / 4);
    private readonly uniform: ComputeBuffer;
    private readonly pass: FluidSubstepPass;

    constructor(sim: FluidSimulation, shape: PlanarContainerShape, options: Partial<FluidContainerOptions> = {}) {
        this.options = { ...DEFAULT_CONTAINER_OPTIONS, ...options };
        this._shape = shape;
        this.uniform = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_UNIFORM,
            usage: BufferBase.BUFFER_USAGE_UNIFORM | BufferBase.BUFFER_USAGE_COPY_DST,
            buffer: this.uniformF32,
        });
        const nodes = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_READ_ONLY_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: shape.nodes,
        });
        this.pass = sim.substepPass(CONTAINER_WGSL, [this.uniform, nodes]);
        this.upload();
    }

    public get shape(): PlanarContainerShape {
        return this._shape;
    }

    /** Upload `options` after changing them. */
    public upload(): void {
        const s = this._shape;
        const f = this.uniformF32;
        const u = new Uint32Array(f.buffer);
        f[0] = s.origin[0]; f[1] = s.origin[1];
        f[2] = s.cell;
        f[3] = s.wallOffset;
        u[4] = s.dims[0]; u[5] = s.dims[1];
        f[6] = this.options.margin;
        f[7] = this.options.restitution;
        f[8] = this.options.friction;
        this.uniform.needsUpdate = true;
    }

    public dispatch(pass: GPUComputePassEncoder, particleCount: number, device: GPUDevice): void {
        this.pass.dispatch(pass, particleCount, device);
    }
}

export { PlanarContainerShape, FluidContainer, signedDistance, fillBox, CONTAINER_WGSL, CONTAINER_UNIFORM_BYTES };
export type { FluidContainerOptions };
