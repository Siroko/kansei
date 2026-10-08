import type { FluidSimulation } from './FluidSimulation';

type Vec3 = [number, number, number];

const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** A unit vector perpendicular to the unit vector `d` (glam's `any_orthonormal_vector`). */
function anyOrthonormal(d: Vec3): Vec3 {
    const sign = d[2] >= 0 ? 1 : -1;
    const a = -1 / (sign + d[2]);
    const b = d[0] * d[1] * a;
    return [b, sign + d[1] * d[1] * a, -d[1]];
}

/**
 * Adding fluid at runtime: a round nozzle turns a stream (a hose, a spout, a cannon's muzzle)
 * into particles, which `FluidSimulation.emit` appends into the simulation's spare capacity
 * (`maxParticles` above the live count). Port of the Rust engine's `FluidNozzle`
 * (`rust/kansei-core/src/simulations/fluid/emitter.rs`).
 *
 * The nozzle lays the stream down in layers across its disc, a particle spacing apart both
 * across and along the stream, so the new water starts at about the density the rest of the
 * fluid is at: no burst from packed particles, no gaps. Each layer is placed as far along the
 * stream as it has travelled since it left within the step, and turned and jittered a little so
 * the stream does not read as a lattice.
 */
class FluidNozzle {
    /** The disc's centre and the stream's direction (normalised), in the simulation's space. */
    public origin: Vec3;
    public direction: Vec3 = [0, 1, 0];
    /** The disc's radius. */
    public radius: number;
    /** How fast the stream leaves (the simulation's units per simulated second). */
    public speed: number;
    /** The particles' spacing across and along the stream: the fluid's rest spacing. */
    public spacing: number;
    /** Random offsets of each particle, as a fraction of the spacing. */
    public jitter = 0.1;
    /**
     * Random deviation of each particle's velocity, as a fraction of `speed`: the stream
     * spreading as it flies.
     */
    public spread = 0.02;
    /** How far the stream has run since its last layer. */
    private travelled = 0;
    private seed = 0x9e3779b9;

    constructor(origin: Vec3, direction: Vec3, radius: number, speed: number, spacing: number) {
        this.origin = [...origin];
        this.radius = radius;
        this.speed = speed;
        this.spacing = spacing;
        this.aim(origin, direction);
    }

    /** Move and turn it (`direction` need not be normalised). */
    public aim(origin: Vec3, direction: Vec3): void {
        const len = Math.hypot(direction[0], direction[1], direction[2]);
        this.origin = [...origin];
        this.direction = len > 0 && Number.isFinite(len)
            ? [direction[0] / len, direction[1] / len, direction[2] / len]
            : [0, 1, 0];
    }

    /** The particles in one layer across the disc (before any is cut by the capacity). */
    public layerSize(): number {
        return this.disc(0).length;
    }

    /** Particles per simulated second at full flow. */
    public rate(): number {
        return this.layerSize() * this.speed / this.spacing;
    }

    /** Restart the stream: its next layer leaves at once. */
    public restart(): void {
        this.travelled = this.spacing;
    }

    /**
     * The disc's points, a spacing apart on a hexagonal pattern turned by `angle`, relative to
     * the centre and across the axis.
     */
    private disc(angle: number): Vec3[] {
        const d = this.direction;
        const u0 = anyOrthonormal(d);
        const v0 = cross(d, u0);
        const s = Math.sin(angle), c = Math.cos(angle);
        const u: Vec3 = [u0[0] * c + v0[0] * s, u0[1] * c + v0[1] * s, u0[2] * c + v0[2] * s];
        const v: Vec3 = [v0[0] * c - u0[0] * s, v0[1] * c - u0[1] * s, v0[2] * c - u0[2] * s];
        const h = this.spacing * 0.8660254;
        const rows = Math.floor(this.radius / h);
        const cols = Math.ceil(this.radius / this.spacing) + 1;
        const r2 = this.radius * this.radius + 1e-6;
        const points: Vec3[] = [];
        for (let j = -rows; j <= rows; j++) {
            const shift = ((j % 2) + 2) % 2 === 1 ? 0.5 : 0;
            for (let i = -cols; i <= cols; i++) {
                const x = (i + shift) * this.spacing, y = j * h;
                if (x * x + y * y <= r2) {
                    points.push([u[0] * x + v[0] * y, u[1] * x + v[1] * y, u[2] * x + v[2] * y]);
                }
            }
        }
        return points;
    }

    /** xorshift32, in [-1, 1). */
    private random(): number {
        let x = this.seed;
        x = (x ^ (x << 13)) >>> 0;
        x = (x ^ (x >>> 17)) >>> 0;
        x = (x ^ (x << 5)) >>> 0;
        this.seed = x;
        return (x >>> 8) / (1 << 23) - 1;
    }

    /**
     * The particles for `dt` simulated seconds of flow: positions and velocities, layer by layer,
     * the first to leave the farthest along.
     */
    public flow(dt: number): { positions: Vec3[]; velocities: Vec3[] } {
        const positions: Vec3[] = [];
        const velocities: Vec3[] = [];
        const d = this.direction;
        const o = this.origin;
        this.travelled += this.speed * Math.max(dt, 0);
        while (this.travelled >= this.spacing) {
            this.travelled -= this.spacing;
            const along = this.travelled;
            const angle = this.random() * Math.PI;
            for (const p of this.disc(angle)) {
                const js = this.jitter * this.spacing;
                const jitter: Vec3 = [this.random() * js, this.random() * js, this.random() * js];
                const ds = this.spread * this.speed;
                const deviation: Vec3 = [this.random() * ds, this.random() * ds, this.random() * ds];
                positions.push([
                    o[0] + p[0] + d[0] * along + jitter[0],
                    o[1] + p[1] + d[1] * along + jitter[1],
                    o[2] + p[2] + d[2] * along + jitter[2],
                ]);
                velocities.push([
                    d[0] * this.speed + deviation[0],
                    d[1] * this.speed + deviation[1],
                    d[2] * this.speed + deviation[2],
                ]);
            }
        }
        return { positions, velocities };
    }

    /**
     * `flow` for `dt` simulated seconds into `sim`, as much as its spare capacity takes.
     * Returns how many particles were added.
     */
    public emitInto(sim: FluidSimulation, dt: number): number {
        const { positions, velocities } = this.flow(dt);
        return positions.length > 0 ? sim.emit(positions, velocities) : 0;
    }
}

export { FluidNozzle };
