import { simParamsStruct } from './sim-params.wgsl';

/**
 * Position Based Fluids' shaders (Macklin & Müller 2013), ported from the Rust engine's
 * `rust/kansei-core/src/simulations/fluid/pbf.rs`, which keeps them inline (so they cannot be
 * imported with `?raw`) and validates them with naga: keep the two in step.
 */

/** Workgroup size of the neighbour-search passes (lambda, delta, vorticity, XSPH). */
export const PBF_NEIGHBOR_WG = 64;

/** `PbfParams`' size in floats (32 bytes; Rust `GpuPbf`). */
export const PBF_PARAMS_FLOATS = 8;

const pbfParamsStruct = /* wgsl */`
struct PbfParams {
    restDensity: f32,
    relaxation: f32,
    scorrK: f32,
    scorrN: f32,
    scorrWdq: f32,
    xsph: f32,
    vorticity: f32,
    maxSpeed: f32,
};
`;

/**
 * The loop over a particle's neighbours in the sorted grid, around `body` (which sees `j`,
 * `diff = pos - sortedPositions[j].xyz`, `d2` and `h`, `h2`), skipping itself and anything past
 * the smoothing radius.
 */
const neighbors = (body: string) => /* wgsl */`
    let coord = getCellCoord(pos, params);
    let h = params.smoothingRadius;
    let h2 = h * h;
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let nc = coord + vec3<i32>(dx, dy, dz);
                if (any(nc < vec3<i32>(0)) || any(nc >= vec3<i32>(params.gridDims))) { continue; }
                let cell = cellHash(nc, params);
                let cellEnd = select(cellOffsets[cell + 1u], params.particleCount, cell + 1u >= params.totalCells);
                for (var j = cellOffsets[cell]; j < cellEnd; j++) {
                    if (j == slot) { continue; }
                    let diff = pos - sortedPositions[j].xyz;
                    let d2 = dot(diff, diff);
                    if (d2 >= h2 || d2 < 1e-12) { continue; }
                    ${body}
                }
            }
        }
    }
`;

/** Kernels: poly6 `W` and the spiky gradient, normalised with `SimParams`' 3D factors. */
const kernels = /* wgsl */`
fn poly6(d2: f32, h2: f32) -> f32 {
    let q = h2 - d2;
    return params.poly6Factor * q * q * q;
}
// grad of the spiky kernel at diff (length r): -(45 / pi h^6) (h - r)^2 diff / r
fn spikyGrad(diff: vec3<f32>, r: f32, h: f32) -> vec3<f32> {
    let v = h - r;
    return -params.spikyPow2DerivFactor * v * v * diff / r;
}
`;

/** 1. Keep the start, apply gravity and damping, predict. */
const predict = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> previous: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }
    let p = positions[idx];
    var v = velocities[idx];
    previous[idx] = p;
    // w = 1: held by an external attractor (no gravity), as in the SPH solver
    if (v.w < 0.5) {
        if (params.radialGravity > 0.5) {
            let to = params.gravityCenter - p.xyz;
            v = vec4<f32>(v.xyz + to / (length(to) + 1e-4) * length(params.gravity) * params.dt, v.w);
        } else {
            v = vec4<f32>(v.xyz + params.gravity * params.dt, v.w);
        }
    }
    v = vec4<f32>(v.xyz * params.damping, v.w);
    velocities[idx] = v;
    positions[idx] = vec4<f32>(clamp(p.xyz + v.xyz * params.dt, params.worldBoundsMin, params.worldBoundsMax), p.w);
}
`;

/** 2. The density constraint's multiplier: lambda = -C / (sum |grad C|^2 + relaxation). */
const lambda = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> sortedPositions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellOffsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> lambdas: array<f32>;
@group(0) @binding(3) var<uniform> params: SimParams;
@group(0) @binding(4) var<uniform> pbf: PbfParams;
${kernels}
@compute @workgroup_size(__NEIGHBOR_WG__)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.particleCount) { return; }
    let pos = sortedPositions[slot].xyz;
    var density = params.poly6Factor * pow(params.smoothingRadius, 6.0);
    var gradI = vec3<f32>(0.0);
    var sumGrad2 = 0.0;
    ${neighbors(`
                    let r = sqrt(d2);
                    density += poly6(d2, h2);
                    let g = spikyGrad(diff, r, h) / pbf.restDensity;
                    gradI += g;
                    sumGrad2 += dot(g, g);`)}
    let c = max(density / pbf.restDensity - 1.0, 0.0);
    lambdas[slot] = -c / (sumGrad2 + dot(gradI, gradI) + pbf.relaxation);
}
`;

/** 3. The position correction, with the tensile correction s_corr = -k (W / W(dq))^n. */
const delta = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> sortedPositions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellOffsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> lambdas: array<f32>;
@group(0) @binding(3) var<storage, read_write> deltas: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: SimParams;
@group(0) @binding(5) var<uniform> pbf: PbfParams;
${kernels}
@compute @workgroup_size(__NEIGHBOR_WG__)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.particleCount) { return; }
    let pos = sortedPositions[slot].xyz;
    let li = lambdas[slot];
    var delta = vec3<f32>(0.0);
    ${neighbors(`
                    let r = sqrt(d2);
                    let scorr = -pbf.scorrK * pow(poly6(d2, h2) / pbf.scorrWdq, pbf.scorrN);
                    delta += (li + lambdas[j] + scorr) * spikyGrad(diff, r, h);`)}
    deltas[slot] = vec4<f32>(delta / pbf.restDensity, 0.0);
}
`;

/** 4. Apply the corrections (Jacobi: all at once, after all were computed). */
const apply = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> sortedPositions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> deltas: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.particleCount) { return; }
    let p = sortedPositions[slot];
    sortedPositions[slot] = vec4<f32>(clamp(p.xyz + deltas[slot].xyz, params.worldBoundsMin, params.worldBoundsMax), p.w);
}
`;

/** 5. Back to the particles' own order. */
const unsort = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> sortedPositions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> sortedIndices: array<u32>;
@group(0) @binding(2) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.particleCount) { return; }
    positions[sortedIndices[slot]] = sortedPositions[slot];
}
`;

/**
 * 6. The velocity is the move over the step, but a collider's (the particles a collider pushed
 * out this substep, marked w + 2: their move is the push, not a motion), and capped.
 */
const velocity = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> previous: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: SimParams;
@group(0) @binding(4) var<uniform> pbf: PbfParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }
    let v4 = velocities[idx];
    var v = (positions[idx].xyz - previous[idx].xyz) / max(params.dt, 1e-6);
    var w = v4.w;
    if (w >= 1.5) {
        v = v4.xyz;
        w -= 2.0;
    }
    let speed = length(v);
    if (pbf.maxSpeed > 0.0 && speed > pbf.maxSpeed) {
        v *= pbf.maxSpeed / speed;
    }
    velocities[idx] = vec4<f32>(v, w);
}
`;

/** 7. Sorted copies of the new positions and velocities, for the velocity passes. */
const gather = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> sortedIndices: array<u32>;
@group(0) @binding(3) var<storage, read_write> sortedPositions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> sortedVelocities: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.particleCount) { return; }
    let idx = sortedIndices[slot];
    sortedPositions[slot] = positions[idx];
    sortedVelocities[slot] = velocities[idx];
}
`;

/** 8. Vorticity: omega = sum (v_j - v_i) x grad W. */
const vorticity = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> sortedPositions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> sortedVelocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cellOffsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> omega: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: SimParams;
${kernels}
@compute @workgroup_size(__NEIGHBOR_WG__)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.particleCount) { return; }
    let pos = sortedPositions[slot].xyz;
    let vi = sortedVelocities[slot].xyz;
    var w = vec3<f32>(0.0);
    ${neighbors(`
                    w += cross(sortedVelocities[j].xyz - vi, spikyGrad(diff, sqrt(d2), h));`)}
    omega[slot] = vec4<f32>(w, length(w));
}
`;

/**
 * 9. XSPH viscosity, v += c sum (v_j - v_i) W, and vorticity confinement, v += dt eps (N x omega)
 * with N the direction up the gradient of |omega|.
 */
const xsph = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> sortedPositions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> sortedVelocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cellOffsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> omega: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> sortedIndices: array<u32>;
@group(0) @binding(5) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(6) var<uniform> params: SimParams;
@group(0) @binding(7) var<uniform> pbf: PbfParams;
${kernels}
@compute @workgroup_size(__NEIGHBOR_WG__)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.particleCount) { return; }
    let pos = sortedPositions[slot].xyz;
    let vi = sortedVelocities[slot].xyz;
    var xsph = vec3<f32>(0.0);
    var eta = vec3<f32>(0.0);
    ${neighbors(`
                    xsph += (sortedVelocities[j].xyz - vi) * poly6(d2, h2) / pbf.restDensity;
                    eta += omega[j].w * spikyGrad(diff, sqrt(d2), h);`)}
    var v = vi + pbf.xsph * xsph;
    let len = length(eta);
    if (pbf.vorticity > 0.0 && len > 1e-6) {
        v += params.dt * pbf.vorticity * cross(eta / len, omega[slot].xyz);
    }
    let idx = sortedIndices[slot];
    velocities[idx] = vec4<f32>(v, velocities[idx].w);
}
`;

const full = (code: string) =>
    `${simParamsStruct}\n${pbfParamsStruct}\n${code}`.replace(/__NEIGHBOR_WG__/g, String(PBF_NEIGHBOR_WG));

/** Every PBF shader, complete (with `SimParams` and `PbfParams`), by pass. */
export const pbfShaders = {
    predict: full(predict),
    lambda: full(lambda),
    delta: full(delta),
    apply: full(apply),
    unsort: full(unsort),
    velocity: full(velocity),
    gather: full(gather),
    vorticity: full(vorticity),
    xsph: full(xsph),
};
