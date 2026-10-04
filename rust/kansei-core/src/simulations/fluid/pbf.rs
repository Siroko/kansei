//! Position Based Fluids (Macklin & Müller 2013, "Position Based Fluids"): the particles'
//! positions are predicted, then projected a few times onto a per-particle density constraint
//! (`C_i = rho_i / rho_0 - 1`, solved for a multiplier `lambda_i` with a relaxation `epsilon`),
//! with the tensile correction `s_corr` against clustering at the free surface; the velocity is
//! the move over the step, then XSPH viscosity and vorticity confinement act on it.
//!
//! An alternative to the SPH solver of [`FluidSimulation`](super::FluidSimulation), on its
//! buffers: the same particles, the same neighbour grid (built on the predicted positions, once
//! per substep), and the same [`FluidSubstepPass`](super::FluidSubstepPass)es (containers,
//! colliders), which run on the predicted positions and again after the projection. 3D only.

use bytemuck::{Pod, Zeroable};

/// Position Based Fluids settings (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PbfOptions {
    /// Constraint projections per substep.
    pub iterations: u32,
    /// Rest density: what the poly6 kernel (unit mass) sums to at a particle at rest. For a
    /// lattice fill that is `fluid::lattice_density(spacing, h)`, which tends to `1 / spacing³` only
    /// once the spacing is well under the smoothing radius.
    pub rest_density: f32,
    /// The constraint's relaxation (CFM): larger is softer and steadier.
    pub relaxation: f32,
    /// Tensile correction `s_corr = -k (W(r) / W(dq))^n`, `dq` as a fraction of the smoothing
    /// radius.
    pub scorr_k: f32,
    pub scorr_n: f32,
    pub scorr_dq: f32,
    /// XSPH viscosity: how much of the neighbours' mean relative velocity each particle takes.
    pub xsph: f32,
    /// Vorticity confinement strength.
    pub vorticity: f32,
    /// A cap on the particles' speed (simulation units per second; 0: none).
    pub max_speed: f32,
}

impl PbfOptions {
    pub const DEFAULT: Self = Self { iterations: 3, rest_density: 6.4, relaxation: 5.0, scorr_k: 0.05, scorr_n: 4.0, scorr_dq: 0.2, xsph: 0.1, vorticity: 0.0, max_speed: 0.0 };
}

impl Default for PbfOptions {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// GPU layout of `PbfParams` (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuPbf {
    rest_density: f32,
    relaxation: f32,
    scorr_k: f32,
    scorr_n: f32,
    scorr_w_dq: f32,
    xsph: f32,
    vorticity: f32,
    max_speed: f32,
}

impl GpuPbf {
    pub(crate) fn new(o: &PbfOptions, h: f32) -> Self {
        // poly6 at dq·h, normalised as the shaders' W
        let r = o.scorr_dq.clamp(0.0, 1.0) * h;
        let q = h * h - r * r;
        let w_dq = 315.0 / (64.0 * std::f32::consts::PI * h.powi(9)) * q * q * q;
        Self { rest_density: o.rest_density.max(1e-3), relaxation: o.relaxation.max(1e-6), scorr_k: o.scorr_k, scorr_n: o.scorr_n, scorr_w_dq: w_dq.max(1e-12), xsph: o.xsph, vorticity: o.vorticity, max_speed: o.max_speed }
    }
}

pub(crate) const PBF_PARAMS_WGSL: &str = r#"
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
"#;

/// The loop over a particle's neighbours in the sorted grid, around `BODY` (which sees `j`,
/// `diff = pos - sortedPositions[j].xyz`, `d2` and `h`, `h2`), skipping itself and anything past
/// the smoothing radius.
const NEIGHBORS: &str = r#"
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
                    BODY
                }
            }
        }
    }
"#;

fn neighbors(body: &str) -> String {
    NEIGHBORS.replace("BODY", body)
}

/// Kernels: poly6 `W` and the spiky gradient, normalised with `SimParams`' 3D factors.
const KERNELS: &str = r#"
fn poly6(d2: f32, h2: f32) -> f32 {
    let q = h2 - d2;
    return params.poly6Factor * q * q * q;
}
// grad of the spiky kernel at diff (length r): -(45 / pi h^6) (h - r)^2 diff / r
fn spikyGrad(diff: vec3<f32>, r: f32, h: f32) -> vec3<f32> {
    let v = h - r;
    return -params.spikyPow2DerivFactor * v * v * diff / r;
}
"#;

/// 1. Keep the start, apply gravity and damping, predict.
pub(crate) const PREDICT_WGSL: &str = r#"
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
"#;

fn lambda_wgsl() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> sortedPositions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellOffsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> lambdas: array<f32>;
@group(0) @binding(3) var<uniform> params: SimParams;
@group(0) @binding(4) var<uniform> pbf: PbfParams;
{KERNELS}
// 2. The density constraint's multiplier: lambda = -C / (sum |grad C|^2 + relaxation).
@compute @workgroup_size(__NEIGHBOR_WG__)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let slot = gid.x;
    if (slot >= params.particleCount) {{ return; }}
    let pos = sortedPositions[slot].xyz;
    var density = params.poly6Factor * pow(params.smoothingRadius, 6.0);
    var gradI = vec3<f32>(0.0);
    var sumGrad2 = 0.0;
    {loop}
    let c = max(density / pbf.restDensity - 1.0, 0.0);
    lambdas[slot] = -c / (sumGrad2 + dot(gradI, gradI) + pbf.relaxation);
}}
"#,
        loop = neighbors(
            r#"
                    let r = sqrt(d2);
                    density += poly6(d2, h2);
                    let g = spikyGrad(diff, r, h) / pbf.restDensity;
                    gradI += g;
                    sumGrad2 += dot(g, g);"#
        )
    )
}

fn delta_wgsl() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> sortedPositions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellOffsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> lambdas: array<f32>;
@group(0) @binding(3) var<storage, read_write> deltas: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: SimParams;
@group(0) @binding(5) var<uniform> pbf: PbfParams;
{KERNELS}
// 3. The position correction, with the tensile correction s_corr = -k (W / W(dq))^n.
@compute @workgroup_size(__NEIGHBOR_WG__)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let slot = gid.x;
    if (slot >= params.particleCount) {{ return; }}
    let pos = sortedPositions[slot].xyz;
    let li = lambdas[slot];
    var delta = vec3<f32>(0.0);
    {loop}
    deltas[slot] = vec4<f32>(delta / pbf.restDensity, 0.0);
}}
"#,
        loop = neighbors(
            r#"
                    let r = sqrt(d2);
                    let scorr = -pbf.scorrK * pow(poly6(d2, h2) / pbf.scorrWdq, pbf.scorrN);
                    delta += (li + lambdas[j] + scorr) * spikyGrad(diff, r, h);"#
        )
    )
}

/// 4. Apply the corrections (Jacobi: all at once, after all were computed).
pub(crate) const APPLY_WGSL: &str = r#"
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
"#;

/// 5. Back to the particles' own order.
pub(crate) const UNSORT_WGSL: &str = r#"
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
"#;

/// 6. The velocity is the move over the step, but a collider's (the particles a collider pushed
/// out this substep, marked w + 2: their move is the push, not a motion), and capped.
pub(crate) const VELOCITY_WGSL: &str = r#"
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
"#;

/// 7. Sorted copies of the new positions and velocities, for the velocity passes.
pub(crate) const GATHER_WGSL: &str = r#"
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
"#;

fn vorticity_wgsl() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> sortedPositions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> sortedVelocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cellOffsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> omega: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: SimParams;
{KERNELS}
// 8. Vorticity: omega = sum (v_j - v_i) x grad W.
@compute @workgroup_size(__NEIGHBOR_WG__)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let slot = gid.x;
    if (slot >= params.particleCount) {{ return; }}
    let pos = sortedPositions[slot].xyz;
    let vi = sortedVelocities[slot].xyz;
    var w = vec3<f32>(0.0);
    {loop}
    omega[slot] = vec4<f32>(w, length(w));
}}
"#,
        loop = neighbors(
            r#"
                    w += cross(sortedVelocities[j].xyz - vi, spikyGrad(diff, sqrt(d2), h));"#
        )
    )
}

fn xsph_wgsl() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> sortedPositions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> sortedVelocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cellOffsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> omega: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> sortedIndices: array<u32>;
@group(0) @binding(5) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(6) var<uniform> params: SimParams;
@group(0) @binding(7) var<uniform> pbf: PbfParams;
{KERNELS}
// 9. XSPH viscosity, v += c sum (v_j - v_i) W, and vorticity confinement, v += dt eps (N x omega)
// with N the direction up the gradient of |omega|.
@compute @workgroup_size(__NEIGHBOR_WG__)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let slot = gid.x;
    if (slot >= params.particleCount) {{ return; }}
    let pos = sortedPositions[slot].xyz;
    let vi = sortedVelocities[slot].xyz;
    var xsph = vec3<f32>(0.0);
    var eta = vec3<f32>(0.0);
    {loop}
    var v = vi + pbf.xsph * xsph;
    let len = length(eta);
    if (pbf.vorticity > 0.0 && len > 1e-6) {{
        v += params.dt * pbf.vorticity * cross(eta / len, omega[slot].xyz);
    }}
    let idx = sortedIndices[slot];
    velocities[idx] = vec4<f32>(v, velocities[idx].w);
}}
"#,
        loop = neighbors(
            r#"
                    xsph += (sortedVelocities[j].xyz - vi) * poly6(d2, h2) / pbf.restDensity;
                    eta += omega[j].w * spikyGrad(diff, sqrt(d2), h);"#
        )
    )
}

/// Every PBF shader, complete (with `SimParams` and `PbfParams`), for validation and pipelines.
pub(crate) fn shader_sources(sim_params: &str, neighbor_wg: u32) -> Vec<(&'static str, String)> {
    let full = |code: String| format!("{sim_params}\n{PBF_PARAMS_WGSL}\n{code}").replace("__NEIGHBOR_WG__", &neighbor_wg.to_string());
    vec![
        ("predict", full(PREDICT_WGSL.into())),
        ("lambda", full(lambda_wgsl())),
        ("delta", full(delta_wgsl())),
        ("apply", full(APPLY_WGSL.into())),
        ("unsort", full(UNSORT_WGSL.into())),
        ("velocity", full(VELOCITY_WGSL.into())),
        ("gather", full(GATHER_WGSL.into())),
        ("vorticity", full(vorticity_wgsl())),
        ("xsph", full(xsph_wgsl())),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shaders_validate_and_the_params_layout_matches() {
        let sim_params = include_str!("shaders/sim-params.wgsl");
        for (name, code) in shader_sources(sim_params, 64) {
            let module = naga::front::wgsl::parse_str(&code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&code)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap_or_else(|e| panic!("{name}: {e:?}"));
            let span = module.types.iter().find_map(|(_, t)| match (&t.name, &t.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == "PbfParams" => Some(*span as usize),
                _ => None,
            });
            assert_eq!(span, Some(std::mem::size_of::<GpuPbf>()), "{name}");
        }
    }
}
