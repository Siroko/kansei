// Transmittance LUT: transmittance from a point at radius r to the top of the atmosphere along a
// ray with cosine zenith mu, in Bruneton's (x_mu, x_r) parameterisation. Depends only on the
// atmosphere, so it is rebuilt only when the atmosphere changes.

@group(0) @binding(0) var<uniform> atm : Atmosphere;
@group(0) @binding(1) var lutOut : texture_storage_2d<rgba16float, write>;

const SAMPLES : u32 = 40u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(lutOut);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let sizeF = vec2f(size);
    let uv = (vec2f(gid.xy) + 0.5) / sizeF;
    let xMu = texelUvToUnit(uv.x, sizeF.x);
    let xR = texelUvToUnit(uv.y, sizeF.y);

    let top = atm.topRadius;
    let bottom = atm.bottomRadius;
    let H = sqrt((top - bottom) * (top + bottom));
    let rho = H * xR;
    let r = sqrt(rho * rho + bottom * bottom);
    let dMin = top - r;
    let dMax = rho + H;
    let d = dMin + xMu * (dMax - dMin);
    var mu = 1.0;
    if (d > 0.0) {
        mu = clamp((H * H - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
    }

    // optical depth to the top boundary, midpoint rule
    let ro = vec3f(0.0, r, 0.0);
    let rd = vec3f(sqrt(max(1.0 - mu * mu, 0.0)), mu, 0.0);
    let dt = d / f32(SAMPLES);
    var depth = vec3f(0.0);
    for (var i = 0u; i < SAMPLES; i++) {
        let p = ro + rd * ((f32(i) + 0.5) * dt);
        depth += sampleMedium(length(p) - bottom).extinction * dt;
    }
    textureStore(lutOut, gid.xy, vec4f(exp(-depth), 1.0));
}
