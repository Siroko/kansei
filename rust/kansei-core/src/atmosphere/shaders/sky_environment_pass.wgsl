// Sky environment cubemap, one mip per dispatch: the sky's radiance around the camera (the
// sky-view LUT with the clouds in front of it above the horizon, the air and a Lambertian ground
// below it; no sun disk, whose highlight belongs to the directional light), GGX-prefiltered for
// the mip's roughness (Karis 2013's split sum, N = V = R). Mip 0 is the mirror; mip m has
// roughness m / (mips - 1).

struct EnvPass {
    roughness : f32,
    samples   : u32,
    _pad0     : u32,
    _pad1     : u32,
}

@group(0) @binding(0) var<uniform> atm : Atmosphere;
@group(0) @binding(1) var<uniform> frame : SkyFrame;
@group(0) @binding(2) var skyViewLut : texture_2d<f32>;
@group(0) @binding(3) var skyViewSampler : sampler;
@group(0) @binding(4) var<uniform> skyLighting : SkyLighting;
@group(0) @binding(5) var envOut : texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(6) var<uniform> envPass : EnvPass;
@group(0) @binding(7) var cloudMap : texture_2d<f32>;

// The sky with the clouds in front of it (cloud_map.wgsl)
fn skyWithClouds(d: vec3f) -> vec3f {
    let c = textureSampleLevel(cloudMap, skyViewSampler, cloudMapUv(d), 0.0);
    return skyViewLuminance(d) * (1.0 - c.a) + c.rgb;
}

// Direction through texel uv of a cube face, in the WebGPU (D3D/Vulkan) face layout:
// +X, -X, +Y, -Y, +Z, -Z, with t running down each face.
fn cubeDirection(face: u32, uv: vec2f) -> vec3f {
    let s = uv.x * 2.0 - 1.0;
    let t = uv.y * 2.0 - 1.0;
    switch (face) {
        case 0u: { return normalize(vec3f(1.0, -t, -s)); }
        case 1u: { return normalize(vec3f(-1.0, -t, s)); }
        case 2u: { return normalize(vec3f(s, 1.0, t)); }
        case 3u: { return normalize(vec3f(s, -1.0, -t)); }
        case 4u: { return normalize(vec3f(s, -t, 1.0)); }
        default: { return normalize(vec3f(-s, -t, -1.0)); }
    }
}

// Radiance arriving from d: the sky, or below the horizon the air in front of a ground lit by the
// sky and the sun (as the sky lighting's lower hemisphere).
fn environmentRadiance(d: vec3f) -> vec3f {
    var lum = skyWithClouds(d);
    let up = normalize(frame.cameraPos);
    if (dot(d, up) < horizonCos(length(frame.cameraPos))) {
        let e = skyIrradiance(skyLighting, up) + skyLighting.sunIlluminance.rgb * max(dot(up, skyLighting.sunDirection.xyz), 0.0)
              + skyLighting.moonIlluminance.rgb * max(dot(up, skyLighting.moonDirection.xyz), 0.0);
        lum += frame.skyLightGroundAlbedo / PI * e;
    }
    return lum;
}

fn radicalInverse(i: u32) -> f32 {
    var b = (i << 16u) | (i >> 16u);
    b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
    b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
    b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
    b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
    return f32(b) * 2.3283064365386963e-10;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(envOut);
    if (gid.x >= size.x || gid.y >= size.y || gid.z >= 6u) { return; }
    let n = cubeDirection(gid.z, (vec2f(gid.xy) + 0.5) / vec2f(size));

    if (envPass.roughness <= 0.0) {
        textureStore(envOut, gid.xy, gid.z, vec4f(environmentRadiance(n), 1.0));
        return;
    }

    // GGX importance sampling around n, weighted by N.L
    let a = envPass.roughness * envPass.roughness;
    let tangentX = normalize(cross(select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.y) < 0.999), n));
    let tangentY = cross(n, tangentX);
    var sum = vec3f(0.0);
    var weight = 0.0;
    for (var i = 0u; i < envPass.samples; i++) {
        let xi = vec2f((f32(i) + 0.5) / f32(envPass.samples), radicalInverse(i));
        let phi = 2.0 * PI * xi.x;
        let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
        let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
        let h = tangentX * (sinTheta * cos(phi)) + tangentY * (sinTheta * sin(phi)) + n * cosTheta;
        let l = 2.0 * dot(n, h) * h - n;
        let nl = dot(n, l);
        if (nl > 0.0) {
            sum += environmentRadiance(l) * nl;
            weight += nl;
        }
    }
    textureStore(envOut, gid.xy, gid.z, vec4f(sum / max(weight, 1e-6), 1.0));
}
