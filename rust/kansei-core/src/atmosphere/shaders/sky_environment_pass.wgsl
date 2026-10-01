// Sky environment cubemap, one mip per dispatch: the sky's radiance around the camera (the
// sky-view LUT with the clouds in front of it above the horizon, the air and a Lambertian ground
// below it, and the capture fog in front of all of it; no sun disk, whose highlight belongs to the
// directional light), GGX-prefiltered for
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
@group(0) @binding(8) var<uniform> capture : SkyCapture;

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
// sky and the sun (as the sky lighting's lower hemisphere), then what the capture adds.
fn environmentRadiance(d: vec3f) -> vec3f {
    var lum = skyWithClouds(d);
    let up = normalize(frame.cameraPos);
    if (!captureCoversGround() && dot(d, up) < horizonCos(length(frame.cameraPos))) {
        let e = skyIrradiance(skyLighting, up) + skyLighting.sunIlluminance.rgb * max(dot(up, skyLighting.sunDirection.xyz), 0.0)
              + skyLighting.moonIlluminance.rgb * max(dot(up, skyLighting.moonDirection.xyz), 0.0);
        lum += frame.skyLightGroundAlbedo / PI * e;
    }
    return capturedSky(d, lum, skyLighting.distantSkyLight.rgb);
}

fn radicalInverse(i: u32) -> f32 {
    var b = (i << 16u) | (i >> 16u);
    b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
    b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
    b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
    b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
    return f32(b) * 2.3283064365386963e-10;
}

// GGX sample i of envPass.samples around n (tangent frame tx, ty): the radiance it brings,
// weighted by N.L, and that weight (zero below the surface).
fn ggxSample(n: vec3f, tx: vec3f, ty: vec3f, i: u32) -> vec4f {
    let a = envPass.roughness * envPass.roughness;
    let xi = vec2f((f32(i) + 0.5) / f32(envPass.samples), radicalInverse(i));
    let phi = 2.0 * PI * xi.x;
    let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    let h = tx * (sinTheta * cos(phi)) + ty * (sinTheta * sin(phi)) + n * cosTheta;
    let l = 2.0 * dot(n, h) * h - n;
    let nl = dot(n, l);
    if (nl <= 0.0) { return vec4f(0.0); }
    return vec4f(environmentRadiance(l) * nl, nl);
}

fn tangentFrame(n: vec3f) -> mat2x3f {
    // +z as up, but where n is +-z (the 1x1 mip's texel centres), +x
    let tx = normalize(cross(select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.z) < 0.999), n));
    return mat2x3f(tx, cross(n, tx));
}

// One invocation per texel: the mirror mip (and any mip, sampling alone).
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
    let t = tangentFrame(n);
    var sum = vec4f(0.0);
    for (var i = 0u; i < envPass.samples; i++) {
        sum += ggxSample(n, t[0], t[1], i);
    }
    textureStore(envOut, gid.xy, gid.z, vec4f(sum.rgb / max(sum.a, 1e-6), 1.0));
}

const ROUGH_GROUP : u32 = 32u;
var<workgroup> partial : array<vec4f, ROUGH_GROUP>;

// The rough mips: one workgroup per texel (xy: texel, z: face), its invocations sharing the GGX
// samples. The small mips have too few texels to fill the GPU at one invocation each, so each
// texel's whole sample loop would run in turn; shared, the loops are ROUGH_GROUP times shorter.
@compute @workgroup_size(32, 1, 1)
fn rough(@builtin(workgroup_id) wg : vec3u, @builtin(local_invocation_index) lid : u32) {
    let size = textureDimensions(envOut);
    let texel = wg.xy;
    let n = cubeDirection(wg.z, (vec2f(texel) + 0.5) / vec2f(size));
    let t = tangentFrame(n);
    var sum = vec4f(0.0);
    for (var i = lid; i < envPass.samples; i += ROUGH_GROUP) {
        sum += ggxSample(n, t[0], t[1], i);
    }
    partial[lid] = sum;
    workgroupBarrier();
    for (var stride = ROUGH_GROUP / 2u; stride > 0u; stride /= 2u) {
        if (lid < stride) {
            partial[lid] += partial[lid + stride];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        textureStore(envOut, texel, wg.z, vec4f(partial[0].rgb / max(partial[0].a, 1e-6), 1.0));
    }
}
