// Prepares an HDR environment (environment::HdrEnvironment), once at load and again when its
// settings change: the equirectangular image into a radiance cubemap (turned about +y, scaled by
// the intensity and the tint), that cubemap's box-filtered mips, the GGX-prefiltered environment
// cubemap SKY_ENVIRONMENT_WGSL samples (mip m of roughness m / (mips - 1); mip 0 is copied from
// the radiance), and the SkyLighting uniform the GI, the fog and the materials read (order-2 SH
// of the radiance). Prefixed with SKY_LIGHTING_WGSL.

struct HdrPrepare {
    rotation : vec4f,   // cos and sin of the turn about +y, the intensity
    tint     : vec4f,   // rgb
}

struct HdrMipPass {
    roughness : f32,
    samples   : u32,
    _pad0     : u32,
    _pad1     : u32,
}

const HDR_PI : f32 = 3.14159265;
// rgba16float's largest finite value, a little under
const HDR_MAX : f32 = 65000.0;

@group(0) @binding(0) var<uniform> prep : HdrPrepare;
@group(0) @binding(1) var equirect : texture_2d<f32>;
@group(0) @binding(2) var equirectSampler : sampler;
@group(0) @binding(3) var cubeOut : texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var cubeSrc : texture_2d_array<f32>;
@group(0) @binding(5) var radiance : texture_cube<f32>;
@group(0) @binding(6) var radianceSampler : sampler;
@group(0) @binding(7) var<uniform> mipPass : HdrMipPass;
@group(0) @binding(8) var<storage, read_write> skyLightingOut : SkyLighting;

// Direction through texel uv of a cube face, in the WebGPU (D3D/Vulkan) face layout:
// +X, -X, +Y, -Y, +Z, -Z, with t running down each face.
fn hdrCubeDirection(face: u32, uv: vec2f) -> vec3f {
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

// Where the equirectangular image holds direction d: its middle column looks along -z, the
// columns to its right toward +x, its top row straight up.
fn hdrEquirectUv(d: vec3f) -> vec2f {
    return vec2f(0.5 + atan2(d.x, -d.z) / (2.0 * HDR_PI), acos(clamp(d.y, -1.0, 1.0)) / HDR_PI);
}

// The radiance arriving from d: the image turned by the rotation (what it shows along -z turned
// toward +x), by the intensity and the tint.
fn hdrSource(d: vec3f) -> vec3f {
    let c = prep.rotation.x;
    let s = prep.rotation.y;
    let r = vec3f(c * d.x + s * d.z, d.y, c * d.z - s * d.x);
    let rgb = textureSampleLevel(equirect, equirectSampler, hdrEquirectUv(r), 0.0).rgb;
    return min(rgb * prep.rotation.z * prep.tint.rgb, vec3f(HDR_MAX));
}

// The radiance cube's mip 0, four samples a texel (an image finer than the face doesn't alias).
@compute @workgroup_size(8, 8, 1)
fn to_cube(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(cubeOut);
    if (gid.x >= size.x || gid.y >= size.y || gid.z >= 6u) { return; }
    var sum = vec3f(0.0);
    for (var k = 0u; k < 4u; k++) {
        let offset = vec2f(f32(k & 1u), f32(k >> 1u)) * 0.5 + 0.25;
        sum += hdrSource(hdrCubeDirection(gid.z, (vec2f(gid.xy) + offset) / vec2f(size)));
    }
    textureStore(cubeOut, gid.xy, gid.z, vec4f(sum * 0.25, 1.0));
}

// A box-filtered mip of the radiance cube from the one above it (cubeSrc).
@compute @workgroup_size(8, 8, 1)
fn downsample(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(cubeOut);
    if (gid.x >= size.x || gid.y >= size.y || gid.z >= 6u) { return; }
    let p = gid.xy * 2u;
    let c = textureLoad(cubeSrc, p, gid.z, 0) + textureLoad(cubeSrc, p + vec2u(1u, 0u), gid.z, 0)
          + textureLoad(cubeSrc, p + vec2u(0u, 1u), gid.z, 0) + textureLoad(cubeSrc, p + vec2u(1u, 1u), gid.z, 0);
    textureStore(cubeOut, gid.xy, gid.z, vec4f(c.rgb * 0.25, 1.0));
}

fn hdrRadicalInverse(i: u32) -> f32 {
    var b = (i << 16u) | (i >> 16u);
    b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
    b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
    b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
    b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
    return f32(b) * 2.3283064365386963e-10;
}

// A rough mip of the environment cube: the radiance around each texel's direction n under a GGX
// lobe of mipPass.roughness (Karis 2013's split sum, N = V = R), importance sampled with each
// sample read from the radiance mip whose texels match its share of the lobe (Krivanek and Colbert
// 2008's filtered importance sampling), so few samples stay smooth around small bright lights.
@compute @workgroup_size(8, 8, 1)
fn prefilter(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(cubeOut);
    if (gid.x >= size.x || gid.y >= size.y || gid.z >= 6u) { return; }
    let n = hdrCubeDirection(gid.z, (vec2f(gid.xy) + 0.5) / vec2f(size));
    // +z as up, but where n is +-z, +x
    let tx = normalize(cross(select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.z) < 0.999), n));
    let ty = cross(n, tx);
    let a = mipPass.roughness * mipPass.roughness;
    let a2 = a * a;
    let face = f32(textureDimensions(radiance).x);
    let texelSolidAngle = 4.0 * HDR_PI / (6.0 * face * face);
    let maxLod = f32(textureNumLevels(radiance) - 1u);
    var sum = vec4f(0.0);
    for (var i = 0u; i < mipPass.samples; i++) {
        let xi = vec2f((f32(i) + 0.5) / f32(mipPass.samples), hdrRadicalInverse(i));
        let phi = 2.0 * HDR_PI * xi.x;
        let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
        let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
        let h = tx * (sinTheta * cos(phi)) + ty * (sinTheta * sin(phi)) + n * cosTheta;
        let l = 2.0 * cosTheta * h - n;
        let nl = dot(n, l);
        if (nl <= 0.0) { continue; }
        // the sample's pdf over directions (N = V: D / 4) and so the solid angle it stands for
        let q = cosTheta * cosTheta * (a2 - 1.0) + 1.0;
        let pdf = a2 / (4.0 * HDR_PI * q * q);
        let sampleSolidAngle = 1.0 / (f32(mipPass.samples) * pdf + 1e-6);
        let lod = clamp(0.5 * log2(sampleSolidAngle / texelSolidAngle) + 1.0, 0.0, maxLod);
        sum += vec4f(textureSampleLevel(radiance, radianceSampler, l, lod).rgb * nl, nl);
    }
    textureStore(cubeOut, gid.xy, gid.z, vec4f(sum.rgb / max(sum.a, 1e-6), 1.0));
}

// 64 threads keep the shared sums (64 x 9 x 16 bytes) inside WebGPU's 16 KiB default
const SH_THREADS : u32 = 64u;
const SH_PHI : u32 = 128u;
const SH_COS : u32 = 64u;

var<workgroup> sharedSh : array<array<vec3f, 9>, SH_THREADS>;

// The SkyLighting: the radiance projected onto order-2 SH over equal-area cells (uniform in
// azimuth and in the cosine of the zenith angle), read from the radiance mip of about a cell's
// size. No sun or moon: the environment's lights are in its SH and its cubemap. One workgroup.
@compute @workgroup_size(64)
fn project_sh(@builtin(local_invocation_index) li : u32) {
    let cellSolidAngle = 4.0 * HDR_PI / f32(SH_PHI * SH_COS);
    let lod = clamp(log2(f32(textureDimensions(radiance).x) / 32.0), 0.0, f32(textureNumLevels(radiance) - 1u));
    var sh : array<vec3f, 9>;
    for (var i = 0u; i < 9u; i++) { sh[i] = vec3f(0.0); }
    for (var cell = li; cell < SH_PHI * SH_COS; cell += SH_THREADS) {
        let phi = (f32(cell % SH_PHI) + 0.5) / f32(SH_PHI) * 2.0 * HDR_PI;
        let cosT = 1.0 - 2.0 * (f32(cell / SH_PHI) + 0.5) / f32(SH_COS);
        let sinT = sqrt(max(1.0 - cosT * cosT, 0.0));
        let d = vec3f(sinT * cos(phi), cosT, sinT * sin(phi));
        let lum = textureSampleLevel(radiance, radianceSampler, d, lod).rgb * cellSolidAngle;
        var y : array<f32, 9>;
        skyShBasis(d, &y);
        for (var i = 0u; i < 9u; i++) { sh[i] += lum * y[i]; }
    }
    sharedSh[li] = sh;
    workgroupBarrier();
    for (var stride = SH_THREADS / 2u; stride > 0u; stride = stride / 2u) {
        if (li < stride) {
            for (var i = 0u; i < 9u; i++) { sharedSh[li][i] += sharedSh[li + stride][i]; }
        }
        workgroupBarrier();
    }
    if (li != 0u) { return; }
    var out : SkyLighting;
    for (var i = 0u; i < 9u; i++) { out.sh[i] = vec4f(sharedSh[0][i], 0.0); }
    // nothing in front of the scene's own lights (the height fog's sun visibility)
    out.sunIlluminance = vec4f(0.0, 0.0, 0.0, 1.0);
    out.sunDirection = vec4f(0.0, 1.0, 0.0, 0.0);
    out.moonIlluminance = vec4f(0.0);
    out.moonDirection = vec4f(0.0, 1.0, 0.0, 0.0);
    out.clearSkyUp = vec4f(skyIrradiance(out, vec3f(0.0, 1.0, 0.0)), 0.0);
    // the mean radiance
    out.distantSkyLight = vec4f(out.sh[0].rgb * 0.282095, 0.0);
    skyLightingOut = out;
}
