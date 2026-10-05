// Ray-traced reflections (rt::RtReflectionsEffect), trace: at a reduced resolution (one pixel of
// each block a frame, rtJitter), the reflection of a surface whose material wrote an F0: a ray
// through the grid of triangles (rt_grid.wgsl), the hit lit by the voxels (the clipmap or the
// volume, the light leaving the surface there: the voxels as the surface cache); past the
// grid's box, or on a miss, a narrow cone through the voxels, then the sky. Glossy surfaces jitter
// the ray over their lobe (the history averages it). Output: rgb the radiance, a 1 where traced
// (with the cost view, 1 + the cells and triangles the ray visited), 0 elsewhere. Prefixed with
// SKY_LIGHTING_WGSL, rt_reflect_common.wgsl, RT_GRID_WGSL with its bindings in group 1, the voxel
// source (srcVoxelSize, srcHitRadiance, srcCone) and `kansei_rt_covered`.

@group(0) @binding(0) var<uniform> rp : RtReflectParams;
@group(0) @binding(1) var depthTex : texture_depth_2d;
@group(0) @binding(2) var normalTex : texture_2d<f32>;
@group(0) @binding(3) var albedoTex : texture_2d<f32>;
@group(0) @binding(4) var outTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> sky : SkyLighting;
// rays, hits, cells, triangles tested, the costliest ray (RT_REFLECT_STATS)
@group(1) @binding(5) var<storage, read_write> rtStats : array<atomic<u32>, 8>;

fn rtHash(p: vec2u, frame: u32) -> vec2f {
    var h = p.x * 1973u + p.y * 9277u + frame * 26699u;
    h = (h ^ (h >> 15u)) * 0x2c1b3c6du;
    h = (h ^ (h >> 12u)) * 0x297a2d39u;
    let a = h ^ (h >> 15u);
    let b = a * 0x9e3779b9u;
    return vec2f(f32(a & 0xffffu), f32(b >> 16u)) / 65536.0;
}

// A direction round `r`, uniform over the cone of half-angle atan(tanHalf).
fn rtGlossy(r: vec3f, tanHalf: f32, p: vec2u, frame: u32) -> vec3f {
    let xi = rtHash(p, frame);
    let phi = 6.2831853 * xi.x;
    let cosMax = 1.0 / sqrt(1.0 + tanHalf * tanHalf);
    let cosT = 1.0 - xi.y * (1.0 - cosMax);
    let sinT = sqrt(max(1.0 - cosT * cosT, 0.0));
    // an orthonormal basis round r (Duff et al. 2017)
    let s = select(-1.0, 1.0, r.z >= 0.0);
    let a = -1.0 / (s + r.z);
    let b = r.x * r.y * a;
    let t = vec3f(1.0 + s * r.x * r.x * a, s * b, -s * r.x);
    let bt = vec3f(b, s + r.y * r.y * a, -r.y);
    return normalize(r * cosT + (t * cos(phi) + bt * sin(phi)) * sinT);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= rp.traceSize)) { return; }
    let px = min(gid.xy * rp.downscale + rtJitter(rp.frame, rp.downscale), vec2u(rp.fullSize) - 1u);
    let s = rtSurface(px);
    if (!s.valid) {
        textureStore(outTex, gid.xy, vec4f(0.0));
        return;
    }
    let eye = rp.invView[3].xyz;
    var r = reflect(normalize(s.world - eye), s.n);
    let lobe = rtLobeTan(s.roughness);
    if (lobe > 1e-3) {
        r = rtGlossy(r, lobe, gid.xy, rp.frame);
    }
    // above the surface (a grazing ray turned under it by the lobe, or by an interpolated normal)
    let rise = dot(r, s.n);
    if (rise < 0.02) {
        r = normalize(r + s.n * (0.02 - rise));
    }
    // off the surface by a little of a cell, more far away (the depth's precision)
    let origin = s.world + s.n * (0.05 * kansei_rt_grid.cell + 1e-3 * distance(eye, s.world));
    var flags = KANSEI_RT_SOLID;
    if ((rp.flags & RT_REFLECT_ALPHA) != 0u) {
        flags = 0u;
    }
    var hit : KanseiRtHit;
    var exit = 0.0;
    if ((rp.flags & RT_REFLECT_GRID) != 0u && kansei_rt_contains(origin)) {
        hit = kansei_rt_trace(origin, r, 0.0, rp.maxDistance, flags);
        exit = kansei_rt_exit(origin, r);
    }
    var radiance = vec3f(0.0);
    if (hit.found) {
        radiance = srcHitRadiance(origin + r * hit.t, hit.normal);
    } else {
        // past the grid (or outside it): the voxels from where the ray left it, then the sky
        let size = srcVoxelSize(origin);
        let c = srcCone(origin, r, s.n, max(rp.coneTan, lobe), max(exit, size), rp.maxDistance, rp.coneSteps);
        radiance = c.rgb + c.a * rp.skyScale * skyRadiance(sky, r);
    }
    let cost = hit.cells + hit.tests;
    if ((rp.flags & RT_REFLECT_STATS) != 0u) {
        atomicAdd(&rtStats[0], 1u);
        atomicAdd(&rtStats[1], select(0u, 1u, hit.found));
        atomicAdd(&rtStats[2], hit.cells);
        atomicAdd(&rtStats[3], hit.tests);
        atomicMax(&rtStats[4], cost);
    }
    var a = 1.0;
    if (rp.view == 3u) {
        a = 1.0 + f32(cost);
    }
    textureStore(outTex, gid.xy, vec4f(min(radiance, vec3f(60000.0)), a));
}
