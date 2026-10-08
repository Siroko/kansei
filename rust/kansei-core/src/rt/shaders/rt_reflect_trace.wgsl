// Ray-traced reflections (rt::RtReflectionsEffect), trace: at a reduced resolution (one pixel of
// each block a frame, rtJitter), the reflection of a surface whose material wrote an F0: a ray
// through the grid of triangles (rt_grid.wgsl), the hit lit as rt_reflect_hit.wgsl says (the lit
// image where the camera sees it, else the spot lights and the voxels, or the voxels alone: the
// surface cache); past the grid's box, or on a miss, a narrow cone through the voxels, then the
// sky. Glossy surfaces jitter the ray over their lobe (the history averages it). Output: rgb the
// radiance, a 1 where traced (with the cost view, 1 + the cells and triangles the ray visited), 0
// elsewhere. Prefixed with SKY_LIGHTING_WGSL, the spot light types, rt_reflect_common.wgsl,
// RT_GRID_WGSL with its bindings in group 1, `kansei_rt_covered`, the voxel source (srcVoxelSize,
// srcHitRadiance, srcCone, srcIrradiance) and rt_reflect_hit.wgsl.

@group(0) @binding(0) var<uniform> rp : RtReflectParams;
@group(0) @binding(1) var depthTex : texture_depth_2d;
@group(0) @binding(2) var normalTex : texture_2d<f32>;
@group(0) @binding(3) var albedoTex : texture_2d<f32>;
@group(0) @binding(4) var outTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> sky : SkyLighting;
// rays, hits, cells, triangles tested, the costliest ray (RT_REFLECT_STATS)
@group(1) @binding(5) var<storage, read_write> rtStats : array<atomic<u32>, 8>;
// the image the hits are looked up in has the glass drawn (rt_reflect_hit.wgsl)
const RT_SCREEN_HAS_GLASS : bool = true;

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
    let ray = rtSceneRay(origin, r, s.n, lobe, flags);
    let radiance = ray.radiance;
    let cost = ray.cost.x + ray.cost.y;
    if ((rp.flags & RT_REFLECT_STATS) != 0u) {
        atomicAdd(&rtStats[0], 1u);
        atomicAdd(&rtStats[1], select(0u, 1u, ray.found));
        atomicAdd(&rtStats[2], ray.cost.x);
        atomicAdd(&rtStats[3], ray.cost.y);
        atomicMax(&rtStats[4], cost);
    }
    var a = 1.0;
    if (rp.view == 3u) {
        a = 1.0 + f32(cost);
    }
    textureStore(outTex, gid.xy, vec4f(min(radiance, vec3f(60000.0)), a));
}
