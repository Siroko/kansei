// Ray-traced diffuse GI (rt::RtDiffuseGiEffect), trace: one ray a pixel of the trace resolution,
// from the GBuffer's surface in a cosine-distributed direction (the frame's R2 sample, rotated
// per pixel).
//
// - Hybrid (the default): the ray walks the grid of triangles (rt_grid.wgsl), up to
//   nearDistance. A hit is lit by its exact direct light (the renderer's lights, shadowed by rays
//   through the grid or by the shadow maps) plus the indirect light the voxels hold round it (one
//   voxel cone in a cosine-distributed direction), or with RT_GI_HIT_DIRECT off by the voxels'
//   radiance alone (the voxels as the surface cache). A ray that leaves the grid's box (or its
//   near field) goes on as a voxel cone, then the sky.
// - Reference (RT_GI_REFERENCE): a path tracer through the grid alone, `bounces` vertices with
//   the direct light at each (shadow rays through the grid) and Russian roulette past the
//   second; past the grid's box the same voxel cone as the hybrid (the far field is shared).
//
// Output: rgb the radiance arriving along the ray (its mean over directions is the irradiance
// over pi), a 1 where the pixel holds a surface (1 + the rays' cost with the cost view), 0
// elsewhere; with RT_GI_ACCUMULATE also added to the running sums. Prefixed with
// SKY_LIGHTING_WGSL, SPOT_LIGHT_TYPES_WGSL, compute_shadows.wgsl, rt_gi_common.wgsl,
// RT_GRID_WGSL with its bindings in group 1, `kansei_rt_covered` and the voxel source
// (srcVoxelSize, srcHitRadiance, srcCone).

@group(0) @binding(21) var depthTex : texture_depth_2d;
@group(0) @binding(22) var normalTex : texture_2d<f32>;
@group(0) @binding(23) var outTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(24) var<uniform> sky : SkyLighting;
// running sums (rgb) and frames (a), a texel of the trace resolution each
@group(0) @binding(25) var<storage, read_write> accum : array<vec4f>;
// rays, hits, cells, triangles tested, the costliest pixel, shadow rays
@group(1) @binding(5) var<storage, read_write> giStats : array<atomic<u32>, 8>;

fn skyLight(d: vec3f) -> vec3f {
    if ((gp.flags & RT_GI_HAS_SKY) == 0u) { return vec3f(0.0); }
    return skyRadiance(sky, d) * gp.skyScale;
}

fn traceFlags() -> u32 {
    return select(KANSEI_RT_SOLID, 0u, (gp.flags & RT_GI_ALPHA) != 0u);
}

// Off a surface by a little of a cell (more far from the eye: the depth's precision).
fn surfaceBias(p: vec3f) -> f32 {
    return 0.05 * kansei_rt_grid.cell + 2e-4 * distance(gp.invView[3].xyz, p);
}

// Whether nothing in the grid lies between o and `tMax` along unit `d`; with `beyond` (a
// directional light), past that the shadow map at the point the ray stops: where it leaves the
// grid's box, or at `tMax` (the near field).
fn shadowRay(o: vec3f, d: vec3f, tMax: f32, beyond: bool, cost: ptr<function, u32>) -> f32 {
    if (!kansei_rt_contains(o)) {
        return 1.0;
    }
    let h = kansei_rt_trace(o, d, 0.0, tMax, KANSEI_RT_ANY_HIT | traceFlags());
    *cost += h.cells + h.tests;
    if ((gp.flags & RT_GI_STATS) != 0u) {
        atomicAdd(&giStats[5], 1u);
    }
    if (h.found) {
        return 0.0;
    }
    if (beyond && gp.hasShadowMap != 0u) {
        let q = o + d * min(kansei_rt_exit(o, d), tMax);
        if (dirShadowCovers(q)) {
            return dirShadowLookup(q);
        }
    }
    return 1.0;
}

// The irradiance the renderer's lights put on a surface at p (geometric normal ng, shading
// normal n): directional, point and spot lights, shadowed by rays through the grid
// (RT_GI_SHADOW_RAY; a spot light's toward a point of its disk) or by the shadow maps.
fn directLight(p: vec3f, ng: vec3f, n: vec3f, px: vec2u, dim: u32, cost: ptr<function, u32>) -> vec3f {
    var e = vec3f(0.0);
    let o = p + ng * surfaceBias(p);
    let rays = (gp.flags & RT_GI_SHADOW_RAY) != 0u;
    for (var i = 0u; i < gp.numDirLights; i++) {
        let dl = dirLights[i];
        let l = -normalize(dl.direction);
        let ndl = dot(n, l);
        if (ndl <= 0.0) { continue; }
        var vis = 1.0;
        let mapped = dl.shadowed != 0u && gp.hasShadowMap != 0u;
        if (rays) {
            vis = shadowRay(o, l, select(1e5, gp.nearDistance, gp.nearDistance > 0.0), mapped, cost);
        } else if (mapped && dirShadowCovers(o)) {
            vis = dirShadowLookup(o + l * kansei_rt_grid.cell);
        }
        e += dl.color * ndl * vis;
    }
    for (var i = 0u; i < gp.numPointLights; i++) {
        let pl = ptLights[i];
        let d = pl.position - p;
        let dist = length(d);
        if (dist > pl.radius || dist < 1e-4) { continue; }
        let ndl = dot(n, d / dist);
        if (ndl <= 0.0) { continue; }
        let falloff = (1.0 - dist / pl.radius) * (1.0 - dist / pl.radius);
        var vis = 1.0;
        if (rays) {
            vis = shadowRay(o, d / dist, dist - 1e-3, false, cost);
        } else if (pl.shadowLayer != NO_SHADOW) {
            vis = pointShadowLookup(o, pl.position, pl.shadowLayer);
        }
        e += pl.color * ndl * falloff * vis;
    }
    for (var i = 0u; i < spotLights.count; i++) {
        let light = spotLights.lights[i];
        let s = kansei_spot_sample(light, p);
        if (max(s.illuminance.r, max(s.illuminance.g, s.illuminance.b)) <= 0.0) { continue; }
        let ndl = dot(n, s.toLight);
        if (ndl <= 0.0) { continue; }
        var vis = 1.0;
        if (rays) {
            // toward a point of the lamp's disk (its soft shadow, over frames)
            let xi = vec2f(giHash(px.x, px.y, gp.frame * 31u + dim), giHash(px.y, px.x, gp.frame * 17u + dim + 1u));
            let r = light.sourceRadius * sqrt(xi.x);
            let w = s.toLight;
            let sgn = select(-1.0, 1.0, w.z >= 0.0);
            let a = -1.0 / (sgn + w.z);
            let b = w.x * w.y * a;
            let t = vec3f(1.0 + sgn * w.x * w.x * a, sgn * b, -sgn * w.x);
            let bt = vec3f(b, sgn + w.y * w.y * a, -w.y);
            let onDisk = light.position + (t * cos(6.2831853 * xi.y) + bt * sin(6.2831853 * xi.y)) * r;
            let to = onDisk - o;
            let dist = length(to);
            vis = shadowRay(o, to / dist, dist - 1e-3, false, cost);
        } else if (light.shadowLayer >= 0) {
            let coord = kansei_spot_shadow_coord(light, o + s.toLight * kansei_rt_grid.cell);
            if (coord.w > 0.0) {
                vis = textureSampleCompareLevel(spotShadowAtlas, spotShadowSampler, coord.xy, light.shadowLayer, coord.z);
            }
        }
        e += s.illuminance * ndl * vis;
    }
    return e;
}

// The light leaving a hit (a Lambertian surface of the triangle's albedo) toward the ray: the
// voxels' radiance there, or its exact direct light plus the indirect light round it from one
// voxel cone in a cosine-distributed direction (a sample of the irradiance over pi, which the
// denoiser averages; one wide cone along the normal loses a third of it through thin walls).
// Shaded with the triangle's interpolated vertex normals where it carries them
// (RtSurface::with_smooth_normals), offset along its geometric normal ng.
fn shadeHit(p: vec3f, ng: vec3f, triangle: u32, bary: vec2f, px: vec2u, cost: ptr<function, u32>) -> vec3f {
    if ((gp.flags & RT_GI_HIT_DIRECT) == 0u) {
        return srcHitRadiance(p, ng);
    }
    let n = kansei_rt_shading_normal(triangle, bary, ng);
    let albedo = kansei_rt_albedo(triangle);
    let e = directLight(p, ng, n, px, 2u, cost);
    var indirect = vec3f(0.0);
    if (gp.hitConeSteps > 0u) {
        let size = srcVoxelSize(p);
        let xi = vec2f(giHash(px.x, px.y, gp.frame * 64u + 40u), giHash(px.y, px.x, gp.frame * 64u + 41u));
        let d = above(giCosineDir(n, xi), ng);
        let c = srcSurfaceCone(p, d, n, gp.hitConeTan, 1.5 * size, gp.maxDistance, gp.hitConeSteps);
        indirect = c.rgb + c.a * skyLight(d);
    }
    return albedo / RT_GI_PI * e + albedo * indirect;
}

// Past the grid's box (or with the grid off): a voxel cone from where the ray left it, the sky
// past the voxels.
fn farField(o: vec3f, d: vec3f, n: vec3f, exit: f32) -> vec3f {
    let size = srcVoxelSize(o);
    let c = srcCone(o, d, n, gp.coneTan, max(exit, size), gp.maxDistance, gp.coneSteps);
    return c.rgb + c.a * skyLight(d);
}

// A direction above the surface (an interpolated normal can tilt a sample under it).
fn above(d: vec3f, n: vec3f) -> vec3f {
    let rise = dot(d, n);
    return select(d, normalize(d + n * (0.02 - rise)), rise < 0.02);
}

fn tracePath(origin: vec3f, n0: vec3f, px: vec2u, cost: ptr<function, u32>, found: ptr<function, bool>) -> vec3f {
    var o = origin;
    var n = n0;
    var d = above(giCosineDir(n, giSample2(px, gp.frame, 0u)), n);
    var throughput = vec3f(1.0);
    var radiance = vec3f(0.0);
    for (var b = 0u; b < max(gp.bounces, 1u); b++) {
        var h : KanseiRtHit;
        h.found = false;
        var exit = 0.0;
        if (kansei_rt_contains(o)) {
            h = kansei_rt_trace(o, d, 0.0, gp.maxDistance, traceFlags());
            exit = kansei_rt_exit(o, d);
            *cost += h.cells + h.tests;
        }
        if (!h.found) {
            radiance += throughput * farField(o, d, n, exit);
            break;
        }
        if (b == 0u) {
            *found = true;
        }
        let p = o + d * h.t;
        let ng = h.normal;
        let nh = kansei_rt_shading_normal(h.triangle, h.bary, ng);
        let albedo = kansei_rt_albedo(h.triangle);
        radiance += throughput * albedo / RT_GI_PI * directLight(p, ng, nh, px, 8u + 4u * b, cost);
        // cosine sampling: the BRDF times the cosine over the pdf is the albedo
        throughput *= albedo;
        if (b >= 1u) {
            let q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 0.95);
            if (giHash(px.x, px.y, gp.frame * 64u + b) > q) { break; }
            throughput /= q;
        }
        o = p + ng * surfaceBias(p);
        n = ng;
        let xi = vec2f(giHash(px.x, px.y, gp.frame * 64u + 2u * b + 20u), giHash(px.y, px.x, gp.frame * 64u + 2u * b + 21u));
        d = above(giCosineDir(nh, xi), ng);
    }
    return radiance;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= gp.traceSize)) { return; }
    let px = giPixel(gid.xy);
    let depth = textureLoad(depthTex, px, 0);
    let raw = textureLoad(normalTex, px, 0);
    if (depth >= 1.0 || dot(raw.xyz, raw.xyz) < 1e-4) {
        textureStore(outTex, gid.xy, vec4f(0.0));
        return;
    }
    let n = normalize(raw.xyz * 2.0 - 1.0);
    let world = giWorldPos(giUv(px), depth);
    let origin = world + n * surfaceBias(world);
    var cost = 0u;
    var found = false;
    var radiance = vec3f(0.0);
    if ((gp.flags & RT_GI_REFERENCE) != 0u) {
        radiance = tracePath(origin, n, px, &cost, &found);
    } else {
        let d = above(giCosineDir(n, giSample2(px, gp.frame, 0u)), n);
        var hit : KanseiRtHit;
        hit.found = false;
        var exit = 0.0;
        if ((gp.flags & RT_GI_GRID) != 0u && kansei_rt_contains(origin)) {
            // the near field: the grid up to nearDistance, the voxels past it
            let near = select(gp.maxDistance, gp.nearDistance, gp.nearDistance > 0.0);
            hit = kansei_rt_trace(origin, d, 0.0, near, traceFlags());
            exit = min(kansei_rt_exit(origin, d), near);
            cost += hit.cells + hit.tests;
        }
        found = hit.found;
        if (hit.found) {
            radiance = shadeHit(origin + d * hit.t, hit.normal, hit.triangle, hit.bary, px, &cost);
        } else {
            radiance = farField(origin, d, n, exit);
        }
    }
    // (a NaN would spread through every filter)
    radiance = select(min(radiance, vec3f(60000.0)), vec3f(0.0), radiance != radiance);
    if ((gp.flags & RT_GI_STATS) != 0u) {
        atomicAdd(&giStats[0], 1u);
        atomicAdd(&giStats[1], select(0u, 1u, found));
        atomicAdd(&giStats[3], cost);
        atomicMax(&giStats[4], cost);
    }
    if ((gp.flags & RT_GI_ACCUMULATE) != 0u) {
        let i = gid.y * u32(gp.traceSize.x) + gid.x;
        let before = select(accum[i], vec4f(0.0), gp.accumCount == 0u);
        accum[i] = before + vec4f(radiance, 1.0);
    }
    textureStore(outTex, gid.xy, vec4f(radiance, select(1.0, 1.0 + f32(cost), gp.view == 5u)));
}
