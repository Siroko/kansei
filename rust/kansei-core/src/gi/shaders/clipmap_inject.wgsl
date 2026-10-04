// Light injection into a voxel clipmap level (gi::SceneVoxelClipmap): inject.wgsl's, a level at
// a time. One thread per texel of the level's window turns the voxelized surface it holds (the
// clipmap voxelizer's average albedo and normal, and its emission) into the radiance leaving it,
// written to the scratch level and copied over the level after:
// - direct light: the renderer's directional, point and spot lights, each shadowed by its map
//   (compute_shadows.wgsl) or, where no map covers the voxel (a sun past its cascades) or for
//   every light (`coneShadows`), by a narrow cone toward the light through last frame's clipmap:
//   its transmittance;
// - one more bounce: the irradiance the hemisphere's cones gather from last frame's clipmap, from
//   footprints twice the level's voxels (as inject.wgsl reads a volume's mips 1..), so the bounces
//   add up over frames;
// - the surface's emission.
// A voxel is a Lambertian surface: albedo / pi times its irradiance, plus its emission. Its
// opacity comes from the surface's area in it (in voxel faces): a surface whose normals agree on
// an axis (a wall, the ground, a sheet seen from both sides) covers the voxel as its area says,
// at most fully; scattered ones (needles, leaves, cards turned every way) block light as randomly
// turned leaves do, 1 - exp(-area / 2) (Ross' G = 1/2), so a sparse crown lets light through and
// a dense one does not. Needs clipmap.wgsl,
// particle_emission.wgsl (its hash), SKY_LIGHTING_WGSL, spot_light_types.wgsl and
// compute_shadows.wgsl.

struct ClipInjectParams {
    numDirLights    : u32,
    numPointLights  : u32,
    hasShadowMap    : u32,
    hasPointShadows : u32,
    bounce          : f32,   // share of last frame's indirect light fed back (0: direct light only)
    skyScale        : f32,   // how much of the sky's light the bounce cones bring in
    emissionScale   : f32,
    shadowOffset    : f32,   // voxels the shadow lookups move out along the normal
    maxSteps        : u32,   // per bounce cone
    hasDynamic      : u32,   // 1: the level's dynamic surfaces are bound and preferred where they exist
    level           : u32,   // the level lit
    coneShadows     : u32,   // 0: shadow maps only; 1: cones where no map covers; 2: always cones
    shadowTan       : f32,   // tan of the shadow cones' half angle
    shadowSteps     : u32,   // per shadow cone
    bounceMinAlbedo : f32,   // darker voxels skip the bounce
    _pad2           : u32,
}

@group(0) @binding(0) var<uniform> ip : ClipInjectParams;
// five u32 per voxel by texel (voxel_write.wgsl): albedo rgb8 + count, normal xyz8 + count, the
// folded normal xyz8 + count, emission RGB9E5, the surface's area in voxel faces (1/256)
@group(0) @binding(10) var<storage, read> staticSurfaces : array<u32>;
@group(0) @binding(11) var<storage, read> dynamicSurfaces : array<u32>;
@group(0) @binding(14) var radianceOut : texture_storage_3d<rgba16float, write>;
@group(0) @binding(15) var<uniform> sky : SkyLighting;

fn unpack8(v: u32) -> vec4f {
    return vec4f(f32(v & 255u), f32((v >> 8u) & 255u), f32((v >> 16u) & 255u), f32(v >> 24u));
}

fn unpackRgb9e5(v: u32) -> vec3f {
    let scale = exp2(f32(v >> 27u) - 24.0);
    return vec3f(f32(v & 511u), f32((v >> 9u) & 511u), f32((v >> 18u) & 511u)) * scale;
}

// The share of a light `maxT` metres away along `toLight` that reaches a voxel at `ps` (normal n,
// none for a two-sided sheet) through last frame's clipmap: a narrow cone's transmittance, from a
// voxel out.
fn coneShadow(ps: vec3f, n: vec3f, twoSided: bool, toLight: vec3f, maxT: f32, size: f32) -> f32 {
    let lift = select(n, vec3f(0.0), twoSided);
    return clipConeTrace(ps, toLight, lift, ip.shadowTan, size, size, maxT, ip.shadowSteps).a;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid >= clipmap.dims)) { return; }
    let k = ip.level;
    let size = clipVoxelSize(k);
    // the window's voxel this texel holds: congruent to it, inside the window
    let d = vec3i(clipmap.dims);
    let origin = clipmap.levels[k].origin;
    let c = origin + ((((vec3i(gid) - origin) % d) + d) % d);
    let idx = (gid.z * clipmap.dims.y + gid.y) * clipmap.dims.x + gid.x;
    let base = 5u * idx;
    var surface = vec4u(staticSurfaces[base], staticSurfaces[base + 1u], staticSurfaces[base + 2u], staticSurfaces[base + 3u]);
    var area = f32(staticSurfaces[base + 4u]) / 256.0;
    if (ip.hasDynamic != 0u) {
        let dyn = vec4u(dynamicSurfaces[base], dynamicSurfaces[base + 1u], dynamicSurfaces[base + 2u], dynamicSurfaces[base + 3u]);
        if ((dyn.x >> 24u) != 0u || dyn.w != 0u) {
            surface = dyn;
            area = f32(dynamicSurfaces[base + 4u]) / 256.0;
        }
    }
    let albedoRaw = unpack8(surface.x);
    let emission = unpackRgb9e5(surface.w) * ip.emissionScale;
    if (albedoRaw.w == 0.0 && all(emission <= vec3f(0.0))) {
        textureStore(radianceOut, gid, vec4f(0.0));
        return;
    }
    let albedo = albedoRaw.rgb / 255.0;
    // the average normal, or a sheet's axis where its faces cancel (inject.wgsl)
    let nRaw = unpack8(surface.y).xyz / 255.0 * 2.0 - 1.0;
    let nLen = length(nRaw);
    let twoSided = nLen < 0.35;
    let axis = unpack8(surface.z).xyz / 255.0 * 2.0 - 1.0;
    let n = select(nRaw / max(nLen, 1e-4), axis / max(length(axis), 1e-4), twoSided);
    // how opaque: as a flat surface where the normals agree on an axis (either way: a sheet's two
    // faces), as scattered leaves where they don't
    let flat = smoothstep(0.35, 0.85, length(axis));
    let opacity = clamp(mix(1.0 - exp(-0.5 * area), area, flat), 0.0, 1.0);
    let p = (vec3f(c) + 0.5) * size;
    let ps = p + select(n, vec3f(0.0), twoSided) * (ip.shadowOffset * size);

    var e = vec3f(0.0);
    for (var i = 0u; i < ip.numDirLights; i++) {
        let dl = dirLights[i];
        let l = -normalize(dl.direction);
        var ndl = dot(n, l);
        ndl = select(max(ndl, 0.0), abs(ndl), twoSided);
        if (ndl <= 0.0) { continue; }
        var visibility = 1.0;
        let mapped = dl.shadowed != 0u && ip.hasShadowMap != 0u && dirShadowCovers(ps);
        if (ip.coneShadows == 2u || (ip.coneShadows == 1u && !mapped)) {
            visibility = coneShadow(ps, n, twoSided, l, 1e4, size);
        } else if (mapped) {
            visibility = dirShadowLookup(ps + l * size);
        }
        e += dl.color * ndl * visibility;
    }
    for (var i = 0u; i < ip.numPointLights; i++) {
        let pl = ptLights[i];
        let toLight = pl.position - p;
        let dist = length(toLight);
        if (dist > pl.radius || dist < 1e-4) { continue; }
        var ndl = dot(n, toLight / dist);
        ndl = select(max(ndl, 0.0), abs(ndl), twoSided);
        if (ndl <= 0.0) { continue; }
        let falloff = (1.0 - dist / pl.radius) * (1.0 - dist / pl.radius);
        var visibility = 1.0;
        let mapped = pl.shadowLayer != NO_SHADOW && ip.hasPointShadows != 0u;
        if (ip.coneShadows == 2u || (ip.coneShadows == 1u && !mapped)) {
            visibility = coneShadow(ps, n, twoSided, toLight / dist, dist, size);
        } else if (mapped) {
            visibility = pointShadowLookup(ps + toLight / dist * size, pl.position, pl.shadowLayer);
        }
        e += pl.color * ndl * falloff * visibility;
    }
    for (var i = 0u; i < spotLights.count; i++) {
        let light = spotLights.lights[i];
        let s = kansei_spot_sample(light, p);
        if (max(s.illuminance.r, max(s.illuminance.g, s.illuminance.b)) <= 0.0) { continue; }
        var ndl = dot(n, s.toLight);
        ndl = select(max(ndl, 0.0), abs(ndl), twoSided);
        if (ndl <= 0.0) { continue; }
        var visibility = 1.0;
        var coord = vec4f(0.0);
        if (light.shadowLayer >= 0) {
            coord = kansei_spot_shadow_coord(light, ps + s.toLight * size);
        }
        if (ip.coneShadows == 2u || (ip.coneShadows == 1u && coord.w == 0.0)) {
            visibility = coneShadow(ps, n, twoSided, s.toLight, length(light.position - ps), size);
        } else if (coord.w > 0.0) {
            visibility = textureSampleCompareLevel(spotShadowAtlas, spotShadowSampler, coord.xy, light.shadowLayer, coord.z);
        }
        e += s.illuminance * ndl * visibility;
    }

    // the bounce, from last frame's clipmap, footprints twice this level's voxels (but on dark
    // voxels, which pass on little of it)
    if (ip.bounce > 0.0 && max(albedo.r, max(albedo.g, albedo.b)) >= ip.bounceMinAlbedo) {
        // the tilted cones turned per voxel, the same every frame (noise here would flicker)
        let angle = giHash01(idx * 7919u + k * 104729u) * 6.2831853;
        var gathered = clipIrradiance(sky, ip.skyScale, p + n * size, n, angle, 2.0 * size, 2.0 * size, 1e4, ip.maxSteps).rgb;
        if (twoSided) {
            // a sheet: the brighter of its sides (an isotropic voxel can't keep both)
            let back = clipIrradiance(sky, ip.skyScale, p - n * size, -n, angle, 2.0 * size, 2.0 * size, 1e4, ip.maxSteps).rgb;
            gathered = max(gathered, back);
        }
        e += gathered * ip.bounce;
    }

    // (premultiplied by the opacity, as the volume's voxels are by their coverage)
    let out = (albedo / 3.14159265 * e + emission) * opacity;
    textureStore(radianceOut, gid, vec4f(out / clipmap.radianceScale, opacity));
}
