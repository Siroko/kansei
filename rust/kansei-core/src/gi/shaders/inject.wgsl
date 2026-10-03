// Light injection (gi::SceneVoxelGi): one thread per voxel turns the voxelized surface (the mesh
// voxelizer's average albedo and normal, and its emission) into the radiance leaving it, the
// volume's mip 0:
// - direct light: the renderer's directional, point and spot lights, each shadowed by its map
//   (compute_shadows.wgsl), looked up a little out along the normal so the voxel does not
//   shadow itself;
// - one more bounce: the irradiance the hemisphere's cones gather from mips 1.. of the volume
//   itself, which still hold last frame's light (the chain is rebuilt after this pass, and they
//   are other subresources than the mip 0 written here), so the bounces add up over frames;
// - the surface's emission.
// A voxel is a Lambertian surface: albedo / pi times its irradiance, plus its emission. It is
// opaque. Needs voxel_volume.wgsl, voxel_cones.wgsl, voxel_irradiance.wgsl, particle_emission.wgsl
// (its hash), SKY_LIGHTING_WGSL, spot_light_types.wgsl and compute_shadows.wgsl.

struct InjectParams {
    numDirLights    : u32,
    numPointLights  : u32,
    hasShadowMap    : u32,
    hasPointShadows : u32,
    bounce          : f32,   // share of last frame's indirect light fed back (0: direct light only)
    skyScale        : f32,   // how much of the sky's light the bounce cones bring in
    emissionScale   : f32,
    shadowOffset    : f32,   // voxels the shadow lookups move out along the normal
    maxSteps        : u32,   // per bounce cone
    hasDynamic      : u32,   // 1: the dynamic surfaces are bound and preferred where they exist
    _pad0           : u32,
    _pad1           : u32,
}

@group(0) @binding(0) var<uniform> vol : VoxelVolume;
@group(0) @binding(2) var<uniform> ip : InjectParams;
// three u32 per voxel (voxel_write.wgsl): albedo rgb8 + count, normal xyz8 + count, RGB9E5
@group(0) @binding(10) var<storage, read> staticSurfaces : array<u32>;
@group(0) @binding(11) var<storage, read> dynamicSurfaces : array<u32>;
// mips 1.. of the radiance texture being written
@group(0) @binding(12) var previous : texture_3d<f32>;
@group(0) @binding(13) var linearClamp : sampler;
@group(0) @binding(14) var radianceOut : texture_storage_3d<rgba16float, write>;
@group(0) @binding(15) var<uniform> sky : SkyLighting;

fn unpack8(v: u32) -> vec4f {
    return vec4f(f32(v & 255u), f32((v >> 8u) & 255u), f32((v >> 16u) & 255u), f32(v >> 24u));
}

fn unpackRgb9e5(v: u32) -> vec3f {
    let scale = exp2(f32(v >> 27u) - 24.0);
    return vec3f(f32(v & 511u), f32((v >> 9u) & 511u), f32((v >> 18u) & 511u)) * scale;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid >= vol.dims)) { return; }
    let idx = voxelLinearIndex(vol, gid);
    var surface = vec3u(staticSurfaces[3u * idx], staticSurfaces[3u * idx + 1u], staticSurfaces[3u * idx + 2u]);
    if (ip.hasDynamic != 0u) {
        let d = vec3u(dynamicSurfaces[3u * idx], dynamicSurfaces[3u * idx + 1u], dynamicSurfaces[3u * idx + 2u]);
        if ((d.x >> 24u) != 0u || d.z != 0u) { surface = d; }
    }
    let albedoRaw = unpack8(surface.x);
    let emission = unpackRgb9e5(surface.z) * ip.emissionScale;
    if (albedoRaw.w == 0.0 && all(emission <= vec3f(0.0))) {
        textureStore(radianceOut, gid, vec4f(0.0));
        return;
    }
    let albedo = albedoRaw.rgb / 255.0;
    // the average normal; faces that point both ways in one voxel (a sheet thinner than a voxel)
    // average out, and are lit from both sides
    let nRaw = unpack8(surface.y).xyz / 255.0 * 2.0 - 1.0;
    let nLen = length(nRaw);
    let twoSided = nLen < 0.35;
    let n = select(nRaw / max(nLen, 1e-4), vec3f(0.0, 1.0, 0.0), nLen < 1e-3);
    let p = vol.origin + (vec3f(gid) + 0.5) * vol.voxelSize;
    let ps = p + select(n, vec3f(0.0), twoSided) * (ip.shadowOffset * vol.voxelSize);

    var e = vec3f(0.0);
    for (var i = 0u; i < ip.numDirLights; i++) {
        let dl = dirLights[i];
        let l = -normalize(dl.direction);
        var ndl = dot(n, l);
        ndl = select(max(ndl, 0.0), abs(ndl), twoSided);
        if (ndl <= 0.0) { continue; }
        var visibility = 1.0;
        if (dl.shadowed != 0u && ip.hasShadowMap != 0u) {
            visibility = dirShadowLookup(ps);
        }
        e += dl.color * ndl * visibility;
    }
    for (var i = 0u; i < ip.numPointLights; i++) {
        let pl = ptLights[i];
        let d = pl.position - p;
        let dist = length(d);
        if (dist > pl.radius || dist < 1e-4) { continue; }
        var ndl = dot(n, d / dist);
        ndl = select(max(ndl, 0.0), abs(ndl), twoSided);
        if (ndl <= 0.0) { continue; }
        // basic_lit.wgsl's falloff
        let falloff = (1.0 - dist / pl.radius) * (1.0 - dist / pl.radius);
        var visibility = 1.0;
        if (pl.shadowLayer != NO_SHADOW && ip.hasPointShadows != 0u) {
            visibility = pointShadowLookup(ps, pl.position, pl.shadowLayer);
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
        if (light.shadowLayer >= 0) {
            let coord = kansei_spot_shadow_coord(light, ps);
            if (coord.w > 0.0) {
                visibility = textureSampleCompareLevel(spotShadowAtlas, spotShadowSampler, coord.xy, light.shadowLayer, coord.z);
            }
        }
        e += s.illuminance * ndl * visibility;
    }

    // the bounce, from last frame's mips 1..: a volume of half the resolution over the same box
    if (ip.bounce > 0.0 && vol.mipCount > 1u) {
        var half = vol;
        half.voxelSize = vol.voxelSize * 2.0;
        half.dims = max(vol.dims / 2u, vec3u(1u));
        half.mipCount = vol.mipCount - 1u;
        // the tilted cones turned per voxel, the same every frame (noise here would flicker)
        let angle = giHash01(idx * 7919u) * 6.2831853;
        let origin = p + n * vol.voxelSize;
        let gathered = voxelIrradiance(half, previous, linearClamp, sky, ip.skyScale, origin, n, angle, half.voxelSize, 1e4, ip.maxSteps);
        e += gathered.rgb * ip.bounce;
    }

    let out = albedo / VOXEL_GI_PI * e + emission;
    textureStore(radianceOut, gid, vec4f(out / vol.radianceScale, 1.0));
}
