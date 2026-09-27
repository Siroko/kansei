// Spot-light data as the renderer uploads it (lights::spot_lights_gpu), and the light's own
// falloff, shared by the material chunk (spot_lights.wgsl) and the volumetric fog.

struct KanseiSpotLight {
    position        : vec3f,
    range           : f32,      // metres; the light reaches zero here
    direction       : vec3f,    // unit vector the light points along
    cosOuter        : f32,
    color           : vec3f,    // linear RGB times candela
    cosInner        : f32,
    shadowLayer     : i32,      // layer in the spot shadow atlas, -1 = unshadowed
    volumetricScale : f32,      // fog scattering relative to surface lighting
    sourceRadius    : f32,      // metres, sets the PCSS penumbra
    normalBias      : f32,      // shadow receiver offset, in shadow-map texels
    shadowNear      : f32,
    tanHalfFov      : f32,      // of the shadow projection
    texelSize       : f32,      // 1 / atlas resolution
    _pad            : f32,
    viewProj        : mat4x4f,  // world -> shadow clip space
}

struct KanseiSpotLights {
    count  : u32,
    _pad0  : u32,
    _pad1  : u32,
    _pad2  : u32,
    lights : array<KanseiSpotLight>,
}

// Clustered culling (light_clusters.wgsl): per cluster, a count then up to 31 light indices.
const KANSEI_CLUSTER_SLOTS : u32 = 32u;

struct KanseiClusterParams {
    view    : mat4x4f,   // world -> view space of the camera the clusters were built for
    invProj : mat4x4f,   // its (jittered) projection, inverted
    screen  : vec2f,     // pixels
    near    : f32,       // depth range the slices span, exponentially
    far     : f32,
    grid    : vec3u,     // tiles x, tiles y, slices
    enabled : u32,       // 0: shade with every light (views such as planar reflections)
}

struct KanseiLightSample {
    toLight     : vec3f,   // unit vector from the point to the light
    illuminance : vec3f,   // lux times colour on a surface facing the light
}

// Inverse-square falloff with a smooth window to zero at `range` (Karis 2013), and a squared
// smoothstep between the outer and inner cones.
fn kansei_spot_sample(light: KanseiSpotLight, worldPos: vec3f) -> KanseiLightSample {
    let d = light.position - worldPos;
    let dist2 = max(dot(d, d), 1e-4);
    let l = d * inverseSqrt(dist2);
    let r = dist2 / (light.range * light.range);
    let window = saturate(1.0 - r * r);
    let cone = saturate((dot(-l, light.direction) - light.cosOuter) / max(light.cosInner - light.cosOuter, 1e-4));
    var s: KanseiLightSample;
    s.toLight = l;
    s.illuminance = light.color * (window * window * cone * cone / dist2);
    return s;
}

// Shadow-map depth ([0, 1], perspective_rh) back to distance along the light's axis.
fn kansei_spot_linear_depth(light: KanseiSpotLight, ndcDepth: f32) -> f32 {
    let n = light.shadowNear;
    let f = light.range;
    return n * f / (f - ndcDepth * (f - n));
}

// Shadow-map uv and depth of a world point; w = 0 when it falls outside the map.
fn kansei_spot_shadow_coord(light: KanseiSpotLight, worldPos: vec3f) -> vec4f {
    let clip = light.viewProj * vec4f(worldPos, 1.0);
    if (clip.w <= 1e-4) { return vec4f(0.0); }
    let ndc = clip.xyz / clip.w;
    let uv = vec2f(ndc.x, -ndc.y) * 0.5 + 0.5;
    let inside = all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0)) && ndc.z < 1.0;
    return vec4f(uv, ndc.z, select(0.0, 1.0, inside));
}
