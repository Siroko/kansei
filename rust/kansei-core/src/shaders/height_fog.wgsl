// Analytic exponential height fog: the line integral of one or two exponential density layers
// from the camera to each pixel (to `skyDistance` for the sky), coloured by a constant
// inscattering luminance, the sky's distant light (SkyLighting, when bound) and a directional
// lobe toward the sun. Unreal's ExponentialHeightFog model, for the distance beyond the
// volumetric fog's froxels: start it where they end and put it before them in the chain.
// Prefixed with SKY_LIGHTING_WGSL.

struct HeightFogParams {
    invViewProj           : mat4x4f,
    cameraPos             : vec3f,
    startDistance         : f32,
    inscattering          : vec3f,   // fog luminance at full opacity
    cutoffDistance        : f32,     // no fog on what is farther; 0: no cutoff
    directional           : vec3f,   // luminance of the lobe toward the light
    directionalExponent   : f32,
    lightDirection        : vec3f,   // toward the light, when there is no sky lighting
    directionalStart      : f32,
    layer0                : vec4f,   // density (per metre at height), falloff (per metre), height (m)
    layer1                : vec4f,
    maxOpacity            : f32,
    skyAmbientScale       : f32,
    skyDistance           : f32,
    hasSkyLighting        : u32,
    viewForward           : vec3f,   // the camera's forward axis
    volumetricFogDistance : f32,     // the volumetric fog's end as a view depth (0: none)
}

@group(0) @binding(0) var inputTex  : texture_2d<f32>;
@group(0) @binding(1) var depthTex  : texture_depth_2d;
@group(0) @binding(2) var outputTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> fog : HeightFogParams;
@group(0) @binding(4) var<uniform> skyLighting : SkyLighting;

// Optical depth of one layer along c + d t for t in [t0, t1] (d unit), in closed form.
fn layerOpticalDepth(layer: vec4f, c: vec3f, d: vec3f, t0: f32, t1: f32) -> f32 {
    let density = layer.x;
    if (density <= 0.0 || t1 <= t0) { return 0.0; }
    // density where the segment starts, clamped far below the layer so it cannot overflow
    let start = density * exp(min(-layer.y * (c.y + d.y * t0 - layer.z), 80.0));
    let k = layer.y * d.y;
    let len = t1 - t0;
    let x = k * len;
    var shape = len;
    if (abs(x) > 1e-4) {
        shape = (1.0 - exp(min(-x, 80.0))) / k;
    }
    return start * shape;
}

fn opticalDepth(c: vec3f, d: vec3f, t0: f32, t1: f32) -> f32 {
    return layerOpticalDepth(fog.layer0, c, d, t0, t1) + layerOpticalDepth(fog.layer1, c, d, t0, t1);
}

fn unproject(uv: vec2f, depth: f32) -> vec3f {
    let p = fog.invViewProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return p.xyz / p.w;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(outputTex);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let color = textureLoad(inputTex, gid.xy, 0);
    let depth = textureLoad(depthTex, gid.xy, 0);
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(size);

    var dir : vec3f;
    var dist : f32;
    if (depth < 1.0) {
        let offset = unproject(uv, depth) - fog.cameraPos;
        dist = length(offset);
        dir = offset / max(dist, 1e-6);
    } else {
        dir = normalize(unproject(uv, 1.0) - unproject(uv, 0.0));
        dist = fog.skyDistance;
    }
    // the fog starts past startDistance along the ray and past the volumetric fog's end plane
    let start = max(fog.startDistance, fog.volumetricFogDistance / max(dot(dir, fog.viewForward), 1e-4));
    if ((fog.cutoffDistance > 0.0 && dist > fog.cutoffDistance) || dist <= start) {
        textureStore(outputTex, gid.xy, color);
        return;
    }

    let opacity = min(1.0 - exp(-opticalDepth(fog.cameraPos, dir, start, dist)), fog.maxOpacity);
    var fogColor = fog.inscattering;
    var lightDir = fog.lightDirection;
    var lightVisibility = 1.0;
    if (fog.hasSkyLighting != 0u) {
        // the sky's distant light, as Unreal's height fog adds it (the SH would hold the fog
        // itself once the sky lighting captures it)
        fogColor += skyLighting.distantSkyLight.rgb * fog.skyAmbientScale;
        lightDir = skyLighting.sunDirection.xyz;
        lightVisibility = skyLighting.sunIlluminance.w;
    }
    var result = color.rgb * (1.0 - opacity) + fogColor * opacity;

    // the lobe toward the light, only past its own start distance
    if (any(fog.directional > vec3f(0.0))) {
        let t0 = max(start, fog.directionalStart);
        let dirOpacity = min(1.0 - exp(-opticalDepth(fog.cameraPos, dir, t0, dist)), fog.maxOpacity);
        let lobe = pow(saturate(dot(dir, normalize(lightDir))), fog.directionalExponent);
        result += fog.directional * lobe * lightVisibility * dirOpacity;
    }
    textureStore(outputTex, gid.xy, vec4f(result, color.a));
}
