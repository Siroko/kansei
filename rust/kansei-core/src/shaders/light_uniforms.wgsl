// Kansei scene lights for materials: the renderer packs the scene's directional and point lights
// (lights/light_uniforms.rs) into the camera group's binding 2. Prepend `lights::LIGHTS_WGSL` and
// loop over `kansei_lights.directional[0..num_directional]` and `.point[0..num_point]`. Colours
// are colour times intensity (lux for directional lights). Spot lights come through
// `lights::SPOT_LIGHTS_WGSL` instead.

struct KanseiDirectionalLight {
    direction : vec3f,   // the direction the light travels
    _pad0     : f32,
    color     : vec3f,   // colour times illuminance
    intensity : f32,
}

struct KanseiPointLight {
    position  : vec3f,
    radius    : f32,     // the light reaches no further
    color     : vec3f,   // colour times intensity
    intensity : f32,
}

struct KanseiLights {
    num_directional : u32,
    num_point       : u32,
    _pad0           : u32,
    _pad1           : u32,
    directional     : array<KanseiDirectionalLight, 4>,
    point           : array<KanseiPointLight, 8>,
}

@group(1) @binding(2) var<uniform> kansei_lights : KanseiLights;

// A point light's falloff to zero at its radius (the curve basic_lit and the voxel GI use).
fn kansei_point_falloff(light: KanseiPointLight, worldPos: vec3f) -> f32 {
    let f = max(1.0 - distance(light.position, worldPos) / light.radius, 0.0);
    return f * f;
}
