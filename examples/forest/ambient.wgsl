// The sky light every surface receives, by the page's GI mode: the sky's SH whole (its own
// ambient: off, and the screen GI modes, which replace it on screen), dimmed by the top-down sky
// occlusion the Raggare film uses, or by the sky visibility the clipmap's probes measure. Group 0
// bindings 1 (the sky) and 2-7. Prefixed with SKY_LIGHTING_WGSL, SKY_OCCLUSION_WGSL and
// CLIPMAP_PROBES_WGSL. Shared by the outdoor-gi example's Rust (rust/kansei-wasm/examples/
// outdoor-gi) and TS (index_outdoor_gi.html) pages.
struct Ambient { mode: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(1) var<uniform> sky: SkyLighting;
@group(0) @binding(2) var<uniform> ambient: Ambient;
@group(0) @binding(3) var sky_volume: texture_3d<f32>;
@group(0) @binding(4) var sky_sampler: sampler;
@group(0) @binding(5) var<uniform> sky_occlusion: SkyOcclusionParams;
@group(0) @binding(6) var<uniform> kansei_clip_probe_grid: ClipProbeGrid;
@group(0) @binding(7) var<storage, read> kansei_clip_probes: array<vec4<f32>>;

// The share of the sky a point sees, as the GI mode measures it in the material (1: all of it).
fn sky_visibility(world: vec3<f32>, n: vec3<f32>) -> f32 {
    switch (ambient.mode) {
        case 1u: { return skyVisibility(sky_volume, sky_sampler, sky_occlusion, world); }
        case 2u: { return kansei_clipmap_sky_visibility(world, n); }
        default: { return 1.0; }
    }
}

// The sky's irradiance on a surface facing n, dimmed by what the mode sees of it.
fn sky_light(world: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return skyIrradiance(sky, n) * sky_visibility(world, n);
}
