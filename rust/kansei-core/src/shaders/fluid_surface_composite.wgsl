// `FluidSurfaceEffect`'s composite: refraction (with chromatic aberration), Fresnel, screen-space
// and sky reflection, key-light specular and rim, over the fluid's pixels. Shared with the TS
// engine (`src/postprocessing/effects/FluidSurfaceEffect.ts`).
struct Params {
    view_matrix: mat4x4<f32>,
    color: vec4<f32>,
    ior: f32,
    chromatic_aberration: f32,
    tint_strength: f32,
    fresnel_power: f32,
    roughness: f32,
    thickness: f32,
    screen_width: f32,
    screen_height: f32,
    light_dir: vec4<f32>,   // xyz = direction light travels (world), w = intensity
    light_color: vec4<f32>, // rgb, w = rim strength
    sky: vec4<f32>,         // rgb, w = sky reflection
    mask: u32,              // 0: any GBuffer normal is fluid; 1: only where emissive alpha >= 0.5
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var scene_color: texture_2d<f32>;    // input (MC surface already rendered)
@group(0) @binding(2) var background_tex: texture_2d<f32>; // opaque scene before MC
@group(0) @binding(3) var normal_tex: texture_2d<f32>;     // GBuffer normals
@group(0) @binding(4) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var emissive_tex: texture_2d<f32>;   // GBuffer emissive (alpha: the fluid's mask)

// Whether the fluid's surface covers this texel: a GBuffer normal, and with the emissive mask the
// fluid material's mark (other materials writing normals, for GI, are not the fluid).
fn is_fluid(texel: vec2u) -> bool {
    if (length(textureLoad(normal_tex, texel, 0).rgb) < 0.01) { return false; }
    return params.mask == 0u || textureLoad(emissive_tex, texel, 0).a >= 0.5;
}

// A refracted sample, where the fluid covers it: elsewhere the texel shows something in front of
// the surface (a body standing out of the water), which must not appear through it, and the
// pixel's own texel stands in.
fn refracted_texel(uv: vec2<f32>, own: vec2u, dims: vec2<f32>) -> vec3<f32> {
    let texel = min(vec2u(dims * uv), vec2u(dims) - vec2u(1u));
    let covered = is_fluid(texel);
    return textureLoad(background_tex, select(own, texel, covered), 0).rgb;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let coord = gid.xy;
    let w = u32(params.screen_width);
    let h = u32(params.screen_height);
    if (coord.x >= w || coord.y >= h) { return; }

    let scene = textureLoad(scene_color, coord, 0);
    let normal_data = textureLoad(normal_tex, coord, 0).rgb;

    // No MC surface here → pass through scene color
    if (!is_fluid(coord)) {
        textureStore(output_tex, coord, scene);
        return;
    }

    let N_world = normalize(normal_data);
    // Transform world normal to view space (use upper 3x3 of view matrix)
    let view3 = mat3x3<f32>(
        params.view_matrix[0].xyz,
        params.view_matrix[1].xyz,
        params.view_matrix[2].xyz,
    );
    // Flip back-facing normals so they always face the camera (fluid is double-sided)
    var N_view = normalize(view3 * N_world);
    if (N_view.z < 0.0) { N_view = -N_view; }
    let screen_uv = (vec2<f32>(f32(coord.x), f32(coord.y)) + 0.5) / vec2<f32>(f32(w), f32(h));

    // Refraction offset — use view-space normal's xy (screen-aligned)
    let refract_strength = params.thickness * (1.0 - 1.0 / params.ior);
    let offset = N_view.xy * refract_strength * 0.05;

    // Chromatic aberration
    let ca = params.chromatic_aberration;
    let dims_f = vec2<f32>(f32(w), f32(h));
    let uv_r = clamp(screen_uv + offset * (1.0 + ca), vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_g = clamp(screen_uv + offset, vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_b = clamp(screen_uv + offset * (1.0 - ca), vec2<f32>(0.0), vec2<f32>(1.0));

    let bg_r = refracted_texel(uv_r, coord, dims_f).r;
    let bg_g = refracted_texel(uv_g, coord, dims_f).g;
    let bg_b = refracted_texel(uv_b, coord, dims_f).b;
    var refracted = vec3<f32>(bg_r, bg_g, bg_b);

    // Tint refracted light by fluid color (absorption)
    refracted *= mix(vec3<f32>(1.0), params.color.rgb, params.tint_strength);

    // Fresnel (view-space: N_view.z is proper NdotV since view dir is (0,0,1) in view space)
    let f0 = pow((1.0 - params.ior) / (1.0 + params.ior), 2.0);
    let ndotv = N_view.z;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - ndotv, params.fresnel_power);

    // Screen-space reflection: reflect the view ray off the view-space normal.
    // In view space, the view direction is (0,0,1) (looking toward the surface).
    let reflect_dir = reflect(vec3<f32>(0.0, 0.0, -1.0), N_view);
    let reflect_offset = reflect_dir.xy * 0.3;
    let reflect_uv = clamp(screen_uv + reflect_offset, vec2<f32>(0.0), vec2<f32>(1.0));
    var reflected = textureLoad(background_tex, vec2u(dims_f * reflect_uv), 0).rgb;
    // Open sky where the reflected ray points up in the world (view to world: the transpose)
    let up = (transpose(view3) * reflect_dir).y;
    reflected = mix(reflected, params.sky.rgb, params.sky.w * smoothstep(0.0, 0.15, up));

    // Key-light GGX specular (view space: V = (0,0,1)).
    let L = normalize(view3 * normalize(-params.light_dir.xyz));
    let V = vec3<f32>(0.0, 0.0, 1.0);
    let H = normalize(L + V);
    let ndotl = max(dot(N_view, L), 0.0);
    let ndoth = max(dot(N_view, H), 0.0);
    let alpha = max(params.roughness * params.roughness, 1e-3);
    let a2 = alpha * alpha;
    let denom = ndoth * ndoth * (a2 - 1.0) + 1.0;
    let D = a2 / (3.14159 * denom * denom + 1e-4);
    let k = alpha * 0.5;
    let G = (ndotv / (ndotv * (1.0 - k) + k)) * (ndotl / (ndotl * (1.0 - k) + k));
    let light_rgb = params.light_color.rgb * params.light_dir.w;
    let specular = fresnel * D * G * ndotl * light_rgb * 0.5;

    // Rim light for edge glow, tinted by the key light
    let rim = pow(1.0 - ndotv, 3.0) * params.light_color.w * params.light_color.rgb * (0.5 + 0.25 * params.light_dir.w);

    // Final: mix refracted (transmitted) and reflected (environment) via Fresnel, + specular + rim
    let result = mix(refracted, reflected, fresnel) + specular + rim;

    textureStore(output_tex, coord, vec4<f32>(result, 1.0));
}
