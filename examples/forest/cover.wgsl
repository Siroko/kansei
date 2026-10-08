// Instanced ground cover (after forest.wgsl): the Raggare intro's undergrowth.wgsl, grass tufts,
// flower clumps and shrubs as alpha-tested cards, lit by the sun and the sky (the film's breeze is
// left out). Per instance (the intro's exported instances/*.f32, 32 B, and the fade Kansei's
// culling appends):
//   location 3: X, Y, Z, scale        location 4: bearing (rad), variant, extra0, extra1
//   location 5: the LOD crossfade's fade (1 = whole)
// Card UVs are region-local (0..1, v = 1 at the ground); even variants use region A, odd ones
// region B of the ground-cover atlas.
struct CoverVertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) inst_pos_scale: vec4<f32>,
    @location(4) inst_bearing_variant: vec4<f32>,
    @location(5) lod_fade: f32,
};

struct Regions {
    a: vec4<f32>, // u0, v0, u1, v1
    b: vec4<f32>,
};
@group(0) @binding(0) var<uniform> regions: Regions;
@group(0) @binding(8) var albedo_tex: texture_2d<f32>;
@group(0) @binding(9) var normal_tex: texture_2d<f32>;
@group(0) @binding(10) var tex_sampler: sampler;

struct VertexOutput {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) ao: f32,
    @location(4) tint: f32,
    @location(5) @interpolate(flat) lod_fade: f32,
};

@vertex
fn vertex_main(in: CoverVertexInput) -> VertexOutput {
    let origin = in.inst_pos_scale.xyz;
    let rnd = hash12(origin.xz * 1.37);
    // the Unreal verge's: uniform scale with a 0.85-1.2 stretch in height, sunk 3 cm into the ground
    let scale = in.inst_pos_scale.w * vec3<f32>(1.0, 0.85 + 0.35 * rnd, 1.0);
    let a = -in.inst_bearing_variant.x;
    let c = cos(a);
    let s = sin(a);
    let rot = mat3x3<f32>(vec3<f32>(c, 0.0, -s), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(s, 0.0, c));

    var out: VertexOutput;
    out.world = origin + rot * (in.position.xyz * scale) - vec3<f32>(0.0, 0.03, 0.0);
    out.clip = projection_matrix * view_matrix * vec4<f32>(out.world, 1.0);
    out.normal = rot * in.normal;
    let variant = u32(abs(in.inst_bearing_variant.y) + 0.5);
    let r = select(regions.a, regions.b, (variant & 1u) == 1u);
    // half a texel inside the region, so mips don't bleed across
    let inset = 2.0 / 1024.0;
    out.uv = mix(r.xy + inset, r.zw - inset, in.uv);
    out.ao = in.position.w;
    out.tint = rnd;
    out.lod_fade = in.lod_fade;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput, @builtin(front_facing) front: bool) -> KanseiGBufferOut {
    let texel = textureSample(albedo_tex, tex_sampler, in.uv);
    let nm = textureSample(normal_tex, tex_sampler, in.uv).xyz * 2.0 - 1.0;
    let dp1 = dpdx(in.world);
    let dp2 = dpdy(in.world);
    let duv1 = dpdx(in.uv);
    let duv2 = dpdy(in.uv);
    if (texel.a < 0.5 || lod_fade_out(in.lod_fade, in.clip.xy)) {
        discard;
    }
    // the tangent frame from derivatives (as tree.wgsl), over the up-leaning card normal
    let n0 = normalize(in.normal);
    let n = normalize(mix(n0, cotangent_frame(n0, dp1, dp2, duv1, duv2) * nm, 0.5));

    let albedo = texel.rgb * (0.8 + 0.4 * in.tint);
    var radiance = albedo / PI * (sun_light_foliage(in.world, n, in.clip.xy) + sky_light(in.world, n) * in.ao);
    // light through the blades and leaves from the sky behind them
    radiance += albedo / PI * sky_light(in.world, vec3<f32>(0.0, 1.0, 0.0)) * 0.05 * in.ao;
    return kansei_gbuffer_out(radiance, vec3<f32>(0.0), n, albedo * in.ao);
}

// The shadow maps: the cards cut out by their alpha.
@fragment
fn shadow_fragment(in: VertexOutput) {
    let a = textureSample(albedo_tex, tex_sampler, in.uv).a;
    if (a < 0.5 || lod_fade_out(in.lod_fade, in.clip.xy)) {
        discard;
    }
}
