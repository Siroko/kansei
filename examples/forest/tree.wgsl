// Instanced procedural trees (after forest.wgsl): the Raggare intro's tree.wgsl, one material for
// bark (opaque) and one for foliage (alpha-tested cards on the foliage atlas), per species and LOD,
// lit by the sun and the sky, written to the GBuffer and the GI's voxels. The trees stand still
// (the film's sway is left out, so the shadow maps, the voxels and the ray tracing grid hold the
// trees the camera sees).
//
// Per vertex: position.xyz (1 m tall tree), position.w = ambient occlusion, a normal (bent
// toward the crown's outside for foliage), uv (atlas UV for foliage; bark: u around the trunk,
// v = 12 x distance along the stem of the 1 m tree).
// Per instance (the intro's exported trees.f32, 32 B, and the fade Kansei's culling appends):
//   location 3: X, Y, Z, height (m)      location 4: bearing (rad), kind, raw size, _
//   location 5: the LOD crossfade's fade (1 = whole)
struct TreeVertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) inst_pos_height: vec4<f32>,
    @location(4) inst_bearing_kind: vec4<f32>,
    @location(5) lod_fade: f32,
};

struct TreeParams {
    tint: vec4<f32>,
    flags: vec4<f32>, // x: 1 = foliage (alpha test), y: 1 = birch
};
@group(0) @binding(0) var<uniform> params: TreeParams;
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
    @location(5) above_ground: f32,
    @location(6) @interpolate(flat) lod_fade: f32,
};

const BARK_TILE_M: f32 = 1.6; // metres of trunk per bark texture repeat

// An instance's width (m) for its height: a little wider or narrower by a hash of where it stands.
// (TREE_PLACEMENT_WGSL places the ray tracing grid's trees the same way.)
fn tree_width(origin: vec3<f32>, height: f32) -> f32 {
    return height * (0.9 + 0.2 * hash12(origin.xz));
}

// A bearing b turns a mesh authored facing -Z by -b about +Y.
fn tree_rotation(bearing: f32) -> mat3x3<f32> {
    let c = cos(-bearing);
    let s = sin(-bearing);
    return mat3x3<f32>(vec3<f32>(c, 0.0, -s), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(s, 0.0, c));
}

@vertex
fn vertex_main(in: TreeVertexInput) -> VertexOutput {
    let origin = in.inst_pos_height.xyz;
    let height = in.inst_pos_height.w;
    let width = tree_width(origin, height);
    let scale = vec3<f32>(width, height, width);
    let rot = tree_rotation(in.inst_bearing_kind.x);

    var out: VertexOutput;
    out.world = origin + rot * (in.position.xyz * scale);
    out.clip = projection_matrix * view_matrix * vec4<f32>(out.world, 1.0);
    out.normal = rot * normalize(in.normal / scale);
    out.ao = in.position.w;
    out.tint = hash12(origin.xz);
    out.above_ground = in.position.y * height;
    out.lod_fade = in.lod_fade;
    if (params.flags.x > 0.5) {
        out.uv = in.uv;
    } else {
        // bark: metric tiling along the stem, whatever the tree's height
        out.uv = vec2<f32>(in.uv.x, in.uv.y / 12.0 * height / BARK_TILE_M);
    }
    return out;
}

fn tree_albedo(in: VertexOutput, texel: vec3<f32>) -> vec3<f32> {
    let foliage = params.flags.x > 0.5;
    var albedo = texel * params.tint.rgb * (0.85 + 0.3 * in.tint);
    if (!foliage && params.flags.y > 0.5 && in.uv.x > 1.0) {
        // birch limbs: grey-brown twig bark, not the trunk's white
        albedo = vec3<f32>(0.05, 0.045, 0.04) * (0.8 + 0.4 * in.tint);
    } else if (!foliage && params.flags.y > 0.5) {
        // birch: dark, fissured bark at the foot of the trunk
        let base = 1.0 - smoothstep(0.6, 2.2 + 0.8 * in.tint, in.above_ground + 0.5 * hash12(floor(in.uv * vec2<f32>(6.0, 18.0))));
        albedo = mix(albedo, vec3<f32>(0.035, 0.03, 0.028), base);
    }
    return albedo;
}

@fragment
fn fragment_main(in: VertexOutput, @builtin(front_facing) front: bool) -> KanseiGBufferOut {
    let texel = textureSample(albedo_tex, tex_sampler, in.uv);
    let nm = textureSample(normal_tex, tex_sampler, in.uv).xyz * 2.0 - 1.0;
    let dp1 = dpdx(in.world);
    let dp2 = dpdy(in.world);
    let duv1 = dpdx(in.uv);
    let duv2 = dpdy(in.uv);
    let foliage = params.flags.x > 0.5;
    if ((foliage && texel.a < 0.5) || lod_fade_out(in.lod_fade, in.clip.xy)) {
        discard;
    }

    var n = normalize(in.normal);
    if (!foliage && !front) {
        n = -n;
    }
    let tbn = cotangent_frame(n, dp1, dp2, duv1, duv2);
    n = normalize(mix(n, tbn * nm, select(0.8, 0.6, foliage)));

    let albedo = tree_albedo(in, texel.rgb);
    // the needles and leaves take the sun from either side, the bark as a surface; inside the
    // crown (its occlusion below 1) the bark mostly in the needles' shade, finer than the shadow
    // maps resolve
    let sun = select(sun_light(in.world, n, in.clip.xy) * in.ao * in.ao, sun_light_foliage(in.world, n, in.clip.xy), foliage);
    var radiance = albedo / PI * (sun + sky_light(in.world, n) * in.ao);
    if (foliage) {
        // skylight through the needles and leaves
        radiance += albedo / PI * sky_light(in.world, vec3<f32>(0.0, 1.0, 0.0)) * 0.06 * in.ao;
    }
    // the albedo written with the ambient occlusion: the GI's bounces and the impostors take it
    return kansei_gbuffer_out(radiance, vec3<f32>(0.0), n, albedo * in.ao);
}

// The shadow maps: foliage cards cut out by their alpha, and both parts dissolved across the LOD
// crossfades (in the shadow map's own pixels).
@fragment
fn shadow_fragment(in: VertexOutput) {
    let a = textureSample(albedo_tex, tex_sampler, in.uv).a;
    if ((params.flags.x > 0.5 && a < 0.5) || lod_fade_out(in.lod_fade, in.clip.xy)) {
        discard;
    }
}

// The GI's voxels: the sprays' and leaves' area alone (the cut-out texels write nothing), whole
// across the crossfades.
@fragment
fn voxel_main(in: VertexOutput, @builtin(front_facing) front: bool) {
    let texel = textureSample(albedo_tex, tex_sampler, in.uv);
    let covered = params.flags.x < 0.5 || texel.a >= 0.5;
    kansei_voxel_write_coverage(in.clip, front, tree_albedo(in, texel.rgb) * in.ao, vec3<f32>(0.0), select(0.0, 1.0, covered));
}
