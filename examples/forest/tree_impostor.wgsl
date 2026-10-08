// A tree layer's far LOD (after IMPOSTOR_WGSL and forest.wgsl): the Raggare intro's
// tree_impostor.wgsl, Kansei's octahedral impostor baked from LOD0 at the layer's median height,
// drawn as a billboard per tree facing each view that draws it (the camera, a cascade). Per
// instance as tree.wgsl: location 3 X, Y, Z, height (m); location 4 bearing (rad), kind, size, _;
// location 5 the LOD crossfade's fade (it fades in from the cards or LOD2). The atlases hold the
// albedo with the ambient occlusion (tree.wgsl writes it so), so the far trees are shaded as the
// meshes are, without per-vertex AO.
struct TreeVertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) inst_pos_height: vec4<f32>,
    @location(4) inst_bearing_kind: vec4<f32>,
    @location(5) lod_fade: f32,
};

struct TreeImpostor {
    // x: the height it was baked at (m); y: the width factor and z: the tint of the instance it
    // was baked from (tree.wgsl's, at its hash)
    baked: vec4<f32>,
    impostor: KanseiImpostor,
};
@group(0) @binding(0) var<uniform> params: TreeImpostor;
@group(0) @binding(8) var albedo_atlas: texture_2d<f32>;
@group(0) @binding(9) var normal_depth_atlas: texture_2d<f32>;
@group(0) @binding(10) var atlas_sampler: sampler;

struct VertexOutput {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) local: vec3<f32>,
    @location(1) eye: vec3<f32>,
    @location(2) inst_pos_height: vec4<f32>,
    @location(3) bearing: f32,
    @location(4) @interpolate(flat) lod_fade: f32,
};

// The baked tree to this one: tree.wgsl's height and hashed width, relative to the baked ones.
fn tree_scale(inst: vec4<f32>) -> vec3<f32> {
    let h = inst.w / params.baked.x;
    let w = h * (0.9 + 0.2 * hash12(inst.xz)) / params.baked.y;
    return vec3<f32>(w, h, w);
}

// tree.wgsl's rotation: a bearing b turns a mesh authored facing -Z by -b about +Y
fn tree_rotation(bearing: f32) -> mat3x3<f32> {
    let c = cos(-bearing);
    let s = sin(-bearing);
    return mat3x3<f32>(vec3<f32>(c, 0.0, -s), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(s, 0.0, c));
}

@vertex
fn vertex_main(in: TreeVertexInput) -> VertexOutput {
    let origin = in.inst_pos_height.xyz;
    let scale = tree_scale(in.inst_pos_height);
    let rot = tree_rotation(in.inst_bearing_kind.x);
    let eye = (transpose(rot) * (view_eye() - origin)) / scale;
    let local = kansei_impostor_corner(params.impostor, in.position.xy, eye);
    var out: VertexOutput;
    out.clip = projection_matrix * view_matrix * vec4<f32>(origin + rot * (local * scale), 1.0);
    out.local = local;
    out.eye = eye;
    out.inst_pos_height = in.inst_pos_height;
    out.bearing = in.inst_bearing_kind.x;
    out.lod_fade = in.lod_fade;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> KanseiGBufferOut {
    let s = kansei_impostor_sample(params.impostor, albedo_atlas, normal_depth_atlas, atlas_sampler, in.local, in.eye);
    if (s.alpha < 0.5 || lod_fade_out(in.lod_fade, in.clip.xy)) {
        discard;
    }
    let origin = in.inst_pos_height.xyz;
    let scale = tree_scale(in.inst_pos_height);
    let rot = tree_rotation(in.bearing);
    let world = origin + rot * (s.position * scale);
    let n = normalize(rot * (s.normal / scale));
    // baked with that instance's tint: this tree's own
    let albedo = s.albedo * (0.85 + 0.3 * hash12(origin.xz)) / params.baked.z;
    // shaded as tree.wgsl's foliage: the sun from either side, the sky and the skylight through
    // the needles and leaves
    var radiance = albedo / PI * (sun_light_foliage(world, n, in.clip.xy) + sky_light(world, n));
    radiance += albedo / PI * sky_light(world, vec3<f32>(0.0, 1.0, 0.0)) * 0.06;
    return kansei_gbuffer_out(radiance, vec3<f32>(0.0), n, albedo);
}

// The shadow maps: the billboard facing the light, cut out by the coverage and dissolved across
// the LOD crossfade.
@fragment
fn shadow_fragment(in: VertexOutput) {
    let s = kansei_impostor_sample(params.impostor, albedo_atlas, normal_depth_atlas, atlas_sampler, in.local, in.eye);
    if (s.alpha < 0.5 || lod_fade_out(in.lod_fade, in.clip.xy)) {
        discard;
    }
}
