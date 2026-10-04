// Kansei's MSDF text material (Material::msdf_text): one instanced quad per glyph, its corners
// from the glyph's plane rect around the instance's position, sampled from a multi-channel
// signed distance field atlas (sdf::FontAtlas), anti-aliased by the distance's screen-space
// derivative. Per instance: location 3 the position (xyz; w the glyph's turn about x when
// KANSEI_ROTATE_X), 4 the atlas rect (left, top, right, bottom in texture uv, v down), 5 the plane
// rect (left, top, right, bottom in world units, y up), 6 the colour. Forward, alpha-blended or
// cut out below an alpha of 0.15.

@group(0) @binding(0) var kansei_atlas : texture_2d<f32>;
@group(0) @binding(1) var kansei_atlas_sampler : sampler;
@group(1) @binding(0) var<uniform> view_matrix : mat4x4f;
@group(1) @binding(1) var<uniform> projection_matrix : mat4x4f;
@group(2) @binding(0) var<uniform> normal_matrix : mat4x4f;
@group(2) @binding(1) var<uniform> world_matrix : mat4x4f;

struct KanseiGlyphVaryings {
    @builtin(position) clip : vec4f,
    @location(0) uv : vec2f,
    @location(1) color : vec4f,
}

@vertex
fn vertex_main(
    @location(0) position : vec4f,   // a PlaneGeometry quad, corners at ±0.5
    @location(1) normal : vec3f,
    @location(2) uv : vec2f,
    @location(3) glyph_position : vec4f,
    @location(4) atlas_rect : vec4f,
    @location(5) plane_rect : vec4f,
    @location(6) color : vec4f,
) -> KanseiGlyphVaryings {
    let right = step(0.0, position.x);
    let up = step(0.0, position.y);
    let x = mix(plane_rect.x, plane_rect.z, right);
    var offset = vec3f(x, mix(plane_rect.w, plane_rect.y, up), 0.0);
    if (KANSEI_ROTATE_X) {
        let angle = glyph_position.w;
        offset = vec3f(x, offset.y * cos(angle), offset.y * sin(angle));
    }
    var out: KanseiGlyphVaryings;
    out.clip = projection_matrix * view_matrix * world_matrix * vec4f(glyph_position.xyz + offset, 1.0);
    out.uv = vec2f(mix(atlas_rect.x, atlas_rect.z, right), mix(atlas_rect.w, atlas_rect.y, up));
    out.color = color;
    return out;
}

@fragment
fn fragment_main(in: KanseiGlyphVaryings) -> @location(0) vec4f {
    let s = textureSample(kansei_atlas, kansei_atlas_sampler, in.uv);
    let d = max(min(s.r, s.g), min(max(s.r, s.g), s.b));
    let fw = fwidth(d);
    let alpha = smoothstep(0.5 - fw, 0.5 + fw, d);
    // the atlas's filtering reaches the black around a glyph: cut the faint edge
    if (alpha < 0.15) {
        discard;
    }
    return vec4f(in.color.rgb, in.color.a * alpha);
}
