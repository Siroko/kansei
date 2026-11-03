// SmaaEffect's three passes: edges, blending weights and neighbourhood blending, each its own
// compute pass over the frame, reading the input, the edges and the weights from textures.
// Appended to smaa_common.wgsl, whose accessors it defines.

@group(0) @binding(4) var smaaOutput     : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var smaaEdgesIn    : texture_2d<f32>;
@group(0) @binding(6) var smaaEdgesOut   : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(7) var smaaWeightsIn  : texture_2d<f32>;
@group(0) @binding(8) var smaaWeightsOut : texture_storage_2d<rgba8unorm, write>;

fn smaaColorAt(p: vec2i) -> vec3f {
    return textureLoad(smaaInput, smaaClampToFrame(p), 0).rgb;
}

fn smaaLumaAt(p: vec2i) -> f32 {
    return smaaLumaOf(smaaColorAt(p));
}

fn smaaEdgesAt(p: vec2i) -> vec2f {
    if (!smaaInFrame(p)) {
        return vec2f(0.0);
    }
    return textureLoad(smaaEdgesIn, p, 0).rg;
}

fn smaaWeightsAt(p: vec2i) -> vec4f {
    if (!smaaInFrame(p)) {
        return vec4f(0.0);
    }
    return textureLoad(smaaWeightsIn, p, 0);
}

@compute @workgroup_size(8, 8)
fn edges(@builtin(global_invocation_id) gid: vec3u) {
    if (any(gid.xy >= smaa.size)) {
        return;
    }
    textureStore(smaaEdgesOut, gid.xy, vec4f(smaaDetectEdges(vec2i(gid.xy)), 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn weights(@builtin(global_invocation_id) gid: vec3u) {
    if (any(gid.xy >= smaa.size)) {
        return;
    }
    let p = vec2i(gid.xy);
    // a pixel with no edge has no weights (iryoku's stencil)
    if (dot(smaaEdgesAt(p), vec2f(1.0)) == 0.0) {
        textureStore(smaaWeightsOut, gid.xy, vec4f(0.0));
        return;
    }
    textureStore(smaaWeightsOut, gid.xy, smaaWeights(p, i32(smaa.reach)));
}

@compute @workgroup_size(8, 8)
fn blend(@builtin(global_invocation_id) gid: vec3u) {
    if (any(gid.xy >= smaa.size)) {
        return;
    }
    let p = vec2i(gid.xy);
    let color = smaaBlend(p, smaaWeightsAt(p), smaaWeightsAt(p + vec2i(1, 0)), smaaWeightsAt(p + vec2i(0, 1)));
    textureStore(smaaOutput, gid.xy, vec4f(color, 1.0));
}
