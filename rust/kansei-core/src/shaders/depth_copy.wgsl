@group(0) @binding(0) var depth_msaa: texture_depth_multisampled_2d;

@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(vec2<f32>(-1.0,-1.0), vec2<f32>(3.0,-1.0), vec2<f32>(-1.0,3.0));
    return vec4<f32>(pos[vi], 0.0, 1.0);
}

@fragment fn fs(@builtin(position) frag: vec4<f32>) -> @builtin(frag_depth) f32 {
    return textureLoad(depth_msaa, vec2<i32>(frag.xy), 0);
}
