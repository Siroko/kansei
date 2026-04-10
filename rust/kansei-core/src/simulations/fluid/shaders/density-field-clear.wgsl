@group(0) @binding(0) var densityTex: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(densityTex);
    if (any(gid >= dims)) { return; }
    textureStore(densityTex, gid, vec4<f32>(0.0));
}
