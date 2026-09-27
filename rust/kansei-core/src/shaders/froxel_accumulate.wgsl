// Front-to-back integration of (in-scatter, extinction) along each froxel column.
// Output: rgb = light scattered toward the camera up to the slice, a = transmittance.

struct GridParams {
    near  : f32,
    far   : f32,
    gridW : u32,
    gridH : u32,
    gridD : u32,
    _pad0 : f32,
    _pad1 : f32,
    _pad2 : f32,
}

@group(0) @binding(0) var scatterExtTex : texture_3d<f32>;
@group(0) @binding(1) var accumOut      : texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> gp   : GridParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let x = gid.x;
    let y = gid.y;
    if (x >= gp.gridW || y >= gp.gridH) { return; }

    var transmittance = 1.0;
    var accLight = vec3f(0.0);

    for (var z = 0u; z < gp.gridD; z++) {
        let data = textureLoad(scatterExtTex, vec3u(x, y, z), 0);
        let scatter    = data.rgb;
        let extinction = data.a;

        let d0 = sliceDepth(f32(z), gp.near, gp.far, f32(gp.gridD));
        let d1 = sliceDepth(f32(z + 1u), gp.near, gp.far, f32(gp.gridD));
        let thickness = d1 - d0;

        let sliceT = exp(-extinction * thickness);

        // energy-conserving integration of in-scattered light over the slice
        accLight += transmittance * scatter * (1.0 - sliceT) / max(extinction, 0.0001);
        transmittance *= sliceT;

        textureStore(accumOut, vec3u(x, y, z), vec4f(accLight, transmittance));
    }
}
