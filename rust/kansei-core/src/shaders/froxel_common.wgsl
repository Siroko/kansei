// Froxel grid helpers: exponential depth slicing and the camera's [0,1] depth convention
// (glam::Mat4::perspective_rh, as kansei's Camera). Prepended to every froxel/fog shader.

fn sliceDepth(i: f32, near: f32, far: f32, numSlices: f32) -> f32 {
    return near * pow(far / near, i / numSlices);
}

fn depthToSlice(linearDepth: f32, near: f32, far: f32, numSlices: f32) -> f32 {
    return numSlices * log(linearDepth / near) / log(far / near);
}

// View-space distance -> NDC depth for perspective_rh ([0,1] depth).
fn linearToNdcDepth(d: f32, n: f32, f: f32) -> f32 {
    return f * (d - n) / (d * (f - n));
}

// NDC depth ([0,1]) -> view-space distance.
fn ndcToLinearDepth(z: f32, n: f32, f: f32) -> f32 {
    return n * f / (f - z * (f - n));
}

// World position of the centre of froxel `cell` (x, y in grid cells, z = slice + 0.5).
fn froxelToWorld(cell: vec3f, gridSize: vec3f, gridNear: f32, gridFar: f32,
                 camNear: f32, camFar: f32, invViewProj: mat4x4f) -> vec3f {
    let uv = (cell.xy + 0.5) / gridSize.xy;
    let linearD = sliceDepth(cell.z, gridNear, gridFar, gridSize.z);
    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, linearToNdcDepth(linearD, camNear, camFar), 1.0);
    let world = invViewProj * ndc;
    return world.xyz / world.w;
}
