// A smoke volume's fill (SmokeFill): density and temperature per cell, written into its current
// scalar grid, and the velocity grid cleared (Fluid; a 1-texel stand-in otherwise).

struct SmokeFillParams {
    dims           : vec3u,
    kind           : u32,     // 0 none, 1 box, 2 sphere, 3 layers
    size           : vec3f,   // the box's full extents, metres
    density        : f32,
    sphereCenter   : vec3f,   // along the box's axes from its centre, metres
    radius         : f32,
    temperature    : f32,
    layerCount     : u32,
    layerThickness : f32,
    _pad           : f32,
}

@group(0) @binding(0) var<uniform> fp : SmokeFillParams;
@group(0) @binding(1) var scalarsOut  : texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var velocityOut : texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn fill(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid >= fp.dims)) { return; }
    let local = (vec3f(gid) + 0.5) / vec3f(fp.dims);
    let q = (local - 0.5) * fp.size;
    let cell = max(fp.size.x, max(fp.size.y, fp.size.z)) / f32(max(fp.dims.x, max(fp.dims.y, fp.dims.z)));
    var w = 0.0;
    if (fp.kind == 1u) {
        w = 1.0;
    } else if (fp.kind == 2u) {
        // a ball, its density falling smoothly to 0 over the outer quarter of its radius (and at
        // least two cells), so the advection's limiter has no step to clip
        let band = max(0.25 * fp.radius, 2.0 * cell);
        let t = saturate((fp.radius - length(q - fp.sphereCenter)) / band);
        w = t * t * (3.0 - 2.0 * t);
    } else if (fp.kind == 3u) {
        w = smokeLayers(local, fp.layerCount, fp.layerThickness, 0.0);
    }
    textureStore(scalarsOut, gid, vec4f(fp.density * w, fp.temperature * w, 0.0, 0.0));
    textureStore(velocityOut, gid, vec4f(0.0));
}
