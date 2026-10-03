// The fragment side of mesh voxelization (gi::MeshVoxelizer): its group 3, and
// `kansei_voxel_write`, which puts a fragment's surface in its voxel. The engine's own
// `voxel_fragment` (voxel_fragment.wgsl) passes the renderable's constant `GiSurface`; a
// material's `voxel_fragment_entry` passes what its surface reflects and emits there (a texture
// lookup, say), and must call it in uniform control flow (it takes derivatives).
//
// The voxelizer draws each renderable three times, through its material's own vertex_main, with
// an orthographic camera along x, y and z over the volume and a pixel per voxel. The target is
// multisampled, so a fragment runs wherever any sample is covered: close to conservative
// rasterization, which WebGPU lacks. A fragment's voxel is its position and depth; its normal
// comes from how its voxel position moves across the screen (exact for a flat triangle under an
// orthographic camera), so a material need not output one.
//
// Per voxel, three u32 (`kansei_voxel_surfaces`): the running averages of the albedo (rgb8 and
// a count) and of the normal (xyz8 and a count), after Crassin and Green (OpenGL Insights, ch.
// 22) on atomicCompareExchangeWeak, since there are no float atomics; and the brightest emission
// (RGB9E5 under atomicMax, whose shared exponent sits in the high bits).

struct KanseiVoxelizeParams {
    clipToVoxel : mat4x4f,   // this axis' clip space to voxel coordinates (voxel c spans [c, c + 1))
    viewDir     : vec3f,     // the direction this axis' camera looks along
    _pad0       : f32,
    viewport    : vec2f,     // pixels of this axis' viewport
    _pad1       : vec2f,
    dims        : vec3u,
    _pad2       : u32,
}

// the renderable's constant surface (gi::GiSurface), at a dynamic offset per draw
struct KanseiVoxelDraw {
    albedo   : vec3f,
    _pad0    : f32,
    emission : vec3f,   // scene radiance
    _pad1    : f32,
}

@group(3) @binding(100) var<uniform> kansei_voxelize : KanseiVoxelizeParams;
@group(3) @binding(101) var<uniform> kansei_voxel_draw : KanseiVoxelDraw;
@group(3) @binding(102) var<storage, read_write> kansei_voxel_surfaces : array<atomic<u32>>;

fn kansei_voxel_unpack(v: u32) -> vec4f {
    return vec4f(f32(v & 255u), f32((v >> 8u) & 255u), f32((v >> 16u) & 255u), f32(v >> 24u));
}

fn kansei_voxel_pack(c: vec4f) -> u32 {
    let q = vec4u(clamp(round(c), vec4f(0.0), vec4f(255.0)));
    return q.x | (q.y << 8u) | (q.z << 16u) | (q.w << 24u);
}

// Fold `value` (0..1 per channel) into the running average in `slot`. The count stops at 255;
// past it the average keeps moving by 1/256 per sample.
fn kansei_voxel_average(slot: u32, value: vec3f) {
    let add = clamp(value, vec3f(0.0), vec3f(1.0)) * 255.0;
    var expected = atomicLoad(&kansei_voxel_surfaces[slot]);
    for (var tries = 0u; tries < 32u; tries++) {
        let cur = kansei_voxel_unpack(expected);
        let n = cur.w;
        let next = kansei_voxel_pack(vec4f((cur.xyz * n + add) / (n + 1.0), min(n + 1.0, 255.0)));
        let r = atomicCompareExchangeWeak(&kansei_voxel_surfaces[slot], expected, next);
        if (r.exchanged) { return; }
        expected = r.old_value;
    }
}

// RGB9E5 (EXT_texture_shared_exponent): nine bits of mantissa per channel, a shared exponent.
fn kansei_pack_rgb9e5(c: vec3f) -> u32 {
    let rgb = clamp(c, vec3f(0.0), vec3f(65408.0));
    let m = max(rgb.r, max(rgb.g, rgb.b));
    var e = max(-16, i32(floor(log2(max(m, 1e-30))))) + 16;
    var denom = exp2(f32(e - 24));
    if (u32(floor(m / denom + 0.5)) == 512u) {
        denom *= 2.0;
        e += 1;
    }
    let q = vec3u(floor(rgb / denom + 0.5));
    return min(q.r, 511u) | (min(q.g, 511u) << 9u) | (min(q.b, 511u) << 18u) | (u32(e) << 27u);
}

// Put a surface into the fragment's voxel: what it reflects (`albedo`) and emits (`emission`,
// scene radiance). Call it in uniform control flow.
fn kansei_voxel_write(fragPos: vec4f, front: bool, albedo: vec3f, emission: vec3f) {
    let ndc = vec2f(fragPos.x / kansei_voxelize.viewport.x * 2.0 - 1.0, 1.0 - fragPos.y / kansei_voxelize.viewport.y * 2.0);
    let h = kansei_voxelize.clipToVoxel * vec4f(ndc, fragPos.z, 1.0);
    let v = h.xyz / h.w;
    // the plane's normal toward this axis' camera, then toward the outside of the surface (a
    // front face is seen from outside)
    var n = cross(dpdx(v), dpdy(v));
    n = n * inverseSqrt(max(dot(n, n), 1e-20));
    n = select(n, -n, dot(n, kansei_voxelize.viewDir) > 0.0);
    n = select(-n, n, front);
    let dims = vec3f(kansei_voxelize.dims);
    if (any(v < vec3f(0.0)) || any(v >= dims)) { return; }
    let c = vec3u(min(floor(v), dims - 1.0));
    let idx = (c.z * kansei_voxelize.dims.y + c.y) * kansei_voxelize.dims.x + c.x;
    kansei_voxel_average(3u * idx, albedo);
    kansei_voxel_average(3u * idx + 1u, n * 0.5 + 0.5);
    if (any(emission > vec3f(0.0))) {
        atomicMax(&kansei_voxel_surfaces[3u * idx + 2u], kansei_pack_rgb9e5(emission));
    }
}
