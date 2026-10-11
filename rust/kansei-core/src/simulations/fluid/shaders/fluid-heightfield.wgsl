// A fluid's surface as a height field (simulations::fluid::FluidHeightfield), from its density
// field (FluidDensityField, iso level `iso`): `heights` finds, in each column of voxels, the top
// of the body of fluid from below (the first crossing out of it, so the spray above it is left
// out), as a world height (DRY, far below, where the column holds none); `rt_mesh` writes a grid
// of vertices over it into an RtMesh's words for the ray tracing grid, and its quads' indices: a
// quad with a dry corner as one point (no area: the grid leaves it out). The field's voxel i sits at boundsMin + i / dims * extent (the marching
// cubes' placing); world = simulation units * scale.

struct HeightfieldParams {
    boundsMin : vec3f,
    iso       : f32,
    boundsMax : vec3f,
    scale     : f32,
    dims      : vec3u,
    _pad0     : u32,
    meshDims  : vec2u,
    spacing   : f32,
    _pad1     : u32,
    meshMin   : vec2f,
    _pad2     : vec2f,
}

@group(0) @binding(0) var<uniform> hp : HeightfieldParams;

const DRY : f32 = -1e9;

fn isDry(h: f32) -> bool {
    return h < -1e8;
}

// ---- heights

@group(0) @binding(1) var density : texture_3d<f32>;
@group(0) @binding(2) var heightsOut : texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn heights(@builtin(global_invocation_id) gid : vec3u) {
    if (gid.x >= hp.dims.x || gid.y >= hp.dims.z) { return; }
    var inside = false;
    var below = 0.0;
    var h = DRY;
    for (var j = 0u; j < hp.dims.y; j++) {
        let v = textureLoad(density, vec3u(gid.x, j, gid.y), 0).r;
        if (v >= hp.iso) {
            inside = true;
        } else if (inside) {
            // between voxel j - 1 (inside) and j
            let f = clamp((below - hp.iso) / max(below - v, 1e-6), 0.0, 1.0);
            let y = (f32(j) - 1.0 + f) / f32(hp.dims.y);
            h = mix(hp.boundsMin.y, hp.boundsMax.y, y) * hp.scale;
            break;
        }
        below = v;
    }
    textureStore(heightsOut, vec2u(gid.x, gid.y), vec4f(h, 0.0, 0.0, 0.0));
}

// ---- the mesh for the grid

@group(0) @binding(3) var heightsIn : texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> mesh : array<u32>;

// The height at world xz, bilinear between the columns (DRY if any of the four is dry).
fn heightAt(xz: vec2f) -> f32 {
    let sim = xz / hp.scale;
    let c = (sim - hp.boundsMin.xz) / (hp.boundsMax.xz - hp.boundsMin.xz) * vec2f(hp.dims.xz);
    let last = vec2i(hp.dims.xz) - 1;
    let i = clamp(vec2i(floor(c)), vec2i(0), last);
    let f = clamp(c - vec2f(i), vec2f(0.0), vec2f(1.0));
    let j = min(i + 1, last);
    let a = textureLoad(heightsIn, vec2u(i), 0).r;
    let b = textureLoad(heightsIn, vec2u(vec2i(j.x, i.y)), 0).r;
    let d = textureLoad(heightsIn, vec2u(vec2i(i.x, j.y)), 0).r;
    let e = textureLoad(heightsIn, vec2u(j), 0).r;
    if (isDry(a) || isDry(b) || isDry(d) || isDry(e)) {
        return DRY;
    }
    return mix(mix(a, b, f.x), mix(d, e, f.x), f.y);
}

@compute @workgroup_size(8, 8)
fn rt_mesh(@builtin(global_invocation_id) gid : vec3u) {
    if (gid.x >= hp.meshDims.x || gid.y >= hp.meshDims.y) { return; }
    let xz = hp.meshMin + vec2f(gid.xy) * hp.spacing;
    let h = heightAt(xz);
    // the normal from the heights a vertex either side (its own height where one is dry)
    let s = hp.spacing;
    var hx0 = heightAt(xz - vec2f(s, 0.0));
    var hx1 = heightAt(xz + vec2f(s, 0.0));
    var hz0 = heightAt(xz - vec2f(0.0, s));
    var hz1 = heightAt(xz + vec2f(0.0, s));
    hx0 = select(hx0, h, isDry(hx0));
    hx1 = select(hx1, h, isDry(hx1));
    hz0 = select(hz0, h, isDry(hz0));
    hz1 = select(hz1, h, isDry(hz1));
    var n = normalize(vec3f(hx0 - hx1, 2.0 * s, hz0 - hz1));
    if (isDry(h)) {
        n = vec3f(0.0, 1.0, 0.0);
    }
    let v = gid.y * hp.meshDims.x + gid.x;
    let at = mesh[0] + v * 5u;
    mesh[at] = bitcast<u32>(xz.x);
    mesh[at + 1u] = bitcast<u32>(select(h, hp.boundsMin.y * hp.scale, isDry(h)));
    mesh[at + 2u] = bitcast<u32>(xz.y);
    mesh[at + 3u] = 0u;
    mesh[at + 4u] = kansei_rt_pack_normal(n);
}

// The quad from vertex (x, z) to (x + 1, z + 1): two triangles counter-clockwise seen from above,
// or a point where a corner is dry.
@compute @workgroup_size(8, 8)
fn rt_quads(@builtin(global_invocation_id) gid : vec3u) {
    let quads = hp.meshDims - 1u;
    if (gid.x >= quads.x || gid.y >= quads.y) { return; }
    let nx = hp.meshDims.x;
    let a = gid.y * nx + gid.x;
    let b = a + 1u;
    let c = a + nx + 1u;
    let d = a + nx;
    let p = hp.meshMin + vec2f(gid.xy) * hp.spacing;
    let s = hp.spacing;
    let dry = isDry(heightAt(p)) || isDry(heightAt(p + vec2f(s, 0.0))) || isDry(heightAt(p + vec2f(0.0, s))) || isDry(heightAt(p + vec2f(s, s)));
    let at = mesh[1] + (gid.y * quads.x + gid.x) * 6u;
    if (dry) {
        for (var k = 0u; k < 6u; k++) {
            mesh[at + k] = a;
        }
        return;
    }
    mesh[at] = a;
    mesh[at + 1u] = d;
    mesh[at + 2u] = c;
    mesh[at + 3u] = a;
    mesh[at + 4u] = c;
    mesh[at + 5u] = b;
}
