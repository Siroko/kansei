struct Params {
    dims_and_max_tris: vec4<u32>,  // xyz = voxel dims, w = max triangles
    bounds_min_and_iso: vec4<f32>, // xyz = bounds min, w = iso level
    bounds_max_pad: vec4<f32>,     // xyz = bounds max
};

struct Counter {
    value: atomic<u32>,
};

struct McVertex {
    position: vec4<f32>,
    normal: vec4<f32>,
};

@group(0) @binding(0) var densityTex: texture_3d<f32>;
@group(0) @binding(1) var densitySampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> vertices: array<McVertex>;
@group(0) @binding(4) var<storage, read_write> indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> counter: Counter;

fn voxel_center_world(coord: vec3<u32>, dims: vec3<u32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec3<f32> {
    let d = max(vec3<f32>(dims), vec3<f32>(1.0, 1.0, 1.0));
    let uvw = (vec3<f32>(coord) + vec3<f32>(0.5, 0.5, 0.5)) / d;
    return mix(bmin, bmax, uvw);
}

fn sample_density(dims: vec3<u32>, coord: vec3<u32>) -> f32 {
    let uvw = (vec3<f32>(coord) + vec3<f32>(0.5, 0.5, 0.5)) / max(vec3<f32>(dims), vec3<f32>(1.0, 1.0, 1.0));
    return textureSampleLevel(densityTex, densitySampler, uvw, 0.0).x;
}

fn emit_triangle(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, n: vec3<f32>, max_tris: u32) {
    let tri_idx = atomicAdd(&counter.value, 1u);
    if (tri_idx >= max_tris) {
        return;
    }
    let vbase = tri_idx * 3u;
    let ibase = tri_idx * 3u;

    vertices[vbase + 0u].position = vec4<f32>(p0, 1.0);
    vertices[vbase + 1u].position = vec4<f32>(p1, 1.0);
    vertices[vbase + 2u].position = vec4<f32>(p2, 1.0);
    vertices[vbase + 0u].normal = vec4<f32>(n, 0.0);
    vertices[vbase + 1u].normal = vec4<f32>(n, 0.0);
    vertices[vbase + 2u].normal = vec4<f32>(n, 0.0);

    indices[ibase + 0u] = vbase + 0u;
    indices[ibase + 1u] = vbase + 1u;
    indices[ibase + 2u] = vbase + 2u;
}

fn emit_quad(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>, n: vec3<f32>, max_tris: u32) {
    emit_triangle(a, b, c, n, max_tris);
    emit_triangle(a, c, d, n, max_tris);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = params.dims_and_max_tris.xyz;
    if (any(gid >= dims)) {
        return;
    }

    let bmin = params.bounds_min_and_iso.xyz;
    let bmax = params.bounds_max_pad.xyz;
    let iso = params.bounds_min_and_iso.w;
    let max_tris = params.dims_and_max_tris.w;

    let density = sample_density(dims, gid);
    if (density < iso) {
        return;
    }

    let dims_f = max(vec3<f32>(dims), vec3<f32>(1.0, 1.0, 1.0));
    let cell = (bmax - bmin) / dims_f;
    let c0 = bmin + vec3<f32>(gid) * cell;
    let c1 = c0 + cell;

    let p000 = vec3<f32>(c0.x, c0.y, c0.z);
    let p001 = vec3<f32>(c0.x, c0.y, c1.z);
    let p010 = vec3<f32>(c0.x, c1.y, c0.z);
    let p011 = vec3<f32>(c0.x, c1.y, c1.z);
    let p100 = vec3<f32>(c1.x, c0.y, c0.z);
    let p101 = vec3<f32>(c1.x, c0.y, c1.z);
    let p110 = vec3<f32>(c1.x, c1.y, c0.z);
    let p111 = vec3<f32>(c1.x, c1.y, c1.z);

    let gid_i = vec3<i32>(gid);
    let dims_i = vec3<i32>(dims);

    // +X
    {
        let ncoord = gid_i + vec3<i32>(1, 0, 0);
        var outside = any(ncoord >= dims_i);
        var nd = 0.0;
        if (!outside) {
            nd = sample_density(dims, vec3<u32>(ncoord));
        }
        if (outside || nd < iso) {
            emit_quad(p100, p101, p111, p110, vec3<f32>(1.0, 0.0, 0.0), max_tris);
        }
    }
    // -X
    {
        let ncoord = gid_i + vec3<i32>(-1, 0, 0);
        var outside = any(ncoord < vec3<i32>(0));
        var nd = 0.0;
        if (!outside) {
            nd = sample_density(dims, vec3<u32>(ncoord));
        }
        if (outside || nd < iso) {
            emit_quad(p001, p000, p010, p011, vec3<f32>(-1.0, 0.0, 0.0), max_tris);
        }
    }
    // +Y
    {
        let ncoord = gid_i + vec3<i32>(0, 1, 0);
        var outside = any(ncoord >= dims_i);
        var nd = 0.0;
        if (!outside) {
            nd = sample_density(dims, vec3<u32>(ncoord));
        }
        if (outside || nd < iso) {
            emit_quad(p010, p110, p111, p011, vec3<f32>(0.0, 1.0, 0.0), max_tris);
        }
    }
    // -Y
    {
        let ncoord = gid_i + vec3<i32>(0, -1, 0);
        var outside = any(ncoord < vec3<i32>(0));
        var nd = 0.0;
        if (!outside) {
            nd = sample_density(dims, vec3<u32>(ncoord));
        }
        if (outside || nd < iso) {
            emit_quad(p000, p001, p101, p100, vec3<f32>(0.0, -1.0, 0.0), max_tris);
        }
    }
    // +Z
    {
        let ncoord = gid_i + vec3<i32>(0, 0, 1);
        var outside = any(ncoord >= dims_i);
        var nd = 0.0;
        if (!outside) {
            nd = sample_density(dims, vec3<u32>(ncoord));
        }
        if (outside || nd < iso) {
            emit_quad(p001, p011, p111, p101, vec3<f32>(0.0, 0.0, 1.0), max_tris);
        }
    }
    // -Z
    {
        let ncoord = gid_i + vec3<i32>(0, 0, -1);
        var outside = any(ncoord < vec3<i32>(0));
        var nd = 0.0;
        if (!outside) {
            nd = sample_density(dims, vec3<u32>(ncoord));
        }
        if (outside || nd < iso) {
            emit_quad(p110, p010, p000, p100, vec3<f32>(0.0, 0.0, -1.0), max_tris);
        }
    }
}
