// Classic Marching Cubes with Paul Bourke lookup tables.
// Produces smooth iso-surface triangles with edge interpolation and gradient normals.

struct Params {
    dims_and_max_tris: vec4<u32>,
    bounds_min_and_iso: vec4<f32>,
    bounds_max_pad: vec4<f32>,
};

struct Counter { value: atomic<u32> };

// Vertex data written as flat f32 array to avoid WGSL struct alignment padding.
// 9 floats per vertex = 36 bytes: position(4) + normal(3) + uv(2), matching engine Vertex layout.
const FLOATS_PER_VERT: u32 = 9u;

@group(0) @binding(0) var densityTex: texture_3d<f32>;
@group(0) @binding(1) var densitySampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> vertex_data: array<f32>;
@group(0) @binding(4) var<storage, read_write> indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> counter: Counter;
// Paul Bourke lookup tables. Originally stored as module-scope const arrays,
// moved to storage buffers because Safari/WebKit's WGSL→Metal translator
// hangs for tens of seconds compiling a shader with a 4096-entry const array.
// Uploaded once at pipeline init from CPU-side constants (see
// marching_cubes_tables.rs / marching-cubes-tables.ts).
@group(0) @binding(6) var<storage, read> edge_table: array<u32>;
@group(0) @binding(7) var<storage, read> tri_table:  array<i32>;

fn write_vertex(idx: u32, pos: vec3<f32>, norm: vec3<f32>) {
    let b = idx * FLOATS_PER_VERT;
    vertex_data[b + 0u] = pos.x;
    vertex_data[b + 1u] = pos.y;
    vertex_data[b + 2u] = pos.z;
    vertex_data[b + 3u] = 1.0;
    vertex_data[b + 4u] = norm.x;
    vertex_data[b + 5u] = norm.y;
    vertex_data[b + 6u] = norm.z;
    vertex_data[b + 7u] = 0.0;
    vertex_data[b + 8u] = 0.0;
}



// ── Cube corner offsets (standard MC numbering) ──────────────────────────
//   0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0)
//   4=(0,0,1) 5=(1,0,1) 6=(1,1,1) 7=(0,1,1)

fn sample(dims: vec3<f32>, coord: vec3<f32>) -> f32 {
    let uvw = (coord + vec3<f32>(0.5)) / dims;
    let d = textureSampleLevel(densityTex, densitySampler, uvw, 0.0).x;
    // Fade density to 0 near domain boundary so iso-surface closes on all sides.
    let lo = smoothstep(vec3<f32>(0.0), vec3<f32>(2.0), coord);
    let hi = smoothstep(vec3<f32>(0.0), vec3<f32>(2.0), dims - vec3<f32>(1.0) - coord);
    let fade = lo.x * lo.y * lo.z * hi.x * hi.y * hi.z;
    return d * fade;
}

fn gradient(dims: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    let e = vec3<f32>(1.0, 0.0, 0.0);
    return vec3<f32>(
        sample(dims, p + e.xyy) - sample(dims, p - e.xyy),
        sample(dims, p + e.yxy) - sample(dims, p - e.yxy),
        sample(dims, p + e.yyx) - sample(dims, p - e.yyx),
    );
}

fn interp_vertex(p1: vec3<f32>, p2: vec3<f32>, v1: f32, v2: f32, iso: f32) -> vec3<f32> {
    let d = v2 - v1;
    if (abs(d) < 0.00001) { return (p1 + p2) * 0.5; }
    let t = clamp((iso - v1) / d, 0.0, 1.0);
    return mix(p1, p2, t);
}

fn emit_mc_triangle(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>,
                    dims: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>, max_tris: u32) {
    let tri_idx = atomicAdd(&counter.value, 1u);
    if (tri_idx >= max_tris) { return; }
    let vbase = tri_idx * 3u;

    let w0 = mix(bmin, bmax, p0 / dims);
    let w1 = mix(bmin, bmax, p1 / dims);
    let w2 = mix(bmin, bmax, p2 / dims);

    let n0 = -normalize(gradient(dims, p0));
    let n1 = -normalize(gradient(dims, p1));
    let n2 = -normalize(gradient(dims, p2));

    write_vertex(vbase + 0u, w0, n0);
    write_vertex(vbase + 1u, w1, n1);
    write_vertex(vbase + 2u, w2, n2);
    indices[vbase + 0u] = vbase + 0u;
    indices[vbase + 1u] = vbase + 1u;
    indices[vbase + 2u] = vbase + 2u;
}

// ── Edge endpoints: edge i connects corner EDGE_A[i] to corner EDGE_B[i] ─
const EDGE_A: array<u32, 12> = array<u32, 12>(0,1,2,3, 4,5,6,7, 0,1,2,3);
const EDGE_B: array<u32, 12> = array<u32, 12>(1,2,3,0, 5,6,7,4, 4,5,6,7);

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = params.dims_and_max_tris.xyz;
    // We need dims-1 cells (each cell spans two voxels along each axis)
    if (any(gid >= dims - vec3<u32>(1u))) { return; }

    let bmin = params.bounds_min_and_iso.xyz;
    let bmax = params.bounds_max_pad.xyz;
    let iso  = params.bounds_min_and_iso.w;
    let max_tris = params.dims_and_max_tris.w;
    let dims_f = vec3<f32>(dims);
    let g = vec3<f32>(gid);

    // Sample 8 corners
    var corner_pos: array<vec3<f32>, 8>;
    corner_pos[0] = g + vec3<f32>(0.0, 0.0, 0.0);
    corner_pos[1] = g + vec3<f32>(1.0, 0.0, 0.0);
    corner_pos[2] = g + vec3<f32>(1.0, 1.0, 0.0);
    corner_pos[3] = g + vec3<f32>(0.0, 1.0, 0.0);
    corner_pos[4] = g + vec3<f32>(0.0, 0.0, 1.0);
    corner_pos[5] = g + vec3<f32>(1.0, 0.0, 1.0);
    corner_pos[6] = g + vec3<f32>(1.0, 1.0, 1.0);
    corner_pos[7] = g + vec3<f32>(0.0, 1.0, 1.0);

    var corner_val: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) {
        corner_val[i] = sample(dims_f, corner_pos[i]);
    }

    // Build case index
    var cube_index = 0u;
    if (corner_val[0] >= iso) { cube_index |= 1u; }
    if (corner_val[1] >= iso) { cube_index |= 2u; }
    if (corner_val[2] >= iso) { cube_index |= 4u; }
    if (corner_val[3] >= iso) { cube_index |= 8u; }
    if (corner_val[4] >= iso) { cube_index |= 16u; }
    if (corner_val[5] >= iso) { cube_index |= 32u; }
    if (corner_val[6] >= iso) { cube_index |= 64u; }
    if (corner_val[7] >= iso) { cube_index |= 128u; }

    let edges = edge_table[cube_index];
    if (edges == 0u) { return; }

    // Interpolate edge vertices
    var edge_verts: array<vec3<f32>, 12>;
    for (var e = 0u; e < 12u; e++) {
        if ((edges & (1u << e)) != 0u) {
            let a = EDGE_A[e];
            let b = EDGE_B[e];
            edge_verts[e] = interp_vertex(
                corner_pos[a], corner_pos[b],
                corner_val[a], corner_val[b], iso,
            );
        }
    }

    // Emit triangles from lookup table
    let base = cube_index * 16u;
    for (var t = 0u; t < 15u; t += 3u) {
        let e0 = tri_table[base + t];
        if (e0 < 0) { break; }
        let e1 = tri_table[base + t + 1u];
        let e2 = tri_table[base + t + 2u];
        emit_mc_triangle(
            edge_verts[e0], edge_verts[e1], edge_verts[e2],
            dims_f, bmin, bmax, max_tris,
        );
    }
}
