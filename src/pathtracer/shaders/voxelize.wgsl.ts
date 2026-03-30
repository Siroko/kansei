export const voxelClearShader = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> voxelGrid : array<u32>;
@group(0) @binding(1) var<storage, read_write> voxelColors : array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let idx = gid.x;
    if (idx >= arrayLength(&voxelGrid)) { return; }
    voxelGrid[idx] = 0u;
    voxelColors[idx] = 0u;
}
`;

export const voxelizeShader = /* wgsl */`
struct VoxelizeParams {
    gridMin      : vec3f,
    voxelSize    : f32,
    gridRes      : u32,
    instTriCount : u32,   // total (instance, triangle) pairs
    instCount    : u32,
    _pad         : u32,
}

struct Instance {
    transform0     : vec4f,
    transform1     : vec4f,
    transform2     : vec4f,
    invTransform0  : vec4f,
    invTransform1  : vec4f,
    invTransform2  : vec4f,
    blasNodeOffset : u32,
    blasTriOffset  : u32,
    blasTriCount   : u32,
    materialIndex  : u32,
}

struct MaterialData {
    albedo       : vec3f,
    roughness    : f32,
    metallic     : f32,
    ior          : f32,
    maxBounces   : f32,
    transmission : f32,
    absorptionColor : vec3f,
    absorptionDensity : f32,
    emissive     : vec3f,
    emissiveIntensity : f32,
}

@group(0) @binding(0) var<uniform> params : VoxelizeParams;
@group(0) @binding(1) var<storage, read> triangles : array<f32>;
@group(0) @binding(2) var<storage, read> instances : array<Instance>;
@group(0) @binding(3) var<storage, read_write> voxelGrid : array<u32>;
@group(0) @binding(4) var<storage, read> materials : array<MaterialData>;
@group(0) @binding(5) var<storage, read_write> voxelColors : array<u32>;

fn packRGBA8(c: vec4f) -> u32 {
    return u32(clamp(c.r, 0.0, 1.0) * 255.0)
         | (u32(clamp(c.g, 0.0, 1.0) * 255.0) << 8u)
         | (u32(clamp(c.b, 0.0, 1.0) * 255.0) << 16u)
         | (u32(clamp(c.a, 0.0, 1.0) * 255.0) << 24u);
}

const TRI_STRIDE = 24u;

fn transformPoint(inst: Instance, p: vec3f) -> vec3f {
    return vec3f(
        dot(inst.transform0.xyz, p) + inst.transform0.w,
        dot(inst.transform1.xyz, p) + inst.transform1.w,
        dot(inst.transform2.xyz, p) + inst.transform2.w,
    );
}

// Morton / Z-order curve: interleave bits of x,y,z for spatial locality
fn inflate(n: u32) -> u32 {
    var v = n;
    v = (v | (v << 16u)) & 0xff0000ffu;
    v = (v | (v <<  8u)) & 0x0f00f00fu;
    v = (v | (v <<  4u)) & 0xc30c30c3u;
    v = (v | (v <<  2u)) & 0x49249249u;
    return v;
}

fn voxelIndex(coord: vec3u, res: u32) -> u32 {
    return inflate(coord.x) | (inflate(coord.y) << 1u) | (inflate(coord.z) << 2u);
}

// Per-(instance, triangle) voxelization: handles shared BLAS correctly
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let globalIdx = gid.x;
    if (globalIdx >= params.instTriCount) { return; }

    // Find instance via cumulative blasTriCount scan
    var instIdx = 0u;
    var cumCount = 0u;
    for (var i = 0u; i < params.instCount; i++) {
        let nextCum = cumCount + instances[i].blasTriCount;
        if (globalIdx < nextCum) {
            instIdx = i;
            break;
        }
        cumCount = nextCum;
    }

    let inst = instances[instIdx];
    let matVal = inst.materialIndex + 1u; // 0 = empty, >0 = material+1
    let localTriIdx = globalIdx - cumCount;
    let base = (inst.blasTriOffset + localTriIdx) * TRI_STRIDE;

    // Read local-space vertices
    let v0 = vec3f(triangles[base], triangles[base + 1u], triangles[base + 2u]);
    let v1 = vec3f(triangles[base + 3u], triangles[base + 4u], triangles[base + 5u]);
    let v2 = vec3f(triangles[base + 6u], triangles[base + 7u], triangles[base + 8u]);

    // Transform to world space
    let wv0 = transformPoint(inst, v0);
    let wv1 = transformPoint(inst, v1);
    let wv2 = transformPoint(inst, v2);

    // Compute AABB in grid coordinates
    let triMin = min(min(wv0, wv1), wv2);
    let triMax = max(max(wv0, wv1), wv2);
    let invVoxel = 1.0 / params.voxelSize;
    let gMin = vec3i(floor((triMin - params.gridMin) * invVoxel));
    let gMax = vec3i(floor((triMax - params.gridMin) * invVoxel));
    let gridRes = i32(params.gridRes);
    let lo = clamp(gMin, vec3i(0), vec3i(gridRes - 1));
    let hi = clamp(gMax, vec3i(0), vec3i(gridRes - 1));

    // Triangle edges and normal for intersection test
    let e0 = wv1 - wv0;
    let e1 = wv2 - wv0;
    let triNorm = cross(e0, e1);
    let triNormLen = length(triNorm);
    if (triNormLen < 1e-10) { return; } // degenerate triangle
    let N = triNorm / triNormLen;
    let d = dot(N, wv0);

    // Precompute barycentric dot products for projection test
    let d00 = dot(e0, e0);
    let d01 = dot(e0, e1);
    let d11 = dot(e1, e1);
    let invDenom = 1.0 / (d00 * d11 - d01 * d01);
    // Margin: expand triangle by ~half voxel diagonal relative to edge length
    let halfDiag = params.voxelSize * 0.866; // sqrt(3)/2
    let maxEdgeLen = max(sqrt(d00), sqrt(d11));
    let margin = halfDiag / max(maxEdgeLen, 0.001);

    // Rasterize: for each voxel in AABB, check plane distance + triangle overlap
    for (var z = lo.z; z <= hi.z; z++) {
        for (var y = lo.y; y <= hi.y; y++) {
            for (var x = lo.x; x <= hi.x; x++) {
                let center = params.gridMin + (vec3f(f32(x), f32(y), f32(z)) + 0.5) * params.voxelSize;

                // Check distance from voxel center to triangle plane
                let planeDist = abs(dot(N, center) - d);
                if (planeDist > halfDiag) { continue; }

                // Project voxel center onto triangle plane, check barycentric coords
                let projPoint = center - N * (dot(N, center) - d);
                let v0p = projPoint - wv0;
                let d20 = dot(v0p, e0);
                let d21 = dot(v0p, e1);
                let bv = (d11 * d20 - d01 * d21) * invDenom;
                let bw = (d00 * d21 - d01 * d20) * invDenom;
                let bu = 1.0 - bv - bw;
                // Reject voxels whose projection falls outside expanded triangle
                if (bu < -margin || bv < -margin || bw < -margin) { continue; }

                let coord = vec3u(u32(x), u32(y), u32(z));
                let idx = voxelIndex(coord, params.gridRes);
                voxelGrid[idx] = matVal;
                voxelColors[idx] = packRGBA8(vec4f(materials[matVal - 1u].albedo, 1.0));
            }
        }
    }
}
`;

export const voxelMipBuildShader = /* wgsl */`
struct MipBuildParams {
    fineOffset   : u32,
    coarseOffset : u32,
    fineRes      : u32,
    coarseRes    : u32,
}

@group(0) @binding(0) var<storage, read_write> voxelGrid : array<u32>;
@group(0) @binding(1) var<uniform> params : MipBuildParams;
@group(0) @binding(2) var<storage, read_write> voxelColors : array<u32>;

fn mipUnpackRGBA8(v: u32) -> vec4f {
    return vec4f(
        f32(v & 0xffu) / 255.0,
        f32((v >> 8u) & 0xffu) / 255.0,
        f32((v >> 16u) & 0xffu) / 255.0,
        f32((v >> 24u) & 0xffu) / 255.0,
    );
}

fn mipPackRGBA8(c: vec4f) -> u32 {
    return u32(clamp(c.r, 0.0, 1.0) * 255.0)
         | (u32(clamp(c.g, 0.0, 1.0) * 255.0) << 8u)
         | (u32(clamp(c.b, 0.0, 1.0) * 255.0) << 16u)
         | (u32(clamp(c.a, 0.0, 1.0) * 255.0) << 24u);
}

// Morton / Z-order curve helpers (must match voxelizeShader + DDA shader)
fn mipInflate(n: u32) -> u32 {
    var v = n;
    v = (v | (v << 16u)) & 0xff0000ffu;
    v = (v | (v <<  8u)) & 0x0f00f00fu;
    v = (v | (v <<  4u)) & 0xc30c30c3u;
    v = (v | (v <<  2u)) & 0x49249249u;
    return v;
}

fn mipMorton(x: u32, y: u32, z: u32) -> u32 {
    return mipInflate(x) | (mipInflate(y) << 1u) | (mipInflate(z) << 2u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let idx = gid.x;
    let cRes = params.coarseRes;
    if (idx >= cRes * cRes * cRes) { return; }

    // Derive 3D coord from linear dispatch index
    let cz = idx / (cRes * cRes);
    let cy = (idx / cRes) % cRes;
    let cx = idx % cRes;

    let fx = cx * 2u;
    let fy = cy * 2u;
    let fz = cz * 2u;
    let fOff = params.fineOffset;

    // Read 2x2x2 block from finer level (Morton-indexed)
    // Propagate first non-zero material for SVO occupancy; average colors for cone tracing
    var matVal = 0u;
    var totalColor = vec3f(0.0);
    var totalAlpha = 0.0;
    for (var dz = 0u; dz < 2u; dz++) {
        for (var dy = 0u; dy < 2u; dy++) {
            for (var dx = 0u; dx < 2u; dx++) {
                let fi = fOff + mipMorton(fx + dx, fy + dy, fz + dz);
                let v = voxelGrid[fi];
                if (v > 0u && matVal == 0u) { matVal = v; }

                let colorPacked = voxelColors[fi];
                if (colorPacked != 0u) {
                    let c = mipUnpackRGBA8(colorPacked);
                    totalColor += c.rgb * c.a;
                    totalAlpha += c.a;
                }
            }
        }
    }

    let coarseIdx = params.coarseOffset + mipMorton(cx, cy, cz);
    voxelGrid[coarseIdx] = matVal;
    if (totalAlpha > 0.0) {
        voxelColors[coarseIdx] = mipPackRGBA8(vec4f(totalColor / totalAlpha, totalAlpha / 8.0));
    } else {
        voxelColors[coarseIdx] = 0u;
    }
}
`;
