/**
 * GPU compute shader for expanding per-instance data into TLAS instance entries.
 *
 * Two entry points:
 *   expandPositions  — reads vec4f positions (xyz offset), combines with parent transform
 *   expandTransforms — reads 3×vec4f row-major affine transforms, composes with parent transform
 *
 * Both produce full Instance records (transform + inverse + BLAS metadata) in the
 * instance storage buffer so the TLAS refit passes can build world-space AABBs.
 */
export const instanceExpandShader = /* wgsl */`

struct Instance {
    transform0    : vec4f,
    transform1    : vec4f,
    transform2    : vec4f,
    invTransform0 : vec4f,
    invTransform1 : vec4f,
    invTransform2 : vec4f,
    metadata      : vec4u,
}

struct ExpandParams {
    // Parent rotation/scale rows + translation packed as w
    r0 : vec4f,   // r00, r01, r02, ptx
    r1 : vec4f,   // r10, r11, r12, pty
    r2 : vec4f,   // r20, r21, r22, ptz
    // Inverse of parent rotation/scale (shared across all instances in position mode)
    ir0 : vec4f,  // inv00, inv01, inv02, 0
    ir1 : vec4f,  // inv10, inv11, inv12, 0
    ir2 : vec4f,  // inv20, inv21, inv22, 0
    // BLAS metadata
    blasNodeOffset : u32,
    blasTriOffset  : u32,
    blasTriCount   : u32,
    materialIndex  : u32,
    // Range within the instance buffer
    instanceOffset : u32,
    count          : u32,
    _pad0          : u32,
    _pad1          : u32,
}

@group(0) @binding(0) var<storage, read>       srcData   : array<vec4f>;
@group(0) @binding(1) var<storage, read_write> instances : array<Instance>;
@group(0) @binding(2) var<uniform>             params    : ExpandParams;

// ── Position-only mode ─────────────────────────────────────────────────
// srcData[idx] = vec4f(x, y, z, _)  — local-space position offset.
// Composed transform = parent rotScale * pos + parent translation.
@compute @workgroup_size(256)
fn expandPositions(@builtin(global_invocation_id) gid : vec3u) {
    let idx = gid.x;
    if (idx >= params.count) { return; }

    let pos = srcData[idx];
    let lx = pos.x; let ly = pos.y; let lz = pos.z;

    // World position = R_parent * localPos + T_parent
    let tx = params.r0.x * lx + params.r0.y * ly + params.r0.z * lz + params.r0.w;
    let ty = params.r1.x * lx + params.r1.y * ly + params.r1.z * lz + params.r1.w;
    let tz = params.r2.x * lx + params.r2.y * ly + params.r2.z * lz + params.r2.w;

    let i = params.instanceOffset + idx;

    // Transform rows: parent rotscale + per-instance translation
    instances[i].transform0 = vec4f(params.r0.xyz, tx);
    instances[i].transform1 = vec4f(params.r1.xyz, ty);
    instances[i].transform2 = vec4f(params.r2.xyz, tz);

    // Inverse translation = -invRotScale * t
    let ivx = -(params.ir0.x * tx + params.ir0.y * ty + params.ir0.z * tz);
    let ivy = -(params.ir1.x * tx + params.ir1.y * ty + params.ir1.z * tz);
    let ivz = -(params.ir2.x * tx + params.ir2.y * ty + params.ir2.z * tz);

    instances[i].invTransform0 = vec4f(params.ir0.xyz, ivx);
    instances[i].invTransform1 = vec4f(params.ir1.xyz, ivy);
    instances[i].invTransform2 = vec4f(params.ir2.xyz, ivz);

    instances[i].metadata = vec4u(
        params.blasNodeOffset, params.blasTriOffset,
        params.blasTriCount,   params.materialIndex,
    );
}

// ── Full-transform mode ────────────────────────────────────────────────
// srcData stores 3 consecutive vec4f per instance (row-major 4×3 affine):
//   srcData[idx*3+0] = vec4f(r00, r01, r02, tx)
//   srcData[idx*3+1] = vec4f(r10, r11, r12, ty)
//   srcData[idx*3+2] = vec4f(r20, r21, r22, tz)
// Composed transform = M_parent * M_instance.
@compute @workgroup_size(256)
fn expandTransforms(@builtin(global_invocation_id) gid : vec3u) {
    let idx = gid.x;
    if (idx >= params.count) { return; }

    let base = idx * 3u;
    let iR0 = srcData[base + 0u];
    let iR1 = srcData[base + 1u];
    let iR2 = srcData[base + 2u];

    // Compose rotation/scale: R_comp = R_parent * R_inst  (row × col)
    let c00 = params.r0.x*iR0.x + params.r0.y*iR1.x + params.r0.z*iR2.x;
    let c01 = params.r0.x*iR0.y + params.r0.y*iR1.y + params.r0.z*iR2.y;
    let c02 = params.r0.x*iR0.z + params.r0.y*iR1.z + params.r0.z*iR2.z;
    let c10 = params.r1.x*iR0.x + params.r1.y*iR1.x + params.r1.z*iR2.x;
    let c11 = params.r1.x*iR0.y + params.r1.y*iR1.y + params.r1.z*iR2.y;
    let c12 = params.r1.x*iR0.z + params.r1.y*iR1.z + params.r1.z*iR2.z;
    let c20 = params.r2.x*iR0.x + params.r2.y*iR1.x + params.r2.z*iR2.x;
    let c21 = params.r2.x*iR0.y + params.r2.y*iR1.y + params.r2.z*iR2.y;
    let c22 = params.r2.x*iR0.z + params.r2.y*iR1.z + params.r2.z*iR2.z;

    // Compose translation: T_comp = R_parent * T_inst + T_parent
    let it = vec3f(iR0.w, iR1.w, iR2.w);
    let tx = params.r0.x*it.x + params.r0.y*it.y + params.r0.z*it.z + params.r0.w;
    let ty = params.r1.x*it.x + params.r1.y*it.y + params.r1.z*it.z + params.r1.w;
    let tz = params.r2.x*it.x + params.r2.y*it.y + params.r2.z*it.z + params.r2.w;

    let i = params.instanceOffset + idx;

    instances[i].transform0 = vec4f(c00, c01, c02, tx);
    instances[i].transform1 = vec4f(c10, c11, c12, ty);
    instances[i].transform2 = vec4f(c20, c21, c22, tz);

    // Inverse of composed 3×3
    let det = c00*(c11*c22 - c12*c21) - c01*(c10*c22 - c12*c20) + c02*(c10*c21 - c11*c20);
    let id  = select(1.0, 1.0 / det, abs(det) > 1e-20);

    let iv00 = (c11*c22 - c12*c21)*id;
    let iv01 = (c02*c21 - c01*c22)*id;
    let iv02 = (c01*c12 - c02*c11)*id;
    let iv10 = (c12*c20 - c10*c22)*id;
    let iv11 = (c00*c22 - c02*c20)*id;
    let iv12 = (c02*c10 - c00*c12)*id;
    let iv20 = (c10*c21 - c11*c20)*id;
    let iv21 = (c01*c20 - c00*c21)*id;
    let iv22 = (c00*c11 - c01*c10)*id;

    instances[i].invTransform0 = vec4f(iv00, iv01, iv02, -(iv00*tx + iv01*ty + iv02*tz));
    instances[i].invTransform1 = vec4f(iv10, iv11, iv12, -(iv10*tx + iv11*ty + iv12*tz));
    instances[i].invTransform2 = vec4f(iv20, iv21, iv22, -(iv20*tx + iv21*ty + iv22*tz));

    instances[i].metadata = vec4u(
        params.blasNodeOffset, params.blasTriOffset,
        params.blasTriCount,   params.materialIndex,
    );
}
`;
