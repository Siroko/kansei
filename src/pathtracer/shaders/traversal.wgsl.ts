export const traversalShader = /* wgsl */`
// Requires: intersection.wgsl (Ray, HitInfo, rayAABB, rayTriangle)
// Requires: BVH bindings (triangles, bvh4Nodes, tlasBvh4Nodes, instances) declared externally

const TLAS_STACK_SIZE = 16u;
const BLAS_STACK_SIZE = 64u;
const TRI_STRIDE = 24u; // 24 floats per triangle

struct Instance {
    transform0   : vec4f, // row 0 of 3×4 affine
    transform1   : vec4f, // row 1
    transform2   : vec4f, // row 2
    invTransform0: vec4f, // row 0 of inverse 3×4
    invTransform1: vec4f, // row 1
    invTransform2: vec4f, // row 2
    blasNodeOffset: u32,
    blasTriOffset : u32,
    blasTriCount  : u32,
    materialIndex: u32,
}

fn transformRayToLocal(ray: Ray, inst: Instance) -> Ray {
    var local: Ray;
    // Apply inverse affine transform
    local.origin = vec3f(
        dot(inst.invTransform0.xyz, ray.origin) + inst.invTransform0.w,
        dot(inst.invTransform1.xyz, ray.origin) + inst.invTransform1.w,
        dot(inst.invTransform2.xyz, ray.origin) + inst.invTransform2.w,
    );
    local.dir = vec3f(
        dot(inst.invTransform0.xyz, ray.dir),
        dot(inst.invTransform1.xyz, ray.dir),
        dot(inst.invTransform2.xyz, ray.dir),
    );
    return local;
}

fn transformPointToWorld(p: vec3f, inst: Instance) -> vec3f {
    return vec3f(
        dot(inst.transform0.xyz, p) + inst.transform0.w,
        dot(inst.transform1.xyz, p) + inst.transform1.w,
        dot(inst.transform2.xyz, p) + inst.transform2.w,
    );
}

fn transformNormalToWorld(n: vec3f, inst: Instance) -> vec3f {
    // Normal transform = (M^-1)^T * n, i.e. columns of the inverse matrix
    return normalize(vec3f(
        inst.invTransform0.x * n.x + inst.invTransform1.x * n.y + inst.invTransform2.x * n.z,
        inst.invTransform0.y * n.x + inst.invTransform1.y * n.y + inst.invTransform2.y * n.z,
        inst.invTransform0.z * n.x + inst.invTransform1.z * n.y + inst.invTransform2.z * n.z,
    ));
}

fn traverseBLAS(
    ray: Ray,
    bvh4Root: u32,
    triOffset: u32,
    triCount: u32,
    tMax: f32,
    anyHit: bool,
) -> HitInfo {
    var hit: HitInfo;
    hit.t = tMax;
    hit.hit = false;

    if (triCount == 0u) { return hit; }

    let invDir = 1.0 / ray.dir;
    let ox = vec4f(ray.origin.x);
    let oy = vec4f(ray.origin.y);
    let oz = vec4f(ray.origin.z);
    let idx4 = vec4f(invDir.x);
    let idy4 = vec4f(invDir.y);
    let idz4 = vec4f(invDir.z);

    var stack: array<u32, BLAS_STACK_SIZE>;
    var stackPtr = 0u;
    stack[0] = bvh4Root;
    stackPtr = 1u;

    var iter = 0u;
    while (stackPtr > 0u && iter < 16384u) {
        iter += 1u;
        stackPtr -= 1u;
        let base = stack[stackPtr] * 8u;

        // Read BVH4 node (8 vec4f = 128 bytes)
        let childMinX = bvh4Nodes[base + 0u];
        let childMaxX = bvh4Nodes[base + 1u];
        let childMinY = bvh4Nodes[base + 2u];
        let childMaxY = bvh4Nodes[base + 3u];
        let childMinZ = bvh4Nodes[base + 4u];
        let childMaxZ = bvh4Nodes[base + 5u];
        let children  = bitcast<vec4i>(bvh4Nodes[base + 6u]);
        let triCounts = bitcast<vec4u>(bvh4Nodes[base + 7u]);

        // Test all 4 child AABBs simultaneously
        let t1x = (childMinX - ox) * idx4;
        let t2x = (childMaxX - ox) * idx4;
        let t1y = (childMinY - oy) * idy4;
        let t2y = (childMaxY - oy) * idy4;
        let t1z = (childMinZ - oz) * idz4;
        let t2z = (childMaxZ - oz) * idz4;

        let tminV = max(max(min(t1x, t2x), min(t1y, t2y)), min(t1z, t2z));
        let tmaxV = min(min(max(t1x, t2x), max(t1y, t2y)), max(t1z, t2z));

        // Collect and sort hit children by distance (nearest first on stack top)
        var hitDist: array<f32, 4>;
        var hitIdx: array<u32, 4>;
        var hitIsLeaf: array<bool, 4>;
        var hitCount = 0u;

        for (var ci = 0u; ci < 4u; ci++) {
            let tmin_c = tminV[ci];
            let tmax_c = tmaxV[ci];
            if (tmax_c >= 0.0 && tmin_c <= tmax_c && tmin_c <= hit.t) {
                let childIdx = children[ci];

                if (childIdx < 0) {
                    // Leaf: test triangles immediately
                    let triStart = u32(-(childIdx + 1));
                    let triCnt = triCounts[ci];
                    for (var ti = 0u; ti < triCnt; ti++) {
                        let triIdx = triStart + ti;
                        let tbase = (triOffset + triIdx) * TRI_STRIDE;
                        let v0 = vec3f(triangles[tbase+0u], triangles[tbase+1u], triangles[tbase+2u]);
                        let v1 = vec3f(triangles[tbase+3u], triangles[tbase+4u], triangles[tbase+5u]);
                        let v2 = vec3f(triangles[tbase+6u], triangles[tbase+7u], triangles[tbase+8u]);
                        let tuv = rayTriangle(ray, v0, v1, v2);
                        if (tuv.x > 0.0 && tuv.x < hit.t) {
                            hit.t = tuv.x;
                            hit.u = tuv.y;
                            hit.v = tuv.z;
                            hit.triIndex = triOffset + triIdx;
                            hit.hit = true;
                            if (anyHit) { return hit; }
                            let n0 = vec3f(triangles[tbase+9u],  triangles[tbase+10u], triangles[tbase+11u]);
                            let n1 = vec3f(triangles[tbase+12u], triangles[tbase+13u], triangles[tbase+14u]);
                            let n2 = vec3f(triangles[tbase+15u], triangles[tbase+16u], triangles[tbase+17u]);
                            let w = 1.0 - tuv.y - tuv.z;
                            hit.worldNorm = normalize(n0 * w + n1 * tuv.y + n2 * tuv.z);
                            hit.worldPos = ray.origin + ray.dir * tuv.x;
                        }
                    }
                } else {
                    // Internal: collect for sorted push
                    hitDist[hitCount] = max(tmin_c, 0.0);
                    hitIdx[hitCount] = u32(childIdx);
                    hitCount += 1u;
                }
            }
        }

        // Push internal children sorted farthest-first (nearest popped first)
        if (hitCount > 0u) {
            // Simple insertion sort for up to 4 elements (ascending = farthest first)
            for (var i = 1u; i < hitCount; i++) {
                let kd = hitDist[i];
                let ki = hitIdx[i];
                var j = i;
                while (j > 0u && hitDist[j - 1u] < kd) {
                    hitDist[j] = hitDist[j - 1u];
                    hitIdx[j] = hitIdx[j - 1u];
                    j -= 1u;
                }
                hitDist[j] = kd;
                hitIdx[j] = ki;
            }
            // Push as many as fit (sorted farthest-first, so nearest are pushed last = popped first)
            let pushCount = min(hitCount, BLAS_STACK_SIZE - stackPtr);
            for (var i = 0u; i < pushCount; i++) {
                stack[stackPtr] = hitIdx[i];
                stackPtr += 1u;
            }
        }
    }
    return hit;
}

fn traceBVHInternal(ray: Ray, anyHit: bool, maxDist: f32) -> HitInfo {
    var hit: HitInfo;
    hit.t = maxDist;
    hit.hit = false;

    let invDir = 1.0 / ray.dir;
    let ox = vec4f(ray.origin.x);
    let oy = vec4f(ray.origin.y);
    let oz = vec4f(ray.origin.z);
    let idx4 = vec4f(invDir.x);
    let idy4 = vec4f(invDir.y);
    let idz4 = vec4f(invDir.z);

    var stack: array<u32, TLAS_STACK_SIZE>;
    var stackPtr = 0u;
    stack[0] = 0u;
    stackPtr = 1u;

    var iter = 0u;
    while (stackPtr > 0u && iter < 16384u) {
        iter += 1u;
        stackPtr -= 1u;
        let base = stack[stackPtr] * 8u;

        // Read BVH4 node (8 vec4f = 128 bytes, 1 cache line)
        let childMinX = tlasBvh4Nodes[base + 0u];
        let childMaxX = tlasBvh4Nodes[base + 1u];
        let childMinY = tlasBvh4Nodes[base + 2u];
        let childMaxY = tlasBvh4Nodes[base + 3u];
        let childMinZ = tlasBvh4Nodes[base + 4u];
        let childMaxZ = tlasBvh4Nodes[base + 5u];
        let children  = bitcast<vec4i>(tlasBvh4Nodes[base + 6u]);
        let childCnts = bitcast<vec4u>(tlasBvh4Nodes[base + 7u]);

        // Test all 4 child AABBs simultaneously
        let t1x = (childMinX - ox) * idx4;
        let t2x = (childMaxX - ox) * idx4;
        let t1y = (childMinY - oy) * idy4;
        let t2y = (childMaxY - oy) * idy4;
        let t1z = (childMinZ - oz) * idz4;
        let t2z = (childMaxZ - oz) * idz4;

        let tminV = max(max(min(t1x, t2x), min(t1y, t2y)), min(t1z, t2z));
        let tmaxV = min(min(max(t1x, t2x), max(t1y, t2y)), max(t1z, t2z));

        var hitDist: array<f32, 4>;
        var hitIdx: array<u32, 4>;
        var hitCount = 0u;

        for (var ci = 0u; ci < 4u; ci++) {
            let tmin_c = tminV[ci];
            let tmax_c = tmaxV[ci];
            if (tmax_c >= 0.0 && tmin_c <= tmax_c && tmin_c <= hit.t) {
                let childIdx = children[ci];

                if (childIdx < 0) {
                    // Leaf: instance intersection (skip sentinels with count 0)
                    if (childCnts[ci] > 0u) {
                        let instIdx = u32(-(childIdx + 1));
                        let inst = instances[instIdx];
                        let localRay = transformRayToLocal(ray, inst);

                        let blasHit = traverseBLAS(
                            localRay,
                            inst.blasNodeOffset,
                            inst.blasTriOffset,
                            inst.blasTriCount,
                            hit.t,
                            anyHit,
                        );

                        if (blasHit.hit && blasHit.t < hit.t) {
                            hit = blasHit;
                            hit.instanceId = instIdx;
                            hit.matIndex = inst.materialIndex;
                            if (anyHit) { return hit; }
                            hit.worldPos = transformPointToWorld(blasHit.worldPos, inst);
                            hit.worldNorm = transformNormalToWorld(blasHit.worldNorm, inst);
                        }
                    }
                } else {
                    // Internal: collect for sorted push
                    hitDist[hitCount] = max(tmin_c, 0.0);
                    hitIdx[hitCount] = u32(childIdx);
                    hitCount += 1u;
                }
            }
        }

        // Push internal children sorted farthest-first (nearest popped first)
        if (hitCount > 0u) {
            for (var i = 1u; i < hitCount; i++) {
                let kd = hitDist[i];
                let ki = hitIdx[i];
                var j = i;
                while (j > 0u && hitDist[j - 1u] < kd) {
                    hitDist[j] = hitDist[j - 1u];
                    hitIdx[j] = hitIdx[j - 1u];
                    j -= 1u;
                }
                hitDist[j] = kd;
                hitIdx[j] = ki;
            }
            let pushCount = min(hitCount, TLAS_STACK_SIZE - stackPtr);
            for (var i = 0u; i < pushCount; i++) {
                stack[stackPtr] = hitIdx[i];
                stackPtr += 1u;
            }
        }
    }
    return hit;
}

fn traceBVH(ray: Ray) -> HitInfo {
    return traceBVHInternal(ray, false, 1e30);
}

fn traceBVHShadow(ray: Ray, maxDist: f32) -> bool {
    let hit = traceBVHInternal(ray, true, maxDist);
    return hit.hit;
}

// World-space triangle area for MIS on emissive geometry hits
fn getWorldTriArea(triIdx: u32, instId: u32) -> f32 {
    let base = triIdx * TRI_STRIDE;
    let v0 = vec3f(triangles[base], triangles[base+1u], triangles[base+2u]);
    let v1 = vec3f(triangles[base+3u], triangles[base+4u], triangles[base+5u]);
    let v2 = vec3f(triangles[base+6u], triangles[base+7u], triangles[base+8u]);
    let inst = instances[instId];
    let w0 = transformPointToWorld(v0, inst);
    let w1 = transformPointToWorld(v1, inst);
    let w2 = transformPointToWorld(v2, inst);
    return 0.5 * length(cross(w1 - w0, w2 - w0));
}
`;
