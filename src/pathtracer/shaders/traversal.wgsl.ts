export const traversalShader = /* wgsl */`
// Requires: intersection.wgsl (Ray, HitInfo, rayAABB, rayTriangle)
// Requires: BVHNode struct, Instance struct, triangle/node/instance storage bindings

const TLAS_STACK_SIZE = 32u;
const BLAS_STACK_SIZE = 32u;
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
    nodeOffset: u32,
    triOffset: u32,
    triCount: u32,
    tMax: f32,
) -> HitInfo {
    var hit: HitInfo;
    hit.t = tMax;
    hit.hit = false;

    if (triCount == 0u) { return hit; }

    var stack: array<u32, BLAS_STACK_SIZE>;
    var stackPtr = 0u;
    stack[0] = nodeOffset; // BLAS root is at nodeOffset in the combined buffer
    stackPtr = 1u;

    var iter = 0u;
    while (stackPtr > 0u && iter < 16384u) {
        iter += 1u;
        stackPtr -= 1u;
        let nodeIdx = stack[stackPtr];
        let node = blasNodes[nodeIdx];

        let tHit = rayAABB(ray, node.boundsMin, node.boundsMax, hit.t);
        if (tHit < 0.0) { continue; }

        if (node.leftChild < 0) {
            // Leaf: triangle index encoded as -(triIdx + 1)
            let triIdx = u32(-(node.leftChild + 1));
            let base = (triOffset + triIdx) * TRI_STRIDE;
            let v0 = vec3f(triangles[base+0u], triangles[base+1u], triangles[base+2u]);
            let v1 = vec3f(triangles[base+3u], triangles[base+4u], triangles[base+5u]);
            let v2 = vec3f(triangles[base+6u], triangles[base+7u], triangles[base+8u]);
            let tuv = rayTriangle(ray, v0, v1, v2);
            if (tuv.x > 0.0 && tuv.x < hit.t) {
                hit.t = tuv.x;
                hit.u = tuv.y;
                hit.v = tuv.z;
                hit.triIndex = triOffset + triIdx;
                hit.hit = true;

                let n0 = vec3f(triangles[base+9u],  triangles[base+10u], triangles[base+11u]);
                let n1 = vec3f(triangles[base+12u], triangles[base+13u], triangles[base+14u]);
                let n2 = vec3f(triangles[base+15u], triangles[base+16u], triangles[base+17u]);
                let w = 1.0 - tuv.y - tuv.z;
                hit.worldNorm = normalize(n0 * w + n1 * tuv.y + n2 * tuv.z);
                hit.worldPos = ray.origin + ray.dir * tuv.x;
            }
        } else {
            // Internal: add nodeOffset since child indices are local to this BLAS
            if (stackPtr < BLAS_STACK_SIZE - 2u) {
                stack[stackPtr] = nodeOffset + u32(node.leftChild);
                stackPtr += 1u;
                stack[stackPtr] = nodeOffset + u32(node.rightChild);
                stackPtr += 1u;
            }
        }
    }
    return hit;
}

fn traceBVH(ray: Ray) -> HitInfo {
    var hit: HitInfo;
    hit.t = 1e30;
    hit.hit = false;

    var stack: array<u32, TLAS_STACK_SIZE>;
    var stackPtr = 0u;
    stack[0] = 0u; // TLAS root
    stackPtr = 1u;

    var iter = 0u;
    while (stackPtr > 0u && iter < 16384u) {
        iter += 1u;
        stackPtr -= 1u;
        let nodeIdx = stack[stackPtr];
        let node = tlasNodes[nodeIdx];

        let tHit = rayAABB(ray, node.boundsMin, node.boundsMax, hit.t);
        if (tHit < 0.0) { continue; }

        if (node.leftChild < 0) {
            // Leaf: encodes instance index as -(instIdx + 1)
            let instIdx = u32(-(node.leftChild + 1));
            let inst = instances[instIdx];

            // Transform ray to instance local space
            let localRay = transformRayToLocal(ray, inst);

            // Traverse this instance's BLAS
            let blasHit = traverseBLAS(
                localRay,
                inst.blasNodeOffset,
                inst.blasTriOffset,
                inst.blasTriCount,
                hit.t,
            );

            if (blasHit.hit && blasHit.t < hit.t) {
                hit = blasHit;
                hit.instanceId = instIdx;
                hit.matIndex = inst.materialIndex;
                // Transform hit back to world space
                hit.worldPos = transformPointToWorld(blasHit.worldPos, inst);
                hit.worldNorm = transformNormalToWorld(blasHit.worldNorm, inst);
            }
        } else {
            // Internal: push children
            if (stackPtr < TLAS_STACK_SIZE - 1u) {
                stack[stackPtr] = u32(node.leftChild);
                stackPtr += 1u;
                stack[stackPtr] = u32(node.rightChild);
                stackPtr += 1u;
            }
        }
    }
    return hit;
}
`;
