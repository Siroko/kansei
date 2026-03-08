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
    anyHit: bool,
) -> HitInfo {
    var hit: HitInfo;
    hit.t = tMax;
    hit.hit = false;

    if (triCount == 0u) { return hit; }

    var stack: array<u32, BLAS_STACK_SIZE>;
    var stackPtr = 0u;
    stack[0] = nodeOffset;
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
            let triStart = u32(-(node.leftChild + 1));
            let triCount = u32(node.rightChild);
            for (var ti = 0u; ti < triCount; ti++) {
                let triIdx = triStart + ti;
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

                    if (anyHit) { return hit; }

                    let n0 = vec3f(triangles[base+9u],  triangles[base+10u], triangles[base+11u]);
                    let n1 = vec3f(triangles[base+12u], triangles[base+13u], triangles[base+14u]);
                    let n2 = vec3f(triangles[base+15u], triangles[base+16u], triangles[base+17u]);
                    let w = 1.0 - tuv.y - tuv.z;
                    hit.worldNorm = normalize(n0 * w + n1 * tuv.y + n2 * tuv.z);
                    hit.worldPos = ray.origin + ray.dir * tuv.x;
                }
            }
        } else {
            if (stackPtr < BLAS_STACK_SIZE - 2u) {
                // Ordered traversal: visit nearer child first
                let leftIdx = nodeOffset + u32(node.leftChild);
                let rightIdx = nodeOffset + u32(node.rightChild);
                let leftNode = blasNodes[leftIdx];
                let rightNode = blasNodes[rightIdx];
                let tLeft = rayAABB(ray, leftNode.boundsMin, leftNode.boundsMax, hit.t);
                let tRight = rayAABB(ray, rightNode.boundsMin, rightNode.boundsMax, hit.t);

                if (tLeft >= 0.0 && tRight >= 0.0) {
                    // Push farther first (popped last = visited second)
                    if (tLeft < tRight) {
                        stack[stackPtr] = rightIdx; stackPtr += 1u;
                        stack[stackPtr] = leftIdx;  stackPtr += 1u;
                    } else {
                        stack[stackPtr] = leftIdx;  stackPtr += 1u;
                        stack[stackPtr] = rightIdx; stackPtr += 1u;
                    }
                } else if (tLeft >= 0.0) {
                    stack[stackPtr] = leftIdx; stackPtr += 1u;
                } else if (tRight >= 0.0) {
                    stack[stackPtr] = rightIdx; stackPtr += 1u;
                }
            }
        }
    }
    return hit;
}

fn traceBVHInternal(ray: Ray, anyHit: bool) -> HitInfo {
    var hit: HitInfo;
    hit.t = 1e30;
    hit.hit = false;

    var stack: array<u32, TLAS_STACK_SIZE>;
    var stackPtr = 0u;
    stack[0] = 0u;
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
            let instIdx = u32(-(node.leftChild + 1));
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
        } else {
            if (stackPtr < TLAS_STACK_SIZE - 2u) {
                let leftIdx = u32(node.leftChild);
                let rightIdx = u32(node.rightChild);
                let leftNode = tlasNodes[leftIdx];
                let rightNode = tlasNodes[rightIdx];
                let tLeft = rayAABB(ray, leftNode.boundsMin, leftNode.boundsMax, hit.t);
                let tRight = rayAABB(ray, rightNode.boundsMin, rightNode.boundsMax, hit.t);

                if (tLeft >= 0.0 && tRight >= 0.0) {
                    if (tLeft < tRight) {
                        stack[stackPtr] = rightIdx; stackPtr += 1u;
                        stack[stackPtr] = leftIdx;  stackPtr += 1u;
                    } else {
                        stack[stackPtr] = leftIdx;  stackPtr += 1u;
                        stack[stackPtr] = rightIdx; stackPtr += 1u;
                    }
                } else if (tLeft >= 0.0) {
                    stack[stackPtr] = leftIdx; stackPtr += 1u;
                } else if (tRight >= 0.0) {
                    stack[stackPtr] = rightIdx; stackPtr += 1u;
                }
            }
        }
    }
    return hit;
}

fn traceBVH(ray: Ray) -> HitInfo {
    return traceBVHInternal(ray, false);
}

fn traceBVHShadow(ray: Ray) -> bool {
    let hit = traceBVHInternal(ray, true);
    return hit.hit;
}
`;
