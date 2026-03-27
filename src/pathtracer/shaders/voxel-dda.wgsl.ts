export const voxelDDAShader = /* wgsl */`
// Sparse Voxel Octree traversal — stack-based descent through occupancy mip pyramid.
// Uses Morton/Z-order indexed dense mip levels as an implicit octree.
// Front-to-back child ordering via ray direction sign bits.

struct VoxelGridParams {
    gridMin    : vec3f,
    voxelSize  : f32,     // finest-level voxel size
    gridMax    : vec3f,
    gridRes    : u32,     // finest-level resolution
    numLevels  : u32,
    _pad0      : u32,
    _pad1      : u32,
    _pad2      : u32,
    mipOffsets : array<vec4u, 2>,  // 8 u32s packed as 2 vec4u (level offsets into voxelGrid)
}

@group(0) @binding(5) var<storage, read> voxelGrid : array<u32>;
@group(0) @binding(6) var<uniform> voxelParams : VoxelGridParams;
@group(0) @binding(7) var<storage, read> voxelColors : array<u32>;

fn unpackRGBA8(v: u32) -> vec4f {
    return vec4f(
        f32(v & 0xffu) / 255.0,
        f32((v >> 8u) & 0xffu) / 255.0,
        f32((v >> 16u) & 0xffu) / 255.0,
        f32((v >> 24u) & 0xffu) / 255.0,
    );
}

const MAX_SVO_ITERATIONS = 2048u;
const MAX_LEVELS = 10u;

fn getMipOffset(level: u32) -> u32 {
    let vecIdx = level / 4u;
    let compIdx = level % 4u;
    if (vecIdx == 0u) { return voxelParams.mipOffsets[0][compIdx]; }
    return voxelParams.mipOffsets[1][compIdx];
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

fn mortonIdx(coord: vec3u) -> u32 {
    return inflate(coord.x) | (inflate(coord.y) << 1u) | (inflate(coord.z) << 2u);
}

// Also used for signed coords (clamped positive by callers)
fn voxelIdx3D(coord: vec3i, res: i32) -> u32 {
    return inflate(u32(coord.x)) | (inflate(u32(coord.y)) << 1u) | (inflate(u32(coord.z)) << 2u);
}

// Ray-AABB intersection returning (tNear, tFar). Returns tNear > tFar on miss.
fn intersectAABB(ray: Ray, invDir: vec3f, boxMin: vec3f, boxMax: vec3f) -> vec2f {
    let t1 = (boxMin - ray.origin) * invDir;
    let t2 = (boxMax - ray.origin) * invDir;
    let tSmall = min(t1, t2);
    let tBig   = max(t1, t2);
    let tNear = max(max(tSmall.x, tSmall.y), tSmall.z);
    let tFar  = min(min(tBig.x, tBig.y), tBig.z);
    return vec2f(tNear, tFar);
}

// Compute entry normal from the face the ray enters through
fn computeHitNormal(ray: Ray, invDir: vec3f, boxMin: vec3f, boxMax: vec3f) -> vec3f {
    let t1 = (boxMin - ray.origin) * invDir;
    let t2 = (boxMax - ray.origin) * invDir;
    let tEntry = min(t1, t2); // entry planes per axis
    var n = vec3f(0.0);
    if (tEntry.x > tEntry.y && tEntry.x > tEntry.z) {
        n.x = select(1.0, -1.0, ray.dir.x > 0.0);
    } else if (tEntry.y > tEntry.z) {
        n.y = select(1.0, -1.0, ray.dir.y > 0.0);
    } else {
        n.z = select(1.0, -1.0, ray.dir.z > 0.0);
    }
    return n;
}

// Stack-based SVO traversal through the occupancy mip pyramid
fn traceSVO(ray: Ray, maxDist: f32, skipOriginVoxel: bool) -> HitInfo {
    var result: HitInfo;
    result.hit = false;
    result.t = maxDist;

    let invDir = 1.0 / ray.dir;
    let numLevels = voxelParams.numLevels;
    if (numLevels == 0u) { return result; }

    // Intersect the entire grid AABB
    let gridAABB = intersectAABB(ray, invDir, voxelParams.gridMin, voxelParams.gridMax);
    if (gridAABB.y < 0.0 || gridAABB.x > gridAABB.y || gridAABB.x > maxDist) {
        return result;
    }

    let rootLevel = numLevels - 1u;

    // Check root occupancy
    if (voxelGrid[getMipOffset(rootLevel)] == 0u) { return result; }

    // Front-to-back child ordering: flip octant bits based on ray direction
    // When dir component < 0, flip that bit so child 0 is always "near" side
    let flipMask = vec3u(
        select(0u, 1u, ray.dir.x < 0.0),
        select(0u, 1u, ray.dir.y < 0.0),
        select(0u, 1u, ray.dir.z < 0.0),
    );

    // Stack: one entry per octree depth level
    // Each entry stores: parent coord, next child to process, parent level
    var stackCoordX : array<i32, MAX_LEVELS>;
    var stackCoordY : array<i32, MAX_LEVELS>;
    var stackCoordZ : array<i32, MAX_LEVELS>;
    var stackChild  : array<u32, MAX_LEVELS>;
    var stackLvl    : array<u32, MAX_LEVELS>;
    var stackDepth  = 0u;

    // Precompute the origin voxel coord for self-intersection avoidance
    let vs = voxelParams.voxelSize;
    let res = i32(voxelParams.gridRes);
    var originVoxel = vec3i(-1);
    if (skipOriginVoxel) {
        let gp = (ray.origin - voxelParams.gridMin) / vs;
        originVoxel = vec3i(floor(clamp(gp, vec3f(0.0), vec3f(f32(res) - 0.001))));
    }

    // Start traversal at root level
    var level = rootLevel;
    var coord = vec3i(0);
    var child = 0u;
    var iterations = 0u;

    loop {
        if (iterations >= MAX_SVO_ITERATIONS) { break; }
        iterations++;

        if (level == 0u) {
            // ── Finest level: check actual voxel material ──
            let matVal = voxelGrid[mortonIdx(vec3u(coord))];
            if (matVal > 0u) {
                // Self-intersection avoidance: skip the voxel the ray starts in
                let isOrigin = skipOriginVoxel &&
                    coord.x == originVoxel.x && coord.y == originVoxel.y && coord.z == originVoxel.z;
                if (!isOrigin) {
                    let voxMin = voxelParams.gridMin + vec3f(coord) * vs;
                    let voxMax = voxMin + vec3f(vs);
                    let tt = intersectAABB(ray, invDir, voxMin, voxMax);
                    let tHit = max(tt.x, 0.0);
                    if (tHit < result.t) {
                        result.hit = true;
                        result.t = tHit;
                        result.worldPos = ray.origin + ray.dir * tHit;
                        result.matIndex = matVal - 1u;
                        result.triIndex = 0u;
                        result.instanceId = 0u;
                        result.u = 0.0;
                        result.v = 0.0;
                        result.worldNorm = computeHitNormal(ray, invDir, voxMin, voxMax);
                        // First hit in front-to-back order is closest — return immediately
                        return result;
                    }
                }
            }

            // Pop stack
            if (stackDepth == 0u) { break; }
            stackDepth--;
            let sIdx = stackDepth;
            coord = vec3i(stackCoordX[sIdx], stackCoordY[sIdx], stackCoordZ[sIdx]);
            child = stackChild[sIdx];
            level = stackLvl[sIdx];
            continue;
        }

        // ── Internal node: iterate children in front-to-back order ──
        let childLevel = level - 1u;
        let childShift = childLevel;
        let childRes = voxelParams.gridRes >> childShift;
        let childVs = vs * f32(1u << childShift);
        let childMipOffset = getMipOffset(childLevel);
        var foundChild = false;

        while (child < 8u) {
            // Map child index through flipMask for front-to-back ordering
            let dx = (child & 1u) ^ flipMask.x;
            let dy = ((child >> 1u) & 1u) ^ flipMask.y;
            let dz = ((child >> 2u) & 1u) ^ flipMask.z;
            let childCoord = vec3i(
                coord.x * 2 + i32(dx),
                coord.y * 2 + i32(dy),
                coord.z * 2 + i32(dz),
            );
            child++;

            // Bounds check
            let cRes = i32(childRes);
            if (childCoord.x < 0 || childCoord.y < 0 || childCoord.z < 0 ||
                childCoord.x >= cRes || childCoord.y >= cRes || childCoord.z >= cRes) {
                continue;
            }

            // Check occupancy at child's mip level
            let occupancy = voxelGrid[childMipOffset + mortonIdx(vec3u(childCoord))];
            if (occupancy == 0u) { continue; }

            // Ray-AABB intersection with child's world-space box
            let childMin = voxelParams.gridMin + vec3f(childCoord) * childVs;
            let childMax = childMin + vec3f(childVs);
            let tt = intersectAABB(ray, invDir, childMin, childMax);

            // Skip if ray misses, or child is behind us, or farther than best hit
            if (tt.y < 0.0 || tt.x > tt.y || tt.x > result.t) { continue; }

            // Descend into this child — push current state onto stack
            let sIdx = stackDepth;
            stackCoordX[sIdx] = coord.x;
            stackCoordY[sIdx] = coord.y;
            stackCoordZ[sIdx] = coord.z;
            stackChild[sIdx] = child; // next sibling to try on return
            stackLvl[sIdx] = level;
            stackDepth++;

            level = childLevel;
            coord = childCoord;
            child = 0u;
            foundChild = true;
            break;
        }

        if (!foundChild) {
            // All children exhausted at this level — pop stack
            if (stackDepth == 0u) { break; }
            stackDepth--;
            let sIdx = stackDepth;
            coord = vec3i(stackCoordX[sIdx], stackCoordY[sIdx], stackCoordZ[sIdx]);
            child = stackChild[sIdx];
            level = stackLvl[sIdx];
        }
    }

    return result;
}

// ── Voxel color sampling for indirect lighting ──────────────────────────

// Sample voxel color with trilinear interpolation from the color mip pyramid.
// Returns (rgb, alpha) where alpha represents occupancy fraction.
fn sampleVoxelMip(pos: vec3f, level: u32) -> vec4f {
    let levelRes = voxelParams.gridRes >> level;
    let levelVs = voxelParams.voxelSize * f32(1u << level);
    let gridPos = (pos - voxelParams.gridMin) / levelVs;

    // Bounds check with margin for trilinear
    let fRes = f32(levelRes);
    if (gridPos.x < -0.5 || gridPos.y < -0.5 || gridPos.z < -0.5 ||
        gridPos.x > fRes + 0.5 || gridPos.y > fRes + 0.5 || gridPos.z > fRes + 0.5) {
        return vec4f(0.0);
    }

    // Shift to voxel centers for trilinear
    let centered = gridPos - 0.5;
    let base = vec3i(floor(centered));
    let frac = centered - vec3f(base);
    let iRes = i32(levelRes);
    let mipOff = getMipOffset(level);

    var result = vec4f(0.0);
    for (var dz = 0; dz <= 1; dz++) {
        for (var dy = 0; dy <= 1; dy++) {
            for (var dx = 0; dx <= 1; dx++) {
                let c = clamp(base + vec3i(dx, dy, dz), vec3i(0), vec3i(iRes - 1));
                let packed = voxelColors[mipOff + mortonIdx(vec3u(c))];
                let s = unpackRGBA8(packed);
                let wx = select(1.0 - frac.x, frac.x, dx == 1);
                let wy = select(1.0 - frac.y, frac.y, dy == 1);
                let wz = select(1.0 - frac.z, frac.z, dz == 1);
                result += s * wx * wy * wz;
            }
        }
    }
    return result;
}

fn getVoxelParams() -> VoxelGridParams {
    return voxelParams;
}

fn traceBVH(ray: Ray) -> HitInfo {
    return traceSVO(ray, 1e30, false);
}

fn traceBVHShadow(ray: Ray, maxDist: f32) -> bool {
    let hit = traceSVO(ray, maxDist, true);
    return hit.hit;
}

fn traceBVHInternal(ray: Ray, anyHit: bool, maxDist: f32) -> HitInfo {
    return traceSVO(ray, maxDist, true);
}

// Dummy for voxel mode — no triangle geometry, MIS falls back gracefully
fn getWorldTriArea(triIdx: u32, instId: u32) -> f32 {
    return 1.0;
}
`;
