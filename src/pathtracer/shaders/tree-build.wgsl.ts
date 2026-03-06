export const treeBuildShader = /* wgsl */`
struct BVHNode {
    boundsMin  : vec3f,
    leftChild  : i32,
    boundsMax  : vec3f,
    rightChild : i32,
}

struct Params {
    leafCount : u32,
    _pad      : vec3u,
}

@group(0) @binding(0) var<storage, read>       sortedMorton : array<u32>;
@group(0) @binding(1) var<storage, read_write> nodes        : array<BVHNode>;
@group(0) @binding(2) var<storage, read_write> parents      : array<i32>;
@group(0) @binding(3) var<uniform>             params       : Params;

fn clz(v: u32) -> u32 {
    if (v == 0u) { return 32u; }
    var n = 0u;
    var x = v;
    if ((x & 0xFFFF0000u) == 0u) { n += 16u; x <<= 16u; }
    if ((x & 0xFF000000u) == 0u) { n +=  8u; x <<=  8u; }
    if ((x & 0xF0000000u) == 0u) { n +=  4u; x <<=  4u; }
    if ((x & 0xC0000000u) == 0u) { n +=  2u; x <<=  2u; }
    if ((x & 0x80000000u) == 0u) { n +=  1u; }
    return n;
}

fn delta(i: i32, j: i32, n: i32) -> i32 {
    if (j < 0 || j >= n) { return -1; }
    let ki = sortedMorton[u32(i)];
    let kj = sortedMorton[u32(j)];
    if (ki == kj) {
        return i32(32u - clz(u32(i) ^ u32(j))) + 32;
    }
    return i32(32u - clz(ki ^ kj));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = i32(gid.x);
    let n = i32(params.leafCount);
    if (idx >= n - 1) { return; }

    // Determine direction of the range
    let dPlus  = delta(idx, idx + 1, n);
    let dMinus = delta(idx, idx - 1, n);
    let d = select(-1, 1, dPlus > dMinus);
    let dMin = select(dPlus, dMinus, dPlus > dMinus);

    // Compute upper bound for the length of the range
    var lMax = 2;
    while (delta(idx, idx + lMax * d, n) > dMin) {
        lMax *= 2;
    }

    // Find the other end using binary search
    var l = 0;
    var t = lMax / 2;
    while (t >= 1) {
        if (delta(idx, idx + (l + t) * d, n) > dMin) {
            l += t;
        }
        t /= 2;
    }
    let j = idx + l * d;

    // Find the split position
    let dNode = delta(idx, j, n);
    var s = 0;
    var divider = 2;
    t = (l + divider - 1) / divider;
    while (t >= 1) {
        if (delta(idx, idx + (s + t) * d, n) > dNode) {
            s += t;
        }
        divider *= 2;
        t = (l + divider - 1) / divider;
    }
    let split = idx + s * d + min(d, 0);

    // Internal node indices: 0..n-2
    // Leaf node indices: n-1..2n-2
    let leftIsLeaf  = (split == min(idx, j));
    let rightIsLeaf = (split + 1 == max(idx, j));
    let leftIdx  = select(split, split + n - 1, leftIsLeaf);
    let rightIdx = select(split + 1, split + 1 + n - 1, rightIsLeaf);

    nodes[u32(idx)].leftChild  = i32(leftIdx);
    nodes[u32(idx)].rightChild = i32(rightIdx);

    // Record parent pointers
    parents[leftIdx]  = idx;
    parents[rightIdx] = idx;
}
`;
