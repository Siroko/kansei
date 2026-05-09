// Repulsion: push ALL particles apart from nearby particles of DIFFERENT words.
// Uses the same spatial hash as steering. Dispatched over particleCount.
// Runs after integrate, before verlet, so verlet can re-snap the chain.

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var<storage, read> sortedIndices: array<u32>;
@group(0) @binding(3) var<storage, read> cellOffsets: array<u32>;
@group(0) @binding(4) var<storage, read> cellCounts: array<u32>;
@group(0) @binding(5) var<storage, read> wordMetaBuf: array<vec4<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }

    let pos = positions[idx].xyz;
    let myWordId = wordMetaBuf[idx].x;
    let myCoord = getCellCoord(pos, params);

    var pushAccum = vec3<f32>(0.0);
    var pushCount = 0u;
    // Use a tighter radius for inter-letter repulsion (half of separation radius)
    let repulsionRadius = params.repulsionRadius;

    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighborCoord = myCoord + vec3<i32>(dx, dy, dz);
                let dims = vec3<i32>(i32(params.gridDimsX), i32(params.gridDimsY), i32(params.gridDimsZ));
                if (any(neighborCoord < vec3<i32>(0)) || any(neighborCoord >= dims)) { continue; }

                let cell = cellHash(neighborCoord, params);
                let start = cellOffsets[cell];
                let count = cellCounts[cell];

                // Cap iterations per cell to avoid O(N²) in dense clusters
                let maxPerCell = min(count, params.maxPerCell);
                for (var i = 0u; i < maxPerCell; i++) {
                    let otherIdx = sortedIndices[start + i];
                    if (otherIdx == idx) { continue; }

                    // Only repel particles from DIFFERENT words
                    if (wordMetaBuf[otherIdx].x == myWordId) { continue; }

                    let otherPos = positions[otherIdx].xyz;
                    let diff = pos - otherPos;
                    let dist = length(diff);

                    if (dist > 0.001 && dist < repulsionRadius) {
                        // Soft push: stronger when closer
                        pushAccum += normalize(diff) * (1.0 - dist / repulsionRadius);
                        pushCount += 1u;
                    }
                }
            }
        }
    }

    if (pushCount > 0u) {
        let push = pushAccum / f32(pushCount);
        let nudge = push * params.repulsionStrength * params.dt;
        positions[idx] = vec4<f32>(pos + nudge, 1.0);
    }
}
