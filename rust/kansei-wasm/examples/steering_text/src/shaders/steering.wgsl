// Steering: separation, cohesion, alignment, attractor, wander, mouse and bounds forces for
// each vehicle.
// Dispatched over vehicleCount.

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<storage, read> sortedIndices: array<u32>;
@group(0) @binding(4) var<storage, read> cellOffsets: array<u32>;
@group(0) @binding(5) var<storage, read> cellCounts: array<u32>;
@group(0) @binding(6) var<storage, read> vehicleIndices: array<u32>;
@group(0) @binding(7) var<storage, read> wordMeta: array<vec4<u32>>;
@group(0) @binding(8) var<uniform> grid: NeighbourGrid;

// Simple hash for per-vehicle random seed
fn pcgHash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hashToFloat(h: u32) -> f32 {
    return f32(h) / f32(0xFFFFFFFFu);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vid = gid.x;
    if (vid >= params.vehicleCount) { return; }

    let particleIdx = vehicleIndices[vid];
    let pos = positions[particleIdx].xyz;
    let vel = velocities[particleIdx].xyz;

    var force = vec3<f32>(0.0);

    // ── Boids: separation + cohesion + alignment via spatial hash ──
    let myCoord = neighbourCell(pos, grid);
    var sepAccum = vec3<f32>(0.0);
    var cohAccum = vec3<f32>(0.0);   // average neighbor position
    var aliAccum = vec3<f32>(0.0);   // average neighbor velocity
    var neighborCount = 0u;

    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighborCoord = myCoord + vec3<i32>(dx, dy, dz);
                if (!neighbourCellInside(neighborCoord, grid)) {
                    continue;
                }

                let neighborCell = neighbourCellIndex(neighborCoord, grid);
                let start = cellOffsets[neighborCell];
                let count = cellCounts[neighborCell];

                let maxPerCell = min(count, params.maxPerCell);
                for (var i = 0u; i < maxPerCell; i++) {
                    let otherPIdx = sortedIndices[start + i];
                    if (otherPIdx == particleIdx) { continue; }

                    // Skip particles from the same word
                    let myWordId = wordMeta[particleIdx].x;
                    let otherWordId = wordMeta[otherPIdx].x;
                    if (otherWordId == myWordId) { continue; }

                    let otherPos = positions[otherPIdx].xyz;
                    let otherVel = velocities[otherPIdx].xyz;
                    let diff = pos - otherPos;
                    let dist = length(diff);

                    if (dist > 0.0 && dist < params.separationRadius) {
                        sepAccum += normalize(diff) / dist;
                        cohAccum += otherPos;
                        aliAccum += otherVel;
                        neighborCount += 1u;
                    }
                }
            }
        }
    }

    // Separation — steer AWAY from nearby boids
    if (neighborCount > 0u) {
        var sep = sepAccum / f32(neighborCount);
        if (length(sep) > 0.0) {
            sep = normalize(sep) * params.maxSpeed - vel;
            if (length(sep) > params.maxForce) { sep = normalize(sep) * params.maxForce; }
        }
        force += sep * params.separationStrength;
    }

    // Cohesion — steer TOWARD average position of neighbors
    if (neighborCount > 0u) {
        let center = cohAccum / f32(neighborCount);
        var coh = center - pos;
        if (length(coh) > 0.0) {
            coh = normalize(coh) * params.maxSpeed - vel;
            if (length(coh) > params.maxForce) { coh = normalize(coh) * params.maxForce; }
        }
        force += coh * params.cohesionStrength;
    }

    // Alignment — steer toward average HEADING of neighbors
    if (neighborCount > 0u) {
        var ali = aliAccum / f32(neighborCount);
        if (length(ali) > 0.0) {
            ali = normalize(ali) * params.maxSpeed - vel;
            if (length(ali) > params.maxForce) { ali = normalize(ali) * params.maxForce; }
        }
        force += ali * params.alignmentStrength;
    }

    // ── Attractor: steer toward a point in space ──
    // Strength scales with distance: particles far away feel the full pull,
    // particles inside the attractor radius feel almost none (they've "arrived").
    // smoothstep ramps from 0 at dist=0 to 1 at dist=separationRadius*2.
    if (params.attractorStrength > 0.001) {
        let attractorPos = vec3<f32>(params.attractorX, params.attractorY, params.attractorZ);
        let toAttractor = attractorPos - pos;
        let dist = length(toAttractor);
        if (dist > 0.001) {
            let falloff = smoothstep(0.0, params.separationRadius * 2.0, dist);
            var att = normalize(toAttractor) * params.maxSpeed - vel;
            if (length(att) > params.maxForce) { att = normalize(att) * params.maxForce; }
            force += att * params.attractorStrength * falloff;
        }
    }

    // ── Wander: sin/cos-based 3D random walk ──
    let seed = pcgHash(vid * 17u + bitcast<u32>(params.time * 100.0));
    let phase = params.time * params.wanderSpeed;
    let angle1 = phase + hashToFloat(seed) * 6.283185;
    let angle2 = phase * 0.7 + hashToFloat(pcgHash(seed)) * 6.283185;
    let wanderDir = vec3<f32>(
        cos(angle1) * cos(angle2),
        sin(angle1),
        cos(angle1) * sin(angle2),
    );
    force += wanderDir * params.wanderStrength;

    // ── Mouse force: push along the cursor's motion, strongest on letters near the ray under it ──
    if (params.mouseStrength > 0.01) {
        let rayOrigin = vec3<f32>(params.mouseRayOriginX, params.mouseRayOriginY, params.mouseRayOriginZ);
        let rayDir = vec3<f32>(params.mouseRayDirX, params.mouseRayDirY, params.mouseRayDirZ);
        let along = max(dot(pos - rayOrigin, rayDir), 0.0);
        let mouseDist = length(pos - (rayOrigin + rayDir * along));
        let mouseInfluence = exp(-mouseDist * 0.1) * params.mouseStrength;
        let mouseDir = vec3<f32>(params.mouseDirX, params.mouseDirY, params.mouseDirZ);
        force += mouseDir * mouseInfluence * params.mouseForce;
    }

    // ── Bounds: soft wall force near +/- boundsSize ──
    let bs = params.boundsSize;
    let margin = bs * 0.1;
    let boundStrength = params.maxForce * 2.0;

    if (pos.x > bs - margin) { force.x -= boundStrength * ((pos.x - (bs - margin)) / margin); }
    if (pos.x < -bs + margin) { force.x += boundStrength * ((-bs + margin - pos.x) / margin); }
    if (pos.y > bs - margin) { force.y -= boundStrength * ((pos.y - (bs - margin)) / margin); }
    if (pos.y < -bs + margin) { force.y += boundStrength * ((-bs + margin - pos.y) / margin); }
    if (pos.z > bs - margin) { force.z -= boundStrength * ((pos.z - (bs - margin)) / margin); }
    if (pos.z < -bs + margin) { force.z += boundStrength * ((-bs + margin - pos.z) / margin); }

    // ── Clamp force to maxForce ──
    if (length(force) > params.maxForce) {
        force = normalize(force) * params.maxForce;
    }

    // ── Apply force to velocity, clamp to maxSpeed ──
    var newVel = vel + force * params.dt;
    if (length(newVel) > params.maxSpeed) {
        newVel = normalize(newVel) * params.maxSpeed;
    }

    velocities[particleIdx] = vec4<f32>(newVel, 0.0);
}
