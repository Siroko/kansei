import { simParamsStruct } from './sim-params.wgsl';
import { bodySdfHelpers } from './body-sdf.wgsl';

export const shaderCode = /* wgsl */`
${simParamsStruct}
${bodySdfHelpers}

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> bodyStates: array<BodyState>;
@group(0) @binding(3) var<storage, read_write> bodyPrimitives: array<BodyPrimitive>;
@group(0) @binding(4) var<storage, read_write> bodyForces: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: SimParams;
@group(0) @binding(6) var<uniform> bodyCount: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }

    var pos = positions[idx];
    var vel = velocities[idx];
    let particlePos = pos.xy;

    for (var b = 0u; b < bodyCount; b++) {
        let body = bodyStates[b];
        let sdfVal = evaluateBodySDF(particlePos, body, &bodyPrimitives);

        if (sdfVal < 0.0) {
            let gradient = sdfGradient(particlePos, body, &bodyPrimitives);

            let pushDist = min(-sdfVal, body.maxPushDist);

            // Push particle out
            let newPos = particlePos + gradient * pushDist;
            pos.x = newPos.x;
            pos.y = newPos.y;

            // Body velocity at contact point (linear + angular)
            let r = newPos - body.pos.xy;
            let bodyVelAt = body.vel.xy + body.angVel * vec2<f32>(-r.y, r.x);

            // Relative velocity collision — particle picks up body's motion
            let relVel = vel.xy - bodyVelAt;
            let dotVN = dot(relVel, gradient);
            if (dotVN < 0.0) {
                // Only reflect if approaching the surface
                let impulse = -(1.0 + body.restitution) * dotVN;
                vel.x += impulse * gradient.x;
                vel.y += impulse * gradient.y;
            }

            // Accumulate reaction force on body (Newton's 3rd law)
            let reactionForce = -gradient * pushDist * body.reactionMultiplier;
            let torque = cross2D(r, reactionForce);

            atomicAddF32(&bodyForces[b * 4u + 0u], reactionForce.x);
            atomicAddF32(&bodyForces[b * 4u + 1u], reactionForce.y);
            atomicAddF32(&bodyForces[b * 4u + 2u], torque);
        }
    }

    positions[idx] = pos;
    velocities[idx] = vel;
}
`;
