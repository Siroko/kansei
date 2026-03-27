import { simParamsStruct } from './sim-params.wgsl';
import { bodySdfHelpers } from './body-sdf.wgsl';

export const shaderCode = /* wgsl */`
${simParamsStruct}
${bodySdfHelpers}

@group(0) @binding(0) var<storage, read_write> bodyStates: array<BodyState>;
@group(0) @binding(1) var<storage, read_write> bodyForces: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> bodyTransforms: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: SimParams;
@group(0) @binding(4) var<uniform> bodyCount: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= bodyCount) { return; }

    var body = bodyStates[b];

    // Read accumulated forces
    let forceX = bitcast<f32>(atomicLoad(&bodyForces[b * 4u + 0u]));
    let forceY = bitcast<f32>(atomicLoad(&bodyForces[b * 4u + 1u]));
    let torque = bitcast<f32>(atomicLoad(&bodyForces[b * 4u + 2u]));

    // Clear accumulators
    atomicStore(&bodyForces[b * 4u + 0u], 0u);
    atomicStore(&bodyForces[b * 4u + 1u], 0u);
    atomicStore(&bodyForces[b * 4u + 2u], 0u);

    // Total force = fluid reaction + gravity
    var totalForce = vec2<f32>(forceX, forceY);
    totalForce += vec2<f32>(params.gravity.x, params.gravity.y) * body.mass;

    // Linear integration
    body.vel = vec3<f32>(
        body.vel.x + (totalForce.x / body.mass) * params.dt,
        body.vel.y + (totalForce.y / body.mass) * params.dt,
        body.vel.z
    );
    body.pos = vec3<f32>(
        body.pos.x + body.vel.x * params.dt,
        body.pos.y + body.vel.y * params.dt,
        body.pos.z
    );

    // Angular integration
    let safeInertia = max(body.inertia, 0.001);
    body.angVel += (torque / safeInertia) * params.dt;
    body.angle += body.angVel * params.dt;

    // Damping
    body.vel = vec3<f32>(body.vel.x * 0.999, body.vel.y * 0.999, body.vel.z);
    body.angVel *= 0.998;

    // Boundary collision
    let bMin = params.worldBoundsMin;
    let bMax = params.worldBoundsMax;
    if (body.pos.x < bMin.x) { body.pos.x = bMin.x; body.vel.x *= -0.5; }
    if (body.pos.x > bMax.x) { body.pos.x = bMax.x; body.vel.x *= -0.5; }
    if (body.pos.y < bMin.y) { body.pos.y = bMin.y; body.vel.y *= -0.5; }
    if (body.pos.y > bMax.y) { body.pos.y = bMax.y; body.vel.y *= -0.5; }

    // Write back
    bodyStates[b] = body;
    bodyTransforms[b] = vec4<f32>(body.pos.x, body.pos.y, body.pos.z, body.angle);
}
`;
