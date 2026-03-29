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
@group(0) @binding(5) var<uniform> viewMatrix: mat4x4<f32>;
@group(0) @binding(6) var<uniform> projectionMatrix: mat4x4<f32>;
@group(0) @binding(7) var<uniform> inverseViewMatrix: mat4x4<f32>;
@group(0) @binding(8) var<uniform> worldMatrix: mat4x4<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= bodyCount) { return; }

    var body = bodyStates[b];

    // Clear force accumulators (must happen even for static bodies)
    let forceX = bitcast<f32>(atomicLoad(&bodyForces[b * 4u + 0u]));
    let forceY = bitcast<f32>(atomicLoad(&bodyForces[b * 4u + 1u]));
    let torque = bitcast<f32>(atomicLoad(&bodyForces[b * 4u + 2u]));
    atomicStore(&bodyForces[b * 4u + 0u], 0u);
    atomicStore(&bodyForces[b * 4u + 1u], 0u);
    atomicStore(&bodyForces[b * 4u + 2u], 0u);

    // Static body: mass ≈ 0 → skip integration, stay in place
    if (body.mass < 0.001) {
        bodyTransforms[b] = vec4<f32>(body.pos.x, body.pos.y, body.pos.z, body.angle);
        return;
    }

    var fluidForce = vec2<f32>(forceX, forceY);

    // Clamp fluid reaction to prevent explosive spikes
    let fluidMag = length(fluidForce);
    let gravVec = vec2<f32>(params.gravity.x, params.gravity.y);
    let gravMag = length(gravVec);
    let maxFluidForce = body.mass * gravMag * body.forceClampFactor;
    if (fluidMag > maxFluidForce) {
        fluidForce *= maxFluidForce / fluidMag;
    }

    // Buoyancy from density (Archimedes): only when in contact with fluid
    // density < 1 → buoyancy exceeds weight → floats
    // density = 1 → neutral
    // density > 1 → sinks in fluid
    let submersion = clamp(fluidMag / (body.mass * gravMag + 0.001), 0.0, 1.0);
    let buoyancy = -normalize(gravVec) * gravMag * body.mass * (1.0 / max(body.density, 0.01) - 1.0) * submersion;

    // Total force = fluid reaction + buoyancy + gravity
    var totalForce = fluidForce + buoyancy;
    totalForce += gravVec * body.mass;

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
    let clampedTorque = clamp(torque, -maxFluidForce, maxFluidForce);
    // Righting torque — acts like a weighted keel pulling the boat upright
    let rightingTorque = -sin(body.angle) * body.mass * body.rightingStrength;
    body.angVel += ((clampedTorque + rightingTorque) / safeInertia) * params.dt;
    body.angle += body.angVel * params.dt;

    // Mouse interaction — applied directly to velocity (matching particle behavior)
    if (params.mouseStrength > 0.001) {
        let projected = projectionMatrix * viewMatrix * worldMatrix * vec4<f32>(body.pos, 1.0);
        let ndc = projected.xy / projected.w;
        var ndcMouse = params.mousePos;
        ndcMouse.y *= -1.0;
        let distToMouse = distance(ndcMouse, ndc);
        let mouseR = params.mouseRadius * 4.0;
        if (distToMouse < mouseR) {
            let nDist = distToMouse / mouseR;
            let displaceNDC = vec2<f32>(
                params.mouseDir.x * params.mouseForce * (1.0 - nDist) * -1.0,
                params.mouseDir.y * params.mouseForce * (1.0 - nDist)
            );
            let worldDisplace = (inverseViewMatrix * vec4<f32>(displaceNDC, 0.0, 0.0)).xy;
            body.vel.x += worldDisplace.x * params.mouseStrength * body.mouseScale;
            body.vel.y += worldDisplace.y * params.mouseStrength * body.mouseScale;
        }
    }

    // Damping
    body.vel = vec3<f32>(body.vel.x * body.linearDamping, body.vel.y * body.linearDamping, body.vel.z);
    body.angVel *= body.angularDamping;

    // Boundary collision — inset by body extent so edges don't clip
    let pad = 5.0;
    let bMin = params.worldBoundsMin + vec3<f32>(pad, pad, 0.0);
    let bMax = params.worldBoundsMax - vec3<f32>(pad, pad, 0.0);
    if (body.pos.x < bMin.x) { body.pos.x = bMin.x; body.vel.x *= -0.3; }
    if (body.pos.x > bMax.x) { body.pos.x = bMax.x; body.vel.x *= -0.3; }
    if (body.pos.y < bMin.y) { body.pos.y = bMin.y; body.vel.y *= -0.3; }
    if (body.pos.y > bMax.y) { body.pos.y = bMax.y; body.vel.y *= -0.3; }

    // Write back
    bodyStates[b] = body;
    bodyTransforms[b] = vec4<f32>(body.pos.x, body.pos.y, body.pos.z, body.angle);
}
`;
