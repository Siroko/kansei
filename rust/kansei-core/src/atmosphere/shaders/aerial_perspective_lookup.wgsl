// Aerial-perspective lookup. The including shader binds `frame`, `apScattering`,
// `apTransmittance` (3D, filterable) and a clamping linear `lutSampler`.

struct AerialPerspective {
    scattering    : vec3f,   // light scattered toward the camera in front of the surface
    transmittance : vec3f,   // of the air between the camera and the surface
}

// The atmosphere between the camera and a world-space point (metres) seen at screen uv:
// composite as color * transmittance + scattering.
fn aerialPerspective(uv: vec2f, worldPos: vec3f) -> AerialPerspective {
    let slices = f32(textureDimensions(apScattering, 0).z);
    let tKm = max(length(worldPos - frame.cameraWorld) * 0.001 - frame.apStartDepth, 0.0) * frame.apDistanceScale;
    let w = sqrt(saturate(tKm / frame.apDistance));
    // slice k holds distance ((k + 1) / slices)^2; in front of slice 0, fade from nothing
    let z = w - 0.5 / slices;
    let fade = saturate(w * slices);
    let coord = vec3f(uv, max(z, 0.5 / slices));
    var ap : AerialPerspective;
    ap.scattering = textureSampleLevel(apScattering, lutSampler, coord, 0.0).rgb * fade;
    ap.transmittance = mix(vec3f(1.0), textureSampleLevel(apTransmittance, lutSampler, coord, 0.0).rgb, fade);
    return ap;
}
