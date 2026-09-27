// Sky-view LUT lookup and the sun and moon disks. The including shader binds `frame`,
// `skyViewLut` and `skyViewSampler` (linear, repeating in u).

fn skyViewLuminance(rd: vec3f) -> vec3f {
    let r = length(frame.cameraPos);
    let lf = localFrame(frame.cameraPos);
    let u = fract(atan2(dot(rd, lf.z), dot(rd, lf.x)) / (2.0 * PI));
    let v = skyViewV(dot(rd, lf.up), r);
    return textureSampleLevel(skyViewLut, skyViewSampler, vec2f(u, v), 0.0).rgb * frame.skyLuminanceFactor;
}

// Luminance of a light's disk along rd (before the atmosphere), with linear limb darkening
// I(mu) = 1 - k (1 - mu); `luminanceScale` is normalised so the disk integrates to the light's
// illuminance.
fn diskLuminance(rd: vec3f, lightDir: vec3f, angularRadius: f32, illuminance: vec3f, luminanceScale: f32,
                 limbDarkening: f32) -> vec3f {
    let cosR = cos(angularRadius);
    let c = dot(rd, lightDir);
    if (c < cosR || luminanceScale <= 0.0) { return vec3f(0.0); }
    let sinR2 = max(1.0 - cosR * cosR, 1e-12);
    let mu = sqrt(saturate(1.0 - (1.0 - c * c) / sinR2));
    return illuminance * luminanceScale * (1.0 - limbDarkening * (1.0 - mu));
}
