// AtmosphereEffect composite: the sky, the sun and the moon behind the scene, where the depth
// buffer is still at the far plane. The rest of the image passes through.

@group(0) @binding(0) var<uniform> atm : Atmosphere;
@group(0) @binding(1) var<uniform> frame : SkyFrame;
@group(0) @binding(2) var transmittanceLut : texture_2d<f32>;
@group(0) @binding(3) var skyViewLut : texture_2d<f32>;
@group(0) @binding(4) var lutSampler : sampler;
@group(0) @binding(5) var skyViewSampler : sampler;
@group(0) @binding(6) var inputTex : texture_2d<f32>;
@group(0) @binding(7) var depthTex : texture_depth_2d;
@group(0) @binding(8) var outputTex : texture_storage_2d<rgba16float, write>;

const SUN_LIMB_DARKENING : f32 = 0.6;
const MAX_HALF : f32 = 65000.0;

// World-space direction of the ray through a pixel centre.
fn viewRay(pixel: vec2u, size: vec2u) -> vec3f {
    let uv = (vec2f(pixel) + 0.5) / vec2f(size);
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let n = frame.invViewProj * vec4f(ndc, 0.0, 1.0);
    let f = frame.invViewProj * vec4f(ndc, 1.0, 1.0);
    return normalize(f.xyz / f.w - n.xyz / n.w);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(outputTex);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let color = textureLoad(inputTex, gid.xy, 0);
    let depth = textureLoad(depthTex, gid.xy, 0);
    if (depth < 1.0) {
        textureStore(outputTex, gid.xy, color);
        return;
    }

    let rd = viewRay(gid.xy, size);
    let ro = frame.cameraPos;
    let r = length(ro);
    var lum = skyViewLuminance(rd);
    let tGround = rayGround(ro, rd);
    if (tGround < 0.0) {
        let toSpace = transmittanceToTop(r, dot(ro, rd) / r);
        lum += toSpace * diskLuminance(rd, frame.sunDirection, frame.sunAngularRadius, frame.sunIlluminance,
                                       frame.sunDiskLuminance, SUN_LIMB_DARKENING);
        lum += toSpace * diskLuminance(rd, frame.moonDirection, frame.moonAngularRadius, frame.moonIlluminance,
                                       frame.moonDiskLuminance, 0.0);
    } else {
        // the planet's surface where no scene geometry covers it: diffuse, lit by the sun and the
        // moon through the atmosphere, and seen through it
        let n = normalize(ro + rd * tGround);
        let muSun = dot(n, frame.sunDirection);
        let muMoon = dot(n, frame.moonDirection);
        let e = frame.sunIlluminance * transmittanceToTop(atm.bottomRadius, muSun) * saturate(muSun)
              + frame.moonIlluminance * transmittanceToTop(atm.bottomRadius, muMoon) * saturate(muMoon);
        lum += transmittanceBetween(ro, rd, tGround) * atm.groundAlbedo / PI * e;
    }
    textureStore(outputTex, gid.xy, vec4f(min(lum, vec3f(MAX_HALF)), color.a));
}
