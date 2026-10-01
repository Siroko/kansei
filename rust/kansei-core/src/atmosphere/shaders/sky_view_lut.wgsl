// Sky-view LUT: the luminance of the sky around the camera, (world azimuth, zenith angle with
// rows packed toward the horizon). Rebuilt every frame: it depends on the camera's altitude and
// on the sun and the moon. Rays that hit the ground stop there (the ground itself is not in it).

@group(0) @binding(0) var<uniform> atm : Atmosphere;
@group(0) @binding(1) var<uniform> frame : SkyFrame;
@group(0) @binding(2) var transmittanceLut : texture_2d<f32>;
@group(0) @binding(3) var multiScatteringLut : texture_2d<f32>;
@group(0) @binding(4) var lutSampler : sampler;
@group(0) @binding(5) var skyViewOut : texture_storage_2d<rgba16float, write>;

const SAMPLES : u32 = 32u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(skyViewOut);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(size);

    let ro = frame.cameraPos;
    let r = length(ro);
    let zenith = skyViewZenith(uv.y, r);
    let phi = uv.x * 2.0 * PI;
    let lf = localFrame(ro);
    let rd = lf.x * (sin(zenith) * cos(phi)) + lf.up * cos(zenith) + lf.z * (sin(zenith) * sin(phi));

    let tGround = rayGround(ro, rd);
    let tMax = select(rayTop(ro, rd), tGround, tGround > 0.0);
    let result = integrateScattering(ro, rd, tMax, SAMPLES);
    textureStore(skyViewOut, gid.xy, vec4f(result.luminance, 1.0));
}
