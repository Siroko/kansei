export const shaderCode = /* wgsl */`
// Screen-space refraction composite.
// Reads:
//   - input         : scene colour with the transmissive mesh already drawn on top
//   - background    : scene colour from BEFORE the transmissive mesh was drawn
//   - normalTex     : GBuffer world-space normal (acts as a mask + refraction input)
// Writes:
//   - output        : composited result (refracted + fresnel + rim + chromatic aberration)

struct Params {
    viewMatrix          : mat4x4<f32>, // 64 bytes — transform world normal to view space
    color               : vec4<f32>,
    ior                 : f32,
    chromaticAberration : f32,
    tintStrength        : f32,
    fresnelPower        : f32,
    roughness           : f32,
    thickness           : f32,
    screenWidth         : f32,
    screenHeight        : f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var sceneColor   : texture_2d<f32>;
@group(0) @binding(2) var backgroundTex: texture_2d<f32>;
@group(0) @binding(3) var normalTex    : texture_2d<f32>;
@group(0) @binding(4) var outputTex    : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let w = u32(params.screenWidth);
    let h = u32(params.screenHeight);
    if (gid.x >= w || gid.y >= h) { return; }

    let scene = textureLoad(sceneColor, coord, 0);
    let normalSample = textureLoad(normalTex, coord, 0);
    // normal.w is a "transmissive mask": the fluid surface shader writes 1.0 here,
    // opaque materials write 0.0. Untouched pixels are 0 from the clear.
    if (normalSample.w < 0.5) {
        textureStore(outputTex, coord, scene);
        return;
    }

    let nWorld = normalize(normalSample.xyz);
    // Upper 3x3 of the view matrix transforms a world-space direction into view space.
    let view3 = mat3x3<f32>(
        params.viewMatrix[0].xyz,
        params.viewMatrix[1].xyz,
        params.viewMatrix[2].xyz,
    );
    // Flip back-facing normals so they always face the camera (mesh is double-sided).
    var nView = normalize(view3 * nWorld);
    if (nView.z < 0.0) { nView = -nView; }

    let screenUv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / vec2<f32>(f32(w), f32(h));
    let dimsF = vec2<f32>(f32(w), f32(h));

    // Refraction offset — view-space normal's xy is screen-aligned.
    let refractStrength = params.thickness * (1.0 - 1.0 / params.ior);
    let offset = nView.xy * refractStrength * 0.05;

    // Chromatic aberration — sample RGB at slightly different offsets.
    let ca = params.chromaticAberration;
    let uvR = clamp(screenUv + offset * (1.0 + ca), vec2<f32>(0.0), vec2<f32>(1.0));
    let uvG = clamp(screenUv + offset,                vec2<f32>(0.0), vec2<f32>(1.0));
    let uvB = clamp(screenUv + offset * (1.0 - ca), vec2<f32>(0.0), vec2<f32>(1.0));

    let bgR = textureLoad(backgroundTex, vec2<i32>(dimsF * uvR), 0).r;
    let bgG = textureLoad(backgroundTex, vec2<i32>(dimsF * uvG), 0).g;
    let bgB = textureLoad(backgroundTex, vec2<i32>(dimsF * uvB), 0).b;
    var refracted = vec3<f32>(bgR, bgG, bgB);

    // Tint refracted light by fluid colour (absorption).
    refracted *= mix(vec3<f32>(1.0), params.color.rgb, params.tintStrength);

    // Fresnel (view-space: ndotv = nView.z since the view dir is (0,0,1) in view space).
    let f0 = pow((1.0 - params.ior) / (1.0 + params.ior), 2.0);
    let ndotv = nView.z;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - ndotv, params.fresnelPower);

    // Screen-space reflection by offsetting in the reflected direction.
    let reflectDir = reflect(vec3<f32>(0.0, 0.0, -1.0), nView);
    let reflectOffset = reflectDir.xy * 0.3;
    let reflectUv = clamp(screenUv + reflectOffset, vec2<f32>(0.0), vec2<f32>(1.0));
    let reflected = textureLoad(backgroundTex, vec2<i32>(dimsF * reflectUv), 0).rgb;

    // Rim light — bright edge glow.
    let rim = pow(1.0 - ndotv, 3.0) * 0.15;

    // Fresnel mix between refracted (transmitted) and reflected (environment).
    let result = mix(refracted, reflected, fresnel) + vec3<f32>(rim);

    textureStore(outputTex, coord, vec4<f32>(result, 1.0));
}
`;
