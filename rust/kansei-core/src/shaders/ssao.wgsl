// Screen-space ambient occlusion from depth alone (a port of the TS engine's SSAOEffect): view
// positions and normals are rebuilt from the depth buffer, a hemisphere of samples around each
// point is projected back to the screen and compared with the depth there, and the colour is
// multiplied by what is left unoccluded.

struct SSAOParams {
    projMatrix    : mat4x4f,
    invProjMatrix : mat4x4f,
    screenWidth   : f32,
    screenHeight  : f32,
    radius        : f32,
    bias          : f32,
    kernelSize    : u32,
    strength      : f32,
    _pad0         : f32,
    _pad1         : f32,
}

@group(0) @binding(0) var depthTex  : texture_depth_2d;
@group(0) @binding(1) var colorTex  : texture_2d<f32>;
@group(0) @binding(2) var outputTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params : SSAOParams;
@group(0) @binding(4) var<uniform> kernel : array<vec4f, 32>;

// The depth under screen pixel `px`; the depth buffer is at the render size, below the screen's
// after a temporal upscaler.
fn loadDepth(px: vec2u) -> f32 {
    let dims = textureDimensions(depthTex);
    let q = vec2u((vec2f(px) + 0.5) * vec2f(dims) / vec2f(params.screenWidth, params.screenHeight));
    return textureLoad(depthTex, min(q, dims - 1u), 0);
}

// A view-space position from a UV coordinate and a raw depth value.
fn viewPosFromDepth(uv: vec2f, depth: f32) -> vec3f {
    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let viewPos = params.invProjMatrix * ndc;
    return viewPos.xyz / viewPos.w;
}

fn viewPosAt(coord: vec2u) -> vec3f {
    return viewPosFromDepth(vec2f(f32(coord.x) / params.screenWidth, f32(coord.y) / params.screenHeight), loadDepth(coord));
}

// The view-space normal at a pixel, by finite differences on the depth buffer.
fn viewNormalFromDepth(coord: vec2u) -> vec3f {
    let w = u32(params.screenWidth);
    let h = u32(params.screenHeight);
    let pR = viewPosAt(vec2u(min(coord.x + 1u, w - 1u), coord.y));
    let pL = viewPosAt(vec2u(select(0u, coord.x - 1u, coord.x > 0u), coord.y));
    let pU = viewPosAt(vec2u(coord.x, select(0u, coord.y - 1u, coord.y > 0u)));
    let pD = viewPosAt(vec2u(coord.x, min(coord.y + 1u, h - 1u)));
    return normalize(cross(pR - pL, pD - pU));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let coord = gid.xy;
    let w = u32(params.screenWidth);
    let h = u32(params.screenHeight);
    if (coord.x >= w || coord.y >= h) { return; }

    let uv = vec2f(f32(coord.x) / params.screenWidth, f32(coord.y) / params.screenHeight);
    let depth = loadDepth(coord);

    // the sky: no occlusion, the colour passes through
    if (depth >= 1.0) {
        textureStore(outputTex, coord, textureLoad(colorTex, coord, 0));
        return;
    }

    let viewPos = viewPosFromDepth(uv, depth);
    let viewNormal = viewNormalFromDepth(coord);

    // a tangent basis around the surface normal
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(viewNormal.y) > 0.99);
    let tangent = normalize(up - viewNormal * dot(up, viewNormal));
    let bitangent = cross(viewNormal, tangent);
    let tbn = mat3x3f(tangent, bitangent, viewNormal);

    var occlusion = 0.0;
    for (var i = 0u; i < params.kernelSize; i++) {
        // the hemisphere sample, oriented along the normal
        let sampleViewPos = viewPos + (tbn * kernel[i].xyz) * params.radius;

        // projected to the screen
        let clip = params.projMatrix * vec4f(sampleViewPos, 1.0);
        let ndc = clip.xyz / clip.w;
        let sampleUV = vec2f(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
        if (sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0) {
            continue;
        }

        let sampleCoord = vec2u(u32(sampleUV.x * params.screenWidth), u32(sampleUV.y * params.screenHeight));
        let sampleViewRef = viewPosFromDepth(sampleUV, loadDepth(min(sampleCoord, vec2u(w - 1u, h - 1u))));

        // the range check keeps occlusion from bleeding over large depth discontinuities
        let rangeCheck = smoothstep(0.0, 1.0, params.radius / abs(viewPos.z - sampleViewRef.z));
        occlusion += select(0.0, 1.0, sampleViewRef.z >= sampleViewPos.z + params.bias) * rangeCheck;
    }

    let ao = 1.0 - (occlusion / f32(params.kernelSize)) * params.strength;
    let color = textureLoad(colorTex, coord, 0);
    textureStore(outputTex, coord, vec4f(color.rgb * ao, color.a));
}
