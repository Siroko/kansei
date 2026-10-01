// Aerial-perspective volume (Hillaire 2020 section 5.4): a camera-frustum grid whose xy follow the
// screen and whose slices lie at view distances apDistance * ((k + 1) / depth)^2 (dense near the
// camera). Each slice holds the light scattered toward the camera between it and the camera, and
// the (chromatic) transmittance over that distance. One thread per froxel column, marching front
// to back and storing every slice. Rebuilt every frame.

@group(0) @binding(0) var<uniform> atm : Atmosphere;
@group(0) @binding(1) var<uniform> frame : SkyFrame;
@group(0) @binding(2) var transmittanceLut : texture_2d<f32>;
@group(0) @binding(3) var multiScatteringLut : texture_2d<f32>;
@group(0) @binding(4) var lutSampler : sampler;
@group(0) @binding(5) var apScatteringOut : texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var apTransmittanceOut : texture_storage_3d<rgba16float, write>;

const SUBSTEPS : u32 = 2u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(apScatteringOut);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(size.xy);
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let n = frame.invViewProj * vec4f(ndc, 0.0, 1.0);
    let f = frame.invViewProj * vec4f(ndc, 1.0, 1.0);
    let rd = normalize(f.xyz / f.w - n.xyz / n.w);
    let ro = frame.cameraPos;

    var lum = vec3f(0.0);
    var trans = vec3f(1.0);
    var tPrev = 0.0;
    let slices = f32(size.z);
    for (var k = 0u; k < size.z; k++) {
        let w = (f32(k) + 1.0) / slices;
        let tNext = frame.apDistance * w * w;
        let dt = (tNext - tPrev) / f32(SUBSTEPS);
        for (var s = 0u; s < SUBSTEPS; s++) {
            let p = ro + rd * (tPrev + (f32(s) + 0.5) * dt);
            let r = length(p);
            let med = sampleMedium(r - atm.bottomRadius);
            let sc = scatteringAt(p, r, med, rd);
            let segT = exp(-med.extinction * dt);
            lum += trans * (sc - sc * segT) / max(med.extinction, vec3f(1e-9));
            trans *= segT;
        }
        tPrev = tNext;
        textureStore(apScatteringOut, vec3u(gid.xy, k), vec4f(lum, 1.0));
        textureStore(apTransmittanceOut, vec3u(gid.xy, k), vec4f(trans, 1.0));
    }
}
