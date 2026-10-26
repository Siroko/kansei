// Distance fog: each surface fades toward a colour by its distance from the camera (or from a
// point, `usePoint`), from nothing at `start` to all of it at `end` (a smoothstep, so neither end
// shows a ring). The colour is a
// gradient sky by direction (Material::gradient_sky's: zenith, horizon, ground and its curve; one
// colour when they are equal). With a dome (`radius` > 0), the colour is what that dome shows
// behind the pixel and the fade ends where the ray meets the dome (that point's distance), so a
// floor running into the dome meets it with no edge. The sky (depth 1) is left as it is.

struct DistanceFogParams {
    invViewProj : mat4x4f,
    cameraPos   : vec3f,
    start       : f32,
    zenith      : vec4f,   // w: the curve's exponent
    horizon     : vec4f,
    ground      : vec4f,
    centre      : vec3f,   // the dome's centre
    radius      : f32,     // the dome's radius; 0: no dome
    end         : f32,
    maxOpacity  : f32,
    usePoint    : f32,     // 1: distances from `point`, not the camera
    _pad        : f32,
    point       : vec4f,
}

@group(0) @binding(0) var inputTex  : texture_2d<f32>;
@group(0) @binding(1) var depthTex  : texture_depth_2d;
@group(0) @binding(2) var outputTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> fog : DistanceFogParams;

// as gradient_sky.wgsl's fragment_main
fn gradientSky(d: vec3f) -> vec3f {
    let up = pow(saturate(d.y), fog.zenith.w);
    let down = sqrt(saturate(-d.y));
    return select(mix(fog.horizon.rgb, fog.ground.rgb, down), mix(fog.horizon.rgb, fog.zenith.rgb, up), d.y >= 0.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(outputTex);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let color = textureLoad(inputTex, gid.xy, 0);
    let depth = textureLoad(depthTex, gid.xy, 0);
    if (depth >= 1.0) {
        textureStore(outputTex, gid.xy, color);
        return;
    }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(size);
    let p = fog.invViewProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = p.xyz / p.w;
    let offset = world - fog.cameraPos;
    let dir = offset / max(length(offset), 1e-6);
    // the distance the fade goes by: from the camera, or from the point
    let origin = select(fog.cameraPos, fog.point.xyz, fog.usePoint > 0.5);
    let dist = distance(world, origin);

    var end = fog.end;
    var skyDir = dir;
    if (fog.radius > 0.0) {
        // where the ray leaves the dome
        let oc = fog.cameraPos - fog.centre;
        let b = dot(oc, dir);
        let h = b * b - (dot(oc, oc) - fog.radius * fog.radius);
        if (h >= 0.0) {
            let tDome = -b + sqrt(h);
            if (tDome > 0.0) {
                let hit = fog.cameraPos + dir * tDome;
                end = min(end, distance(hit, origin));
                skyDir = normalize(hit - fog.centre);
            }
        }
    }
    let start = min(fog.start, end);
    let amount = select(step(end, dist), smoothstep(start, end, dist), end > start) * fog.maxOpacity;
    textureStore(outputTex, gid.xy, vec4f(mix(color.rgb, gradientSky(skyDir), amount), color.a));
}
