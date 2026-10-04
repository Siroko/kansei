// Voxel GI on screen, composite: the accumulated irradiance upsampled and filtered by depth (the
// traced texels whose surface matches this pixel's), then added to the scene as albedo / pi
// times it.
//
// With a near field (screen-space GI traced first, ssgi_trace.wgsl), the screen supplies the
// light of what it saw hide each direction and the voxels the rest: E = E_screen + open * E_voxels,
// `open` the share of the hemisphere the screen found no occluder in (Lumen's split: screen
// traces first, the scene representation for what they miss). The screen's occluders carry only
// their direct light, though (the frame's lit colour before GI), so where they are lit by
// bounces alone it would leave those directions dark: the voxels' light from the same share of
// the hemisphere, (1 - open) * E_voxels, is the least the screen's part gives.
//
// With probes (gi::SdfProbes, `probes`), the far field is their irradiance at each pixel
// (`kansei_gi_irradiance`), in place of the traced cones: the same split, with the probes as the
// scene representation.
//
// The voxels' irradiance holds the sky they see past the volume, so with the sky's lighting
// bound the material's own sky ambient (albedo / pi times the sky's irradiance around its normal)
// is taken out, `ambient` times: it is what the GI replaces. The debug views show the GI alone, or
// the volume's mip 0 as the camera sees it (the voxelized scene and its light), a slice of the
// distance field, or the probes.

@group(0) @binding(0) var<uniform> gp : VoxelGiParams;
@group(0) @binding(1) var colorTex  : texture_2d<f32>;
@group(0) @binding(2) var depthTex  : texture_depth_2d;
@group(0) @binding(3) var giTex     : texture_2d<f32>;
@group(0) @binding(4) var nearTex   : texture_2d<f32>;
@group(0) @binding(5) var albedoTex : texture_2d<f32>;
@group(0) @binding(6) var normalTex : texture_2d<f32>;
@group(0) @binding(7) var<uniform> sky : SkyLighting;
@group(0) @binding(8) var outTex    : texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var<uniform> vol : VoxelVolume;
@group(0) @binding(10) var radiance : texture_3d<f32>;
@group(0) @binding(11) var linearClamp : sampler;
// the scene's distance field (gi::JumpFloodSdf; a 1-texel stand-in without one)
@group(0) @binding(12) var sdfTex : texture_3d<f32>;
// the probes (gi::SdfProbes; 1-element stand-ins without them), for probe_irradiance.wgsl
@group(0) @binding(13) var<uniform> kansei_probe_grid : ProbeGrid;
@group(0) @binding(14) var<storage, read> kansei_probe_sh : array<vec4f>;
@group(0) @binding(15) var<storage, read> kansei_probe_state : array<vec4f>;
@group(0) @binding(16) var<storage, read> kansei_probe_depth : array<vec2f>;

// The probes as small balls on the camera ray through `uv`, each a white diffuse ball lit by its
// own irradiance (E / pi toward each of its normals, so its SH shows), dark red where a probe is
// left out; a = 0 where the ray meets none before `sceneDist`. The ray walks the grid's cells (one
// around each probe; Amanatides and Woo) and tests each cell's probe.
fn probeBalls(uv: vec2f, sceneDist: f32) -> vec4f {
    let grid = kansei_probe_grid;
    let eye = (gp.invView * vec4f(0.0, 0.0, 0.0, 1.0)).xyz;
    let far = (gp.invView * vec4f(gpViewPos(uv, 1.0), 1.0)).xyz;
    let dir = normalize(far - eye);
    let lo = grid.origin - 0.5 * grid.spacing;
    let hi = lo + vec3f(grid.dims) * grid.spacing;
    let inv = 1.0 / select(dir, vec3f(1e-8), abs(dir) < vec3f(1e-8));
    let t0 = (lo - eye) * inv;
    let t1 = (hi - eye) * inv;
    let enter = max(max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z)), 0.0);
    let exit = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    if (enter >= exit) { return vec4f(0.0); }
    let start = eye + dir * (enter + 1e-4);
    var cell = clamp(vec3i(floor((start - lo) / grid.spacing)), vec3i(0), vec3i(grid.dims) - 1);
    let stepDir = vec3i(select(vec3f(-1.0), vec3f(1.0), dir >= vec3f(0.0)));
    let delta = abs(grid.spacing * inv);
    var next = (lo + (vec3f(cell) + select(vec3f(0.0), vec3f(1.0), dir >= vec3f(0.0))) * grid.spacing - eye) * inv;
    let radius = 0.12 * grid.spacing;
    var best = min(sceneDist, exit);
    var color = vec4f(0.0);
    let cells = grid.dims.x + grid.dims.y + grid.dims.z;
    for (var i = 0u; i < cells; i++) {
        let slot = probeSlot(grid, grid.base + cell);
        let state = kansei_probe_state[2u * slot];
        let center = grid.origin + vec3f(cell) * grid.spacing + state.xyz;
        let oc = eye - center;
        let b = dot(oc, dir);
        let h = b * b - (dot(oc, oc) - radius * radius);
        if (h >= 0.0) {
            let t = -b - sqrt(h);
            if (t > 0.0 && t < best) {
                best = t;
                let n = normalize(eye + dir * t - center);
                let lit = kanseiProbeShIrradiance(slot, n) / 3.14159265;
                color = vec4f(select(lit, vec3f(8.0, 0.4, 0.3), state.w > grid.backfaceLimit), 1.0);
            }
        }
        // past this cell's far side, the later cells' balls lie farther than the one found
        let leave = min(min(next.x, next.y), next.z);
        if (color.a > 0.0 && best <= leave) { break; }
        if (next.x <= next.y && next.x <= next.z) {
            cell.x += stepDir.x;
            next.x += delta.x;
        } else if (next.y <= next.z) {
            cell.y += stepDir.y;
            next.y += delta.y;
        } else {
            cell.z += stepDir.z;
            next.z += delta.z;
        }
        if (any(cell < vec3i(0)) || any(cell >= vec3i(grid.dims))) { break; }
    }
    return color;
}

// The distance field on the horizontal plane at the debug slice's height, where the camera ray
// through `uv` meets it: bands every 10 cm, darker toward the surfaces, red inside them, as scene
// radiance of about 40 (bright under an EV100 of 5); None (a = 0) off the plane or the volume.
fn sdfSliceColor(uv: vec2f) -> vec4f {
    let eye = (gp.invView * vec4f(0.0, 0.0, 0.0, 1.0)).xyz;
    let far = (gp.invView * vec4f(gpViewPos(uv, 1.0), 1.0)).xyz;
    let dir = normalize(far - eye);
    if (abs(dir.y) < 1e-5) { return vec4f(0.0); }
    let t = (gp.sdfSlice - eye.y) / dir.y;
    if (t <= 0.0) { return vec4f(0.0); }
    let p = eye + dir * t;
    let uvw = voxelUvw(vol, p);
    if (any(uvw < vec3f(0.0)) || any(uvw > vec3f(1.0))) { return vec4f(0.0); }
    let d = textureSampleLevel(sdfTex, linearClamp, uvw, 0.0).r;
    if (d < 0.5 * vol.voxelSize) { return vec4f(0.8, 0.12, 0.08, 1.0); }
    let band = fract(d / 0.1);
    let line = select(1.0, 0.35, band < 0.08);
    let shade = 1.0 - exp(-d / 0.6);
    return vec4f(mix(vec3f(0.05, 0.12, 0.35), vec3f(0.95, 0.95, 0.85), shade) * line, 1.0);
}

// The voxels' light along the camera ray through `uv`: mip 0 marched front to back, half a voxel
// a step (from a start jittered per pixel, so the steps don't band), from where the ray enters
// the volume.
fn marchVoxels(uv: vec2f) -> vec3f {
    let eye = (gp.invView * vec4f(0.0, 0.0, 0.0, 1.0)).xyz;
    let far = (gp.invView * vec4f(gpViewPos(uv, 1.0), 1.0)).xyz;
    let dir = normalize(far - eye);
    let lo = vol.origin;
    let hi = vol.origin + vec3f(vol.dims) * vol.voxelSize;
    let t0 = (lo - eye) / dir;
    let t1 = (hi - eye) / dir;
    let enter = max(max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z)), 0.0);
    let exit = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    var color = vec3f(0.0);
    var transmittance = 1.0;
    let jitter = fract(52.9829189 * fract(dot(uv * gp.fullSize, vec2f(0.06711056, 0.00583715))));
    var t = enter + jitter * 0.5 * vol.voxelSize;
    for (var i = 0u; i < 1024u; i++) {
        if (t >= exit || transmittance < 0.01) { break; }
        let s = textureSampleLevel(radiance, linearClamp, voxelUvw(vol, eye + dir * t), 0.0);
        let a = 1.0 - sqrt(max(1.0 - s.a, 0.0));
        color += transmittance * s.rgb * select(0.5, a / s.a, s.a > 1e-4);
        transmittance *= 1.0 - a;
        t += 0.5 * vol.voxelSize;
    }
    return color * vol.radianceScale;
}

// `tex` (traced at `size`) at `uv`, filtered over the 4x4 traced texels around it with a tent
// two texels wide (a wider bilinear: the irradiance is smooth, the cones' per-pixel rotation is
// not), among the texels whose surface lies near this pixel's view depth `z`.
fn upsample(tex: texture_2d<f32>, size: vec2f, uv: vec2f, z: f32) -> vec4f {
    let pos = uv * size - 0.5;
    let base = floor(pos) - 1.0;
    var sum = vec4f(0.0);
    var weight = 0.0;
    for (var i = 0u; i < 16u; i++) {
        let o = vec2f(f32(i & 3u), f32(i >> 2u));
        let t = clamp(vec2i(base + o), vec2i(0), vec2i(size) - 1);
        let tuv = (vec2f(t) + 0.5) / size;
        let tz = -gpViewPos(tuv, gpDepth(gpPixel(tuv))).z;
        let d = abs(base + o - pos) * 0.5;
        let tent = max(1.0 - d.x, 0.0) * max(1.0 - d.y, 0.0);
        let w = tent * exp(-abs(tz - z) / (0.02 * z + 0.05)) + 1e-5 * tent;
        sum += textureLoad(tex, t, 0) * w;
        weight += w;
    }
    return sum / max(weight, 1e-6);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= gp.fullSize)) { return; }
    let color = textureLoad(colorTex, gid.xy, 0);
    let px = vec2i(gid.xy);
    if (gp.debug == 2u) {
        textureStore(outTex, gid.xy, vec4f(marchVoxels((vec2f(gid.xy) + 0.5) / gp.fullSize), color.a));
        return;
    }
    if (gp.debug == 3u) {
        // the slice where it lies in front of the scene, which shows dimmed around it
        let suv = (vec2f(gid.xy) + 0.5) / gp.fullSize;
        var slice = sdfSliceColor(suv);
        let sceneDepth = gpDepth(vec2i(gid.xy));
        if (slice.a > 0.0 && sceneDepth < 1.0) {
            let eye = (gp.invView * vec4f(0.0, 0.0, 0.0, 1.0)).xyz;
            let surface = (gp.invView * vec4f(gpViewPos(suv, sceneDepth), 1.0)).xyz;
            let far = (gp.invView * vec4f(gpViewPos(suv, 1.0), 1.0)).xyz;
            let dir = normalize(far - eye);
            let t = (gp.sdfSlice - eye.y) / dir.y;
            if (length(surface - eye) < t) { slice.a = 0.0; }
        }
        textureStore(outTex, gid.xy, vec4f(select(color.rgb * 0.25, slice.rgb * 40.0, slice.a > 0.0), color.a));
        return;
    }
    let depth = gpDepth(px);
    let albedo = textureLoad(albedoTex, px, 0).rgb;
    if (depth >= 1.0 || all(albedo <= vec3f(0.0))) {
        textureStore(outTex, gid.xy, select(color, vec4f(0.0, 0.0, 0.0, color.a), gp.debug != 0u));
        return;
    }
    let uv = (vec2f(gid.xy) + 0.5) / gp.fullSize;
    let z = -gpViewPos(uv, depth).z;
    var e = upsample(giTex, gp.traceSize, uv, z).rgb;
    if (gp.nearField != 0u) {
        let near = upsample(nearTex, gp.nearSize, uv, z);
        e = max(near.rgb, (1.0 - near.a) * e) + near.a * e;
    }
    let n = gpWorldNormal(px);
    // the distance field's contact occlusion, which the coarse cones and the screen miss
    if (gp.hasSdf != 0u && gp.sdfAo > 0.0 && n.w > 0.0) {
        let world = (gp.invView * vec4f(gpViewPos(uv, depth), 1.0)).xyz;
        e *= mix(1.0, sdfAo(vol, sdfTex, linearClamp, world, n.xyz), gp.sdfAo);
    }
    let bounce = albedo * e * (gp.intensity / 3.14159265);
    if (gp.debug != 0u) {
        textureStore(outTex, gid.xy, vec4f(bounce, color.a));
        return;
    }
    var result = color.rgb + bounce;
    if (gp.hasSky != 0u && gp.ambient > 0.0 && n.w > 0.0) {
        result = max(result - albedo * skyIrradiance(sky, n.xyz) / 3.14159265 * gp.ambient, vec3f(0.0));
    }
    textureStore(outTex, gid.xy, vec4f(result, color.a));
}

// The composite with the probes as the far field (`VoxelGIEffect::set_probes`): main's, with each
// pixel's irradiance from the probes around it in place of the traced cones, and the probes' debug
// view (the voxels and the slice views run main). A separate entry point, so main's code (and what
// it outputs) stays as it was.
@compute @workgroup_size(8, 8)
fn main_probes(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= gp.fullSize)) { return; }
    let color = textureLoad(colorTex, gid.xy, 0);
    let px = vec2i(gid.xy);
    let uv = (vec2f(gid.xy) + 0.5) / gp.fullSize;
    let depth = gpDepth(px);
    if (gp.debug == 4u) {
        // the probes over the lit scene (without its GI)
        var sceneDist = 1e30;
        if (depth < 1.0) { sceneDist = length(gpViewPos(uv, depth)); }
        let ball = probeBalls(uv, sceneDist);
        textureStore(outTex, gid.xy, vec4f(select(color.rgb, ball.rgb, ball.a > 0.0), color.a));
        return;
    }
    let albedo = textureLoad(albedoTex, px, 0).rgb;
    if (depth >= 1.0 || all(albedo <= vec3f(0.0))) {
        textureStore(outTex, gid.xy, select(color, vec4f(0.0, 0.0, 0.0, color.a), gp.debug != 0u));
        return;
    }
    let view = gpViewPos(uv, depth);
    let ns = surfaceNormal(px, view);
    let world = (gp.invView * vec4f(view, 1.0)).xyz;
    // the lookup moved off the surface toward the viewer as well as along the normal (DDGI)
    let toEye = normalize((gp.invView * vec4f(-view, 0.0)).xyz);
    var e = kanseiProbeIrradiance(world, ns, ns * 0.4 + toEye * 0.6);
    if (gp.nearField != 0u) {
        let near = upsample(nearTex, gp.nearSize, uv, -view.z);
        e = max(near.rgb, (1.0 - near.a) * e) + near.a * e;
    }
    let n = gpWorldNormal(px);
    if (gp.hasSdf != 0u && gp.sdfAo > 0.0 && n.w > 0.0) {
        e *= mix(1.0, sdfAo(vol, sdfTex, linearClamp, world, n.xyz), gp.sdfAo);
    }
    let bounce = albedo * e * (gp.intensity / 3.14159265);
    if (gp.debug != 0u) {
        textureStore(outTex, gid.xy, vec4f(bounce, color.a));
        return;
    }
    var result = color.rgb + bounce;
    if (gp.hasSky != 0u && gp.ambient > 0.0 && n.w > 0.0) {
        result = max(result - albedo * skyIrradiance(sky, n.xyz) / 3.14159265 * gp.ambient, vec3f(0.0));
    }
    textureStore(outTex, gid.xy, vec4f(result, color.a));
}
