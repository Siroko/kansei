// The asphalt road (after forest.wgsl): the Raggare intro's road.wgsl, its WGSL port of the
// Unreal intro's road material (M_IntroRoad and MidsommarRoad.ush in the game repo): four scanned
// CC0 asphalts (ambientCG, served with the intro's data) mixed with painted lines, wheel tracks,
// repairs, cracks, gritty edges and a damp, puddled surface, normal mapped and lit by the sun and
// the sky. The film's road decals are left out.
//
// Mesh uv = (lateral -1..1 across carriageway + shoulders, station m); the port works in road
// metres like the .ush: u across (0 on the centreline, positive right of travel), v along.

struct RoadParams {
    half_width: f32,     // half the asphalt's width (m)
    line_offset: f32,    // the edge lines' distance from the centreline (m)
    wetness: f32,        // 0..1
    lateral_scale: f32,  // mesh uv.x -> metres (the ribbon's half-width)
    albedo_scale: f32,   // calibrates the scanned albedo to the Unreal frames
    normal_scale: f32,   // strength of the textures' relief
    sky_reflection: f32, // strength of the sky seen in the damp asphalt
    roughness_scale: f32, // calibrates the scanned roughness to the Unreal frames
    // > 0: the road is wet for the ray-traced reflections (reflect=1): this F0 and roughness go
    // into the GBuffer and the material leaves its own sky reflection out
    wet_f0: f32,
    wet_roughness: f32,
    _pad0: f32,
    _pad1: f32,
};
@group(0) @binding(0) var<uniform> road: RoadParams;
// per surface: colour (sRGB) and normal.xy + roughness + occlusion
@group(0) @binding(8) var base_color: texture_2d<f32>;
@group(0) @binding(9) var base_nra: texture_2d<f32>;
@group(0) @binding(10) var var_color: texture_2d<f32>;
@group(0) @binding(11) var var_nra: texture_2d<f32>;
@group(0) @binding(12) var patch_color: texture_2d<f32>;
@group(0) @binding(13) var patch_nra: texture_2d<f32>;
@group(0) @binding(14) var edge_color: texture_2d<f32>;
@group(0) @binding(15) var edge_nra: texture_2d<f32>;
@group(0) @binding(16) var surface_sampler: sampler;

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vertex_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.world = in.position.xyz;
    out.clip = projection_matrix * view_matrix * vec4<f32>(out.world, 1.0);
    out.normal = in.normal;
    out.uv = in.uv;
    return out;
}

// ---- MidsommarRoad.ush helpers ---------------------------------------------------------------

fn road_hash(q: vec2<f32>) -> f32 {
    var p = fract(q * vec2<f32>(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

fn road_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = road_hash(i);
    let b = road_hash(i + vec2<f32>(1.0, 0.0));
    let c = road_hash(i + vec2<f32>(0.0, 1.0));
    let d = road_hash(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn road_fbm(q: vec2<f32>) -> f32 {
    var p = q;
    var sum = 0.0;
    var amplitude = 0.5;
    for (var octave = 0; octave < 4; octave++) {
        sum += amplitude * road_noise(p);
        p = p * 2.03 + vec2<f32>(17.1, 5.3);
        amplitude *= 0.5;
    }
    return sum;
}

// coverage of a band |x - centre| < hw, antialiased over the pixel's footprint w
fn band(x: f32, centre: f32, hw: f32, w: f32) -> f32 {
    return clamp((hw - abs(x - centre)) / max(w, 1e-4) + 0.5, 0.0, 1.0);
}

// coverage of dashes along v: `dash` metres of paint every `period` metres, from `offset`
fn dashes(v: f32, period: f32, dash: f32, offset: f32, w: f32) -> f32 {
    let t = fract((v - offset) / period) * period;
    return clamp((dash - t) / max(w, 1e-4) + 0.5, 0.0, 1.0) * clamp(t / max(w, 1e-4) + 0.5, 0.0, 1.0);
}

// ---- the scanned surfaces --------------------------------------------------------------------

struct Surface {
    color: vec3<f32>,
    normal: vec3<f32>, // road tangent space: x across (right), y along, z up
    rough: f32,
    ao: f32,
};

// Sample a surface at texture coordinates `c`. The normal maps are OpenGL-style (green points up
// the image, i.e. toward -c.y); `swapped` surfaces are sampled at (v, u), so their normal's
// components swap back into road space. `gain` evens out the scans' albedo.
fn surface(ct: texture_2d<f32>, nt: texture_2d<f32>, c: vec2<f32>, swapped: bool, gain: f32) -> Surface {
    let col = textureSample(ct, surface_sampler, c).rgb;
    let t = textureSample(nt, surface_sampler, c);
    var xy = vec2<f32>(t.r * 2.0 - 1.0, 1.0 - t.g * 2.0) * road.normal_scale;
    xy = select(xy, xy.yx, swapped);
    let n = vec3<f32>(xy, sqrt(clamp(1.0 - dot(xy, xy), 0.0, 1.0)));
    return Surface(col * (gain * road.albedo_scale), normalize(n), min(t.b * road.roughness_scale, 1.0), t.a);
}

struct RoadSurface {
    color: vec3<f32>,
    rough: f32,
    normal: vec3<f32>,
    ao: f32,
};

// MidsommarRoadSurface: uv in road metres.
fn road_surface(uv: vec2<f32>, fw: vec2<f32>, base: Surface, vari: Surface, repair: Surface, edge: Surface) -> RoadSurface {
    let lat = uv.x;
    let along = uv.y;
    let a = abs(lat);
    let wx = max(fw.x, 1e-4);
    let wy = max(fw.y, 1e-4);
    let half_width = road.half_width;
    let flat_n = vec3<f32>(0.0, 0.0, 1.0);

    // two worn asphalts in long, lazy patches
    let macro_mix = road_fbm(vec2<f32>(along * 0.045, lat * 0.18 + 3.1));
    let var_mix = smoothstep(0.42, 0.62, macro_mix);
    var color = mix(base.color, vari.color, var_mix);
    var normal = normalize(mix(base.normal, vari.normal, var_mix));
    var rough = mix(base.rough, vari.rough, var_mix);
    var occlusion = mix(base.ao, vari.ao, var_mix);

    // wheel tracks: two per lane, polished smoother and a shade darker
    let track = exp(-pow((a - 0.82) / 0.3, 2.0)) + exp(-pow((a - 2.42) / 0.3, 2.0));
    color *= 1.0 - 0.12 * track;
    rough -= 0.12 * track;

    // repairs: a rectangle of fresher asphalt in some 23 m stretches, over one lane or the road
    let cell = floor(along / 23.0);
    let h0 = road_hash(vec2<f32>(cell, 7.0));
    let h1 = road_hash(vec2<f32>(cell, 13.0));
    let h2 = road_hash(vec2<f32>(cell, 29.0));
    let patch_start = cell * 23.0 + h1 * 9.0;
    let patch_length = 2.5 + h2 * 9.0;
    let patch_side = select(select(-1.65, 1.65, h2 < 0.5), 0.0, h1 < 0.3);
    let patch_half = select(1.0 + 0.5 * h0, half_width - 0.35, patch_side == 0.0);
    let ragged = 0.08 * (road_noise(vec2<f32>(along * 3.0, lat * 3.0)) - 0.5);
    let patched = step(h0, 0.42)
        * band(along, patch_start + patch_length * 0.5, patch_length * 0.5 + ragged, wy)
        * band(lat, patch_side, patch_half + ragged, wx);
    color = mix(color, repair.color * 0.85, patched);
    normal = normalize(mix(normal, repair.normal, patched));
    rough = mix(rough, repair.rough, patched);
    occlusion = mix(occlusion, repair.ao, patched);

    // the edges: sandier, crumbling asphalt with grit washed onto it
    let edge_noise = road_noise(vec2<f32>(along * 0.7, 11.0)) * 0.35;
    let edged = smoothstep(half_width - 0.55 - edge_noise, half_width - 0.1, a);
    color = mix(color, edge.color, edged);
    normal = normalize(mix(normal, edge.normal, edged));
    rough = mix(rough, edge.rough, edged);
    occlusion = mix(occlusion, edge.ao, edged);

    // cracks: one wandering along the lane joint and the odd one across the road, sealed with
    // darker bitumen in places
    let joint = 0.28 + 0.12 * (road_noise(vec2<f32>(along * 0.23, 3.0)) - 0.5);
    let joint_on = step(0.45, road_noise(vec2<f32>(along * 0.09, 5.0)));
    var crack = joint_on * band(lat, joint, 0.006 + 0.01 * road_noise(vec2<f32>(along * 2.0, 1.0)), wx);
    let cross_cell = floor(along / 17.0);
    let cross_at = cross_cell * 17.0 + 3.0 + 11.0 * road_hash(vec2<f32>(cross_cell, 3.0));
    let cross_wobble = 0.25 * (road_noise(vec2<f32>(lat * 1.3, cross_cell)) - 0.5);
    crack = max(crack, step(road_hash(vec2<f32>(cross_cell, 41.0)), 0.5) * band(along, cross_at + cross_wobble, 0.008, wy)
        * step(a, half_width - 0.3 * road_hash(vec2<f32>(cross_cell, 9.0))));
    let sealed = step(0.6, road_noise(vec2<f32>(along * 0.05, 21.0)));
    let seal = sealed * band(lat, joint, 0.05, wx) * joint_on;
    color *= (1.0 - 0.55 * seal) * (1.0 - 0.7 * crack);
    rough = mix(rough, 0.55, seal);

    // paint: a dashed centre line (3 m every 12) and dashed edge lines (1 m every 3), worn thin
    // by the tyres and the plough blades
    let wear = smoothstep(0.2, 0.6, road_fbm(vec2<f32>(along * 1.6, lat * 9.0)));
    var paint = band(lat, 0.0, 0.05, wx) * dashes(along, 12.0, 3.0, 0.0, wy);
    paint += (band(lat, road.line_offset, 0.05, wx) + band(lat, -road.line_offset, 0.05, wx)) * dashes(along, 3.0, 1.0, 0.4, wy);
    paint = clamp(paint, 0.0, 1.0) * (0.35 + 0.65 * wear) * (1.0 - 0.5 * patched);
    color = mix(color, vec3<f32>(0.62, 0.62, 0.58), paint);
    rough = mix(rough, 0.42, paint);
    normal = normalize(mix(normal, flat_n, paint * 0.7));

    // damp everywhere after the evening rain, with puddles standing in the wheel tracks and the
    // low spots: darker, smoother and flatter where the water is
    let low = road_fbm(vec2<f32>(along * 0.11, lat * 0.45 + 7.0));
    let puddle = clamp(road.wetness * smoothstep(0.58, 0.7, low + 0.12 * track - 0.08 * edged), 0.0, 1.0);
    let damp = road.wetness * 0.65;
    color *= mix(1.0, 0.62, max(damp * (1.0 - paint * 0.6), puddle));
    rough = mix(rough * mix(1.0, 0.62, damp), 0.05, puddle);
    normal = normalize(mix(normal, flat_n, puddle * 0.9 + damp * 0.25));

    return RoadSurface(color, clamp(rough, 0.0, 1.0), normal, occlusion);
}

struct Shaded {
    surface: RoadSurface,
    normal: vec3<f32>,
    rough: f32,
};

fn shade(in: VertexOutput) -> Shaded {
    let uv = vec2<f32>(in.uv.x * road.lateral_scale, in.uv.y);
    let fw = fwidth(uv);

    // cotangent frame from screen-space derivatives: t follows +u (right), b follows +v (along)
    let n0 = normalize(in.normal);
    let tbn = cotangent_frame(n0, dpdx(in.world), dpdy(in.world), dpdx(uv), dpdy(uv));

    // the four surfaces at their Unreal tile sizes; the variation and the edge are turned 90 deg.
    // The repair scan (Asphalt010) is a third as bright as the base: lift it to a fresher,
    // darker-but-not-black patch.
    let base = surface(base_color, base_nra, uv / 2.2, false, 1.0);
    let vari = surface(var_color, var_nra, uv.yx / 3.1 + vec2<f32>(0.37, 0.61), true, 0.95);
    let repair = surface(patch_color, patch_nra, uv / 2.0 + vec2<f32>(0.71, 0.13), false, 2.6);
    let edge = surface(edge_color, edge_nra, uv.yx / 1.8, true, 1.0);
    let s = road_surface(uv, fw, base, vari, repair, edge);

    let n = normalize(tbn * s.normal);
    // specular antialiasing (Kaplanyan & Hoffman 2016): widen the lobe by how much the normal
    // varies across the pixel, so the relief does not sparkle in the distance
    let dndx = dpdx(n);
    let dndy = dpdy(n);
    let variance = 0.25 * (dot(dndx, dndx) + dot(dndy, dndy));
    let alpha = max(s.rough, 0.06) * max(s.rough, 0.06);
    let rough = sqrt(sqrt(clamp(alpha * alpha + min(2.0 * variance, 0.18), 0.0, 1.0)));
    return Shaded(s, n, rough);
}

@fragment
fn fragment_main(in: VertexOutput) -> KanseiGBufferOut {
    let shaded = shade(in);
    let s = shaded.surface;
    let n = shaded.normal;
    let p = in.world;
    let v = normalize(view_eye() - p);
    let l = -kansei_cascades.lightDirection;
    let sun = kansei_cascades.lightColor * kansei_sun_shadow(p, n, in.clip.xy);
    let diffuse = s.color / PI * (sun * max(dot(n, l), 0.0) + sky_light(p, n) * s.ao);
    let specular = sun * ggx_specular(n, v, l, shaded.rough, 0.04);
    if (road.wet_f0 > 0.0) {
        // the ray-traced reflections add the rest
        return kansei_gbuffer_out_specular(diffuse + specular, vec3<f32>(0.0), n, s.color, road.wet_f0, road.wet_roughness);
    }
    // the damp asphalt mirrors a little of the sky, more at grazing angles and where smooth
    let nv = max(dot(n, v), 0.0);
    let fresnel = 0.04 + 0.96 * pow(1.0 - nv, 5.0);
    let gloss = (1.0 - shaded.rough) * (1.0 - shaded.rough);
    let r = reflect(-v, n);
    let reflection = skyRadiance(sky, r) * fresnel * gloss * road.sky_reflection * sky_visibility(p, r) * s.ao;
    return kansei_gbuffer_out(diffuse + specular + reflection, vec3<f32>(0.0), n, s.color);
}

@fragment
fn voxel_main(in: VertexOutput, @builtin(front_facing) front: bool) {
    kansei_voxel_write(in.clip, front, shade(in).surface.color, vec3<f32>(0.0));
}
