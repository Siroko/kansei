// The terrain (after forest.wgsl): CC0 ground scans (ambientCG, served with the Raggare intro's
// data) blended by the exported splat weights (R meadow, G forest floor, B gravel, A dirt, the
// rest sand). Each scan is calibrated to a mean albedo (`layers`), sampled at two scales and
// rotations against visible tiling, and the layers meet along their relief (height blending on the
// scans' occlusion) rather than in soft cross-fades. The Raggare intro's terrain.wgsl, lit by the
// sun and the sky.
struct Layers {
    // per layer: rgb = albedo gain, w = tile size (m)
    layer: array<vec4<f32>, 6>,
};
@group(0) @binding(0) var<uniform> layers: Layers;
@group(0) @binding(8) var splat_tex: texture_2d<f32>;
@group(0) @binding(9) var splat_sampler: sampler;
// the six scans as array layers: meadow, forest, moss, gravel, dirt, sand
@group(0) @binding(10) var ground_color: texture_2d_array<f32>;
@group(0) @binding(11) var ground_nra: texture_2d_array<f32>;
@group(0) @binding(12) var ground_sampler: sampler;

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
    out.world = in.position.xyz; // terrain vertices are already in world space
    out.clip = projection_matrix * view_matrix * vec4<f32>(out.world, 1.0);
    out.normal = in.normal;
    out.uv = in.uv;
    return out;
}

fn ground_noise(q: vec2<f32>) -> f32 {
    let i = floor(q);
    let f = fract(q);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i + vec2<f32>(1.0, 0.0)), u.x), mix(hash12(i + vec2<f32>(0.0, 1.0)), hash12(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

fn ground_fbm(q: vec2<f32>) -> f32 {
    return ground_noise(q) * 0.5 + ground_noise(q * 2.07 + 17.0) * 0.3 + ground_noise(q * 4.3 + 31.0) * 0.2;
}

struct Layer {
    color: vec3<f32>,
    normal: vec2<f32>, // tangent-space xy: x along +X, y along -Z
    rough: f32,
    height: f32, // the scan's occlusion, standing in for its relief
};

// One layer at ground position `g` (m; x east, y north): the scan at its tile size, and again
// turned and at 2.7 times the size, mixed by a slow noise so neither repeat reads.
fn layer(g: vec2<f32>, k: u32, mixer: f32) -> Layer {
    let p = layers.layer[k];
    let c1 = g / p.w;
    let rot = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
    let c2 = rot * g / (p.w * 2.7) + vec2<f32>(0.31, 0.77) * f32(k + 1u);
    let a = textureSample(ground_color, ground_sampler, c1, k).rgb;
    let b = textureSample(ground_color, ground_sampler, c2, k).rgb;
    let na = textureSample(ground_nra, ground_sampler, c1, k);
    let nb = textureSample(ground_nra, ground_sampler, c2, k);
    // normals: OpenGL maps, green pointing up the image (toward -v); the turned sample's normal is
    // turned back into ground space
    let xa = vec2<f32>(na.r * 2.0 - 1.0, 1.0 - na.g * 2.0);
    let xb = transpose(rot) * vec2<f32>(nb.r * 2.0 - 1.0, 1.0 - nb.g * 2.0);
    var out: Layer;
    out.color = mix(a, b, mixer) * p.rgb;
    out.normal = mix(xa, xb, mixer);
    out.rough = mix(na.b, nb.b, mixer);
    out.height = mix(na.a, nb.a, mixer);
    return out;
}

struct Ground {
    albedo: vec3<f32>,
    normal: vec3<f32>,
    ao: f32,
};

fn ground(in: VertexOutput) -> Ground {
    let w = textureSample(splat_tex, splat_sampler, in.uv);
    let sand_w = max(0.0, 1.0 - (w.r + w.g + w.b + w.a));
    let g = vec2<f32>(in.world.x, -in.world.z);
    let mixer = smoothstep(0.3, 0.7, ground_fbm(g * 0.09));

    let meadow = layer(g, 0u, mixer);
    let forest = layer(g, 1u, mixer);
    let moss = layer(g, 2u, mixer);
    let gravel = layer(g, 3u, mixer);
    let dirt = layer(g, 4u, mixer);
    let sand = layer(g, 5u, mixer);

    // the forest floor: needle litter with cushions of moss, in patches a few metres across
    let moss_share = smoothstep(0.45, 0.65, ground_fbm(g * 0.23 + 5.0));
    var floor_: Layer;
    let fm = select(0.0, 1.0, moss.height + moss_share * 1.2 > forest.height + 0.6) * moss_share;
    floor_.color = mix(forest.color, moss.color, fm);
    floor_.normal = mix(forest.normal, moss.normal, fm);
    floor_.rough = mix(forest.rough, moss.rough, fm);
    floor_.height = mix(forest.height, moss.height, fm);

    // height blending: each layer's weight lifted by its relief, and only what stands within
    // `depth` of the highest survives, so the layers interlock at their edges
    let depth = 0.2;
    let h = array<f32, 5>(
        meadow.height + w.r * 1.5,
        floor_.height + w.g * 1.5,
        gravel.height + w.b * 1.5,
        dirt.height + w.a * 1.5,
        sand.height + sand_w * 1.5,
    );
    let wts = array<f32, 5>(w.r, w.g, w.b, w.a, sand_w);
    var top = -1.0;
    for (var i = 0; i < 5; i++) {
        if (wts[i] > 0.001) {
            top = max(top, h[i]);
        }
    }
    var b = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
    var sum = 0.0;
    for (var i = 0; i < 5; i++) {
        b[i] = select(0.0, max(h[i] - top + depth, 0.0), wts[i] > 0.001);
        sum += b[i];
    }
    let inv = 1.0 / max(sum, 1e-4);
    let colour = (meadow.color * b[0] + floor_.color * b[1] + gravel.color * b[2] + dirt.color * b[3] + sand.color * b[4]) * inv;
    let nxy = (meadow.normal * b[0] + floor_.normal * b[1] + gravel.normal * b[2] + dirt.normal * b[3] + sand.normal * b[4]) * inv;
    let ao = (meadow.height * b[0] + floor_.height * b[1] + gravel.height * b[2] + dirt.height * b[3] + sand.height * b[4]) * inv;

    // broad variation in tone over tens of metres, as damp and dry ground alternate
    let macro_ = 0.8 + 0.4 * ground_fbm(g * 0.02 + 3.0);

    // tangent frame of the heightfield: x along +X, y along -Z (right-handed with the normal)
    let n0 = normalize(in.normal);
    let t = normalize(vec3<f32>(1.0, 0.0, 0.0) - n0 * n0.x);
    let bt = normalize(vec3<f32>(0.0, 0.0, -1.0) + n0 * n0.z);
    let nz = sqrt(clamp(1.0 - dot(nxy, nxy), 0.0, 1.0));
    var out: Ground;
    out.albedo = colour * macro_;
    out.normal = normalize(t * nxy.x + bt * nxy.y + n0 * nz);
    out.ao = ao;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> KanseiGBufferOut {
    let s = ground(in);
    let radiance = s.albedo / PI * (sun_light(in.world, s.normal, in.clip.xy) + sky_light(in.world, s.normal) * s.ao);
    return kansei_gbuffer_out(radiance, vec3<f32>(0.0), s.normal, s.albedo);
}

@fragment
fn voxel_main(in: VertexOutput, @builtin(front_facing) front: bool) {
    kansei_voxel_write(in.clip, front, ground(in).albedo, vec3<f32>(0.0));
}
