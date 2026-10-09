//! Dust: motes drifting in the room's air, stepped by a compute shader and drawn as small soft
//! sprites. They drift in a slow divergence-free flow (the curl of a few sine potentials) and the
//! character's legs and body push them aside. They live in a box round the camera (clamped to the
//! room), wrapping round its sides and fading near them, so the ones near the eye stay dense
//! wherever it goes.
//!
//! Their sprites come from an atlas a compute shader draws once at start (seeded, so the same every
//! time), with its mips: soft round motes, thin curled fibres, irregular flecks, and bokeh discs
//! (a bright rim, a faint hexagon) for the motes the depth of field blurs. Each mote picks a sprite
//! (`mix` sets how many are fibres and flecks), turns slowly, and grows into a soft disc as far as
//! the lens defocuses it, its light spread over the disc.
//!
//! Each mote is lit where it is by the sun (its cascades) and the spot lights (their shadow maps;
//! one tap: they are a few pixels across), scattering forward toward the eye looking into a beam,
//! so they catch the light inside the beams and vanish in shadow; the volumetric fog, later in the
//! chain, veils them with the same lights. They blend nearly additively (a tiny alpha), never
//! darkening what is behind them.

use kansei_core::buffers::{BufferType, ComputeBuffer, Sampler, Texture};
use kansei_core::geometries::{InstancedGeometry, PlaneGeometry};
use kansei_core::lights::{LIGHTS_WGSL, SPOT_LIGHTS_WGSL};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::Renderer;
use kansei_core::shadows::CASCADED_SHADOWS_WGSL;

use super::layout::{HALF, HEIGHT};

/// The box round the camera the motes live in: half its width and depth (m); it is the room's
/// height.
const BOX_HALF: f32 = 11.0;
/// The character's capsules the motes feel.
const CAPSULES: usize = 8;
/// The atlas: 4 x 4 sprites of 64 texels.
const ATLAS: u32 = 256;

const STEP_WGSL: &str = r#"
struct Capsule { a: vec4f, b: vec4f, velocity: vec4f };   // a.w: radius
struct Params {
    center: vec4f,      // xyz: the box's centre; w: time (s)
    half: vec4f,        // xyz: the box's half size; w: dt (s)
    room: vec4f,        // x: the room's half width, y: its height; z: the flow's speed (m/s); w: capsules
    capsules: array<Capsule, 8>,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4f>;

// the curl of three sine potentials, two octaves: swirls a few metres across, drifting in time
fn curl(p: vec3f, t: f32) -> vec3f {
    var v = vec3f(0.0);
    var scale = 0.32;
    var amp = 1.0;
    for (var o = 0; o < 2; o++) {
        let a = vec3f(0.9, 0.4, 0.2) * scale;
        let b = vec3f(0.3, 1.1, -0.5) * scale;
        let c = vec3f(-0.4, 0.3, 0.8) * scale;
        let ga = a * cos(dot(a, p) + t * 0.11);
        let gb = b * cos(dot(b, p) + t * 0.07 + 1.3);
        let gc = c * cos(dot(c, p) - t * 0.09 + 2.1);
        v += amp * vec3f(gc.y - gb.z, ga.z - gc.x, gb.x - ga.y) / scale;
        scale *= 2.3;
        amp *= 0.45;
    }
    return v;
}

fn closest_on_segment(p: vec3f, a: vec3f, b: vec3f) -> vec3f {
    let ab = b - a;
    let t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    return a + ab * t;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= arrayLength(&positions)) { return; }
    var p = positions[i].xyz;
    var v = velocities[i].xyz;
    let seed = positions[i].w;
    let t = params.center.w;
    let dt = params.half.w;
    // ease toward the flow (each mote a little differently), with a faint settling
    let flow = curl(p + vec3f(seed * 3.1), t) * params.room.z * (0.6 + 0.8 * fract(seed * 7.3)) + vec3f(0.0, -0.004, 0.0);
    v = mix(flow, v, exp(-dt * 0.7));
    // pushed aside by the character: out from each capsule and along with it
    for (var k = 0u; k < u32(params.room.w); k++) {
        let c = params.capsules[k];
        let q = closest_on_segment(p, c.a.xyz, c.b.xyz);
        let d = p - q;
        let dist = length(d);
        let reach = c.a.w + 0.45;
        if (dist < reach) {
            let s = 1.0 - dist / reach;
            v += (normalize(d + vec3f(1e-4, 0.0, 0.0)) * 0.9 + c.velocity.xyz * 0.6) * s * s * min(dt * 10.0, 1.0);
        }
    }
    p += v * dt;
    // wrap round the box's sides, and between the floor and the ceiling
    let size = 2.0 * params.half.xyz;
    let rel = p - params.center.xyz;
    p = params.center.xyz + rel - size * floor(rel / size + 0.5);
    p.y = 0.05 + (p.y - 0.05) - (params.room.y - 0.15) * floor((p.y - 0.05) / (params.room.y - 0.15));
    positions[i] = vec4f(p, seed);
    velocities[i] = vec4f(v, 0.0);
}
"#;

/// The atlas, drawn once: sprite k (row-major in a 4 x 4 grid) from a seeded hash. 0-5 soft
/// motes, 6-9 fibres, 10-13 flecks, 14-15 bokeh discs. Alpha holds the coverage; rgb a faint warm
/// or cool cast.
const ATLAS_WGSL: &str = r#"
@group(0) @binding(0) var atlas: texture_storage_2d<rgba8unorm, write>;

fn hash(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn noise(p: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2f(1.0, 0.0)), u.x), mix(hash(i + vec2f(0.0, 1.0)), hash(i + vec2f(1.0, 1.0)), u.x), u.y);
}

fn segment_distance(p: vec2f, a: vec2f, b: vec2f) -> f32 {
    let ab = b - a;
    let t = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
    return length(p - a - ab * t);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (any(id.xy >= vec2u(256u))) { return; }
    let cell = id.xy / 64u;
    let k = cell.y * 4u + cell.x;
    let seed = f32(k) * 17.13 + 3.7;
    // -1..1 across the cell
    let p = (vec2f(id.xy % 64u) + 0.5) / 64.0 * 2.0 - 1.0;
    var a = 0.0;
    if (k < 6u) {
        // a soft mote, a little elongated
        let e = 1.0 + hash(vec2f(seed, 1.0)) * 0.6;
        let q = vec2f(p.x * e, p.y);
        let r = length(q) / 0.55;
        a = exp(-r * r * 3.0);
    } else if (k < 10u) {
        // a fibre: a thin curved stroke of three segments
        var d = 9.0;
        var prev = vec2f(-0.7, (hash(vec2f(seed, 2.0)) - 0.5) * 0.6);
        for (var s = 1; s <= 3; s++) {
            let x = -0.7 + 1.4 * f32(s) / 3.0;
            let next = vec2f(x, (hash(vec2f(seed, f32(s) + 3.0)) - 0.5) * 0.7);
            d = min(d, segment_distance(p, prev, next));
            prev = next;
        }
        let width = 0.035 + 0.02 * noise(p * 6.0 + seed);
        a = (1.0 - smoothstep(width * 0.5, width, d)) * 0.9;
    } else if (k < 14u) {
        // a fleck: an irregular blob
        let r = length(p);
        let edge = 0.35 + 0.25 * noise(p * 3.0 + seed) + 0.1 * noise(p * 9.0 - seed);
        a = (1.0 - smoothstep(edge - 0.12, edge, r)) * (0.6 + 0.4 * noise(p * 12.0 + seed * 2.0));
    } else {
        // a bokeh disc: a faint hexagon, brighter at its rim
        let sector = 3.14159265 / 3.0;
        let ang = atan2(p.y, p.x) + 3.14159265;
        let hex = cos(sector * 0.5) / cos(ang - sector * floor(ang / sector) - sector * 0.5);
        let r = length(p) / (0.82 * mix(1.0, hex, 0.35));
        a = (1.0 - smoothstep(0.92, 1.0, r)) * (0.55 + 0.45 * smoothstep(0.6, 0.95, r));
    }
    let tint = mix(vec3f(1.0, 0.96, 0.9), vec3f(0.92, 0.96, 1.0), hash(vec2f(seed, 9.0)));
    textureStore(atlas, id.xy, vec4f(tint, clamp(a, 0.0, 1.0)));
}
"#;

/// A mip of the atlas from the one above it (a 2 x 2 box).
const MIP_WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(dst);
    if (any(id.xy >= size)) { return; }
    let s = id.xy * 2u;
    let c = textureLoad(src, s, 0) + textureLoad(src, s + vec2u(1u, 0u), 0) + textureLoad(src, s + vec2u(0u, 1u), 0) + textureLoad(src, s + vec2u(1u, 1u), 0);
    textureStore(dst, id.xy, c * 0.25);
}
"#;

/// The motes' material: sprites a few pixels across (or their size in the world, nearer), turned,
/// grown by the lens's defocus into bokeh, lit by the sun and the spot lights, faded near the box's
/// sides, far off and outside the room.
const DRAW_WGSL: &str = r#"
struct Dust {
    center: vec4f,      // xyz: the box's centre; w: the motes' radius (m)
    half: vec4f,        // xyz: the box's half size; w: the viewport's height (px)
    room: vec4f,        // x: the room's half width, y: its height; z: brightness; w: opacity
    look: vec4f,        // x: time (s); y: the share of fibres and flecks; z: the share drawn; w: unused
    lens: vec4f,        // x: focus distance (m); y: defocus (px of circle per unit of |1 - focus / z|); z: on
};
@group(0) @binding(0) var<uniform> dust: Dust;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var atlas_sampler: sampler;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4f;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4f;

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) corner: vec2f,
    @location(2) @interpolate(flat) cell: vec2f,
    @location(3) fade: f32,
};

fn mhash(x: f32) -> f32 {
    return fract(sin(x * 91.3458) * 47453.5453);
}

@vertex
fn vertex_main(@location(0) position: vec4f, @location(1) normal: vec3f, @location(2) uv: vec2f, @location(3) mote: vec4f) -> VOut {
    var out: VOut;
    let p = mote.xyz;
    let seed = mote.w;
    // outside the room (the camera's box reaches past a wall), or past the share drawn: not drawn
    let r = dust.room;
    let inside = all(abs(p.xz) < vec2f(r.x - 0.05)) && p.y > 0.0 && p.y < r.y && fract(seed * 7.77) < dust.look.z;
    let rel = abs(p - dust.center.xyz) / dust.half.xyz;
    let edge = max(rel.x, rel.z);
    let view = view_matrix * vec4f(p, 1.0);
    let z = -view.z;
    // dimmer far off
    out.fade = select(0.0, smoothstep(1.0, 0.75, edge), inside) * (0.5 + fract(seed * 13.7)) * smoothstep(22.0, 6.0, z);
    let pixel = z * 2.0 / (projection_matrix[1][1] * dust.half.w);
    var size = max(dust.center.w * (0.6 + 0.8 * fract(seed * 5.1)), 1.5 * pixel);
    // the sprite: a mote, or (by the mix) a fibre or a fleck
    var k = u32(fract(seed * 3.3) * 6.0);
    if (fract(seed * 11.1) < dust.look.y) {
        k = 6u + u32(fract(seed * 17.7) * 8.0);
    }
    // defocused: a bokeh disc the size of the circle of confusion, its light spread over it
    if (dust.lens.z > 0.5) {
        let coc = dust.lens.y * abs(1.0 - dust.lens.x / max(z, 0.05));
        let blur = coc * pixel * 0.5;
        if (blur > size) {
            out.fade *= clamp(size * size / (blur * blur), 0.02, 1.0);
            size = blur;
            if (coc > 4.0) { k = 14u + u32(fract(seed * 5.7) * 2.0); }
        }
    }
    out.cell = vec2f(f32(k % 4u), f32(k / 4u));
    // turning slowly, each its own way
    let angle = seed * 6.2831853 + dust.look.x * (mhash(seed) - 0.5) * 0.8;
    let c = cos(angle);
    let s = sin(angle);
    let corner = vec2f(position.x * c - position.y * s, position.x * s + position.y * c);
    let right = vec3f(view_matrix[0][0], view_matrix[1][0], view_matrix[2][0]);
    let up = vec3f(view_matrix[0][1], view_matrix[1][1], view_matrix[2][1]);
    let world = p + (right * corner.x + up * corner.y) * size * 2.0;
    out.clip = projection_matrix * view_matrix * vec4f(world, 1.0);
    if (out.fade <= 0.0 || z < 0.2) { out.clip = vec4f(0.0, 0.0, 2.0, 1.0); }
    out.world = p;
    out.corner = position.xy + 0.5;
    return out;
}

@fragment
fn fragment_main(in: VOut) -> @location(0) vec4f {
    let uv = (in.cell + mix(vec2f(0.06), vec2f(0.94), in.corner)) / 4.0;
    let sprite = textureSample(atlas, atlas_sampler, uv);
    if (sprite.a <= 0.003) { discard; }
    let view3 = mat3x3f(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let eye = -(transpose(view3) * view_matrix[3].xyz);
    let toEye = normalize(eye - in.world);
    let g = 0.6;
    var light = vec3f(0.0);
    // the sun, through the cascades
    for (var i = 0u; i < kansei_lights.num_directional; i++) {
        let d = kansei_lights.directional[i];
        let l = -normalize(d.direction);
        let lit = kansei_sun_shadow(in.world, l, in.clip.xy);
        let cosTheta = dot(-l, -toEye);
        let phase = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cosTheta, 1.5);
        light += d.color * lit * (0.25 + phase);
    }
    for (var k = 0u; k < kansei_spot_lights.count; k++) {
        let l = kansei_spot_lights.lights[k];
        let s = kansei_spot_sample(l, in.world);
        if (max(s.illuminance.r, max(s.illuminance.g, s.illuminance.b)) <= 0.0) { continue; }
        var lit = 1.0;
        if (l.shadowLayer >= 0) {
            let c = kansei_spot_shadow_coord(l, in.world);
            if (c.w > 0.0) {
                lit = textureSampleCompareLevel(kansei_spot_shadow_atlas, kansei_spot_shadow_sampler, c.xy, l.shadowLayer, c.z - 0.0005);
            }
        }
        // forward scattering (Henyey-Greenstein, g = 0.6): bright looking into a beam
        let cosTheta = dot(-s.toLight, -toEye);
        let phase = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cosTheta, 1.5);
        light += s.illuminance * lit * (0.25 + phase);
    }
    let radiance = light * dust.room.z * sprite.rgb;
    // nearly additive: a tiny alpha, the colour scaled up to match
    let a = sprite.a * in.fade * dust.room.w;
    let alpha = 0.02;
    return vec4f(radiance * a / alpha, alpha * a);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Capsule {
    a: [f32; 4],
    b: [f32; 4],
    velocity: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StepParams {
    center: [f32; 4],
    half: [f32; 4],
    room: [f32; 4],
    capsules: [Capsule; CAPSULES],
}

/// How the motes look: radius (m; at least 1.5 px), brightness (radiance per lux), opacity, the
/// flow's speed (m/s), the share of fibres and flecks, the share of the motes drawn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DustLook {
    pub size: f32,
    pub brightness: f32,
    pub opacity: f32,
    pub speed: f32,
    pub mix: f32,
    pub amount: f32,
}

impl Default for DustLook {
    fn default() -> Self {
        Self { size: 0.005, brightness: 0.0016, opacity: 0.7, speed: 0.06, mix: 0.3, amount: 1.0 }
    }
}

/// The lens, for the motes' bokeh: focus distance (m) and defocus (px per unit of
/// |1 - focus / z|), or none.
pub type DustLens = Option<(f32, f32)>;

pub struct Dust {
    count: u32,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    renderable: usize,
    time: f32,
    previous: Vec<[glam::Vec3; 2]>,
    pub look: DustLook,
}

/// A small fast hash in 0..1.
fn hash(i: u32) -> f32 {
    let mut x = i.wrapping_mul(0x9E37_79B9) ^ 0x85EB_CA6B;
    x ^= x >> 15;
    x = x.wrapping_mul(0x2C1B_3C6D);
    x ^= x >> 12;
    x = x.wrapping_mul(0x297A_2D39);
    x ^= x >> 15;
    (x >> 8) as f32 / (1u32 << 24) as f32
}

/// Draw the sprite atlas and its mips (once).
fn atlas(renderer: &Renderer) -> Texture {
    let device = renderer.device();
    let levels = ATLAS.trailing_zeros() + 1;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Room/Dust/Atlas"),
        size: wgpu::Extent3d { width: ATLAS, height: ATLAS, depth_or_array_layers: 1 },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let mip = |level| texture.create_view(&wgpu::TextureViewDescriptor { base_mip_level: level, mip_level_count: Some(1), ..Default::default() });
    let pipeline = |label, code: &str| {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(code.into()) });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
    };
    let draw = pipeline("Room/Dust/Atlas", ATLAS_WGSL);
    let down = pipeline("Room/Dust/AtlasMip", MIP_WGSL);
    let mut encoder = renderer.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Room/Dust/Atlas") });
    {
        let view = mip(0);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &draw.get_bind_group_layout(0), entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) }] });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&draw);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(ATLAS / 8, ATLAS / 8, 1);
    }
    for level in 1..levels {
        let (src, dst) = (mip(level - 1), mip(level));
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &down.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dst) }],
        });
        let size = (ATLAS >> level).max(1);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&down);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(size.div_ceil(8), size.div_ceil(8), 1);
    }
    renderer.submit(std::iter::once(encoder.finish()));
    let view = texture.create_view(&Default::default());
    Texture::from_view("Room/Dust/Atlas", texture, view)
}

impl Dust {
    /// `count` motes in the box round `center` (the camera's start).
    pub fn new(renderer: &Renderer, scene: &mut Scene, count: u32, center: glam::Vec3) -> Self {
        let device = renderer.device();
        let mut positions = Vec::with_capacity(count as usize * 4);
        for i in 0..count {
            positions.extend([
                center.x + (hash(i * 4) * 2.0 - 1.0) * BOX_HALF,
                0.05 + hash(i * 4 + 1) * (HEIGHT - 0.15),
                center.z + (hash(i * 4 + 2) * 2.0 - 1.0) * BOX_HALF,
                hash(i * 4 + 3),
            ]);
        }
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;
        let position_buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Room/Dust/Positions"), size: positions.len() as u64 * 4, usage, mapped_at_creation: false });
        renderer.queue().write_buffer(&position_buffer, 0, bytemuck::cast_slice(&positions));
        let velocities = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Room/Dust/Velocities"), size: count as u64 * 16, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        renderer.queue().write_buffer(&velocities, 0, &vec![0u8; count as usize * 16]);
        let params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Room/Dust/Params"), size: std::mem::size_of::<StepParams>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Room/Dust/Step"), source: wgpu::ShaderSource::Wgsl(STEP_WGSL.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Room/Dust/Step"), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Room/Dust/Step"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: position_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: velocities.as_entire_binding() },
            ],
        });

        // the sprites: a unit quad per mote, the positions as its instances
        let instances = ComputeBuffer::from_external("Room/Dust/Positions", position_buffer, BufferType::Storage).with_vertex_vec4(3);
        let geometry = InstancedGeometry::new(PlaneGeometry::new(1.0, 1.0), count, vec![instances]);
        let mut material = Material::new(
            "Room/Dust",
            &format!("{LIGHTS_WGSL}\n{CASCADED_SHADOWS_WGSL}\n{SPOT_LIGHTS_WGSL}\n{DRAW_WGSL}"),
            vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT), Binding::texture_2d(1, ShaderStages::FRAGMENT), Binding::sampler(2, ShaderStages::FRAGMENT)],
            MaterialOptions { transparent: true, depth_write: Some(false), cull_mode: CullMode::None, ..Default::default() },
        );
        material.set_uniform_bindable(0, "Room/Dust", &[0.0f32; 20]);
        material.set_bindable(1, atlas(renderer));
        material.set_bindable(2, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear));
        let mut r = Renderable::new(geometry, material);
        r.cast_shadow = false;
        r.dynamic = true;
        let renderable = scene.add(SceneNode::Renderable(r));
        log::info!("room: {count} dust motes");
        Self { count, pipeline, bind_group, params, renderable, time: 0.0, previous: Vec::new(), look: DustLook::default() }
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    pub fn visible(&self, scene: &Scene) -> bool {
        scene.get_renderable(self.renderable).is_some_and(|r| r.visible)
    }

    pub fn set_visible(&self, scene: &mut Scene, on: bool) {
        if let Some(r) = scene.get_renderable_mut(self.renderable) {
            r.visible = on;
        }
    }

    /// Step the motes by `dt` round `eye`, pushed by `capsules` (world ends and radius), and
    /// upload what their material reads (`viewport_height` in pixels, `lens` for the bokeh).
    #[allow(clippy::too_many_arguments)]
    pub fn update(&mut self, renderer: &Renderer, scene: &mut Scene, eye: glam::Vec3, capsules: &[(glam::Vec3, glam::Vec3, f32)], dt: f32, viewport_height: f32, lens: DustLens) {
        if !self.visible(scene) {
            return;
        }
        self.time += dt;
        let center = glam::Vec3::new(eye.x.clamp(-HALF + BOX_HALF * 0.5, HALF - BOX_HALF * 0.5), HEIGHT * 0.5, eye.z.clamp(-HALF + BOX_HALF * 0.5, HALF - BOX_HALF * 0.5));
        let half = [BOX_HALF, HEIGHT * 0.5, BOX_HALF];
        let capsules = &capsules[..capsules.len().min(CAPSULES)];
        if self.previous.len() != capsules.len() {
            self.previous = capsules.iter().map(|(a, b, _)| [*a, *b]).collect();
        }
        let look = self.look;
        let mut params = StepParams { center: [center.x, center.y, center.z, self.time], half: [half[0], half[1], half[2], dt], room: [HALF, HEIGHT, look.speed, capsules.len() as f32], capsules: [Capsule { a: [0.0; 4], b: [0.0; 4], velocity: [0.0; 4] }; CAPSULES] };
        for (k, ((a, b, r), [pa, pb])) in capsules.iter().zip(&self.previous).enumerate() {
            let v = ((*a - *pa) + (*b - *pb)) * 0.5 / dt.max(1e-3);
            let v = if v.length() > 15.0 { glam::Vec3::ZERO } else { v };
            params.capsules[k] = Capsule { a: [a.x, a.y, a.z, *r], b: [b.x, b.y, b.z, 0.0], velocity: [v.x, v.y, v.z, 0.0] };
        }
        self.previous = capsules.iter().map(|(a, b, _)| [*a, *b]).collect();
        renderer.queue().write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let mut encoder = renderer.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Room/Dust") });
        {
            let stamp = kansei_core::profiling::gpu_pass("Room/Dust");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Room/Dust"), timestamp_writes: stamp.as_ref().map(kansei_core::profiling::PassStamp::compute) });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.count.div_ceil(256), 1, 1);
        }
        renderer.submit(std::iter::once(encoder.finish()));
        let (focus, defocus, on) = lens.map_or((10.0, 0.0, 0.0), |(f, d)| (f, d, 1.0));
        #[rustfmt::skip]
        let draw = [
            center.x, center.y, center.z, look.size,
            half[0], half[1], half[2], viewport_height,
            HALF, HEIGHT, look.brightness, look.opacity,
            self.time, look.mix, look.amount, 0.0,
            focus, defocus, on, 0.0,
        ];
        if let Some(buffer) = scene.get_renderable_mut(self.renderable).and_then(|r| r.material.bindable_buffer(0)) {
            renderer.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(&draw));
        }
    }
}
