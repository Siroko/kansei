//! Dust: motes drifting in the room's air, stepped by a compute shader and drawn as tiny soft
//! billboards. They drift in a slow divergence-free flow (the curl of a few sine potentials) and
//! the character's legs and body push them aside. They live in a box round the camera (clamped to
//! the room), wrapping round its sides and fading near them, so the ones near the eye stay dense
//! wherever it goes.
//!
//! Each mote is lit where it is by the room's spot lights through their shadow maps (one
//! comparison tap: they are a pixel or two across), scattering more toward the eye looking into a
//! beam, so they catch the light inside the beams and vanish in shadow, like the fog around them:
//! the volumetric fog, later in the chain, attenuates and veils them with the same lights. They
//! blend nearly additively (a tiny alpha), never darkening what is behind them.

use kansei_core::buffers::{BufferType, ComputeBuffer};
use kansei_core::geometries::{InstancedGeometry, PlaneGeometry};
use kansei_core::lights::SPOT_LIGHTS_WGSL;
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::Renderer;

use super::layout::{HALF, HEIGHT};

/// The box round the camera the motes live in: half its width and depth (m); it is the room's
/// height.
const BOX_HALF: f32 = 11.0;
/// The character's capsules the motes feel.
const CAPSULES: usize = 8;

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

/// The motes' material: billboards a pixel or two across (or their size in the world, nearer),
/// lit by the spot lights, faded near the box's sides and outside the room.
const DRAW_WGSL: &str = r#"
struct Dust {
    center: vec4f,      // xyz: the box's centre; w: the motes' radius (m)
    half: vec4f,        // xyz: the box's half size; w: the viewport's height (px)
    room: vec4f,        // x: the room's half width, y: its height; z: brightness; w: opacity
};
@group(0) @binding(0) var<uniform> dust: Dust;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4f;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4f;

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) corner: vec2f,
    @location(2) fade: f32,
};

@vertex
fn vertex_main(@location(0) position: vec4f, @location(1) normal: vec3f, @location(2) uv: vec2f, @location(3) mote: vec4f) -> VOut {
    var out: VOut;
    let p = mote.xyz;
    // outside the room (the camera's box reaches past a wall): not drawn
    let r = dust.room;
    let inside = all(abs(p.xz) < vec2f(r.x - 0.05)) && p.y > 0.0 && p.y < r.y;
    let rel = abs(p - dust.center.xyz) / dust.half.xyz;
    let edge = max(rel.x, rel.z);
    out.fade = select(0.0, smoothstep(1.0, 0.75, edge), inside) * (0.5 + fract(mote.w * 13.7));
    let view = view_matrix * vec4f(p, 1.0);
    // at least 1.5 px across
    let pixel = -view.z * 2.0 / (projection_matrix[1][1] * dust.half.w);
    let size = max(dust.center.w * (0.6 + 0.8 * fract(mote.w * 5.1)), 1.5 * pixel);
    let right = vec3f(view_matrix[0][0], view_matrix[1][0], view_matrix[2][0]);
    let up = vec3f(view_matrix[0][1], view_matrix[1][1], view_matrix[2][1]);
    let world = p + (right * position.x + up * position.y) * size;
    out.clip = projection_matrix * view_matrix * vec4f(world, 1.0);
    if (out.fade <= 0.0 || view.z > -0.2) { out.clip = vec4f(0.0, 0.0, 2.0, 1.0); }
    out.world = p;
    out.corner = position.xy * 2.0;
    return out;
}

@fragment
fn fragment_main(in: VOut) -> @location(0) vec4f {
    let disc = 1.0 - smoothstep(0.3, 1.0, length(in.corner));
    if (disc <= 0.0) { discard; }
    let view3 = mat3x3f(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let eye = -(transpose(view3) * view_matrix[3].xyz);
    let toEye = normalize(eye - in.world);
    var light = vec3f(0.0);
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
        let g = 0.6;
        let phase = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cosTheta, 1.5);
        light += s.illuminance * lit * (0.25 + phase);
    }
    let radiance = light * dust.room.z;
    // nearly additive: a tiny alpha, the colour scaled up to match
    let a = disc * in.fade * dust.room.w;
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

pub struct Dust {
    count: u32,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    renderable: usize,
    time: f32,
    previous: Vec<[glam::Vec3; 2]>,
    /// The flow's speed (m/s), the motes' radius (m; they are drawn at least 1.5 px across), their
    /// brightness (radiance per lux) and opacity.
    pub speed: f32,
    pub size: f32,
    pub brightness: f32,
    pub opacity: f32,
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

        // the billboards: a unit quad per mote, the positions as its instances
        let instances = ComputeBuffer::from_external("Room/Dust/Positions", position_buffer, BufferType::Storage).with_vertex_vec4(3);
        let geometry = InstancedGeometry::new(PlaneGeometry::new(1.0, 1.0), count, vec![instances]);
        let mut material = Material::new(
            "Room/Dust",
            &format!("{SPOT_LIGHTS_WGSL}\n{DRAW_WGSL}"),
            vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)],
            MaterialOptions { transparent: true, depth_write: Some(false), cull_mode: CullMode::None, ..Default::default() },
        );
        material.set_uniform_bindable(0, "Room/Dust", &[0.0f32; 12]);
        let mut r = Renderable::new(geometry, material);
        r.cast_shadow = false;
        r.dynamic = true;
        let renderable = scene.add(SceneNode::Renderable(r));
        log::info!("room: {count} dust motes");
        Self { count, pipeline, bind_group, params, renderable, time: 0.0, previous: Vec::new(), speed: 0.06, size: 0.006, brightness: 0.14, opacity: 0.6 }
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
    /// upload what their material reads (`viewport_height` in pixels).
    pub fn update(&mut self, renderer: &Renderer, scene: &mut Scene, eye: glam::Vec3, capsules: &[(glam::Vec3, glam::Vec3, f32)], dt: f32, viewport_height: f32) {
        let visible = scene.get_renderable(self.renderable).is_some_and(|r| r.visible);
        if !visible {
            return;
        }
        self.time += dt;
        let center = glam::Vec3::new(eye.x.clamp(-HALF + BOX_HALF * 0.5, HALF - BOX_HALF * 0.5), HEIGHT * 0.5, eye.z.clamp(-HALF + BOX_HALF * 0.5, HALF - BOX_HALF * 0.5));
        let half = [BOX_HALF, HEIGHT * 0.5, BOX_HALF];
        let capsules = &capsules[..capsules.len().min(CAPSULES)];
        if self.previous.len() != capsules.len() {
            self.previous = capsules.iter().map(|(a, b, _)| [*a, *b]).collect();
        }
        let mut params = StepParams { center: [center.x, center.y, center.z, self.time], half: [half[0], half[1], half[2], dt], room: [HALF, HEIGHT, self.speed, capsules.len() as f32], capsules: [Capsule { a: [0.0; 4], b: [0.0; 4], velocity: [0.0; 4] }; CAPSULES] };
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
        // the material's uniform: the box, the motes' radius, the viewport, brightness, opacity
        let draw = [center.x, center.y, center.z, self.size, half[0], half[1], half[2], viewport_height, HALF, HEIGHT, self.brightness, self.opacity];
        if let Some(buffer) = scene.get_renderable_mut(self.renderable).and_then(|r| r.material.bindable_buffer(0)) {
            renderer.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(&draw));
        }
    }
}
