//! Moving colliders for the fluid: capsules the app places each frame (a character's legs, an
//! oar, a boat's hull made of a few) that push the particles out of them and carry them along
//! with their own velocity: bow waves, wakes and splashes. Applied after each substep's
//! integration by [`FluidColliders`] (a [`FluidSubstepPass`]).
//!
//! The coupling is one way: the fluid does not push back on the colliders.

use bytemuck::{Pod, Zeroable};

use super::simulation::{FluidSimulation, FluidSubstepPass, SIM_PARAMS_WGSL};

/// A capsule: the segment `a`-`b` grown by `radius`, and the velocity of each end (per second of
/// simulation time), in the simulation's space. A sphere is a capsule with `a == b`.
/// `expansion` is how fast its surface moves out along its normal (a body displacing the fluid
/// all round, such as a landing's impact), on top of the ends' velocities.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Pod, Zeroable)]
pub struct FluidCapsule {
    pub a: [f32; 3],
    pub radius: f32,
    pub b: [f32; 3],
    pub expansion: f32,
    pub velocity_a: [f32; 3],
    pub _pad1: f32,
    pub velocity_b: [f32; 3],
    pub _pad2: f32,
}

impl FluidCapsule {
    pub fn new(a: [f32; 3], b: [f32; 3], radius: f32, velocity_a: [f32; 3], velocity_b: [f32; 3]) -> Self {
        Self { a, radius, b, velocity_a, velocity_b, ..Default::default() }
    }

    /// The same capsule in a space scaled by `s`, with its velocities scaled by `velocity_scale`
    /// (e.g. `s` over the simulation's time scale).
    pub fn scaled(&self, s: f32, velocity_scale: f32) -> Self {
        Self { expansion: self.expansion * velocity_scale, ..Self::new(self.a.map(|v| v * s), self.b.map(|v| v * s), self.radius * s, self.velocity_a.map(|v| v * velocity_scale), self.velocity_b.map(|v| v * velocity_scale)) }
    }
}

/// How the colliders treat the particles they touch.
#[derive(Debug, Clone, Copy)]
pub struct FluidCollidersOptions {
    /// The fraction of the speed into a collider (relative to its surface) that bounces back.
    pub restitution: f32,
    /// The fraction of the speed along a collider (relative to its surface) lost at each
    /// contact: 1 carries the touching fluid along with it, 0 lets it slide.
    pub drag: f32,
}

impl Default for FluidCollidersOptions {
    fn default() -> Self {
        Self { restitution: 0.2, drag: 0.3 }
    }
}

/// GPU layout of the colliders' uniform (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuColliders {
    count: u32,
    restitution: f32,
    drag: f32,
    _pad: f32,
}

pub(crate) const COLLIDERS_WGSL: &str = r#"
struct Colliders {
    count: u32,
    restitution: f32,
    drag: f32,
    _pad: f32,
};
struct Capsule {
    a: vec3<f32>,
    radius: f32,
    b: vec3<f32>,
    expansion: f32,
    velocity_a: vec3<f32>,
    _pad1: f32,
    velocity_b: vec3<f32>,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<uniform> colliders: Colliders;
@group(0) @binding(4) var<storage, read> capsules: array<Capsule>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount || colliders.count == 0u) { return; }
    var pos = positions[idx].xyz;
    let v4 = velocities[idx];
    var vel = v4.xyz;
    var touched = false;

    for (var k = 0u; k < colliders.count; k++) {
        let c = capsules[k];
        let ab = c.b - c.a;
        let t = clamp(dot(pos - c.a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
        let q = c.a + ab * t;
        let d = pos - q;
        let dist = length(d);
        if (dist >= c.radius) { continue; }
        // out onto the surface; a particle on the axis leaves upward
        var n = vec3<f32>(0.0, 1.0, 0.0);
        if (dist > 1e-5) { n = d / dist; }
        pos = q + n * c.radius;
        // the collider's surface moves with its axis: relative to it, what goes in bounces back
        // and what slides along is dragged
        let surface = mix(c.velocity_a, c.velocity_b, t) + n * c.expansion;
        let rel = vel - surface;
        let vn = dot(rel, n);
        let vt = (rel - n * vn) * (1.0 - colliders.drag);
        vel = surface + vt + n * max(vn, -vn * colliders.restitution);
        touched = true;
    }

    if (touched) {
        positions[idx] = vec4<f32>(pos, positions[idx].w);
        velocities[idx] = vec4<f32>(vel, v4.w);
    }
}
"#;

/// The colliders' pass: up to `capacity` capsules, placed with [`FluidColliders::set`] (once a
/// frame: every substep of the frame sees the same ones).
pub struct FluidColliders {
    pub options: FluidCollidersOptions,
    capacity: usize,
    count: usize,
    queue: wgpu::Queue,
    uniform: wgpu::Buffer,
    capsules: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl FluidColliders {
    pub fn new(sim: &FluidSimulation, capacity: usize, options: FluidCollidersOptions) -> Self {
        let (device, queue) = sim.gpu();
        let capacity = capacity.max(1);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FluidColliders/Uniform"),
            size: std::mem::size_of::<GpuColliders>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let capsules = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FluidColliders/Capsules"),
            size: (capacity * std::mem::size_of::<FluidCapsule>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (pipeline, bind_group) = sim.substep_pipeline("FluidColliders", &format!("{SIM_PARAMS_WGSL}\n{COLLIDERS_WGSL}"), &[(&uniform, false), (&capsules, true)]);
        let colliders = Self { options, capacity, count: 0, queue: queue.clone(), uniform, capsules, pipeline, bind_group };
        colliders.upload_uniform();
        colliders
    }

    /// Place the capsules (the first `capacity` of them) for the next steps.
    pub fn set(&mut self, capsules: &[FluidCapsule]) {
        self.count = capsules.len().min(self.capacity);
        if self.count > 0 {
            self.queue.write_buffer(&self.capsules, 0, bytemuck::cast_slice(&capsules[..self.count]));
        }
        self.upload_uniform();
    }

    /// Upload `options` after changing them.
    pub fn upload_uniform(&self) {
        let gpu = GpuColliders { count: self.count as u32, restitution: self.options.restitution, drag: self.options.drag, _pad: 0.0 };
        self.queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&gpu));
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl FluidSubstepPass for FluidColliders {
    fn dispatch(&self, pass: &mut wgpu::ComputePass<'_>, particle_count: u32) {
        if self.count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(particle_count.div_ceil(64), 1, 1);
    }
}
