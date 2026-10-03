use bytemuck::{Pod, Zeroable};

use super::volume::VoxelVolume;
use crate::buffers::{BufferType, ComputeBuffer, InstanceAttribute, VertexFormat};
use crate::materials::{Binding, BindingResource, Compute};

pub(crate) const PARTICLE_CONES_WGSL: &str = concat!(
    include_str!("shaders/voxel_volume.wgsl"),
    include_str!("shaders/voxel_cones.wgsl"),
    include_str!("shaders/particle_emission.wgsl"),
    include_str!("../atmosphere/shaders/sky_lighting.wgsl"),
    include_str!("shaders/particle_cones.wgsl"),
);

/// Bytes per particle of `ParticleConeShading`'s lighting buffer: two vec4.
pub const PARTICLE_LIGHTING_STRIDE: u64 = 32;

/// How particles gather light from the volume (`ParticleConeShading::encode`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleConeSettings {
    /// Direction toward the sun: the narrow cone's axis.
    pub to_sun: [f32; 3],
    /// tan of the sun cone's half aperture: the softness of the particles' shadow.
    pub sun_cone_tan: f32,
    /// tan of the six diffuse cones' half aperture (1: 90-degree cones tiling the sphere).
    pub diffuse_cone_tan: f32,
    /// Voxels out the cones start, past the particle's own splat (1 to 2).
    pub start_voxels: f32,
    /// How far the cones look, metres.
    pub max_distance_m: f32,
    /// Weight of this frame in each particle's running average (1: no history).
    pub temporal_blend: f32,
    /// Voxels the cones' start moves by, per particle and frame (the average smooths it).
    pub jitter_voxels: f32,
    /// The most steps a cone takes (`VoxelGiQuality::cone_steps`).
    pub max_steps: u32,
    /// false: trace no cones and give every particle the whole sky and the sun (voxel GI off;
    /// the volume is not read, so it need not be built).
    pub use_volume: bool,
}

impl Default for ParticleConeSettings {
    fn default() -> Self {
        Self {
            to_sun: [0.0, 1.0, 0.0],
            sun_cone_tan: 0.05,
            diffuse_cone_tan: 1.0,
            start_voxels: 1.5,
            max_distance_m: 1e4,
            temporal_blend: 0.2,
            jitter_voxels: 0.5,
            max_steps: 48,
            use_volume: true,
        }
    }
}

/// The WGSL `ConeParams` (particle_cones.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ConeParamsGpu {
    to_sun: [f32; 3],
    sun_cone_tan: f32,
    diffuse_cone_tan: f32,
    start_voxels: f32,
    max_distance: f32,
    temporal_blend: f32,
    particle_count: u32,
    max_steps: u32,
    frame: u32,
    jitter: f32,
    use_volume: u32,
    _pad: [u32; 3],
}

/// The WGSL `SkyLighting` (`SKY_LIGHTING_WGSL`): 15 vec4.
pub type SkyLightingData = [[f32; 4]; 15];

/// A `SkyLighting` whose radiance runs from `down` (straight below) to `up` (straight above),
/// linearly in the direction's height, with no sun or moon: the sky for cones that leave a
/// volume when the scene has no `SkyAtmosphere`. A constant sky is `up == down`.
pub fn gradient_sky_lighting(up: [f32; 3], down: [f32; 3]) -> SkyLightingData {
    let mut sky = [[0.0; 4]; 15];
    for c in 0..3 {
        // skyRadiance(d) = 0.282095 sh0 + 0.488603 sh1 d.y
        sky[0][c] = (up[c] + down[c]) * 0.5 / 0.282095;
        sky[1][c] = (up[c] - down[c]) * 0.5 / 0.488603;
    }
    sky
}

/// Per-particle light from cone tracing a `VoxelVolume`, miaumiau.cat/?p=1476's gather: each
/// particle traces six 90-degree cones along the axes (escaping to the sky) and one narrow cone
/// toward the sun, and keeps a running average, since particle indices are stable. It writes
/// `PARTICLE_LIGHTING_STRIDE` bytes per particle, for the particles' material to read as instance
/// attributes (`lighting_instance_buffer`):
/// - `vec4(mean incoming radiance, sun visibility)`: a Lambertian particle of albedo `k` reflects
///   `k * rgb`, plus its sun light times `a`;
/// - `vec4(the particle's emission, 0)`, as the splat put it in the volume.
pub struct ParticleConeShading {
    params: wgpu::Buffer,
    lighting: wgpu::Buffer,
    shade: Compute,
    capacity: u32,
    frame: u32,
}

impl ParticleConeShading {
    /// Shade up to `capacity` particles of `positions` (and `velocities`, for speed emission;
    /// `emission` is the voxelizer's `ParticleEmission` uniform) from `volume`, escaping to `sky`
    /// (a `SkyLighting` uniform).
    pub fn new(
        device: &wgpu::Device,
        volume: &VoxelVolume,
        positions: &wgpu::Buffer,
        velocities: Option<&wgpu::Buffer>,
        emission: &wgpu::Buffer,
        sky: &wgpu::Buffer,
        capacity: u32,
    ) -> Self {
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VoxelGI/ConeParams"),
            size: std::mem::size_of::<ConeParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lighting = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VoxelGI/ParticleLighting"),
            size: capacity.max(1) as u64 * PARTICLE_LIGHTING_STRIDE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let compute = wgpu::ShaderStages::COMPUTE;
        let mut shade = Compute::new(
            "VoxelGI/ParticleCones",
            PARTICLE_CONES_WGSL,
            vec![
                Binding::uniform(0, compute),
                Binding::uniform(1, compute),
                Binding::uniform(2, compute),
                Binding::uniform(3, compute),
                Binding::texture_3d(4, compute),
                Binding::sampler(5, compute),
                Binding::storage(6, compute, true),
                Binding::storage(7, compute, true),
                Binding::storage(8, compute, false),
            ],
        );
        shade.initialize(device);
        let mut shading = Self { params, lighting, shade, capacity, frame: 0 };
        shading.bind(device, volume, positions, velocities, emission, sky);
        shading
    }

    /// Rebind the inputs, for example another sky (`SkyAtmosphereBindings::sky_lighting`).
    pub fn bind(
        &mut self,
        device: &wgpu::Device,
        volume: &VoxelVolume,
        positions: &wgpu::Buffer,
        velocities: Option<&wgpu::Buffer>,
        emission: &wgpu::Buffer,
        sky: &wgpu::Buffer,
    ) {
        let buffer = |buffer| BindingResource::Buffer { buffer, offset: 0, size: None };
        self.shade.set_bind_group(
            device,
            &[
                (0, buffer(volume.uniform())),
                (1, buffer(&self.params)),
                (2, buffer(emission)),
                (3, buffer(sky)),
                (4, BindingResource::TextureView(volume.view())),
                (5, BindingResource::Sampler(volume.sampler())),
                (6, buffer(positions)),
                (7, buffer(velocities.unwrap_or(positions))),
                (8, buffer(&self.lighting)),
            ],
        );
    }

    /// `PARTICLE_LIGHTING_STRIDE` bytes per particle, in the particles' order.
    pub fn lighting_buffer(&self) -> &wgpu::Buffer {
        &self.lighting
    }

    /// The lighting buffer as instance attributes for `InstancedGeometry`: the incoming light and
    /// sun visibility at `shader_location`, the emission at `shader_location + 1`.
    pub fn lighting_instance_buffer(&self, shader_location: u32) -> ComputeBuffer {
        ComputeBuffer::from_external("VoxelGI/ParticleLighting", self.lighting.clone(), BufferType::Storage).with_vertex_layout(
            PARTICLE_LIGHTING_STRIDE,
            vec![
                InstanceAttribute { shader_location, offset: 0, format: VertexFormat::Float32x4 },
                InstanceAttribute { shader_location: shader_location + 1, offset: 16, format: VertexFormat::Float32x4 },
            ],
        )
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Start the running average over: the next frame takes its light as it is.
    pub fn reset_history(&mut self) {
        self.frame = 0;
    }

    /// Record the shading of the first `particle_count` particles (after the volume's mips are
    /// built).
    pub fn encode(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, particle_count: u32, settings: &ParticleConeSettings) {
        let particle_count = particle_count.min(self.capacity);
        let params = ConeParamsGpu {
            to_sun: glam::Vec3::from(settings.to_sun).normalize_or(glam::Vec3::Y).to_array(),
            sun_cone_tan: settings.sun_cone_tan,
            diffuse_cone_tan: settings.diffuse_cone_tan,
            start_voxels: settings.start_voxels,
            max_distance: settings.max_distance_m,
            temporal_blend: settings.temporal_blend.clamp(0.0, 1.0),
            particle_count,
            max_steps: settings.max_steps,
            frame: self.frame,
            jitter: settings.jitter_voxels,
            use_volume: settings.use_volume as u32,
            _pad: [0; 3],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        self.frame = self.frame.wrapping_add(1).max(1);
        if particle_count == 0 {
            return;
        }
        let stamp = crate::profiling::gpu_pass("VoxelGI/ParticleCones");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VoxelGI/ParticleCones"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
        self.shade.dispatch(&mut pass, particle_count.div_ceil(64), 1, 1);
    }
}
