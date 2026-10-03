use bytemuck::{Pod, Zeroable};

use super::volume::{VoxelVolume, ACCUMULATOR_BYTES_PER_VOXEL};
use crate::materials::{Binding, BindingResource, Compute};

pub(crate) const SPLAT_WGSL: &str = concat!(
    include_str!("shaders/voxel_volume.wgsl"),
    include_str!("shaders/particle_emission.wgsl"),
    include_str!("shaders/particle_splat.wgsl"),
);
pub(crate) const RESOLVE_WGSL: &str = concat!(
    include_str!("shaders/voxel_volume.wgsl"),
    include_str!("../atmosphere/shaders/sky_lighting.wgsl"),
    include_str!("shaders/particle_resolve.wgsl"),
);

/// Most analytic boxes a `ParticleVoxelizer` holds.
pub const MAX_GI_BOXES: usize = 32;

/// The light particles emit (WGSL `ParticleEmission`): a share of them, picked by a hash of
/// their index, glows with `color`, and each adds `speed_color` per unit of speed. Shared by the
/// splat, which puts it in the volume, and the cone shading, which hands it to the particles'
/// material, so both agree on which particles glow.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct ParticleEmission {
    /// Scene radiance of a glowing particle.
    pub color: [f32; 3],
    /// The share of particles that glow, 0..1.
    pub share: f32,
    /// Scene radiance added per unit of speed (needs the particles' velocities).
    pub speed_color: [f32; 3],
    pub seed: u32,
}

impl Default for ParticleEmission {
    fn default() -> Self {
        Self { color: [0.0; 3], share: 0.0, speed_color: [0.0; 3], seed: 0x9e37_79b9 }
    }
}

/// An analytic box in the volume (WGSL `GiBox`): a wall, a container, a collider. It covers its
/// exact share of each voxel it overlaps and reflects the sun (`N.L`, no shadow) and the sky on
/// the face `normal` points out of, plus its own emission. Make it at least a voxel thick, or wide
/// cones see through it at coarse mips.
///
/// `albedo` and `emission` are the constants this producer voxelizes with, as a per-renderable
/// constant will be for meshes (an optional material entry point can later supply textured
/// albedo to the mesh voxelizer instead).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GiBox {
    pub min: [f32; 3],
    _pad0: f32,
    pub max: [f32; 3],
    _pad1: f32,
    pub albedo: [f32; 3],
    _pad2: f32,
    /// Scene radiance.
    pub emission: [f32; 3],
    _pad3: f32,
    pub normal: [f32; 3],
    _pad4: f32,
}

impl GiBox {
    pub fn new(min: [f32; 3], max: [f32; 3], albedo: [f32; 3], normal: [f32; 3]) -> Self {
        Self { min, max, albedo, normal, ..Zeroable::zeroed() }
    }

    pub fn with_emission(mut self, emission: [f32; 3]) -> Self {
        self.emission = emission;
        self
    }
}

/// How particles fill the volume, and how the boxes are lit (`ParticleVoxelizer::encode`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleSplatSettings {
    /// Density one particle adds (spread over the voxels it touches).
    pub density_per_particle: f32,
    /// Opacity per unit density across a voxel: `1 - exp(-extinction * density)`.
    pub extinction: f32,
    /// The particle's radius, metres: under a voxel the splat is trilinear (8 voxels), above it
    /// spreads over the footprint (up to 3 voxels out).
    pub particle_radius_m: f32,
    /// Direction toward the sun, and its illuminance (scene units), for the boxes.
    pub to_sun: [f32; 3],
    pub sun_illuminance: [f32; 3],
    /// How much of the sky's irradiance reaches the boxes.
    pub box_sky_scale: f32,
}

impl Default for ParticleSplatSettings {
    fn default() -> Self {
        Self {
            density_per_particle: 1.0,
            extinction: 0.5,
            particle_radius_m: 0.0,
            to_sun: [0.0, 1.0, 0.0],
            sun_illuminance: [3.0; 3],
            box_sky_scale: 1.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SplatParamsGpu {
    particle_count: u32,
    radius_voxels: f32,
    density_per_particle: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ResolveParamsGpu {
    to_sun: [f32; 3],
    extinction: f32,
    sun_illuminance: [f32; 3],
    box_count: u32,
    box_sky_scale: f32,
    _pad: [f32; 3],
}

/// Particles and analytic boxes into a `VoxelVolume`'s mip 0, miaumiau.cat/?p=1476's scatter
/// done with atomics:
/// - **splat**: one thread per particle adds its density and density-weighted emission into four
///   u32 accumulators per voxel, trilinearly (or over a smooth footprint for large particles),
///   with weights that sum to one, so moving a particle changes the volume continuously;
/// - **resolve**: turns density into opacity (Beer-Lambert), emission into its mean times that
///   opacity, adds the boxes and clears the accumulators.
pub struct ParticleVoxelizer {
    accumulators: wgpu::Buffer,
    splat_params: wgpu::Buffer,
    resolve_params: wgpu::Buffer,
    emission: wgpu::Buffer,
    boxes: wgpu::Buffer,
    box_count: u32,
    has_velocities: bool,
    splat: Compute,
    resolve: Compute,
    dims: [u32; 3],
    voxel_size: f32,
}

impl ParticleVoxelizer {
    /// Particles from `positions` (`array<vec4f>`, xyz in metres) and optionally `velocities`
    /// (for speed emission), into `volume`; `sky` is a `SkyLighting` uniform for the boxes.
    pub fn new(device: &wgpu::Device, volume: &VoxelVolume, positions: &wgpu::Buffer, velocities: Option<&wgpu::Buffer>, sky: &wgpu::Buffer) -> Self {
        let uniform = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
        };
        let accumulators = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VoxelGI/Accumulators"),
            size: volume.layout().voxel_count() * ACCUMULATOR_BYTES_PER_VOXEL,
            // (COPY_SRC: readable in tests)
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let splat_params = uniform("VoxelGI/SplatParams", std::mem::size_of::<SplatParamsGpu>() as u64);
        let resolve_params = uniform("VoxelGI/ResolveParams", std::mem::size_of::<ResolveParamsGpu>() as u64);
        let emission = uniform("VoxelGI/ParticleEmission", std::mem::size_of::<ParticleEmission>() as u64);
        let boxes = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VoxelGI/Boxes"),
            size: (MAX_GI_BOXES * std::mem::size_of::<GiBox>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let compute = wgpu::ShaderStages::COMPUTE;
        let mut splat = Compute::new(
            "VoxelGI/Splat",
            SPLAT_WGSL,
            vec![
                Binding::uniform(0, compute),
                Binding::uniform(1, compute),
                Binding::uniform(2, compute),
                Binding::storage(3, compute, true),
                Binding::storage(4, compute, true),
                Binding::storage(5, compute, false),
            ],
        );
        splat.initialize(device);
        let buffer = |buffer| BindingResource::Buffer { buffer, offset: 0, size: None };
        splat.set_bind_group(
            device,
            &[
                (0, buffer(volume.uniform())),
                (1, buffer(&splat_params)),
                (2, buffer(&emission)),
                (3, buffer(positions)),
                (4, buffer(velocities.unwrap_or(positions))),
                (5, buffer(&accumulators)),
            ],
        );
        let mut resolve = Compute::new(
            "VoxelGI/Resolve",
            RESOLVE_WGSL,
            vec![
                Binding::uniform(0, compute),
                Binding::uniform(1, compute),
                Binding::uniform(2, compute),
                Binding::storage(3, compute, false),
                Binding::storage(4, compute, true),
                Binding::storage_texture_3d(5, compute, wgpu::TextureFormat::Rgba16Float),
            ],
        );
        resolve.initialize(device);
        let mut voxelizer = Self {
            accumulators,
            splat_params,
            resolve_params,
            emission,
            boxes,
            box_count: 0,
            has_velocities: velocities.is_some(),
            splat,
            resolve,
            dims: volume.dims(),
            voxel_size: volume.voxel_size(),
        };
        voxelizer.set_sky(device, volume, sky);
        voxelizer
    }

    /// Read the sky's light for the boxes from `sky` (a `SkyLighting` uniform, such as
    /// `SkyAtmosphereBindings::sky_lighting`).
    pub fn set_sky(&mut self, device: &wgpu::Device, volume: &VoxelVolume, sky: &wgpu::Buffer) {
        let buffer = |buffer| BindingResource::Buffer { buffer, offset: 0, size: None };
        self.resolve.set_bind_group(
            device,
            &[
                (0, buffer(volume.uniform())),
                (1, buffer(&self.resolve_params)),
                (2, buffer(sky)),
                (3, buffer(&self.accumulators)),
                (4, buffer(&self.boxes)),
                (5, BindingResource::StorageTexture(volume.mip0_storage_view())),
            ],
        );
    }

    /// Replace the analytic boxes (at most `MAX_GI_BOXES`; the rest are dropped).
    pub fn set_boxes(&mut self, queue: &wgpu::Queue, boxes: &[GiBox]) {
        let boxes = &boxes[..boxes.len().min(MAX_GI_BOXES)];
        if !boxes.is_empty() {
            queue.write_buffer(&self.boxes, 0, bytemuck::cast_slice(boxes));
        }
        self.box_count = boxes.len() as u32;
    }

    /// Set what the particles emit (speed emission is ignored without velocities).
    pub fn set_emission(&self, queue: &wgpu::Queue, emission: ParticleEmission) {
        let emission = ParticleEmission { speed_color: if self.has_velocities { emission.speed_color } else { [0.0; 3] }, ..emission };
        queue.write_buffer(&self.emission, 0, bytemuck::bytes_of(&emission));
    }

    /// The `ParticleEmission` uniform, for passes that must agree with the splat on it.
    pub fn emission_buffer(&self) -> &wgpu::Buffer {
        &self.emission
    }

    /// Four u32 per voxel, x fastest: emission r, g, b (fixed point, 1/1024, of stored radiance
    /// times density) and density (fixed point, 1/4096). Zero outside a splat-to-resolve span.
    pub fn accumulators(&self) -> &wgpu::Buffer {
        &self.accumulators
    }

    fn write_params(&self, queue: &wgpu::Queue, particle_count: u32, settings: &ParticleSplatSettings) {
        let splat = SplatParamsGpu {
            particle_count,
            radius_voxels: settings.particle_radius_m / self.voxel_size,
            density_per_particle: settings.density_per_particle,
            _pad: 0.0,
        };
        queue.write_buffer(&self.splat_params, 0, bytemuck::bytes_of(&splat));
        let to_sun = glam::Vec3::from(settings.to_sun).normalize_or(glam::Vec3::Y).to_array();
        let resolve = ResolveParamsGpu {
            to_sun,
            extinction: settings.extinction,
            sun_illuminance: settings.sun_illuminance,
            box_count: self.box_count,
            box_sky_scale: settings.box_sky_scale,
            _pad: [0.0; 3],
        };
        queue.write_buffer(&self.resolve_params, 0, bytemuck::bytes_of(&resolve));
    }

    /// Record the splat of `particle_count` particles alone (the accumulators then hold them
    /// until `encode_resolve`).
    pub fn encode_splat(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, particle_count: u32, settings: &ParticleSplatSettings) {
        self.write_params(queue, particle_count, settings);
        if particle_count == 0 {
            return;
        }
        let stamp = crate::profiling::gpu_pass("VoxelGI/Splat");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VoxelGI/Splat"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
        self.splat.dispatch(&mut pass, particle_count.div_ceil(64), 1, 1);
    }

    /// Record the resolve into the volume's mip 0 (clearing the accumulators).
    pub fn encode_resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let [w, h, d] = self.dims;
        let stamp = crate::profiling::gpu_pass("VoxelGI/Resolve");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VoxelGI/Resolve"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
        self.resolve.dispatch(&mut pass, w.div_ceil(4), h.div_ceil(4), d.div_ceil(4));
    }

    /// Record the splat and the resolve: the volume's mip 0 then holds this frame's particles and
    /// boxes (build its mips next).
    pub fn encode(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, particle_count: u32, settings: &ParticleSplatSettings) {
        self.encode_splat(queue, encoder, particle_count, settings);
        self.encode_resolve(encoder);
    }
}
