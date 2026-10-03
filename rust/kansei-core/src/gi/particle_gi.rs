use super::cones::{gradient_sky_lighting, ParticleConeSettings, ParticleConeShading};
use super::particles::{GiBox, ParticleEmission, ParticleSplatSettings, ParticleVoxelizer};
use super::volume::{VoxelGiQuality, VoxelVolume};
use crate::buffers::ComputeBuffer;

/// What `ParticleGi::new` builds.
#[derive(Clone, Copy, Debug)]
pub struct ParticleGiOptions {
    /// The tier asked for; `ParticleGi` steps down from it to what the device holds within
    /// `budget_bytes` (see `VoxelGiQuality::fit`).
    pub quality: VoxelGiQuality,
    /// The box the volume covers (the particles' container and its walls), metres.
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    /// Most particles shaded (the lighting buffer's size).
    pub capacity: u32,
    /// The reference radiance is stored against (see `VoxelVolume`).
    pub radiance_scale: f32,
    /// Most bytes the volume may take (0: no limit). 24 MiB keeps a phone at `Low`.
    pub budget_bytes: u64,
}

impl Default for ParticleGiOptions {
    fn default() -> Self {
        Self {
            quality: VoxelGiQuality::Medium,
            bounds_min: [-1.0; 3],
            bounds_max: [1.0; 3],
            capacity: 0,
            radiance_scale: 1.0,
            budget_bytes: 0,
        }
    }
}

/// Everything `ParticleGi` reads each frame; change it freely between frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleGiSettings {
    pub splat: ParticleSplatSettings,
    pub cones: ParticleConeSettings,
    pub emission: ParticleEmission,
}

impl ParticleGiSettings {
    /// Point the sun (direction toward it, and its illuminance) for both the boxes and the cones.
    pub fn set_sun(&mut self, to_sun: [f32; 3], illuminance: [f32; 3]) {
        self.splat.to_sun = to_sun;
        self.splat.sun_illuminance = illuminance;
        self.cones.to_sun = to_sun;
    }
}

impl Default for ParticleGiSettings {
    fn default() -> Self {
        Self { splat: ParticleSplatSettings::default(), cones: ParticleConeSettings::default(), emission: ParticleEmission::default() }
    }
}

/// Particles that light, occlude and shadow each other through a voxel volume (miaumiau.cat's
/// "indirect lighting on particles", p=1476, on WebGPU compute). Each frame, in one encoder:
/// 1. the particles splat their density and emission into the volume, and analytic boxes (the
///    room's walls) add their lit surfaces (`ParticleVoxelizer`);
/// 2. the volume's mips are rebuilt (`Mip3d`);
/// 3. each particle cone traces its incoming light and its sun visibility (`ParticleConeShading`),
///    which its material reads from `lighting_instance_buffer`.
///
/// ```ignore
/// let mut gi = ParticleGi::new(device, ParticleGiOptions { bounds_min, bounds_max, capacity, ..Default::default() },
///     sim.positions_buffer().unwrap(), sim.velocities_buffer());
/// gi.set_boxes(queue, &walls);
/// // InstancedGeometry::new(billboard, capacity, vec![positions, gi.lighting_instance_buffer(4)])
/// // each frame, after the simulation step:
/// gi.encode(queue, &mut encoder, sim.particle_count());
/// ```
pub struct ParticleGi {
    pub settings: ParticleGiSettings,
    quality: VoxelGiQuality,
    volume: VoxelVolume,
    voxelizer: ParticleVoxelizer,
    shading: ParticleConeShading,
    sky: wgpu::Buffer,
}

impl ParticleGi {
    /// Particles from `positions` (`array<vec4f>`) and optionally `velocities` (speed emission).
    pub fn new(device: &wgpu::Device, options: ParticleGiOptions, positions: &wgpu::Buffer, velocities: Option<&wgpu::Buffer>) -> Self {
        let quality = options.quality.fit(&device.limits(), options.bounds_min, options.bounds_max, options.budget_bytes);
        let volume = VoxelVolume::new(device, options.bounds_min, options.bounds_max, quality.resolution(), options.radiance_scale);
        use wgpu::util::DeviceExt;
        let sky = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VoxelGI/Sky"),
            contents: bytemuck::cast_slice(&gradient_sky_lighting([0.4, 0.5, 0.7], [0.1, 0.1, 0.1])),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let voxelizer = ParticleVoxelizer::new(device, &volume, positions, velocities, &sky);
        let shading = ParticleConeShading::new(device, &volume, positions, velocities, voxelizer.emission_buffer(), &sky, options.capacity);
        let mut settings = ParticleGiSettings::default();
        settings.cones.max_steps = quality.cone_steps();
        Self { settings, quality, volume, voxelizer, shading, sky }
    }

    /// The tier in use (the one asked for, or lower if the device could not hold it).
    pub fn quality(&self) -> VoxelGiQuality {
        self.quality
    }

    pub fn volume(&self) -> &VoxelVolume {
        &self.volume
    }

    pub fn voxelizer(&self) -> &ParticleVoxelizer {
        &self.voxelizer
    }

    pub fn shading(&self) -> &ParticleConeShading {
        &self.shading
    }

    /// Replace the analytic boxes (walls, containers): see `GiBox`.
    pub fn set_boxes(&mut self, queue: &wgpu::Queue, boxes: &[GiBox]) {
        self.voxelizer.set_boxes(queue, boxes);
    }

    /// The sky past the volume, from `down` to `up` (scene radiance): see `gradient_sky_lighting`.
    /// Ignored after `use_sky_lighting`.
    pub fn set_sky_gradient(&self, queue: &wgpu::Queue, up: [f32; 3], down: [f32; 3]) {
        queue.write_buffer(&self.sky, 0, bytemuck::cast_slice(&gradient_sky_lighting(up, down)));
    }

    /// Take the sky from `sky_lighting` (a `SkyLighting` uniform such as
    /// `SkyAtmosphereBindings::sky_lighting`) instead of the gradient.
    pub fn use_sky_lighting(&mut self, device: &wgpu::Device, sky_lighting: &wgpu::Buffer, positions: &wgpu::Buffer, velocities: Option<&wgpu::Buffer>) {
        self.voxelizer.set_sky(device, &self.volume, sky_lighting);
        self.shading.bind(device, &self.volume, positions, velocities, self.voxelizer.emission_buffer(), sky_lighting);
    }

    /// See `ParticleConeShading::lighting_instance_buffer`.
    pub fn lighting_instance_buffer(&self, shader_location: u32) -> ComputeBuffer {
        self.shading.lighting_instance_buffer(shader_location)
    }

    pub fn lighting_buffer(&self) -> &wgpu::Buffer {
        self.shading.lighting_buffer()
    }

    /// Start the particles' running averages over.
    pub fn reset_history(&mut self) {
        self.shading.reset_history();
    }

    /// Record a frame: splat and resolve, mips, then the particles' cones. With
    /// `settings.cones.use_volume` false only the last runs, giving the particles the sky and
    /// the sun unoccluded: voxel GI off at a fraction of the cost.
    pub fn encode(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, particle_count: u32) {
        self.voxelizer.set_emission(queue, self.settings.emission);
        if self.settings.cones.use_volume {
            self.voxelizer.encode(queue, encoder, particle_count, &self.settings.splat);
            self.volume.build_mips(encoder);
        }
        self.shading.encode(queue, encoder, particle_count, &self.settings.cones);
    }
}
