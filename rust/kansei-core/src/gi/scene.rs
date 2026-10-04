use super::cones::gradient_sky_lighting;
use super::inject::{RadianceInjection, SceneGiSettings};
use super::voxelize::{MeshVoxelizer, SURFACE_WORDS_PER_VOXEL};
use super::volume::{VoxelGiQuality, VoxelVolume};
use crate::renderers::SharedLayouts;

/// What `Renderer::enable_voxel_gi` builds.
#[derive(Clone, Copy, Debug)]
pub struct SceneVoxelGiOptions {
    /// The tier asked for; it steps down to what the device holds within `budget_bytes` (see
    /// `VoxelGiQuality::fit_scene`).
    pub quality: VoxelGiQuality,
    /// The box the volume covers, metres: the room, or the part of the scene whose light
    /// bounces. Outside it there is no voxel GI.
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    /// The reference radiance is stored against (see `VoxelVolume`).
    pub radiance_scale: f32,
    /// Most bytes the volume and its static surfaces may take (0: no limit). 24 MiB keeps a
    /// phone at `Low`.
    pub budget_bytes: u64,
}

impl Default for SceneVoxelGiOptions {
    fn default() -> Self {
        Self { quality: VoxelGiQuality::Medium, bounds_min: [-1.0; 3], bounds_max: [1.0; 3], radiance_scale: 1.0, budget_bytes: 0 }
    }
}

/// Voxel GI for a scene's meshes (the renderer's, `Renderer::enable_voxel_gi`). Each frame,
/// after the shadow maps and before the GBuffer:
/// 1. the renderables with a `Renderable::gi` surface are drawn into voxels through their own
///    `vertex_main` (`MeshVoxelizer`): the static ones when they change, the dynamic ones every
///    frame;
/// 2. the voxels are lit into the volume's mip 0 by the renderer's lights through their shadow
///    maps, plus their emission and one more bounce of last frame's light (`settings`);
/// 3. the volume's mips are rebuilt.
///
/// Read it with `gi::VoxelGIEffect` (screen-space cones), or with `VOXEL_CONES_WGSL` from
/// any pass or material.
pub struct SceneVoxelGi {
    pub settings: SceneGiSettings,
    quality: VoxelGiQuality,
    volume: VoxelVolume,
    voxelizer: MeshVoxelizer,
    pub(crate) injection: RadianceInjection,
    sky: wgpu::Buffer,
}

impl SceneVoxelGi {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue, shared: &SharedLayouts, light_buf: &wgpu::Buffer, options: SceneVoxelGiOptions) -> Self {
        let quality = options.quality.fit_scene(&device.limits(), options.bounds_min, options.bounds_max, options.budget_bytes);
        let mut volume = VoxelVolume::new(device, options.bounds_min, options.bounds_max, quality.resolution(), options.radiance_scale);
        // walls show a cone the face it meets first, and stay opaque for it
        volume.set_anisotropic_mips(device, true);
        let voxelizer = MeshVoxelizer::new(device, queue, shared, light_buf, *volume.layout());
        use wgpu::util::DeviceExt;
        // no sky past the volume until one is set
        let sky = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VoxelGI/SceneSky"),
            contents: bytemuck::cast_slice(&gradient_sky_lighting([0.0; 3], [0.0; 3])),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let injection = RadianceInjection::new(device, &volume, &sky);
        let settings = SceneGiSettings { bounce_steps: quality.cone_steps() / 2, ..Default::default() };
        Self { settings, quality, volume, voxelizer, injection, sky }
    }

    /// The tier in use (the one asked for, or lower if the device could not hold it).
    pub fn quality(&self) -> VoxelGiQuality {
        self.quality
    }

    /// The volume of the scene's light, for its consumers.
    pub fn volume(&self) -> &VoxelVolume {
        &self.volume
    }

    pub fn voxelizer(&self) -> &MeshVoxelizer {
        &self.voxelizer
    }

    pub(crate) fn voxelizer_mut(&mut self) -> &mut MeshVoxelizer {
        &mut self.voxelizer
    }

    /// Voxelize the static renderables again next frame (after changing something the
    /// voxelizer can't see, such as a material's texture).
    pub fn invalidate(&mut self) {
        self.voxelizer.invalidate();
    }

    /// The sky past the volume, from `down` to `up` (scene radiance), for the voxels' bounce
    /// cones: see `gradient_sky_lighting`. Black until set. Ignored after `use_sky_lighting`.
    pub fn set_sky_gradient(&self, queue: &wgpu::Queue, up: [f32; 3], down: [f32; 3]) {
        queue.write_buffer(&self.sky, 0, bytemuck::cast_slice(&gradient_sky_lighting(up, down)));
    }

    /// Take the sky from `sky_lighting` (a `SkyLighting` uniform such as
    /// `SkyAtmosphereBindings::sky_lighting`) instead of the gradient.
    pub fn use_sky_lighting(&mut self, sky_lighting: &wgpu::Buffer) {
        self.injection.set_sky(sky_lighting);
    }

    /// Bytes on the GPU: the radiance with its mips (and anisotropic chains) and the surface
    /// buffers.
    pub fn memory_bytes(&self) -> u64 {
        let layout = self.volume.layout();
        let surfaces = 1 + self.voxelizer.dynamic_surfaces().is_some() as u64;
        layout.radiance_bytes() + self.volume.anisotropic_bytes() + surfaces * layout.voxel_count() * SURFACE_WORDS_PER_VOXEL * 4
    }

    /// Record the injection and the mips (after the voxelization).
    pub(crate) fn encode_lighting(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        self.injection.encode(device, queue, encoder, &self.volume, &self.voxelizer, &self.settings);
        self.volume.build_mips(encoder);
    }
}
