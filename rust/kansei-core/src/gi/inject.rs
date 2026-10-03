use bytemuck::{Pod, Zeroable};

use super::voxelize::MeshVoxelizer;
use super::VoxelVolume;
use crate::shadows::compute_shadows::ComputeShadows;

pub(crate) const INJECT_WGSL: &str = concat!(
    include_str!("shaders/voxel_volume.wgsl"),
    include_str!("shaders/voxel_cones.wgsl"),
    include_str!("shaders/voxel_irradiance.wgsl"),
    include_str!("shaders/particle_emission.wgsl"),
    include_str!("../atmosphere/shaders/sky_lighting.wgsl"),
    include_str!("../shaders/spot_light_types.wgsl"),
    include_str!("../shaders/compute_shadows.wgsl"),
    include_str!("shaders/inject.wgsl"),
);

/// How the scene's voxels are lit (`SceneVoxelGi::settings`); change it freely between frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneGiSettings {
    /// false: the volume is left as it is (nothing is voxelized or lit).
    pub enabled: bool,
    /// The share of last frame's indirect light each voxel bounces again: 1 is physical (the
    /// bounces add up over frames, each dimmed by the albedo), 0 keeps direct light only.
    pub bounce: f32,
    /// The most steps each of a voxel's bounce cones takes.
    pub bounce_steps: u32,
    /// How much of the sky's light the bounce cones bring in where they leave the volume.
    pub sky_scale: f32,
    /// Scale of the surfaces' emission.
    pub emission_scale: f32,
    /// Voxels the shadow lookups move out along the normal, so a surface voxel (up to half a
    /// voxel inside its surface) does not shadow itself.
    pub shadow_offset_voxels: f32,
}

impl Default for SceneGiSettings {
    fn default() -> Self {
        Self { enabled: true, bounce: 1.0, bounce_steps: 16, sky_scale: 1.0, emission_scale: 1.0, shadow_offset_voxels: 1.0 }
    }
}

/// The WGSL `InjectParams` (inject.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct InjectParamsGpu {
    num_dir_lights: u32,
    num_point_lights: u32,
    has_shadow_map: u32,
    has_point_shadows: u32,
    bounce: f32,
    sky_scale: f32,
    emission_scale: f32,
    shadow_offset: f32,
    max_steps: u32,
    has_dynamic: u32,
    _pad: [u32; 2],
}

/// The voxelized surfaces into light: the volume's mip 0 (inject.wgsl), from the renderer's
/// lights and shadow maps (`ComputeShadows`), last frame's mips (the bounce) and emission.
pub(crate) struct RadianceInjection {
    pub(crate) shadows: ComputeShadows,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    group: Option<wgpu::BindGroup>,
    params: wgpu::Buffer,
    previous: wgpu::TextureView,
    sky: wgpu::Buffer,
    /// A one-voxel stand-in for the dynamic surfaces before there are any.
    no_surfaces: wgpu::Buffer,
    /// Whether `group` binds the voxelizer's dynamic surfaces.
    bound_dynamic: bool,
}

impl RadianceInjection {
    pub(crate) fn new(device: &wgpu::Device, volume: &VoxelVolume, sky: &wgpu::Buffer) -> Self {
        let compute = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty, count: None };
        let uniform = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
        let storage = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None };
        let mut entries = vec![
            entry(0, uniform),
            entry(2, uniform),
            entry(10, storage),
            entry(11, storage),
            entry(12, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false }),
            entry(13, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
            entry(14, wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D3 }),
            entry(15, uniform),
        ];
        entries.extend(ComputeShadows::layout_entries());
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("VoxelGI/InjectBGL"), entries: &entries });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("VoxelGI/Inject"), source: wgpu::ShaderSource::Wgsl(INJECT_WGSL.into()) });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("VoxelGI/Inject"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VoxelGI/Inject"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let previous = volume.texture().create_view(&wgpu::TextureViewDescriptor {
            label: Some("VoxelGI/PreviousMips"),
            base_mip_level: 1.min(volume.mip_count() - 1),
            mip_level_count: Some((volume.mip_count() - 1).max(1)),
            ..Default::default()
        });
        Self {
            shadows: ComputeShadows::new(),
            pipeline,
            bgl,
            group: None,
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("VoxelGI/InjectParams"),
                size: std::mem::size_of::<InjectParamsGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            previous,
            sky: sky.clone(),
            no_surfaces: device.create_buffer(&wgpu::BufferDescriptor { label: Some("VoxelGI/NoSurfaces"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false }),
            bound_dynamic: false,
        }
    }

    /// Read the sky (past the volume) from `sky`, a `SkyLighting` uniform.
    pub(crate) fn set_sky(&mut self, sky: &wgpu::Buffer) {
        self.sky = sky.clone();
        self.group = None;
    }

    /// Record the injection into `volume`'s mip 0 (build its mips next).
    pub(crate) fn encode(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, volume: &VoxelVolume, voxelizer: &MeshVoxelizer, settings: &SceneGiSettings) {
        let has_dynamic = voxelizer.dynamic_surfaces().is_some();
        if self.shadows.prepare(device, queue) || has_dynamic != self.bound_dynamic {
            self.group = None;
        }
        if self.group.is_none() {
            let mut entries = vec![
                wgpu::BindGroupEntry { binding: 0, resource: volume.uniform().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: voxelizer.static_surfaces().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: voxelizer.dynamic_surfaces().unwrap_or(&self.no_surfaces).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(&self.previous) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::Sampler(volume.sampler()) },
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(volume.mip0_storage_view()) },
                wgpu::BindGroupEntry { binding: 15, resource: self.sky.as_entire_binding() },
            ];
            entries.extend(self.shadows.entries());
            self.group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("VoxelGI/InjectBG"), layout: &self.bgl, entries: &entries }));
            self.bound_dynamic = has_dynamic;
        }
        let params = InjectParamsGpu {
            num_dir_lights: self.shadows.dir.len() as u32,
            num_point_lights: self.shadows.point.len() as u32,
            has_shadow_map: self.shadows.has_shadow_map() as u32,
            has_point_shadows: self.shadows.has_point_shadows() as u32,
            bounce: settings.bounce.max(0.0),
            sky_scale: settings.sky_scale.max(0.0),
            emission_scale: settings.emission_scale.max(0.0),
            shadow_offset: settings.shadow_offset_voxels,
            max_steps: settings.bounce_steps.max(1),
            has_dynamic: has_dynamic as u32,
            _pad: [0; 2],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let [w, h, d] = volume.dims();
        let stamp = crate::profiling::gpu_pass("VoxelGI/Inject");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VoxelGI/Inject"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.group.as_ref(), &[]);
        pass.dispatch_workgroups(w.div_ceil(4), h.div_ceil(4), d.div_ceil(4));
    }
}
