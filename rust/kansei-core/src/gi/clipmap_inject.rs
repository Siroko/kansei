use bytemuck::{Pod, Zeroable};

use super::clipmap::{clipmap_entries, clipmap_layout_entries, VoxelClipmap};
use super::clipmap_voxelize::ClipmapVoxelizer;
use crate::shadows::compute_shadows::ComputeShadows;

pub(crate) const CLIPMAP_INJECT_WGSL: &str = concat!(
    include_str!("shaders/clipmap.wgsl"),
    include_str!("shaders/particle_emission.wgsl"),
    include_str!("../atmosphere/shaders/sky_lighting.wgsl"),
    include_str!("../shaders/spot_light_types.wgsl"),
    include_str!("../shaders/compute_shadows.wgsl"),
    include_str!("shaders/clipmap_inject.wgsl"),
);

/// Where a voxel clipmap's injection takes shadows from cones traced through the clipmap
/// (`ClipmapGiSettings::cone_shadows`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConeShadows {
    /// The shadow maps alone (and none where they don't reach).
    Off,
    /// Cones where no shadow map covers the voxel: a sun past its cascades, lights without a map.
    #[default]
    Fallback,
    /// Cones for every light.
    Always,
}

impl ConeShadows {
    pub fn name(self) -> &'static str {
        match self {
            ConeShadows::Off => "off",
            ConeShadows::Fallback => "fallback",
            ConeShadows::Always => "always",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "off" => Some(ConeShadows::Off),
            "fallback" => Some(ConeShadows::Fallback),
            "always" => Some(ConeShadows::Always),
            _ => None,
        }
    }
}

/// How a voxel clipmap's voxels are lit (`SceneVoxelClipmap::settings`); change it freely
/// between frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipmapGiSettings {
    /// false: the clipmap is left as it is (nothing is voxelized, moved or lit).
    pub enabled: bool,
    /// The share of last frame's indirect light each voxel bounces again (1 is physical).
    pub bounce: f32,
    /// The most steps each of a voxel's bounce and shadow cones takes.
    pub bounce_steps: u32,
    /// How much of the sky's light the bounce cones bring in where they leave the clipmap.
    pub sky_scale: f32,
    /// Scale of the surfaces' emission.
    pub emission_scale: f32,
    /// Voxels the shadow lookups move out along the normal.
    pub shadow_offset_voxels: f32,
    /// Shadows from cones through the clipmap, where no map reaches or always.
    pub cone_shadows: ConeShadows,
    /// Tangent of the shadow cones' half angle: wider is softer (and cheaper).
    pub cone_shadow_tan: f32,
    /// Levels lit each frame, in turn, the finest every frame (0: all of them every frame). The
    /// coarse levels change slowly; lighting fewer a frame saves most of the injection's cost.
    pub levels_per_frame: u32,
}

impl Default for ClipmapGiSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            bounce: 1.0,
            bounce_steps: 24,
            sky_scale: 1.0,
            emission_scale: 1.0,
            shadow_offset_voxels: 1.0,
            cone_shadows: ConeShadows::Fallback,
            cone_shadow_tan: 0.08,
            levels_per_frame: 0,
        }
    }
}

/// The WGSL `ClipInjectParams` (clipmap_inject.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ClipInjectParamsGpu {
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
    level: u32,
    cone_shadows: u32,
    shadow_tan: f32,
    _pad: [u32; 3],
}

/// The clipmap's voxelized surfaces into light, a level at a time (clipmap_inject.wgsl), from
/// the renderer's lights and shadow maps (`ComputeShadows`), last frame's clipmap (the bounce, the
/// cone shadows) and emission.
pub(crate) struct ClipmapInjection {
    pub(crate) shadows: ComputeShadows,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    /// Per level, and whether it bound the level's dynamic surfaces.
    groups: Vec<Option<(wgpu::BindGroup, bool)>>,
    /// A slot per level (`params_stride` apart), written once a frame.
    params: wgpu::Buffer,
    params_stride: u64,
    sky: wgpu::Buffer,
    no_surfaces: wgpu::Buffer,
}

impl ClipmapInjection {
    pub(crate) fn new(device: &wgpu::Device, clipmap: &VoxelClipmap, sky: &wgpu::Buffer) -> Self {
        let compute = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty, count: None };
        let storage = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None };
        let mut entries = vec![
            entry(0, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<ClipInjectParamsGpu>() as u64) }),
            entry(10, storage),
            entry(11, storage),
            entry(14, wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D3 }),
            entry(15, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
        ];
        entries.extend(ComputeShadows::layout_entries());
        entries.extend(clipmap_layout_entries(compute));
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("VoxelClipmap/InjectBGL"), entries: &entries });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("VoxelClipmap/Inject"), source: wgpu::ShaderSource::Wgsl(CLIPMAP_INJECT_WGSL.into()) });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("VoxelClipmap/Inject"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VoxelClipmap/Inject"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let levels = clipmap.layout().levels as u64;
        let params_stride = (std::mem::size_of::<ClipInjectParamsGpu>() as u64).next_multiple_of(device.limits().min_uniform_buffer_offset_alignment as u64);
        Self {
            shadows: ComputeShadows::new(),
            pipeline,
            bgl,
            groups: (0..levels).map(|_| None).collect(),
            params: device.create_buffer(&wgpu::BufferDescriptor { label: Some("VoxelClipmap/InjectParams"), size: params_stride * levels, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
            params_stride,
            sky: sky.clone(),
            no_surfaces: device.create_buffer(&wgpu::BufferDescriptor { label: Some("VoxelClipmap/NoSurfaces"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false }),
        }
    }

    /// Read the sky (past the clipmap) from `sky`, a `SkyLighting` uniform.
    pub(crate) fn set_sky(&mut self, sky: &wgpu::Buffer) {
        self.sky = sky.clone();
        self.groups.iter_mut().for_each(|g| *g = None);
    }

    /// Record lighting `levels` of `clipmap` (those with a window), each into the scratch level
    /// then copied over it, with `settings`; `has_dynamic(level)`: its dynamic surfaces hold
    /// renderables this frame.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, clipmap: &VoxelClipmap, voxelizer: &ClipmapVoxelizer, levels: &[u32], has_dynamic: &dyn Fn(u32) -> bool, settings: &super::ClipmapGiSettings) {
        if self.shadows.prepare(device, queue) {
            self.groups.iter_mut().for_each(|g| *g = None);
        }
        let layout = clipmap.layout();
        // every level's slot in one write (`queue.write_buffer` lands before the frame's work)
        let mut bytes = vec![0u8; (self.params_stride * layout.levels as u64) as usize];
        for level in 0..layout.levels {
            let params = ClipInjectParamsGpu {
                num_dir_lights: self.shadows.dir.len() as u32,
                num_point_lights: self.shadows.point.len() as u32,
                has_shadow_map: self.shadows.has_shadow_map() as u32,
                has_point_shadows: self.shadows.has_point_shadows() as u32,
                bounce: settings.bounce.max(0.0),
                sky_scale: settings.sky_scale.max(0.0),
                emission_scale: settings.emission_scale.max(0.0),
                shadow_offset: settings.shadow_offset_voxels,
                max_steps: settings.bounce_steps.max(1),
                has_dynamic: has_dynamic(level) as u32,
                level,
                cone_shadows: settings.cone_shadows as u32,
                shadow_tan: settings.cone_shadow_tan.max(1e-3),
                _pad: [0; 3],
            };
            let at = (level as u64 * self.params_stride) as usize;
            bytes[at..at + std::mem::size_of::<ClipInjectParamsGpu>()].copy_from_slice(bytemuck::bytes_of(&params));
        }
        queue.write_buffer(&self.params, 0, &bytes);
        let [w, h, d] = layout.dims;
        for &level in levels {
            if clipmap.origin(level).is_none() {
                continue;
            }
            let dynamic = voxelizer.dynamic_surfaces(level).filter(|_| has_dynamic(level));
            if self.groups[level as usize].as_ref().is_none_or(|(_, bound)| *bound != dynamic.is_some()) {
                let mut entries = vec![
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.params, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<ClipInjectParamsGpu>() as u64) }),
                    },
                    wgpu::BindGroupEntry { binding: 10, resource: voxelizer.static_surfaces(level).as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 11, resource: dynamic.unwrap_or(&self.no_surfaces).as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(clipmap.scratch_view()) },
                    wgpu::BindGroupEntry { binding: 15, resource: self.sky.as_entire_binding() },
                ];
                entries.extend(self.shadows.entries());
                entries.extend(clipmap_entries(clipmap));
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("VoxelClipmap/InjectBG"), layout: &self.bgl, entries: &entries });
                self.groups[level as usize] = Some((group, dynamic.is_some()));
            }
            {
                let stamp = crate::profiling::gpu_pass("VoxelClipmap/Inject");
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VoxelClipmap/Inject"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.groups[level as usize].as_ref().unwrap().0, &[(level as u64 * self.params_stride) as u32]);
                pass.dispatch_workgroups(w.div_ceil(4), h.div_ceil(4), d.div_ceil(4));
            }
            clipmap.copy_scratch_to(encoder, level);
        }
    }
}
