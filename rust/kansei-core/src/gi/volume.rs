use bytemuck::{Pod, Zeroable};

use crate::buffers::Texture;

const MIP3D_WGSL: &str = include_str!("shaders/mip3d.wgsl");

/// Resolution and cost tiers of voxel GI. Each sets the voxels across the volume's longest axis
/// and the steps a cone may take; `fit` steps down to a tier the device can hold, so a phone
/// degrades to `Low` instead of failing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoxelGiQuality {
    /// 64 voxels across: about 7 MiB for a cube (radiance with mips and accumulators), within a
    /// phone's budget of 24 MiB.
    Low,
    /// 96 voxels across: about 21 MiB for a cube.
    Medium,
    /// 128 voxels across: about 50 MiB for a cube.
    High,
}

impl VoxelGiQuality {
    /// Voxels across the volume's longest axis.
    pub fn resolution(self) -> u32 {
        match self {
            VoxelGiQuality::Low => 64,
            VoxelGiQuality::Medium => 96,
            VoxelGiQuality::High => 128,
        }
    }

    /// The most steps one cone takes (wide cones need far fewer: each step doubles in length).
    pub fn cone_steps(self) -> u32 {
        match self {
            VoxelGiQuality::Low => 32,
            VoxelGiQuality::Medium => 48,
            VoxelGiQuality::High => 64,
        }
    }

    /// The tier below, if any.
    pub fn lower(self) -> Option<Self> {
        match self {
            VoxelGiQuality::Low => None,
            VoxelGiQuality::Medium => Some(VoxelGiQuality::Low),
            VoxelGiQuality::High => Some(VoxelGiQuality::Medium),
        }
    }

    /// `low`, `medium` or `high`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "low" => Some(VoxelGiQuality::Low),
            "medium" => Some(VoxelGiQuality::Medium),
            "high" => Some(VoxelGiQuality::High),
            _ => None,
        }
    }

    /// This tier, or the highest one below it whose volume over `bounds` fits the device's
    /// limits (its accumulators in one storage binding, its sides in a 3D texture) and
    /// `budget_bytes` (0: no budget). `Low` is returned when nothing fits.
    pub fn fit(self, limits: &wgpu::Limits, bounds_min: [f32; 3], bounds_max: [f32; 3], budget_bytes: u64) -> Self {
        self.fit_with(limits, bounds_min, bounds_max, budget_bytes, ACCUMULATOR_BYTES_PER_VOXEL, false)
    }

    /// `fit` for a scene's meshes (`SceneVoxelGi`): the radiance with its anisotropic mips and the
    /// static surface buffer (`gi::SURFACE_WORDS_PER_VOXEL` u32 a voxel) instead of particle
    /// accumulators.
    pub fn fit_scene(self, limits: &wgpu::Limits, bounds_min: [f32; 3], bounds_max: [f32; 3], budget_bytes: u64) -> Self {
        self.fit_with(limits, bounds_min, bounds_max, budget_bytes, super::voxelize::SURFACE_WORDS_PER_VOXEL * 4, true)
    }

    /// `fit` with `bytes_per_voxel` in one storage binding next to the radiance (and its
    /// anisotropic chains).
    fn fit_with(self, limits: &wgpu::Limits, bounds_min: [f32; 3], bounds_max: [f32; 3], budget_bytes: u64, bytes_per_voxel: u64, anisotropic: bool) -> Self {
        let mut q = self;
        loop {
            let layout = VolumeLayout::new(bounds_min, bounds_max, q.resolution());
            let voxels = layout.voxel_count();
            let fits = voxels * bytes_per_voxel <= limits.max_storage_buffer_binding_size as u64
                && layout.dims.iter().all(|&d| d <= limits.max_texture_dimension_3d)
                && (budget_bytes == 0 || layout.radiance_bytes() + anisotropic as u64 * layout.anisotropic_bytes() + voxels * bytes_per_voxel <= budget_bytes);
            match (fits, q.lower()) {
                (false, Some(lower)) => q = lower,
                _ => return q,
            }
        }
    }
}

/// Bytes of the particle accumulators per voxel (four u32).
pub(crate) const ACCUMULATOR_BYTES_PER_VOXEL: u64 = 16;

/// Where a volume's voxels lie: cubic voxels over a box, `resolution` across its longest axis,
/// each side rounded up to a multiple of 8 (three mips then halve exactly) and centred on the box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeLayout {
    /// World position of voxel (0, 0, 0)'s corner.
    pub origin: [f32; 3],
    pub voxel_size: f32,
    pub dims: [u32; 3],
}

impl VolumeLayout {
    pub fn new(bounds_min: [f32; 3], bounds_max: [f32; 3], resolution: u32) -> Self {
        let extent: [f32; 3] = std::array::from_fn(|i| (bounds_max[i] - bounds_min[i]).abs().max(1e-3));
        let voxel_size = extent[0].max(extent[1]).max(extent[2]) / resolution.max(1) as f32;
        let dims: [u32; 3] = std::array::from_fn(|i| ((extent[i] / voxel_size).ceil() as u32).max(1).next_multiple_of(8));
        let origin = std::array::from_fn(|i| (bounds_min[i] + bounds_max[i]) * 0.5 - dims[i] as f32 * voxel_size * 0.5);
        Self { origin, voxel_size, dims }
    }

    pub fn voxel_count(&self) -> u64 {
        self.dims.iter().map(|&d| d as u64).product()
    }

    /// The mips of the volume's chain: down to one voxel on its longest side.
    pub fn mip_count(&self) -> u32 {
        Texture::full_mip_count(self.dims[0], self.dims[1], self.dims[2])
    }

    /// The radiance texture with all its mips plus the particle accumulators.
    pub fn memory_bytes(&self) -> u64 {
        self.radiance_bytes() + self.voxel_count() * ACCUMULATOR_BYTES_PER_VOXEL
    }

    /// The six anisotropic chains (`VoxelVolume::set_anisotropic_mips`): each as the volume's
    /// mips 1.. are.
    pub fn anisotropic_bytes(&self) -> u64 {
        6 * (self.radiance_bytes() - self.voxel_count() * 8)
    }

    /// The radiance texture with all its mips.
    pub fn radiance_bytes(&self) -> u64 {
        (0..self.mip_count()).map(|l| self.dims.iter().map(|&d| (d >> l).max(1) as u64).product::<u64>() * 8).sum()
    }
}

/// The WGSL `VoxelVolume` (voxel_volume.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct VoxelVolumeGpu {
    pub origin: [f32; 3],
    pub voxel_size: f32,
    pub dims: [u32; 3],
    pub mip_count: u32,
    pub inv_extent: [f32; 3],
    pub radiance_scale: f32,
}

/// A voxel volume of the scene's light, the core every voxel-GI producer writes and every
/// consumer reads (`VOXEL_VOLUME_WGSL`):
/// - an `rgba16float` 3D texture whose rgb is the radiance leaving each voxel, premultiplied by
///   its coverage, and a its opacity across one voxel, so mips are a plain 2x2x2 average
///   (`Mip3d`), rebuilt by `build_mips` after mip 0 is written;
/// - its placement and scale as a uniform (`VoxelVolume` in WGSL);
/// - a trilinear, clamping sampler for cone tracing (`VOXEL_CONES_WGSL`).
///
/// Radiance is stored divided by `radiance_scale`, a reference such as the scene's exposure or
/// its sun's illuminance, so that sums of bright emitters keep to f16 and to the producers'
/// fixed point; readers multiply it back (`voxelConeTrace` does).
pub struct VoxelVolume {
    layout: VolumeLayout,
    texture: Texture,
    view: wgpu::TextureView,
    mip0_storage: wgpu::TextureView,
    mips: Mip3d,
    anisotropic: Option<super::aniso::AnisotropicMips>,
    uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    gpu: VoxelVolumeGpu,
}

impl VoxelVolume {
    /// A volume over `bounds_min..bounds_max`, `resolution` voxels across its longest axis.
    pub fn new(device: &wgpu::Device, bounds_min: [f32; 3], bounds_max: [f32; 3], resolution: u32, radiance_scale: f32) -> Self {
        let layout = VolumeLayout::new(bounds_min, bounds_max, resolution);
        let [w, h, d] = layout.dims;
        let mut texture = Texture::new_3d(
            "VoxelVolume/Radiance",
            w,
            h,
            d,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
        )
        .with_mip_levels(layout.mip_count());
        texture.initialize(device);
        let view = texture.view().unwrap().clone();
        let mip0_storage = texture.mip_view(0).unwrap();
        let mips = Mip3d::new(device, texture.gpu_texture().unwrap());
        let gpu = VoxelVolumeGpu {
            origin: layout.origin,
            voxel_size: layout.voxel_size,
            dims: layout.dims,
            mip_count: layout.mip_count(),
            inv_extent: std::array::from_fn(|i| 1.0 / (layout.dims[i] as f32 * layout.voxel_size)),
            radiance_scale: radiance_scale.max(1e-6),
        };
        use wgpu::util::DeviceExt;
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VoxelVolume/Params"),
            contents: bytemuck::bytes_of(&gpu),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("VoxelVolume/LinearClamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self { layout, texture, view, mip0_storage, mips, anisotropic: None, uniform, sampler, gpu }
    }

    pub fn layout(&self) -> &VolumeLayout {
        &self.layout
    }

    pub fn dims(&self) -> [u32; 3] {
        self.layout.dims
    }

    pub fn voxel_size(&self) -> f32 {
        self.layout.voxel_size
    }

    pub fn origin(&self) -> [f32; 3] {
        self.layout.origin
    }

    pub fn mip_count(&self) -> u32 {
        self.gpu.mip_count
    }

    pub fn radiance_scale(&self) -> f32 {
        self.gpu.radiance_scale
    }

    /// Change the reference radiance is stored against (takes effect from the next producer
    /// pass: last frame's content stays in the old scale until rewritten).
    pub fn set_radiance_scale(&mut self, queue: &wgpu::Queue, scale: f32) {
        self.gpu.radiance_scale = scale.max(1e-6);
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&self.gpu));
    }

    /// The radiance texture (all mips).
    pub fn texture(&self) -> &wgpu::Texture {
        self.texture.gpu_texture().unwrap()
    }

    /// Every mip, to bind as `texture_3d<f32>` and cone trace with `sampler()`.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Mip 0 alone, for producers to write as `texture_storage_3d<rgba16float, write>`.
    pub fn mip0_storage_view(&self) -> &wgpu::TextureView {
        &self.mip0_storage
    }

    /// The WGSL `VoxelVolume` uniform.
    pub fn uniform(&self) -> &wgpu::Buffer {
        &self.uniform
    }

    /// Trilinear across voxels and mips, clamping at the volume's sides.
    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// The volume as a `Texture` (shares the GPU texture), to attach to a material that cone
    /// traces it: bind with `Binding::texture_3d`.
    pub fn as_texture(&self) -> Texture {
        Texture::from_view("VoxelVolume/Radiance", self.texture().clone(), self.view.clone())
    }

    /// Also build anisotropic mips (Crassin et al. 2011): from mip 1 up, six directional chains
    /// whose voxels composite their children front to back along each axis direction, so a cone
    /// meets the face of a wall it reaches first instead of the mean of both faces (which halves
    /// a lit room's walls at coarse mips). `build_mips` rebuilds them; cones read them through
    /// `anisotropic_views` (the scene GI's do). About 0.86 times the memory of mip 0 more.
    pub fn set_anisotropic_mips(&mut self, device: &wgpu::Device, anisotropic: bool) {
        self.anisotropic = anisotropic.then(|| super::aniso::AnisotropicMips::new(device, self.texture.gpu_texture().unwrap()));
    }

    /// The anisotropic chains (`set_anisotropic_mips`), in the order +x, +y, +z, -x, -y, -z of
    /// the direction a cone travels; their level 0 is the volume's mip 1.
    pub fn anisotropic_views(&self) -> Option<&[wgpu::TextureView; 6]> {
        self.anisotropic.as_ref().map(|a| a.views())
    }

    /// Record the rebuild of mips 1.. (and the anisotropic chains) from mip 0.
    pub fn build_mips(&self, encoder: &mut wgpu::CommandEncoder) {
        self.mips.encode(encoder);
        if let Some(anisotropic) = &self.anisotropic {
            anisotropic.encode(encoder);
        }
    }

    /// Bytes of the anisotropic chains (0 without them).
    pub fn anisotropic_bytes(&self) -> u64 {
        self.anisotropic.as_ref().map_or(0, |a| a.memory_bytes())
    }

    /// The radiance texture with all its mips plus the particle accumulators.
    pub fn memory_bytes(&self) -> u64 {
        self.layout.memory_bytes()
    }
}

/// Builds a 3D texture's mip chain on the GPU (`rgba16float`, with `TEXTURE_BINDING` and
/// `STORAGE_BINDING`): one compute pass, one dispatch per level, each a 2x2x2 box filter of the
/// level before. WebGPU has no mip generation, and Kansei had none for 3D textures.
pub struct Mip3d {
    pipeline: wgpu::ComputePipeline,
    // per level 1..: its bind group (reading the level before) and its size
    levels: Vec<(wgpu::BindGroup, [u32; 3])>,
}

impl Mip3d {
    pub fn new(device: &wgpu::Device, texture: &wgpu::Texture) -> Self {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mip3d/BGL"),
            entries: &[
                entry(0, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false }),
                entry(1, wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D3 }),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Mip3d"), source: wgpu::ShaderSource::Wgsl(MIP3D_WGSL.into()) });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("Mip3d"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mip3d"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let level_view = |level: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor { label: Some("Mip3d/Level"), base_mip_level: level, mip_level_count: Some(1), ..Default::default() })
        };
        let size = texture.size();
        let levels = (1..texture.mip_level_count())
            .map(|level| {
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Mip3d/Level"),
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&level_view(level - 1)) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&level_view(level)) },
                    ],
                });
                let dims = [size.width, size.height, size.depth_or_array_layers].map(|d| (d >> level).max(1));
                (bind_group, dims)
            })
            .collect();
        Self { pipeline, levels }
    }

    /// Record the chain's rebuild from level 0.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.levels.is_empty() {
            return;
        }
        let stamp = crate::profiling::gpu_pass("VoxelGI/Mips");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VoxelGI/Mips"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
        pass.set_pipeline(&self.pipeline);
        for (bind_group, [w, h, d]) in &self.levels {
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(w.div_ceil(4), h.div_ceil(4), d.div_ceil(4));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layout_keeps_voxels_cubic_and_covers_the_bounds() {
        let layout = VolumeLayout::new([-2.0, 0.0, -1.0], [2.0, 3.0, 1.0], 64);
        assert_eq!(layout.voxel_size, 4.0 / 64.0);
        assert_eq!(layout.dims, [64, 48, 32]);
        for i in 0..3 {
            let lo = [-2.0, 0.0, -1.0][i];
            let hi = [2.0, 3.0, 1.0][i];
            assert!(layout.origin[i] <= lo + 1e-5 && layout.origin[i] + layout.dims[i] as f32 * layout.voxel_size >= hi - 1e-5);
        }
        assert_eq!(layout.mip_count(), 7);
    }

    #[test]
    fn low_fits_a_phone_budget_and_fit_steps_down() {
        let cube = VolumeLayout::new([0.0; 3], [1.0; 3], VoxelGiQuality::Low.resolution());
        assert!(cube.memory_bytes() <= 24 << 20, "{} bytes", cube.memory_bytes());
        let limits = wgpu::Limits::default();
        assert_eq!(VoxelGiQuality::High.fit(&limits, [0.0; 3], [1.0; 3], 0), VoxelGiQuality::High);
        assert_eq!(VoxelGiQuality::High.fit(&limits, [0.0; 3], [1.0; 3], 24 << 20), VoxelGiQuality::Medium);
        let small = wgpu::Limits { max_storage_buffer_binding_size: 8 << 20, ..limits };
        assert_eq!(VoxelGiQuality::High.fit(&small, [0.0; 3], [1.0; 3], 0), VoxelGiQuality::Low);
    }
}
