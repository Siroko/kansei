use bytemuck::{Pod, Zeroable};
use glam::{IVec3, UVec3, Vec3};

/// The most levels a `VoxelClipmap` has.
pub const MAX_CLIPMAP_LEVELS: usize = 6;

/// Where a voxel clipmap's levels lie: `levels` boxes of `dims` voxels each, centred on a point
/// that moves (the camera), the finest of voxels `voxel_size` metres wide and each next one twice
/// as coarse, so each covers twice the extent of the one before. A level's voxels are cells of a
/// fixed world lattice (voxel `c` spans `c * size .. (c + 1) * size`), and a level keeps a window
/// of `dims` of them, stored toroidally: voxel `c` in texel `c mod dims`. Moving the window moves
/// its origin by whole voxels and rewrites only the slabs that came into it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipmapLayout {
    pub levels: u32,
    /// Voxels of each level, multiples of 8.
    pub dims: [u32; 3],
    /// The finest level's voxels, metres.
    pub voxel_size: f32,
}

impl ClipmapLayout {
    /// `levels` (at most `MAX_CLIPMAP_LEVELS`) of `dims` voxels (each rounded up to a multiple of
    /// 8), the finest `voxel_size` metres.
    pub fn new(levels: u32, dims: [u32; 3], voxel_size: f32) -> Self {
        Self { levels: levels.clamp(1, MAX_CLIPMAP_LEVELS as u32), dims: dims.map(|d| d.max(8).next_multiple_of(8)), voxel_size: voxel_size.max(1e-4) }
    }

    /// Level `level`'s voxels, metres.
    pub fn level_voxel_size(&self, level: u32) -> f32 {
        self.voxel_size * (1u32 << level) as f32
    }

    /// Level `level`'s extent, metres.
    pub fn level_extent(&self, level: u32) -> Vec3 {
        UVec3::from(self.dims).as_vec3() * self.level_voxel_size(level)
    }

    /// Voxels of one level.
    pub fn voxel_count(&self) -> u64 {
        self.dims.iter().map(|&d| d as u64).product()
    }

    /// The radiance of every level, and the scratch level the injection writes into.
    pub fn radiance_bytes(&self) -> u64 {
        (self.levels as u64 + 1) * self.voxel_count() * 8
    }

    /// The origin (first voxel) of level `level`'s window centred on `eye`, a multiple of `snap`
    /// voxels.
    pub fn centred_origin(&self, level: u32, eye: Vec3, snap: u32) -> IVec3 {
        let snap = snap.max(1) as i32;
        let cell = (eye / self.level_voxel_size(level)).floor().as_ivec3();
        let half = UVec3::from(self.dims).as_ivec3() / 2;
        // the snapped voxel nearest the eye's
        let snapped = (cell.as_vec3() / snap as f32).round().as_ivec3() * snap;
        snapped - half
    }

    /// Where level `level`'s window at `origin` goes for an eye at `eye`: it stays while the eye is
    /// less than `snap` voxels from its centre along each axis, and otherwise moves by whole
    /// multiples of `snap` toward the eye (so an eye moving to and fro across a step does not move
    /// it back and forth).
    pub fn follow(&self, level: u32, origin: IVec3, eye: Vec3, snap: u32) -> IVec3 {
        let snap = snap.max(1) as i32;
        let size = self.level_voxel_size(level);
        let centre = origin.as_vec3() + UVec3::from(self.dims).as_vec3() * 0.5;
        let offset = eye / size - centre;
        // whole steps toward the eye, rounding toward zero
        let steps = (offset / snap as f32).trunc().as_ivec3();
        origin + steps * snap
    }

    /// The level whose voxels a footprint `diameter` metres wide reads, before the levels that
    /// don't hold the point: `log2(diameter / voxel_size)`, clamped to the levels.
    pub fn level_for(&self, diameter: f32) -> u32 {
        ((diameter / self.voxel_size).max(1.0).log2().floor() as u32).min(self.levels - 1)
    }
}

/// The WGSL `ClipLevel` (clipmap.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub(crate) struct ClipLevelGpu {
    pub origin: [i32; 3],
    pub valid: u32,
}

/// The WGSL `VoxelClipmap` (clipmap.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct VoxelClipmapGpu {
    pub dims: [u32; 3],
    pub level_count: u32,
    pub voxel_size: f32,
    pub radiance_scale: f32,
    pub _pad: [f32; 2],
    pub levels: [ClipLevelGpu; MAX_CLIPMAP_LEVELS],
}

/// A voxel clipmap of the scene's light (the outdoor counterpart of `VoxelVolume`, which covers
/// one fixed box): `ClipmapLayout::levels` nested windows around a moving point, each its own
/// `rgba16float` 3D texture (premultiplied radiance in rgb, opacity across one voxel in a, as in
/// `VoxelVolume`), stored toroidally and sampled with a repeating sampler, so a world position
/// maps to its texel without an offset (`CLIPMAP_WGSL`'s `clipSample`). The levels stand in for
/// a volume's mips: a cone reads the level whose voxels are as wide as it is, or the next coarser
/// one that holds the point.
///
/// Each level's window has an origin once it holds valid data (`origin`); readers ignore levels
/// without one. A producer (`SceneVoxelClipmap`) writes a level's texture through a scratch level
/// (`scratch_view`), copied over it, so a pass that writes a level can read every level.
pub struct VoxelClipmap {
    layout: ClipmapLayout,
    textures: Vec<wgpu::Texture>,
    views: Vec<wgpu::TextureView>,
    scratch: wgpu::Texture,
    scratch_view: wgpu::TextureView,
    /// Bound for the levels a layout lacks.
    missing: wgpu::TextureView,
    uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    gpu: VoxelClipmapGpu,
    written: Option<VoxelClipmapGpu>,
}

impl VoxelClipmap {
    pub fn new(device: &wgpu::Device, layout: ClipmapLayout, radiance_scale: f32) -> Self {
        let [w, h, d] = layout.dims;
        let texture = |label: &str, w: u32, h: u32, d: u32, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: d },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba16Float,
                usage,
                view_formats: &[],
            })
        };
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC;
        let textures: Vec<wgpu::Texture> = (0..layout.levels).map(|_| texture("VoxelClipmap/Level", w, h, d, usage)).collect();
        let views = textures.iter().map(|t| t.create_view(&Default::default())).collect();
        let scratch = texture("VoxelClipmap/Scratch", w, h, d, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
        let scratch_view = scratch.create_view(&Default::default());
        let missing = texture("VoxelClipmap/NoLevel", 1, 1, 1, wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default());
        let gpu = VoxelClipmapGpu {
            dims: layout.dims,
            level_count: layout.levels,
            voxel_size: layout.voxel_size,
            radiance_scale: radiance_scale.max(1e-6),
            _pad: [0.0; 2],
            levels: [ClipLevelGpu::default(); MAX_CLIPMAP_LEVELS],
        };
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VoxelClipmap/Params"),
            size: std::mem::size_of::<VoxelClipmapGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("VoxelClipmap/LinearRepeat"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self { layout, textures, views, scratch, scratch_view, missing, uniform, sampler, gpu, written: None }
    }

    pub fn layout(&self) -> &ClipmapLayout {
        &self.layout
    }

    /// Level `level`'s origin (its window's first voxel, in its lattice), once it holds data.
    pub fn origin(&self, level: u32) -> Option<IVec3> {
        let l = self.gpu.levels.get(level as usize)?;
        (l.valid != 0).then(|| IVec3::from(l.origin))
    }

    /// Set level `level`'s origin (None: no valid data). Written to the uniform by `upload`.
    pub(crate) fn set_origin(&mut self, level: u32, origin: Option<IVec3>) {
        self.gpu.levels[level as usize] = match origin {
            Some(o) => ClipLevelGpu { origin: o.to_array(), valid: 1 },
            None => ClipLevelGpu::default(),
        };
    }

    /// World box of level `level`'s window, if it holds data.
    pub fn level_bounds(&self, level: u32) -> Option<(Vec3, Vec3)> {
        let size = self.layout.level_voxel_size(level);
        self.origin(level).map(|o| (o.as_vec3() * size, (o + UVec3::from(self.layout.dims).as_ivec3()).as_vec3() * size))
    }

    pub fn radiance_scale(&self) -> f32 {
        self.gpu.radiance_scale
    }

    /// Change the reference radiance is stored against (as `VoxelVolume::set_radiance_scale`).
    pub fn set_radiance_scale(&mut self, scale: f32) {
        self.gpu.radiance_scale = scale.max(1e-6);
    }

    /// Write the uniform if it changed (once a frame, after the origins are set).
    pub(crate) fn upload(&mut self, queue: &wgpu::Queue) {
        if self.written != Some(self.gpu) {
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&self.gpu));
            self.written = Some(self.gpu);
        }
    }

    /// The WGSL `VoxelClipmap` uniform.
    pub fn uniform(&self) -> &wgpu::Buffer {
        &self.uniform
    }

    /// Trilinear, repeating: world positions wrap onto the toroidal levels.
    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// Level `level`'s texture.
    pub fn texture(&self, level: u32) -> &wgpu::Texture {
        &self.textures[level as usize]
    }

    /// Level `level`'s view (sampled, or written as storage by a producer).
    pub fn view(&self, level: u32) -> &wgpu::TextureView {
        &self.views[level as usize]
    }

    /// The `MAX_CLIPMAP_LEVELS` views `CLIPMAP_WGSL` binds (a 1-texel stand-in past the layout's
    /// levels).
    pub fn level_views(&self) -> [&wgpu::TextureView; MAX_CLIPMAP_LEVELS] {
        std::array::from_fn(|k| self.views.get(k).unwrap_or(&self.missing))
    }

    /// The scratch level a producer writes, then copies into a level (`copy_scratch_to`).
    pub(crate) fn scratch_view(&self) -> &wgpu::TextureView {
        &self.scratch_view
    }

    /// Record copying the scratch level over level `level`.
    pub(crate) fn copy_scratch_to(&self, encoder: &mut wgpu::CommandEncoder, level: u32) {
        let [width, height, depth_or_array_layers] = self.layout.dims;
        encoder.copy_texture_to_texture(self.scratch.as_image_copy(), self.textures[level as usize].as_image_copy(), wgpu::Extent3d { width, height, depth_or_array_layers });
    }

    /// The radiance of every level and the scratch level.
    pub fn memory_bytes(&self) -> u64 {
        self.layout.radiance_bytes()
    }
}

/// Bind group layout entries for `CLIPMAP_WGSL`'s group 0 bindings 50-57 (the uniform, the
/// levels, the sampler), visible to `visibility`.
pub(crate) fn clipmap_layout_entries(visibility: wgpu::ShaderStages) -> Vec<wgpu::BindGroupLayoutEntry> {
    let texture = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false };
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 50,
        visibility,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }];
    entries.extend((0..MAX_CLIPMAP_LEVELS as u32).map(|k| wgpu::BindGroupLayoutEntry { binding: 51 + k, visibility, ty: texture, count: None }));
    entries.push(wgpu::BindGroupLayoutEntry { binding: 57, visibility, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None });
    entries
}

/// The entries binding `clipmap` at `clipmap_layout_entries`' bindings.
pub(crate) fn clipmap_entries(clipmap: &VoxelClipmap) -> Vec<wgpu::BindGroupEntry<'_>> {
    let mut entries = vec![wgpu::BindGroupEntry { binding: 50, resource: clipmap.uniform().as_entire_binding() }];
    entries.extend(clipmap.level_views().into_iter().enumerate().map(|(k, view)| wgpu::BindGroupEntry { binding: 51 + k as u32, resource: wgpu::BindingResource::TextureView(view) }));
    entries.push(wgpu::BindGroupEntry { binding: 57, resource: wgpu::BindingResource::Sampler(clipmap.sampler()) });
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_double_and_centre_on_the_eye() {
        let layout = ClipmapLayout::new(4, [64, 30, 64], 0.5);
        assert_eq!(layout.dims, [64, 32, 64]);
        assert_eq!(layout.level_voxel_size(3), 4.0);
        assert_eq!(layout.level_extent(1), Vec3::new(64.0, 32.0, 64.0));
        let eye = Vec3::new(10.3, 1.7, -40.2);
        for level in 0..4 {
            let origin = layout.centred_origin(level, eye, 4);
            assert_eq!(origin % 4, IVec3::new(0, 0, 0), "snapped");
            let size = layout.level_voxel_size(level);
            let centre = (origin.as_vec3() + Vec3::new(32.0, 16.0, 32.0)) * size;
            // within half a snap of the eye
            assert!((centre - eye).abs().max_element() <= 2.0 * size + 1e-4, "level {level}: centre {centre} for eye {eye}");
        }
    }

    #[test]
    fn a_window_follows_in_whole_steps_without_flicker() {
        let layout = ClipmapLayout::new(2, [64, 64, 64], 1.0);
        let origin = layout.centred_origin(0, Vec3::ZERO, 4);
        assert_eq!(origin, IVec3::splat(-32));
        // less than a step from the centre: stays
        assert_eq!(layout.follow(0, origin, Vec3::new(3.9, -3.9, 0.0), 4), origin);
        // a step and a bit: one step
        assert_eq!(layout.follow(0, origin, Vec3::new(4.1, 0.0, -9.0), 4), origin + IVec3::new(4, 0, -8));
        // back to just under a step the other way from the new centre: stays
        let moved = origin + IVec3::new(4, 0, 0);
        assert_eq!(layout.follow(0, moved, Vec3::new(0.5, 0.0, 0.0), 4), moved);
    }

    #[test]
    fn a_footprint_picks_the_level_as_wide() {
        let layout = ClipmapLayout::new(5, [64; 3], 0.5);
        assert_eq!(layout.level_for(0.1), 0);
        assert_eq!(layout.level_for(0.99), 0);
        assert_eq!(layout.level_for(1.0), 1);
        assert_eq!(layout.level_for(3.9), 2);
        assert_eq!(layout.level_for(1e3), 4);
    }
}
