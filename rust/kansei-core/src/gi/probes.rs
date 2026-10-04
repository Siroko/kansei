use bytemuck::{Pod, Zeroable};

use super::sdf::JumpFloodSdf;
use super::voxelize::MeshVoxelizer;
use super::volume::VoxelVolume;

pub(crate) const PROBE_UPDATE_WGSL: &str = concat!(
    include_str!("shaders/voxel_volume.wgsl"),
    include_str!("shaders/voxel_cones.wgsl"),
    include_str!("shaders/voxel_irradiance.wgsl"),
    include_str!("../atmosphere/shaders/sky_lighting.wgsl"),
    include_str!("shaders/probe_common.wgsl"),
    include_str!("shaders/probe_update.wgsl"),
);

/// Rays a probe traces each update (one workgroup).
pub const PROBE_RAYS: u32 = 64;
const SH_WORDS: u64 = 9;
const DEPTH_TEXELS: u64 = 64;

/// What `SceneVoxelGi::enable_probes` sets up, and how the probes update (`SdfProbes::options`,
/// change it between frames; `dims` and `spacing_voxels` take effect only at creation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfProbeOptions {
    /// Voxels of the volume between probes.
    pub spacing_voxels: f32,
    /// The most probes per axis (0: as many as cover the volume). The grid follows the camera
    /// when it is smaller than the volume.
    pub max_probes_per_axis: u32,
    /// Probes updated each frame, in turn (0: all of them).
    pub probes_per_frame: u32,
    /// Weight of the history in each update of the irradiance (0.97: about 30 frames to settle).
    pub hysteresis: f32,
    /// The same for the depth moments.
    pub depth_hysteresis: f32,
    /// Sharpness of a ray's weight on the depth map's texels around its direction.
    pub depth_sharpness: f32,
    /// Voxels a probe is moved off the surfaces near its lattice point.
    pub min_clearance_voxels: f32,
    /// Voxels a lookup moves off its surface along the normal before weighing the probes.
    pub normal_bias_voxels: f32,
    /// A probe whose rays meet more back faces than this share (it sits inside geometry) is
    /// left out of the lookups.
    pub backface_limit: f32,
    /// Weigh the probes by whether they see the point (their depth moments): stops light leaking
    /// through walls.
    pub visibility: bool,
    /// Scale of the sky the rays that leave the volume see.
    pub sky_scale: f32,
    /// Most sphere-tracing steps per ray.
    pub max_steps: u32,
}

impl Default for SdfProbeOptions {
    fn default() -> Self {
        Self {
            spacing_voxels: 8.0,
            max_probes_per_axis: 0,
            probes_per_frame: 0,
            hysteresis: 0.97,
            depth_hysteresis: 0.97,
            depth_sharpness: 32.0,
            min_clearance_voxels: 2.0,
            normal_bias_voxels: 3.0,
            backface_limit: 0.25,
            visibility: true,
            sky_scale: 1.0,
            max_steps: 96,
        }
    }
}

/// The WGSL `ProbeGrid` (probe_common.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub(crate) struct ProbeGridGpu {
    origin: [f32; 3],
    spacing: f32,
    dims: [u32; 3],
    probe_count: u32,
    base: [i32; 3],
    normal_bias: f32,
    backface_limit: f32,
    max_depth: f32,
    visibility: u32,
    _pad: u32,
}

/// The WGSL `ProbeUpdate` (probe_update.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ProbeUpdateGpu {
    rotation: [f32; 16],
    first_probe: u32,
    hysteresis: f32,
    depth_hysteresis: f32,
    sky_scale: f32,
    max_steps: u32,
    min_clearance: f32,
    depth_sharpness: f32,
    reset_all: u32,
    has_dynamic: u32,
    _pad: [u32; 3],
}

/// Irradiance probes traced in a voxel volume's distance field (the Lumen-like consumer of the
/// scene's voxel GI; `SceneVoxelGi::enable_probes`, which the renderer updates each frame after
/// the volume is lit). A grid of probes on a world lattice follows the camera (inside the volume:
/// by whole cells, each probe keeping its history while it stays in the grid). Each update, a
/// probe:
/// 1. moves off the surfaces near its lattice point, up the distance field's gradient;
/// 2. sphere-traces 64 rays (a spherical Fibonacci set turned at random each frame) through the
///    field, reading the lit volume where they hit (mip 0 over the anisotropic chain the ray faces:
///    no material or shadow lookups, the injection did those) and the sky where they leave it;
/// 3. projects their light onto order-2 SH convolved with the cosine lobe (its irradiance) and
///    their distances onto an 8x8 octahedral map of depth moments, both blended into its history;
/// 4. counts the rays that met the back of a surface: a probe inside geometry is left out.
///
/// Read it with `PROBES_WGSL`'s `kansei_gi_irradiance(p, n)` (materials, with `bindings_wgsl`
/// and `bind_group_entries`), or with `VoxelGIEffect::set_probes` (the far field on screen). The
/// lookups weigh the eight probes around a point as DDGI does (Majercik et al. 2019): trilinear,
/// by facing, and by Chebyshev visibility from the depth moments, so walls don't leak.
pub struct SdfProbes {
    pub options: SdfProbeOptions,
    dims: [u32; 3],
    spacing: f32,
    /// World position of lattice cell (0, 0, 0)'s probe.
    anchor: [f32; 3],
    /// Lattice cells the volume spans per axis.
    volume_cells: [i32; 3],
    base: [i32; 3],
    grid: wgpu::Buffer,
    grid_data: Option<ProbeGridGpu>,
    params: wgpu::Buffer,
    sh: wgpu::Buffer,
    state: wgpu::Buffer,
    depth: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    group: Option<wgpu::BindGroup>,
    /// (dynamic surfaces bound, sky buffer bound) of `group`
    bound: (bool, Option<wgpu::Buffer>),
    no_surfaces: wgpu::Buffer,
    cursor: u32,
    frame: u32,
    reset: bool,
}

impl SdfProbes {
    pub(crate) fn new(device: &wgpu::Device, volume: &VoxelVolume, options: SdfProbeOptions) -> Self {
        let layout = volume.layout();
        let spacing = layout.voxel_size * options.spacing_voxels.max(1.0);
        let extent = layout.dims.map(|d| d as f32 * layout.voxel_size);
        let volume_cells = extent.map(|e| ((e / spacing).floor() as i32).max(1));
        let cap = if options.max_probes_per_axis == 0 { u32::MAX } else { options.max_probes_per_axis.max(2) };
        // (one workgroup each, within a dispatch's 65535)
        let cap = cap.min(40);
        let dims = volume_cells.map(|c| (c as u32).clamp(2, cap));
        // the lattice is centred on the volume, so a grid that covers it is symmetric in it
        let anchor = std::array::from_fn(|i| layout.origin[i] + 0.5 * (extent[i] - (volume_cells[i] - 1) as f32 * spacing));
        let count = dims.iter().product::<u32>() as u64;
        let storage = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false })
        };
        let uniform = |label: &str, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
        };

        let compute = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty, count: None };
        let uniform_ty = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
        let storage_ty = |read_only| wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None };
        let texture_ty = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false };
        let mut entries = vec![
            entry(0, uniform_ty),
            entry(1, uniform_ty),
            entry(2, uniform_ty),
            entry(3, texture_ty),
            entry(4, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
            entry(5, uniform_ty),
            entry(6, storage_ty(true)),
            entry(7, storage_ty(true)),
            entry(8, storage_ty(false)),
            entry(9, storage_ty(false)),
            entry(10, storage_ty(false)),
        ];
        // the anisotropic chains and the distance field (voxel_irradiance.wgsl)
        entries.extend((40..47).map(|binding| entry(binding, texture_ty)));
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("VoxelGI/ProbesBGL"), entries: &entries });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("VoxelGI/Probes"), source: wgpu::ShaderSource::Wgsl(PROBE_UPDATE_WGSL.into()) });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("VoxelGI/Probes"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VoxelGI/Probes"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut probes = Self {
            options,
            dims,
            spacing,
            anchor,
            volume_cells,
            base: [0; 3],
            grid: uniform("VoxelGI/ProbeGrid", std::mem::size_of::<ProbeGridGpu>()),
            grid_data: None,
            params: uniform("VoxelGI/ProbeUpdate", std::mem::size_of::<ProbeUpdateGpu>()),
            sh: storage("VoxelGI/ProbeSh", count * SH_WORDS * 16),
            state: storage("VoxelGI/ProbeState", count * 2 * 16),
            depth: storage("VoxelGI/ProbeDepth", count * DEPTH_TEXELS * 8),
            pipeline,
            bgl,
            group: None,
            bound: (false, None),
            no_surfaces: device.create_buffer(&wgpu::BufferDescriptor { label: Some("VoxelGI/ProbeNoSurfaces"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false }),
            cursor: 0,
            frame: 0,
            reset: true,
        };
        probes.base = probes.base_for(None);
        probes
    }

    /// Probes per axis.
    pub fn dims(&self) -> [u32; 3] {
        self.dims
    }

    pub fn probe_count(&self) -> u32 {
        self.dims.iter().product()
    }

    /// Metres between probes.
    pub fn spacing(&self) -> f32 {
        self.spacing
    }

    /// World position of the grid's first probe now (before its offset).
    pub fn grid_origin(&self) -> [f32; 3] {
        std::array::from_fn(|i| self.anchor[i] + self.base[i] as f32 * self.spacing)
    }

    /// Bytes on the GPU.
    pub fn memory_bytes(&self) -> u64 {
        self.sh.size() + self.state.size() + self.depth.size()
    }

    /// Start every probe over next update (after a cut, or a change of the scene's lighting the
    /// hysteresis should not blend through).
    pub fn reset(&mut self) {
        self.reset = true;
    }

    /// The `ProbeGrid` uniform, for readers.
    pub fn grid_buffer(&self) -> &wgpu::Buffer {
        &self.grid
    }

    /// Each probe's irradiance SH: 9 `vec4f` (rgb).
    pub fn sh_buffer(&self) -> &wgpu::Buffer {
        &self.sh
    }

    /// Each probe's state: 2 `vec4f` (offset and back-face share; lattice cell and frames).
    pub fn state_buffer(&self) -> &wgpu::Buffer {
        &self.state
    }

    /// Each probe's depth moments: 64 `vec2f`.
    pub fn depth_buffer(&self) -> &wgpu::Buffer {
        &self.depth
    }

    /// The declarations `PROBES_WGSL` reads, at `group` and the four bindings from `first`:
    /// the grid, the SH, the state and the depth moments (`bind_group_entries` binds them).
    pub fn bindings_wgsl(group: u32, first: u32) -> String {
        format!(
            "@group({group}) @binding({}) var<uniform> kansei_probe_grid : ProbeGrid;\n\
             @group({group}) @binding({}) var<storage, read> kansei_probe_sh : array<vec4f>;\n\
             @group({group}) @binding({}) var<storage, read> kansei_probe_state : array<vec4f>;\n\
             @group({group}) @binding({}) var<storage, read> kansei_probe_depth : array<vec2f>;\n",
            first,
            first + 1,
            first + 2,
            first + 3
        )
    }

    /// The buffers for `bindings_wgsl(_, first)`, in order.
    pub fn bind_group_entries(&self, first: u32) -> [wgpu::BindGroupEntry<'_>; 4] {
        [
            wgpu::BindGroupEntry { binding: first, resource: self.grid.as_entire_binding() },
            wgpu::BindGroupEntry { binding: first + 1, resource: self.sh.as_entire_binding() },
            wgpu::BindGroupEntry { binding: first + 2, resource: self.state.as_entire_binding() },
            wgpu::BindGroupEntry { binding: first + 3, resource: self.depth.as_entire_binding() },
        ]
    }

    /// The lattice cell of the grid's first probe for a camera at `eye`: the grid centred on it,
    /// kept inside the volume (centred on the volume when it is larger).
    fn base_for(&self, eye: Option<[f32; 3]>) -> [i32; 3] {
        std::array::from_fn(|i| {
            let dims = self.dims[i] as i32;
            let room = self.volume_cells[i] - dims;
            if room <= 0 {
                return room / 2;
            }
            let center = eye.map_or(self.volume_cells[i] / 2, |e| ((e[i] - self.anchor[i]) / self.spacing).round() as i32);
            (center - dims / 2).clamp(0, room)
        })
    }

    fn grid_data(&self) -> ProbeGridGpu {
        let voxel = self.spacing / self.options.spacing_voxels.max(1.0);
        ProbeGridGpu {
            origin: self.grid_origin(),
            spacing: self.spacing,
            dims: self.dims,
            probe_count: self.probe_count(),
            base: self.base,
            normal_bias: self.options.normal_bias_voxels.max(0.0) * voxel,
            backface_limit: self.options.backface_limit,
            max_depth: 1.5 * self.spacing * 3f32.sqrt(),
            visibility: self.options.visibility as u32,
            _pad: 0,
        }
    }

    /// Record an update of the probes (after the volume's light and mips): the grid follows
    /// `eye`, then the next `probes_per_frame` probes trace.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        volume: &VoxelVolume,
        sdf: &JumpFloodSdf,
        voxelizer: &MeshVoxelizer,
        sky: &wgpu::Buffer,
        eye: Option<[f32; 3]>,
    ) {
        self.base = self.base_for(eye);
        let grid = self.grid_data();
        if self.grid_data != Some(grid) {
            queue.write_buffer(&self.grid, 0, bytemuck::bytes_of(&grid));
            self.grid_data = Some(grid);
        }
        let count = self.probe_count();
        let batch = if self.options.probes_per_frame == 0 { count } else { self.options.probes_per_frame.min(count) }.min(65535);
        // a random turn of the ray set each frame (a uniform quaternion, Shoemake 1992)
        let h = |k: u32| {
            let mut x = self.frame.wrapping_mul(0x9e37_79b9).wrapping_add(k.wrapping_mul(0x85eb_ca6b));
            x ^= x >> 16;
            x = x.wrapping_mul(0x7feb_352d);
            x ^= x >> 15;
            x = x.wrapping_mul(0x846c_a68b);
            x ^= x >> 16;
            x as f32 / u32::MAX as f32
        };
        let (u1, u2, u3) = (h(1), h(2) * std::f32::consts::TAU, h(3) * std::f32::consts::TAU);
        let q = glam::Quat::from_xyzw((1.0 - u1).sqrt() * u2.sin(), (1.0 - u1).sqrt() * u2.cos(), u1.sqrt() * u3.sin(), u1.sqrt() * u3.cos()).normalize();
        let has_dynamic = voxelizer.dynamic_surfaces().is_some();
        let o = &self.options;
        let params = ProbeUpdateGpu {
            rotation: glam::Mat4::from_quat(q).to_cols_array(),
            first_probe: self.cursor,
            hysteresis: o.hysteresis.clamp(0.0, 0.999),
            depth_hysteresis: o.depth_hysteresis.clamp(0.0, 0.999),
            sky_scale: o.sky_scale.max(0.0),
            max_steps: o.max_steps.max(1),
            min_clearance: o.min_clearance_voxels.max(0.0) * volume.voxel_size(),
            depth_sharpness: o.depth_sharpness.max(1.0),
            reset_all: std::mem::take(&mut self.reset) as u32,
            has_dynamic: has_dynamic as u32,
            _pad: [0; 3],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        self.cursor = (self.cursor + batch) % count;
        self.frame = self.frame.wrapping_add(1);

        if self.bound.0 != has_dynamic || self.bound.1.as_ref() != Some(sky) {
            self.group = None;
        }
        let group = self.group.get_or_insert_with(|| {
            let tex = wgpu::BindingResource::TextureView;
            let mut entries = vec![
                wgpu::BindGroupEntry { binding: 0, resource: volume.uniform().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: tex(volume.view()) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(volume.sampler()) },
                wgpu::BindGroupEntry { binding: 5, resource: sky.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: voxelizer.static_surfaces().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: voxelizer.dynamic_surfaces().unwrap_or(&self.no_surfaces).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.sh.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.state.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.depth.as_entire_binding() },
            ];
            let aniso = volume.anisotropic_views().expect("SdfProbes read a volume with anisotropic mips");
            entries.extend(aniso.iter().enumerate().map(|(i, v)| wgpu::BindGroupEntry { binding: 40 + i as u32, resource: tex(v) }));
            entries.push(wgpu::BindGroupEntry { binding: 46, resource: tex(sdf.view()) });
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("VoxelGI/ProbesBG"), layout: &self.bgl, entries: &entries })
        });
        self.bound = (has_dynamic, Some(sky.clone()));

        let stamp = crate::profiling::gpu_pass("VoxelGI/Probes");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VoxelGI/Probes"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &*group, &[]);
        pass.dispatch_workgroups(batch, 1, 1);
    }

}
