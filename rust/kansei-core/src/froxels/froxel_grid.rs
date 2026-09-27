use bytemuck::{Pod, Zeroable};

use crate::math::Mat4;

/// WGSL helpers for exponential depth slicing and froxel <-> world conversion, using the
/// camera's [0,1] depth convention. Prepend to consumer shaders that read the grid.
pub const FROXEL_WGSL_HELPERS: &str = include_str!("../shaders/froxel_common.wgsl");

const ACCUMULATE_WGSL: &str = concat!(
    include_str!("../shaders/froxel_common.wgsl"),
    include_str!("../shaders/froxel_accumulate.wgsl"),
);
const TEMPORAL_WGSL: &str = concat!(
    include_str!("../shaders/froxel_common.wgsl"),
    include_str!("../shaders/froxel_temporal.wgsl"),
);

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub struct FroxelGridOptions {
    pub grid_w: u32,
    pub grid_h: u32,
    pub grid_d: u32,
    /// View distance of the first slice.
    pub near: f32,
    /// View distance of the last slice; fog beyond it is clamped to it.
    pub far: f32,
    /// Blend injected froxels with reprojected history (smooths jitter, costs a pass).
    pub temporal: bool,
    /// Weight of the current frame when `temporal` is on.
    pub blend_factor: f32,
}

impl Default for FroxelGridOptions {
    fn default() -> Self {
        Self { grid_w: 160, grid_h: 90, grid_d: 64, near: 0.1, far: 1000.0, temporal: false, blend_factor: 0.05 }
    }
}

// ── Uniform layouts (must match the WGSL structs) ──

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GridParamsGpu {
    near: f32,
    far: f32,
    grid_w: u32,
    grid_h: u32,
    grid_d: u32,
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct TemporalParamsGpu {
    current_inv_vp: [f32; 16],
    prev_vp: [f32; 16],
    grid_near: f32,
    grid_far: f32,
    camera_near: f32,
    camera_far: f32,
    grid_w: u32,
    grid_h: u32,
    grid_d: u32,
    blend_factor: f32,
    has_prev_frame: u32,
    _pad: [u32; 3],
}

struct Temporal {
    // the history textures and the grid params are kept alive by the bind groups
    params_buf: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    /// [i] reads history[i] and writes history[1 - i].
    blend_bg: [wgpu::BindGroup; 2],
    /// [i] accumulates from history[i].
    accum_bg: [wgpu::BindGroup; 2],
    prev_vp: [f32; 16],
    /// History slot the next blend reads.
    read_idx: usize,
    has_prev_frame: bool,
}

/// A frustum-aligned 3D grid of participating-media samples with exponential depth slices,
/// as the TS `FroxelGrid`. An injection pass writes (in-scatter, extinction) into
/// `scatter_extinction`; `temporal_blend` (optional) and `accumulate` integrate it front to back
/// into `accum`, where rgb is the light scattered toward the camera up to each slice and a is
/// the transmittance.
pub struct FroxelGrid {
    grid_w: u32,
    grid_h: u32,
    grid_d: u32,
    near: f32,
    far: f32,
    blend_factor: f32,
    scatter_extinction: wgpu::Texture,
    scatter_extinction_view: wgpu::TextureView,
    accum: wgpu::Texture,
    accum_view: wgpu::TextureView,
    accum_pipeline: wgpu::ComputePipeline,
    accum_bg: wgpu::BindGroup,
    temporal: Option<Temporal>,
}

fn storage_texture_3d(device: &wgpu::Device, label: &str, w: u32, h: u32, d: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: d },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn texture_entry(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D3,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: FORMAT,
            view_dimension: wgpu::TextureViewDimension::D3,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn compute_pipeline(device: &wgpu::Device, label: &str, code: &str, bgl: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(code.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn view(t: &wgpu::Texture) -> wgpu::TextureView {
    t.create_view(&Default::default())
}

impl FroxelGrid {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, options: &FroxelGridOptions) -> Self {
        let (w, h, d) = (options.grid_w.max(1), options.grid_h.max(1), options.grid_d.max(1));
        let scatter_extinction = storage_texture_3d(device, "FroxelGrid/ScatterExtinction", w, h, d);
        let accum = storage_texture_3d(device, "FroxelGrid/Accum", w, h, d);
        let scatter_extinction_view = view(&scatter_extinction);
        let accum_view = view(&accum);

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FroxelGrid/Params"),
            size: std::mem::size_of::<GridParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = GridParamsGpu { near: options.near, far: options.far, grid_w: w, grid_h: h, grid_d: d, _pad: [0.0; 3] };
        queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let accum_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FroxelGrid/AccumBGL"),
            entries: &[texture_entry(0, false), storage_entry(1), uniform_entry(2)],
        });
        let accum_pipeline = compute_pipeline(device, "FroxelGrid/Accumulate", ACCUMULATE_WGSL, &accum_bgl);
        let accum_bind_group = |source: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("FroxelGrid/AccumBG"),
                layout: &accum_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(source) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&accum_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
                ],
            })
        };
        let accum_bg = accum_bind_group(&scatter_extinction_view);

        let temporal = options.temporal.then(|| {
            let history = [
                storage_texture_3d(device, "FroxelGrid/History0", w, h, d),
                storage_texture_3d(device, "FroxelGrid/History1", w, h, d),
            ];
            let history_views = [view(&history[0]), view(&history[1])];
            let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("FroxelGrid/TemporalParams"),
                size: std::mem::size_of::<TemporalParamsGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("FroxelGrid/HistorySampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("FroxelGrid/TemporalBGL"),
                entries: &[
                    texture_entry(0, false),
                    texture_entry(1, true),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    storage_entry(3),
                    uniform_entry(4),
                ],
            });
            let pipeline = compute_pipeline(device, "FroxelGrid/Temporal", TEMPORAL_WGSL, &bgl);
            let blend_bg = |read: usize| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("FroxelGrid/TemporalBG"),
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&scatter_extinction_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&history_views[read]) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&history_views[1 - read]) },
                        wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
                    ],
                })
            };
            Temporal {
                blend_bg: [blend_bg(0), blend_bg(1)],
                accum_bg: [accum_bind_group(&history_views[0]), accum_bind_group(&history_views[1])],
                params_buf,
                pipeline,
                prev_vp: [0.0; 16],
                read_idx: 0,
                has_prev_frame: false,
            }
        });

        Self {
            grid_w: w,
            grid_h: h,
            grid_d: d,
            near: options.near,
            far: options.far,
            blend_factor: options.blend_factor,
            scatter_extinction,
            scatter_extinction_view,
            accum,
            accum_view,
            accum_pipeline,
            accum_bg,
            temporal,
        }
    }

    pub fn grid_w(&self) -> u32 { self.grid_w }
    pub fn grid_h(&self) -> u32 { self.grid_h }
    pub fn grid_d(&self) -> u32 { self.grid_d }
    pub fn near(&self) -> f32 { self.near }
    pub fn far(&self) -> f32 { self.far }
    pub fn is_temporal(&self) -> bool { self.temporal.is_some() }

    /// Injection target: rgba16float 3D storage texture (rgb in-scatter, a extinction).
    pub fn scatter_extinction_texture(&self) -> &wgpu::Texture { &self.scatter_extinction }
    pub fn scatter_extinction_view(&self) -> &wgpu::TextureView { &self.scatter_extinction_view }
    /// Integrated result: rgb scattered light toward the camera, a transmittance.
    pub fn accum_view(&self) -> &wgpu::TextureView { &self.accum_view }
    pub fn accum_texture(&self) -> &wgpu::Texture { &self.accum }

    /// Forget the temporal history, so the next frame uses only its own injection. Call on a
    /// camera cut, or reprojection smears the previous shot's fog into the new one.
    pub fn reset_history(&mut self) {
        if let Some(t) = &mut self.temporal {
            t.has_prev_frame = false;
        }
    }

    /// Temporal reprojection blend. Call after injection and before `accumulate`.
    /// No-op unless the grid was created with `temporal: true`.
    pub fn temporal_blend(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        current_inv_vp: &Mat4,
        current_vp: &Mat4,
        camera_near: f32,
        camera_far: f32,
    ) {
        let (w, h, d) = (self.grid_w, self.grid_h, self.grid_d);
        let (near, far, blend) = (self.near, self.far, self.blend_factor);
        let Some(t) = &mut self.temporal else { return };
        let params = TemporalParamsGpu {
            current_inv_vp: current_inv_vp.data,
            prev_vp: t.prev_vp,
            grid_near: near,
            grid_far: far,
            camera_near,
            camera_far,
            grid_w: w,
            grid_h: h,
            grid_d: d,
            blend_factor: blend,
            has_prev_frame: t.has_prev_frame as u32,
            _pad: [0; 3],
        };
        queue.write_buffer(&t.params_buf, 0, bytemuck::bytes_of(&params));

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FroxelGrid/TemporalBlend"),
                ..Default::default()
            });
            pass.set_pipeline(&t.pipeline);
            pass.set_bind_group(0, &t.blend_bg[t.read_idx], &[]);
            pass.dispatch_workgroups(w.div_ceil(4), h.div_ceil(4), d.div_ceil(4));
        }

        t.prev_vp = current_vp.data;
        t.has_prev_frame = true;
        t.read_idx = 1 - t.read_idx; // the slot just written is read next frame
    }

    /// Front-to-back accumulation. Call after injection (and `temporal_blend`, if temporal).
    pub fn accumulate(&self, encoder: &mut wgpu::CommandEncoder) {
        let bind_group = match &self.temporal {
            // after a blend, read_idx points at the history slot that was just written
            Some(t) if t.has_prev_frame => &t.accum_bg[t.read_idx],
            _ => &self.accum_bg,
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FroxelGrid/Accumulate"),
            ..Default::default()
        });
        pass.set_pipeline(&self.accum_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(self.grid_w.div_ceil(8), self.grid_h.div_ceil(8), 1);
    }

    #[cfg(test)]
    pub(crate) fn shader_sources() -> [(&'static str, &'static str); 2] {
        [("froxel_accumulate", ACCUMULATE_WGSL), ("froxel_temporal", TEMPORAL_WGSL)]
    }
}
