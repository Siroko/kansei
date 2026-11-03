//! Smoke through the volumetric fog (`VolumetricFogEffect::set_smoke`): the march of every
//! enabled smoke volume at half resolution, lit by the fog's lights and composited with it.
//! See `shaders/volumetric_smoke_march.wgsl`.

use bytemuck::{Pod, Zeroable};

use crate::simulations::smoke::{SmokeHandle, SmokePushGpu, SmokeVolumeGpu, MAX_SMOKE_PUSHES, MAX_SMOKE_VOLUMES};

/// How the fog marches smoke.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmokeRenderOptions {
    /// Steps along each ray through all the volumes it crosses (the steps grow past it).
    pub max_steps: u32,
    /// Steps of each sample's march toward each light, for the smoke's own shadow.
    pub shadow_steps: u32,
    /// A step's length in cells of the finest volume it is in.
    pub step_scale: f32,
    /// Jitter the steps per pixel and frame and resolve them over frames (the smoke's own
    /// history, separate from TAA). Off: every step samples its middle.
    pub temporal: bool,
    /// The new frame's weight in the history.
    pub blend: f32,
    /// The farthest the self-shadow march looks toward a light, metres: smoke farther along
    /// the light's path than this does not shadow (thin smoke over metres scatters as the fog).
    pub shadow_distance: f32,
}

impl Default for SmokeRenderOptions {
    fn default() -> Self {
        Self { max_steps: 96, shadow_steps: 6, step_scale: 0.75, temporal: true, blend: 0.25, shadow_distance: 1.0 }
    }
}

/// WGSL `SmokeFrame`, written once a frame.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SmokeFrameGpu {
    pub(crate) inv_view_proj: [f32; 16],
    pub(crate) prev_view_proj: [f32; 16],
    pub(crate) camera_pos: [f32; 3],
    pub(crate) frame: u32,
    pub(crate) view_forward: [f32; 3],
    pub(crate) num_volumes: u32,
    pub(crate) size: [u32; 2],
    pub(crate) full_size: [u32; 2],
    pub(crate) grid_near: f32,
    pub(crate) grid_far: f32,
    pub(crate) grid_d: f32,
    pub(crate) camera_near: f32,
    pub(crate) camera_far: f32,
    pub(crate) time: f32,
    pub(crate) advanced: f32,
    pub(crate) step_scale: f32,
    pub(crate) max_steps: u32,
    pub(crate) shadow_steps: u32,
    pub(crate) jitter: u32,
    pub(crate) num_pushes: u32,
    pub(crate) shadow_distance: f32,
    pub(crate) history_valid: u32,
    pub(crate) blend: f32,
    pub(crate) _pad: f32,
    pub(crate) volumes: [SmokeVolumeGpu; MAX_SMOKE_VOLUMES],
    pub(crate) pushes: [SmokePushGpu; MAX_SMOKE_PUSHES],
}

/// A half-resolution target.
fn target(device: &wgpu::Device, label: &str, (width, height): (u32, u32), format: wgpu::TextureFormat) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// The march's targets at one size.
struct SmokeTargets {
    size: (u32, u32),
    /// (R, Ts), this frame's
    light: wgpu::TextureView,
    /// (linear depth, previous uv, distance of the representative point), this frame's and the
    /// previous one's
    aux: [wgpu::TextureView; 2],
    /// the resolved (R, Ts), this frame's and the previous one's
    history: [wgpu::TextureView; 2],
}

/// What the composite reads of the smoke: its (R, Ts) and the aux with its depth.
pub(crate) struct SmokeComposite {
    pub(crate) light: wgpu::TextureView,
    pub(crate) aux: wgpu::TextureView,
}

/// The camera and froxel grid the march needs from the fog's frame.
pub(crate) struct SmokeView {
    pub(crate) inv_view_proj: glam::Mat4,
    pub(crate) view_proj: glam::Mat4,
    pub(crate) camera_pos: glam::Vec3,
    pub(crate) view_forward: glam::Vec3,
    pub(crate) camera_near: f32,
    pub(crate) camera_far: f32,
    pub(crate) camera_frame: u32,
    pub(crate) grid_near: f32,
    pub(crate) grid_far: f32,
    pub(crate) grid_d: f32,
    pub(crate) full_size: (u32, u32),
}

/// The smoke's GPU side inside the fog: its march and its composite.
pub(crate) struct SmokeRenderer {
    march: wgpu::ComputePipeline,
    march_bgl: wgpu::BindGroupLayout,
    temporal: wgpu::ComputePipeline,
    temporal_bgl: wgpu::BindGroupLayout,
    pub(crate) composite: wgpu::ComputePipeline,
    pub(crate) composite_bgl: wgpu::BindGroupLayout,
    frame_buffer: wgpu::Buffer,
    /// bound for a volume's missing grids
    stand_in: wgpu::TextureView,
    targets: Option<SmokeTargets>,
    frame: u32,
    prev_view_proj: Option<glam::Mat4>,
    last_camera_frame: Option<u32>,
}

impl SmokeRenderer {
    /// `fog_entries`: the layout of the fog's bindings the march shares (its parameters, media,
    /// sky and lights: 1-12, 18-22); `composite_entries`: the fog composite's (0-6).
    pub(crate) fn new(device: &wgpu::Device, march_wgsl: &str, composite_wgsl: &str, fog_entries: &[wgpu::BindGroupLayoutEntry], composite_entries: &[wgpu::BindGroupLayoutEntry]) -> Self {
        let compute = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty, count: None };
        let tex3d = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false };
        let tex2d = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false };
        let storage = |format| wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format, view_dimension: wgpu::TextureViewDimension::D2 };
        let mut march_entries: Vec<_> = fog_entries.to_vec();
        march_entries.extend([
            entry(30, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
            entry(31, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }),
            entry(32, tex3d),
            entry(33, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
        ]);
        march_entries.extend((34..42).map(|b| entry(b, tex3d)));
        march_entries.extend([entry(42, storage(wgpu::TextureFormat::Rgba16Float)), entry(43, storage(wgpu::TextureFormat::Rgba32Float))]);
        let march_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("VolumetricFog/SmokeMarchBGL"), entries: &march_entries });
        let temporal_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VolumetricFog/SmokeTemporalBGL"),
            entries: &[
                entry(30, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
                entry(33, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                entry(44, tex2d),
                entry(45, tex2d),
                entry(46, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }),
                entry(47, tex2d),
                entry(48, storage(wgpu::TextureFormat::Rgba16Float)),
            ],
        });
        let mut comp_entries: Vec<_> = composite_entries.to_vec();
        comp_entries.extend([entry(7, tex2d), entry(8, tex2d)]);
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("VolumetricFog/SmokeCompositeBGL"), entries: &comp_entries });
        let module = |label: &str, code: &str| device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(code.into()) });
        let pipeline = |label: &str, module: &wgpu::ShaderModule, entry: &str, bgl: &wgpu::BindGroupLayout| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[Some(bgl)], immediate_size: 0 });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: Some(&layout), module, entry_point: Some(entry), compilation_options: Default::default(), cache: None })
        };
        // the march and its temporal filter: one module, two entry points
        let march_module = module("VolumetricFog/SmokeMarch", march_wgsl);
        let composite_module = module("VolumetricFog/SmokeComposite", composite_wgsl);
        let stand_in = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("VolumetricFog/NoSmokeGrid"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        Self {
            march: pipeline("VolumetricFog/SmokeMarch", &march_module, "smokeMarch", &march_bgl),
            temporal: pipeline("VolumetricFog/SmokeTemporal", &march_module, "smokeTemporal", &temporal_bgl),
            composite: pipeline("VolumetricFog/SmokeComposite", &composite_module, "compositeSmoke", &composite_bgl),
            march_bgl,
            temporal_bgl,
            composite_bgl,
            frame_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("VolumetricFog/SmokeFrame"),
                size: std::mem::size_of::<SmokeFrameGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            stand_in,
            targets: None,
            frame: 0,
            prev_view_proj: None,
            last_camera_frame: None,
        }
    }

    /// Forget the history (camera cuts, a new handle).
    pub(crate) fn reset_history(&mut self) {
        self.prev_view_proj = None;
    }

    /// March the handle's volumes into the targets, and return them for the composite; `None`
    /// when no volume is enabled. `fog_entries`: the fog's bindings the march shares.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn march(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        cache: &crate::renderers::BindGroupCache,
        handle: &SmokeHandle,
        options: &SmokeRenderOptions,
        view: &SmokeView,
        fog_entries: Vec<wgpu::BindGroupEntry>,
        depth: &wgpu::TextureView,
        accum: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> Option<SmokeComposite> {
        let (volumes, pushes, time, advanced) = {
            let mut shared = handle.shared.lock().unwrap();
            if shared.volumes.is_empty() {
                return None;
            }
            let advanced = std::mem::take(&mut shared.advanced);
            (shared.volumes.clone(), shared.pushes.clone(), shared.time, advanced)
        };
        let size = (view.full_size.0.div_ceil(2), view.full_size.1.div_ceil(2));
        if self.targets.as_ref().is_none_or(|t| t.size != size) {
            let aux = || target(device, "VolumetricFog/SmokeAux", size, wgpu::TextureFormat::Rgba32Float);
            let history = || target(device, "VolumetricFog/SmokeHistory", size, wgpu::TextureFormat::Rgba16Float);
            self.targets = Some(SmokeTargets {
                size,
                light: target(device, "VolumetricFog/SmokeLight", size, wgpu::TextureFormat::Rgba16Float),
                aux: [aux(), aux()],
                history: [history(), history()],
            });
            self.prev_view_proj = None;
        }
        // a gap in the camera's frames (a cut, or the smoke was off) invalidates the history
        if self.last_camera_frame.is_some_and(|f| view.camera_frame != f && view.camera_frame != f.wrapping_add(1)) {
            self.prev_view_proj = None;
        }
        self.last_camera_frame = Some(view.camera_frame);

        let mut frame = SmokeFrameGpu {
            inv_view_proj: view.inv_view_proj.to_cols_array(),
            prev_view_proj: self.prev_view_proj.unwrap_or(view.view_proj).to_cols_array(),
            camera_pos: view.camera_pos.to_array(),
            frame: self.frame,
            view_forward: view.view_forward.to_array(),
            num_volumes: volumes.len().min(MAX_SMOKE_VOLUMES) as u32,
            size: [size.0, size.1],
            full_size: [view.full_size.0, view.full_size.1],
            grid_near: view.grid_near,
            grid_far: view.grid_far,
            grid_d: view.grid_d,
            camera_near: view.camera_near,
            camera_far: view.camera_far,
            time,
            advanced,
            step_scale: options.step_scale.max(0.05),
            max_steps: options.max_steps.max(1),
            shadow_steps: options.shadow_steps,
            jitter: options.temporal as u32,
            num_pushes: pushes.len().min(MAX_SMOKE_PUSHES) as u32,
            shadow_distance: options.shadow_distance.max(0.0),
            history_valid: (options.temporal && self.prev_view_proj.is_some()) as u32,
            blend: options.blend.clamp(0.02, 1.0),
            ..Zeroable::zeroed()
        };
        for (dst, v) in frame.volumes.iter_mut().zip(&volumes) {
            *dst = v.data;
        }
        for (dst, p) in frame.pushes.iter_mut().zip(&pushes) {
            *dst = *p;
        }
        queue.write_buffer(&self.frame_buffer, 0, bytemuck::bytes_of(&frame));
        let current = (self.frame % 2) as usize;
        self.frame = self.frame.wrapping_add(1);
        self.prev_view_proj = Some(view.view_proj);

        let targets = self.targets.as_ref().unwrap();
        let tex = wgpu::BindingResource::TextureView;
        let grid = |i: usize, f: fn(&crate::simulations::smoke::SmokeHandleVolume) -> Option<&wgpu::TextureView>| volumes.get(i).and_then(f).unwrap_or(&self.stand_in);
        let mut entries = fog_entries;
        entries.extend([
            wgpu::BindGroupEntry { binding: 30, resource: self.frame_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 31, resource: tex(depth) },
            wgpu::BindGroupEntry { binding: 32, resource: tex(accum) },
            wgpu::BindGroupEntry { binding: 33, resource: wgpu::BindingResource::Sampler(sampler) },
        ]);
        for i in 0..MAX_SMOKE_VOLUMES {
            entries.push(wgpu::BindGroupEntry { binding: 34 + i as u32, resource: tex(grid(i, |v| v.scalars.as_ref())) });
            entries.push(wgpu::BindGroupEntry { binding: 38 + i as u32, resource: tex(grid(i, |v| v.velocity.as_ref())) });
        }
        entries.extend([
            wgpu::BindGroupEntry { binding: 42, resource: tex(&targets.light) },
            wgpu::BindGroupEntry { binding: 43, resource: tex(&targets.aux[current]) },
        ]);
        let bg = cache.get(device, &wgpu::BindGroupDescriptor { label: Some("VolumetricFog/SmokeMarchBG"), layout: &self.march_bgl, entries: &entries });
        let stamp = crate::profiling::gpu_pass("VolumetricFog/SmokeMarch");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VolumetricFog/SmokeMarch"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
        pass.set_pipeline(&self.march);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
        drop(pass);
        if !options.temporal {
            return Some(SmokeComposite { light: targets.light.clone(), aux: targets.aux[current].clone() });
        }
        let temporal_bg = cache.get(
            device,
            &wgpu::BindGroupDescriptor {
                label: Some("VolumetricFog/SmokeTemporalBG"),
                layout: &self.temporal_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 30, resource: self.frame_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 33, resource: wgpu::BindingResource::Sampler(sampler) },
                    wgpu::BindGroupEntry { binding: 44, resource: tex(&targets.light) },
                    wgpu::BindGroupEntry { binding: 45, resource: tex(&targets.aux[current]) },
                    wgpu::BindGroupEntry { binding: 46, resource: tex(&targets.history[1 - current]) },
                    wgpu::BindGroupEntry { binding: 47, resource: tex(&targets.aux[1 - current]) },
                    wgpu::BindGroupEntry { binding: 48, resource: tex(&targets.history[current]) },
                ],
            },
        );
        let stamp = crate::profiling::gpu_pass("VolumetricFog/SmokeTemporal");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VolumetricFog/SmokeTemporal"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
        pass.set_pipeline(&self.temporal);
        pass.set_bind_group(0, &temporal_bg, &[]);
        pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
        drop(pass);
        Some(SmokeComposite { light: targets.history[current].clone(), aux: targets.aux[current].clone() })
    }
}
