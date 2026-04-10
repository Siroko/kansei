use crate::renderers::Renderer;
use bytemuck::{Pod, Zeroable};

/// Uniform parameters for the temporal denoise shader.
///
/// Layout must match the WGSL `TemporalParams` struct exactly:
///   mat4x4f currentInvViewProj  (64 bytes)
///   mat4x4f prevViewProj        (64 bytes)
///   f32     blendFactor          (4 bytes)
///   f32     width                (4 bytes)
///   f32     height               (4 bytes)
///   u32     frameIndex           (4 bytes)
/// Total: 144 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TemporalParams {
    current_inv_view_proj: [f32; 16],
    prev_view_proj: [f32; 16],
    blend_factor: f32,
    width: f32,
    height: f32,
    frame_index: u32,
}

const PARAMS_SIZE: u64 = std::mem::size_of::<TemporalParams>() as u64;

/// SVGF temporal denoiser with motion-compensated reprojection.
///
/// Takes noisy path-traced GI, depth, and normal textures plus the
/// previous frame's history and moments, reprojects the current pixel
/// into the previous frame's screen space, validates the reprojection
/// with depth/normal similarity checks, and blends the result using an
/// adaptive alpha. Variance moments (first and second) are accumulated
/// for guiding the subsequent spatial (A-trous) filter.
///
/// History and moments textures are double-buffered (ping-pong) so
/// the previous frame's output becomes the current frame's history input.
pub struct TemporalDenoise {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    history_a: Option<wgpu::Texture>,
    history_a_view: Option<wgpu::TextureView>,
    history_b: Option<wgpu::Texture>,
    history_b_view: Option<wgpu::TextureView>,
    moments_a: Option<wgpu::Texture>,
    moments_a_view: Option<wgpu::TextureView>,
    moments_b: Option<wgpu::Texture>,
    moments_b_view: Option<wgpu::TextureView>,
    output_tex: Option<wgpu::Texture>,
    output_view: Option<wgpu::TextureView>,
    sampler: wgpu::Sampler,
    ping: bool,
    /// Temporal blend factor (default 0.1). Lower = more temporal
    /// accumulation, higher = more responsive to new frames.
    pub blend: f32,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl TemporalDenoise {
    /// Create the temporal denoise pipeline and parameter buffer.
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();

        let shader_src = include_str!("shaders/denoise-temporal.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TemporalDenoise Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // ── Bind group layout ─────────────────────────────────────
        //
        //  0: currentGI        texture_2d<f32>
        //  1: historyGI        texture_2d<f32>
        //  2: outputGI         texture_storage_2d<rgba16float, write>
        //  3: depthTex         texture_depth_2d
        //  4: normalTex        texture_2d<f32>
        //  5: historySamp      sampler
        //  6: params           uniform
        //  7: momentsHistory   texture_2d<f32>
        //  8: momentsOutput    texture_storage_2d<rgba16float, write>
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TemporalDenoise BGL"),
            entries: &[
                // 0: currentGI
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 1: historyGI
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 2: outputGI (storage write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 3: depthTex (depth texture)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 4: normalTex
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 5: historySamp (linear sampler for bilinear history fetch)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 6: params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(PARAMS_SIZE),
                    },
                    count: None,
                },
                // 7: momentsHistory
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 8: momentsOutput (storage write)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TemporalDenoise PipelineLayout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TemporalDenoise Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("temporal_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TemporalDenoise Params"),
            size: PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("TemporalDenoise Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        Self {
            pipeline,
            bgl,
            params_buf,
            history_a: None,
            history_a_view: None,
            history_b: None,
            history_b_view: None,
            moments_a: None,
            moments_a_view: None,
            moments_b: None,
            moments_b_view: None,
            output_tex: None,
            output_view: None,
            sampler,
            ping: false,
            blend: 0.1,
            device,
            queue,
        }
    }

    /// Recreate all internal textures for the given resolution.
    ///
    /// Must be called before the first `denoise()` and whenever the
    /// viewport dimensions change.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        let make_tex = |label: &str, usage: wgpu::TextureUsages| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage,
                view_formats: &[],
            })
        };

        let history_usage =
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;

        let ha = make_tex("TemporalDenoise/HistoryA", history_usage);
        let hb = make_tex("TemporalDenoise/HistoryB", history_usage);
        let ma = make_tex("TemporalDenoise/MomentsA", history_usage);
        let mb = make_tex("TemporalDenoise/MomentsB", history_usage);
        let output = make_tex(
            "TemporalDenoise/Output",
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        );

        self.history_a_view = Some(ha.create_view(&Default::default()));
        self.history_b_view = Some(hb.create_view(&Default::default()));
        self.moments_a_view = Some(ma.create_view(&Default::default()));
        self.moments_b_view = Some(mb.create_view(&Default::default()));
        self.output_view = Some(output.create_view(&Default::default()));

        self.history_a = Some(ha);
        self.history_b = Some(hb);
        self.moments_a = Some(ma);
        self.moments_b = Some(mb);
        self.output_tex = Some(output);

        // Reset ping so first frame uses A as history input (zeroed)
        self.ping = false;
    }

    /// Run the temporal denoise pass.
    ///
    /// # Arguments
    ///
    /// * `encoder`    - Active command encoder
    /// * `gi_view`    - Current frame's noisy GI texture view
    /// * `depth_view` - GBuffer depth texture view
    /// * `normal_view`- GBuffer normal texture view
    /// * `prev_vp`    - Previous frame's View-Projection matrix (column-major f32x16)
    /// * `curr_vp`    - Current frame's View-Projection matrix (column-major f32x16)
    /// * `frame_index`- Current frame number (0 = first frame / reset)
    pub fn denoise(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        gi_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        prev_vp: &[f32; 16],
        curr_vp: &[f32; 16],
        frame_index: u32,
    ) {
        // Guard: textures must be allocated
        let (Some(output_view), Some(ha_view), Some(hb_view), Some(ma_view), Some(mb_view)) = (
            self.output_view.as_ref(),
            self.history_a_view.as_ref(),
            self.history_b_view.as_ref(),
            self.moments_a_view.as_ref(),
            self.moments_b_view.as_ref(),
        ) else {
            return;
        };

        // Compute inverse of current VP for world-space reconstruction
        let curr_vp_mat = glam::Mat4::from_cols_array(curr_vp);
        let inv_vp = curr_vp_mat.inverse();
        let inv_vp_arr: [f32; 16] = inv_vp.to_cols_array();

        // Determine read/write texture views via ping-pong
        let (history_read, history_write, moments_read, moments_write) = if self.ping {
            (hb_view, ha_view, mb_view, ma_view)
        } else {
            (ha_view, hb_view, ma_view, mb_view)
        };

        // Upload uniform params
        let params = TemporalParams {
            current_inv_view_proj: inv_vp_arr,
            prev_view_proj: *prev_vp,
            blend_factor: self.blend,
            width: self.output_tex.as_ref().map_or(0, |t| t.width()) as f32,
            height: self.output_tex.as_ref().map_or(0, |t| t.height()) as f32,
            frame_index,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TemporalDenoise BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(gi_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(history_read),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(history_write),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(moments_read),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(moments_write),
                },
            ],
        });

        // Dispatch
        let width = self.output_tex.as_ref().map_or(1, |t| t.width());
        let height = self.output_tex.as_ref().map_or(1, |t| t.height());
        let wg_x = (width + 7) / 8;
        let wg_y = (height + 7) / 8;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TemporalDenoise"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Copy the written history to the output texture so downstream
        // consumers always read from a stable view.
        // The output_view is the same format/size, so we do a texture copy.
        if let (Some(output_tex), Some(hist_tex)) = (
            self.output_tex.as_ref(),
            if self.ping {
                self.history_a.as_ref()
            } else {
                self.history_b.as_ref()
            },
        ) {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: hist_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: output_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }

        // Flip ping-pong for next frame
        self.ping = !self.ping;
    }

    /// Returns the denoised output texture view (available after `denoise()`).
    pub fn output_view(&self) -> Option<&wgpu::TextureView> {
        self.output_view.as_ref()
    }

    /// Returns the current moments texture view (for the spatial denoiser).
    ///
    /// Contains (moment1, moment2, historyLength, variance) per pixel.
    pub fn moments_view(&self) -> Option<&wgpu::TextureView> {
        // The moments that were just written are the "write" side of the
        // ping-pong, which will become the "read" side next frame. But
        // the spatial denoiser runs in the *same* frame, so return the
        // side we just wrote to.
        if self.ping {
            // We just flipped, so previous write was into A-side moments
            self.moments_a_view.as_ref()
        } else {
            self.moments_b_view.as_ref()
        }
    }
}
