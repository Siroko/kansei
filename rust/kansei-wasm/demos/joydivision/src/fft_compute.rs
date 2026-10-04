// FFT displacement compute — engine-adjacent code using raw wgpu.
// Reads FFT data from JS each frame and displaces particle Y positions.

use wgpu::util::DeviceExt;

use crate::text_layout::LyricsParticleData;

const FFT_BINS: usize = 256;
const VERTS_PER_LINE: u32 = 256;
const FFT_DISPLACE_WGSL: &str = include_str!("shaders/fft_displace.wgsl");
const ELEVATION_SCROLL_WGSL: &str = include_str!("shaders/elevation_scroll.wgsl");
const WAVE_DISPLACE_WGSL: &str = include_str!("shaders/wave_displace.wgsl");
const WAVE_FILL_WGSL: &str = include_str!("shaders/wave_fill.wgsl");

/// Uniform params matching the WGSL Params struct.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FftParams {
    particle_count: u32,
    line_count: u32,
    active_line_idx: u32,
    scroll_y: f32,
    fft_amplitude: f32,
    line_spacing: f32,
    max_chars_per_line: u32,
    half_width: f32,
    noise_scale: f32,
    noise_strength: f32,
    noise_speed: f32,
    noise_zoom: f32,
    noise_mix: f32,
    noise_time: f32,
    peak_exponent: f32,
    peak_min: f32,
    text_offset_x: f32,
    text_offset_y: f32,
    text_offset_z: f32,
    glyph_rot_x: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ScrollParams {
    time: f32,
    noise_scale: f32,
    noise_strength: f32,
    noise_speed: f32,
    noise_zoom: f32,
    noise_mix: f32,
    peak_exponent: f32,
    temporal_blend: f32,
    fft_pow: f32,
    peak_min: f32,
    _scroll_pad2: u32,
    _scroll_pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct WaveParams {
    line_count: u32,
    verts_per_line: u32,
    half_width: f32,
    active_line_idx: u32,
    line_spacing: f32,
    fft_amplitude: f32,
    thickness: f32,
    noise_scale: f32,
    noise_strength: f32,
    noise_speed: f32,
    noise_zoom: f32,
    noise_mix: f32,
    noise_time: f32,
    peak_exponent: f32,
    peak_min: f32,
    _pad3: u32,
}

struct Pass {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl Pass {
    fn dispatch<'a>(&'a self, cpass: &mut wgpu::ComputePass<'a>, wx: u32, wy: u32, wz: u32) {
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(wx, wy, wz);
    }
}

#[allow(dead_code)]
pub struct FftCompute {
    device: wgpu::Device,
    queue: wgpu::Queue,

    // GPU buffers
    positions_buffer: wgpu::Buffer,
    base_positions_buffer: wgpu::Buffer,
    fft_buffer: wgpu::Buffer,
    line_meta_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,

    // Elevation map: two 256x256 R32Float textures (ping-pong)
    elevation_a: wgpu::Texture,
    elevation_a_view: wgpu::TextureView,
    elevation_b: wgpu::Texture,
    elevation_b_view: wgpu::TextureView,
    elevation_sampler: wgpu::Sampler,
    ping_pong: bool, // false = read A write B, true = read B write A
    scroll_pass_ab: Pass, // reads A, writes B
    scroll_pass_ba: Pass, // reads B, writes A
    scroll_params_buffer: wgpu::Buffer,
    noise_scale: f32,
    noise_strength: f32,
    noise_speed: f32,
    noise_zoom: f32,
    noise_mix: f32,
    peak_exponent: f32,
    temporal_blend: f32,
    fft_pow: f32,
    noise_time: f32,
    peak_min: f32,
    text_offset: [f32; 3],
    glyph_rot_x: f32,

    // Displace passes (one per ping-pong state)
    displace_a: Pass, // reads elevation_a
    displace_b: Pass, // reads elevation_b

    // Wave line displacement
    wave_positions_buffer: wgpu::Buffer,
    wave_params_buffer: wgpu::Buffer,
    wave_displace_a: Pass, // reads elevation_a
    wave_displace_b: Pass, // reads elevation_b
    wave_total_verts: u32,
    wave_thickness: f32,

    // Wave fill (solid black occlusion mesh)
    fill_positions_buffer: wgpu::Buffer,
    fill_displace_a: Pass, // reads elevation_a
    fill_displace_b: Pass, // reads elevation_b

    // State
    particle_count: u32,
    line_count: u32,
    line_spacing: f32,
    fft_amplitude: f32,
    line_half_width: f32,
    timestamps: Vec<f32>,
    max_chars_per_line: u32,
}

impl FftCompute {
    pub fn new(
        renderer: &kansei_core::renderers::Renderer,
        data: &LyricsParticleData,
        line_spacing: f32,
        fft_amplitude: f32,
    ) -> Self {
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();
        let particle_count = data.total_particles;
        let line_count = data.total_lines;

        let usage_sv = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;

        // Positions buffer (read-write by compute, read as vertex by render)
        let positions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("JoyDiv/Positions"),
            contents: bytemuck::cast_slice(&data.positions),
            usage: usage_sv | wgpu::BufferUsages::COPY_SRC,
        });

        // Base positions (read-only, never changes)
        let base_positions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("JoyDiv/BasePositions"),
            contents: bytemuck::cast_slice(&data.positions),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // FFT data (256 u32 values, written from JS each frame)
        let fft_data = vec![0u32; FFT_BINS];
        let fft_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("JoyDiv/FFT"),
            contents: bytemuck::cast_slice(&fft_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Line metadata per particle
        let line_meta_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("JoyDiv/LineMeta"),
            contents: bytemuck::cast_slice(&data.line_meta),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Uniform params
        let params = FftParams {
            particle_count,
            line_count,
            active_line_idx: 0,
            scroll_y: 0.0,
            fft_amplitude,
            line_spacing,
            max_chars_per_line: data.max_chars_per_line,
            half_width: data.line_half_width,
            noise_scale: 0.09, noise_strength: 1.3, noise_speed: 0.4,
            noise_zoom: 0.6, noise_mix: 1.0, noise_time: 0.0,
            peak_exponent: 15.2, peak_min: 0.08,
            text_offset_x: 0.0, text_offset_y: 0.0, text_offset_z: 1.0,
            glyph_rot_x: std::f32::consts::FRAC_PI_2,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("JoyDiv/Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Elevation map textures (256x256 R32Float, ping-pong)
        let tex_desc = wgpu::TextureDescriptor {
            label: Some("JoyDiv/ElevationA"),
            size: wgpu::Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };
        let elevation_a = device.create_texture(&tex_desc);
        let elevation_b = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("JoyDiv/ElevationB"),
            ..tex_desc
        });
        let elevation_a_view = elevation_a.create_view(&wgpu::TextureViewDescriptor::default());
        let elevation_b_view = elevation_b.create_view(&wgpu::TextureViewDescriptor::default());

        let elevation_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("JoyDiv/ElevationSampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let scroll_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("JoyDiv/ScrollParams"),
            size: std::mem::size_of::<ScrollParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Helper closures for bind group layout entries
        let c = wgpu::ShaderStages::COMPUTE;
        let storage_rw = |binding: u32| -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: c,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        };
        let storage_ro = |binding: u32| -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: c,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        };
        let uniform = |binding: u32| -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: c,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        };

        // ── Scroll pass (elevation map ping-pong) ──
        let scroll_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("JoyDiv/ElevScroll/Shader"),
            source: wgpu::ShaderSource::Wgsl(ELEVATION_SCROLL_WGSL.into()),
        });

        let scroll_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("JoyDiv/ScrollBGL"),
            entries: &[
                // binding 0: elevRead (texture_2d<f32>, read via textureLoad)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: c,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 1: elevWrite (storage texture, write-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: c,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                storage_ro(2),  // fftRow
                uniform(3),     // params
            ],
        });

        let scroll_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("JoyDiv/ElevScroll/Layout"),
            bind_group_layouts: &[&scroll_bgl],
            push_constant_ranges: &[],
        });

        let scroll_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("JoyDiv/ElevScroll/Pipeline"),
            layout: Some(&scroll_pipeline_layout),
            module: &scroll_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Scroll A->B bind group
        let scroll_bg_ab = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("JoyDiv/Scroll/BG_AB"),
            layout: &scroll_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&elevation_a_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&elevation_b_view) },
                wgpu::BindGroupEntry { binding: 2, resource: fft_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: scroll_params_buffer.as_entire_binding() },
            ],
        });

        // Scroll B->A bind group
        let scroll_bg_ba = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("JoyDiv/Scroll/BG_BA"),
            layout: &scroll_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&elevation_b_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&elevation_a_view) },
                wgpu::BindGroupEntry { binding: 2, resource: fft_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: scroll_params_buffer.as_entire_binding() },
            ],
        });

        let scroll_pass_ab = Pass { pipeline: scroll_pipeline.clone(), bind_group: scroll_bg_ab };
        let scroll_pass_ba = Pass { pipeline: scroll_pipeline, bind_group: scroll_bg_ba };

        // ── Displace pass ──
        let displace_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("JoyDiv/FftDisplace/Shader"),
            source: wgpu::ShaderSource::Wgsl(FFT_DISPLACE_WGSL.into()),
        });

        let displace_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("JoyDiv/FftDisplace/BGL"),
            entries: &[
                storage_rw(0),  // positions
                storage_ro(1),  // basePositions
                // binding 2: elevationTex (texture_2d<f32>, sampled with linear filtering)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: c,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_ro(3),  // lineMeta
                uniform(4),     // params
                // binding 5: elevationSampler
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: c,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let displace_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("JoyDiv/FftDisplace/Layout"),
            bind_group_layouts: &[&displace_bgl],
            push_constant_ranges: &[],
        });

        let displace_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("JoyDiv/FftDisplace/Pipeline"),
            layout: Some(&displace_pipeline_layout),
            module: &displace_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Displace bind group A (reads elevation_a)
        let displace_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("JoyDiv/FftDisplace/BG_A"),
            layout: &displace_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: positions_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: base_positions_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&elevation_a_view) },
                wgpu::BindGroupEntry { binding: 3, resource: line_meta_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&elevation_sampler) },
            ],
        });

        // Displace bind group B (reads elevation_b)
        let displace_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("JoyDiv/FftDisplace/BG_B"),
            layout: &displace_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: positions_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: base_positions_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&elevation_b_view) },
                wgpu::BindGroupEntry { binding: 3, resource: line_meta_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&elevation_sampler) },
            ],
        });

        let displace_a = Pass { pipeline: displace_pipeline.clone(), bind_group: displace_bg_a };
        let displace_b = Pass { pipeline: displace_pipeline, bind_group: displace_bg_b };

        // ── Wave line displacement ──
        // 2 vertices per sample (top + bottom of ribbon)
        let wave_total_verts = line_count * VERTS_PER_LINE * 2;
        // 9 floats per vertex: pos(4) + normal(3) + uv(2), matching Vertex layout (36 bytes)
        let wave_buf_size = (wave_total_verts as usize * 9 * 4) as u64;

        let wave_positions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("JoyDiv/WavePositions"),
            size: wave_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let wave_params = WaveParams {
            line_count,
            verts_per_line: VERTS_PER_LINE,
            half_width: data.line_half_width,
            active_line_idx: 0,
            line_spacing,
            fft_amplitude,
            thickness: 0.44,
            noise_scale: 0.09,
            noise_strength: 1.3,
            noise_speed: 0.4,
            noise_zoom: 0.6,
            noise_mix: 1.0,
            noise_time: 0.0,
            peak_exponent: 15.2, peak_min: 0.08, _pad3: 0,
        };
        let wave_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("JoyDiv/WaveParams"),
            contents: bytemuck::cast_slice(&[wave_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let wave_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("JoyDiv/WaveDisplace/Shader"),
            source: wgpu::ShaderSource::Wgsl(WAVE_DISPLACE_WGSL.into()),
        });

        let wave_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("JoyDiv/WaveDisplace/BGL"),
            entries: &[
                storage_rw(0),  // vertices (wave positions)
                // binding 2: elevationTex (texture_2d<f32>, sampled with linear filtering)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: c,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                uniform(3),     // wave params
                // binding 4: elevationSampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: c,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let wave_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("JoyDiv/WaveDisplace/Layout"),
            bind_group_layouts: &[&wave_bgl],
            push_constant_ranges: &[],
        });

        let wave_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("JoyDiv/WaveDisplace/Pipeline"),
            layout: Some(&wave_pipeline_layout),
            module: &wave_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Wave bind group A (reads elevation_a)
        let wave_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("JoyDiv/WaveDisplace/BG_A"),
            layout: &wave_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wave_positions_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&elevation_a_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wave_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&elevation_sampler) },
            ],
        });

        // Wave bind group B (reads elevation_b)
        let wave_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("JoyDiv/WaveDisplace/BG_B"),
            layout: &wave_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wave_positions_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&elevation_b_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wave_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&elevation_sampler) },
            ],
        });

        let wave_displace_a = Pass { pipeline: wave_pipeline.clone(), bind_group: wave_bg_a };
        let wave_displace_b = Pass { pipeline: wave_pipeline, bind_group: wave_bg_b };

        // ── Wave fill displacement (solid occlusion mesh) ──
        let fill_positions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("JoyDiv/FillPositions"),
            size: wave_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let fill_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("JoyDiv/WaveFill/Shader"),
            source: wgpu::ShaderSource::Wgsl(WAVE_FILL_WGSL.into()),
        });

        let fill_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("JoyDiv/WaveFill/Layout"),
            bind_group_layouts: &[&wave_bgl], // same layout as wave
            push_constant_ranges: &[],
        });

        let fill_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("JoyDiv/WaveFill/Pipeline"),
            layout: Some(&fill_pipeline_layout),
            module: &fill_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Fill bind group A (reads elevation_a)
        let fill_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("JoyDiv/WaveFill/BG_A"),
            layout: &wave_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: fill_positions_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&elevation_a_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wave_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&elevation_sampler) },
            ],
        });

        // Fill bind group B (reads elevation_b)
        let fill_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("JoyDiv/WaveFill/BG_B"),
            layout: &wave_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: fill_positions_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&elevation_b_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wave_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&elevation_sampler) },
            ],
        });

        let fill_displace_a = Pass { pipeline: fill_pipeline.clone(), bind_group: fill_bg_a };
        let fill_displace_b = Pass { pipeline: fill_pipeline, bind_group: fill_bg_b };

        Self {
            device,
            queue,
            positions_buffer,
            base_positions_buffer,
            fft_buffer,
            line_meta_buffer,
            params_buffer,
            elevation_a,
            elevation_a_view,
            elevation_b,
            elevation_b_view,
            elevation_sampler,
            ping_pong: false,
            scroll_pass_ab,
            scroll_pass_ba,
            scroll_params_buffer,
            noise_scale: 0.09,
            noise_strength: 1.3,
            noise_speed: 0.4,
            noise_zoom: 0.6,
            noise_mix: 1.0,
            peak_exponent: 15.2,
            temporal_blend: 0.70,
            fft_pow: 2.0,
            peak_min: 0.08,
            noise_time: 0.0,
            text_offset: [0.0, 0.0, 1.0],
            glyph_rot_x: std::f32::consts::FRAC_PI_2,
            displace_a,
            displace_b,
            wave_positions_buffer,
            wave_params_buffer,
            wave_displace_a,
            wave_displace_b,
            wave_total_verts,
            wave_thickness: 0.44,
            fill_positions_buffer,
            fill_displace_a,
            fill_displace_b,
            particle_count,
            line_count,
            line_spacing,
            fft_amplitude,
            line_half_width: data.line_half_width,
            timestamps: data.line_timestamps.clone(),
            max_chars_per_line: data.max_chars_per_line,
        }
    }

    pub fn set_fft_amplitude(&mut self, amp: f32) {
        self.fft_amplitude = amp;
    }

    pub fn set_noise_scale(&mut self, v: f32) {
        self.noise_scale = v;
    }

    pub fn set_noise_speed(&mut self, v: f32) { self.noise_speed = v; }
    pub fn set_noise_zoom(&mut self, v: f32) { self.noise_zoom = v; }
    pub fn set_noise_mix(&mut self, v: f32) { self.noise_mix = v; }
    pub fn set_peak_exponent(&mut self, v: f32) { self.peak_exponent = v; }
    pub fn set_temporal_blend(&mut self, v: f32) { self.temporal_blend = v; }
    pub fn set_fft_pow(&mut self, v: f32) { self.fft_pow = v; }
    pub fn set_peak_min(&mut self, v: f32) { self.peak_min = v; }
    pub fn set_wave_thickness(&mut self, v: f32) { self.wave_thickness = v; }
    pub fn set_noise_strength(&mut self, v: f32) {
        self.noise_strength = v;
    }

    pub fn set_text_offset(&mut self, offset: [f32; 3]) {
        self.text_offset = offset;
    }

    pub fn set_glyph_rot_x(&mut self, radians: f32) {
        self.glyph_rot_x = radians;
    }

    /// Upload this frame's FFT row and run the passes; `current_time` is the song's position,
    /// `dt` the seconds since the previous update (the noise field moves in real time, whatever
    /// the frame rate and whether the song plays).
    pub fn update(&mut self, fft_data: &[u8], current_time: f32, dt: f32) {
        self.noise_time += dt;

        // Convert u8 FFT data to u32 array for GPU
        let mut fft_u32 = [0u32; FFT_BINS];
        let len = fft_data.len().min(FFT_BINS);
        for i in 0..len {
            fft_u32[i] = fft_data[i] as u32;
        }

        // Determine active line index from timestamps
        let active_line_idx = self.find_active_line(current_time);

        // Compute scroll offset so active line is near top third of view
        // Each line is at y = -(line_idx * line_spacing)
        // We want active line's y to appear at roughly +1/3 of visible height
        // scrollY shifts all positions down: pos.y -= scrollY
        // So scrollY = -(active_line_y) + target_screen_y
        let target_y = self.line_spacing * 2.0; // ~2 lines above center
        let active_line_y = -(active_line_idx as f32) * self.line_spacing;
        let scroll_y = active_line_y - target_y; // negative means shift up

        // Upload FFT data
        self.queue.write_buffer(&self.fft_buffer, 0, bytemuck::cast_slice(&fft_u32));

        // Upload scroll params
        let scroll_params = ScrollParams {
            time: current_time,
            noise_scale: self.noise_scale,
            noise_strength: self.noise_strength,
            noise_speed: self.noise_speed,
            noise_zoom: self.noise_zoom,
            noise_mix: self.noise_mix,
            peak_exponent: self.peak_exponent,
            temporal_blend: self.temporal_blend,
            fft_pow: self.fft_pow,
            peak_min: self.peak_min,
            _scroll_pad2: 0,
            _scroll_pad3: 0,
        };
        self.queue.write_buffer(&self.scroll_params_buffer, 0, bytemuck::cast_slice(&[scroll_params]));

        // Upload displace params
        let params = FftParams {
            particle_count: self.particle_count,
            line_count: self.line_count,
            active_line_idx: active_line_idx as u32,
            scroll_y,
            fft_amplitude: self.fft_amplitude,
            line_spacing: self.line_spacing,
            max_chars_per_line: self.max_chars_per_line,
            half_width: self.line_half_width,
            noise_scale: self.noise_scale,
            noise_strength: self.noise_strength,
            noise_speed: self.noise_speed,
            noise_zoom: self.noise_zoom,
            noise_mix: self.noise_mix,
            noise_time: self.noise_time,
            peak_exponent: self.peak_exponent,
            peak_min: self.peak_min,
            text_offset_x: self.text_offset[0],
            text_offset_y: self.text_offset[1],
            text_offset_z: self.text_offset[2],
            glyph_rot_x: self.glyph_rot_x,
        };
        self.queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));

        // Upload wave params
        let wave_params = WaveParams {
            line_count: self.line_count,
            verts_per_line: VERTS_PER_LINE,
            half_width: self.line_half_width,
            active_line_idx: active_line_idx as u32,
            line_spacing: self.line_spacing,
            fft_amplitude: self.fft_amplitude,
            thickness: self.wave_thickness,
            noise_scale: self.noise_scale,
            noise_strength: self.noise_strength,
            noise_speed: self.noise_speed,
            noise_zoom: self.noise_zoom,
            noise_mix: self.noise_mix,
            noise_time: self.noise_time,
            peak_exponent: self.peak_exponent,
            peak_min: self.peak_min,
            _pad3: 0,
        };
        self.queue.write_buffer(&self.wave_params_buffer, 0, bytemuck::cast_slice(&[wave_params]));

        // Dispatch compute
        let wg = ((self.particle_count + 63) / 64).max(1);
        let wave_wg = ((self.wave_total_verts + 63) / 64).max(1);
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("JoyDiv/FftCompute") },
        );
        {
            let mut cp = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None },
            );

            // 1. Scroll elevation map (ping-pong)
            let scroll_pass = if self.ping_pong { &self.scroll_pass_ba } else { &self.scroll_pass_ab };
            scroll_pass.dispatch(&mut cp, 16, 16, 1); // 256/16 = 16 workgroups per axis

            // 2. Displace text particles from the just-written elevation buffer
            let displace_pass = if self.ping_pong { &self.displace_a } else { &self.displace_b };
            displace_pass.dispatch(&mut cp, wg, 1, 1);

            // 3. Displace wave line vertices from the same elevation buffer
            let wave_pass = if self.ping_pong { &self.wave_displace_a } else { &self.wave_displace_b };
            wave_pass.dispatch(&mut cp, wave_wg, 1, 1);

            // 4. Displace fill mesh vertices (same elevation, same workgroup count)
            let fill_pass = if self.ping_pong { &self.fill_displace_a } else { &self.fill_displace_b };
            fill_pass.dispatch(&mut cp, wave_wg, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // Toggle ping-pong for next frame
        self.ping_pong = !self.ping_pong;
    }

    /// Find the active line index based on current playback time.
    fn find_active_line(&self, current_time: f32) -> usize {
        let mut active = 0;
        for (i, &t) in self.timestamps.iter().enumerate() {
            if current_time >= t {
                active = i;
            } else {
                break;
            }
        }
        active
    }

    /// Public accessor for the positions buffer (shared with vertex instancing).
    pub fn positions_buffer(&self) -> &wgpu::Buffer {
        &self.positions_buffer
    }

    /// Public accessor for the wave positions buffer (shared as vertex source).
    pub fn wave_positions_buffer(&self) -> &wgpu::Buffer {
        &self.wave_positions_buffer
    }

    /// Public accessor for the fill positions buffer (solid occlusion mesh).
    pub fn fill_positions_buffer(&self) -> &wgpu::Buffer {
        &self.fill_positions_buffer
    }

    /// Number of lines in the wave mesh.
    pub fn wave_line_count(&self) -> u32 {
        self.line_count
    }

    /// Vertices per wave line.
    pub fn wave_verts_per_line(&self) -> u32 {
        VERTS_PER_LINE
    }
}
