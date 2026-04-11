use bytemuck::{Pod, Zeroable};
use super::{SurfaceContractVersion, SurfaceExtractionSourceContract, SurfaceMeshGpuContract};
use super::marching_cubes_tables::{EDGE_TABLE, TRI_TABLE};
use crate::renderers::Renderer;
use wgpu::util::DeviceExt;

const MC_RESET_WGSL: &str = include_str!("shaders/marching-cubes-reset.wgsl");
const MC_EXTRACT_WGSL: &str = include_str!("shaders/marching-cubes-extract.wgsl");
const MC_CLASSIC_WGSL: &str = include_str!("shaders/marching-cubes-classic.wgsl");
const MC_FINALIZE_WGSL: &str = include_str!("shaders/marching-cubes-finalize.wgsl");

/// Matches the standard engine `Vertex` layout: position(vec4) + normal(vec3) + uv(vec2) = 36 bytes.
/// This allows MC output to be consumed directly by standard Materials and the Renderer pipeline.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MarchingCubesVertex {
    pub position: [f32; 4],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MarchingCubesGpuParams {
    dims_and_max_tris: [u32; 4],
    bounds_min_and_iso: [f32; 4],
    bounds_max_pad: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
pub enum MarchingCubesGridSizing {
    FromSource,
    Fixed([u32; 3]),
    MaxAxis(u32),
}

#[derive(Clone, Copy, Debug)]
pub struct MarchingCubesSimulationParams {
    pub grid_sizing: MarchingCubesGridSizing,
    pub iso_level: f32,
    pub max_triangles: u32,
    pub normal_sample_step: f32,
    pub surface_smoothing: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct MarchingCubesOptions {
    pub max_triangles: u32,
    pub iso_level: f32,
}

impl Default for MarchingCubesOptions {
    fn default() -> Self {
        Self {
            max_triangles: 262_144,
            iso_level: 1.0,
        }
    }
}

impl Default for MarchingCubesSimulationParams {
    fn default() -> Self {
        Self {
            grid_sizing: MarchingCubesGridSizing::FromSource,
            iso_level: 1.0,
            max_triangles: 262_144,
            normal_sample_step: 1.0,
            surface_smoothing: 0.0,
        }
    }
}

/// GPU marching-cubes extraction contract for fluid surfaces.
///
/// Output layout is intentionally shared across render paths:
/// - `vertex_buffer`: `MarchingCubesVertex[]`
/// - `index_buffer`: `u32[]`
/// - `indirect_args_buffer`: DrawIndexedIndirect args (5 x u32)
pub struct MarchingCubesSimulation {
    params: MarchingCubesSimulationParams,
    params_buffer: wgpu::Buffer,
    tri_counter_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    indirect_args_buffer: wgpu::Buffer,
    /// Paul Bourke edge intersection bitmask table (256 × u32). Uploaded once.
    edge_table_buffer: wgpu::Buffer,
    /// Paul Bourke triangle lookup table (4096 × i32, sentinel -1). Uploaded once.
    tri_table_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    extract_bgl: wgpu::BindGroupLayout,
    reset_bg: wgpu::BindGroup,
    finalize_bg: wgpu::BindGroup,
    reset_pipeline: wgpu::ComputePipeline,
    extract_pipeline: wgpu::ComputePipeline,
    classic_extract_pipeline: wgpu::ComputePipeline,
    finalize_pipeline: wgpu::ComputePipeline,
    use_classic: bool,
}

impl MarchingCubesSimulation {
    fn new_with_device(device: &wgpu::Device, options: MarchingCubesOptions) -> Self {
        let params = MarchingCubesSimulationParams {
            max_triangles: options.max_triangles.max(1),
            iso_level: options.iso_level,
            ..Default::default()
        };
        let max_tris = params.max_triangles.max(1);
        let max_vertices = max_tris as u64 * 3;
        let max_indices = max_tris as u64 * 3;

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FluidMarchingCubes/Params"),
            size: std::mem::size_of::<MarchingCubesGpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tri_counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FluidMarchingCubes/TriangleCounter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FluidMarchingCubes/Vertices"),
            size: max_vertices * std::mem::size_of::<MarchingCubesVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FluidMarchingCubes/Indices"),
            size: max_indices * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let indirect_args_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FluidMarchingCubes/IndirectArgs"),
            size: 5 * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Marching-cubes lookup tables — uploaded once as uniform buffers so
        // Chrome/Dawn routes the lookups through the GPU's constant cache
        // (much faster than storage memory for random-access tables). In the
        // shader they're declared as `array<vec4<u32>, 64>` / `array<vec4<i32>, 1024>`
        // so the 16-byte uniform-array stride matches the natural packing
        // of 4 consecutive u32/i32 entries into one vec4 slot — meaning the
        // byte layout is identical to a flat storage buffer and we can
        // upload `EDGE_TABLE` / `TRI_TABLE` as-is with no repacking.
        let edge_table_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FluidMarchingCubes/EdgeTable"),
            contents: bytemuck::cast_slice(&EDGE_TABLE),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let tri_table_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FluidMarchingCubes/TriTable"),
            contents: bytemuck::cast_slice(&TRI_TABLE),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("FluidMarchingCubes/Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let reset_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FluidMarchingCubes/ResetShader"),
            source: wgpu::ShaderSource::Wgsl(MC_RESET_WGSL.into()),
        });
        let extract_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FluidMarchingCubes/ExtractShader"),
            source: wgpu::ShaderSource::Wgsl(MC_EXTRACT_WGSL.into()),
        });
        let finalize_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FluidMarchingCubes/FinalizeShader"),
            source: wgpu::ShaderSource::Wgsl(MC_FINALIZE_WGSL.into()),
        });

        let reset_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FluidMarchingCubes/ResetBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let reset_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FluidMarchingCubes/ResetLayout"),
            bind_group_layouts: &[&reset_bgl],
            push_constant_ranges: &[],
        });
        let reset_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FluidMarchingCubes/ResetPipeline"),
            layout: Some(&reset_layout),
            module: &reset_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let extract_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FluidMarchingCubes/ExtractBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 6 — EDGE_TABLE (classic MC only; voxel shell ignores it).
                // Uniform buffer so Dawn routes lookups through constant cache.
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 7 — TRI_TABLE (classic MC only; voxel shell ignores it).
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let extract_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FluidMarchingCubes/ExtractLayout"),
            bind_group_layouts: &[&extract_bgl],
            push_constant_ranges: &[],
        });
        let extract_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FluidMarchingCubes/ExtractPipeline"),
            layout: Some(&extract_layout),
            module: &extract_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let classic_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FluidMarchingCubes/ClassicShader"),
            source: wgpu::ShaderSource::Wgsl(MC_CLASSIC_WGSL.into()),
        });
        let classic_extract_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FluidMarchingCubes/ClassicExtractPipeline"),
            layout: Some(&extract_layout),
            module: &classic_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let finalize_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FluidMarchingCubes/FinalizeBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let finalize_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FluidMarchingCubes/FinalizeLayout"),
            bind_group_layouts: &[&finalize_bgl],
            push_constant_ranges: &[],
        });
        let finalize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FluidMarchingCubes/FinalizePipeline"),
            layout: Some(&finalize_layout),
            module: &finalize_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let reset_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FluidMarchingCubes/ResetBG"),
            layout: &reset_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tri_counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: indirect_args_buffer.as_entire_binding(),
                },
            ],
        });
        let finalize_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FluidMarchingCubes/FinalizeBG"),
            layout: &finalize_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tri_counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: indirect_args_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            params,
            params_buffer,
            tri_counter_buffer,
            vertex_buffer,
            index_buffer,
            indirect_args_buffer,
            edge_table_buffer,
            tri_table_buffer,
            sampler,
            extract_bgl,
            reset_bg,
            finalize_bg,
            reset_pipeline,
            extract_pipeline,
            classic_extract_pipeline,
            finalize_pipeline,
            use_classic: false,
        }
    }

    pub fn new(renderer: &Renderer, options: MarchingCubesOptions) -> Self {
        Self::new_with_device(renderer.raw_device(), options)
    }

    fn resolve_dims(&self, source_dims: [u32; 3]) -> [u32; 3] {
        let src = [source_dims[0].max(1), source_dims[1].max(1), source_dims[2].max(1)];
        match self.params.grid_sizing {
            MarchingCubesGridSizing::FromSource => src,
            MarchingCubesGridSizing::Fixed(d) => [d[0].max(1), d[1].max(1), d[2].max(1)],
            MarchingCubesGridSizing::MaxAxis(max_axis) => {
                let max_axis = max_axis.max(1) as f32;
                let sx = src[0] as f32;
                let sy = src[1] as f32;
                let sz = src[2] as f32;
                let max_src = sx.max(sy).max(sz).max(1.0);
                [
                    ((sx / max_src) * max_axis).round().max(1.0) as u32,
                    ((sy / max_src) * max_axis).round().max(1.0) as u32,
                    ((sz / max_src) * max_axis).round().max(1.0) as u32,
                ]
            }
        }
    }

    fn create_bind_group_with_device(
        &self,
        device: &wgpu::Device,
        density_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FluidMarchingCubes/ExtractBG"),
            layout: &self.extract_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(density_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.tri_counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.edge_table_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.tri_table_buffer.as_entire_binding(),
                },
            ],
        })
    }

    pub fn create_bind_group(
        &self,
        renderer: &Renderer,
        density_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        self.create_bind_group_with_device(renderer.raw_device(), density_view)
    }

    fn update_with_queue(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        extract_bind_group: &wgpu::BindGroup,
        density_dims: [u32; 3],
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
    ) {
        let effective_dims = self.resolve_dims(density_dims);
        let params = MarchingCubesGpuParams {
            dims_and_max_tris: [
                effective_dims[0].max(1),
                effective_dims[1].max(1),
                effective_dims[2].max(1),
                self.params.max_triangles.max(1),
            ],
            bounds_min_and_iso: [
                bounds_min[0],
                bounds_min[1],
                bounds_min[2],
                self.params.iso_level,
            ],
            bounds_max_pad: [bounds_max[0], bounds_max[1], bounds_max[2], 0.0],
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FluidMarchingCubes/Reset"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.reset_pipeline);
            pass.set_bind_group(0, &self.reset_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        {
            let pipeline = if self.use_classic {
                &self.classic_extract_pipeline
            } else {
                &self.extract_pipeline
            };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FluidMarchingCubes/Extract"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, extract_bind_group, &[]);
            pass.dispatch_workgroups(
                effective_dims[0].max(1).div_ceil(4),
                effective_dims[1].max(1).div_ceil(4),
                effective_dims[2].max(1).div_ceil(4),
            );
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FluidMarchingCubes/Finalize"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.finalize_pipeline);
            pass.set_bind_group(0, &self.finalize_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    /// Update with explicit encoder + queue (no Renderer needed).
    pub fn update_with_encoder_and_queue(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        extract_bind_group: &wgpu::BindGroup,
        source: SurfaceExtractionSourceContract,
    ) {
        if source.version != SurfaceContractVersion::V1 { return; }
        self.update_with_queue(encoder, queue, extract_bind_group,
            source.field_dims, source.world_bounds_min, source.world_bounds_max);
    }

    pub fn update_with_encoder(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &Renderer,
        extract_bind_group: &wgpu::BindGroup,
        density_dims: [u32; 3],
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
    ) {
        self.update_with_queue(
            encoder,
            renderer.raw_queue(),
            extract_bind_group,
            density_dims,
            bounds_min,
            bounds_max,
        );
    }

    pub fn update(
        &self,
        renderer: &Renderer,
        extract_bind_group: &wgpu::BindGroup,
        density_dims: [u32; 3],
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
    ) {
        let mut encoder = renderer.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FluidMarchingCubes/UpdateEncoder"),
        });
        self.update_with_queue(
            &mut encoder,
            renderer.raw_queue(),
            extract_bind_group,
            density_dims,
            bounds_min,
            bounds_max,
        );
        renderer.submit(std::iter::once(encoder.finish()));
    }

    pub fn update_from_source_with_encoder(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &Renderer,
        extract_bind_group: &wgpu::BindGroup,
        source: SurfaceExtractionSourceContract,
    ) {
        if source.version != SurfaceContractVersion::V1 {
            return;
        }
        self.update_with_queue(
            encoder,
            renderer.raw_queue(),
            extract_bind_group,
            source.field_dims,
            source.world_bounds_min,
            source.world_bounds_max,
        );
    }

    pub fn update_from_source(
        &self,
        renderer: &Renderer,
        extract_bind_group: &wgpu::BindGroup,
        source: SurfaceExtractionSourceContract,
    ) {
        if source.version != SurfaceContractVersion::V1 {
            return;
        }
        let mut encoder = renderer.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FluidMarchingCubes/UpdateFromSourceEncoder"),
        });
        self.update_with_queue(
            &mut encoder,
            renderer.raw_queue(),
            extract_bind_group,
            source.field_dims,
            source.world_bounds_min,
            source.world_bounds_max,
        );
        renderer.submit(std::iter::once(encoder.finish()));
    }

    pub fn mesh_contract(&self) -> SurfaceMeshGpuContract<'_> {
        SurfaceMeshGpuContract {
            version: SurfaceContractVersion::V1,
            vertex_buffer: &self.vertex_buffer,
            index_buffer: &self.index_buffer,
            indirect_args_buffer: &self.indirect_args_buffer,
            vertex_stride: std::mem::size_of::<MarchingCubesVertex>() as u64,
        }
    }

    pub fn set_params(&mut self, mut params: MarchingCubesSimulationParams) {
        params.max_triangles = params.max_triangles.max(1);
        params.iso_level = params.iso_level.max(0.0);
        params.normal_sample_step = params.normal_sample_step.max(0.0001);
        params.surface_smoothing = params.surface_smoothing.max(0.0);
        self.params = params;
    }

    pub fn params(&self) -> MarchingCubesSimulationParams {
        self.params
    }

    pub fn set_use_classic(&mut self, classic: bool) {
        self.use_classic = classic;
    }

    pub fn use_classic(&self) -> bool {
        self.use_classic
    }

    pub fn set_iso_level(&mut self, iso_level: f32) {
        self.params.iso_level = iso_level.max(0.0);
    }

    pub fn iso_level(&self) -> f32 {
        self.params.iso_level
    }

    pub fn max_triangles(&self) -> u32 {
        self.params.max_triangles
    }

    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertex_buffer
    }

    pub fn index_buffer(&self) -> &wgpu::Buffer {
        &self.index_buffer
    }

    pub fn indirect_args_buffer(&self) -> &wgpu::Buffer {
        &self.indirect_args_buffer
    }

    pub fn triangle_counter_buffer(&self) -> &wgpu::Buffer {
        &self.tri_counter_buffer
    }
}

pub type FluidMarchingCubes = MarchingCubesSimulation;
