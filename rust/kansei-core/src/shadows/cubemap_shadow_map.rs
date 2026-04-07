use crate::geometries::Vertex;
use crate::renderers::Renderer;

/// Face directions for cubemap rendering (+X, -X, +Y, -Y, +Z, -Z).
const FACE_DIRS: [[f32; 3]; 6] = [
    [1.0, 0.0, 0.0],   // +X
    [-1.0, 0.0, 0.0],  // -X
    [0.0, 1.0, 0.0],   // +Y
    [0.0, -1.0, 0.0],  // -Y
    [0.0, 0.0, 1.0],   // +Z
    [0.0, 0.0, -1.0],  // -Z
];

/// Up vectors for each cubemap face.
const FACE_UPS: [[f32; 3]; 6] = [
    [0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
    [0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
];

/// Number of floats per face uniform slot (mat4 + vec3 + 1 pad = 20 floats).
const FACE_UNIFORM_FLOATS: usize = 20;

/// Cubemap shadow map for point light shadows.
///
/// Renders 6 faces per light, storing linear distance in an R32Float texture
/// array. The scratch depth buffer is reused across all faces.
pub struct CubeMapShadowMap {
    pub resolution: u32,
    pub max_lights: u32,
    pub near: f32,
    pub shadow_far: f32,
    /// Distance texture: r32float, `[resolution, resolution, 6 * max_lights]` layers.
    pub distance_texture: wgpu::Texture,
    pub distance_view: wgpu::TextureView,
    /// Scratch depth: depth32float, reused per face render.
    scratch_depth_texture: wgpu::Texture,
    scratch_depth_view: wgpu::TextureView,
    /// Per-face light uniform buffer (dynamic offset per face).
    light_uniform_buf: wgpu::Buffer,
    light_uniform_bgl: wgpu::BindGroupLayout,
    light_uniform_bg: wgpu::BindGroup,
    /// Mesh bind group layout (shadow-specific, 2 dynamic-offset uniforms).
    mesh_bgl: wgpu::BindGroupLayout,
    mesh_world_buf: wgpu::Buffer,
    mesh_normal_buf: wgpu::Buffer,
    mesh_bg: wgpu::BindGroup,
    /// Render pipeline for cubemap shadow passes.
    pipeline: wgpu::RenderPipeline,
    // Internal bookkeeping
    matrix_alignment: u32,
    uniform_alignment: u32,
    face_vps: Vec<f32>,
    world_staging: Vec<f32>,
    normal_staging: Vec<f32>,
    object_capacity: usize,
}

impl CubeMapShadowMap {
    /// Create a new cubemap shadow map.
    ///
    /// - `resolution`: width/height of each face in pixels.
    /// - `max_lights`: maximum number of point lights that cast shadows.
    pub fn new(renderer: &Renderer, resolution: u32, max_lights: u32) -> Self {
        let device = renderer.device();
        let uniform_alignment = device.limits().min_uniform_buffer_offset_alignment;
        let matrix_alignment = uniform_alignment;

        // --- Distance texture (R32Float, 2D array with 6 * max_lights layers) ---
        let total_layers = 6 * max_lights;
        let distance_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("CubeMapShadow/Distance"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: total_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let distance_view = distance_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        // --- Scratch depth (Depth32Float, single face, reused) ---
        let scratch_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("CubeMapShadow/ScratchDepth"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let scratch_depth_view = scratch_depth_texture.create_view(&Default::default());

        // --- Light uniform buffer (dynamic offset per face) ---
        // Each slot: 20 floats (mat4 + vec3 + pad), padded to uniform_alignment.
        let slot_bytes = uniform_alignment.max((FACE_UNIFORM_FLOATS * 4) as u32);
        let light_buf_size = (total_layers as u64) * (slot_bytes as u64);
        let light_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CubeMapShadow/LightUniforms"),
            size: light_buf_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let light_uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CubeMapShadow/LightBGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new((FACE_UNIFORM_FLOATS * 4) as u64),
                },
                count: None,
            }],
        });

        let light_uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CubeMapShadow/LightBG"),
            layout: &light_uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &light_uniform_buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(slot_bytes as u64),
                }),
            }],
        });

        // --- Mesh BGL (shadow-specific, same shape as shared mesh BGL) ---
        let mesh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CubeMapShadow/MeshBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(64),
                    },
                    count: None,
                },
            ],
        });

        // Mesh buffers (start with 0 capacity; will grow on first use).
        let mesh_world_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CubeMapShadow/MeshWorld"),
            size: matrix_alignment as u64, // minimum 1 slot
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mesh_normal_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CubeMapShadow/MeshNormal"),
            size: matrix_alignment as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mesh_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CubeMapShadow/MeshBG"),
            layout: &mesh_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &mesh_normal_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &mesh_world_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(64),
                    }),
                },
            ],
        });

        // --- Render pipeline ---
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CubeMapShadow/PipelineLayout"),
            bind_group_layouts: &[&light_uniform_bgl, &mesh_bgl],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CubeMapShadow/Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/cubemap_shadow.wgsl").into(),
            ),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("CubeMapShadow/Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("shadow_vs"),
                buffers: &[Vertex::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("shadow_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        Self {
            resolution,
            max_lights,
            near: 0.1,
            shadow_far: 100.0,
            distance_texture,
            distance_view,
            scratch_depth_texture,
            scratch_depth_view,
            light_uniform_buf,
            light_uniform_bgl,
            light_uniform_bg,
            mesh_bgl,
            mesh_world_buf,
            mesh_normal_buf,
            mesh_bg,
            pipeline,
            matrix_alignment,
            uniform_alignment,
            face_vps: Vec::new(),
            world_staging: Vec::new(),
            normal_staging: Vec::new(),
            object_capacity: 0,
        }
    }

    /// Compute a view-projection matrix for one cubemap face.
    ///
    /// Returns a column-major `[f32; 16]` suitable for GPU upload.
    /// Uses a 90-degree perspective with WebGPU [0,1] depth range.
    pub fn compute_face_vp(light_pos: &[f32; 3], face: usize, shadow_far: f32) -> [f32; 16] {
        let near = 0.1_f32;
        let eye = glam::Vec3::from_array(*light_pos);
        let dir = glam::Vec3::from_array(FACE_DIRS[face]);
        let up = glam::Vec3::from_array(FACE_UPS[face]);
        let target = eye + dir;

        let view = glam::Mat4::look_at_rh(eye, target, up);
        // 90-degree FOV, aspect 1:1, WebGPU depth [0,1]
        let proj = glam::Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_2,
            1.0,
            near,
            shadow_far,
        );

        (proj * view).to_cols_array()
    }

    /// Grow mesh uniform buffers if the current capacity is insufficient.
    pub fn ensure_mesh_buffers(&mut self, device: &wgpu::Device, count: usize) {
        if count <= self.object_capacity && self.object_capacity > 0 {
            return;
        }
        let new_cap = count.max(4); // minimum 4 slots
        let buf_size = (new_cap as u64) * (self.matrix_alignment as u64);

        self.mesh_world_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CubeMapShadow/MeshWorld"),
            size: buf_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.mesh_normal_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CubeMapShadow/MeshNormal"),
            size: buf_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Recreate bind group with new buffers.
        self.mesh_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CubeMapShadow/MeshBG"),
            layout: &self.mesh_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.mesh_normal_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.mesh_world_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(64),
                    }),
                },
            ],
        });

        let floats_per_slot = self.matrix_alignment as usize / 4;
        let total_floats = new_cap * floats_per_slot;
        self.world_staging.resize(total_floats, 0.0);
        self.normal_staging.resize(total_floats, 0.0);

        self.object_capacity = new_cap;
    }

    /// Number of floats in the world staging buffer.
    pub fn world_staging_len(&self) -> usize {
        self.world_staging.len()
    }

    /// Write a world matrix into the staging buffer at the given object index.
    pub fn write_world_matrix(&mut self, index: usize, data: &[f32]) {
        let floats_per_slot = self.matrix_alignment as usize / 4;
        let offset = index * floats_per_slot;
        let end = (offset + 16).min(self.world_staging.len());
        let count = end - offset;
        self.world_staging[offset..offset + count].copy_from_slice(&data[..count]);
    }

    /// Write a normal matrix into the staging buffer at the given object index.
    pub fn write_normal_matrix(&mut self, index: usize, data: &[f32]) {
        let floats_per_slot = self.matrix_alignment as usize / 4;
        let offset = index * floats_per_slot;
        let end = (offset + 16).min(self.normal_staging.len());
        let count = end - offset;
        self.normal_staging[offset..offset + count].copy_from_slice(&data[..count]);
    }

    /// Upload mesh staging buffers to the GPU.
    pub fn upload_mesh_matrices(&self, queue: &wgpu::Queue) {
        if self.object_capacity == 0 { return; }
        queue.write_buffer(&self.mesh_world_buf, 0, bytemuck::cast_slice(&self.world_staging));
        queue.write_buffer(&self.mesh_normal_buf, 0, bytemuck::cast_slice(&self.normal_staging));
    }

    /// Upload the 6 face view-projection matrices and light position for one
    /// point light into the uniform buffer.
    pub fn upload_face_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        light_index: usize,
        light_pos: &[f32; 3],
        shadow_far: f32,
    ) {
        let slot_bytes = self.uniform_alignment.max((FACE_UNIFORM_FLOATS * 4) as u32) as usize;
        let slot_floats = slot_bytes / 4;

        // Ensure staging buffer is large enough for 6 faces.
        self.face_vps.resize(6 * slot_floats, 0.0);

        for face in 0..6 {
            let vp = Self::compute_face_vp(light_pos, face, shadow_far);
            let base = face * slot_floats;
            // mat4 (16 floats)
            self.face_vps[base..base + 16].copy_from_slice(&vp);
            // vec3 light_world_pos + 1 pad
            self.face_vps[base + 16] = light_pos[0];
            self.face_vps[base + 17] = light_pos[1];
            self.face_vps[base + 18] = light_pos[2];
            self.face_vps[base + 19] = 0.0; // pad
        }

        let buf_offset = (light_index * 6 * slot_bytes) as u64;
        let byte_data = bytemuck::cast_slice(&self.face_vps[..6 * slot_floats]);
        queue.write_buffer(&self.light_uniform_buf, buf_offset, byte_data);
    }

    /// Create a 2D texture view for a single face layer in the distance texture.
    ///
    /// `face_slot` = `light_index * 6 + face` (0-based).
    pub fn face_color_view(&self, face_slot: usize) -> wgpu::TextureView {
        self.distance_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("CubeMapShadow/FaceView"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: face_slot as u32,
            array_layer_count: Some(1),
            ..Default::default()
        })
    }

    /// Reference to the render pipeline.
    pub fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }

    /// Reference to the light uniform bind group (use with dynamic offsets).
    pub fn light_uniform_bg(&self) -> &wgpu::BindGroup {
        &self.light_uniform_bg
    }

    /// Reference to the mesh bind group (use with dynamic offsets).
    pub fn mesh_bg(&self) -> &wgpu::BindGroup {
        &self.mesh_bg
    }

    /// Reference to the mesh world matrix buffer.
    pub fn mesh_world_buf(&self) -> &wgpu::Buffer {
        &self.mesh_world_buf
    }

    /// Reference to the mesh normal matrix buffer.
    pub fn mesh_normal_buf(&self) -> &wgpu::Buffer {
        &self.mesh_normal_buf
    }

    /// Reference to the scratch depth view.
    pub fn scratch_depth_view(&self) -> &wgpu::TextureView {
        &self.scratch_depth_view
    }

    /// The uniform alignment (bytes) used for dynamic offsets.
    pub fn uniform_alignment(&self) -> u32 {
        self.uniform_alignment
    }

    /// The matrix alignment (bytes) used for mesh dynamic offsets.
    pub fn matrix_alignment(&self) -> u32 {
        self.matrix_alignment
    }
}
