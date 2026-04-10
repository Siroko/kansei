/// Describes what resource a binding points to.
pub enum BindingResource<'a> {
    Buffer {
        buffer: &'a wgpu::Buffer,
        offset: u64,
        size: Option<u64>,
    },
    TextureView(&'a wgpu::TextureView),
    Sampler(&'a wgpu::Sampler),
    StorageTexture(&'a wgpu::TextureView),
}

/// A single binding descriptor — binding index + visibility + resource type.
pub struct Binding {
    pub index: u32,
    pub visibility: wgpu::ShaderStages,
    pub ty: BindingType,
}

/// The type of a binding for layout creation.
pub enum BindingType {
    UniformBuffer,
    StorageBuffer { read_only: bool },
    Texture { sample_type: wgpu::TextureSampleType, view_dimension: wgpu::TextureViewDimension },
    StorageTexture { format: wgpu::TextureFormat, view_dimension: wgpu::TextureViewDimension },
    Sampler { filtering: bool },
}

impl Binding {
    pub fn uniform(index: u32, visibility: wgpu::ShaderStages) -> Self {
        Self { index, visibility, ty: BindingType::UniformBuffer }
    }

    pub fn storage(index: u32, visibility: wgpu::ShaderStages, read_only: bool) -> Self {
        Self { index, visibility, ty: BindingType::StorageBuffer { read_only } }
    }

    pub fn texture_2d(index: u32, visibility: wgpu::ShaderStages) -> Self {
        Self { index, visibility, ty: BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
        }}
    }

    pub fn texture_3d(index: u32, visibility: wgpu::ShaderStages) -> Self {
        Self { index, visibility, ty: BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D3,
        }}
    }

    pub fn texture_depth(index: u32, visibility: wgpu::ShaderStages) -> Self {
        Self { index, visibility, ty: BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
        }}
    }

    pub fn storage_texture_3d(index: u32, visibility: wgpu::ShaderStages, format: wgpu::TextureFormat) -> Self {
        Self { index, visibility, ty: BindingType::StorageTexture {
            format,
            view_dimension: wgpu::TextureViewDimension::D3,
        }}
    }

    pub fn storage_texture_2d(index: u32, visibility: wgpu::ShaderStages, format: wgpu::TextureFormat) -> Self {
        Self { index, visibility, ty: BindingType::StorageTexture {
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        }}
    }

    pub fn sampler(index: u32, visibility: wgpu::ShaderStages) -> Self {
        Self { index, visibility, ty: BindingType::Sampler { filtering: true } }
    }

    /// Convert to a wgpu BindGroupLayoutEntry.
    pub fn to_layout_entry(&self) -> wgpu::BindGroupLayoutEntry {
        let ty = match &self.ty {
            BindingType::UniformBuffer => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            BindingType::StorageBuffer { read_only } => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: *read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            BindingType::Texture { sample_type, view_dimension } => wgpu::BindingType::Texture {
                sample_type: *sample_type,
                view_dimension: *view_dimension,
                multisampled: false,
            },
            BindingType::StorageTexture { format, view_dimension } => wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: *format,
                view_dimension: *view_dimension,
            },
            BindingType::Sampler { filtering } => wgpu::BindingType::Sampler(
                if *filtering { wgpu::SamplerBindingType::Filtering } else { wgpu::SamplerBindingType::NonFiltering }
            ),
        };

        wgpu::BindGroupLayoutEntry {
            binding: self.index,
            visibility: self.visibility,
            ty,
            count: None,
        }
    }
}

/// Helper to build bind group layouts and bind groups from descriptors.
pub struct BindGroupBuilder;

impl BindGroupBuilder {
    /// Create a bind group layout from binding descriptors.
    pub fn create_layout(device: &wgpu::Device, label: &str, bindings: &[Binding]) -> wgpu::BindGroupLayout {
        let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings.iter().map(|b| b.to_layout_entry()).collect();
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        })
    }

    /// Create a bind group from a layout and resources.
    pub fn create_bind_group(
        device: &wgpu::Device,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        entries: &[(u32, BindingResource)],
    ) -> wgpu::BindGroup {
        let bg_entries: Vec<wgpu::BindGroupEntry> = entries.iter().map(|(binding, resource)| {
            wgpu::BindGroupEntry {
                binding: *binding,
                resource: match resource {
                    BindingResource::Buffer { buffer, offset, size } => {
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: *offset,
                            size: size.map(|s| std::num::NonZeroU64::new(s).unwrap()),
                        })
                    }
                    BindingResource::TextureView(view) => wgpu::BindingResource::TextureView(view),
                    BindingResource::StorageTexture(view) => wgpu::BindingResource::TextureView(view),
                    BindingResource::Sampler(sampler) => wgpu::BindingResource::Sampler(sampler),
                },
            }
        }).collect();

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &bg_entries,
        })
    }
}
