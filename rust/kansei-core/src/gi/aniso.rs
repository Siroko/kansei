const ANISO_MIP_WGSL: &str = include_str!("shaders/aniso_mip.wgsl");

/// The six directional mip chains of a `VoxelVolume` (`VoxelVolume::set_anisotropic_mips`):
/// from its mip 1 up, one `rgba16float` chain per direction a cone travels along an axis (+x, +y,
/// +z, -x, -y, -z), each voxel its children composited front to back along that direction
/// (aniso_mip.wgsl). About 0.86 times the memory of the volume's mip 0.
pub(crate) struct AnisotropicMips {
    textures: Vec<wgpu::Texture>,
    views: [wgpu::TextureView; 6],
    pipelines: [wgpu::ComputePipeline; 4],
    /// Per level and sign: the pipeline (index into `pipelines`), its bind group and its size.
    passes: Vec<(usize, wgpu::BindGroup, [u32; 3])>,
}

impl AnisotropicMips {
    /// Directions of `views`, in order.
    pub(crate) const DIRECTIONS: [&'static str; 6] = ["+x", "+y", "+z", "-x", "-y", "-z"];

    /// The chains of `volume` (an `rgba16float` 3D texture with a full mip chain).
    pub(crate) fn new(device: &wgpu::Device, volume: &wgpu::Texture) -> Self {
        let size = volume.size();
        let levels = volume.mip_level_count() - 1;
        let [w, h, d] = [size.width, size.height, size.depth_or_array_layers].map(|s| (s / 2).max(1));
        let textures: Vec<wgpu::Texture> = Self::DIRECTIONS
            .iter()
            .map(|_| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("VoxelGI/AnisotropicMips"),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: d },
                    mip_level_count: levels.max(1),
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                })
            })
            .collect();
        let views = std::array::from_fn(|i| textures[i].create_view(&Default::default()));
        let level = |texture: &wgpu::Texture, l: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor { label: Some("VoxelGI/AnisotropicLevel"), base_mip_level: l, mip_level_count: Some(1), ..Default::default() })
        };
        let iso = level(volume, 0);

        let compute = wgpu::ShaderStages::COMPUTE;
        let sampled = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
            count: None,
        };
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D3 },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VoxelGI/AnisotropicMipsBGL"),
            entries: &[sampled(0), sampled(1), sampled(2), sampled(3), storage(4), storage(5), storage(6)],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("VoxelGI/AnisotropicMips"), source: wgpu::ShaderSource::Wgsl(ANISO_MIP_WGSL.into()) });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("VoxelGI/AnisotropicMips"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipelines = ["first_pos", "first_neg", "down_pos", "down_neg"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("VoxelGI/AnisotropicMips"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });

        let mut passes = Vec::new();
        for l in 0..levels {
            for sign in 0..2 {
                let first = 3 * sign;
                let dst: Vec<_> = (first..first + 3).map(|i| level(&textures[i], l)).collect();
                // level 0 reads the isotropic mip 0 (and binds it in the unused slots); the levels
                // above read their own chains' level below
                let src: Vec<_> = (first..first + 3).map(|i| if l == 0 { iso.clone() } else { level(&textures[i], l - 1) }).collect();
                let entries: Vec<_> = std::iter::once(&iso)
                    .chain(&src)
                    .chain(&dst)
                    .enumerate()
                    .map(|(binding, view)| wgpu::BindGroupEntry { binding: binding as u32, resource: wgpu::BindingResource::TextureView(view) })
                    .collect();
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("VoxelGI/AnisotropicMipsBG"), layout: &bgl, entries: &entries });
                let pipeline = if l == 0 { sign } else { 2 + sign };
                passes.push((pipeline, group, [w, h, d].map(|s| (s >> l).max(1))));
            }
        }
        Self { textures, views, pipelines, passes }
    }

    /// Every level of each direction's chain (in `DIRECTIONS` order), to sample.
    pub(crate) fn views(&self) -> &[wgpu::TextureView; 6] {
        &self.views
    }

    /// Bytes on the GPU.
    pub(crate) fn memory_bytes(&self) -> u64 {
        self.textures
            .iter()
            .map(|t| {
                let s = t.size();
                (0..t.mip_level_count()).map(|l| [s.width, s.height, s.depth_or_array_layers].iter().map(|&d| (d >> l).max(1) as u64).product::<u64>() * 8).sum::<u64>()
            })
            .sum()
    }

    /// Record the chains' rebuild from the volume's mip 0.
    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let stamp = crate::profiling::gpu_pass("VoxelGI/AnisotropicMips");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VoxelGI/AnisotropicMips"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
        for (pipeline, group, [w, h, d]) in &self.passes {
            pass.set_pipeline(&self.pipelines[*pipeline]);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(w.div_ceil(4), h.div_ceil(4), d.div_ceil(4));
        }
    }
}
