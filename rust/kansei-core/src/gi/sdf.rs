use bytemuck::{Pod, Zeroable};

use super::volume::VolumeLayout;

pub(crate) const JUMP_FLOOD_WGSL: &str = include_str!("shaders/jump_flood.wgsl");

/// Where a `JumpFloodSdf`'s seeds come from.
pub enum SdfSeeds<'a> {
    /// A mesh voxelizer's surface buffers (`MeshVoxelizer::static_surfaces` and
    /// `dynamic_surfaces`, `SURFACE_WORDS_PER_VOXEL` u32 a voxel): a voxel holding a surface is a
    /// seed.
    Surfaces,
    /// A volume's mip 0 (`VoxelVolume::view`): a voxel at least `threshold` opaque is a seed
    /// (particles, analytic boxes).
    Opacity { radiance: &'a wgpu::TextureView, threshold: f32 },
}

/// The WGSL `SdfParams` (jump_flood.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SdfParamsGpu {
    dims: [u32; 3],
    step: u32,
    voxel_size: f32,
    threshold: f32,
    seed_mode: u32,
    has_dynamic: u32,
}

/// A seed field's two ping-pong textures and the passes that fill them (seed, then floods).
struct Chain {
    #[allow(dead_code)] // (kept alive with their views)
    textures: [wgpu::Texture; 2],
    views: [wgpu::TextureView; 2],
    /// (entry index, bind group) per pass, in order
    passes: Vec<(usize, wgpu::BindGroup)>,
    /// Which of `views` holds the result.
    result: usize,
}

/// A distance field over a voxel volume by jump flooding (miaumiau.cat/?p=1457's distance field,
/// on WebGPU compute; Rong and Tan 2006): the unsigned distance in metres from each voxel to the
/// nearest occupied one, in an `r32float` 3D texture over the same `VolumeLayout`, read with
/// `SDF_WGSL` (`sdfDistance`, `sdfSoftShadow`, `sdfAo`).
///
/// Seeds come from a mesh voxelizer's surfaces or a volume's opacity (`SdfSeeds`). With surfaces,
/// the static renderables' seeds flood only when they change (`encode_static`) and the dynamic
/// ones' every frame they exist (`encode_dynamic`); the distance pass takes the nearer of the two.
/// A flood is log2 of the volume's side passes plus two (JFA+2), 26 texel reads a voxel each.
pub struct JumpFloodSdf {
    layout: VolumeLayout,
    pipelines: [wgpu::ComputePipeline; 3],
    bgl: wgpu::BindGroupLayout,
    /// One per pass (the flood steps differ; each pass reads its own, so they never share a
    /// buffer a frame writes twice), written once.
    params: Vec<wgpu::Buffer>,
    distance_params: [wgpu::Buffer; 2],
    steps: Vec<u32>,
    static_chain: Chain,
    dynamic_chain: Option<Chain>,
    distance: wgpu::Texture,
    distance_view: wgpu::TextureView,
    distance_storage: wgpu::TextureView,
    /// (static only, with dynamic seeds) distance passes
    distance_groups: [Option<wgpu::BindGroup>; 2],
    dummy_buffer: wgpu::Buffer,
    dummy_volume: wgpu::TextureView,
}

impl JumpFloodSdf {
    /// A field over `layout`. With `SdfSeeds::Surfaces`, give the static surfaces now
    /// (`static_surfaces`) and the dynamic ones with `set_dynamic_surfaces` when they appear.
    pub fn new(device: &wgpu::Device, layout: VolumeLayout, seeds: SdfSeeds, static_surfaces: Option<&wgpu::Buffer>) -> Self {
        let compute = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty, count: None };
        let uint_texture = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VoxelGI/SdfBGL"),
            entries: &[
                entry(0, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
                entry(1, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }),
                entry(2, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false }),
                entry(3, uint_texture),
                entry(4, wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::R32Uint, view_dimension: wgpu::TextureViewDimension::D3 }),
                entry(5, uint_texture),
                entry(6, wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::R32Float, view_dimension: wgpu::TextureViewDimension::D3 }),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("VoxelGI/JumpFlood"), source: wgpu::ShaderSource::Wgsl(JUMP_FLOOD_WGSL.into()) });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("VoxelGI/JumpFlood"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipelines = ["seed", "flood", "distance"].map(|entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("VoxelGI/JumpFlood"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        });

        let (seed_mode, threshold, opacity): (u32, f32, Option<wgpu::TextureView>) = match seeds {
            SdfSeeds::Surfaces => (0, 0.0, None),
            SdfSeeds::Opacity { radiance, threshold } => (1, threshold, Some(radiance.clone())),
        };
        // the flood's steps: half the largest side (a power of two) down to 1, then 2 and 1 again
        let side = layout.dims.iter().copied().max().unwrap_or(1).next_power_of_two();
        let mut steps: Vec<u32> = std::iter::successors(Some((side / 2).max(1)), |&s| (s > 1).then_some(s / 2)).collect();
        steps.extend([2, 1]);
        use wgpu::util::DeviceExt;
        let params_with = |step: u32, has_dynamic: u32| {
            let gpu = SdfParamsGpu { dims: layout.dims, step, voxel_size: layout.voxel_size, threshold, seed_mode, has_dynamic };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("VoxelGI/SdfParams"), contents: bytemuck::bytes_of(&gpu), usage: wgpu::BufferUsages::UNIFORM })
        };
        // the seed pass's at 0, then one per flood step
        let params: Vec<wgpu::Buffer> = std::iter::once(0).chain(steps.iter().copied()).map(|s| params_with(s, 0)).collect();
        let distance_params = [params_with(0, 0), params_with(0, 1)];

        let dummy_buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("VoxelGI/SdfNoSurfaces"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let dummy_volume = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("VoxelGI/SdfNoVolume"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let distance = volume_texture(device, &layout, "VoxelGI/Sdf", wgpu::TextureFormat::R32Float);
        let distance_view = distance.create_view(&Default::default());
        let distance_storage = distance.create_view(&Default::default());
        let mut sdf = Self {
            layout,
            pipelines,
            bgl,
            params,
            distance_params,
            steps,
            static_chain: Chain::new(device, &layout, "VoxelGI/SdfStaticSeeds"),
            dynamic_chain: None,
            distance,
            distance_view,
            distance_storage,
            distance_groups: [None, None],
            dummy_buffer,
            dummy_volume,
        };
        let surfaces = static_surfaces.cloned();
        sdf.static_chain.passes = sdf.chain_passes(device, &sdf.static_chain, surfaces.as_ref(), opacity.as_ref());
        sdf.rebuild_distance_groups(device);
        sdf
    }

    /// The seed pass then the floods, ping-ponging between the chain's textures.
    fn chain_passes(&self, device: &wgpu::Device, chain: &Chain, surfaces: Option<&wgpu::Buffer>, opacity: Option<&wgpu::TextureView>) -> Vec<(usize, wgpu::BindGroup)> {
        let group = |params: &wgpu::Buffer, read: usize, write: usize| {
            let tex = wgpu::BindingResource::TextureView;
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("VoxelGI/SdfPass"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: surfaces.unwrap_or(&self.dummy_buffer).as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: tex(opacity.unwrap_or(&self.dummy_volume)) },
                    wgpu::BindGroupEntry { binding: 3, resource: tex(&chain.views[read]) },
                    wgpu::BindGroupEntry { binding: 4, resource: tex(&chain.views[write]) },
                    wgpu::BindGroupEntry { binding: 5, resource: tex(&chain.views[read]) },
                    wgpu::BindGroupEntry { binding: 6, resource: tex(&self.distance_storage) },
                ],
            })
        };
        // the seed pass writes 0 (it reads nothing; 1 stands in), each flood reads what the one
        // before wrote
        let mut passes = vec![(0, group(&self.params[0], 1, 0))];
        for k in 0..self.steps.len() {
            let (read, write) = if k % 2 == 0 { (0, 1) } else { (1, 0) };
            passes.push((1, group(&self.params[k + 1], read, write)));
        }
        passes
    }

    /// Which texture of a chain holds its result after its passes.
    fn result_index(&self) -> usize {
        // the seed pass writes 0; flood k writes 1 when k is even
        if self.steps.len() % 2 == 1 { 1 } else { 0 }
    }

    fn rebuild_distance_groups(&mut self, device: &wgpu::Device) {
        let result = self.result_index();
        self.static_chain.result = result;
        if let Some(chain) = &mut self.dynamic_chain {
            chain.result = result;
        }
        let statics = &self.static_chain.views;
        let group = |params: &wgpu::Buffer, dynamic: &wgpu::TextureView| {
            let tex = wgpu::BindingResource::TextureView;
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("VoxelGI/SdfDistance"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.dummy_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: tex(&self.dummy_volume) },
                    wgpu::BindGroupEntry { binding: 3, resource: tex(&statics[result]) },
                    // (unwritten: the other static texture)
                    wgpu::BindGroupEntry { binding: 4, resource: tex(&statics[1 - result]) },
                    wgpu::BindGroupEntry { binding: 5, resource: tex(dynamic) },
                    wgpu::BindGroupEntry { binding: 6, resource: tex(&self.distance_storage) },
                ],
            })
        };
        self.distance_groups[0] = Some(group(&self.distance_params[0], &statics[result]));
        self.distance_groups[1] = self.dynamic_chain.as_ref().map(|c| group(&self.distance_params[1], &c.views[result]));
    }

    /// Flood the dynamic renderables' seeds from `surfaces` (`MeshVoxelizer::dynamic_surfaces`),
    /// making their textures the first time.
    pub fn set_dynamic_surfaces(&mut self, device: &wgpu::Device, surfaces: Option<&wgpu::Buffer>) {
        match surfaces {
            None => self.dynamic_chain = None,
            Some(buffer) => {
                let mut chain = Chain::new(device, &self.layout, "VoxelGI/SdfDynamicSeeds");
                chain.passes = self.chain_passes(device, &chain, Some(buffer), None);
                self.dynamic_chain = Some(chain);
            }
        }
        self.rebuild_distance_groups(device);
    }

    pub fn has_dynamic(&self) -> bool {
        self.dynamic_chain.is_some()
    }

    fn encode_chain(&self, encoder: &mut wgpu::CommandEncoder, chain: &Chain, label: &'static str) {
        let [w, h, d] = self.layout.dims;
        let stamp = crate::profiling::gpu_pass(label);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
        for (entry, group) in &chain.passes {
            pass.set_pipeline(&self.pipelines[*entry]);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(w.div_ceil(4), h.div_ceil(4), d.div_ceil(4));
        }
    }

    /// Record the static seeds' flood (when they changed; then `encode_distance`).
    pub fn encode_static(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_chain(encoder, &self.static_chain, "VoxelGI/SdfStaticFlood");
    }

    /// Record the dynamic seeds' flood (every frame there are any; then `encode_distance`).
    pub fn encode_dynamic(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(chain) = &self.dynamic_chain {
            self.encode_chain(encoder, chain, "VoxelGI/SdfDynamicFlood");
        }
    }

    /// Record the distance pass: the nearer of the static and (if any) dynamic seeds.
    pub fn encode_distance(&self, encoder: &mut wgpu::CommandEncoder) {
        let [w, h, d] = self.layout.dims;
        let group = match (&self.distance_groups[1], &self.distance_groups[0]) {
            (Some(both), _) => both,
            (None, Some(statics)) => statics,
            _ => return,
        };
        let stamp = crate::profiling::gpu_pass("VoxelGI/SdfDistance");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VoxelGI/SdfDistance"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
        pass.set_pipeline(&self.pipelines[2]);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(w.div_ceil(4), h.div_ceil(4), d.div_ceil(4));
    }

    /// Record everything for a volume seeded by its opacity (`SdfSeeds::Opacity`), every frame.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_static(encoder);
        self.encode_distance(encoder);
    }

    pub fn layout(&self) -> &VolumeLayout {
        &self.layout
    }

    /// The field (metres), to bind as `texture_3d<f32>` and sample with the volume's sampler.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.distance_view
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.distance
    }

    /// The field as a `Texture` (shares the GPU texture), to attach to a material that reads it
    /// with `SDF_WGSL`: bind with `Binding::texture_3d`.
    pub fn as_texture(&self) -> crate::buffers::Texture {
        crate::buffers::Texture::from_view("VoxelGI/Sdf", self.distance.clone(), self.distance_view.clone())
    }

    /// Flood passes per rebuild (seed and distance aside).
    pub fn flood_passes(&self) -> usize {
        self.steps.len()
    }

    /// Bytes on the GPU: the field and the seed textures.
    pub fn memory_bytes(&self) -> u64 {
        let voxels = self.layout.voxel_count();
        voxels * 4 * (1 + 2 + 2 * self.dynamic_chain.is_some() as u64)
    }
}

/// A 3D texture over `layout`'s voxels, written by storage and read by sampling.
fn volume_texture(device: &wgpu::Device, layout: &VolumeLayout, label: &str, format: wgpu::TextureFormat) -> wgpu::Texture {
    let [w, h, d] = layout.dims;
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: d },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format,
        // (COPY_SRC: readable in tests)
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

impl Chain {
    fn new(device: &wgpu::Device, layout: &VolumeLayout, label: &str) -> Self {
        let textures = [volume_texture(device, layout, label, wgpu::TextureFormat::R32Uint), volume_texture(device, layout, label, wgpu::TextureFormat::R32Uint)];
        let views = [textures[0].create_view(&Default::default()), textures[1].create_view(&Default::default())];
        Self { textures, views, passes: Vec::new(), result: 0 }
    }
}
