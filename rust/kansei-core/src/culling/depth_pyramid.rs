const WGSL: &str = include_str!("../shaders/depth_pyramid.wgsl");

/// What each texel of a `DepthPyramid` keeps of the depths it covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthReduction {
    /// The largest depth (`r32float`): the farthest with the renderer's `[0, 1]` depth, which is
    /// what an occlusion test compares against.
    Max,
    /// The smallest depth (`r32float`): the farthest with reverse Z.
    Min,
    /// Both (`rg32float`): r the smallest, g the largest.
    MinMax,
}

impl DepthReduction {
    pub fn format(self) -> wgpu::TextureFormat {
        match self {
            DepthReduction::Max | DepthReduction::Min => wgpu::TextureFormat::R32Float,
            DepthReduction::MinMax => wgpu::TextureFormat::Rg32Float,
        }
    }

    fn mode(self) -> u32 {
        match self {
            DepthReduction::Max => 0,
            DepthReduction::Min => 1,
            DepthReduction::MinMax => 2,
        }
    }

    fn shader(self) -> String {
        let format = match self.format() {
            wgpu::TextureFormat::R32Float => "r32float",
            _ => "rg32float",
        };
        WGSL.replace("MODE_VALUE", &format!("{}u", self.mode())).replace("FORMAT", format)
    }
}

/// A hierarchical depth ("Hi-Z") pyramid of a depth buffer, built by compute.
///
/// Mip 0 is half the depth buffer's size, rounded up to a power of two in each axis, and each mip
/// halves the one before, down to 1 x 1. Texel `(x, y)` of mip `L` reduces exactly the depth
/// pixels `[x, x + 1) * 2^(L + 1)` in each axis (clipped to the buffer; texels wholly past it
/// hold edge values and are never needed), so a query maps a pixel rectangle to texels by
/// shifting its corners right by `L + 1`, for any buffer size: a rectangle whose extent in pixels
/// is under `2^(L + 1)` touches at most 2 x 2 texels of mip `L`.
///
/// ```ignore
/// let mut pyramid = DepthPyramid::new(device, width, height, DepthReduction::Max);
/// pyramid.build(device, &mut encoder, &gbuffer.depth_view); // after the depth is drawn
/// // bind pyramid.view() as texture_2d<f32> and textureLoad(pyramid, texel, level)
/// ```
pub struct DepthPyramid {
    reduction: DepthReduction,
    source_size: (u32, u32),
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    // per mip: its size and its storage view; mips 1.. also their bind group (reading the mip before)
    mips: Vec<Mip>,
    from_depth: wgpu::ComputePipeline,
    from_mip: wgpu::ComputePipeline,
    depth_bgl: wgpu::BindGroupLayout,
    mip_bgl: wgpu::BindGroupLayout,
}

struct Mip {
    size: (u32, u32),
    storage: wgpu::TextureView,
    bind_group: Option<wgpu::BindGroup>,
}

/// The size of each mip for a depth buffer of `width` x `height`: from half the buffer, rounded
/// up to a power of two (so that every mip halves exactly, as the texture's mip chain does), down
/// to 1 x 1.
pub(crate) fn mip_sizes(width: u32, height: u32) -> Vec<(u32, u32)> {
    let base = |n: u32| n.div_ceil(2).max(1).next_power_of_two();
    let mut sizes = vec![(base(width), base(height))];
    while sizes.last() != Some(&(1, 1)) {
        let (w, h) = *sizes.last().unwrap();
        sizes.push(((w / 2).max(1), (h / 2).max(1)));
    }
    sizes
}

impl DepthPyramid {
    /// A pyramid for a depth buffer of `width` x `height` (a `texture_depth_2d`, single-sampled).
    pub fn new(device: &wgpu::Device, width: u32, height: u32, reduction: DepthReduction) -> Self {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let dst = entry(
            2,
            wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: reduction.format(),
                view_dimension: wgpu::TextureViewDimension::D2,
            },
        );
        let depth_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DepthPyramid/FromDepthBGL"),
            entries: &[
                entry(0, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }),
                dst,
            ],
        });
        let mip_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DepthPyramid/FromMipBGL"),
            entries: &[
                entry(1, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }),
                dst,
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("DepthPyramid"), source: wgpu::ShaderSource::Wgsl(reduction.shader().into()) });
        let pipeline = |bgl: &wgpu::BindGroupLayout, entry_point: &str| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("DepthPyramid"), bind_group_layouts: &[bgl], push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("DepthPyramid/{entry_point}")),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let from_depth = pipeline(&depth_bgl, "from_depth");
        let from_mip = pipeline(&mip_bgl, "from_mip");
        let (texture, view, mips) = Self::create_mips(device, &mip_bgl, width, height, reduction);
        Self { reduction, source_size: (width, height), texture, view, mips, from_depth, from_mip, depth_bgl, mip_bgl }
    }

    fn create_mips(device: &wgpu::Device, mip_bgl: &wgpu::BindGroupLayout, width: u32, height: u32, reduction: DepthReduction) -> (wgpu::Texture, wgpu::TextureView, Vec<Mip>) {
        let sizes = mip_sizes(width, height);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DepthPyramid"),
            size: wgpu::Extent3d { width: sizes[0].0, height: sizes[0].1, depth_or_array_layers: 1 },
            mip_level_count: sizes.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: reduction.format(),
            // (COPY_SRC: readable for debugging and tests)
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor { label: Some("DepthPyramid"), ..Default::default() });
        let mip_view = |level: usize| {
            texture.create_view(&wgpu::TextureViewDescriptor { label: Some("DepthPyramid/Mip"), base_mip_level: level as u32, mip_level_count: Some(1), ..Default::default() })
        };
        let storage: Vec<wgpu::TextureView> = (0..sizes.len()).map(mip_view).collect();
        let mips = sizes
            .iter()
            .enumerate()
            .map(|(level, &size)| Mip {
                size,
                storage: storage[level].clone(),
                bind_group: (level > 0).then(|| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("DepthPyramid/FromMip"),
                        layout: mip_bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&mip_view(level - 1)) },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&storage[level]) },
                        ],
                    })
                }),
            })
            .collect();
        (texture, view, mips)
    }

    /// Resize for a depth buffer of `width` x `height` (a no-op at the current size). The
    /// texture and its views are recreated: rebind them.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.source_size == (width, height) {
            return;
        }
        let (texture, view, mips) = Self::create_mips(device, &self.mip_bgl, width, height, self.reduction);
        (self.texture, self.view, self.mips) = (texture, view, mips);
        self.source_size = (width, height);
    }

    /// Record the build from `depth` (a single-sampled depth view of `source_size()`): one
    /// compute pass, one dispatch per mip.
    pub fn build(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, depth: &wgpu::TextureView) {
        let first = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DepthPyramid/FromDepth"),
            layout: &self.depth_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.mips[0].storage) },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("DepthPyramid"), ..Default::default() });
        for (level, mip) in self.mips.iter().enumerate() {
            if level == 0 {
                pass.set_pipeline(&self.from_depth);
                pass.set_bind_group(0, &first, &[]);
            } else {
                if level == 1 {
                    pass.set_pipeline(&self.from_mip);
                }
                pass.set_bind_group(0, mip.bind_group.as_ref().unwrap(), &[]);
            }
            pass.dispatch_workgroups(mip.size.0.div_ceil(8), mip.size.1.div_ceil(8), 1);
        }
    }

    pub fn reduction(&self) -> DepthReduction {
        self.reduction
    }

    /// The pyramid's texture (`reduction().format()`, `mip_count()` mips).
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// A view of every mip, to bind as `texture_2d<f32>` and read with `textureLoad`.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn mip_count(&self) -> u32 {
        self.mips.len() as u32
    }

    /// The size of mip `level`.
    pub fn mip_size(&self, level: u32) -> (u32, u32) {
        self.mips[level as usize].size
    }

    /// The size of the depth buffer it is built from.
    pub fn source_size(&self) -> (u32, u32) {
        self.source_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mips_halve_exactly_down_to_one_texel() {
        assert_eq!(mip_sizes(8, 4), vec![(4, 2), (2, 1), (1, 1)]);
        assert_eq!(mip_sizes(5, 3), vec![(4, 2), (2, 1), (1, 1)]);
        assert_eq!(mip_sizes(1, 1), vec![(1, 1)]);
        assert_eq!(mip_sizes(1920, 1080)[0], (1024, 1024));
        assert_eq!(mip_sizes(1440, 810)[0], (1024, 512));
        assert_eq!(mip_sizes(1920, 1080).len(), 11);
        // the top mip covers the whole buffer: (size - 1) >> (levels) == 0
        for (w, h) in [(1920, 1080), (1287, 723), (37, 23), (2, 1), (3, 1)] {
            let levels = mip_sizes(w, h).len() as u32;
            assert_eq!(((w - 1) >> levels, (h - 1) >> levels), (0, 0), "{w} x {h}");
        }
    }

    #[test]
    fn shaders_validate() {
        for reduction in [DepthReduction::Max, DepthReduction::Min, DepthReduction::MinMax] {
            let source = reduction.shader();
            let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{reduction:?}: {e:?}"));
        }
    }
}
