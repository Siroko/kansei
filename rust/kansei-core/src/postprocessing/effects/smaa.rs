//! SMAA 1x, Jorge Jimenez et al.'s Enhanced Subpixel Morphological Antialiasing
//! (<https://github.com/iryoku/smaa>), whose shaders this ports and whose area texture it embeds
//! under their MIT licence:
//!
//! > Copyright (C) 2013 Jorge Jimenez (jorge@iryoku.com)
//! > Copyright (C) 2013 Jose I. Echevarria (joseignacioechevarria@gmail.com)
//! > Copyright (C) 2013 Belen Masia (bmasia@unizar.es)
//! > Copyright (C) 2013 Fernando Navarro (fernandn@microsoft.com)
//! > Copyright (C) 2013 Diego Gutierrez (diegog@unizar.es)
//! >
//! > Permission is hereby granted, free of charge, to any person obtaining a copy
//! > this software and associated documentation files (the "Software"), to deal in
//! > the Software without restriction, including without limitation the rights to
//! > use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
//! > of the Software, and to permit persons to whom the Software is furnished to
//! > do so, subject to the following conditions:
//! >
//! > The above copyright notice and this permission notice shall be included in
//! > all copies or substantial portions of the Software. As clarification, there
//! > is no requirement that the copyright notice and permission be included in
//! > binary distributions of the Software.
//! >
//! > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! > IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! > FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! > AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! > LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! > OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! > SOFTWARE.
//!
//! SMAA anti-aliases one frame on its own: it finds the frame's luma edges, measures how far each
//! edge's run reaches and how its ends cross, looks up in the precomputed area texture how much of
//! each pixel the true edge would have covered, and blends each pixel with its neighbour across
//! the edge by that much. There is no history, so fast motion stays crisp where a temporal
//! resolve smears; it is the scene's choice *instead of* [`TemporalAAEffect`](super::TemporalAAEffect),
//! never beside it.
//!
//! # Preset
//!
//! iryoku's High settings (threshold 0.1, local contrast adaptation factor 2, corner rounding
//! 25 %), with two changes: the orthogonal searches reach [`SmaaEffect::REACH`] (16) pixels each
//! way, half of High's 32 (High's 16 search steps cover 2 px each; Medium's 8 reach 16), and there
//! is no diagonal search (High has 8 diagonal steps). The three passes' reach is
//! [`SmaaOptions::three_pass_reach`].
//!
//! # Where it differs from iryoku's reference
//!
//! - Compute passes, not fragment passes, and the weights pass skips pixels with no edge by an
//!   early return instead of a stencil mask.
//! - The searches walk one pixel at a time over exact edge flags, where iryoku reads pairs of
//!   flags with one bilinear fetch and decodes their last step with the search texture: the
//!   search texture is not needed and not shipped. Its crossing edges (fetched at -0.25 px) are
//!   the two flags read and mixed 0.75 / 0.25.
//! - SMAA 1x only: no diagonal search, predication, colour or depth edge detection, temporal
//!   (T2x, 4x) or reprojection modes.
//! - No edges outside the frame: the luma is clamped to it, and the flags outside it are 0, so a
//!   run stops at the frame's border.
//! - The edges and weights are `rgba8unorm` (`rg8unorm` is no core WebGPU storage format).
//! - The blend decodes the two texels it mixes to linear light and encodes the result, as
//!   iryoku's sRGB reads do, explicitly (no sRGB views of the input), and skips that for a linear
//!   input; the edges are found on the encoded luma either way.
//!
//! # The area texture
//!
//! `smaa_area.bin` is the first 80 rows of iryoku's `Textures/AreaTex.h` (RG8, 160 x 80, the 1x
//! subtexture), made with:
//!
//! ```text
//! D=$(mktemp -d); S=$(mktemp -d)
//! curl -sSL https://raw.githubusercontent.com/iryoku/smaa/master/Textures/AreaTex.h -o "$D/AreaTex.h"
//! cat > "$S/area.py" <<'PY'
//! import re, sys
//! src = open(sys.argv[1], encoding='latin-1').read()
//! body = src[src.index('areaTexBytes[] = {'):]; body = body[:body.index('};')]
//! data = bytes(int(h, 16) for h in re.findall(r'0x([0-9a-fA-F]{2})', body))
//! assert len(data) == 160 * 560 * 2
//! sys.stdout.buffer.write(data[:80 * 320])   # rows 0-79: SMAA 1x's subtexture
//! PY
//! python3 -I "$S/area.py" "$D/AreaTex.h" > rust/kansei-core/src/postprocessing/effects/smaa_area.bin
//! shasum -a 256 rust/kansei-core/src/postprocessing/effects/smaa_area.bin
//! # a365373fbcece6d1589add8d993f3541217fdacbe2cc6ca4beb578cc1dfa9b3b
//! ```

use bytemuck::{Pod, Zeroable};

use super::FilmGrain;
use crate::cameras::Camera;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

const THREE_PASS_WGSL: &str = concat!(include_str!("../../shaders/smaa_common.wgsl"), include_str!("../../shaders/smaa_passes.wgsl"));
/// SMAA's area texture, its 1x subtexture (see the module docs).
const AREA: &[u8] = include_bytes!("smaa_area.bin");
const AREA_SIZE: (u32, u32) = (160, 80);
/// The edges and the weights between the three passes.
const INTERMEDIATE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// The input is sRGB-encoded (smaa_common.wgsl's SMAA_FLAG_ENCODED).
const FLAG_ENCODED: u32 = 1;
/// A dither after the blend (SMAA_FLAG_DITHER).
const FLAG_DITHER: u32 = 2;

/// How [`SmaaEffect`] runs SMAA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SmaaMode {
    /// Three compute passes (edges, blending weights, neighbourhood blending) over textures, the
    /// reference, at [`SmaaOptions::three_pass_reach`].
    #[default]
    ThreePass,
}

/// [`SmaaEffect`]'s settings; the defaults are the module docs' preset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmaaOptions {
    /// The luma step (of the sRGB-encoded signal) that makes an edge.
    pub threshold: f32,
    /// An edge is kept only where its step is at least 1 / this of the largest step round it.
    pub local_contrast: f32,
    /// How much of a corner's blending is kept (0..1; iryoku's SMAA_CORNER_ROUNDING / 100).
    pub corner_rounding: f32,
    /// How far (px) the three passes' searches reach each way, 1..=64.
    pub three_pass_reach: u32,
}

impl Default for SmaaOptions {
    fn default() -> Self {
        Self { threshold: 0.1, local_contrast: 2.0, corner_rounding: 0.25, three_pass_reach: SmaaEffect::REACH }
    }
}

// ── GPU layout (must match SmaaParams in smaa_common.wgsl) ──

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SmaaParamsGpu {
    size: [u32; 2],
    threshold: f32,
    local_contrast: f32,
    corner_rounding: f32,
    reach: u32,
    flags: u32,
    frame: u32,
    grain: f32,
    grain_size: f32,
    _pad: [u32; 2],
}

/// A compute pipeline and its (derived) bind group layout.
struct Pipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl Pipeline {
    fn derived(device: &wgpu::Device, module: &wgpu::ShaderModule, label: &str, entry_point: &str) -> Self {
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });
        let bgl = pipeline.get_bind_group_layout(0);
        Self { pipeline, bgl }
    }
}

/// The edges and weights textures at a frame size.
struct Targets {
    size: (u32, u32),
    edges: wgpu::Texture,
    edges_view: wgpu::TextureView,
    weights: wgpu::Texture,
    weights_view: wgpu::TextureView,
}

impl Targets {
    fn new(device: &wgpu::Device, (width, height): (u32, u32)) -> Self {
        let texture = |label| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: INTERMEDIATE_FORMAT,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let (edges, weights) = (texture("SMAA/Edges"), texture("SMAA/Weights"));
        let (edges_view, weights_view) = (edges.create_view(&Default::default()), weights.create_view(&Default::default()));
        Self { size: (width, height), edges, edges_view, weights, weights_view }
    }

    fn destroy(&self) {
        self.edges.destroy();
        self.weights.destroy();
    }
}

/// The three passes' pipelines and textures.
struct ThreePass {
    edges: Pipeline,
    weights: Pipeline,
    blend: Pipeline,
    targets: Option<Targets>,
}

struct Gpu {
    params: wgpu::Buffer,
    area_texture: wgpu::Texture,
    area: wgpu::TextureView,
    sampler: wgpu::Sampler,
    /// Made on the first three-pass frame.
    three: Option<ThreePass>,
}

/// SMAA 1x on the display signal, a single-frame anti-aliasing for scenes where
/// [`TemporalAAEffect`](super::TemporalAAEffect)'s history smears (fast pans, fast thin things):
/// pick one of the two, never both. See the module docs for the preset and the algorithm.
///
/// Put it last, after [`ToneMapEffect`](super::ToneMapEffect) and any display-space
/// [`ColorGradingEffect`](super::ColorGradingEffect): SMAA finds edges by luma contrast and wants
/// perceptual values. The tonemapper writes the display signal, sRGB-encoded with
/// `encode_srgb` (browsers' `Bgra8Unorm` surfaces) and display-linear without it (sRGB
/// surfaces), and the volume's blit copies the chain's last texture to the surface as is; tell
/// this effect which it is with [`set_film`](Self::set_film) (`FilmGrain::encode_srgb`). It finds
/// edges on the encoded luma either way and blends in linear light.
///
/// It renders at full resolution (it is no upscaler) and asks for no camera jitter.
///
/// ```ignore
/// let tonemap = ToneMapEffect::new(ToneMapOptions::for_surface(renderer.presentation_format()));
/// let mut smaa = SmaaEffect::new(SmaaOptions::default());
/// smaa.set_film(FilmGrain { encode_srgb: tonemap.options.encode_srgb, ..Default::default() });
/// let volume = PostProcessingVolume::new(&renderer, vec![Box::new(tonemap), Box::new(smaa)]);
/// ```
pub struct SmaaEffect {
    pub options: SmaaOptions,
    mode: SmaaMode,
    film: FilmGrain,
    frame: u32,
    gpu: Option<Gpu>,
    /// the bind groups its passes make from the frame's resources, kept
    bind_groups: crate::renderers::BindGroupCache,
}

impl SmaaEffect {
    /// How far (px) the searches reach each way.
    pub const REACH: u32 = 16;

    pub fn new(options: SmaaOptions) -> Self {
        Self { options, mode: SmaaMode::default(), film: FilmGrain::default(), frame: 0, gpu: None, bind_groups: Default::default() }
    }

    pub fn with_mode(mut self, mode: SmaaMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn mode(&self) -> SmaaMode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: SmaaMode) {
        self.mode = mode;
    }

    /// The signal it reads (`encode_srgb`) and the film it adds after blending.
    pub fn set_film(&mut self, film: FilmGrain) {
        self.film = film;
    }

    pub fn film(&self) -> FilmGrain {
        self.film
    }

    fn params(&self, width: u32, height: u32) -> SmaaParamsGpu {
        let (o, f) = (&self.options, &self.film);
        SmaaParamsGpu {
            size: [width, height],
            threshold: o.threshold,
            local_contrast: o.local_contrast,
            corner_rounding: o.corner_rounding.clamp(0.0, 1.0),
            reach: o.three_pass_reach.clamp(1, 64),
            flags: if f.encode_srgb { FLAG_ENCODED } else { 0 } | if f.dither { FLAG_DITHER } else { 0 },
            frame: self.frame,
            grain: f.grain.max(0.0),
            grain_size: f.grain_size.max(1.0),
            _pad: [0; 2],
        }
    }

    fn init_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SMAA/Params"),
            size: std::mem::size_of::<SmaaParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let size = wgpu::Extent3d { width: AREA_SIZE.0, height: AREA_SIZE.1, depth_or_array_layers: 1 };
        let area_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SMAA/Area"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &area_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            AREA,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(AREA_SIZE.0 * 2), rows_per_image: Some(AREA_SIZE.1) },
            size,
        );
        let area = area_texture.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SMAA/Linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        self.gpu = Some(Gpu { params, area_texture, area, sampler, three: None });
    }

    /// Edges, weights and blend, each its own compute pass (`params` already written).
    fn encode_three_pass(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, input: &wgpu::TextureView, output: &wgpu::TextureView, width: u32, height: u32) {
        {
            let three = self.gpu.as_mut().unwrap().three.get_or_insert_with(|| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("SMAA/ThreePass"), source: wgpu::ShaderSource::Wgsl(THREE_PASS_WGSL.into()) });
                ThreePass {
                    edges: Pipeline::derived(device, &module, "SMAA/Edges", "edges"),
                    weights: Pipeline::derived(device, &module, "SMAA/Weights", "weights"),
                    blend: Pipeline::derived(device, &module, "SMAA/Blend", "blend"),
                    targets: None,
                }
            });
            if three.targets.as_ref().map(|t| t.size) != Some((width, height)) {
                if let Some(old) = three.targets.take() {
                    old.destroy();
                }
                three.targets = Some(Targets::new(device, (width, height)));
            }
        }
        let gpu = self.gpu.as_ref().unwrap();
        let three = gpu.three.as_ref().unwrap();
        let targets = three.targets.as_ref().unwrap();
        let tex = wgpu::BindingResource::TextureView;
        let group = |pipeline: &Pipeline, entries: &[(u32, wgpu::BindingResource)]| {
            let entries: Vec<_> = entries.iter().map(|(binding, resource)| wgpu::BindGroupEntry { binding: *binding, resource: resource.clone() }).collect();
            self.bind_groups.get(device, &wgpu::BindGroupDescriptor { label: Some("SMAA/BG"), layout: &pipeline.bgl, entries: &entries })
        };
        let params = gpu.params.as_entire_binding();
        let passes = [
            ("SMAA/Edges", &three.edges, group(&three.edges, &[(0, params.clone()), (1, tex(input)), (6, tex(&targets.edges_view))])),
            (
                "SMAA/Weights",
                &three.weights,
                group(&three.weights, &[(0, params.clone()), (2, tex(&gpu.area)), (3, wgpu::BindingResource::Sampler(&gpu.sampler)), (5, tex(&targets.edges_view)), (8, tex(&targets.weights_view))]),
            ),
            ("SMAA/Blend", &three.blend, group(&three.blend, &[(0, params), (1, tex(input)), (4, tex(output)), (7, tex(&targets.weights_view))])),
        ];
        for (label, pipeline, bind_group) in &passes {
            let stamp = crate::profiling::gpu_pass(label);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
    }

    #[cfg(test)]
    pub(crate) fn three_pass_source() -> &'static str {
        THREE_PASS_WGSL
    }
}

impl PostProcessingEffect for SmaaEffect {
    fn initialize(&mut self, _device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {}

    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        _gbuffer: &GBuffer,
        input: &wgpu::TextureView,
        _depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        _camera: &Camera,
        width: u32,
        height: u32,
    ) {
        if self.gpu.is_none() {
            self.init_gpu(device, queue);
        }
        let params = self.params(width, height);
        self.frame = self.frame.wrapping_add(1);
        queue.write_buffer(&self.gpu.as_ref().unwrap().params, 0, bytemuck::bytes_of(&params));
        match self.mode {
            SmaaMode::ThreePass => self.encode_three_pass(device, encoder, input, output, width, height),
        }
    }

    // The intermediates follow the size `render` is given.
    fn resize(&mut self, _width: u32, _height: u32, _gbuffer: &GBuffer) {}

    fn destroy(&mut self) {
        if let Some(gpu) = self.gpu.take() {
            gpu.area_texture.destroy();
            if let Some(targets) = gpu.three.and_then(|t| t.targets) {
                targets.destroy();
            }
        }
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AREA: &[u8] = include_bytes!("smaa_area.bin");
    fn texel(x: usize, y: usize) -> [u8; 2] {
        [AREA[y * 320 + x * 2], AREA[y * 320 + x * 2 + 1]]
    }

    #[test]
    fn the_area_texture_is_smaa_s_1x_subtexture() {
        assert_eq!(AREA.len(), 160 * 80 * 2);
        assert_eq!(AREA.iter().map(|&b| b as u64).sum::<u64>(), 889_395);
        // no crossing edge at either end (e1 = e2 = 0): nothing to blend, whatever the distances
        assert!((0..16).all(|y| (0..16).all(|x| texel(x, y) == [0, 0])));
        assert_eq!((texel(16, 1), texel(48, 0), texel(0, 48), texel(1, 17)), ([0, 63], [31, 0], [31, 0], [0, 10]));
    }

    #[test]
    fn shaders_validate_and_params_layout_matches() {
        let code = SmaaEffect::three_pass_source();
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{}", e.emit_to_string(code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let span = module
            .types
            .iter()
            .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == "SmaaParams" => Some(*span as usize),
                _ => None,
            })
            .unwrap();
        assert_eq!(span, std::mem::size_of::<SmaaParamsGpu>());
        assert_eq!(span, 48);
    }

    #[test]
    fn smaa_never_asks_for_jitter_or_upscaling() {
        let smaa = SmaaEffect::new(Default::default());
        assert!(!smaa.wants_jitter() && !smaa.upscales_to_display());
    }
}
