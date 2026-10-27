use crate::cameras::Camera;
use crate::lights::Light;
use crate::objects::Scene;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::{GBuffer, Renderer};

use super::{BVHBuilder, GPUBVHData, PathTracer, PathTracerDenoise, PathTracerMaterial, TLASBuilder};

/// High-level wrapper that orchestrates the full path tracing pipeline
/// (BVH build, trace, denoise, output) as a single [`PostProcessingEffect`].
///
/// # Pipeline stages
///
/// 1. **Trace** — dispatch the path trace compute shader using the BLAS/TLAS
///    acceleration structure built at construction time, with the lights from
///    `set_lights`/`set_lights_from_scene`, writing the guide buffers the
///    denoiser reads (from the traced surfaces, not the GBuffer).
/// 2. **Denoise** — [`PathTracerDenoise`] (SVGF), configured through `denoise`
///    (its mode `Off` shows each traced frame as it is).
/// 3. **Output** — the denoised radiance, in HDR: the trace already holds direct
///    light, albedo, emission and the sky, so the effect replaces its input and
///    leaves exposure and the tone curve to a `ToneMapEffect` after it.
pub struct PathTracerEffect {
    #[allow(dead_code)]
    bvh: BVHBuilder,
    tlas: TLASBuilder,
    tracer: PathTracer,
    /// The denoiser and its settings.
    pub denoise: PathTracerDenoise,
    gpu_data: Option<GPUBVHData>,
    prev_vp: [f32; 16],
    /// Samples per pixel per frame (default 1).
    pub spp: u32,
    /// Maximum ray bounce depth (default 4).
    pub max_bounces: u32,
    light_count: u32,
    initialized: bool,
}

impl PathTracerEffect {
    /// Build BVH from the scene and create all sub-pipelines.
    ///
    /// This is a relatively expensive operation: it packs triangles,
    /// runs SAH construction on the CPU, collapses to BVH4, uploads
    /// buffers, and dispatches the GPU TLAS build.
    pub fn new(renderer: &Renderer, scene: &Scene) -> Self {
        let mut bvh = BVHBuilder::new();
        let mut tlas = TLASBuilder::new(renderer);
        let gpu_data = bvh.build_full(renderer, scene, &mut tlas);

        let mut tracer = PathTracer::new(renderer);
        tracer.set_materials(&[PathTracerMaterial::default()]);
        tracer.set_guides(true);

        Self {
            bvh,
            tlas,
            tracer,
            denoise: PathTracerDenoise::new(renderer),
            gpu_data: Some(gpu_data),
            prev_vp: [0.0; 16],
            spp: 1,
            max_bounces: 4,
            light_count: 0,
            initialized: false,
        }
    }

    /// Upload path tracer materials (one per renderable in the scene).
    pub fn set_materials(&mut self, materials: &[PathTracerMaterial]) {
        self.tracer.set_materials(materials);
    }

    /// Trace `lights` (directional, point and area; spot lights are skipped).
    pub fn set_lights(&mut self, lights: &[Light]) {
        self.light_count = self.tracer.set_lights(lights);
    }

    /// `set_lights` with the scene's lights.
    pub fn set_lights_from_scene(&mut self, scene: &Scene) {
        self.light_count = self.tracer.set_lights_from_scene(scene);
    }

    /// Upload raw light data for the trace shader (16 floats per light).
    pub fn set_lights_raw(&mut self, data: &[f32]) {
        self.tracer.set_lights_raw(data);
        self.light_count = (data.len() / 16) as u32;
    }

    /// Returns the GPU BVH data (for external use or inspection).
    pub fn gpu_data(&self) -> Option<&GPUBVHData> {
        self.gpu_data.as_ref()
    }

    /// Returns a reference to the underlying TLAS builder.
    pub fn tlas(&self) -> &TLASBuilder {
        &self.tlas
    }

    /// Returns a reference to the underlying path tracer.
    pub fn tracer(&self) -> &PathTracer {
        &self.tracer
    }

    /// Returns a mutable reference to the underlying path tracer.
    pub fn tracer_mut(&mut self) -> &mut PathTracer {
        &mut self.tracer
    }

    /// The full pipeline into `output` (an rgba16float storage texture), tracing `light_count`
    /// of the lights uploaded last (what `PathTracer::set_lights` returned). As a
    /// [`PostProcessingEffect`] it does the same with the count it keeps.
    pub fn render_with_gbuffer(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        gbuffer: &GBuffer,
        output: &wgpu::TextureView,
        camera: &Camera,
        light_count: u32,
    ) {
        self.light_count = light_count;
        self.run(encoder, output, camera, gbuffer.width, gbuffer.height);
    }

    fn run(&mut self, encoder: &mut wgpu::CommandEncoder, output: &wgpu::TextureView, camera: &Camera, width: u32, height: u32) {
        // Ensure internal textures match viewport size
        self.tracer.resize(width, height);

        // Apply configuration
        self.tracer.set_spp(self.spp);
        self.tracer.set_max_bounces(self.max_bounces);

        // The trace's own output averages every frame since its last reset (`tracer().output_view()`):
        // a moved camera starts it again. The denoiser reads each frame's guides.
        let vp = camera.projection_matrix.mul(&camera.view_matrix);
        let curr_vp = *vp.as_slice();
        if curr_vp != self.prev_vp {
            self.tracer.reset_accumulation();
        }

        // 1. Trace
        if let Some(ref gpu_data) = self.gpu_data {
            if let Some(tlas_buf) = self.tlas.tlas_nodes_buf.as_ref() {
                self.tracer
                    .trace(encoder, gpu_data, tlas_buf, camera, self.light_count);
            }
        }

        // 2. Denoise into the output
        self.denoise.denoise_into(encoder, &self.tracer, camera, output);

        self.prev_vp = curr_vp;
    }

    /// Reset temporal accumulation and the denoiser's history (call after a scene change).
    pub fn reset_accumulation(&mut self) {
        self.tracer.reset_accumulation();
        self.denoise.reset();
    }
}

impl PostProcessingEffect for PathTracerEffect {
    fn initialize(&mut self, _device: &wgpu::Device, gbuffer: &GBuffer, _camera: &Camera) {
        if self.initialized {
            return;
        }
        self.tracer.resize(gbuffer.width, gbuffer.height);
        self.initialized = true;
    }

    fn render(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        _gbuffer: &GBuffer,
        _input: &wgpu::TextureView,
        _depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        camera: &Camera,
        width: u32,
        height: u32,
    ) {
        self.run(encoder, output, camera, width, height);
    }

    fn resize(&mut self, width: u32, height: u32, _gbuffer: &GBuffer) {
        self.tracer.resize(width, height);
    }

    fn destroy(&mut self) {
        // Resources are dropped automatically when the struct is dropped.
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
