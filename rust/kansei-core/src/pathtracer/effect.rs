use crate::cameras::Camera;
use crate::lights::Light;
use crate::objects::Scene;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::{GBuffer, Renderer};

use super::{
    BVHBuilder, Compositor, GPUBVHData, PathTracer, PathTracerMaterial, SpatialDenoise,
    TLASBuilder, TemporalDenoise,
};

/// High-level wrapper that orchestrates the full path tracing pipeline
/// (BVH build, trace, temporal denoise, spatial denoise, output) as a
/// single [`PostProcessingEffect`].
///
/// # Pipeline stages
///
/// 1. **Trace** — dispatch the path trace compute shader using the BLAS/TLAS
///    acceleration structure built at construction time, with the lights from
///    `set_lights`/`set_lights_from_scene`. Frames accumulate while the camera
///    holds still and restart when it moves.
/// 2. **Temporal denoise** — motion-compensated reprojection blending the
///    current noisy frame with the accumulated history.
/// 3. **Spatial denoise** — A-trous wavelet filter guided by depth, normals,
///    and variance moments.
/// 4. **Output** — the denoised radiance, in HDR: the trace already holds direct
///    light, albedo, emission and the sky, so the effect replaces its input and
///    leaves exposure and the tone curve to a `ToneMapEffect` after it.
pub struct PathTracerEffect {
    bvh: BVHBuilder,
    tlas: TLASBuilder,
    tracer: PathTracer,
    temporal: TemporalDenoise,
    spatial: SpatialDenoise,
    compositor: Compositor,
    gpu_data: Option<GPUBVHData>,
    #[allow(dead_code)]
    blas_built: bool,
    prev_vp: [f32; 16],
    frame_index: u32,
    /// Samples per pixel per frame (default 1).
    pub spp: u32,
    /// Maximum ray bounce depth (default 4).
    pub max_bounces: u32,
    /// Number of A-trous spatial filter iterations (default 3).
    pub spatial_passes: u32,
    /// Temporal blend factor — lower values accumulate more history (default 0.1).
    pub temporal_blend: f32,
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

        Self {
            bvh,
            tlas,
            tracer,
            temporal: TemporalDenoise::new(renderer),
            spatial: SpatialDenoise::new(renderer),
            compositor: Compositor::new(renderer),
            gpu_data: Some(gpu_data),
            blas_built: true,
            prev_vp: [0.0; 16],
            frame_index: 0,
            spp: 1,
            max_bounces: 4,
            spatial_passes: 3,
            temporal_blend: 0.1,
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

    /// The full pipeline into `output` (an rgba16float storage texture) with the GBuffer's normals
    /// and depth, tracing `light_count` of the lights uploaded last (what `PathTracer::set_lights`
    /// returned). As a [`PostProcessingEffect`] it does the same with the count it keeps.
    pub fn render_with_gbuffer(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        gbuffer: &GBuffer,
        output: &wgpu::TextureView,
        camera: &Camera,
        light_count: u32,
    ) {
        self.light_count = light_count;
        self.run(encoder, gbuffer, &gbuffer.depth_view, output, camera, gbuffer.width, gbuffer.height);
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        gbuffer: &GBuffer,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        camera: &Camera,
        width: u32,
        height: u32,
    ) {
        // Ensure internal textures match viewport size
        self.tracer.resize(width, height);
        self.temporal.resize(width, height);
        self.spatial.resize(width, height);

        // Apply configuration
        self.tracer.set_spp(self.spp);
        self.tracer.set_max_bounces(self.max_bounces);
        self.temporal.blend = self.temporal_blend;

        // The trace averages every frame since its last reset: a moved camera starts it again,
        // and the temporal filter reprojects what was seen before.
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

        // 2. Temporal denoise
        if let Some(gi_view) = self.tracer.output_view() {
            self.temporal.denoise(
                encoder,
                gi_view,
                depth,
                &gbuffer.normal_view,
                &self.prev_vp,
                &curr_vp,
                self.frame_index,
            );
        }

        // 3. Spatial denoise
        if let (Some(temporal_out), Some(moments)) =
            (self.temporal.output_view(), self.temporal.moments_view())
        {
            self.spatial.iterations = self.spatial_passes;
            let denoised = self.spatial.denoise(
                encoder,
                temporal_out,
                depth,
                &gbuffer.normal_view,
                moments,
            );

            // 4. Output: the traced radiance is complete, so nothing from the raster is added
            self.compositor.composite(
                encoder,
                denoised,
                &gbuffer.albedo_view,
                &gbuffer.color_view,
                &gbuffer.emissive_view,
                output,
                width,
                height,
                false,
            );
        }

        // Store VP for next frame's reprojection
        self.prev_vp = curr_vp;
        self.frame_index += 1;
    }

    /// Reset temporal accumulation (call after camera movement or scene change).
    pub fn reset_accumulation(&mut self) {
        self.frame_index = 0;
        self.tracer.reset_accumulation();
    }
}

impl PostProcessingEffect for PathTracerEffect {
    fn initialize(&mut self, _device: &wgpu::Device, gbuffer: &GBuffer, _camera: &Camera) {
        if self.initialized {
            return;
        }
        self.tracer.resize(gbuffer.width, gbuffer.height);
        self.temporal.resize(gbuffer.width, gbuffer.height);
        self.spatial.resize(gbuffer.width, gbuffer.height);
        self.initialized = true;
    }

    fn render(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        gbuffer: &GBuffer,
        _input: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        camera: &Camera,
        width: u32,
        height: u32,
    ) {
        self.run(encoder, gbuffer, depth, output, camera, width, height);
    }

    fn resize(&mut self, width: u32, height: u32, _gbuffer: &GBuffer) {
        self.tracer.resize(width, height);
        self.temporal.resize(width, height);
        self.spatial.resize(width, height);
    }

    fn destroy(&mut self) {
        // Resources are dropped automatically when the struct is dropped.
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
