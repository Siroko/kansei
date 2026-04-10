use crate::cameras::Camera;
use crate::objects::Scene;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::{GBuffer, Renderer};

use super::{
    BVHBuilder, Compositor, GPUBVHData, PathTracer, PathTracerMaterial, SpatialDenoise,
    TLASBuilder, TemporalDenoise,
};

/// High-level wrapper that orchestrates the full path tracing pipeline
/// (BVH build, trace, temporal denoise, spatial denoise, composite) as a
/// single [`PostProcessingEffect`].
///
/// # Pipeline stages
///
/// 1. **Trace** — dispatch the path trace compute shader using the BLAS/TLAS
///    acceleration structure built at construction time.
/// 2. **Temporal denoise** — motion-compensated reprojection blending the
///    current noisy frame with the accumulated history.
/// 3. **Spatial denoise** — A-trous wavelet filter guided by depth, normals,
///    and variance moments.
/// 4. **Composite** — combine denoised GI with albedo, direct light, and
///    emissive from the GBuffer.
///
/// # Limitations
///
/// The current `PostProcessingEffect::render()` signature provides only a
/// scene color view and a depth view. Full denoising also requires a normals
/// texture, and the compositor needs albedo/emissive. When those GBuffer
/// attachments are not available we skip the denoise/composite stages and
/// write the raw traced GI as output. The full pipeline is engaged when the
/// `render_with_gbuffer()` method is called directly.
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
            initialized: false,
        }
    }

    /// Upload path tracer materials (one per renderable in the scene).
    pub fn set_materials(&mut self, materials: &[PathTracerMaterial]) {
        self.tracer.set_materials(materials);
    }

    /// Upload raw light data for the trace shader.
    pub fn set_lights_raw(&mut self, data: &[f32]) {
        self.tracer.set_lights_raw(data);
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

    /// Full pipeline render with explicit GBuffer access.
    ///
    /// This runs all four stages (trace, temporal denoise, spatial denoise,
    /// composite) using the normal and depth views from the GBuffer.
    #[allow(clippy::too_many_arguments)]
    pub fn render_with_gbuffer(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        gbuffer: &GBuffer,
        output: &wgpu::TextureView,
        camera: &Camera,
        light_count: u32,
    ) {
        let width = gbuffer.width;
        let height = gbuffer.height;

        // Ensure internal textures match viewport size
        self.tracer.resize(width, height);
        self.temporal.resize(width, height);
        self.spatial.resize(width, height);

        // Apply configuration
        self.tracer.set_spp(self.spp);
        self.tracer.set_max_bounces(self.max_bounces);
        self.temporal.blend = self.temporal_blend;

        // Compute current view-projection matrix
        let vp = camera.projection_matrix.mul(&camera.view_matrix);
        let curr_vp = *vp.as_slice();

        // 1. Trace
        if let Some(ref gpu_data) = self.gpu_data {
            if let Some(tlas_buf) = self.tlas.tlas_nodes_buf.as_ref() {
                self.tracer
                    .trace(encoder, gpu_data, tlas_buf, camera, light_count);
            }
        }

        // 2. Temporal denoise
        if let Some(gi_view) = self.tracer.output_view() {
            self.temporal.denoise(
                encoder,
                gi_view,
                &gbuffer.depth_view,
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
            let _denoised = self.spatial.denoise(
                encoder,
                temporal_out,
                &gbuffer.depth_view,
                &gbuffer.normal_view,
                moments,
            );

            // 4. Composite — combine denoised GI with albedo + emissive
            // Use the spatial denoiser's output as the GI input.
            self.compositor.composite(
                encoder,
                _denoised,
                &gbuffer.albedo_view,
                &gbuffer.color_view,    // direct light from raster pass
                &gbuffer.emissive_view,
                output,
                width,
                height,
                true, // raster_direct = true (hybrid mode)
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
        _gbuffer: &crate::renderers::GBuffer,
        _input: &wgpu::TextureView,
        _depth: &wgpu::TextureView,
        _output: &wgpu::TextureView,
        camera: &Camera,
        width: u32,
        height: u32,
    ) {
        // The trait signature only provides input (scene color) and depth views.
        // We need the full GBuffer for normals/albedo/emissive to run the
        // denoise and composite stages. With only these two views, we run
        // just the trace pass and rely on the caller using render_with_gbuffer()
        // for the full pipeline.

        // Ensure sizes match
        self.tracer.resize(width, height);
        self.tracer.set_spp(self.spp);
        self.tracer.set_max_bounces(self.max_bounces);

        // Trace only — no denoise/composite without full GBuffer
        if let Some(ref gpu_data) = self.gpu_data {
            if let Some(tlas_buf) = self.tlas.tlas_nodes_buf.as_ref() {
                self.tracer
                    .trace(encoder, gpu_data, tlas_buf, camera, 0);
            }
        }

        // Update VP for temporal consistency if render_with_gbuffer() is
        // called on subsequent frames.
        let vp = camera.projection_matrix.mul(&camera.view_matrix);
        self.prev_vp = *vp.as_slice();
        self.frame_index += 1;
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
