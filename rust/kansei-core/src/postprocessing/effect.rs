use crate::cameras::Camera;
use crate::renderers::GBuffer;

/// Trait for compute-shader post-processing effects.
///
/// `render`'s `width` and `height` are the size of `output`: the GBuffer's size (the renderer's
/// render size) before the first effect that `upscales_to_display`, and the display (surface)
/// size from that effect on (whose `input` is still at the GBuffer's size). `gbuffer` and
/// `depth` stay at the GBuffer's size: an effect that runs after the upscaler and reads them by
/// pixel must scale its coordinates by `textureDimensions(depth) / (width, height)`. `resize` is
/// called with the GBuffer's size when it changes.
pub trait PostProcessingEffect {
    fn initialize(&mut self, device: &wgpu::Device, gbuffer: &GBuffer, camera: &Camera);
    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        gbuffer: &GBuffer,
        input: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        camera: &Camera,
        width: u32,
        height: u32,
    );
    fn resize(&mut self, width: u32, height: u32, gbuffer: &GBuffer);
    /// Whether the effect runs this frame. An inactive effect is skipped: the next effect reads
    /// what it would have read, and it costs nothing (a toggle for expensive effects).
    fn is_active(&self) -> bool {
        true
    }
    /// Whether the effect wants the scene rendered with a sub-pixel jittered projection (TAA).
    fn wants_jitter(&self) -> bool {
        false
    }
    /// Whether the effect reads its input at the GBuffer's size and writes its output at the
    /// display size (a temporal upscaler). Every effect after it runs at the display size. The
    /// two sizes are the same unless the renderer has a render scale below 1.
    fn upscales_to_display(&self) -> bool {
        false
    }
    fn destroy(&mut self);
    /// Downcast support for runtime access to concrete effect types.
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}
