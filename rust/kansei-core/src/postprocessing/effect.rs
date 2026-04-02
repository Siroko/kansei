use crate::cameras::Camera;
use crate::renderers::GBuffer;

/// Trait for compute-shader post-processing effects.
pub trait PostProcessingEffect {
    fn initialize(&mut self, device: &wgpu::Device, gbuffer: &GBuffer, camera: &Camera);
    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        camera: &Camera,
        width: u32,
        height: u32,
    );
    fn resize(&mut self, width: u32, height: u32, gbuffer: &GBuffer);
    fn destroy(&mut self);
}
