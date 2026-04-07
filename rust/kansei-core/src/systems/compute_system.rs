/// Trait for GPU compute systems driven by the Renderer.
/// Implementors own their GPU resources and are updated each frame.
pub trait ComputeSystem {
    /// Initialize GPU resources. Called once by the Renderer.
    fn initialize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue);

    /// Run the compute work for this frame.
    fn update(&mut self, dt: f32);

    /// Whether the system has been initialized.
    fn is_initialized(&self) -> bool;
}
