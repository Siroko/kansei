use crate::materials::BindingResource;

/// Trait for GPU resources that can be attached to a Material or ComputePass
/// binding slot. The renderer calls `ensure_ready()` once (lazily), then
/// `binding_resource()` to build the bind group.
///
/// Implemented by `ComputeBuffer`, `Texture`, and `Sampler`.
pub trait Bindable {
    /// Create / upload the GPU resource if not yet initialized.
    fn ensure_ready(&mut self, device: &wgpu::Device, queue: &wgpu::Queue);
    /// Return the binding resource for bind group construction.
    /// Returns `None` if the resource hasn't been initialized yet.
    fn binding_resource(&self) -> Option<BindingResource>;
}
