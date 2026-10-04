mod binding;
mod compute;
mod material;
mod shader_utils;
mod standard;
mod stock;

pub use binding::{Binding, BindingResource, BindingType, BindGroupBuilder};
pub use compute::ComputePass;
pub type Compute = ComputePass;
pub use material::{Material, MaterialOptions, CullMode};
pub(crate) use material::{DepthPipelineKey, PipelineKey};
pub use shader_utils::{parse_includes, ShaderChunks};
pub use standard::{GradientSkyOptions, StandardInstancing, StandardLitOptions, GBUFFER_OUT_WGSL};
pub use stock::{BASIC_INSTANCED_WGSL, BASIC_LIT_WGSL, PARTICLE_BILLBOARD_WGSL};

// Re-export ShaderStages so user code doesn't need to import wgpu directly
pub use wgpu::ShaderStages;
