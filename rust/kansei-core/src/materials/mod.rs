mod binding;
mod compute;
mod material;
mod shader_utils;

pub use binding::{Binding, BindingResource, BindGroupBuilder};
pub use compute::ComputePass;
pub type Compute = ComputePass;
pub use material::{Material, MaterialOptions, CullMode};
pub(crate) use material::PipelineKey;
pub use shader_utils::{parse_includes, ShaderChunks};
