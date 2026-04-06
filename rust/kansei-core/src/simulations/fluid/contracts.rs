#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceContractVersion {
    V1 = 1,
}

#[derive(Clone, Copy, Debug)]
pub struct SurfaceMeshGpuContract<'a> {
    pub version: SurfaceContractVersion,
    pub vertex_buffer: &'a wgpu::Buffer,
    pub index_buffer: &'a wgpu::Buffer,
    pub indirect_args_buffer: &'a wgpu::Buffer,
    pub vertex_stride: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct SurfaceExtractionSourceContract {
    pub version: SurfaceContractVersion,
    pub field_dims: [u32; 3],
    pub world_bounds_min: [f32; 3],
    pub world_bounds_max: [f32; 3],
    pub iso_value: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct SimulationRenderableInputContract<'a> {
    pub version: SurfaceContractVersion,
    pub mesh: SurfaceMeshGpuContract<'a>,
}

pub type SimulationRendererInputContract<'a> = SimulationRenderableInputContract<'a>;
