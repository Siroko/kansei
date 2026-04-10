/// Multi-render-target GBuffer (color, emissive, normal, albedo, depth).
pub struct GBuffer {
    pub width: u32,
    pub height: u32,
    pub sample_count: u32,
    // TODO: textures
}

impl GBuffer {
    pub fn new(_device: &wgpu::Device, width: u32, height: u32, sample_count: u32) -> Self {
        Self { width, height, sample_count }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        // TODO: recreate textures
    }
}
