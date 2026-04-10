/// A GPU sampler.
pub struct Sampler {
    gpu_sampler: Option<wgpu::Sampler>,
    mag_filter: wgpu::FilterMode,
    min_filter: wgpu::FilterMode,
    address_mode: wgpu::AddressMode,
    max_anisotropy: u16,
}

impl Sampler {
    pub fn new(mag: wgpu::FilterMode, min: wgpu::FilterMode) -> Self {
        Self {
            gpu_sampler: None,
            mag_filter: mag,
            min_filter: min,
            address_mode: wgpu::AddressMode::Repeat,
            max_anisotropy: 1,
        }
    }

    pub fn with_address_mode(mut self, mode: wgpu::AddressMode) -> Self {
        self.address_mode = mode;
        self
    }

    pub fn with_anisotropy(mut self, anisotropy: u16) -> Self {
        self.max_anisotropy = anisotropy;
        self
    }

    pub fn initialize(&mut self, device: &wgpu::Device) {
        self.gpu_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Sampler"),
            mag_filter: self.mag_filter,
            min_filter: self.min_filter,
            mipmap_filter: self.min_filter,
            address_mode_u: self.address_mode,
            address_mode_v: self.address_mode,
            address_mode_w: self.address_mode,
            anisotropy_clamp: self.max_anisotropy,
            ..Default::default()
        }));
    }

    pub fn gpu_sampler(&self) -> Option<&wgpu::Sampler> {
        self.gpu_sampler.as_ref()
    }
}
