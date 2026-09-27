use crate::atmosphere::sky_atmosphere::{
    compute_pipeline, sampler_entry, texture_3d_entry, texture_entry, uniform_entry, AERIAL_PERSPECTIVE_LOOKUP_WGSL, COMMON_WGSL,
    FRAME_WGSL, LOOKUP_TRANSMITTANCE_WGSL, SKY_LOOKUP_WGSL,
};
use crate::atmosphere::{SkyAtmosphere, SkyAtmosphereBindings};
use crate::cameras::Camera;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

const COMPOSITE_WGSL: &str = include_str!("../../atmosphere/shaders/sky_composite.wgsl");

fn composite_source() -> String {
    [COMMON_WGSL, FRAME_WGSL, LOOKUP_TRANSMITTANCE_WGSL, SKY_LOOKUP_WGSL, AERIAL_PERSPECTIVE_LOOKUP_WGSL, COMPOSITE_WGSL].concat()
}

/// Renders a [`SkyAtmosphere`]: the sky, the sun and the moon wherever the scene left the depth
/// buffer at the far plane, and aerial perspective (the atmosphere between the camera and each
/// surface) everywhere else. Put it first in the chain, before the fog and the tonemapper, and
/// call `SkyAtmosphere::update` every frame before rendering.
pub struct AtmosphereEffect {
    sky: SkyAtmosphereBindings,
    gpu: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
}

impl AtmosphereEffect {
    pub fn new(sky: &SkyAtmosphere) -> Self {
        Self { sky: sky.bindings().clone(), gpu: None }
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        let compute = wgpu::ShaderStages::COMPUTE;
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Atmosphere/CompositeBGL"),
            entries: &[
                uniform_entry(0),
                uniform_entry(1),
                texture_entry(2),
                texture_entry(3),
                sampler_entry(4),
                sampler_entry(5),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: compute,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: GBuffer::COLOR_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                texture_3d_entry(9),
                texture_3d_entry(10),
            ],
        });
        let pipeline = compute_pipeline(device, "Atmosphere/Composite", &composite_source(), &bgl);
        self.gpu = Some((pipeline, bgl));
    }

    #[cfg(test)]
    pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
        use crate::atmosphere::sky_atmosphere as s;
        vec![
            ("transmittance_lut", s::transmittance_source()),
            ("multi_scattering_lut", s::multi_scattering_source()),
            ("sky_view_lut", s::sky_view_source()),
            ("aerial_perspective_lut", s::aerial_perspective_source()),
            ("sky_lighting", s::sky_lighting_source()),
            ("sky_composite", composite_source()),
        ]
    }
}

impl PostProcessingEffect for AtmosphereEffect {
    fn initialize(&mut self, device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {
        if self.gpu.is_none() {
            self.init_gpu(device);
        }
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        _gbuffer: &GBuffer,
        input: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        _camera: &Camera,
        width: u32,
        height: u32,
    ) {
        if self.gpu.is_none() {
            self.init_gpu(device);
        }
        let (pipeline, bgl) = self.gpu.as_ref().unwrap();
        let s = &self.sky;
        let tex = wgpu::BindingResource::TextureView;
        let resources = [
            s.atmosphere.as_entire_binding(),
            s.frame.as_entire_binding(),
            tex(&s.transmittance),
            tex(&s.sky_view),
            wgpu::BindingResource::Sampler(&s.lut_sampler),
            wgpu::BindingResource::Sampler(&s.sky_view_sampler),
            tex(input),
            tex(depth),
            tex(output),
            tex(&s.ap_scattering),
            tex(&s.ap_transmittance),
        ];
        let entries: Vec<_> = resources
            .into_iter()
            .enumerate()
            .map(|(i, resource)| wgpu::BindGroupEntry { binding: i as u32, resource })
            .collect();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("Atmosphere/CompositeBG"), layout: bgl, entries: &entries });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Atmosphere/Composite"), ..Default::default() });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }

    fn resize(&mut self, _width: u32, _height: u32, _gbuffer: &GBuffer) {}

    fn destroy(&mut self) {
        self.gpu = None;
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::params::{AtmosphereGpu, SkyFrameGpu, SkyLightingGpu};

    /// Every atmosphere WGSL module parses and validates with naga, and the Rust uniform structs
    /// have exactly the size of their WGSL counterparts.
    #[test]
    fn shaders_validate_and_uniform_layouts_match() {
        let mut sizes = std::collections::HashMap::new();
        for (name, code) in AtmosphereEffect::shader_sources() {
            let module = naga::front::wgsl::parse_str(&code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&code)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            for (_, ty) in module.types.iter() {
                if let (Some(n), naga::TypeInner::Struct { span, .. }) = (&ty.name, &ty.inner) {
                    sizes.insert(n.clone(), *span as usize);
                }
            }
        }
        assert_eq!(sizes["Atmosphere"], std::mem::size_of::<AtmosphereGpu>());
        assert_eq!(sizes["SkyFrame"], std::mem::size_of::<SkyFrameGpu>());
        assert_eq!(sizes["SkyLighting"], std::mem::size_of::<SkyLightingGpu>());
    }
}
