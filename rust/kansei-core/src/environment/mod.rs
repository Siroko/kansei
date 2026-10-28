//! Environment lighting: the light that reaches a scene from past its geometry, from one of three
//! interchangeable sources, each handing its consumers the same [`EnvironmentLighting`]:
//! - [`GradientEnvironment`]: radiance running from below to above (`gi::gradient_sky_lighting`),
//!   SH only;
//! - the atmosphere (`atmosphere::SkyAtmosphere::lighting`): its SH and its sky cubemap;
//! - [`HdrEnvironment`]: an equirectangular HDR image (a studio HDRI), its SH and a cubemap sharp
//!   at mip 0 and GGX-prefiltered below, prepared on the GPU when it loads or its settings change.
//!
//! A renderer picks one and gives it to every consumer:
//!
//! ```ignore
//! let environment = HdrEnvironment::from_bytes(device, queue, &bytes, HdrEnvironmentOptions::default())?;
//! let lighting = environment.lighting();
//! renderer.set_environment(Some(&lighting));  // the voxel GI's sky (volume and clipmap)
//! volume.set_environment(&lighting);          // every effect: the RT reflections' and glass's misses,
//!                                             // the RT and voxel GI's, the fog's, the background
//! // a material lit by it where nothing traces it
//! let material = Material::standard_lit_environment("Metal", &options, &lighting);
//! // and each frame, before rendering (a no-op unless the settings changed):
//! environment.prepare(device, queue);
//! ```
//!
//! The background is [`crate::postprocessing::effects::EnvironmentBackgroundEffect`] (the
//! cubemap where the scene left the depth at the far plane); the gradient's is
//! `Material::gradient_sky` on a sphere, the atmosphere's `AtmosphereEffect`.

mod hdr;

pub use hdr::{HdrEnvironment, HdrEnvironmentOptions, HdrEnvironmentSettings, HdrImage};

use crate::gi::{gradient_sky_lighting, SkyLightingData};

/// The light past a scene's geometry, as its consumers bind it.
#[derive(Clone, Debug)]
pub struct EnvironmentLighting {
    /// The `SkyLighting` uniform (`atmosphere::SKY_LIGHTING_WGSL`): the radiance as order-2 SH,
    /// for diffuse light (`skyIrradiance`) and rays and cones that leave the scene (`skyRadiance`).
    pub sky_lighting: wgpu::Buffer,
    /// A cubemap of the radiance, GGX-prefiltered down its mips (`atmosphere::SKY_ENVIRONMENT_WGSL`):
    /// sharp and glossy reflections' misses and the materials' specular ambient. None: consumers
    /// reflect the SH instead.
    pub environment: Option<EnvironmentMap>,
}

/// A prefiltered environment cubemap: mip m holds the radiance under a GGX lobe of roughness
/// m / (mips - 1), mip 0 the mirror.
#[derive(Clone, Debug)]
pub struct EnvironmentMap {
    /// The cube view (`texture_cube<f32>`).
    pub view: wgpu::TextureView,
    /// Trilinear, clamping.
    pub sampler: wgpu::Sampler,
    pub mips: u32,
}

/// The gradient source: radiance from `down` (straight below) to `up` (straight above), linear in
/// the direction's height, as `gi::gradient_sky_lighting` lays it out, in a uniform of its own.
pub struct GradientEnvironment {
    sky_lighting: wgpu::Buffer,
}

impl GradientEnvironment {
    pub fn new(device: &wgpu::Device, up: [f32; 3], down: [f32; 3]) -> Self {
        use wgpu::util::DeviceExt;
        let sky_lighting = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GradientEnvironment/SkyLighting"),
            contents: bytemuck::cast_slice(&gradient_sky_lighting(up, down)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        Self { sky_lighting }
    }

    pub fn set(&self, queue: &wgpu::Queue, up: [f32; 3], down: [f32; 3]) {
        queue.write_buffer(&self.sky_lighting, 0, bytemuck::cast_slice(&gradient_sky_lighting(up, down)));
    }

    pub fn lighting(&self) -> EnvironmentLighting {
        EnvironmentLighting { sky_lighting: self.sky_lighting.clone(), environment: None }
    }
}

/// The `SkyLighting` of an environment of constant radiance `c` (for tests and comparisons):
/// band 0 alone, `c / 0.282095`.
pub fn constant_sky_lighting(c: [f32; 3]) -> SkyLightingData {
    gradient_sky_lighting(c, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prepare_shader_validates() {
        let code = hdr::prepare_source();
        let module = naga::front::wgsl::parse_str(&code).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&code)));
        // the strictest capabilities, closest to core WebGPU
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let size = |name: &str| {
            module.types.iter().find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == name => Some(*span as usize),
                _ => None,
            })
        };
        assert_eq!(size("HdrPrepare"), Some(std::mem::size_of::<hdr::HdrPrepareGpu>()));
        assert_eq!(size("HdrMipPass"), Some(std::mem::size_of::<hdr::HdrMipPassGpu>()));
        assert_eq!(size("SkyLighting"), Some(std::mem::size_of::<SkyLightingData>()));
        for entry in ["to_cube", "downsample", "prefilter", "project_sh"] {
            assert!(module.entry_points.iter().any(|e| e.name == entry), "{entry}");
        }
    }

    // A Radiance file of 2 x 1 pixels written by hand: the header, then flat (not run-length)
    // RGBE scanlines. RGBE (r, g, b, e) is (r, g, b) * 2^(e - 136).
    fn tiny_hdr() -> Vec<u8> {
        let mut bytes = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n".to_vec();
        // white: 128 * 2^(129 - 136) = 1; then (0.5, 0.25, 2) at 2^(130 - 136) = 1 / 64
        bytes.extend_from_slice(&[128, 128, 128, 129]);
        bytes.extend_from_slice(&[32, 16, 128, 130]);
        bytes
    }

    #[test]
    fn a_radiance_file_decodes_to_its_radiance() {
        let image = HdrImage::decode(&tiny_hdr()).unwrap();
        assert_eq!((image.width, image.height), (2, 1));
        assert_eq!(image.pixels[0], [1.0, 1.0, 1.0]);
        assert_eq!(image.pixels[1], [0.5, 0.25, 2.0]);
        assert!(HdrImage::decode(b"not an image").is_err());
    }

    #[test]
    fn a_constant_environment_is_band_zero() {
        let sky = constant_sky_lighting([2.0, 1.0, 0.5]);
        assert!((sky[0][0] * 0.282095 - 2.0).abs() < 1e-5);
        assert!(sky[1..9].iter().all(|b| b[..3].iter().all(|v| *v == 0.0)));
    }
}
