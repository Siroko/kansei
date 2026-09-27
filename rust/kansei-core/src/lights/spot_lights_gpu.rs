use bytemuck::{Pod, Zeroable};

use super::{Light, SpotLight};

/// Most spot lights the renderer uploads per frame; later ones are ignored.
pub const MAX_SPOT_LIGHTS: usize = 128;

/// Near plane of the spot shadow projections, metres.
pub(crate) const SPOT_SHADOW_NEAR: f32 = 0.05;

/// One spot light as `KanseiSpotLight` in `spot_light_types.wgsl` (144 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub(crate) struct SpotLightGpu {
    pub position: [f32; 3],
    pub range: f32,
    pub direction: [f32; 3],
    pub cos_outer: f32,
    pub color: [f32; 3],
    pub cos_inner: f32,
    pub shadow_layer: i32,
    pub volumetric_scale: f32,
    pub source_radius: f32,
    pub normal_bias: f32,
    pub shadow_near: f32,
    pub tan_half_fov: f32,
    pub texel_size: f32,
    pub _pad: f32,
    pub view_proj: [f32; 16],
}

/// Header of `KanseiSpotLights`: the light count, padded to the array's 16-byte alignment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SpotLightsHeaderGpu {
    pub count: u32,
    pub _pad: [u32; 3],
}

pub(crate) const SPOT_LIGHTS_BUFFER_BYTES: usize =
    std::mem::size_of::<SpotLightsHeaderGpu>() + MAX_SPOT_LIGHTS * std::mem::size_of::<SpotLightGpu>();

/// A shadow-casting spot light's slot in the shadow atlas this frame.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SpotShadowSlot {
    pub layer: u32,
    pub view: glam::Mat4,
    pub projection: glam::Mat4,
}

/// CPU staging for the renderer's spot-light storage buffer.
pub(crate) struct SpotLightsGpu {
    pub lights: Vec<SpotLightGpu>,
    /// Shadow slots assigned this frame, in layer order.
    pub shadows: Vec<SpotShadowSlot>,
    bytes: Vec<u8>,
}

impl SpotLightsGpu {
    pub fn new() -> Self {
        Self { lights: Vec::new(), shadows: Vec::new(), bytes: Vec::new() }
    }

    /// Pack the scene's spot lights. The first `shadow_layers` lights with `cast_shadow` get the
    /// atlas layers in scene order; `shadow_resolution` is the atlas size (0 without an atlas).
    pub fn pack<'a>(&mut self, lights: impl IntoIterator<Item = &'a Light>, shadow_layers: u32, shadow_resolution: u32) {
        self.lights.clear();
        self.shadows.clear();
        for light in lights {
            let Light::Spot(spot) = light else { continue };
            if self.lights.len() == MAX_SPOT_LIGHTS {
                break;
            }
            let layer = if spot.cast_shadow && (self.shadows.len() as u32) < shadow_layers {
                let layer = self.shadows.len() as u32;
                let (view, projection) = spot_shadow_matrices(spot);
                self.shadows.push(SpotShadowSlot { layer, view, projection });
                layer as i32
            } else {
                -1
            };
            self.lights.push(pack_spot(spot, layer, shadow_resolution));
        }
    }

    /// The storage-buffer bytes: header then lights.
    pub fn as_bytes(&mut self) -> &[u8] {
        let header = SpotLightsHeaderGpu { count: self.lights.len() as u32, _pad: [0; 3] };
        self.bytes.clear();
        self.bytes.extend_from_slice(bytemuck::bytes_of(&header));
        self.bytes.extend_from_slice(bytemuck::cast_slice(&self.lights));
        &self.bytes
    }
}

fn spot_shadow_matrices(spot: &SpotLight) -> (glam::Mat4, glam::Mat4) {
    (spot.shadow_view(), spot.shadow_projection(SPOT_SHADOW_NEAR))
}

fn pack_spot(spot: &SpotLight, shadow_layer: i32, shadow_resolution: u32) -> SpotLightGpu {
    let (outer, inner) = spot.cone();
    let dir = glam::Vec3::new(spot.direction.x, spot.direction.y, spot.direction.z).normalize_or(glam::Vec3::NEG_Z);
    let c = spot.effective_color();
    let (view, projection) = spot_shadow_matrices(spot);
    SpotLightGpu {
        position: [spot.position.x, spot.position.y, spot.position.z],
        range: spot.range.max(1e-3),
        direction: dir.to_array(),
        cos_outer: outer.cos(),
        color: [c.x, c.y, c.z],
        cos_inner: inner.cos(),
        shadow_layer,
        volumetric_scale: spot.volumetric_scale.max(0.0),
        source_radius: spot.source_radius.max(0.0),
        normal_bias: spot.shadow_normal_bias.max(0.0),
        shadow_near: SPOT_SHADOW_NEAR,
        // the projection's vertical scale is 1 / tan(fov / 2)
        tan_half_fov: 1.0 / projection.y_axis.y.abs().max(1e-6),
        texel_size: if shadow_resolution > 0 { 1.0 / shadow_resolution as f32 } else { 0.0 },
        _pad: 0.0,
        view_proj: (projection * view).to_cols_array(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;

    fn spot(cast_shadow: bool) -> Light {
        let mut s = SpotLight::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, -2.0), Vec3::new(1.0, 0.5, 0.25), 22000.0, 70.0, 10f32.to_radians(), 30f32.to_radians());
        s.cast_shadow = cast_shadow;
        Light::Spot(s)
    }

    /// The material chunk validates with naga (also as part of a module with an entry point),
    /// and the WGSL structs match the Rust layouts.
    #[test]
    fn wgsl_validates_and_matches_the_gpu_layout() {
        let shader = format!(
            "{}\n@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {{\n    \
             let c = kansei_spot_lights_radiance(p.xyz, vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(0.5), 0.5, 0.0, p.xy);\n    \
             return vec4f(c, 1.0);\n}}\n",
            super::super::SPOT_LIGHTS_WGSL
        );
        let module = naga::front::wgsl::parse_str(&shader).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&shader)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let span = |name: &str| {
            module
                .types
                .iter()
                .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                    (Some(n), naga::TypeInner::Struct { span, .. }) if n == name => Some(*span as usize),
                    _ => None,
                })
                .unwrap()
        };
        assert_eq!(span("KanseiSpotLight"), std::mem::size_of::<SpotLightGpu>());
        // runtime-sized: the header, then one element's stride
        assert_eq!(span("KanseiSpotLights"), std::mem::size_of::<SpotLightsHeaderGpu>() + std::mem::size_of::<SpotLightGpu>());
    }

    #[test]
    fn pack_assigns_shadow_layers_in_scene_order_and_skips_other_lights() {
        use crate::lights::PointLight;
        let lights = [
            spot(true),
            Light::Point(PointLight::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0), 1.0, 5.0)),
            spot(false),
            spot(true),
            spot(true),
        ];
        let mut gpu = SpotLightsGpu::new();
        gpu.pack(lights.iter(), 2, 1024);
        assert_eq!(gpu.lights.iter().map(|l| l.shadow_layer).collect::<Vec<_>>(), [0, -1, 1, -1]);
        assert_eq!(gpu.shadows.iter().map(|s| s.layer).collect::<Vec<_>>(), [0, 1]);

        let l = &gpu.lights[0];
        assert_eq!(l.color, [22000.0, 11000.0, 5500.0]);
        assert!((l.cos_inner - 10f32.to_radians().cos()).abs() < 1e-6);
        assert!((l.cos_outer - 30f32.to_radians().cos()).abs() < 1e-6);
        assert_eq!(l.direction, [0.0, 0.0, -1.0]);
        assert_eq!(l.texel_size, 1.0 / 1024.0);
        // the shadow projection covers the outer cone with a margin
        assert!(l.tan_half_fov > 30f32.to_radians().tan() && l.tan_half_fov < 35f32.to_radians().tan());

        // without an atlas no light is shadowed; header + lights in the buffer bytes
        gpu.pack(lights.iter(), 0, 0);
        assert!(gpu.lights.iter().all(|l| l.shadow_layer == -1) && gpu.shadows.is_empty());
        let bytes = gpu.as_bytes().to_vec();
        assert_eq!(bytes.len(), 16 + 4 * std::mem::size_of::<SpotLightGpu>());
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 4);
        assert!(bytes.len() <= SPOT_LIGHTS_BUFFER_BYTES);
    }
}
