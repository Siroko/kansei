//! The standard lit material: a physically based surface lit by everything the scene has, for
//! scenes drawn through a `PostProcessingVolume` (it writes the GBuffer's targets).

use super::{Binding, Material, MaterialOptions};

/// Writes the GBuffer's four colour targets; see [`Material::standard_lit`].
pub const GBUFFER_OUT_WGSL: &str = include_str!("../shaders/gbuffer_out.wgsl");

const STANDARD_LIT_WGSL: &str = include_str!("../shaders/standard_lit.wgsl");

/// How an instanced [`Material::standard_lit`] places each instance: a vec4 per instance at
/// vertex location 3 (`ComputeBuffer::with_vertex_vec4(3)`), the layout `culling::InstanceCulling`
/// reads (16-byte stride, position in xyz).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StandardInstancing {
    /// xyz: the offset, w: a uniform scale.
    OffsetScale,
    /// xyz: the offset, w: a scale of the height (y) only, for trunks and posts.
    OffsetHeight,
}

/// What [`Material::standard_lit`] draws. Radiances are in cd/m² and the lights' colours in the
/// units the scene's lights use, so the result goes through `ToneMapEffect` like the rest of an
/// HDR scene.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StandardLitOptions {
    /// Diffuse albedo (linear rgb), or the specular colour of a metal.
    pub base_color: [f32; 3],
    /// GGX roughness, 0 (mirror) to 1.
    pub roughness: f32,
    /// 0 (dielectric) to 1 (metal).
    pub metallic: f32,
    /// Radiance the surface emits (cd/m²), also written to the GBuffer's emissive target.
    pub emissive: [f32; 3],
    /// Sky radiance from straight up (cd/m²): a hemisphere of ambient light, reflected by the
    /// albedo. Zero for none.
    pub sky_up: [f32; 3],
    /// Radiance from straight down (the ground's bounce), blended with `sky_up` by the normal.
    pub sky_down: [f32; 3],
    /// Instanced placement (`None`: one mesh at its transform).
    pub instancing: Option<StandardInstancing>,
    /// Also write screen-space motion (GBuffer target 4) for `TemporalAAEffect` and motion blur:
    /// sets `MaterialOptions::outputs_velocity`. Instances must not move between frames.
    pub outputs_velocity: bool,
}

impl Default for StandardLitOptions {
    fn default() -> Self {
        Self {
            base_color: [0.5, 0.5, 0.5],
            roughness: 0.6,
            metallic: 0.0,
            emissive: [0.0; 3],
            sky_up: [0.0; 3],
            sky_down: [0.0; 3],
            instancing: None,
            outputs_velocity: false,
        }
    }
}

impl StandardLitOptions {
    /// An unlit surface emitting `radiance` (cd/m²): lamp heads, windows, a bright backdrop.
    pub fn emissive(radiance: [f32; 3]) -> Self {
        Self { base_color: [0.0; 3], emissive: radiance, ..Default::default() }
    }

    /// The material's group 0 uniform, as `KanseiStandardSurface` lays it out.
    fn uniform(&self) -> [f32; 20] {
        let [r, g, b] = self.base_color;
        let [er, eg, eb] = self.emissive;
        let [ur, ug, ub] = self.sky_up;
        let [dr, dg, db] = self.sky_down;
        [
            r, g, b, 1.0,
            er, eg, eb, 0.0,
            ur, ug, ub, 0.0,
            dr, dg, db, 0.0,
            self.roughness, self.metallic, 0.0, 0.0,
        ]
    }

    /// The shader: the chunks it uses, then the material with its placeholders filled.
    fn shader(&self) -> String {
        let velocity = self.outputs_velocity;
        let (instance_input, instance_place) = match self.instancing {
            None => ("", ""),
            Some(StandardInstancing::OffsetScale) => ("@location(3) instance : vec4f,", "local = local * v.instance.w + v.instance.xyz;"),
            Some(StandardInstancing::OffsetHeight) => ("@location(3) instance : vec4f,", "local = vec3f(local.x, local.y * v.instance.w, local.z) + v.instance.xyz;"),
        };
        let body = STANDARD_LIT_WGSL
            .replace("KANSEI_INSTANCE_INPUT", instance_input)
            .replace("KANSEI_INSTANCE_PLACE", instance_place)
            .replace(
                "KANSEI_WORLD_BINDING",
                if velocity { "@group(2) @binding(1) var<uniform> mesh : KanseiMeshTransforms;" } else { "@group(2) @binding(1) var<uniform> world_matrix : mat4x4f;" },
            )
            .replace("KANSEI_WORLD", if velocity { "mesh.world" } else { "world_matrix" })
            .replace("KANSEI_VELOCITY_VARYINGS", if velocity { "@location(2) currClip : vec4f,\n    @location(3) prevClip : vec4f," } else { "" })
            .replace("KANSEI_VELOCITY_OUTPUT", if velocity { "@location(4) velocity : vec2f," } else { "" })
            .replace(
                "KANSEI_VELOCITY_VERTEX",
                if velocity {
                    "out.currClip = kansei_camera_temporal.viewProj * world;\n    out.prevClip = kansei_camera_temporal.prevViewProj * (mesh.prevWorld * vec4f(local, 1.0));"
                } else {
                    ""
                },
            )
            .replace("KANSEI_VELOCITY_FRAGMENT", if velocity { "out.velocity = kansei_motion_vector(in.currClip, in.prevClip);" } else { "" });
        let motion = if velocity { crate::cameras::MOTION_VECTORS_WGSL } else { "" };
        format!(
            "{}\n{}\n{}\n{}\n{motion}\n{body}",
            crate::lights::LIGHTS_WGSL,
            crate::shadows::SHADOW_MAP_WGSL,
            crate::shadows::CASCADED_SHADOWS_WGSL,
            crate::lights::SPOT_LIGHTS_WGSL,
        )
    }
}

impl Material {
    /// The standard lit material: a GGX / Lambert surface lit by the scene's directional lights
    /// (the sun through the cascaded shadow map when `Renderer::enable_cascaded_shadows` is on,
    /// else through the single map of `enable_shadows`, which follows the first directional
    /// light), its point lights (the one `enable_point_shadows` renders through its cube shadow), its spot lights with their shadows, a hemisphere of sky and its
    /// own emission. It writes the GBuffer's four targets, so draw it through
    /// `Renderer::render_with_postprocessing`.
    pub fn standard_lit(label: &str, options: &StandardLitOptions) -> Material {
        let material_options = MaterialOptions { mrt_output_count: Some(4), outputs_velocity: options.outputs_velocity, ..Default::default() };
        let mut material = Material::new(label, &options.shader(), vec![Binding::uniform(0, wgpu::ShaderStages::FRAGMENT)], material_options);
        material.set_uniform_bindable(0, &format!("{label}/Surface"), &options.uniform());
        material
    }

    /// An unlit material emitting `radiance` (cd/m²) into the GBuffer: lamp heads, windows, a
    /// backdrop. Shorthand for `standard_lit` with [`StandardLitOptions::emissive`].
    pub fn emissive(label: &str, radiance: [f32; 3]) -> Material {
        Material::standard_lit(label, &StandardLitOptions::emissive(radiance))
    }
}

const GRADIENT_SKY_WGSL: &str = include_str!("../shaders/gradient_sky.wgsl");

/// A sky of three radiances for [`Material::gradient_sky`] (cd/m², linear rgb).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientSkyOptions {
    pub zenith: [f32; 3],
    pub horizon: [f32; 3],
    /// Below the horizon.
    pub ground: [f32; 3],
    /// How fast the horizon gives way to the zenith: the height's exponent (0.5 by default:
    /// a wide horizon band).
    pub curve: f32,
}

impl Default for GradientSkyOptions {
    fn default() -> Self {
        Self { zenith: [3000.0, 5000.0, 9000.0], horizon: [9000.0, 9500.0, 10500.0], ground: [1500.0, 1500.0, 1400.0], curve: 0.5 }
    }
}

impl Material {
    /// A cheap sky: put it on a large sphere around the scene (`SphereGeometry`, with
    /// `cast_shadow = false`) and it shows `options`' gradient by direction from the sphere's
    /// centre, into the GBuffer. For a physical sky with aerial perspective and sky lighting, use
    /// `atmosphere::SkyAtmosphere`.
    pub fn gradient_sky(label: &str, options: &GradientSkyOptions) -> Material {
        let shader = format!("{GBUFFER_OUT_WGSL}\n{GRADIENT_SKY_WGSL}");
        let material_options = MaterialOptions { mrt_output_count: Some(4), cull_mode: super::CullMode::None, ..Default::default() };
        let mut material = Material::new(label, &shader, vec![Binding::uniform(0, wgpu::ShaderStages::FRAGMENT)], material_options);
        let ([zr, zg, zb], [hr, hg, hb], [gr, gg, gb]) = (options.zenith, options.horizon, options.ground);
        let uniform: [f32; 12] = [zr, zg, zb, options.curve, hr, hg, hb, 0.0, gr, gg, gb, 0.0];
        material.set_uniform_bindable(0, &format!("{label}/Sky"), &uniform);
        material
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(label: &str, code: &str) -> naga::Module {
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        module
    }

    fn struct_size(module: &naga::Module, name: &str) -> usize {
        let size = module.types.iter().find_map(|(_, ty)| match (&ty.name, &ty.inner) {
            (Some(n), naga::TypeInner::Struct { span, .. }) if n == name => Some(*span as usize),
            _ => None,
        });
        size.unwrap_or_else(|| panic!("no struct {name}"))
    }

    #[test]
    fn every_standard_variant_validates_and_its_uniform_matches() {
        for instancing in [None, Some(StandardInstancing::OffsetScale), Some(StandardInstancing::OffsetHeight)] {
            for outputs_velocity in [false, true] {
                let options = StandardLitOptions { instancing, outputs_velocity, ..Default::default() };
                let module = validate(&format!("{instancing:?} velocity {outputs_velocity}"), &options.shader());
                assert_eq!(struct_size(&module, "KanseiStandardSurface"), std::mem::size_of_val(&options.uniform()));
            }
        }
    }

    #[test]
    fn the_chunks_validate_and_the_light_layout_matches_the_packer() {
        let lights = validate(
            "lights",
            &format!("{}\n@fragment fn f() -> @location(0) vec4f {{ return vec4f(kansei_lights.directional[0].color * kansei_point_falloff(kansei_lights.point[0], vec3f(0.0)), 1.0); }}", crate::lights::LIGHTS_WGSL),
        );
        assert_eq!(struct_size(&lights, "KanseiLights"), crate::lights::LIGHT_UNIFORM_BYTES);
        let shadow = validate(
            "shadow map",
            &format!("{}\n@fragment fn f() -> @location(0) vec4f {{ return vec4f(kansei_shadow_map(vec3f(0.0), vec3f(0.0, 1.0, 0.0)) * kansei_point_shadow(vec3f(1.0))); }}", crate::shadows::SHADOW_MAP_WGSL),
        );
        assert_eq!(struct_size(&shadow, "KanseiShadowMap"), 24 * 4);
        let sky = validate("gradient sky", &format!("{GBUFFER_OUT_WGSL}\n{GRADIENT_SKY_WGSL}"));
        assert_eq!(struct_size(&sky, "KanseiGradientSky"), 12 * 4);
        validate("gbuffer out", &format!("{GBUFFER_OUT_WGSL}\n@fragment fn f() -> KanseiGBufferOut {{ return kansei_gbuffer_out(vec3f(1.0), vec3f(0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.5)); }}"));
    }
}
