//! The standard lit material: a physically based surface lit by everything the scene has, for
//! scenes drawn through a `PostProcessingVolume` (it writes the GBuffer's targets).

use super::{Binding, Material, MaterialOptions};

/// Writes the GBuffer's four colour targets (`kansei_gbuffer_out`; `kansei_gbuffer_out_specular`
/// for a surface `rt::RtReflectionsEffect` reflects in); see [`Material::standard_lit`].
pub const GBUFFER_OUT_WGSL: &str = include_str!("../shaders/gbuffer_out.wgsl");

const STANDARD_LIT_WGSL: &str = include_str!("../shaders/standard_lit.wgsl");
const ENVIRONMENT_AMBIENT_WGSL: &str = include_str!("../shaders/environment_ambient.wgsl");

/// Where a [`Material::standard_lit`] surface's ambient comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StandardAmbient {
    /// `sky_up` and `sky_down`'s hemisphere.
    Hemisphere,
    /// An environment's SH, and its cubemap for the specular where it has one
    /// (`standard_lit_environment`).
    Environment { map: bool },
}

/// An environment's view or sampler, bound as it was made.
enum External {
    View(wgpu::TextureView),
    Sampler(wgpu::Sampler),
}

impl crate::buffers::Bindable for External {
    fn ensure_ready(&mut self, _device: &wgpu::Device, _queue: &wgpu::Queue) {}
    fn binding_resource(&self) -> Option<super::BindingResource<'_>> {
        Some(match self {
            Self::View(view) => super::BindingResource::TextureView(view),
            Self::Sampler(sampler) => super::BindingResource::Sampler(sampler),
        })
    }
}

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

/// How `rt::RtReflectionsEffect` traces a [`Material::standard_lit`] surface.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum StandardTraced {
    /// Not at all (its GBuffer alphas left at 1).
    #[default]
    None,
    /// Reflective: its reflection traced through the grid, by its F0 (0.04 for a dielectric, the
    /// base colour's brightest channel for a metal) and roughness.
    Reflective,
    /// Glass of this index of refraction (1.5 for window glass), drawn by the effect's glass pass
    /// (`RtReflectionsEffect::set_glass`): the base colour is its tint (the light left after a
    /// metre inside) and the roughness frosts it. Give its renderable `rt::RtSurface::glass` and
    /// `cast_shadow = false`, and no `gi`.
    Glass { ior: f32 },
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
    /// The specular factor, 0 to 1 (glTF's `KHR_materials_specular` `specularFactor`; the
    /// loader's `GLTFMaterialInfo::specular`): it scales a dielectric's Fresnel, its F0 of 0.04 and
    /// its rise to 1 at grazing angles alike, so 1 (the default) is the usual dielectric and 0 is
    /// matte, Lambert alone with no highlight or sheen at any angle (unfinished wood, chalk,
    /// cloth). A metal's Fresnel is its own. The forward lights and the environment's specular
    /// take it, and so does `rt::RtShadowsEffect` from the GBuffer (the normal target's alpha,
    /// `kansei_gbuffer_with_specular`); a reflective (`traced`) surface's F0 for the traced
    /// reflections is scaled by it, and at 0 it isn't traced.
    pub specular: f32,
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
    /// Traced by `rt::RtReflectionsEffect`: reflective, or glass.
    pub traced: StandardTraced,
}

impl Default for StandardLitOptions {
    fn default() -> Self {
        Self {
            base_color: [0.5, 0.5, 0.5],
            roughness: 0.6,
            metallic: 0.0,
            specular: 1.0,
            emissive: [0.0; 3],
            sky_up: [0.0; 3],
            sky_down: [0.0; 3],
            instancing: None,
            outputs_velocity: false,
            traced: StandardTraced::None,
        }
    }
}

impl StandardLitOptions {
    /// An unlit surface emitting `radiance` (cd/m²): lamp heads, windows, a bright backdrop.
    pub fn emissive(radiance: [f32; 3]) -> Self {
        Self { base_color: [0.0; 3], emissive: radiance, ..Default::default() }
    }

    /// A mirror or polished metal of this colour (its reflectance, linear rgb; 0.95 for chrome)
    /// and GGX roughness, whose reflection `rt::RtReflectionsEffect` traces.
    pub fn mirror(color: [f32; 3], roughness: f32) -> Self {
        Self { base_color: color, roughness, metallic: 1.0, traced: StandardTraced::Reflective, ..Default::default() }
    }

    /// Glass of this tint (the light left after a metre inside, linear rgb; 1 is clear), index of
    /// refraction and roughness (0 clear, frosted towards 1), drawn by `rt::RtReflectionsEffect`'s
    /// glass pass (see [`StandardTraced::Glass`]).
    pub fn glass(tint: [f32; 3], ior: f32, roughness: f32) -> Self {
        Self { base_color: tint, roughness, traced: StandardTraced::Glass { ior }, ..Default::default() }
    }

    /// A matte surface of this albedo and roughness (`specular` 0): Lambert alone.
    pub fn matte(color: [f32; 3], roughness: f32) -> Self {
        Self { base_color: color, roughness, specular: 0.0, ..Default::default() }
    }

    /// The material's group 0 uniform, as `KanseiStandardSurface` lays it out.
    fn uniform(&self) -> [f32; 20] {
        let [r, g, b] = self.base_color;
        let [er, eg, eb] = self.emissive;
        let [ur, ug, ub] = self.sky_up;
        let [dr, dg, db] = self.sky_down;
        let (traced, ior) = match self.traced {
            StandardTraced::None => (0.0, 1.0),
            StandardTraced::Reflective => (1.0, 1.0),
            StandardTraced::Glass { ior } => (2.0, ior.max(1.0)),
        };
        [
            r, g, b, self.specular.clamp(0.0, 1.0),
            er, eg, eb, 0.0,
            ur, ug, ub, 0.0,
            dr, dg, db, 0.0,
            self.roughness, self.metallic, traced, ior,
        ]
    }

    /// The shader: the chunks it uses, then the material with its placeholders filled.
    fn shader(&self) -> String {
        self.shader_with(StandardAmbient::Hemisphere)
    }

    fn shader_with(&self, ambient: StandardAmbient) -> String {
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
            .replace("KANSEI_VELOCITY_FRAGMENT", if velocity { "out.velocity = kansei_motion_vector(in.currClip, in.prevClip);" } else { "" })
            .replace(
                "KANSEI_AMBIENT",
                match ambient {
                    StandardAmbient::Hemisphere => "",
                    StandardAmbient::Environment { .. } => "radiance += kansei_environment_ambient(N, V, base, roughness, metallic, kansei_surface.params.z, specular);",
                },
            );
        let motion = if velocity { crate::cameras::MOTION_VECTORS_WGSL } else { "" };
        let environment = match ambient {
            StandardAmbient::Hemisphere => String::new(),
            StandardAmbient::Environment { map } => {
                let (bindings, reflection) = if map {
                    (
                        "@group(0) @binding(2) var kansei_environment : texture_cube<f32>;\n@group(0) @binding(3) var kansei_environment_sampler : sampler;",
                        "return skyEnvironment(kansei_environment, kansei_environment_sampler, r, roughness);",
                    )
                } else {
                    ("", "return skyRadiance(kansei_sky, r);")
                };
                let chunk = ENVIRONMENT_AMBIENT_WGSL.replace("KANSEI_ENVIRONMENT_BINDINGS", bindings).replace("KANSEI_ENVIRONMENT_REFLECTION", reflection);
                format!("{}\n{}\n{chunk}", crate::atmosphere::SKY_LIGHTING_WGSL, crate::atmosphere::SKY_ENVIRONMENT_WGSL)
            }
        };
        format!(
            "{}\n{}\n{}\n{}\n{motion}\n{environment}\n{body}",
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

    /// `standard_lit` with its ambient from an environment (`environment::EnvironmentLighting`:
    /// a `GradientEnvironment`, `SkyAtmosphere::lighting` or an `HdrEnvironment`) on top of
    /// `sky_up` and `sky_down` (leave them at zero): the irradiance of its SH on the diffuse albedo
    /// (the materials' own sky light the GI effects' `material_ambient` replaces), and, unless
    /// `rt::RtReflectionsEffect` traces the surface (`traced`), its specular from the
    /// environment's cubemap at the roughness (its SH without one). It binds the environment's
    /// handles, which follow its settings (an `HdrEnvironment`'s turn or intensity); another
    /// environment needs the material made again.
    pub fn standard_lit_environment(label: &str, options: &StandardLitOptions, environment: &crate::environment::EnvironmentLighting) -> Material {
        let ambient = StandardAmbient::Environment { map: environment.environment.is_some() };
        let fragment = wgpu::ShaderStages::FRAGMENT;
        let mut bindings = vec![Binding::uniform(0, fragment), Binding::uniform(1, fragment)];
        if environment.environment.is_some() {
            bindings.extend([Binding::texture_cube(2, fragment), Binding::sampler(3, fragment)]);
        }
        let material_options = MaterialOptions { mrt_output_count: Some(4), outputs_velocity: options.outputs_velocity, ..Default::default() };
        let mut material = Material::new(label, &options.shader_with(ambient), bindings, material_options);
        material.set_uniform_bindable(0, &format!("{label}/Surface"), &options.uniform());
        let sky = crate::buffers::ComputeBuffer::from_external(&format!("{label}/Sky"), environment.sky_lighting.clone(), crate::buffers::BufferType::Uniform);
        material.set_bindable(1, sky);
        if let Some(map) = &environment.environment {
            material.set_bindable(2, External::View(map.view.clone()));
            material.set_bindable(3, External::Sampler(map.sampler.clone()));
        }
        material
    }

    /// Change a `standard_lit` material's surface (colour, roughness, metallic, specular, emission,
    /// sky, traced) to `options`'; its instancing and velocity stay as it was made.
    pub fn set_standard_lit(&mut self, renderer: &crate::renderers::Renderer, options: &StandardLitOptions) {
        match self.bindable_buffer(0) {
            // drawn already: its uniform written in place (the scene's bundles keep its bind group)
            Some(buffer) => renderer.raw_queue().write_buffer(&buffer, 0, bytemuck::cast_slice(&options.uniform())),
            None => {
                let label = format!("{}/Surface", self.label);
                self.set_uniform_bindable(0, &label, &options.uniform());
            }
        }
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
    fn the_environment_variants_validate() {
        for map in [false, true] {
            for traced in [StandardTraced::None, StandardTraced::Reflective] {
                let options = StandardLitOptions { traced, outputs_velocity: map, ..Default::default() };
                let module = validate(&format!("environment map {map}"), &options.shader_with(StandardAmbient::Environment { map }));
                assert_eq!(struct_size(&module, "KanseiStandardSurface"), std::mem::size_of_val(&options.uniform()));
                assert_eq!(struct_size(&module, "SkyLighting"), std::mem::size_of::<crate::gi::SkyLightingData>());
            }
        }
        // the hemisphere's shader reads no environment
        assert!(!StandardLitOptions::default().shader().contains("kansei_environment"));
    }

    #[test]
    fn traced_surfaces_tell_the_shader_their_mode_and_ior() {
        let params = |o: StandardLitOptions| o.uniform()[16..20].to_vec();
        assert_eq!(params(StandardLitOptions { roughness: 0.6, ..Default::default() }), [0.6, 0.0, 0.0, 1.0]);
        assert_eq!(params(StandardLitOptions::mirror([0.95; 3], 0.2)), [0.2, 1.0, 1.0, 1.0]);
        assert_eq!(params(StandardLitOptions::glass([0.9; 3], 1.5, 0.1)), [0.1, 0.0, 2.0, 1.5]);
        // an index under air's is air's
        assert_eq!(params(StandardLitOptions::glass([1.0; 3], 0.5, 0.0))[3], 1.0);
        validate(
            "gbuffer out glass",
            &format!("{GBUFFER_OUT_WGSL}\n@fragment fn f() -> KanseiGBufferOut {{ return kansei_gbuffer_out_glass(vec3f(0.0, 1.0, 0.0), vec3f(0.9), 1.5, 0.2); }}"),
        );
    }

    #[test]
    fn the_specular_factor_rides_in_the_base_colours_alpha() {
        // the default writes the 1 the uniform held there before the factor (the TS packer's too)
        assert_eq!(StandardLitOptions::default().uniform()[3], 1.0);
        assert_eq!(StandardLitOptions::mirror([0.95; 3], 0.2).uniform()[3], 1.0);
        assert_eq!(StandardLitOptions::matte([0.5; 3], 0.7).uniform()[..4], [0.5, 0.5, 0.5, 0.0]);
        assert_eq!(StandardLitOptions { specular: 0.25, ..Default::default() }.uniform()[3], 0.25);
        // clamped to 0..1: over 1 would read as a reflective surface's F0 in the GBuffer
        assert_eq!(StandardLitOptions { specular: 3.0, ..Default::default() }.uniform()[3], 1.0);
        assert_eq!(StandardLitOptions { specular: -1.0, ..Default::default() }.uniform()[3], 0.0);
        validate(
            "gbuffer with specular",
            &format!(
                "{GBUFFER_OUT_WGSL}\n@fragment fn f() -> KanseiGBufferOut {{ return kansei_gbuffer_with_specular(kansei_gbuffer_out(vec3f(1.0), vec3f(0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.5)), 0.0); }}"
            ),
        );
        validate(
            "brdf with specular",
            &format!(
                "{}\n{}\n{}\n@fragment fn f(@builtin(position) p: vec4f) -> @location(0) vec4f {{ let n = vec3f(0.0, 1.0, 0.0); return vec4f(kansei_brdf_specular(n, n, n, vec3f(0.5), 0.5, 0.0, 0.0) + kansei_spot_lights_radiance_specular(p.xyz, n, n, vec3f(0.5), 0.5, 0.0, 0.0, p.xy), 1.0); }}",
                crate::lights::LIGHTS_WGSL,
                crate::shadows::SHADOW_MAP_WGSL,
                crate::lights::SPOT_LIGHTS_WGSL
            ),
        );
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
