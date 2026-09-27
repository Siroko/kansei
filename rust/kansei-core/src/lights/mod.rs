mod directional;
mod point;
mod area;
mod spot;
pub(crate) mod spot_lights_gpu;
mod light_uniforms;

pub use directional::DirectionalLight;
pub use point::PointLight;
pub use area::AreaLight;
pub use spot::SpotLight;
pub use spot_lights_gpu::MAX_SPOT_LIGHTS;
pub use light_uniforms::{LightUniforms, LIGHT_UNIFORM_BYTES};

/// WGSL for the spot-light data (`KanseiSpotLight`, `KanseiSpotLights`) and the light's own
/// falloff, cone and shadow-map coordinates. Prepend it to shaders that bind the renderer's
/// spot-light buffer themselves (`Renderer::spot_lights_buffer`).
pub const SPOT_LIGHT_TYPES_WGSL: &str = include_str!("../shaders/spot_light_types.wgsl");

/// WGSL for materials lit by the renderer's spot lights: `SPOT_LIGHT_TYPES_WGSL`, the group 3
/// bindings (shadow atlas, light buffer, comparison sampler), PCSS shadows and a GGX/Lambert
/// BRDF. Prepend it to a material shader and call `kansei_spot_lights_radiance`.
pub const SPOT_LIGHTS_WGSL: &str = concat!(
    include_str!("../shaders/spot_light_types.wgsl"),
    include_str!("../shaders/spot_lights.wgsl"),
);

/// A scene light.
pub enum Light {
    Directional(DirectionalLight),
    Point(PointLight),
    Area(AreaLight),
    Spot(SpotLight),
}
