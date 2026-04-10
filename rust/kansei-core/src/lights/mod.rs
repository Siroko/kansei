mod directional;
mod point;
mod area;
mod light_uniforms;

pub use directional::DirectionalLight;
pub use point::PointLight;
pub use area::AreaLight;
pub use light_uniforms::{LightUniforms, LIGHT_UNIFORM_BYTES};

/// A scene light.
pub enum Light {
    Directional(DirectionalLight),
    Point(PointLight),
    Area(AreaLight),
}
