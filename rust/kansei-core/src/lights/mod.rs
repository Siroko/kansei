mod directional;
mod point;
mod area;

pub use directional::DirectionalLight;
pub use point::PointLight;
pub use area::AreaLight;

/// A scene light.
pub enum Light {
    Directional(DirectionalLight),
    Point(PointLight),
    Area(AreaLight),
}
