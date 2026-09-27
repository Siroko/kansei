mod shadow_map;
mod cubemap_shadow_map;
mod spot_shadow_atlas;
mod cascaded_shadow_map;

pub use shadow_map::ShadowMap;
pub use cubemap_shadow_map::CubeMapShadowMap;
pub use spot_shadow_atlas::SpotShadowAtlas;
pub use cascaded_shadow_map::{cascade_splits, frustum_slice_sphere, CascadedShadowMap, CascadedShadowOptions, MAX_CASCADES};

/// WGSL for materials shadowed by the renderer's cascaded shadow map: group 3 bindings 10-12 and
/// `kansei_sun_shadow(worldPos, N, fragCoord.xy)`; see `Renderer::enable_cascaded_shadows`.
pub const CASCADED_SHADOWS_WGSL: &str = include_str!("../shaders/cascaded_shadows.wgsl");
