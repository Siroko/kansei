mod shadow_map;
mod cubemap_shadow_map;
mod spot_shadow_atlas;
mod cascaded_shadow_map;
mod sky_occlusion;
pub(crate) mod compute_shadows;

pub use shadow_map::ShadowMap;
pub use cubemap_shadow_map::CubeMapShadowMap;
pub use spot_shadow_atlas::SpotShadowAtlas;
pub use cascaded_shadow_map::{cascade_splits, frustum_slice_sphere, CascadedShadowMap, CascadedShadowOptions, MAX_CASCADES};
pub use sky_occlusion::{SkyOcclusion, SkyOcclusionOptions};

/// WGSL for materials dimmed by the renderer's sky occlusion: `skyVisibility(volume, sampler,
/// params, worldPos)`, the share of the sky a point sees past the canopy; see
/// `Renderer::enable_sky_occlusion`.
pub const SKY_OCCLUSION_WGSL: &str = include_str!("../shaders/sky_occlusion.wgsl");

/// WGSL for materials shadowed by the single directional shadow map (`Renderer::enable_shadows`,
/// group 3 bindings 0-2): call `kansei_shadow_map(worldPos, N)` on the first directional light.
pub const SHADOW_MAP_WGSL: &str = include_str!("../shaders/shadow_map.wgsl");

/// WGSL for materials shadowed by the renderer's cascaded shadow map: group 3 bindings 10-12 and
/// `kansei_sun_shadow(worldPos, N, fragCoord.xy)`; see `Renderer::enable_cascaded_shadows`.
pub const CASCADED_SHADOWS_WGSL: &str = include_str!("../shaders/cascaded_shadows.wgsl");
