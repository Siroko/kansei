/**
 * WGSL shared with the Rust engine: imported from `rust/kansei-core/src` (Vite `?raw`), so both
 * engines run the same source and `cargo test -p kansei-core` validates it. The names match the
 * Rust constants. Bindings follow `renderers/SharedLayouts`.
 */
import shadowMap from '../../../rust/kansei-core/src/shaders/shadow_map.wgsl?raw';
import lightUniforms from '../../../rust/kansei-core/src/shaders/light_uniforms.wgsl?raw';

/**
 * The directional shadow map and point-light cube shadow (group 3 bindings 0-3):
 * `kansei_shadow_map(worldPos, N)` and `kansei_point_shadow(worldPos)`, 1 lit and 0 shadowed.
 * Rust: `shadows::SHADOW_MAP_WGSL`.
 */
export const SHADOW_MAP_WGSL: string = shadowMap;

/**
 * The scene lights at camera binding 2 (`kansei_lights`, `KanseiLights`) and
 * `kansei_point_falloff`. Rust: `lights::LIGHTS_WGSL`.
 */
export const LIGHTS_WGSL: string = lightUniforms;
