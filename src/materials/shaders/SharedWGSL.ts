/**
 * WGSL shared with the Rust engine: imported from `rust/kansei-core/src` (Vite `?raw`), so both
 * engines run the same source and `cargo test -p kansei-core` validates it. The names match the
 * Rust constants. Bindings follow `renderers/SharedLayouts`.
 */
import shadowMap from '../../../rust/kansei-core/src/shaders/shadow_map.wgsl?raw';
import lightUniforms from '../../../rust/kansei-core/src/shaders/light_uniforms.wgsl?raw';
import tonemapParams from '../../../rust/kansei-core/src/shaders/tonemap_params.wgsl?raw';
import tonemap from '../../../rust/kansei-core/src/shaders/tonemap.wgsl?raw';
import localExposure from '../../../rust/kansei-core/src/shaders/local_exposure.wgsl?raw';

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

/**
 * The display transform (`ToneMapEffect`): `ToneMapParams`, then the tonemap's `main` (group 0
 * bindings 0-5). Rust: `postprocessing/effects/tonemap.rs`'s `WGSL`.
 */
export const TONEMAP_WGSL: string = `${tonemapParams}\n${tonemap}`;

/**
 * Unreal's local exposure for `ToneMapEffect`: `ToneMapParams`, then the `grid`, `logLuminance`,
 * `blurX` and `blurY` entry points. Rust: `tonemap.rs`'s `LOCAL_EXPOSURE_WGSL`.
 */
export const LOCAL_EXPOSURE_WGSL: string = `${tonemapParams}\n${localExposure}`;
