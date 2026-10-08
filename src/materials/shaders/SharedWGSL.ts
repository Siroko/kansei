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
import bloomDownsample from '../../../rust/kansei-core/src/shaders/bloom_downsample.wgsl?raw';
import bloomUpsample from '../../../rust/kansei-core/src/shaders/bloom_upsample.wgsl?raw';
import bloomComposite from '../../../rust/kansei-core/src/shaders/bloom_composite.wgsl?raw';
import gbufferOut from '../../../rust/kansei-core/src/shaders/gbuffer_out.wgsl?raw';
import motionVectors from '../../../rust/kansei-core/src/shaders/motion_vectors.wgsl?raw';
import standardLit from '../../../rust/kansei-core/src/shaders/standard_lit.wgsl?raw';
import gradientSky from '../../../rust/kansei-core/src/shaders/gradient_sky.wgsl?raw';
import basicLit from '../../../rust/kansei-core/src/shaders/basic_lit.wgsl?raw';
import basicInstanced from '../../../rust/kansei-core/src/shaders/basic_instanced.wgsl?raw';
import particleBillboard from '../../../rust/kansei-core/src/shaders/particle_billboard.wgsl?raw';
import spotLightTypes from '../../../rust/kansei-core/src/shaders/spot_light_types.wgsl?raw';
import spotLights from '../../../rust/kansei-core/src/shaders/spot_lights.wgsl?raw';
import cascadedShadows from '../../../rust/kansei-core/src/shaders/cascaded_shadows.wgsl?raw';
import lightClusters from '../../../rust/kansei-core/src/shaders/light_clusters.wgsl?raw';
import taaResolve from '../../../rust/kansei-core/src/shaders/taa_resolve.wgsl?raw';
import motionBlurCommon from '../../../rust/kansei-core/src/shaders/motion_blur_common.wgsl?raw';
import motionBlurPrepare from '../../../rust/kansei-core/src/shaders/motion_blur_prepare.wgsl?raw';
import motionBlurNeighbours from '../../../rust/kansei-core/src/shaders/motion_blur_neighbours.wgsl?raw';
import motionBlurGather from '../../../rust/kansei-core/src/shaders/motion_blur_gather.wgsl?raw';

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

/**
 * `BloomEffect`'s three passes, each with its 32-byte `BloomParams` at group 0 binding 2 or 4:
 * the first level's Karis-weighted tent downsample and exposure-aware threshold, the tent
 * upsample (spread by `radius`), and the composite (added on top, or mixed in when
 * `threshold <= 0`). Rust: `postprocessing/effects/bloom.rs`.
 */
export const BLOOM_DOWNSAMPLE_WGSL: string = bloomDownsample;
export const BLOOM_UPSAMPLE_WGSL: string = bloomUpsample;
export const BLOOM_COMPOSITE_WGSL: string = bloomComposite;

/**
 * `TemporalAAEffect`'s resolve (and temporal upscaler): `TaaParams` (240 bytes) at group 0
 * binding 7, the frame, depth, velocity and history at bindings 0-3, the output and next
 * history at 5-6. Rust: `postprocessing/effects/taa.rs`'s `WGSL`.
 */
export const TAA_RESOLVE_WGSL: string = taaResolve;

/**
 * `MotionBlurEffect`'s three passes, each prefixed by the common `MotionBlurParams` (224 bytes):
 * the per-pixel blur vectors and per-tile maximum (group 0 bindings 0-4), the neighbour-tile
 * maximum (0-2), and the gather (0-4). Rust: `postprocessing/effects/motion_blur.rs`.
 */
export const MOTION_BLUR_PREPARE_WGSL: string = `${motionBlurCommon}\n${motionBlurPrepare}`;
export const MOTION_BLUR_NEIGHBOURS_WGSL: string = `${motionBlurCommon}\n${motionBlurNeighbours}`;
export const MOTION_BLUR_GATHER_WGSL: string = `${motionBlurCommon}\n${motionBlurGather}`;

/**
 * The GBuffer's four colour targets for a fragment shader (`KanseiGBufferOut`): return
 * `kansei_gbuffer_out(color, emissive, N, albedo)`, or `kansei_gbuffer_out_specular(..)` for a
 * surface that reflects, from a material with `mrtOutputCount: 4`. Rust: `materials::GBUFFER_OUT_WGSL`.
 *
 * Its alphas follow the Rust GBuffer: the normal's alpha is 1 (1 - F0 for specular surfaces),
 * which the TS `FluidTransmissionEffect` still reads as its fluid mask.
 */
export const GBUFFER_OUT_WGSL: string = gbufferOut;

/**
 * For materials that write motion vectors (`MaterialOptions.outputsVelocity`): the camera's
 * temporal uniform (`kansei_camera_temporal`, group 1 binding 3), `KanseiMeshTransforms` (group 2
 * binding 1: world and previous world matrix) and `kansei_motion_vector`.
 * Rust: `cameras::MOTION_VECTORS_WGSL`.
 */
export const MOTION_VECTORS_WGSL: string = motionVectors;

/**
 * Blinn-Phong under the scene's directional and point lights (camera group binding 2), with the
 * single directional shadow map and the point-light cube shadow (group 3). Forward, one colour
 * output. Group 0 binding 0: `color` then `specular` (rgb, shininess / 256 in `a`), two vec4s.
 * `Material.basicLit` builds it. Rust: `materials::BASIC_LIT_WGSL`.
 */
export const BASIC_LIT_WGSL: string = basicLit;

/**
 * A flat colour lit by a fixed light from above, for instanced geometry: the instance's model
 * matrix comes as four vec4 vertex attributes at locations 3-6. Group 0 binding 0: `color`
 * (vec4). `Material.basicInstanced` builds it. Rust: `materials::BASIC_INSTANCED_WGSL`.
 */
export const BASIC_INSTANCED_WGSL: string = basicInstanced;

/**
 * Camera-facing quads for particles, one per instance at a vec4 position (location 3), coloured
 * by height. Group 0 binding 0: `size`, `height_min`, `height_max`, a pad, then `color_low` and
 * `color_high` (vec4s), 48 bytes. Rust: `materials::PARTICLE_BILLBOARD_WGSL`.
 */
export const PARTICLE_BILLBOARD_WGSL: string = particleBillboard;

/** The standard lit material's body, with its `KANSEI_*` placeholders (`Material.standardLit` fills them). */
export const STANDARD_LIT_BODY_WGSL: string = standardLit;

/** The gradient sky's body, prefixed by `GBUFFER_OUT_WGSL` (`Material.gradientSky`). */
export const GRADIENT_SKY_BODY_WGSL: string = gradientSky;

/**
 * The spot-light data (`KanseiSpotLight`, `KanseiSpotLights`) and the light's own falloff, cone
 * and shadow-map coordinates. Rust: `lights::SPOT_LIGHT_TYPES_WGSL`.
 */
export const SPOT_LIGHT_TYPES_WGSL: string = spotLightTypes;

/**
 * For materials lit by the renderer's spot lights: `SPOT_LIGHT_TYPES_WGSL`, the group 3 bindings
 * 5-9 (shadow atlas, light buffer, comparison sampler, light clusters), PCSS shadows and a GGX /
 * Lambert BRDF (`kansei_brdf`). Call `kansei_spot_lights_radiance`. Rust: `lights::SPOT_LIGHTS_WGSL`.
 */
export const SPOT_LIGHTS_WGSL: string = `${spotLightTypes}${spotLights}`;

/**
 * The cascaded sun shadow for materials (group 3 bindings 10-12): `kansei_sun_shadow(worldPos,
 * N, fragCoord.xy)`, and `kansei_cascades` (`count` 0 while no cascades are on).
 * Rust: `shadows::CASCADED_SHADOWS_WGSL`.
 */
export const CASCADED_SHADOWS_WGSL: string = cascadedShadows;

/**
 * The clustered light culling compute pass (`LightClusters`): `main` at group 0 (cluster
 * parameters, spot lights, per-cluster light lists). Rust: `lights/light_clusters.rs`'s `WGSL`.
 */
export const LIGHT_CLUSTERS_WGSL: string = `${spotLightTypes}\n${lightClusters}`;
