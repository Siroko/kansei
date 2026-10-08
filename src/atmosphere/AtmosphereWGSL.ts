/**
 * The sky atmosphere's WGSL, imported from `rust/kansei-core/src/atmosphere/shaders` (Vite `?raw`)
 * and assembled in the Rust engine's order (`atmosphere/sky_atmosphere.rs`, `effects/atmosphere.rs`),
 * so both engines run the same source and `cargo test -p kansei-core` validates it.
 */
import { assemble } from '../materials/shaders/ShaderUtils';
import common from '../../rust/kansei-core/src/atmosphere/shaders/common.wgsl?raw';
import frame from '../../rust/kansei-core/src/atmosphere/shaders/frame.wgsl?raw';
import lookupTransmittance from '../../rust/kansei-core/src/atmosphere/shaders/lookup_transmittance.wgsl?raw';
import lookupMultiScattering from '../../rust/kansei-core/src/atmosphere/shaders/lookup_multi_scattering.wgsl?raw';
import scattering from '../../rust/kansei-core/src/atmosphere/shaders/scattering.wgsl?raw';
import skyLookup from '../../rust/kansei-core/src/atmosphere/shaders/sky_lookup.wgsl?raw';
import aerialPerspectiveLookup from '../../rust/kansei-core/src/atmosphere/shaders/aerial_perspective_lookup.wgsl?raw';
import transmittanceLut from '../../rust/kansei-core/src/atmosphere/shaders/transmittance_lut.wgsl?raw';
import multiScatteringLut from '../../rust/kansei-core/src/atmosphere/shaders/multi_scattering_lut.wgsl?raw';
import skyViewLut from '../../rust/kansei-core/src/atmosphere/shaders/sky_view_lut.wgsl?raw';
import aerialPerspectiveLut from '../../rust/kansei-core/src/atmosphere/shaders/aerial_perspective_lut.wgsl?raw';
import skyLighting from '../../rust/kansei-core/src/atmosphere/shaders/sky_lighting.wgsl?raw';
import cloudMap from '../../rust/kansei-core/src/atmosphere/shaders/cloud_map.wgsl?raw';
import skyCapture from '../../rust/kansei-core/src/atmosphere/shaders/sky_capture.wgsl?raw';
import distantSkyLight from '../../rust/kansei-core/src/atmosphere/shaders/distant_sky_light.wgsl?raw';
import skyLightingPass from '../../rust/kansei-core/src/atmosphere/shaders/sky_lighting_pass.wgsl?raw';
import skyComposite from '../../rust/kansei-core/src/atmosphere/shaders/sky_composite.wgsl?raw';
import heightFog from '../../rust/kansei-core/src/shaders/height_fog.wgsl?raw';

/**
 * The WGSL `Atmosphere` and `SkyFrame` structs and the atmosphere helpers (medium, phase
 * functions, ray-sphere tests, LUT parameterisations), for shaders that read the atmosphere.
 * Such a shader declares `atm : Atmosphere` and, for the frame helpers, `frame : SkyFrame`.
 * Rust: `atmosphere::ATMOSPHERE_WGSL`.
 */
export const ATMOSPHERE_WGSL: string = assemble([common, frame]);

/**
 * The WGSL `SkyLighting` struct and its helpers, for materials and media lit by the sky:
 * `skyIrradiance(sky, n)`, `skyRadiance(sky, d)` and `skyInscatter(sky, viewDir, g)`. Bind
 * `SkyAtmosphere.bindings.skyLighting` as a uniform of type `SkyLighting`, for example with
 * `ComputeBuffer.fromExternal(buffer, 'uniform')`. Rust: `atmosphere::SKY_LIGHTING_WGSL`. The
 * package's `SKY_LIGHTING_WGSL` export (`gi/GiWGSL`) is the same source.
 */
export const SKY_LIGHTING_WGSL: string = skyLighting;

// The passes, in the Rust engine's concatenation order

export const TRANSMITTANCE_SOURCE = assemble([common, transmittanceLut]);
export const MULTI_SCATTERING_SOURCE = assemble([common, lookupTransmittance, multiScatteringLut]);
export const SKY_VIEW_SOURCE = assemble([common, frame, lookupTransmittance, lookupMultiScattering, scattering, skyViewLut]);
export const SKY_LIGHTING_SOURCE = assemble([common, frame, lookupTransmittance, skyLookup, skyLighting, cloudMap, skyCapture, skyLightingPass]);
export const DISTANT_SKY_LIGHT_SOURCE = assemble([common, frame, lookupTransmittance, lookupMultiScattering, distantSkyLight]);
export const AERIAL_PERSPECTIVE_SOURCE = assemble([common, frame, lookupTransmittance, lookupMultiScattering, scattering, aerialPerspectiveLut]);
/** `AtmosphereEffect`'s composite: the sky, sun and moon behind the scene, aerial perspective over it. */
export const SKY_COMPOSITE_SOURCE = assemble([common, frame, lookupTransmittance, skyLookup, aerialPerspectiveLookup, skyComposite]);
/** `HeightFogEffect`'s pass (Rust `effects/height_fog.rs`). */
export const HEIGHT_FOG_SOURCE = assemble([skyLighting, heightFog]);
