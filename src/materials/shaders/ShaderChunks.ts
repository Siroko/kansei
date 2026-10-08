import { shaderCode as fbm } from "./compute/noise/fbm";
import { shaderCode as curl } from "./compute/noise/curl";
import { GBUFFER_OUT_WGSL, LIGHTS_WGSL, SHADOW_MAP_WGSL, SPOT_LIGHTS_WGSL } from "./SharedWGSL";

/**
 * A collection of shader code chunks used in the rendering process.
 *
 * @property {string} fbm - The shader code for fractional Brownian motion noise.
 * @property {string} curl - The shader code for curl noise.
 * @property {string} shadows - The directional and point shadow lookups (`kansei_shadow_map`, `kansei_point_shadow`), shared with the Rust engine.
 * @property {string} lights - The scene light uniform (`kansei_lights`) and `kansei_point_falloff`, shared with the Rust engine.
 * @property {string} gbufferOut - The GBuffer's four outputs (`kansei_gbuffer_out`, `kansei_gbuffer_out_specular`), shared with the Rust engine.
 * @property {string} spotLights - The renderer's spot lights with their shadows and clusters (`kansei_spot_lights_radiance`), shared with the Rust engine.
 */
export const ShaderChunks = {
    fbm: fbm,
    curl: curl,
    shadows: SHADOW_MAP_WGSL,
    lights: LIGHTS_WGSL,
    gbufferOut: GBUFFER_OUT_WGSL,
    spotLights: SPOT_LIGHTS_WGSL
};
