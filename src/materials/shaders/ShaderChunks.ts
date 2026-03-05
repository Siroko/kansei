import { shaderCode as fbm } from "./compute/noise/fbm";
import { shaderCode as curl } from "./compute/noise/curl";
import { shaderCode as shadows } from "./shadows";

/**
 * A collection of shader code chunks used in the rendering process.
 *
 * @property {string} fbm - The shader code for fractional Brownian motion noise.
 * @property {string} curl - The shader code for curl noise.
 * @property {string} shadows - The shader code for directional shadow sampling.
 */
export const ShaderChunks = {
    fbm: fbm,
    curl: curl,
    shadows: shadows
};
