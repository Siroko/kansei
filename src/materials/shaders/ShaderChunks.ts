import { shaderCode as fbm } from "./compute/noise/fbm";
import { shaderCode as curl } from "./compute/noise/curl";

/**
 * A collection of shader code chunks used in the rendering process.
 * 
 * @property {string} fbm - The shader code for fractional Brownian motion noise.
 * @property {string} curl - The shader code for curl noise.
 */
export const ShaderChunks = {
    fbm: fbm,
    curl: curl
};
