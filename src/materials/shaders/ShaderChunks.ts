import { shaderCode as fbm } from "./compute/noise/fbm";
import { shaderCode as curl } from "./compute/noise/curl";

export const ShaderChunks = {
    fbm: fbm,
    curl: curl
};