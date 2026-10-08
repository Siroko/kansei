import type { DirectionalLight } from "./DirectionalLight";
import type { PointLight } from "./PointLight";
import { LIGHT_UNIFORM_BYTES } from "../renderers/SharedLayouts";

/** Directional lights `LightUniforms` packs (`KanseiLights.directional`); later ones are left out. */
export const MAX_DIRECTIONAL_LIGHTS = 4;
/** Point lights `LightUniforms` packs (`KanseiLights.point`); later ones are left out. */
export const MAX_POINT_LIGHTS = 8;

const HEADER_FLOATS = 4;
const DIR_LIGHT_FLOATS = 8;
const POINT_LIGHT_FLOATS = 8;
const POINT_OFFSET = HEADER_FLOATS + MAX_DIRECTIONAL_LIGHTS * DIR_LIGHT_FLOATS;

/**
 * Packs scene lights into the uniform at camera binding 2 (`KanseiLights` in `LIGHTS_WGSL`), as
 * the Rust engine's `lights::LightUniforms`: the first 4 directional and 8 point lights, their
 * colour times intensity, point lights at their world position with their radius. Spot and area
 * lights are not in it.
 */
export class LightUniforms {
    /** The packed uniform, `LIGHT_UNIFORM_BYTES` long; the two counts are u32s. */
    public readonly data = new Float32Array(LIGHT_UNIFORM_BYTES / 4);
    private readonly _counts = new Uint32Array(this.data.buffer, 0, 2);

    /**
     * Packs `directional` and `point` (in that order of preference within each list) into `data`.
     * Point lights read their world matrix, so their parents' transforms apply.
     */
    public pack(directional: readonly DirectionalLight[], point: readonly PointLight[]): void {
        const data = this.data;
        data.fill(0);

        const numDirectional = Math.min(directional.length, MAX_DIRECTIONAL_LIGHTS);
        for (let i = 0; i < numDirectional; i++) {
            const light = directional[i];
            const o = HEADER_FLOATS + i * DIR_LIGHT_FLOATS;
            const color = light.effectiveColor;
            data[o] = light.direction[0];
            data[o + 1] = light.direction[1];
            data[o + 2] = light.direction[2];
            data[o + 4] = color[0];
            data[o + 5] = color[1];
            data[o + 6] = color[2];
            data[o + 7] = light.intensity;
        }

        const numPoint = Math.min(point.length, MAX_POINT_LIGHTS);
        for (let i = 0; i < numPoint; i++) {
            const light = point[i];
            light.updateModelMatrix();
            const world = light.worldMatrix.internalMat4;
            const o = POINT_OFFSET + i * POINT_LIGHT_FLOATS;
            const color = light.effectiveColor;
            data[o] = world[12];
            data[o + 1] = world[13];
            data[o + 2] = world[14];
            data[o + 3] = light.radius;
            data[o + 4] = color[0];
            data[o + 5] = color[1];
            data[o + 6] = color[2];
            data[o + 7] = light.intensity;
        }

        this._counts[0] = numDirectional;
        this._counts[1] = numPoint;
    }
}
