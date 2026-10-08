import { mat4 } from "gl-matrix";
import { SpotLight } from "./SpotLight";

/** Most spot lights the renderer uploads per frame; later ones are ignored. */
export const MAX_SPOT_LIGHTS = 128;

/** Near plane of the spot shadow projections, metres. */
export const SPOT_SHADOW_NEAR = 0.05;

/** Bytes of one `KanseiSpotLight` (`spot_light_types.wgsl`). */
export const SPOT_LIGHT_BYTES = 144;

/** Bytes of the `KanseiSpotLights` header: the light count, padded to the array's 16-byte alignment. */
export const SPOT_LIGHTS_HEADER_BYTES = 16;

/** A shadow-casting spot light's layer of the shadow atlas this frame, with its matrices. */
export interface SpotShadowSlot {
    layer: number;
    view: mat4;
    projection: mat4;
    /** `projection * view`. */
    viewProj: mat4;
}

/**
 * CPU staging for the renderer's spot-light storage buffer (Rust `spot_lights_gpu`): the
 * `KanseiSpotLights` header, then a `KanseiSpotLight` (144 bytes) per light.
 */
export class SpotLightsGpu {
    /** Lights packed by the last `pack`. */
    public count = 0;
    /** Shadow slots assigned by the last `pack`, in layer order. */
    public readonly shadows: SpotShadowSlot[] = [];

    private readonly _bytes = new ArrayBuffer(SPOT_LIGHTS_HEADER_BYTES + MAX_SPOT_LIGHTS * SPOT_LIGHT_BYTES);
    private readonly _f32 = new Float32Array(this._bytes);
    private readonly _i32 = new Int32Array(this._bytes);
    private readonly _u32 = new Uint32Array(this._bytes);
    // A slot's matrices, kept across frames.
    private readonly _slotPool: SpotShadowSlot[] = [];
    private readonly _viewProj = mat4.create();
    private readonly _view = mat4.create();
    private readonly _projection = mat4.create();

    /**
     * Packs the scene's spot lights (at most `MAX_SPOT_LIGHTS`). The first `shadowLayers` lights
     * with `castShadow` get the atlas layers in scene order; `shadowResolution` is the atlas size
     * (0 without an atlas).
     */
    pack(lights: Iterable<SpotLight>, shadowLayers: number, shadowResolution: number): void {
        this.count = 0;
        this.shadows.length = 0;
        for (const spot of lights) {
            if (this.count === MAX_SPOT_LIGHTS) break;
            let layer = -1;
            let slot: SpotShadowSlot | null = null;
            if (spot.castShadow && this.shadows.length < shadowLayers) {
                layer = this.shadows.length;
                slot = this._slotPool[layer] ??= { layer, view: mat4.create(), projection: mat4.create(), viewProj: mat4.create() };
                this.shadows.push(slot);
            }
            this._packSpot(this.count++, spot, layer, shadowResolution, slot);
        }
        this._u32[0] = this.count;
    }

    /** The storage-buffer bytes of the last `pack`: header then lights. */
    get bytes(): Uint8Array {
        return new Uint8Array(this._bytes, 0, SPOT_LIGHTS_HEADER_BYTES + this.count * SPOT_LIGHT_BYTES);
    }

    private _packSpot(index: number, spot: SpotLight, shadowLayer: number, shadowResolution: number, slot: SpotShadowSlot | null): void {
        const [outer, inner] = spot.cone();
        const view = slot?.view ?? this._view;
        const projection = slot?.projection ?? this._projection;
        spot.shadowView(view);
        spot.shadowProjection(SPOT_SHADOW_NEAR, projection);
        const [px, py, pz] = spot.worldPosition;
        let [dx, dy, dz] = spot.direction;
        const len = Math.hypot(dx, dy, dz);
        if (len > 0) { dx /= len; dy /= len; dz /= len; } else { dx = 0; dy = 0; dz = -1; }

        const f = this._f32;
        const o = (SPOT_LIGHTS_HEADER_BYTES + index * SPOT_LIGHT_BYTES) / 4;
        f[o + 0] = px; f[o + 1] = py; f[o + 2] = pz;
        f[o + 3] = Math.max(spot.range, 1e-3);
        f[o + 4] = dx; f[o + 5] = dy; f[o + 6] = dz;
        f[o + 7] = Math.cos(outer);
        f[o + 8] = spot.color[0] * spot.intensity;
        f[o + 9] = spot.color[1] * spot.intensity;
        f[o + 10] = spot.color[2] * spot.intensity;
        f[o + 11] = Math.cos(inner);
        this._i32[o + 12] = shadowLayer;
        f[o + 13] = Math.max(spot.volumetricScale, 0);
        f[o + 14] = Math.max(spot.sourceRadius, 0);
        f[o + 15] = Math.max(spot.shadowNormalBias, 0);
        f[o + 16] = SPOT_SHADOW_NEAR;
        // the projection's vertical scale is 1 / tan(fov / 2)
        f[o + 17] = 1 / Math.max(Math.abs(projection[5]), 1e-6);
        f[o + 18] = shadowResolution > 0 ? 1 / shadowResolution : 0;
        f[o + 19] = 0;
        f.set(mat4.multiply(slot?.viewProj ?? this._viewProj, projection, view) as Float32Array, o + 20);
    }
}
