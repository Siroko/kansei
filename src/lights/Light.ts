import { Object3D } from "../objects/Object3D";

abstract class Light extends Object3D {
    public readonly isLight = true;
    public readonly lightType: 'directional' | 'point' | 'area';
    public color: [number, number, number];
    public intensity: number;
    public volumetric: boolean;
    /**
     * Whether the renderer's shadow maps are rendered from this light (`Renderer.enableShadows`:
     * the first directional light that casts, else the first area light; `enablePointShadows`:
     * the point lights that cast). Off by default, as in the Rust engine.
     */
    public castShadow: boolean = false;

    constructor(
        lightType: 'directional' | 'point' | 'area',
        color: [number, number, number] = [1, 1, 1],
        intensity: number = 1,
    ) {
        super();
        this.lightType = lightType;
        this.color = color;
        this.intensity = intensity;
        this.volumetric = true;
    }

    get effectiveColor(): [number, number, number] {
        return [
            this.color[0] * this.intensity,
            this.color[1] * this.intensity,
            this.color[2] * this.intensity,
        ];
    }
}

export { Light };
