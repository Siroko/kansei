import { Object3D } from "../objects/Object3D";

abstract class Light extends Object3D {
    public readonly isLight = true;
    public readonly lightType: 'directional' | 'point';
    public color: [number, number, number];
    public intensity: number;
    public volumetric: boolean;

    constructor(
        lightType: 'directional' | 'point',
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
