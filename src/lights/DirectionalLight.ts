import { Light } from "./Light";
import { ShadowMap, ShadowMapOptions } from "../shadows/ShadowMap";

class DirectionalLight extends Light {
    public direction: [number, number, number];
    public shadowMap: ShadowMap | null = null;

    constructor(
        direction: [number, number, number] = [0, -1, 0],
        color: [number, number, number] = [1, 1, 1],
        intensity: number = 1,
    ) {
        super('directional', color, intensity);
        this.direction = direction;
    }

    enableShadows(device: GPUDevice, options?: ShadowMapOptions): ShadowMap {
        this.shadowMap = new ShadowMap(device, options);
        return this.shadowMap;
    }
}

export { DirectionalLight };
