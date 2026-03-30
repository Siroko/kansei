import { Light } from "./Light";
import { CubeMapShadowMap, CubeMapShadowMapOptions } from "../shadows/CubeMapShadowMap";

class PointLight extends Light {
    public radius: number;
    public shadowMap: CubeMapShadowMap | null = null;

    constructor(
        radius: number = 50,
        color: [number, number, number] = [1, 1, 1],
        intensity: number = 1,
    ) {
        super('point', color, intensity);
        this.radius = radius;
    }

    enableShadows(device: GPUDevice, options?: CubeMapShadowMapOptions): CubeMapShadowMap {
        this.shadowMap = new CubeMapShadowMap(device, options);
        return this.shadowMap;
    }
}

export { PointLight };
