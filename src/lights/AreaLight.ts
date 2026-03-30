import { Light } from "./Light";
import { Vector3 } from "../math/Vector3";
import { ShadowMap, ShadowMapOptions } from "../shadows/ShadowMap";

class AreaLight extends Light {
    /** Width and depth of the rectangular emitting surface. */
    public size: [number, number];

    /** Falloff radius for attenuation and shadow range. */
    public radius: number;

    /** World-space point the light faces toward. Direction = normalize(target - position). */
    public target: Vector3;

    public shadowMap: ShadowMap | null = null;

    constructor(
        color: [number, number, number] = [1, 1, 1],
        intensity: number = 1,
        size: [number, number] = [1, 1],
        radius: number = 20,
    ) {
        super('area', color, intensity);
        this.size = size;
        this.radius = radius;
        // Default target: straight down from origin
        this.target = new Vector3(0, -1, 0);
    }

    /** Normalized direction the light faces (from position toward target). */
    get direction(): [number, number, number] {
        const dx = this.target.x - this.position.x;
        const dy = this.target.y - this.position.y;
        const dz = this.target.z - this.position.z;
        const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (len < 1e-8) return [0, -1, 0];
        return [dx / len, dy / len, dz / len];
    }

    enableShadows(device: GPUDevice, options?: ShadowMapOptions): ShadowMap {
        this.shadowMap = new ShadowMap(device, options);
        return this.shadowMap;
    }
}

export { AreaLight };
