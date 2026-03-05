import { Light } from "./Light";

class DirectionalLight extends Light {
    public direction: [number, number, number];

    constructor(
        direction: [number, number, number] = [0, -1, 0],
        color: [number, number, number] = [1, 1, 1],
        intensity: number = 1,
    ) {
        super('directional', color, intensity);
        this.direction = direction;
    }
}

export { DirectionalLight };
