import { Light } from "./Light";

class PointLight extends Light {
    public radius: number;

    constructor(
        radius: number = 50,
        color: [number, number, number] = [1, 1, 1],
        intensity: number = 1,
    ) {
        super('point', color, intensity);
        this.radius = radius;
    }
}

export { PointLight };
