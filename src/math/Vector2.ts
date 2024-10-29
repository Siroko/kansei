import { vec2 } from "gl-matrix";
import { Vector } from "./Vector";

/**
 * A class representing a 2D vector, extending BufferBase.
 * Provides methods to manipulate and access vector components.
 */
class Vector2 extends Vector {
    /**
     * Constructs a new Vector2 instance.`
     * 
     * @param x - The x component of the vector.
     * @param y - The y component of the vector.
     */
    constructor(x: number = 0, y: number = 0) {
        super();
        this.internalVec = vec2.create();
        vec2.set(this.internalVec, x, y);
        this.buffer = new Float32Array(this.internalVec);
    }

    public set(x: number, y: number) {
        vec2.set(this.internalVec as vec2, x, y);
        this.updateBuffer();
    }
}

export { Vector2 }
