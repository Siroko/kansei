import { vec3 } from "gl-matrix";
import { Vector } from "./Vector";

/**
 * A class representing a 3D vector, extending BufferBase.
 * Provides methods to manipulate and access vector components.
 */
class Vector3 extends Vector {

    /**
     * Constructs a new Vector3 instance.
     * 
     * @param x - The x component of the vector.
     * @param y - The y component of the vector.
     * @param z - The z component of the vector.
     */
    constructor(x: number = 0, y: number = 0, z: number = 0) {
        super();
        this.internalVec = vec3.create();
        vec3.set(this.internalVec, x, y, z);
        this.buffer = new Float32Array(this.internalVec);
    }

    public set(x: number, y: number, z: number) {
        vec3.set(this.internalVec as vec3, x, y, z);
        this.updateBuffer();
    }

    /** Gets the z component of the vector. */
    get z() {
        return this.internalVec![2]!;
    }

    /**
     * Sets the z component of the vector and updates the buffer.
     * 
     * @param value - The new value for the z component.
     */
    set z(value: number) {
        this.internalVec![2] = value;
        this.updateBuffer();
    }
}

export { Vector3 }
