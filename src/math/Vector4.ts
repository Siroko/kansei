import { vec4 } from "gl-matrix";
import { Vector } from "./Vector";

/**
 * A class representing a 4-dimensional vector, extending BufferBase.
 * Provides methods to manipulate and update the vector components.
 */
class Vector4 extends Vector {
    /**
     * Constructs a new Vector4 instance.
     * 
     * @param x - The x component of the vector.
     * @param y - The y component of the vector.
     * @param z - The z component of the vector.
     * @param w - The w component of the vector.
     */
    constructor(x: number = 0, y: number = 0, z: number = 0, w: number = 0) {
        super();
        this.internalVec = vec4.create();
        vec4.set(this.internalVec, x, y, z, w);
        this.buffer = new Float32Array(this.internalVec);
    }

    public set(x: number, y: number, z: number, w: number) {
        vec4.set(this.internalVec as vec4, x, y, z, w);
        this.updateBuffer();
    }

    /** Gets the z component of the vector. */
    get z() {
        return this.internalVec![2]!;
    }

    /** @returns The w component of the vector. */
    get w() {
        return this.internalVec![3]!;
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

    /**
     * Sets the w component of the vector and updates the buffer.
     * 
     * @param value - The new value for the w component.
     */
    set w(value: number) {
        this.internalVec![3] = value;
        this.updateBuffer();
    }
}

export { Vector4 }
