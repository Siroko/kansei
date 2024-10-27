import { vec3 } from "gl-matrix";
import { BufferBase } from "../buffers/BufferBase";

/**
 * A class representing a 3D vector, extending BufferBase.
 * Provides methods to manipulate and access vector components.
 */
class Vector3 extends BufferBase {
    /** The type of buffer, set to uniform. */
    public type: string = BufferBase.BUFFER_TYPE_UNIFORM;

    /** The internal representation of the vector using gl-matrix's vec3. */
    public readonly internalVec3: vec3;

    /**
     * Constructs a new Vector3 instance.
     * 
     * @param x - The x component of the vector.
     * @param y - The y component of the vector.
     * @param z - The z component of the vector.
     */
    constructor(x: number = 0, y: number = 0, z: number = 0) {
        super();
        this.internalVec3 = vec3.create();
        vec3.set(this.internalVec3, x, y, z);
        this.buffer = new Float32Array(this.internalVec3);
    }

    /** Gets the x component of the vector. */
    get x() {
        return this.internalVec3[0];
    }

    /** Gets the y component of the vector. */
    get y() {
        return this.internalVec3[1];
    }

    /** Gets the z component of the vector. */
    get z() {
        return this.internalVec3[2];
    }

    /**
     * Updates the buffer with the current vector values and marks it for update.
     * @private
     */
    private updateBuffer() {
        this.buffer!.set(this.internalVec3);
        this.needsUpdate = true;
    }

    /**
     * Sets the x component of the vector and updates the buffer.
     * 
     * @param value - The new value for the x component.
     */
    set x(value: number) {
        this.internalVec3[0] = value;
        this.updateBuffer();
    }

    /**
     * Sets the y component of the vector and updates the buffer.
     * 
     * @param value - The new value for the y component.
     */
    set y(value: number) {
        this.internalVec3[1] = value;
        this.updateBuffer();
    }

    /**
     * Sets the z component of the vector and updates the buffer.
     * 
     * @param value - The new value for the z component.
     */
    set z(value: number) {
        this.internalVec3[2] = value;
        this.updateBuffer();
    }

    /**
     * Sets a specific component of the vector and updates the buffer.
     * 
     * @param component - The index of the component to set (0 for x, 1 for y, 2 for z).
     * @param value - The new value for the specified component.
     */
    public setComponent(component: number, value: number) {
        this.internalVec3[component] = value;
        this.updateBuffer();
    }

    /**
     * Returns the internal vec3 representation of the vector.
     * 
     * @returns The internal vec3.
     */
    public toVec(): vec3 {
        return this.internalVec3;
    }
}

export { Vector3 }
