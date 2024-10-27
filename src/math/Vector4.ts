import { vec4 } from "gl-matrix";
import { BufferBase } from "../buffers/BufferBase";

/**
 * A class representing a 4-dimensional vector, extending BufferBase.
 * Provides methods to manipulate and update the vector components.
 */
class Vector4 extends BufferBase {
    /** The type of buffer, set to uniform by default. */
    public type: string = BufferBase.BUFFER_TYPE_UNIFORM;

    /** Indicates whether the vector has been initialized. */
    public initialized: boolean = false;

    /** Internal representation of the vector using gl-matrix's vec4. */
    private internalVec4: vec4;

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
        this.internalVec4 = vec4.create();
        vec4.set(this.internalVec4, x, y, z, w);
        this.buffer = new Float32Array(this.internalVec4);
    }

    /** @returns The x component of the vector. */
    get x() {
        return this.internalVec4[0];
    }

    /** @returns The y component of the vector. */
    get y() {
        return this.internalVec4[1];
    }

    /** @returns The z component of the vector. */
    get z() {
        return this.internalVec4[2];
    }

    /** @returns The w component of the vector. */
    get w() {
        return this.internalVec4[3];
    }

    /**
     * Updates the buffer with the current vector values and marks it for update.
     * @private
     */
    private updateBuffer() {
        this.buffer!.set(this.internalVec4);
        this.needsUpdate = true;
    }

    /**
     * Sets the x component of the vector and updates the buffer.
     * 
     * @param value - The new value for the x component.
     */
    set x(value: number) {
        vec4.set(this.internalVec4, value, this.y, this.z, this.w);
        this.updateBuffer();
    }

    /**
     * Sets the y component of the vector and updates the buffer.
     * 
     * @param value - The new value for the y component.
     */
    set y(value: number) {
        vec4.set(this.internalVec4, this.x, value, this.z, this.w);
        this.updateBuffer();
    }

    /**
     * Sets the z component of the vector and updates the buffer.
     * 
     * @param value - The new value for the z component.
     */
    set z(value: number) {
        vec4.set(this.internalVec4, this.x, this.y, value, this.w);
        this.updateBuffer();
    }

    /**
     * Sets the w component of the vector and updates the buffer.
     * 
     * @param value - The new value for the w component.
     */
    set w(value: number) {
        vec4.set(this.internalVec4, this.x, this.y, this.z, value);
        this.updateBuffer();
    }

    /**
     * Sets a specific component of the vector and updates the buffer.
     * 
     * @param component - The index of the component to set (0 for x, 1 for y, 2 for z, 3 for w).
     * @param value - The new value for the specified component.
     */
    setComponent(component: number, value: number) {
        this.internalVec4[component] = value;
        this.updateBuffer();
    }
}

export { Vector4 }
