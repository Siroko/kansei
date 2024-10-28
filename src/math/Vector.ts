import { vec2, vec3, vec4 } from "gl-matrix";
import { BufferBase } from "../buffers/BufferBase";

/**
 * A class representing a vector, extending BufferBase.
 * Provides methods to manipulate and access vector components.
 */
class Vector extends BufferBase {
    /** The type of buffer, set to uniform. */
    public type: string = BufferBase.BUFFER_TYPE_UNIFORM;

    /** The internal representation of the vector using gl-matrix's vec-n. */
    public internalVec?: vec2 | vec3 | vec4;

    /**
     * Constructs a new Vector instance.`
     * 
     * @param x - The x component of the vector.
     * @param y - The y component of the vector.
     */
    constructor() {
        super();
    }

    /** Gets the x component of the vector. */
    get x() {
        return this.internalVec![0];
    }

    /** Gets the y component of the vector. */
    get y() {
        return this.internalVec![1];
    }

    /**
     * Updates the buffer with the current vector values and marks it for update.
     * @private
     */
    protected updateBuffer() {
        this.buffer!.set(this.internalVec!);
        this.needsUpdate = true;
    }

    /**
     * Sets the x component of the vector and updates the buffer.
     * 
     * @param value - The new value for the x component.
     */
    set x(value: number) {
        this.internalVec![0] = value;
        this.updateBuffer();
    }

    /**
     * Sets the y component of the vector and updates the buffer.
     * 
     * @param value - The new value for the y component.
     */
    set y(value: number) {
        this.internalVec![1] = value;
        this.updateBuffer();
    }

    /**
     * Sets a specific component of the vector and updates the buffer.
     * 
     * @param component - The index of the component to set (0 for x, 1 for y, 2 for z, 3 for w).
     * @param value - The new value for the specified component.
     */
    public setComponent(component: number, value: number) {
        this.internalVec![component] = value;
        this.updateBuffer();
    }

    /**
     * Returns the internal vec representation of the vector.
     * 
     * @returns The internal gl-matrix vec.
     */
    public toVec(): vec2 | vec3 | vec4 {
        return this.internalVec!;
    }

    public getVec(): vec2 | vec3 | vec4 {
        switch (this.internalVec!.length) {
            case 2:
                return this.internalVec! as vec2;
                break;
            case 3:
                return this.internalVec! as vec3;
                break;
            case 4:
                return this.internalVec! as vec4;
                break;
        }

        return this.internalVec!;
    }

    public copy(vec: Vector) {
        for (let i = 0; i < this.internalVec!.length; i++) {
            this.internalVec![i] = vec.internalVec![i];
        }
        this.updateBuffer();
    }

    public sub(vec: Vector) {
        for (let i = 0; i < this.internalVec!.length; i++) {
            this.internalVec![i] -= vec.internalVec![i];
        }
        this.updateBuffer();
    }

    public length() {
        switch (this.internalVec!.length) {
            case 2:
                return vec2.length(this.internalVec! as vec2);
                break;
            case 3:
                return vec3.length(this.internalVec! as vec3);
                break;
            case 4:
                return vec4.length(this.internalVec! as vec4);
                break;
        }

        return 0;
    }
}

export { Vector }
