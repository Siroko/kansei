import { BufferBase } from "../buffers/BufferBase";

class Float extends BufferBase {
    /** The type of buffer, set to uniform. */
    public type: string = BufferBase.BUFFER_TYPE_UNIFORM;

    /** The internal representation of the float value. */
    public internalFloat: number[] = [0];

    /**
     * Constructs a new Float instance.
     */
    constructor(value: number) {
        super();
        this.internalFloat = [value];
        this.buffer = new Float32Array(this.internalFloat);
    }

    /**
     * Updates the buffer with the current value and marks it for update.
     * @private
     */
    protected updateBuffer() {
        this.buffer!.set(this.internalFloat);
        this.needsUpdate = true;
    }

    public set value(value: number) {
        this.internalFloat[0] = value;
        this.updateBuffer();
    }

    public get value() {
        return this.internalFloat[0];
    }

    /**
     * Returns the value of this instance as a number
     *
     * @returns {number}
     *
     */
    public valueOf(): number {
        return this.internalFloat[0];
    }
}

export { Float }
