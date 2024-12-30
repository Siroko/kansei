import { BufferBase } from "./BufferBase";

// Type to get byte sizes of basic types
const TypeSizes = {
    float: 4,
    int: 4,
    uint: 4,
    vec2: 8,
    vec3: 12,
    vec4: 16,
    mat4: 64,
    // Add other types as needed
} as const;

// Type helper to describe structure fields
type StructField = {
    type: keyof typeof TypeSizes;
    buffer: ArrayBuffer;
    count: number;
};

// Type to describe a structure layout
type StructLayout = {
    [key: string]: StructField;
};

class StructuredBuffer extends BufferBase {
    private layout: StructLayout;

    constructor(structLayout: StructLayout) {
        super();
        this.layout = structLayout;
        this.stride = this.calculateStride();
    }

    private calculateStride(): number {
        return Object.values(this.layout).reduce((total, field) => {
            const baseSize = TypeSizes[field.type];
            const count = field.buffer.byteLength / baseSize || 1;
            field.count = count;
            return total + (baseSize * count);
        }, 0);
    }

    // Helper to get offset of a specific field
    public getOffset(fieldName: string): number {
        let offset = 0;
        for (const [key, field] of Object.entries(this.layout)) {
            if (key === fieldName) break;
            offset += TypeSizes[field.type] * (field.count || 1);
        }
        return offset;
    }
}

export { StructuredBuffer };