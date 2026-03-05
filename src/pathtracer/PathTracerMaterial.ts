export class PathTracerMaterial {
    public albedo: [number, number, number] = [1, 1, 1];
    public roughness: number = 1.0;
    public metallic: number = 0.0;
    public ior: number = 1.0;
    public maxBounces: number = 2;
    public absorptionColor: [number, number, number] = [1, 1, 1];
    public absorptionDensity: number = 0.0;
    public emissive: [number, number, number] = [0, 0, 0];
    public emissiveIntensity: number = 0.0;
    public refractive: boolean = false;

    /** Byte size of the packed GPU struct (std140-aligned). */
    static readonly GPU_STRIDE = 64;

    /**
     * Pack this material into a Float32Array at the given float offset.
     * GPU layout (64 bytes = 16 floats):
     *   [0-2]  albedo.rgb        [3]  roughness
     *   [4]    metallic          [5]  ior           [6] maxBounces(f32)  [7] flags
     *   [8-10] absorptionColor   [11] absorptionDensity
     *   [12-14] emissive         [15] emissiveIntensity
     */
    public packInto(target: Float32Array, offset: number): void {
        target[offset + 0]  = this.albedo[0];
        target[offset + 1]  = this.albedo[1];
        target[offset + 2]  = this.albedo[2];
        target[offset + 3]  = this.roughness;
        target[offset + 4]  = this.metallic;
        target[offset + 5]  = this.ior;
        target[offset + 6]  = this.maxBounces;
        target[offset + 7]  = (this.refractive ? 1 : 0);
        target[offset + 8]  = this.absorptionColor[0];
        target[offset + 9]  = this.absorptionColor[1];
        target[offset + 10] = this.absorptionColor[2];
        target[offset + 11] = this.absorptionDensity;
        target[offset + 12] = this.emissive[0];
        target[offset + 13] = this.emissive[1];
        target[offset + 14] = this.emissive[2];
        target[offset + 15] = this.emissiveIntensity;
    }
}
