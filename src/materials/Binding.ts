/**
 * The resource half of a bind group layout entry: exactly one of these keys is set. A
 * `BindGroupDescriptor.layout` sets it explicitly (mirroring the Rust engine's `Binding`
 * constructors in `materials/binding.rs`); otherwise the bound value describes itself
 * (`IBindable.getBindingLayout`).
 */
export type BindingLayout =
    | { buffer: GPUBufferBindingLayout }
    | { sampler: GPUSamplerBindingLayout }
    | { texture: GPUTextureBindingLayout }
    | { storageTexture: Partial<GPUStorageTextureBindingLayout> }
    | { externalTexture: GPUExternalTextureBindingLayout };

/**
 * Layouts for `BindGroupDescriptor.layout`, named after the Rust engine's `Binding` constructors.
 * A texture or storage texture layout may leave out its view dimension and format: they are
 * filled from the bound `Texture`.
 */
export const BindingLayouts = {
    uniform: (): BindingLayout => ({ buffer: { type: 'uniform' } }),
    storage: (readOnly: boolean = false): BindingLayout =>
        ({ buffer: { type: readOnly ? 'read-only-storage' : 'storage' } }),

    /** A sampled texture of any view dimension and sample type. */
    texture: (viewDimension?: GPUTextureViewDimension, sampleType: GPUTextureSampleType = 'float'): BindingLayout =>
        ({ texture: viewDimension ? { sampleType, viewDimension } : { sampleType } }),
    texture2d: (sampleType: GPUTextureSampleType = 'float'): BindingLayout => BindingLayouts.texture('2d', sampleType),
    /** A `texture_2d_array`: layers of equal size, such as terrain materials. */
    texture2dArray: (sampleType: GPUTextureSampleType = 'float'): BindingLayout => BindingLayouts.texture('2d-array', sampleType),
    texture3d: (sampleType: GPUTextureSampleType = 'float'): BindingLayout => BindingLayouts.texture('3d', sampleType),
    textureCube: (sampleType: GPUTextureSampleType = 'float'): BindingLayout => BindingLayouts.texture('cube', sampleType),
    textureCubeArray: (sampleType: GPUTextureSampleType = 'float'): BindingLayout => BindingLayouts.texture('cube-array', sampleType),
    /** A `texture_depth_*`, read with a comparison sampler or `textureLoad`. */
    textureDepth: (viewDimension: GPUTextureViewDimension = '2d'): BindingLayout => BindingLayouts.texture(viewDimension, 'depth'),

    /** A storage texture; format and view dimension default to the bound texture's. */
    storageTexture: (
        access: GPUStorageTextureAccess = 'write-only',
        format?: GPUTextureFormat,
        viewDimension?: GPUTextureViewDimension,
    ): BindingLayout => {
        const storageTexture: Partial<GPUStorageTextureBindingLayout> = { access };
        if (format) storageTexture.format = format;
        if (viewDimension) storageTexture.viewDimension = viewDimension;
        return { storageTexture };
    },
    storageTexture2d: (format: GPUTextureFormat, access: GPUStorageTextureAccess = 'write-only'): BindingLayout =>
        BindingLayouts.storageTexture(access, format, '2d'),
    storageTexture3d: (format: GPUTextureFormat, access: GPUStorageTextureAccess = 'write-only'): BindingLayout =>
        BindingLayouts.storageTexture(access, format, '3d'),

    sampler: (): BindingLayout => ({ sampler: { type: 'filtering' } }),
    nonFilteringSampler: (): BindingLayout => ({ sampler: { type: 'non-filtering' } }),
    comparisonSampler: (): BindingLayout => ({ sampler: { type: 'comparison' } }),
    externalTexture: (): BindingLayout => ({ externalTexture: {} }),
};

/**
 * The layout of an `IBindable` that only names its `type` (buffers, and bindables written
 * before descriptors existed).
 */
export function bindingLayoutFromType(type: string | undefined): BindingLayout | undefined {
    switch (type) {
        case 'storage':
        case 'read-only-storage':
        case 'uniform':
            return { buffer: { type } };
        case 'sampler':
            return BindingLayouts.sampler();
        case 'texture':
            return BindingLayouts.texture();
        case 'storage-texture':
            return BindingLayouts.storageTexture('write-only', 'rgba8unorm');
        case 'external-texture':
            return BindingLayouts.externalTexture();
        default:
            return undefined;
    }
}
