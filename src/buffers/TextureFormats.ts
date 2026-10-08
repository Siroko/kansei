/** A texel block: 1x1 for uncompressed formats, 4x4 (or the ASTC footprint) for compressed ones. */
export interface TextureFormatBlock {
    width: number;
    height: number;
    /** Bytes per block, as `queue.writeTexture` counts them for the format's copy aspect. */
    bytes: number;
}

/**
 * The block footprint and size of a texture format (the Rust engine reads them from wgpu's
 * `block_dimensions` and `block_copy_size`). Unknown formats count as 4-byte texels.
 */
export function textureFormatBlock(format: GPUTextureFormat): TextureFormatBlock {
    const astc = /^astc-(\d+)x(\d+)-/.exec(format);
    if (astc) return { width: Number(astc[1]), height: Number(astc[2]), bytes: 16 };
    if (format.startsWith('bc')) {
        const eightByte = format.startsWith('bc1-') || format.startsWith('bc4-');
        return { width: 4, height: 4, bytes: eightByte ? 8 : 16 };
    }
    if (format.startsWith('etc2-') || format.startsWith('eac-')) {
        const eightByte = format.startsWith('etc2-rgb8') || format.startsWith('eac-r11');
        return { width: 4, height: 4, bytes: eightByte ? 8 : 16 };
    }
    return { width: 1, height: 1, bytes: texelBytes(format) };
}

function texelBytes(format: GPUTextureFormat): number {
    switch (format) {
        case 'stencil8':
            return 1;
        case 'depth16unorm':
            return 2;
        case 'depth32float':
        case 'depth24plus':
        case 'depth24plus-stencil8':
        case 'depth32float-stencil8':
        case 'rgb10a2unorm':
        case 'rgb10a2uint':
        case 'rg11b10ufloat':
        case 'rgb9e5ufloat':
            return 4;
    }
    const match = /^(r|rg|rgba|bgra)(8|16|32)/.exec(format);
    if (!match) return 4;
    const channels = match[1] === 'bgra' ? 4 : match[1].length;
    return channels * Number(match[2]) / 8;
}

export function isDepthFormat(format: GPUTextureFormat): boolean {
    return format.startsWith('depth');
}

/**
 * The sample type a sampled texture of this format binds with by default: depth, uint and sint
 * formats theirs, 32-bit floats `unfilterable-float` unless the device has `float32-filterable`,
 * everything else `float`.
 */
export function textureFormatSampleType(format: GPUTextureFormat, gpuDevice: GPUDevice): GPUTextureSampleType {
    if (isDepthFormat(format)) return 'depth';
    if (format === 'stencil8' || /uint$/.test(format)) return 'uint';
    if (/sint$/.test(format)) return 'sint';
    if (/32float$/.test(format) && !gpuDevice.features.has('float32-filterable')) return 'unfilterable-float';
    return 'float';
}
