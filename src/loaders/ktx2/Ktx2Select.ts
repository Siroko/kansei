/**
 * Which GPU format a Basis texture becomes on this device: the device's block-compression
 * features, the source codec, the texture's channels and its size decide it.
 * Rust: `loaders/ktx2/select.rs`.
 */
import type { CompressionSupport } from "../../renderers/Renderer";
import { textureFormatBlock } from "../../buffers/TextureFormats";

/** No compressed formats: everything decodes to uncompressed texels. */
export const NO_COMPRESSION: CompressionSupport = Object.freeze({ bc: false, astc: false, etc2: false });

/**
 * The channels a texture carries, which decide whether one- and two-channel formats fit.
 * - `r`: one channel (roughness, occlusion, a mask), read from `.r`.
 * - `rg`: two channels encoded the Basis way, R in colour and G in alpha (`basisu
 *   -separate_rg_to_color_alpha`, for XY normal maps); sampled as `.rg` whatever the GPU format.
 */
export type Channels = 'r' | 'rg' | 'rgb' | 'rgba';

/**
 * The Basis source codec, as far as target choice is concerned.
 * - `etc1s`: small, ETC1-quality; ETC2 is a lossless target.
 * - `uastc-ldr`: UASTC LDR 4x4, high quality; ASTC 4x4 is a lossless target.
 * - `uastc-hdr`: UASTC HDR 4x4: BC6H, ASTC HDR or half floats.
 * - `other-ldr`: another Basis codec (XUASTC, raw ASTC): whatever of the generic LDR targets it supports.
 * - `other-hdr`: another HDR codec.
 */
export type BasisCodec = 'etc1s' | 'uastc-ldr' | 'uastc-hdr' | 'other-ldr' | 'other-hdr';

/**
 * The GPU format a Basis texture is transcoded to. `etc2-rgb` is ETC2 RGB8 (Basis writes ETC1
 * blocks, a subset of it); `rgba8`, `rg8`, `r8` and `rgba16float` are the uncompressed fallbacks.
 */
export type GpuTarget =
    | 'bc1' | 'bc4' | 'bc5' | 'bc6h' | 'bc7'
    | 'astc-4x4' | 'astc-hdr-4x4'
    | 'etc2-rgb' | 'etc2-rgba' | 'eac-r11' | 'eac-rg11'
    | 'rgba8' | 'rg8' | 'r8' | 'rgba16float';

/** Whether the device can sample `target` (the uncompressed ones always; ASTC HDR never on WebGPU). */
export function supportsTarget(support: CompressionSupport, target: GpuTarget): boolean {
    switch (target) {
        case 'bc1': case 'bc4': case 'bc5': case 'bc6h': case 'bc7':
            return support.bc;
        case 'astc-4x4':
            return support.astc;
        case 'astc-hdr-4x4':
            return false;
        case 'etc2-rgb': case 'etc2-rgba': case 'eac-r11': case 'eac-rg11':
            return support.etc2;
        case 'rgba8': case 'rg8': case 'r8': case 'rgba16float':
            return true;
    }
}

/**
 * The texture format, with the sRGB variant where the format has one and `srgb` is set.
 * `astc-hdr-4x4` has no WebGPU format (WebGPU's ASTC is LDR only); it maps to the LDR one and is
 * never chosen, since no WebGPU device supports it.
 */
export function gpuTargetFormat(target: GpuTarget, srgb: boolean): GPUTextureFormat {
    switch (target) {
        case 'bc1': return srgb ? 'bc1-rgba-unorm-srgb' : 'bc1-rgba-unorm';
        case 'bc4': return 'bc4-r-unorm';
        case 'bc5': return 'bc5-rg-unorm';
        case 'bc6h': return 'bc6h-rgb-ufloat';
        case 'bc7': return srgb ? 'bc7-rgba-unorm-srgb' : 'bc7-rgba-unorm';
        case 'astc-4x4': return srgb ? 'astc-4x4-unorm-srgb' : 'astc-4x4-unorm';
        case 'astc-hdr-4x4': return 'astc-4x4-unorm';
        case 'etc2-rgb': return srgb ? 'etc2-rgb8unorm-srgb' : 'etc2-rgb8unorm';
        case 'etc2-rgba': return srgb ? 'etc2-rgba8unorm-srgb' : 'etc2-rgba8unorm';
        case 'eac-r11': return 'eac-r11unorm';
        case 'eac-rg11': return 'eac-rg11unorm';
        case 'rgba8': return srgb ? 'rgba8unorm-srgb' : 'rgba8unorm';
        case 'rg8': return 'rg8unorm';
        case 'r8': return 'r8unorm';
        case 'rgba16float': return 'rgba16float';
    }
}

export function isCompressedTarget(target: GpuTarget): boolean {
    return !(target === 'rgba8' || target === 'rg8' || target === 'r8' || target === 'rgba16float');
}

/**
 * Bytes one mip level of `width` x `height` texels occupies on the GPU (compressed formats
 * round up to whole 4x4 blocks).
 */
export function targetLevelBytes(target: GpuTarget, width: number, height: number): number {
    const block = textureFormatBlock(gpuTargetFormat(target, false));
    return Math.ceil(width / block.width) * Math.ceil(height / block.height) * block.bytes;
}

/** Bytes of a full chain of `levels` mips from `width` x `height`. */
export function targetChainBytes(target: GpuTarget, width: number, height: number, levels: number): number {
    let sum = 0;
    for (let l = 0; l < levels; l++) sum += targetLevelBytes(target, Math.max(1, width >> l), Math.max(1, height >> l));
    return sum;
}

/** Short name for logs and HUDs ("BC7", "ASTC 4x4", ...). */
export function targetName(target: GpuTarget): string {
    switch (target) {
        case 'bc1': return 'BC1';
        case 'bc4': return 'BC4';
        case 'bc5': return 'BC5';
        case 'bc6h': return 'BC6H';
        case 'bc7': return 'BC7';
        case 'astc-4x4': return 'ASTC 4x4';
        case 'astc-hdr-4x4': return 'ASTC 4x4 HDR';
        case 'etc2-rgb': return 'ETC2 RGB';
        case 'etc2-rgba': return 'ETC2 RGBA';
        case 'eac-r11': return 'EAC R11';
        case 'eac-rg11': return 'EAC RG11';
        case 'rgba8': return 'RGBA8';
        case 'rg8': return 'RG8';
        case 'r8': return 'R8';
        case 'rgba16float': return 'RGBA16F';
    }
}

/**
 * Targets in order of preference for a codec and its channels. The lossless targets lead (ETC2
 * holds ETC1S exactly, ASTC 4x4 holds UASTC exactly); among the rest, the smallest format that
 * keeps the channels. Every list ends in an uncompressed format, which is always available.
 */
export function targetPreferences(codec: BasisCodec, channels: Channels): readonly GpuTarget[] {
    if (codec === 'uastc-hdr' || codec === 'other-hdr') return ['bc6h', 'astc-hdr-4x4', 'rgba16float'];
    if (codec === 'etc1s') {
        switch (channels) {
            case 'r': return ['eac-r11', 'bc4', 'r8'];
            case 'rg': return ['eac-rg11', 'bc5', 'rg8'];
            case 'rgb': return ['etc2-rgb', 'bc1', 'astc-4x4', 'rgba8'];
            case 'rgba': return ['etc2-rgba', 'bc7', 'astc-4x4', 'rgba8'];
        }
    }
    switch (channels) {
        case 'r': return ['bc4', 'eac-r11', 'astc-4x4', 'r8'];
        // ASTC keeps UASTC's own RRRG layout for two channels, so it is not offered for rg
        case 'rg': return ['bc5', 'eac-rg11', 'rg8'];
        case 'rgb': return ['astc-4x4', 'bc7', 'etc2-rgb', 'rgba8'];
        case 'rgba': return ['astc-4x4', 'bc7', 'etc2-rgba', 'rgba8'];
    }
}

/**
 * The first preferred target this device supports, the file can produce (`transcodable`), and
 * the size allows: WebGPU needs a block-compressed texture's base size to be a whole number of
 * 4x4 blocks, so other sizes fall back to uncompressed texels.
 */
export function selectTarget(
    codec: BasisCodec,
    channels: Channels,
    width: number,
    height: number,
    support: CompressionSupport,
    transcodable: (target: GpuTarget) => boolean = () => true,
): GpuTarget {
    const blockAligned = width % 4 === 0 && height % 4 === 0;
    const prefs = targetPreferences(codec, channels);
    return prefs.find((t) => (!isCompressedTarget(t) || blockAligned) && supportsTarget(support, t) && transcodable(t))
        ?? prefs[prefs.length - 1];
}
