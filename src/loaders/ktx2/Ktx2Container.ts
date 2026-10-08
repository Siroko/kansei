/**
 * The KTX2 container: header, level index and the Data Format Descriptor's colour fields
 * (KTX 2.0 spec §3-§4, Khronos Data Format spec §5). The Basis payload itself is decoded by the
 * backend in `Basis.ts`; this reads what the engine needs to decide how to upload it.
 * Rust: `loaders/ktx2/container.rs`.
 */

const IDENTIFIER = [0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a];
const HEADER_BYTES = 80;
const LEVEL_INDEX_ENTRY_BYTES = 24;
/** `KHR_DF_TRANSFER_SRGB`. */
const TRANSFER_SRGB = 2;

/** Why a KTX2 file could not be loaded (Rust's `Ktx2Error`). */
export type Ktx2ErrorKind =
    | 'not-ktx2'
    | 'truncated'
    /** A valid file this loader does not handle (cubemaps, 3D, video, non-Basis payloads). */
    | 'unsupported'
    /** The Basis transcoder rejected the payload. */
    | 'transcoder';

export class Ktx2Error extends Error {
    constructor(public readonly kind: Ktx2ErrorKind, detail: string = '') {
        super(Ktx2Error.describe(kind, detail));
        this.name = 'Ktx2Error';
    }

    private static describe(kind: Ktx2ErrorKind, detail: string): string {
        switch (kind) {
            case 'not-ktx2': return 'not a KTX2 file';
            case 'truncated': return `KTX2 file truncated in its ${detail}`;
            case 'unsupported': return `unsupported KTX2 file: ${detail}`;
            case 'transcoder': return `Basis transcode failed: ${detail}`;
        }
    }
}

/** Whether `bytes` start with the KTX2 identifier. */
export function isKtx2(bytes: Uint8Array): boolean {
    return bytes.length >= IDENTIFIER.length && IDENTIFIER.every((b, i) => bytes[i] === b);
}

/**
 * KTX2 level supercompression (`supercompressionScheme`): `basis-lz` holds Basis Universal's
 * ETC1S codebooks + slices; any other value reads as its number.
 */
export type Supercompression = 'none' | 'basis-lz' | 'zstd' | 'zlib' | number;

function supercompressionFromRaw(v: number): Supercompression {
    switch (v) {
        case 0: return 'none';
        case 1: return 'basis-lz';
        case 2: return 'zstd';
        case 3: return 'zlib';
        default: return v;
    }
}

/** What a KTX2 file declares about itself, read without decoding any level. */
export interface Ktx2Header {
    /**
     * `VK_FORMAT_UNDEFINED` (0) for ETC1S and UASTC LDR; UASTC HDR 4x4 files carry
     * `VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK`.
     */
    vkFormat: number;
    width: number;
    height: number;
    /** 0 for 1D/2D textures. */
    depth: number;
    /** 0 for a texture that is not an array. */
    layers: number;
    /** 6 for a cubemap, otherwise 1. */
    faces: number;
    /** Mip levels stored in the file (a zero `levelCount`, "generate at load", reads as 1). */
    levels: number;
    supercompression: Supercompression;
    /** The DFD's colour model (`KHR_DF_MODEL_ETC1S` 163, `KHR_DF_MODEL_UASTC` 166, ...). */
    colorModel: number;
    /** Whether the DFD's transfer function is sRGB (colour data) rather than linear. */
    srgb: boolean;
    /** Stored bytes of each level (after supercompression), level 0 first. */
    levelBytes: number[];
}

/** Read a KTX2 file's header, level index and DFD colour fields; throws a `Ktx2Error`. */
export function parseKtx2Header(bytes: Uint8Array): Ktx2Header {
    if (!isKtx2(bytes)) throw new Ktx2Error('not-ktx2');
    if (bytes.length < HEADER_BYTES) throw new Ktx2Error('truncated', 'header');
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const u32At = (at: number) => view.getUint32(at, true);
    // offsets and lengths are u64; files past 2^53 bytes do not fit an ArrayBuffer anyway
    const u64At = (at: number) => Number(view.getBigUint64(at, true));

    const levels = Math.max(1, u32At(40));
    const indexEnd = HEADER_BYTES + levels * LEVEL_INDEX_ENTRY_BYTES;
    if (bytes.length < indexEnd) throw new Ktx2Error('truncated', 'level index');
    const levelBytes: number[] = [];
    for (let level = 0; level < levels; level++) {
        const at = HEADER_BYTES + level * LEVEL_INDEX_ENTRY_BYTES;
        const offset = u64At(at);
        const length = u64At(at + 8);
        if (offset + length > bytes.length) throw new Ktx2Error('truncated', 'level data');
        levelBytes.push(length);
    }

    // DFD: dfdTotalSize, then the basic descriptor block (colour model, primaries,
    // transfer function, flags at bytes 8..12 of the block)
    const dfdOffset = u32At(48);
    const dfdLength = u32At(52);
    if (dfdLength < 4 + 12 || dfdOffset + dfdLength > bytes.length) {
        throw new Ktx2Error('truncated', 'data format descriptor');
    }
    const block = dfdOffset + 4;
    return {
        vkFormat: u32At(12),
        width: u32At(20),
        height: Math.max(1, u32At(24)),
        depth: u32At(28),
        layers: u32At(32),
        faces: Math.max(1, u32At(36)),
        levels,
        supercompression: supercompressionFromRaw(u32At(44)),
        colorModel: bytes[block + 8],
        srgb: bytes[block + 10] === TRANSFER_SRGB,
        levelBytes,
    };
}
