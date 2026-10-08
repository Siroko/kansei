/**
 * KTX2 textures with Basis Universal supercompression (ETC1S, UASTC, UASTC HDR), transcoded at
 * load to the best block-compressed format the device supports: BC7/BC1/BC4/BC5/BC6H, ASTC 4x4,
 * ETC2/EAC, or uncompressed texels when none fits. See `docs/ktx2.md` for choosing codecs and
 * the encoding tool. Rust: `loaders::ktx2`.
 *
 * ```ts
 * const tex = await KTX2Loader.transcode('Car/BaseColor', bytes, KTX2Loader.color(), renderer.compressionSupport);
 * console.info(tex.summary());
 * const material = new Material(wgsl, { bindings: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, value: tex.toTexture() }, ...] });
 * ```
 *
 * `Renderer` requests whichever of `texture-compression-bc`, `-astc` and `-etc2` the adapter
 * offers. The transcoder (`ktx2/Basis.ts`) loads with the first file, or earlier with
 * `KTX2Loader.init()`.
 */
import { Texture } from "../buffers/Texture";
import type { CompressionSupport } from "../renderers/Renderer";
import { BasisFile, initBasis } from "./ktx2/Basis";
import { Ktx2Error, isKtx2, parseKtx2Header } from "./ktx2/Ktx2Container";
import type { Ktx2Header } from "./ktx2/Ktx2Container";
import {
    gpuTargetFormat, selectTarget, targetChainBytes, targetName,
} from "./ktx2/Ktx2Select";
import type { BasisCodec, Channels, GpuTarget } from "./ktx2/Ktx2Select";

/** How the texture is used, which the file cannot always say for itself. */
export interface Ktx2Options {
    /**
     * Sample as sRGB colour (`true`: base colour, emissive) or linear data (`false`: normals,
     * roughness/metalness/occlusion). Unset follows the file's transfer function.
     */
    srgb?: boolean;
    /**
     * The channels to keep; unset means RGB, or RGBA when the file has alpha. `r` and `rg` let
     * BC4/BC5/EAC hold one- and two-channel data at half the memory.
     */
    channels?: Channels;
}

/** What a KTX2 file holds, read without transcoding. */
export interface Ktx2Info {
    header: Ktx2Header;
    codec: BasisCodec;
    /** "ETC1S", "UASTC", "UASTC HDR", ... */
    codecName: string;
    hasAlpha: boolean;
}

/** A transcoded texture, ready to upload. */
export class TranscodedTexture {
    constructor(
        public readonly label: string,
        /** The file's source codec ("ETC1S", "UASTC", ...). */
        public readonly codec: string,
        public readonly target: GpuTarget,
        public readonly format: GPUTextureFormat,
        public readonly width: number,
        public readonly height: number,
        /** The layer count of a 2D array texture (bound as `texture_2d_array`); unset for a 2D texture. */
        public readonly layers: number | undefined,
        /**
         * Each mip level's blocks (or texels), tightly packed, level 0 first; within a level, each
         * array layer in turn.
         */
        public readonly levels: Uint8Array[],
        /** Size of the KTX2 file. */
        public readonly fileBytes: number,
    ) {}

    /** GPU memory the texture takes, all mips. */
    public gpuBytes(): number {
        return this.levels.reduce((sum, level) => sum + level.byteLength, 0);
    }

    /** GPU memory the same mips would take uncompressed (RGBA8, or RGBA16F for HDR). */
    public uncompressedBytes(): number {
        const hdr = this.target === 'bc6h' || this.target === 'astc-hdr-4x4' || this.target === 'rgba16float';
        return targetChainBytes(hdr ? 'rgba16float' : 'rgba8', this.width, this.height, this.levels.length) * (this.layers ?? 1);
    }

    /** One line for logs and HUDs. */
    public summary(): string {
        const mb = (b: number) => (b / (1024 * 1024)).toFixed(2);
        return `${this.label}: ${this.codec} ${this.width}x${this.height}${this.layers !== undefined ? `x${this.layers} layers` : ''} `
            + `(${this.levels.length} mips, ${mb(this.fileBytes)} MB file) -> ${this.format}, `
            + `${mb(this.gpuBytes())} MB on the GPU (uncompressed ${mb(this.uncompressedBytes())} MB)`;
    }

    /** A `Texture` with every mip (and layer), uploaded when first bound. */
    public toTexture(): Texture {
        return this.layers !== undefined
            ? Texture.fromArrayLevels(this.label, this.format, this.width, this.height, this.layers, this.levels)
            : Texture.fromLevels(this.label, this.format, this.width, this.height, this.levels);
    }
}

/** Opens `bytes` with the transcoder, runs `body` and frees the transcoder's copy. */
async function withFile<T>(bytes: Uint8Array, body: (header: Ktx2Header, file: BasisFile) => T): Promise<T> {
    const header = parseKtx2Header(bytes);
    await initBasis();
    let file: BasisFile;
    try {
        file = BasisFile.open(bytes);
    } catch (e) {
        if (header.vkFormat !== 0) {
            throw new Ktx2Error('unsupported', `vkFormat ${header.vkFormat} without a Basis payload (${(e as Error).message})`);
        }
        throw e;
    }
    try {
        return body(header, file);
    } finally {
        file.close();
    }
}

function targetFor(header: Ktx2Header, file: BasisFile, options: Ktx2Options, support: CompressionSupport): GpuTarget {
    const channels = options.channels ?? (file.hasAlpha() ? 'rgba' : 'rgb');
    return selectTarget(file.codec(), channels, header.width, header.height, support, (t) => file.canTranscode(t));
}

function levelData(file: BasisFile, level: number, layers: number, target: GpuTarget): Uint8Array {
    const parts: Uint8Array[] = [];
    for (let layer = 0; layer < layers; layer++) parts.push(file.transcode(level, layer, target));
    if (parts.length === 1) return parts[0];
    const out = new Uint8Array(parts.reduce((sum, p) => sum + p.byteLength, 0));
    let offset = 0;
    for (const p of parts) {
        out.set(p, offset);
        offset += p.byteLength;
    }
    return out;
}

function checkLevel(header: Ktx2Header, file: BasisFile, level: number, target: GpuTarget) {
    if (!file.canTranscode(target)) {
        throw new Ktx2Error('unsupported', `${file.codecName()} cannot become ${targetName(target)}`);
    }
    if (level >= header.levels) throw new Ktx2Error('unsupported', `level ${level} of ${header.levels}`);
}

class KTX2Loader {
    /** Colour: sRGB, channels from the file. */
    public static color(): Ktx2Options {
        return { srgb: true };
    }

    /** Linear data (normal maps, ORM): channels from the file. */
    public static linear(): Ktx2Options {
        return { srgb: false };
    }

    /** Whether `bytes` start with the KTX2 identifier. */
    public static isKtx2(bytes: Uint8Array): boolean {
        return isKtx2(bytes);
    }

    /** Load the transcoder now rather than with the first file. */
    public static init(): Promise<void> {
        return initBasis();
    }

    /** Read a Basis KTX2 file's header and codec. */
    public static inspect(bytes: Uint8Array): Promise<Ktx2Info> {
        return withFile(bytes, (header, file) => ({
            header,
            codec: file.codec(),
            codecName: file.codecName(),
            hasAlpha: file.hasAlpha(),
        }));
    }

    /** The GPU target `bytes` would become on a device with `support`, without transcoding. */
    public static chooseTarget(bytes: Uint8Array, options: Ktx2Options, support: CompressionSupport): Promise<GpuTarget> {
        return withFile(bytes, (header, file) => targetFor(header, file, options, support));
    }

    /**
     * Every mip transcoded to `target`, whatever the device supports (for tools, tests and
     * forcing a fallback): per level, each array layer's data in turn. Compressed levels are
     * rounded up to whole blocks.
     */
    public static transcodeLevels(bytes: Uint8Array, target: GpuTarget): Promise<Uint8Array[]> {
        return withFile(bytes, (header, file) => {
            const levels: Uint8Array[] = [];
            for (let level = 0; level < header.levels; level++) {
                checkLevel(header, file, level, target);
                levels.push(levelData(file, level, Math.max(1, header.layers), target));
            }
            return levels;
        });
    }

    /**
     * One mip `level` transcoded to `target`, each array layer's data in turn (a small level
     * as `rgba8` gives a texture's mean colour cheaply, say).
     */
    public static transcodeLevel(bytes: Uint8Array, level: number, target: GpuTarget): Promise<Uint8Array> {
        return withFile(bytes, (header, file) => {
            checkLevel(header, file, level, target);
            return levelData(file, level, Math.max(1, header.layers), target);
        });
    }

    /** Transcode every mip of a 2D (or 2D array) Basis KTX2 texture for a device with `support`. */
    public static async transcode(label: string, bytes: Uint8Array, options: Ktx2Options, support: CompressionSupport): Promise<TranscodedTexture> {
        const header = parseKtx2Header(bytes);
        if (header.faces > 1 || header.depth > 1) {
            throw new Ktx2Error('unsupported', `${label}: ${header.faces} faces, depth ${header.depth} (2D textures and 2D arrays load so far)`);
        }
        return withFile(bytes, (header, file) => {
            if (file.isVideo()) throw new Ktx2Error('unsupported', `${label}: a Basis video`);
            const codec = file.codec();
            const hdr = codec === 'uastc-hdr' || codec === 'other-hdr';
            const srgb = !hdr && (options.srgb ?? header.srgb);
            if (options.srgb !== undefined && options.srgb !== header.srgb && !hdr) {
                console.warn(
                    `${label}: encoded as ${header.srgb ? 'sRGB' : 'linear'} but loaded as ${srgb ? 'sRGB' : 'linear'} `
                    + "(check the encoder's -srgb/-linear)",
                );
            }
            const target = targetFor(header, file, options, support);
            const layers = Math.max(1, header.layers);
            const levels: Uint8Array[] = [];
            for (let level = 0; level < header.levels; level++) levels.push(levelData(file, level, layers, target));
            return new TranscodedTexture(
                label,
                file.codecName(),
                target,
                gpuTargetFormat(target, srgb),
                header.width,
                header.height,
                header.layers > 0 ? header.layers : undefined,
                levels,
                bytes.byteLength,
            );
        });
    }

    /** Fetch `url` and transcode it (`transcode`), labelled with the URL. */
    public async load(url: string, options: Ktx2Options, support: CompressionSupport): Promise<TranscodedTexture> {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`KTX2Loader: ${url}: HTTP ${response.status}`);
        const bytes = new Uint8Array(await response.arrayBuffer());
        return KTX2Loader.transcode(url, bytes, options, support);
    }
}

export { KTX2Loader };
