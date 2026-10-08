/**
 * The Basis Universal transcoder backend: the only file that knows which transcoder decodes the
 * payload. It is Binomial's official WebAssembly build (`basis/`, tag v2_50: the tag whose
 * output the Rust engine's fixture tests check its own pure-Rust transcoder against), loaded
 * the first time a KTX2 file is opened. Swapping in another backend means reimplementing
 * `BasisFile` here. Rust: `loaders/ktx2/basis.rs`.
 */
import type { BasisCodec, GpuTarget } from "./Ktx2Select";
import { targetName } from "./Ktx2Select";
import { Ktx2Error } from "./Ktx2Container";

/** `basist::transcoder_texture_format` values of the targets the engine requests. */
const TranscoderFormat = {
    ETC1_RGB: 0,
    ETC2_RGBA: 1,
    BC1_RGB: 2,
    BC4_R: 4,
    BC5_RG: 5,
    BC7_RGBA: 6,
    ASTC_4X4_RGBA: 10,
    RGBA32: 13,
    EAC_R11: 20,
    EAC_RG11: 21,
    BC6H: 22,
    ASTC_HDR_4X4_RGBA: 23,
    RGBA_HALF: 25,
} as const;

function transcoderFormat(target: GpuTarget): number {
    switch (target) {
        case 'bc1': return TranscoderFormat.BC1_RGB;
        case 'bc4': return TranscoderFormat.BC4_R;
        case 'bc5': return TranscoderFormat.BC5_RG;
        case 'bc6h': return TranscoderFormat.BC6H;
        case 'bc7': return TranscoderFormat.BC7_RGBA;
        case 'astc-4x4': return TranscoderFormat.ASTC_4X4_RGBA;
        case 'astc-hdr-4x4': return TranscoderFormat.ASTC_HDR_4X4_RGBA;
        case 'etc2-rgb': return TranscoderFormat.ETC1_RGB;
        case 'etc2-rgba': return TranscoderFormat.ETC2_RGBA;
        case 'eac-r11': return TranscoderFormat.EAC_R11;
        case 'eac-rg11': return TranscoderFormat.EAC_RG11;
        case 'rgba8': case 'rg8': case 'r8': return TranscoderFormat.RGBA32;
        case 'rgba16float': return TranscoderFormat.RGBA_HALF;
    }
}

// The Emscripten module (embind classes are untyped)
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type BasisModule = any;

let loading: Promise<void> | undefined;
let basisModule: BasisModule | undefined;

/**
 * Load and initialize the transcoder (about 1 MB of WebAssembly, fetched once). `KTX2Loader`
 * awaits it before its first file; call it earlier to keep the fetch off the first load.
 */
export function initBasis(): Promise<void> {
    loading ??= (async () => {
        const { createBasisModule } = await import("./BasisModule");
        basisModule = await createBasisModule();
    })();
    return loading;
}

/**
 * An opened Basis Universal texture. Its transcoder state lives in WebAssembly memory: call
 * `close()` when done (`KTX2Loader` does).
 */
export class BasisFile {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    private constructor(private readonly module: BasisModule, private readonly file: any) {}

    /** Open a KTX2 file (after `initBasis()`); throws a `Ktx2Error` when the transcoder refuses it. */
    public static open(bytes: Uint8Array): BasisFile {
        const module = basisModule;
        if (!module) throw new Ktx2Error('transcoder', 'the Basis transcoder is not loaded (await initBasis() first)');
        const file = new module.KTX2File(bytes);
        if (!file.isValid() || !file.startTranscoding()) {
            file.close();
            file.delete();
            throw new Ktx2Error('transcoder', 'not a Basis Universal KTX2 payload');
        }
        return new BasisFile(module, file);
    }

    public codec(): BasisCodec {
        if (this.file.isETC1S()) return 'etc1s';
        if (this.file.isUASTC_LDR_4x4()) return 'uastc-ldr';
        if (this.file.isHDR4x4()) return 'uastc-hdr';
        return this.file.isHDR() ? 'other-hdr' : 'other-ldr';
    }

    /** A short name of the source codec ("ETC1S", "UASTC", ...). */
    public codecName(): string {
        switch (this.codec()) {
            case 'etc1s': return 'ETC1S';
            case 'uastc-ldr': return 'UASTC';
            case 'uastc-hdr': return 'UASTC HDR';
            default: return this.file.isHDR6x6() ? 'HDR 6x6' : this.file.isXUASTC_LDR() ? 'XUASTC' : `basis_tex_format ${this.file.getBasisTexFormat()}`;
        }
    }

    public hasAlpha(): boolean {
        return !!this.file.getHasAlpha();
    }

    public isVideo(): boolean {
        return !!this.file.isVideo();
    }

    /**
     * Whether the transcoder can produce `target` from this file (a direct transcode, or the
     * uncompressed decode the R8/RG8/RGBA8/RGBA16F fallbacks are cut from).
     */
    public canTranscode(target: GpuTarget): boolean {
        return !!this.module.isFormatSupported(transcoderFormat(target), this.file.getBasisTexFormat());
    }

    /** Mip `level` of array layer `layer` (face 0) as `target`'s texels or blocks, tightly packed. */
    public transcode(level: number, layer: number, target: GpuTarget): Uint8Array {
        const format = transcoderFormat(target);
        const size = this.file.getImageTranscodedSizeInBytes(level, layer, 0, format);
        const raw = new Uint8Array(size);
        // no decode flags, channels -1 -1: the transcoder's defaults, as the Rust backend uses
        if (!size || !this.file.transcodeImageWithFlags(raw, level, layer, 0, format, 0, -1, -1)) {
            throw new Ktx2Error('transcoder', `level ${level} layer ${layer} to ${targetName(target)}`);
        }
        switch (target) {
            // cut from RGBA: R in red; the Basis two-channel layout has G in alpha
            case 'r8': {
                const out = new Uint8Array(raw.length / 4);
                for (let i = 0; i < out.length; i++) out[i] = raw[i * 4];
                return out;
            }
            case 'rg8': {
                const out = new Uint8Array(raw.length / 2);
                for (let i = 0; i < out.length / 2; i++) {
                    out[i * 2] = raw[i * 4];
                    out[i * 2 + 1] = raw[i * 4 + 3];
                }
                return out;
            }
            default:
                return raw;
        }
    }

    /** Free the transcoder's copy of the file. */
    public close() {
        this.file.close();
        this.file.delete();
    }
}
