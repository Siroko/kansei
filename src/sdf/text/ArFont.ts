/**
 * Parser for the Artery Font Format (`.arfont`) MTSDF atlases, in plain TS: header, font
 * variants (metrics, glyphs, kerning) and the embedded atlas image, PNG-decoded to RGBA8. The
 * TS side of the Rust engine's `sdf/arfont.rs`; it replaces the artery-font wasm decoder.
 *
 * Layouts from the upstream `artery-font-format` headers (all little-endian, every block
 * padded to 4 bytes, strings written as `length` bytes plus a NUL when `length > 0`):
 *
 * ```text
 * header (112 B)  tag "ARTERY/FONT\0"[16], magic, version, flags, realType, reserved[4],
 *                 metadataFormat, metadataLength, variantCount, variantsLength,
 *                 imageCount, imagesLength, appendixCount, appendicesLength, reserved[8]
 * metadata        metadataLength
 * variant         flags, weight, codepointType, imageType, fallbackVariant, fallbackGlyph,
 *                 reserved[6], metrics REAL[32], nameLength, metadataLength, glyphCount,
 *                 kernPairCount, name, metadata, glyphs (48 B), kern pairs (16 B)
 * image           flags, encoding, width, height, channels, pixelFormat, imageType,
 *                 rowLength, orientation, childImages, textureFlags, reserved[3],
 *                 metadataLength, dataLength, metadata, data
 * appendix        metadataLength, dataLength, metadata, data
 * ```
 */

export interface FontImage {
    width: number;
    height: number;
    /**
     * 8 bits per channel, `channels` per pixel, rows from the atlas's bottom up: glyphs'
     * `image_bounds` (y up) index it directly, as with the artery-font wasm decoder this parser
     * replaced.
     */
    data: Uint8Array;
    channels: number;
    child_images: number;
    flags: number;
    image_type: string;
    metadata: string;
    pixel_format: string;
    texture_flags: number;
}

export interface FontGlyphAdvance {
    horizontal: number;
    vertical: number;
}

export interface FontGlyphBounds {
    left: number;
    bottom: number;
    right: number;
    top: number;
}

export interface FontGlyph {
    codepoint: number;
    advance: FontGlyphAdvance;
    image: number;
    /** In atlas pixels, y counted up from the atlas's bottom. */
    image_bounds: FontGlyphBounds;
    /** In em units around the glyph's origin. */
    plane_bounds: FontGlyphBounds;
}

export interface FontKernPair {
    codepoint1: number;
    codepoint2: number;
    advance: FontGlyphAdvance;
}

export interface FontMetrics {
    font_size: number;
    distance_range: number;
    em_size: number;
    ascender: number;
    descender: number;
    line_height: number;
    underline_y: number;
    underline_thickness: number;
}

export interface FontVariant {
    name: string;
    codepoint_type: string;
    fallback_glyph: number;
    fallback_variant: number;
    flags: number;
    glyphs: FontGlyph[];
    image_type: string;
    kern_pairs: FontKernPair[];
    metadata: string;
    metrics: FontMetrics;
    weight: number;
}

export interface FontAppendix {
    metadata: string;
    data: Uint8Array;
}

/** A parsed `.arfont`: its variants (glyph metrics) and decoded atlas images. */
export interface ArFont {
    metadata_format: string;
    metadata: string;
    variants: FontVariant[];
    images: FontImage[];
    appendices: FontAppendix[];
}

const HEADER_BYTES = 112;
const MAGIC = 0x4d276a5c;
const REAL_TYPE_F32 = 0x14;
const ENCODING_RAW = 1;
const ENCODING_PNG = 8;
const ORIENTATION_TOP_DOWN = 1;

const METADATA_FORMATS: Record<number, string> = { 0: 'None', 1: 'PlainText', 2: 'Json' };
const CODEPOINT_TYPES: Record<number, string> = { 0: 'Unspecified', 1: 'Unicode', 2: 'Indexed', 14: 'Iconographic' };
const IMAGE_TYPES: Record<number, string> = {
    0: 'None', 1: 'SrgbImage', 2: 'LinearMask', 3: 'MaskedSrgbImage', 4: 'Sdf', 5: 'Psdf', 6: 'Msdf', 7: 'Mtsdf', 8: 'MixedContent',
};
const PIXEL_FORMATS: Record<number, string> = { 0: 'Unknown', 1: 'Boolean1', 8: 'Unsigned8', 32: 'Float32' };

/** A `.arfont` that does not parse: what was wrong with it. */
export class ArFontError extends Error {
    constructor(message: string) {
        super(`.arfont: ${message}`);
        this.name = 'ArFontError';
    }
}

/** Parse a `.arfont` file's bytes; the atlas images come back decoded. */
export async function parseArFont(bytes: Uint8Array): Promise<ArFont> {
    const reader = new Reader(bytes);
    if (bytes.length < HEADER_BYTES) throw new ArFontError('too short for a header');
    if (new TextDecoder().decode(bytes.subarray(0, 11)) !== 'ARTERY/FONT') throw new ArFontError('not an Artery Font file');
    if (reader.u32(16) !== MAGIC) throw new ArFontError('bad magic number');
    const version = reader.u32(20);
    if (version !== 1) throw new ArFontError(`unsupported version ${version}`);
    const realType = reader.u32(28);
    if (realType !== REAL_TYPE_F32) throw new ArFontError(`unsupported real type 0x${realType.toString(16)} (only f32)`);
    const metadataFormat = reader.u32(48);
    const metadataLength = reader.u32(52);
    const variantCount = reader.u32(56);
    const imageCount = reader.u32(64);
    const appendixCount = reader.u32(72);

    reader.at = HEADER_BYTES;
    const metadata = reader.string(metadataLength);
    const variants: FontVariant[] = [];
    for (let i = 0; i < variantCount; i++) variants.push(readVariant(reader));
    const images: FontImage[] = [];
    for (let i = 0; i < imageCount; i++) images.push(await readImage(reader));
    const appendices: FontAppendix[] = [];
    for (let i = 0; i < appendixCount; i++) {
        const [metaLength, dataLength] = [reader.next(), reader.next()];
        appendices.push({ metadata: reader.string(metaLength), data: reader.bytes(dataLength).slice() });
    }
    return { metadata_format: METADATA_FORMATS[metadataFormat] ?? 'None', metadata, variants, images, appendices };
}

function readVariant(reader: Reader): FontVariant {
    const [flags, weight, codepointType, imageType, fallbackVariant, fallbackGlyph] = [0, 0, 0, 0, 0, 0].map(() => reader.next());
    reader.at += 6 * 4;
    const m = Array.from({ length: 32 }, () => reader.nextF32());
    const [nameLength, metadataLength, glyphCount, kernPairCount] = [0, 0, 0, 0].map(() => reader.next());
    const name = reader.string(nameLength);
    const metadata = reader.string(metadataLength);
    const glyphs: FontGlyph[] = [];
    for (let i = 0; i < glyphCount; i++) {
        const codepoint = reader.next();
        const image = reader.next();
        const [pl, pb, pr, pt, il, ib, ir, it, h, v] = Array.from({ length: 10 }, () => reader.nextF32());
        glyphs.push({
            codepoint,
            image,
            plane_bounds: { left: pl, bottom: pb, right: pr, top: pt },
            image_bounds: { left: il, bottom: ib, right: ir, top: it },
            advance: { horizontal: h, vertical: v },
        });
    }
    const kern_pairs: FontKernPair[] = [];
    for (let i = 0; i < kernPairCount; i++) {
        const [codepoint1, codepoint2] = [reader.next(), reader.next()];
        kern_pairs.push({ codepoint1, codepoint2, advance: { horizontal: reader.nextF32(), vertical: reader.nextF32() } });
    }
    return {
        flags,
        weight,
        codepoint_type: CODEPOINT_TYPES[codepointType] ?? 'Unspecified',
        image_type: IMAGE_TYPES[imageType] ?? 'None',
        fallback_variant: fallbackVariant,
        fallback_glyph: fallbackGlyph,
        metrics: {
            font_size: m[0], distance_range: m[1], em_size: m[2], ascender: m[3], descender: m[4],
            line_height: m[5], underline_y: m[6], underline_thickness: m[7],
        },
        name,
        metadata,
        glyphs,
        kern_pairs,
    };
}

async function readImage(reader: Reader): Promise<FontImage> {
    const [flags, encoding, width, height, channels, pixelFormat, imageType] = [0, 0, 0, 0, 0, 0, 0].map(() => reader.next());
    reader.at += 4; // rowLength
    const orientation = reader.next();
    const [childImages, textureFlags] = [reader.next(), reader.next()];
    reader.at += 3 * 4;
    const [metadataLength, dataLength] = [reader.next(), reader.next()];
    const metadata = reader.string(metadataLength);
    const encoded = reader.bytes(dataLength);
    let data: Uint8Array;
    if (encoding === ENCODING_PNG) {
        const png = await decodePng(encoded);
        if (png.width !== width || png.height !== height || png.channels !== channels) {
            throw new ArFontError(`the embedded PNG is ${png.width}x${png.height}x${png.channels}, the header says ${width}x${height}x${channels}`);
        }
        // PNG rows run top down
        data = flipRows(png.pixels, width * channels, height);
    } else if (encoding === ENCODING_RAW) {
        data = orientation === ORIENTATION_TOP_DOWN ? flipRows(encoded, width * channels, height) : encoded.slice();
    } else {
        throw new ArFontError(`unsupported image encoding ${encoding} (PNG or raw only)`);
    }
    return {
        flags,
        width,
        height,
        channels,
        pixel_format: PIXEL_FORMATS[pixelFormat] ?? 'Unknown',
        image_type: IMAGE_TYPES[imageType] ?? 'None',
        child_images: childImages,
        texture_flags: textureFlags,
        metadata,
        data,
    };
}

/** `pixels`' rows in the opposite order (a copy). */
function flipRows(pixels: Uint8Array, stride: number, height: number): Uint8Array {
    const flipped = new Uint8Array(stride * height);
    for (let y = 0; y < height; y++) flipped.set(pixels.subarray(y * stride, (y + 1) * stride), (height - 1 - y) * stride);
    return flipped;
}

/** Little-endian reads, with a cursor over the blocks. */
class Reader {
    public at = 0;
    private readonly view: DataView;

    constructor(private readonly buffer: Uint8Array) {
        this.view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    }

    public u32(offset: number): number {
        if (offset + 4 > this.buffer.length) throw new ArFontError('truncated');
        return this.view.getUint32(offset, true);
    }

    public next(): number {
        const value = this.u32(this.at);
        this.at += 4;
        return value;
    }

    public nextF32(): number {
        if (this.at + 4 > this.buffer.length) throw new ArFontError('truncated');
        const value = this.view.getFloat32(this.at, true);
        this.at += 4;
        return value;
    }

    /** `length` bytes, then the cursor past them padded to 4. */
    public bytes(length: number): Uint8Array {
        if (this.at + length > this.buffer.length) throw new ArFontError('truncated');
        const bytes = this.buffer.subarray(this.at, this.at + length);
        this.at += (length + 3) & ~3;
        return bytes;
    }

    /** A string block: nothing for length 0, else `length` bytes and a NUL, padded to 4. */
    public string(length: number): string {
        if (length === 0) return '';
        const text = new TextDecoder().decode(this.buffer.subarray(this.at, this.at + length));
        this.bytes(length + 1);
        return text;
    }
}

/** Decoded PNG pixels: 8 bits a channel, rows top to bottom. */
interface PngImage {
    width: number;
    height: number;
    channels: number;
    pixels: Uint8Array;
}

/**
 * Decode a non-interlaced PNG of 8-bit greyscale, grey+alpha, RGB or RGBA (what msdf-atlas-gen
 * writes): its IDAT stream inflated by the browser's `DecompressionStream`, then unfiltered.
 * The values are the file's, with no colour-space conversion or premultiplied alpha: an atlas's
 * channels are distances, not colour.
 */
async function decodePng(png: Uint8Array): Promise<PngImage> {
    const signature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
    if (png.length < 8 || signature.some((b, i) => png[i] !== b)) throw new ArFontError('the image is not a PNG');
    const view = new DataView(png.buffer, png.byteOffset, png.byteLength);
    let width = 0, height = 0, channels = 0;
    const idat: Uint8Array[] = [];
    for (let at = 8; at + 8 <= png.length;) {
        const length = view.getUint32(at);
        const type = String.fromCharCode(...png.subarray(at + 4, at + 8));
        const body = png.subarray(at + 8, at + 8 + length);
        if (type === 'IHDR') {
            width = view.getUint32(at + 8);
            height = view.getUint32(at + 12);
            const [depth, colour, , , interlace] = body.subarray(8, 13);
            channels = ({ 0: 1, 2: 3, 4: 2, 6: 4 } as Record<number, number>)[colour] ?? 0;
            if (depth !== 8 || channels === 0 || interlace !== 0) {
                throw new ArFontError(`unsupported PNG (bit depth ${depth}, colour type ${colour}, interlace ${interlace})`);
            }
        } else if (type === 'IDAT') {
            idat.push(body);
        } else if (type === 'IEND') {
            break;
        }
        at += 12 + length;
    }
    if (channels === 0 || idat.length === 0) throw new ArFontError('the PNG has no image data');
    const inflated = await inflate(idat);
    const stride = width * channels;
    if (inflated.length < (stride + 1) * height) throw new ArFontError('the PNG image data is truncated');
    const pixels = new Uint8Array(stride * height);
    for (let y = 0; y < height; y++) {
        const filter = inflated[y * (stride + 1)];
        const src = inflated.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
        const row = y * stride;
        const up = row - stride;
        for (let x = 0; x < stride; x++) {
            const a = x >= channels ? pixels[row + x - channels] : 0;
            const b = y > 0 ? pixels[up + x] : 0;
            const c = x >= channels && y > 0 ? pixels[up + x - channels] : 0;
            let predicted: number;
            switch (filter) {
                case 0: predicted = 0; break;
                case 1: predicted = a; break;
                case 2: predicted = b; break;
                case 3: predicted = (a + b) >> 1; break;
                case 4: {
                    const p = a + b - c;
                    const [pa, pb, pc] = [Math.abs(p - a), Math.abs(p - b), Math.abs(p - c)];
                    predicted = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
                    break;
                }
                default: throw new ArFontError(`bad PNG filter ${filter}`);
            }
            pixels[row + x] = (src[x] + predicted) & 0xff;
        }
    }
    return { width, height, channels, pixels };
}

/** Inflate a zlib stream (the concatenated IDAT chunks). */
async function inflate(chunks: Uint8Array[]): Promise<Uint8Array> {
    const stream = new Blob(chunks as BlobPart[]).stream().pipeThrough(new DecompressionStream('deflate'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
}
