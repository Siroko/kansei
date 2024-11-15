import { Texture } from '../../buffers/Texture';
import init, { ArteryFont } from './loader/artery_font';

interface FontImage {
    width: number;
    height: number;
    data: number[];
    channels: number; // 4
    child_images: number; // 0
    flags: number; // 0
    image_type: string; // Mtsdf
    metadata: string; // ""
    pixel_format: string; // Unsigned8
    texture_flags: number; // 0
}

export interface FontInfo {
    images: FontImage[];
    appendices: [];
    variants: FontVariant[];
    metadata_format: string; // None, Json
    sdfTexture: Texture;
}

interface FontGlyphAdvance {
    horizontal: number;
    vertical: number;
}

interface FontGlyphBounds {
    left: number;
    bottom: number;
    right: number;
    top: number;
}

interface FontGlyph {
    codepoint: number;
    advance: FontGlyphAdvance;
    image: number; // 0
    image_bounds: FontGlyphBounds;
    plane_bounds: FontGlyphBounds;
}

interface FontMetrics {
    font_size: number;
    distance_range: number;
    em_size: number;
    ascender: number;
    descender: number;
}

export interface FontVariant {
    name: string;
    codepoint_type: string; // Indexed, Unicode
    fallback_glyph: number; // 0
    fallback_variant: number; // 0
    flags: number; // 0
    glyphs: FontGlyph[];
    image_type: string; // Mtsdf
    kern_pairs: [];
    metadata: string;
    metadata_format: string; // None, Json
    metrics: FontMetrics;
    weight: number; // 0
}

export class FontLoader {
    private wasmInitialized = false;

    public fontInfo?: FontInfo;
    public sdfTexture?: Texture;

    async load(url: string): Promise<FontInfo> {
        // Initialize WASM only once
        if (!this.wasmInitialized) {
            await (init as unknown as () => Promise<void>)();
            this.wasmInitialized = true;
        }

        const fontData = await fetch(url).then(res => res.arrayBuffer());
        const uint8Array = new Uint8Array(fontData);
        this.fontInfo = ArteryFont.read_from_bytes(uint8Array);

        // Return the first processed image
        const image = this.fontInfo?.images[0];
        if (!image) {
            throw new Error('No image found');
        }
        const canvas = document.createElement('canvas');
        canvas.width = image.width;
        canvas.height = image.height;
        const ctx = canvas.getContext('2d');
        const dataArray = new Uint8ClampedArray(image.data);
        const imgData = new ImageData(dataArray, image.width, image.height);
        ctx!.putImageData(imgData, 0, 0);

        this.sdfTexture = new Texture(canvas, true);
        this.fontInfo!.sdfTexture = this.sdfTexture;

        return this.fontInfo!;
    }
} 