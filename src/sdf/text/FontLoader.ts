import { Texture } from '../../buffers/Texture';
import { ArFont, parseArFont } from './ArFont';

export type { FontImage, FontGlyph, FontVariant, FontMetrics, FontKernPair } from './ArFont';

/** A parsed `.arfont` and its first atlas image as a texture. */
export interface FontInfo extends ArFont {
    sdfTexture: Texture;
}

/**
 * Loads an `.arfont` MTSDF atlas (msdf-atlas-gen's Artery Font output): glyph metrics for
 * `TextGeometry` and the atlas as a texture. Parsed in plain TS (`ArFont.ts`), as the Rust
 * engine's `FontAtlas::parse` does.
 */
export class FontLoader {
    public fontInfo?: FontInfo;
    public sdfTexture?: Texture;

    async load(url: string): Promise<FontInfo> {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
        return this.parse(new Uint8Array(await response.arrayBuffer()));
    }

    /** Parse a `.arfont` file's bytes. */
    async parse(bytes: Uint8Array): Promise<FontInfo> {
        const font = await parseArFont(bytes);
        const image = font.images[0];
        if (!image) {
            throw new Error('No image found');
        }
        // The channels are distances, not colour: upload them as they are (linear RGBA8, rows
        // from the atlas's bottom, as the glyphs' image bounds count them).
        let rgba = image.data;
        if (image.channels !== 4) {
            rgba = new Uint8Array(image.width * image.height * 4);
            for (let i = 0, n = image.width * image.height; i < n; i++) {
                for (let c = 0; c < 4; c++) rgba[i * 4 + c] = c < image.channels ? image.data[i * image.channels + c] : 255;
            }
        }
        this.sdfTexture = new Texture({ label: 'FontAtlas', width: image.width, height: image.height, format: 'rgba8unorm', levels: [rgba] });
        this.fontInfo = { ...font, sdfTexture: this.sdfTexture };
        return this.fontInfo;
    }
}
