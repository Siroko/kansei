import { BufferBase } from "../buffers/BufferBase";
import { ComputeBuffer } from "../buffers/ComputeBuffer";
import { Vector4 } from "../math/Vector4";
import { FontInfo } from "../sdf/text/FontLoader";
import { InstancedGeometry } from "./InstancedGeometry";
import { PlaneGeometry } from "./PlaneGeometry";

class TextGeometry extends InstancedGeometry {
    private _text: string;
    private _fontInfo: FontInfo;
    private _width: number;
    private _height: number;
    private _fontSize: number;
    private _color: Vector4;

    constructor(options: {
        text: string,
        fontInfo: FontInfo,
        width: number,
        height: number,
        fontSize: number,
        color?: Vector4
    }) {
        super(new PlaneGeometry(1, 1), options.text.length);

        console.log(options.fontInfo);
        this._text = options.text;
        this._fontInfo = options.fontInfo;
        this._width = options.width;
        this._height = options.height;
        this._fontSize = options.fontSize;
        this._color = options.color || new Vector4(1, 1, 1, 1);

        const posBuffer = new Float32Array(this._text.length * 4);
        const planeBoundsBuffer = new Float32Array(this._text.length * 4);
        const imageBoundsBuffer = new Float32Array(this._text.length * 4);
        const colorsBuffer = new Float32Array(this._text.length * 4);

        let posX = 0;
        let posY = 0;
        let itText = 0;
        const words = this._text.split(" ");
        for (let word of words) {
            let wordWidth = 0;
            for (let _char of word) {
                const glyph = this._fontInfo.variants[0].glyphs.find((char: any) => char.codepoint === _char.charCodeAt(0));
                wordWidth += glyph!.advance.horizontal;
            }
            if (posX + wordWidth > this._width) {
                posX = 0;
                posY -= 1.1;
            }
            for (let wordIt = 0; wordIt < word.length; wordIt++) {
                const glyph = this._fontInfo.variants[0].glyphs.find((char: any) => char.codepoint === word[wordIt].charCodeAt(0));

                posBuffer[itText * 4] = posX; // X coordinate
                posBuffer[itText * 4 + 1] = posY; // Y coordinate
                posBuffer[itText * 4 + 2] = 0; // Z coordinate
                posBuffer[itText * 4 + 3] = 1; // W coordinate

                imageBoundsBuffer[itText * 4] = glyph!.image_bounds.left / this._fontInfo.images[0].width;
                imageBoundsBuffer[itText * 4 + 1] = glyph!.image_bounds.top / this._fontInfo.images[0].height;
                imageBoundsBuffer[itText * 4 + 2] = glyph!.image_bounds.right / this._fontInfo.images[0].width;
                imageBoundsBuffer[itText * 4 + 3] = glyph!.image_bounds.bottom / this._fontInfo.images[0].height;

                planeBoundsBuffer[itText * 4] = glyph!.plane_bounds.left;
                planeBoundsBuffer[itText * 4 + 1] = glyph!.plane_bounds.top;
                planeBoundsBuffer[itText * 4 + 2] = glyph!.plane_bounds.right;
                planeBoundsBuffer[itText * 4 + 3] = glyph!.plane_bounds.bottom;

                colorsBuffer[itText * 4] = this._color.x;
                colorsBuffer[itText * 4 + 1] = this._color.y;
                colorsBuffer[itText * 4 + 2] = this._color.z;
                colorsBuffer[itText * 4 + 3] = this._color.w;

                itText++;
                posX += glyph!.advance.horizontal;
                // Add space between words
                if (wordIt === word.length - 1) {
                    const spaceGlyph = this._fontInfo.variants[0].glyphs.find((char: any) => char.codepoint === ' '.charCodeAt(0));
                    posBuffer[itText * 4] = posX; // X coordinate
                    posBuffer[itText * 4 + 1] = posY; // Y coordinate
                    posBuffer[itText * 4 + 2] = 0; // Z coordinate
                    posBuffer[itText * 4 + 3] = 1; // W coordinate

                    imageBoundsBuffer[itText * 4] = spaceGlyph!.image_bounds.left / this._fontInfo.images[0].width;
                    imageBoundsBuffer[itText * 4 + 1] = spaceGlyph!.image_bounds.top / this._fontInfo.images[0].height;
                    imageBoundsBuffer[itText * 4 + 2] = spaceGlyph!.image_bounds.right / this._fontInfo.images[0].width;
                    imageBoundsBuffer[itText * 4 + 3] = spaceGlyph!.image_bounds.bottom / this._fontInfo.images[0].height;

                    planeBoundsBuffer[itText * 4] = spaceGlyph!.plane_bounds.left;
                    planeBoundsBuffer[itText * 4 + 1] = spaceGlyph!.plane_bounds.top;
                    planeBoundsBuffer[itText * 4 + 2] = spaceGlyph!.plane_bounds.right;
                    planeBoundsBuffer[itText * 4 + 3] = spaceGlyph!.plane_bounds.bottom;

                    colorsBuffer[itText * 4] = this._color.x;
                    colorsBuffer[itText * 4 + 1] = this._color.y;
                    colorsBuffer[itText * 4 + 2] = this._color.z;
                    colorsBuffer[itText * 4 + 3] = this._color.w;

                    posX += spaceGlyph!.advance.horizontal;
                    itText++;
                }
            }
        }

        const computeBufferPositions = new ComputeBuffer({
            usage:
                BufferBase.BUFFER_USAGE_STORAGE |
                BufferBase.BUFFER_USAGE_COPY_SRC |
                BufferBase.BUFFER_USAGE_VERTEX,
            type: ComputeBuffer.BUFFER_TYPE_STORAGE,
            buffer: posBuffer,
            shaderLocation: 3,
            offset: 0,
            stride: 4 * 4,
            format: "float32x4"
        });

        const computeBufferImageBounds = new ComputeBuffer({
            usage:
                BufferBase.BUFFER_USAGE_STORAGE |
                BufferBase.BUFFER_USAGE_COPY_SRC |
                BufferBase.BUFFER_USAGE_VERTEX,
            type: ComputeBuffer.BUFFER_TYPE_STORAGE,
            buffer: imageBoundsBuffer,
            shaderLocation: 4,
            offset: 0,
            stride: 4 * 4,
            format: "float32x4"
        });

        const computeBufferPlaneBounds = new ComputeBuffer({
            usage:
                BufferBase.BUFFER_USAGE_STORAGE |
                BufferBase.BUFFER_USAGE_COPY_SRC |
                BufferBase.BUFFER_USAGE_VERTEX,
            type: ComputeBuffer.BUFFER_TYPE_STORAGE,
            buffer: planeBoundsBuffer,
            shaderLocation: 5,
            offset: 0,
            stride: 4 * 4,
            format: "float32x4"
        });

        const computeBufferColors = new ComputeBuffer({
            usage:
                BufferBase.BUFFER_USAGE_STORAGE |
                BufferBase.BUFFER_USAGE_COPY_SRC |
                BufferBase.BUFFER_USAGE_VERTEX,
            type: ComputeBuffer.BUFFER_TYPE_STORAGE,
            buffer: colorsBuffer,
            shaderLocation: 6,
            offset: 0,
            stride: 4 * 4,
            format: "float32x4"
        });

        this.extraBuffers.push(
            computeBufferPositions,
            computeBufferImageBounds,
            computeBufferPlaneBounds,
            computeBufferColors
        );
    }

    set fontSize(fontSize: number) {
        this._fontSize = fontSize;
    }

    get fontSize() {
        return this._fontSize;
    }

    set text(text: string) {
        this._text = text;
    }

    get text() {
        return this._text;
    }

    set width(width: number) {
        this._width = width;
    }

    get width() {
        return this._width;
    }

    set height(height: number) {
        this._height = height;
    }

    get height() {
        return this._height;
    }
}

export { TextGeometry }