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

    constructor(text: string,
        fontInfo: FontInfo,
        width: number,
        height: number,
        fontSize: number,
        color: Vector4 = new Vector4(1, 1, 1, 1)
    ) {
        super(new PlaneGeometry(1, 1), text.length);

        this._text = text;
        this._fontInfo = fontInfo;
        this._width = width;
        this._height = height;
        this._fontSize = fontSize;
        this._color = color;

        const posBuffer = new Float32Array(text.length * 4);
        const planeBoundsBuffer = new Float32Array(text.length * 4);
        const imageBoundsBuffer = new Float32Array(text.length * 4);
        const colorsBuffer = new Float32Array(text.length * 4);

        let posX = 0;
        for (let i = 0; i < text.length; i++) {
            const glyph = fontInfo.variants[0].glyphs.find((char) => char.codepoint === text.charCodeAt(i));

            posBuffer[i * 4] = posX % this._width; // X coordinate
            posBuffer[i * 4 + 1] = -Math.floor(posX / this._width); // Y coordinate
            posBuffer[i * 4 + 2] = 0; // Z coordinate
            posBuffer[i * 4 + 3] = 1; // W coordinate

            imageBoundsBuffer[i * 4] = glyph!.image_bounds.left / this._fontInfo.images[0].width;
            imageBoundsBuffer[i * 4 + 1] = glyph!.image_bounds.top / this._fontInfo.images[0].height;
            imageBoundsBuffer[i * 4 + 2] = glyph!.image_bounds.right / this._fontInfo.images[0].width;
            imageBoundsBuffer[i * 4 + 3] = glyph!.image_bounds.bottom / this._fontInfo.images[0].height;

            planeBoundsBuffer[i * 4] = glyph!.plane_bounds.left;
            planeBoundsBuffer[i * 4 + 1] = glyph!.plane_bounds.top;
            planeBoundsBuffer[i * 4 + 2] = glyph!.plane_bounds.right;
            planeBoundsBuffer[i * 4 + 3] = glyph!.plane_bounds.bottom;

            colorsBuffer[i * 4] = this._color.x;
            colorsBuffer[i * 4 + 1] = this._color.y;
            colorsBuffer[i * 4 + 2] = this._color.z;
            colorsBuffer[i * 4 + 3] = this._color.w;

            posX += glyph!.advance.horizontal;
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