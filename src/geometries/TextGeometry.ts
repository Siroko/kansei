import { BufferBase } from "../buffers/BufferBase";
import { ComputeBuffer } from "../buffers/ComputeBuffer";
import { FontInfo } from "../sdf/text/FontLoader";
import { InstancedGeometry } from "./InstancedGeometry";
import { PlaneGeometry } from "./PlaneGeometry";

class TextGeometry extends InstancedGeometry {
    private _text: string;
    private _fontInfo: FontInfo;
    private _width: number;
    private _height: number;
    private _fontSize: number;

    constructor(text: string, fontInfo: FontInfo, width: number, height: number, fontSize: number) {
        super(new PlaneGeometry(1, 1), text.length);

        this._text = text;
        this._fontInfo = fontInfo;
        this._width = width;
        this._height = height;
        this._fontSize = fontSize;

        const myBuffer = new Float32Array(text.length * 4);

        for (let i = 0; i < text.length; i++) {
            const glyph = fontInfo.variants[0].glyphs.find((char) => char.codepoint === text.charCodeAt(i));
            const prevGlyph = fontInfo.variants[0].glyphs.find((char) => char.codepoint === text.charCodeAt(i - 1));

            myBuffer[i * 4] = i % width + (glyph?.advance.horizontal ?? 0); // X coordinate
            myBuffer[i * 4 + 1] = Math.floor(i / width); // Y coordinate
            myBuffer[i * 4 + 2] = 0; // Z coordinate
            myBuffer[i * 4 + 3] = 1; // W coordinate
        }

        const computeBufferPositions = new ComputeBuffer({
            usage:
                BufferBase.BUFFER_USAGE_STORAGE |
                BufferBase.BUFFER_USAGE_COPY_SRC |
                BufferBase.BUFFER_USAGE_VERTEX,
            type: ComputeBuffer.BUFFER_TYPE_STORAGE,
            buffer: myBuffer,
            shaderLocation: 3,
            offset: 0,
            stride: 4 * 4,
            format: "float32x4"
        });

        this.extraBuffers.push(computeBufferPositions);
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