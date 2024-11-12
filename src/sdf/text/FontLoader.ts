import init, { ArteryFont } from './loader/artery_font';

// Add this interface above the class
interface FontImage {
    width: number;
    height: number;
    data: number[];
}

export class FontLoader {
    private wasmInitialized = false;

    async load(url: string): Promise<HTMLCanvasElement> {
        // Initialize WASM only once
        if (!this.wasmInitialized) {
            await (init as unknown as () => Promise<void>)();
            this.wasmInitialized = true;
        }

        const fontData = await fetch(url).then(res => res.arrayBuffer());
        const uint8Array = new Uint8Array(fontData);
        const fontInfo = ArteryFont.read_from_bytes(uint8Array);

        // Return the first processed image
        const image = fontInfo.images[0] as FontImage;
        const canvas = document.createElement('canvas');
        canvas.width = image.width;
        canvas.height = image.height;
        const ctx = canvas.getContext('2d');
        const dataArray = new Uint8ClampedArray(image.data);
        const imgData = new ImageData(dataArray, image.width, image.height);
        ctx!.putImageData(imgData, 0, 0);
        document.body.appendChild(canvas);

        console.log(fontInfo);
        return canvas;
    }
} 