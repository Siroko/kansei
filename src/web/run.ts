/**
 * The `requestAnimationFrame` loop. The TS side of `rust/kansei-wasm/src/frame_loop.rs`.
 */
import type { Camera } from "../cameras/Camera";
import type { Renderer } from "../renderers/Renderer";
import type { Canvas } from "./Canvas";
import { now } from "./page";

/** One animation frame, as `run` hands it to the example. */
export class Frame {
    constructor(
        /** Seconds since `run` started. */
        public readonly time: number,
        /** Seconds since the previous frame (0 on the first). */
        public readonly dt: number,
        /** Frames before this one. */
        public readonly index: number,
        /** The canvas's drawing-buffer size in pixels. */
        public readonly size: [number, number],
        /** The new drawing-buffer size when the canvas changed size since the previous frame. */
        public readonly resized: [number, number] | null,
    ) { }

    /**
     * Apply a resize, if this frame has one, to the renderer (canvas and depth targets) and the
     * camera's aspect ratio. Call it before rendering; an example with size-dependent resources
     * of its own checks `resized` as well.
     */
    public resize(renderer: Renderer, camera: Camera): void {
        if (!this.resized) return;
        const [width, height] = this.resized;
        renderer.resize(width, height);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
    }
}

/**
 * Call `frame` on every animation frame from now on, with the frame's timing and any canvas
 * resize. When `frame` returns a promise (an `async` frame), the next frame waits for it.
 * Returns a function that stops the loop.
 */
export function run(canvas: Canvas, frame: (frame: Frame) => void | Promise<void>): () => void {
    const start = now();
    let last = start;
    let index = 0;
    let stopped = false;
    const tick = async () => {
        if (stopped) return;
        const t = now();
        const resized = canvas.pollResize();
        await frame(new Frame(t - start, index === 0 ? 0 : t - last, index, canvas.size, resized));
        last = t;
        index++;
        if (!stopped) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    return () => { stopped = true; };
}
