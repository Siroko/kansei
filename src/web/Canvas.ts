/**
 * The canvas an example draws to: its drawing buffer sized from its CSS box and the device
 * pixel ratio, kept in step when the page resizes. The TS side of `rust/kansei-wasm/src/canvas.rs`.
 */
import { Renderer, RendererOptions } from "../renderers/Renderer";
import { param } from "./page";

/**
 * The canvas's drawing buffer is its CSS size times `devicePixelRatio`, at most this much unless
 * `withMaxPixelRatio` or `?dpr=` says otherwise.
 */
const DEFAULT_MAX_PIXEL_RATIO = 2;

/**
 * The page's canvas. Its drawing buffer (`width`/`height` attributes) is the CSS box times the
 * device pixel ratio; `run` re-measures it when the box or the ratio changes and reports the new
 * size in `Frame.resized`.
 *
 * `?dpr=<ratio>` on the page's URL fixes the ratio (`?dpr=1` draws one pixel per CSS pixel).
 * The page must give the canvas a CSS size (`width: 100vw; height: 100vh`, say): its box is
 * measured, and a box sized by the drawing buffer would grow with it.
 */
export class Canvas {
    private maxPixelRatio = DEFAULT_MAX_PIXEL_RATIO;
    /** `?dpr=`, read once. */
    private readonly pixelRatioParam: number | null;
    /** A drawing-buffer size that ignores the page (`withSize`). */
    private fixedSize: [number, number] | null = null;
    /** Set by the resize observer; the frame loop re-measures on the next frame. */
    private boxChanged = false;
    /** The ratio last measured, to notice a move to a screen with another ratio. */
    private measuredPixelRatio = 0;
    private readonly observer: ResizeObserver | null;

    /** Size `element` for the screen (it needs a CSS size; see the class notes). */
    constructor(public readonly element: HTMLCanvasElement) {
        const dpr = Number(param('dpr'));
        this.pixelRatioParam = Number.isFinite(dpr) && dpr > 0 ? dpr : null;
        this.observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => { this.boxChanged = true; });
        this.observer?.observe(element);
        this.measure();
    }

    /** The `<canvas>` with this id, sized for the screen. */
    public static find(id: string): Canvas {
        const element = document.getElementById(id);
        if (!(element instanceof HTMLCanvasElement)) throw new Error(`no canvas #${id}`);
        return new Canvas(element);
    }

    /**
     * A new `<canvas>` filling `parent` (the page by default: `display: block`, 100% of its
     * width and height; the parent needs a height), sized for the screen.
     */
    public static fill(parent: HTMLElement = document.body): Canvas {
        const element = document.createElement('canvas');
        Object.assign(element.style, { display: 'block', width: '100%', height: '100%' });
        if (parent === document.body) {
            Object.assign(document.documentElement.style, { height: '100%' });
            Object.assign(document.body.style, { height: '100%', margin: '0' });
        }
        parent.appendChild(element);
        return new Canvas(element);
    }

    /**
     * Cap the device pixel ratio at `ratio` (2 by default): a costly example can draw fewer
     * pixels on high-density screens. `?dpr=` still overrides it.
     */
    public withMaxPixelRatio(ratio: number): this {
        this.maxPixelRatio = Math.max(ratio, 0.1);
        this.measure();
        return this;
    }

    /**
     * Draw at exactly `width` x `height` pixels whatever the page's size (the CSS box still
     * stretches it), e.g. for a benchmark at a fixed resolution.
     */
    public withSize(width: number, height: number): this {
        this.fixedSize = [Math.max(1, Math.round(width)), Math.max(1, Math.round(height))];
        this.setSize(this.fixedSize);
        return this;
    }

    /** The drawing buffer's size in pixels. */
    public get size(): [number, number] {
        return [Math.max(this.element.width, 1), Math.max(this.element.height, 1)];
    }

    /** Width over height, for a camera's projection. */
    public get aspect(): number {
        const [width, height] = this.size;
        return width / height;
    }

    /** Drawing-buffer pixels per CSS pixel, as last measured. */
    public get pixelRatio(): number {
        return this.measuredPixelRatio;
    }

    /**
     * A renderer drawing to this canvas, at its size: `options`' other fields (sample count,
     * clear colour, limits, ...) as given.
     */
    public async renderer(options: RendererOptions = {}): Promise<Renderer> {
        const [width, height] = this.size;
        const renderer = new Renderer({ ...options, canvas: this.element, width, height, devicePixelRatio: 1 });
        await renderer.initialize();
        return renderer;
    }

    /**
     * Re-measure if the CSS box or the pixel ratio changed; the new drawing-buffer size when it
     * differs from the current one. `run` calls it every frame.
     */
    public pollResize(): [number, number] | null {
        if (this.fixedSize) return null;
        const ratio = this.currentPixelRatio();
        if (!this.boxChanged && ratio === this.measuredPixelRatio) return null;
        this.boxChanged = false;
        const [w0, h0] = this.size;
        this.measure();
        const [w1, h1] = this.size;
        return w1 !== w0 || h1 !== h0 ? [w1, h1] : null;
    }

    /** Stop following the page's size. */
    public dispose(): void {
        this.observer?.disconnect();
    }

    private currentPixelRatio(): number {
        if (this.pixelRatioParam !== null) return Math.min(Math.max(this.pixelRatioParam, 0.1), 4);
        return Math.min(window.devicePixelRatio || 1, this.maxPixelRatio);
    }

    private measure(): void {
        const ratio = this.currentPixelRatio();
        this.measuredPixelRatio = ratio;
        if (this.fixedSize) {
            this.setSize(this.fixedSize);
            return;
        }
        const width = Math.round(Math.max(this.element.clientWidth, 1) * ratio);
        const height = Math.round(Math.max(this.element.clientHeight, 1) * ratio);
        this.setSize([width, height]);
    }

    private setSize([width, height]: [number, number]): void {
        if (this.element.width !== width) this.element.width = width;
        if (this.element.height !== height) this.element.height = height;
    }
}
