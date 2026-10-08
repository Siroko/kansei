import { Camera } from '../cameras/Camera';
import { Renderer } from '../renderers/Renderer';
import { Scene } from '../objects/Scene';
import { GBuffer } from './GBuffer';
import { PostProcessingEffect } from './PostProcessingEffect';
import { cpuScope, endProfiledFrame, gpuPass } from '../profiling/Profiler';

/** Options of a `PostProcessingVolume`. */
export interface PostProcessingVolumeOptions {
    /**
     * Samples per pixel of the GBuffer's scene pass: 1 (the Rust volume's, anti-aliased by
     * `TemporalAAEffect`) or 4 (MSAA, resolved before the chain; its depth is the nearest of the
     * four samples). Default 1.
     */
    msaaSampleCount?: number;
}

/** Jitter phases for a render scale: 8 per displayed pixel (Rust `jitter_phases`). */
export function jitterPhases(scale: number): number {
    return Math.round(8 / (scale * scale));
}

/** The `index`-th element (from 1) of the Halton sequence in `base`, in [0, 1) (Rust `halton`). */
export function halton(index: number, base: number): number {
    let result = 0;
    let f = 1;
    while (index > 0) {
        f /= base;
        result += f * (index % base);
        index = Math.floor(index / base);
    }
    return result;
}

/**
 * PostProcessingVolume
 * ====================
 * Orchestrates a chain of compute-shader post-processing effects on top of the
 * scene rendered into a GBuffer.
 *
 * Usage
 * -----
 * ```typescript
 * const volume = new PostProcessingVolume(renderer, [
 *     new SSAOEffect({ radius: 0.5 }),
 *     new GodRaysEffect({ lightScreenPos: new Vector2(0.5, 0.3) }),
 *     new DepthOfFieldEffect({ focusDistance: 5.0 }),
 * ]);
 *
 * // In your render loop (replaces renderer.render()):
 * volume.render(scene, camera);
 * ```
 *
 * Rendering pipeline
 * ------------------
 *  1. renderToGBuffer — scene is drawn into the GBuffer's targets (rgba16float colour,
 *                       depth32float depth, single-sample unless `msaaSampleCount` says
 *                       otherwise) at the renderer's `renderSize`, then the velocity pass. The
 *                       camera is jittered by a sub-pixel Halton offset while an effect
 *                       `wantsJitter` (TAA).
 *  2. Effects chain   — each effect reads the previous output and writes to
 *                       the next (ping-pong between outputTexture / pingPongTexture).
 *                       With a render scale below 1 the first effect that
 *                       `upscalesToDisplay` (TAA) takes the chain from the render size to
 *                       the canvas size, and the effects after it run on the volume's own
 *                       canvas-size pair (they read depth and the GBuffer at the render size).
 *  3. Blit            — the final texture is rendered to the canvas via a
 *                       fullscreen triangle pass (stretched when nothing upscaled it).
 *
 * Effects up to the tonemapper work on scene-linear HDR light; `ToneMapEffect` turns it into the
 * display signal, so it goes last in the HDR chain (after fog, depth of field and bloom), as in
 * the Rust engine (`postprocessing/mod.rs`). Without it the canvas shows linear HDR as is.
 * Effects whose `isActive()` is false are skipped.
 */
class PostProcessingVolume {
    private _gbuffer: GBuffer | null = null;
    private _blitPipeline: GPURenderPipeline | null = null;
    private _blitSampler: GPUSampler | null = null;
    // Bind group is rebuilt whenever the source texture changes (ping-pong swap).
    private _blitBindGroup: GPUBindGroup | null = null;
    private _blitLastSource: GPUTexture | null = null;
    // Ping-pong textures at the canvas size, for the effects after an upscaler.
    private _displayTargets: GPUTexture[] = [];
    private readonly _msaaSampleCount: number;

    public effects: PostProcessingEffect[] = [];

    constructor(
        private renderer: Renderer,
        effects: PostProcessingEffect[] = [],
        options: PostProcessingVolumeOptions = {}
    ) {
        this.effects = effects;
        this._msaaSampleCount = options.msaaSampleCount ?? 1;
    }

    /** Add an effect at the end of the chain. */
    public addEffect(effect: PostProcessingEffect): void {
        this.effects.push(effect);
    }

    /** Whether any active effect wants a jittered projection (the renderer then jitters the camera). */
    public get wantsJitter(): boolean {
        return this.effects.some(e => e.isActive() && e.wantsJitter());
    }

    /** The GBuffer the scene is rendered into (created by the first `render`). */
    public get gbuffer(): GBuffer | null {
        return this._gbuffer;
    }

    /** Remove all effects. */
    public clearEffects(): void {
        this.effects.forEach(e => e.destroy());
        this.effects = [];
    }

    /**
     * Render the scene through the post-processing chain and blit to the canvas.
     * Call this instead of renderer.render() every frame.
     */
    public render(scene: Scene, camera: Camera): void {
        const frameScope = cpuScope('frame');
        const device = this.renderer.gpuDevice;
        const [w, h] = this.renderer.renderSize;
        const displayW = this.renderer.renderWidth;
        const displayH = this.renderer.renderHeight;

        // Lazily create / resize the GBuffer, at the render size.
        if (!this._gbuffer) {
            this._gbuffer = new GBuffer(device, w, h, this._msaaSampleCount);
        } else if (this._gbuffer.width !== w || this._gbuffer.height !== h) {
            this._gbuffer.resize(w, h);
            // Invalidate effects so they rebuild their size-dependent state.
            for (const effect of this.effects) {
                if (effect.initialized) {
                    effect.resize(w, h, this._gbuffer);
                }
            }
            this._blitBindGroup = null;
        }

        // Sub-pixel jitter for temporal anti-aliasing, in rendered pixels: a Halton (2, 3)
        // sequence of 8 phases per displayed pixel, so each one still sees 8 samples when
        // rendering below the display size (Rust `render_with_postprocessing`).
        if (this.wantsJitter) {
            const i = camera.frame % jitterPhases(this.renderer.renderScale) + 1;
            camera.jitter = [(2 * halton(i, 2) - 1) / w, (2 * halton(i, 3) - 1) / h];
        } else {
            camera.jitter = [0, 0];
        }

        // Step 1: render scene into the GBuffer.
        this.renderer.renderToGBuffer(scene, camera, this._gbuffer);

        const postScope = cpuScope('post');
        // Step 2: initialise any uninitialised effects.
        const active = this.effects.filter(e => e.isActive());
        for (const effect of active) {
            if (!effect.initialized) {
                effect.initialize(device, this._gbuffer, camera);
            }
        }

        // Step 3: run the effect chain using ping-pong.
        //   First effect reads from colorTexture.
        //   Subsequent effects alternate between outputTexture and pingPongTexture, at the
        //   render size; from the upscaler on, between the two display targets, at the canvas size.
        //   At the end, currentSource holds the final composited image.
        const upscaling = (w !== displayW || h !== displayH) && active.some(e => e.upscalesToDisplay());
        if (upscaling) this._ensureDisplayTargets(device, displayW, displayH);
        let currentSource: GPUTexture = this._gbuffer.colorTexture;
        let atDisplay = false;

        if (active.length > 0) {
            const commandEncoder = device.createCommandEncoder();

            for (const effect of active) {
                atDisplay ||= upscaling && effect.upscalesToDisplay();
                const outputTex = atDisplay
                    ? (currentSource === this._displayTargets[0] ? this._displayTargets[1] : this._displayTargets[0])
                    : (currentSource === this._gbuffer.outputTexture ? this._gbuffer.pingPongTexture : this._gbuffer.outputTexture);

                // the effect's class name, as Rust's `effect.name()` (a minifier may shorten it)
                const effectScope = cpuScope(effect.constructor.name);
                effect.render(
                    commandEncoder,
                    currentSource,
                    this._gbuffer.depthTexture,
                    outputTex,
                    camera,
                    atDisplay ? displayW : w,
                    atDisplay ? displayH : h,
                    this._gbuffer.emissiveTexture,
                    this._gbuffer
                );
                effectScope?.end();

                currentSource = outputTex;
            }

            device.queue.submit([commandEncoder.finish()]);
        }

        // Step 4: blit the final texture to the canvas.
        this._blit(device, currentSource);
        postScope?.end();
        camera.endFrame();
        frameScope?.end();
        endProfiledFrame();
    }

    /** The canvas-size ping-pong textures, (re)created at `width` x `height`. */
    private _ensureDisplayTargets(device: GPUDevice, width: number, height: number): void {
        const current = this._displayTargets[0];
        if (current && current.width === width && current.height === height) return;
        for (const t of this._displayTargets) t.destroy();
        const target = (label: string) => device.createTexture({
            label,
            size: [width, height],
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
        });
        this._displayTargets = [target('PostProcessingVolume/DisplayA'), target('PostProcessingVolume/DisplayB')];
    }

    // ── Blit pass ────────────────────────────────────────────────────────────

    private _ensureBlitPipeline(device: GPUDevice): void {
        if (this._blitPipeline) return;

        const blitShader = /* wgsl */`
            @group(0) @binding(0) var sourceTex : texture_2d<f32>;
            @group(0) @binding(1) var blitSampler : sampler;

            struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) uv : vec2f,
            }

            // Full-screen triangle — covers the whole viewport with 3 vertices.
            @vertex
            fn vertex_main(@builtin(vertex_index) vertIndex : u32) -> VertexOutput {
                const pos = array<vec2f, 3>(
                    vec2f(-1.0, -1.0),
                    vec2f( 3.0, -1.0),
                    vec2f(-1.0,  3.0),
                );
                const uv = array<vec2f, 3>(
                    vec2f(0.0, 1.0),
                    vec2f(2.0, 1.0),
                    vec2f(0.0, -1.0),
                );
                return VertexOutput(vec4f(pos[vertIndex], 0.0, 1.0), uv[vertIndex]);
            }

            @fragment
            fn fragment_main(input : VertexOutput) -> @location(0) vec4f {
                return textureSample(sourceTex, blitSampler, input.uv);
            }
        `;

        const module = device.createShaderModule({ code: blitShader });

        const bgl = device.createBindGroupLayout({
            label: 'Blit BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
            ],
        });

        this._blitSampler = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });

        this._blitPipeline = device.createRenderPipeline({
            label: 'Blit Pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
            vertex: { module, entryPoint: 'vertex_main' },
            fragment: {
                module,
                entryPoint: 'fragment_main',
                targets: [{ format: this.renderer.presentationFormat }],
            },
            primitive: { topology: 'triangle-list' },
        });
    }

    private _blit(device: GPUDevice, sourceTex: GPUTexture): void {
        this._ensureBlitPipeline(device);

        // Rebuild the bind group only when the source texture changes.
        if (this._blitLastSource !== sourceTex) {
            const bgl = this._blitPipeline!.getBindGroupLayout(0);
            this._blitBindGroup = device.createBindGroup({
                label: 'Blit BindGroup',
                layout: bgl,
                entries: [
                    { binding: 0, resource: sourceTex.createView() },
                    { binding: 1, resource: this._blitSampler! },
                ],
            });
            this._blitLastSource = sourceTex;
        }

        const commandEncoder = device.createCommandEncoder();
        const surfaceScope = cpuScope('frame/surface');
        const swapchainView = this.renderer.context!.getCurrentTexture().createView();
        surfaceScope?.end();

        const pass = commandEncoder.beginRenderPass({
            label: 'Blit RenderPass',
            timestampWrites: gpuPass('Blit RenderPass'),
            colorAttachments: [{
                view: swapchainView,
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
            }],
        });
        pass.setPipeline(this._blitPipeline!);
        pass.setBindGroup(0, this._blitBindGroup!);
        pass.draw(3);
        pass.end();

        device.queue.submit([commandEncoder.finish()]);
    }

    /** Release all GPU resources owned by this volume and its effects. */
    public destroy(): void {
        this._gbuffer?.destroy();
        for (const t of this._displayTargets) t.destroy();
        this._displayTargets = [];
        for (const effect of this.effects) effect.destroy();
    }
}

export { PostProcessingVolume };
