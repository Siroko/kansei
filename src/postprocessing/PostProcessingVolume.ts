import { Camera } from '../cameras/Camera';
import { Renderer } from '../renderers/Renderer';
import { Scene } from '../objects/Scene';
import { GBuffer } from './GBuffer';
import { PostProcessingEffect } from './PostProcessingEffect';

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
 *  1. renderToGBuffer — scene is drawn into a rgba16float colour texture +
 *                       depth32float depth texture (sampleCount=1).
 *  2. Effects chain   — each effect reads the previous output and writes to
 *                       the next (ping-pong between outputTexture / pingPongTexture).
 *  3. Blit            — the final texture is rendered to the canvas via a
 *                       fullscreen triangle pass.
 */
class PostProcessingVolume {
    private _gbuffer: GBuffer | null = null;
    private _blitPipeline: GPURenderPipeline | null = null;
    private _blitSampler: GPUSampler | null = null;
    // Bind group is rebuilt whenever the source texture changes (ping-pong swap).
    private _blitBindGroup: GPUBindGroup | null = null;
    private _blitLastSource: GPUTexture | null = null;

    constructor(
        private renderer: Renderer,
        private effects: PostProcessingEffect[] = []
    ) {}

    /** Add an effect at the end of the chain. */
    public addEffect(effect: PostProcessingEffect): void {
        this.effects.push(effect);
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
        const device = this.renderer.gpuDevice;
        const w = this.renderer.renderWidth;
        const h = this.renderer.renderHeight;

        // Lazily create / resize the GBuffer.
        if (!this._gbuffer) {
            this._gbuffer = new GBuffer(device, w, h);
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

        // Step 1: render scene into the GBuffer.
        this.renderer.renderToGBuffer(scene, camera, this._gbuffer);

        // Step 2: initialise any uninitialised effects.
        for (const effect of this.effects) {
            if (!effect.initialized) {
                effect.initialize(device, this._gbuffer, camera);
            }
        }

        // Step 3: run the effect chain using ping-pong.
        //   First effect reads from colorTexture.
        //   Subsequent effects alternate between outputTexture and pingPongTexture.
        //   At the end, currentSource holds the final composited image.
        let currentSource: GPUTexture = this._gbuffer.colorTexture;
        let pingPongIdx = 0; // 0 → write outputTexture, 1 → write pingPongTexture

        if (this.effects.length > 0) {
            const commandEncoder = device.createCommandEncoder();

            for (const effect of this.effects) {
                const outputTex = pingPongIdx === 0
                    ? this._gbuffer.outputTexture
                    : this._gbuffer.pingPongTexture;

                effect.render(
                    commandEncoder,
                    currentSource,
                    this._gbuffer.depthTexture,
                    outputTex,
                    camera,
                    w,
                    h
                );

                currentSource = outputTex;
                pingPongIdx = 1 - pingPongIdx;
            }

            device.queue.submit([commandEncoder.finish()]);
        }

        // Step 4: blit the final texture to the canvas.
        this._blit(device, currentSource);
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
        const swapchainView = this.renderer.context!.getCurrentTexture().createView();

        const pass = commandEncoder.beginRenderPass({
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
        for (const effect of this.effects) effect.destroy();
    }
}

export { PostProcessingVolume };
