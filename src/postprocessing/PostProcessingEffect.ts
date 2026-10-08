import { Camera } from '../cameras/Camera';
import { GBuffer } from './GBuffer';

/**
 * Abstract base class for all post-processing effects.
 *
 * Each effect is a compute-shader pass that reads from an input texture (and the
 * scene depth) and writes its result to an output texture.  The PostProcessingVolume
 * drives a ping-pong between the GBuffer's outputTexture and pingPongTexture so
 * effects can be chained without extra allocations.
 *
 * Lifecycle
 * ---------
 *  1. PostProcessingVolume calls initialize(device, gbuffer) once after the device
 *     is available — create pipelines, buffers, and bind groups here.
 *  2. render() is called every frame with the current input/output pair.
 *  3. resize() is called whenever the viewport changes — recreate size-dependent
 *     resources (bind groups that reference textures, params uniforms, etc.).
 *  4. destroy() releases all GPU resources.
 *
 * Chain hooks (the Rust engine's `PostProcessingEffect`, `postprocessing/effect.rs`), with
 * defaults so an effect overrides only what it needs: `isActive` skips the effect for a frame,
 * `wantsJitter` asks for a sub-pixel jittered projection (TAA), `upscalesToDisplay` marks a
 * temporal upscaler.
 */
abstract class PostProcessingEffect {
    public initialized: boolean = false;

    /**
     * One-time GPU resource creation.
     * @param device  - The WebGPU device.
     * @param gbuffer - The GBuffer whose textures the effect may reference.
     * @param camera  - The active camera (for projection/view data).
     */
    abstract initialize(device: GPUDevice, gbuffer: GBuffer, camera: Camera): void;

    /**
     * Execute the effect for one frame.
     *
     * @param commandEncoder - The command encoder to record compute commands into.
     * @param input          - Source texture (previous effect's output or scene color).
     * @param depth          - The GBuffer depth texture (depth32float).
     * @param output         - Destination storage texture to write results into.
     * @param camera         - Active camera for per-frame projection data.
     * @param width          - Current render width in pixels.
     * @param height         - Current render height in pixels.
     * @param emissive       - The GBuffer's emissive texture.
     * @param gbuffer        - The whole GBuffer (normal, albedo, background...), as the Rust
     *                         engine passes it to every effect.
     */
    abstract render(
        commandEncoder: GPUCommandEncoder,
        input: GPUTexture,
        depth: GPUTexture,
        output: GPUTexture,
        camera: Camera,
        width: number,
        height: number,
        emissive?: GPUTexture,
        gbuffer?: GBuffer
    ): void;

    /**
     * Called when the viewport size changes.  Re-create any bind groups or
     * size-dependent uniform data.
     */
    abstract resize(width: number, height: number, gbuffer: GBuffer): void;

    /** Release all GPU resources owned by this effect. */
    abstract destroy(): void;

    /**
     * Whether the effect runs this frame. An inactive effect is skipped: the next effect reads
     * what it would have read, and it costs nothing (a toggle for expensive effects).
     */
    isActive(): boolean {
        return true;
    }

    /** Whether the effect wants the scene rendered with a sub-pixel jittered projection (TAA). */
    wantsJitter(): boolean {
        return false;
    }

    /**
     * Whether the effect reads its input at the GBuffer's size and writes its output at the
     * display size (a temporal upscaler); every effect after it runs at the display size. The two
     * sizes are the same until the renderer has a render scale below 1.
     */
    upscalesToDisplay(): boolean {
        return false;
    }
}

export { PostProcessingEffect };
