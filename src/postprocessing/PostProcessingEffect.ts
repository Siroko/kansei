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
     */
    abstract render(
        commandEncoder: GPUCommandEncoder,
        input: GPUTexture,
        depth: GPUTexture,
        output: GPUTexture,
        camera: Camera,
        width: number,
        height: number
    ): void;

    /**
     * Called when the viewport size changes.  Re-create any bind groups or
     * size-dependent uniform data.
     */
    abstract resize(width: number, height: number, gbuffer: GBuffer): void;

    /** Release all GPU resources owned by this effect. */
    abstract destroy(): void;
}

export { PostProcessingEffect };
