import { drawGeometry } from '../culling/InstanceCulling';
import type { Renderable } from '../objects/Renderable';
import type { DepthBias, Material } from '../materials/Material';
import type { SpotShadowSlot } from '../lights/SpotLightsGpu';
import { gpuPass } from '../profiling/Profiler';
import { BindGroupSlot, CAMERA_TEMPORAL_BYTES, LIGHT_UNIFORM_BYTES, cameraBindGroupLayoutEntries } from '../renderers/SharedLayouts';

/**
 * Perspective shadow maps for spot lights (Rust `shadows::SpotShadowAtlas`): one layer of a
 * depth-texture array per shadowed light, rendered each frame by the renderer through the
 * casters' own vertex shaders (`Material.getDepthPipeline`). Create it with
 * `Renderer.enableSpotShadows`; materials sample it through `SPOT_LIGHTS_WGSL`.
 */
class SpotShadowAtlas {
    static readonly FORMAT: GPUTextureFormat = 'depth32float';
    /** Depth bias of the shadow pipelines (the shaders add a normal offset on top). */
    static readonly DEPTH_BIAS: DepthBias = { constant: 0, slopeScale: 1.5, clamp: 0 };

    readonly resolution: number;
    readonly layers: number;
    readonly texture: GPUTexture;
    /** All layers, for sampling (`texture_depth_2d_array`). */
    readonly arrayView: GPUTextureView;
    // One view per layer, for rendering.
    private readonly _layerViews: GPUTextureView[];
    // One camera group per layer (group 1 layout): the light's view and projection, then the
    // scene lights and temporal data, which depth pipelines do not read (shared, zeroed).
    private readonly _cameraBuffers: GPUBuffer[] = [];
    private readonly _viewBuffers: GPUBuffer[];
    private readonly _projectionBuffers: GPUBuffer[];
    private readonly _cameraBindGroups: GPUBindGroup[];
    private readonly _device: GPUDevice;

    constructor(device: GPUDevice, resolution: number, layers: number) {
        this._device = device;
        this.resolution = resolution;
        this.layers = Math.max(layers, 1);
        this.texture = device.createTexture({
            label: 'SpotShadowAtlas',
            size: [resolution, resolution, this.layers],
            format: SpotShadowAtlas.FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.arrayView = this.texture.createView({ label: 'SpotShadowAtlas/Array', dimension: '2d-array' });
        this._layerViews = Array.from({ length: this.layers }, (_, layer) => this.texture.createView({
            label: 'SpotShadowAtlas/Layer',
            dimension: '2d',
            baseArrayLayer: layer,
            arrayLayerCount: 1,
        }));

        const buffer = (label: string, size: number) => {
            const b = device.createBuffer({ label, size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            this._cameraBuffers.push(b);
            return b;
        };
        const lights = buffer('SpotShadowAtlas/Lights', LIGHT_UNIFORM_BYTES);
        const temporal = buffer('SpotShadowAtlas/Temporal', CAMERA_TEMPORAL_BYTES);
        const layout = device.createBindGroupLayout({ label: 'SpotShadowAtlas/Camera BGL', entries: cameraBindGroupLayoutEntries() });
        this._viewBuffers = this._layerViews.map((_, layer) => buffer(`SpotShadowAtlas/View${layer}`, 64));
        this._projectionBuffers = this._layerViews.map((_, layer) => buffer(`SpotShadowAtlas/Projection${layer}`, 64));
        this._cameraBindGroups = this._layerViews.map((_, layer) => device.createBindGroup({
            label: `SpotShadowAtlas/Camera${layer}`,
            layout,
            entries: [this._viewBuffers[layer], this._projectionBuffers[layer], lights, temporal]
                .map((b, binding) => ({ binding, resource: { buffer: b } })),
        }));
    }

    /**
     * Renders each slot's layer: clears it and draws the visible `castShadow` renderables among
     * `objects` from the slot's light, each with its matrices in `meshBindGroup` (group 2) at
     * `meshOffset(renderable)`. Renderables with `instanceCulling` draw the instances culled for
     * the layer's light (cull view `firstCullView + layer`), so casters outside the camera still
     * cast; the others draw every instance.
     */
    encode(
        encoder: GPUCommandEncoder,
        slots: readonly SpotShadowSlot[],
        objects: readonly Renderable[],
        meshBindGroup: GPUBindGroup,
        meshOffset: (renderable: Renderable) => number,
        firstCullView: number,
    ): void {
        const device = this._device;
        const queue = device.queue;
        for (const slot of slots) {
            // Each layer has its own buffers, so these writes do not overwrite one another.
            queue.writeBuffer(this._viewBuffers[slot.layer], 0, slot.view as Float32Array);
            queue.writeBuffer(this._projectionBuffers[slot.layer], 0, slot.projection as Float32Array);
        }

        for (const slot of slots) {
            const pass = encoder.beginRenderPass({
                label: 'Renderer/SpotShadowPass',
                timestampWrites: gpuPass('Renderer/SpotShadowPass'),
                colorAttachments: [],
                depthStencilAttachment: {
                    view: this._layerViews[slot.layer],
                    depthClearValue: 1.0,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store',
                },
            });
            pass.setBindGroup(BindGroupSlot.Camera, this._cameraBindGroups[slot.layer]);

            let pipeline: GPURenderPipeline | null = null;
            let material: Material | null = null;
            let layouts: Iterable<GPUVertexBufferLayout | null> | null = null;
            let vertexBuffer: GPUBuffer | null = null;
            let indexBuffer: GPUBuffer | null = null;
            for (const obj of objects) {
                const geometry = obj.geometry;
                if (!obj.castShadow || !geometry.initialized) continue;
                if (obj.material !== material || geometry.vertexBuffersDescriptors !== layouts) {
                    if (obj.material !== material) {
                        // the renderer updated it this frame
                        pass.setBindGroup(BindGroupSlot.Material, obj.material.currentBindGroup ?? obj.material.getBindGroup(device));
                    }
                    material = obj.material;
                    layouts = geometry.vertexBuffersDescriptors;
                    const next = material.getDepthPipeline(device, layouts, SpotShadowAtlas.FORMAT, SpotShadowAtlas.DEPTH_BIAS);
                    if (next !== pipeline) {
                        pass.setPipeline(next);
                        pipeline = next;
                    }
                }
                const offset = meshOffset(obj);
                pass.setBindGroup(BindGroupSlot.Mesh, meshBindGroup, [offset, offset]);
                if (geometry.vertexBuffer !== vertexBuffer) {
                    pass.setVertexBuffer(0, geometry.vertexBuffer!);
                    vertexBuffer = geometry.vertexBuffer!;
                }
                if (geometry.indexBuffer !== indexBuffer) {
                    pass.setIndexBuffer(geometry.indexBuffer!, geometry.indexFormat!);
                    indexBuffer = geometry.indexBuffer!;
                }
                // culled against this light's frustum, not the camera's
                drawGeometry(pass, geometry, obj.instanceCulling?.view(firstCullView + slot.layer) ?? null);
            }
            pass.end();
        }
    }

    destroy(): void {
        this.texture.destroy();
        for (const buffer of this._cameraBuffers) buffer.destroy();
    }
}

export { SpotShadowAtlas };
