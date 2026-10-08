import { mat4, vec3 } from 'gl-matrix';
import type { Renderable } from '../objects/Renderable';
import type { InstancedGeometry } from '../geometries/InstancedGeometry';
import { GBuffer } from '../postprocessing/GBuffer';
import { CAMERA_TEMPORAL_BYTES, cameraBindGroupLayoutEntries, meshBindGroupLayoutEntries } from '../renderers/SharedLayouts';
import { cpuScope } from '../profiling/Profiler';
import { Impostor, ImpostorOptions, impostorFrameDirection, impostorUpReference } from './Impostor';
import packWgsl from '../../rust/kansei-core/src/shaders/impostor_pack.wgsl?raw';
import mipWgsl from '../../rust/kansei-core/src/shaders/impostor_mip.wgsl?raw';

/** Bytes of `Pack` in impostor_pack.wgsl: one frame. */
const PACK_BYTES = 64;

/**
 * Per-frame uniforms (the view and camera temporal data, the pack parameters) sit this far apart:
 * a multiple of every device's uniform offset alignment.
 */
const SLOT = 256;

/** The GBuffer's colour targets (`Renderer.renderToGBuffer`'s): colour, emissive, normal, albedo. */
const MRT_FORMATS: GPUTextureFormat[] = ['rgba16float', 'rgba16float', 'rgba16float', 'rgba8unorm'];

/** What the bake needs of the renderer. */
export interface ImpostorBakeContext {
    device: GPUDevice;
    /** Group 1 binding 2: the scene lights. */
    lightBuffer: GPUBuffer;
    /** Group 3: shadows and lights, as the scene pass binds them. */
    shadowBindGroup: GPUBindGroup;
}

type Vec3 = [number, number, number];

/**
 * Bakes an octahedral impostor of `parts`, drawn together with their own materials' GBuffer
 * pipelines, from `options.frames` squared directions (see `Impostor`); `Renderer.bakeImpostor`
 * calls it with the renderer's lights and group 3. A port of the Rust `Renderer::bake_impostor`
 * (`impostors/bake.rs`), sharing its pack and mip shaders.
 *
 * Parts may be hidden or not drawn yet; each has one instance buffer at most, drawn with
 * `options.instance`. Materials that light in the GBuffer pass only colour their colour output,
 * which the bake keeps only where they write no albedo.
 */
export function bakeImpostor(context: ImpostorBakeContext, parts: readonly Renderable[], options: ImpostorOptions = {}): Impostor {
    const scope = cpuScope('impostor/bake');
    const n = options.frames ?? 12;
    const size = options.frameSize ?? 128;
    const ss = options.supersample ?? 2;
    const layout = options.layout ?? 'octahedral';
    if (!(n > 0 && size > 0 && (size & (size - 1)) === 0 && ss > 0)) {
        throw new Error('bakeImpostor: frames, a power-of-two frame size and supersampling');
    }
    const { device, lightBuffer, shadowBindGroup } = context;
    for (const r of parts) {
        if (!r.geometry.initialized) r.geometry.initialize(device);
        if (r.geometry.isInstancedGeometry) {
            const geo = r.geometry as InstancedGeometry;
            if (geo.extraBuffers.length > 1) throw new Error('bakeImpostor: impostor parts have one instance buffer at most');
            for (const buffer of geo.extraBuffers) if (!buffer.initialized) buffer.initialize(device);
        }
    }
    let center: Vec3, radius: number, extent: Vec3;
    if (options.bounds) {
        const [lo, hi] = options.bounds;
        center = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5];
        extent = [(hi[0] - lo[0]) * 0.5, (hi[1] - lo[1]) * 0.5, (hi[2] - lo[2]) * 0.5];
        radius = Math.max(Math.hypot(...extent), 1e-6);
    } else {
        [center, radius, extent] = bounds(parts);
    }
    const frameCount = n * n;

    // the atlases, with a mip chain down to a texel per frame
    const levels = Math.log2(size) + 1;
    const atlas = (label: string) => device.createTexture({
        label,
        size: [n * size, n * size],
        mipLevelCount: levels,
        format: Impostor.FORMAT,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    const albedo = atlas('Impostor/Albedo');
    const normalDepth = atlas('Impostor/NormalDepth');
    const mip = (t: GPUTexture, level: number) => t.createView({ baseMipLevel: level, mipLevelCount: 1 });

    // a frame's render targets: the GBuffer's, `supersample` times the frame's size
    const renderSize = size * ss;
    const transient: GPUTexture[] = [];
    const target = (label: string, format: GPUTextureFormat) => {
        const texture = device.createTexture({
            label,
            size: [renderSize, renderSize],
            format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        transient.push(texture);
        return texture.createView();
    };
    const targets = MRT_FORMATS.map((f) => target('Impostor/Target', f));
    const depth = target('Impostor/Depth', GBuffer.DEPTH_FORMAT);

    // per frame: an orthographic camera on the bounding sphere looking at its centre, depth
    // [0, 1] over its diameter; and the pack parameters
    const proj = mat4.orthoZO(mat4.create(), -radius, radius, -radius, radius, 0, 2 * radius);
    const cameras = new ArrayBuffer(frameCount * 2 * SLOT);
    const packs = new ArrayBuffer(frameCount * SLOT);
    const view = mat4.create();
    const right = vec3.create(), up = vec3.create();
    for (let k = 0; k < frameCount; k++) {
        const column = k % n, row = Math.floor(k / n);
        const dir = impostorFrameDirection(n, layout, column, row);
        const upRef = impostorUpReference(dir);
        mat4.lookAt(view, [center[0] + dir[0] * radius, center[1] + dir[1] * radius, center[2] + dir[2] * radius], center, upRef);
        const at = k * 2 * SLOT;
        new Float32Array(cameras, at, 16).set(view);
        // KanseiCameraTemporal: this frame's and last frame's view-projection (the same), no
        // jitter, frame 0
        const temporal = new Float32Array(cameras, at + SLOT, CAMERA_TEMPORAL_BYTES / 4);
        mat4.multiply(temporal.subarray(0, 16), proj, view);
        temporal.copyWithin(16, 0, 16);
        vec3.normalize(right, vec3.cross(right, upRef, dir));
        vec3.cross(up, dir, right);
        const pf = new Float32Array(packs, k * SLOT, PACK_BYTES / 4);
        const pu = new Uint32Array(packs, k * SLOT, PACK_BYTES / 4);
        pf.set(right, 0);
        pf[3] = radius;
        pf.set(up, 4);
        pu[7] = size;
        pf.set(dir, 8);
        pu[11] = ss;
        pu[12] = column;
        pu[13] = row;
    }
    const uniform = (label: string, contents: ArrayBuffer | Float32Array) => {
        const bytes = contents instanceof ArrayBuffer ? new Uint8Array(contents) : new Uint8Array(contents.buffer, contents.byteOffset, contents.byteLength);
        const buffer = device.createBuffer({ label, size: bytes.byteLength, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
        new Uint8Array(buffer.getMappedRange()).set(bytes);
        buffer.unmap();
        return buffer;
    };
    const cameraBuffer = uniform('Impostor/Cameras', cameras);
    const projection = uniform('Impostor/Projection', proj as Float32Array);
    const packBuffer = uniform('Impostor/Pack', packs);
    // the object in place: identity normal and world matrices (this frame's and last)
    const identity = mat4.create() as Float32Array;
    const normalMatrix = uniform('Impostor/NormalMatrix', identity);
    const world = new Float32Array(32);
    world.set(identity, 0);
    world.set(identity, 16);
    const worldBuffer = uniform('Impostor/World', world);
    const meshGroup = device.createBindGroup({
        label: 'Impostor/Mesh',
        layout: device.createBindGroupLayout({ label: 'Impostor/MeshBGL', entries: meshBindGroupLayoutEntries() }),
        entries: [
            { binding: 0, resource: { buffer: normalMatrix, offset: 0, size: 64 } },
            { binding: 1, resource: { buffer: worldBuffer, offset: 0, size: 128 } },
        ],
    });
    const cameraLayout = device.createBindGroupLayout({ label: 'Impostor/CameraBGL', entries: cameraBindGroupLayoutEntries() });
    const cameraGroups: GPUBindGroup[] = [];
    for (let k = 0; k < frameCount; k++) {
        cameraGroups.push(device.createBindGroup({
            label: 'Impostor/Camera',
            layout: cameraLayout,
            entries: [
                { binding: 0, resource: { buffer: cameraBuffer, offset: k * 2 * SLOT, size: 64 } },
                { binding: 1, resource: { buffer: projection } },
                { binding: 2, resource: { buffer: lightBuffer } },
                { binding: 3, resource: { buffer: cameraBuffer, offset: k * 2 * SLOT + SLOT, size: CAMERA_TEMPORAL_BYTES } },
            ],
        }));
    }
    // each part's instance: the given record, in a vertex buffer of its own
    const instances: (GPUBuffer | null)[] = parts.map((r) => {
        if (!r.geometry.isInstancedGeometry || (r.geometry as InstancedGeometry).extraBuffers.length === 0) return null;
        const stride = (r.geometry as InstancedGeometry).extraBuffers[0].stride ?? 0;
        const record = options.instance;
        if (!record || record.byteLength < stride) {
            throw new Error(`ImpostorOptions.instance: one record of the parts' instance layout (${stride} bytes)`);
        }
        const buffer = device.createBuffer({ label: 'Impostor/Instance', size: Math.ceil(record.byteLength / 4) * 4, usage: GPUBufferUsage.VERTEX, mappedAtCreation: true });
        new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(record.buffer, record.byteOffset, record.byteLength));
        buffer.unmap();
        return buffer;
    });

    const packPipeline = device.createComputePipeline({
        label: 'Impostor/Pack',
        layout: 'auto',
        compute: { module: device.createShaderModule({ label: 'Impostor/Pack', code: packWgsl }), entryPoint: 'main' },
    });
    const albedoMip0 = mip(albedo, 0), normalDepthMip0 = mip(normalDepth, 0);
    const packGroups: GPUBindGroup[] = [];
    for (let k = 0; k < frameCount; k++) {
        packGroups.push(device.createBindGroup({
            label: 'Impostor/Pack',
            layout: packPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: targets[0] },
                { binding: 1, resource: targets[2] },
                { binding: 2, resource: targets[3] },
                { binding: 3, resource: depth },
                { binding: 4, resource: albedoMip0 },
                { binding: 5, resource: normalDepthMip0 },
                { binding: 6, resource: { buffer: packBuffer, offset: k * SLOT, size: PACK_BYTES } },
            ],
        }));
    }

    const encoder = device.createCommandEncoder({ label: 'Impostor/Bake' });
    const transparent = { r: 0, g: 0, b: 0, a: 0 };
    for (let k = 0; k < frameCount; k++) {
        const pass = encoder.beginRenderPass({
            label: 'Impostor/Frame',
            colorAttachments: targets.map((view) => ({ view, clearValue: transparent, loadOp: 'clear' as GPULoadOp, storeOp: 'store' as GPUStoreOp })),
            depthStencilAttachment: { view: depth, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
        });
        pass.setBindGroup(1, cameraGroups[k]);
        pass.setBindGroup(2, meshGroup, [0, 0]);
        pass.setBindGroup(3, shadowBindGroup);
        parts.forEach((r, i) => {
            const geometry = r.geometry;
            if (!geometry.vertexBuffer || !geometry.indexBuffer) return;
            const pipeline = r.material.getPipelineForConfig(device, geometry.vertexBuffersDescriptors, MRT_FORMATS[0], 1, GBuffer.DEPTH_FORMAT, MRT_FORMATS.length, MRT_FORMATS);
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, r.material.getBindGroup(device));
            pass.setVertexBuffer(0, geometry.vertexBuffer);
            const instance = instances[i];
            if (instance) pass.setVertexBuffer(1, instance);
            pass.setIndexBuffer(geometry.indexBuffer, geometry.indexFormat!);
            pass.drawIndexed(geometry.vertexCount, 1);
        });
        pass.end();
        const compute = encoder.beginComputePass({ label: 'Impostor/Pack' });
        compute.setPipeline(packPipeline);
        compute.setBindGroup(0, packGroups[k]);
        compute.dispatchWorkgroups(Math.ceil(size / 8), Math.ceil(size / 8), 1);
        compute.end();
    }

    // the mip chain
    const mipPipeline = device.createComputePipeline({
        label: 'Impostor/Mip',
        layout: 'auto',
        compute: { module: device.createShaderModule({ label: 'Impostor/Mip', code: mipWgsl }), entryPoint: 'main' },
    });
    for (let level = 1; level < levels; level++) {
        const views = [mip(albedo, level - 1), mip(normalDepth, level - 1), mip(albedo, level), mip(normalDepth, level)];
        const group = device.createBindGroup({
            label: 'Impostor/Mip',
            layout: mipPipeline.getBindGroupLayout(0),
            entries: views.map((resource, binding) => ({ binding, resource })),
        });
        const side = (n * size) >> level;
        const compute = encoder.beginComputePass({ label: 'Impostor/Mip' });
        compute.setPipeline(mipPipeline);
        compute.setBindGroup(0, group);
        compute.dispatchWorkgroups(Math.ceil(side / 8), Math.ceil(side / 8), 1);
        compute.end();
    }
    device.queue.submit([encoder.finish()]);

    // freed once the bake has run
    for (const texture of transient) texture.destroy();
    for (const buffer of [cameraBuffer, projection, packBuffer, normalMatrix, worldBuffer, ...instances]) buffer?.destroy();
    scope?.end();
    return new Impostor(n, size, layout, center, radius, extent, albedo, normalDepth);
}

/**
 * The box round the parts' vertex positions: its centre, the distance of the farthest from it,
 * and its half size.
 */
function bounds(parts: readonly Renderable[]): [Vec3, number, Vec3] {
    const lo: Vec3 = [Infinity, Infinity, Infinity], hi: Vec3 = [-Infinity, -Infinity, -Infinity];
    const each = (visit: (x: number, y: number, z: number) => void) => {
        for (const r of parts) {
            const vertices = r.geometry.vertices;
            const layout = [...r.geometry.vertexBuffersDescriptors][0];
            if (!vertices || !layout) continue;
            const stride = Number(layout.arrayStride) / 4;
            for (let i = 0; i + 2 < vertices.length; i += stride) visit(vertices[i], vertices[i + 1], vertices[i + 2]);
        }
    };
    each((x, y, z) => {
        lo[0] = Math.min(lo[0], x); lo[1] = Math.min(lo[1], y); lo[2] = Math.min(lo[2], z);
        hi[0] = Math.max(hi[0], x); hi[1] = Math.max(hi[1], y); hi[2] = Math.max(hi[2], z);
    });
    if (!(lo[0] <= hi[0])) throw new Error('bakeImpostor: impostor parts have vertices');
    const center: Vec3 = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5];
    let radius = 0;
    each((x, y, z) => { radius = Math.max(radius, Math.hypot(x - center[0], y - center[1], z - center[2])); });
    return [center, Math.max(radius, 1e-6), [(hi[0] - lo[0]) * 0.5, (hi[1] - lo[1]) * 0.5, (hi[2] - lo[2]) * 0.5]];
}
