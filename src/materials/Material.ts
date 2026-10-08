import { BindGroupDescriptor, BindableGroup } from "./BindableGroup";
import { parseIncludes } from "./shaders/ShaderUtils";
import { GBuffer } from "../postprocessing/GBuffer";
// the stock materials construct Materials only when called, so this cycle is safe
import { GradientSkyOptions, StandardLitOptions, emissive, gradientSky, standardLit } from "./StandardLit";
import { basicInstanced, basicLit } from "./Stock";

/**
 * Configuration of a render material (Rust `MaterialOptions`).
 */
export interface MaterialOptions {
    /** The material's own bindings (group 0). */
    bindings?: BindGroupDescriptor[];
    /** Whether the material is transparent: blends its first target, does not write depth and does not cull. Default false. */
    transparent?: boolean;
    /** Drawn after the opaque objects and the background copy (`GBuffer.backgroundTexture`), for screen-space refraction. Default false. */
    transmissive?: boolean;
    /** Whether depth is written. Default true (false for transparent materials unless set). */
    depthWriteEnabled?: boolean;
    /** Depth comparison. Default 'less'. */
    depthCompare?: GPUCompareFunction;
    /** Face culling. Default 'back'. */
    cullMode?: GPUCullMode;
    /** Primitive topology. Default 'triangle-list'. */
    topology?: GPUPrimitiveTopology;
    /** Depth format of the pipeline `initialize` builds for the canvas. Default 'depth24plus'. */
    depthStencilFormat?: GPUTextureFormat;
    /** The shader writes emissive at @location(1). */
    outputsEmissive?: boolean;
    /**
     * Number of colour targets the fragment shader writes: the first N of the pass's targets.
     * The others get an empty write mask, as WebGPU requires for targets without a shader output.
     * Default: every target of the pass (Rust defaults to 1, or 2 with `outputsEmissive`, so its
     * GBuffer materials set 4, as `GBUFFER_OUT_WGSL` expects).
     */
    mrtOutputCount?: number;
    /**
     * Fragment entry point for shadow (depth-only) passes, for alpha-tested casters such as
     * foliage cards: it runs with no colour targets and should `discard` cut-out texels, and it
     * must not use group 3. Unset, shadow depth comes from `vertex_main` alone.
     */
    shadowFragmentEntry?: string;
    /**
     * The fragment shader also writes screen-space motion at @location(4)
     * (`GBuffer.VELOCITY_TARGET`), so a velocity pass can redraw the material keeping only that
     * output (`getVelocityPipeline`).
     */
    outputsVelocity?: boolean;
    /**
     * Fragment entry point for voxel GI's mesh voxelizer (`MeshVoxelizer`), for a surface whose
     * albedo or emission varies, such as a textured one: it takes the material's vertex outputs
     * and the `front_facing` builtin and calls `kansei_voxel_write` from `VOXEL_WRITE_WGSL`
     * (prepend it to the shader) with what the surface reflects and emits there. Group 3 is the
     * voxelizer's in that pass, so it must not read the shadow group. Unset, the voxelizer uses
     * the renderable's constant `GiSurface`.
     */
    voxelFragmentEntry?: string;
    /** Label of the material's GPU objects. */
    label?: string;
}

/** A depth bias for `getDepthPipeline` (`GPUDepthStencilState`'s bias fields). */
export interface DepthBias {
    constant?: number;
    slopeScale?: number;
    clamp?: number;
}

/**
 * A key for a set of vertex buffer layouts: two geometries whose layouts read the same get the
 * same pipelines. Cached by layout list (geometries fill theirs once, when initialized).
 */
const layoutKeys = new WeakMap<object, { count: number, key: string }>();
function vertexLayoutKey(layouts: Iterable<GPUVertexBufferLayout | null>): string {
    const list = Array.isArray(layouts) ? layouts as (GPUVertexBufferLayout | null)[] : [...layouts];
    const cached = layoutKeys.get(layouts);
    if (cached && cached.count === list.length) return cached.key;
    const key = list.map((l) => l
        ? `${l.arrayStride}/${l.stepMode ?? 'vertex'}/` + [...l.attributes].map((a) => `${a.shaderLocation}:${a.offset}:${a.format}`).join(',')
        : '-').join('|');
    layoutKeys.set(layouts, { count: list.length, key });
    return key;
}

/**
 * Represents a material used in rendering, encapsulating shader modules and pipeline configurations.
 *
 * Every pipeline runs the shader's own `vertex_main`, so passes that redraw the scene from
 * another view (shadows, velocity) place vertices as the colour pass does: instancing and vertex
 * animation included. Vertex stages must therefore not read group 3, which depth pipelines leave out.
 */
class Material {

    public shaderRenderModule?: GPUShaderModule;
    public pipeline?: GPURenderPipeline;
    public initialized: boolean = false;
    public uuid: string;
    public label: string;
    public transparent: boolean = false;
    public outputsEmissive: boolean = false;
    /** When true, the renderer draws this object AFTER opaque geometry and AFTER
     *  snapshotting the opaque result into GBuffer.backgroundTexture — so a
     *  downstream post-processing effect can sample the undistorted background
     *  to compute screen-space refraction. */
    public transmissive: boolean = false;
    /** See `MaterialOptions.outputsVelocity`. */
    public outputsVelocity: boolean = false;

    // Pipelines by target formats, sample count, depth format and vertex layout, so the same
    // material can draw into several passes (canvas, GBuffer) and geometries.
    private _pipelineCache: Map<string, GPURenderPipeline> = new Map();
    // Depth-only pipelines by depth format, vertex layout and bias.
    private _depthPipelineCache: Map<string, GPURenderPipeline> = new Map();
    // Velocity-pass pipelines by vertex layout and sample count.
    private _velocityPipelineCache: Map<string, GPURenderPipeline> = new Map();
    // Voxelization pipelines by voxelizer and vertex layout, and their layouts by voxelizer.
    private _voxelPipelineCache: Map<string, GPURenderPipeline> = new Map();
    private _voxelPipelineLayouts: Map<number, GPUPipelineLayout> = new Map();
    // Groups 0-2 only: shadow passes render into textures that group 3 samples.
    private _depthPipelineLayout?: GPUPipelineLayout;

    private bindableGroup: BindableGroup;
    private depthWriteEnabled: boolean = true;
    private depthCompare: GPUCompareFunction = 'less';
    private cullMode: GPUCullMode = 'back';
    private topology: GPUPrimitiveTopology = 'triangle-list';
    private depthStencilFormat: GPUTextureFormat = 'depth24plus';

    /**
     * Constructs a new Material instance.
     *
     * @param shaderCode - The shader code to be used for this material: `vertex_main` and `fragment_main`.
     * @param options - Configuration options for the material (see `MaterialOptions`).
     */
    constructor(
        private shaderCode: string,
        private options: MaterialOptions
    ) {
        this.bindableGroup = new BindableGroup(this.options.bindings || []);
        this.uuid = crypto.randomUUID();
        this.label = this.options.label ?? 'Material';

        this.transparent = this.options.transparent || false;
        this.transmissive = this.options.transmissive || false;
        this.outputsEmissive = this.options.outputsEmissive || false;
        this.outputsVelocity = this.options.outputsVelocity || false;
        this.depthWriteEnabled = this.options.depthWriteEnabled ?? true;
        this.depthCompare = this.options.depthCompare || 'less';
        this.cullMode = this.options.cullMode || 'back';
        this.topology = this.options.topology || 'triangle-list';
        this.depthStencilFormat = this.options.depthStencilFormat || 'depth24plus';
    }

    /**
     * The standard lit material (Rust `Material::standard_lit`): a GGX / Lambert surface lit by
     * the scene's directional lights (the first through the single shadow map of
     * `Renderer.enableShadows`), its point lights (the one `enablePointShadows` renders through
     * its cube shadow), the spot lights and cascades group 3 carries, a hemisphere of sky and its
     * own emission. It writes the GBuffer's four targets, so draw it through a
     * `PostProcessingVolume`.
     */
    public static standardLit(label: string, options: StandardLitOptions = {}): Material {
        return standardLit(label, options);
    }

    /**
     * An unlit material emitting `radiance` (cd/m²) into the GBuffer: lamp heads, windows, a
     * backdrop. Shorthand for `standardLit` with a black base and that emission.
     */
    public static emissive(label: string, radiance: [number, number, number]): Material {
        return emissive(label, radiance);
    }

    /**
     * A cheap sky (Rust `Material::gradient_sky`): put it on a large sphere around the scene
     * (`SphereGeometry`, with `castShadow = false`) and it shows `options`' gradient by direction
     * from the sphere's centre, into the GBuffer.
     */
    public static gradientSky(label: string, options: GradientSkyOptions = {}): Material {
        return gradientSky(label, options);
    }

    /**
     * A `BASIC_LIT_WGSL` material (Rust `Material::basic_lit`): `color` (rgba, linear) under the
     * scene's lights, with a Blinn-Phong highlight of `specular` (rgb; `a` is the shininess /
     * 256). Forward, one colour output: draw it with `Renderer.render`. `options` adds to the
     * material's (cull mode and the like); its bindings are the colour's.
     */
    public static basicLit(label: string, color: [number, number, number, number], specular: [number, number, number, number], options: MaterialOptions = {}): Material {
        return basicLit(label, color, specular, options);
    }

    /**
     * A `BASIC_INSTANCED_WGSL` material of one `color` (rgba, linear), for an `InstancedGeometry`
     * whose instance matrices sit at vertex locations 3-6 (Rust `Material::basic_instanced`).
     */
    public static basicInstanced(label: string, color: [number, number, number, number], options: MaterialOptions = {}): Material {
        return basicInstanced(label, color, options);
    }

    /** See `MaterialOptions.mrtOutputCount`. */
    public get mrtOutputCount(): number | undefined { return this.options.mrtOutputCount; }

    /** See `MaterialOptions.shadowFragmentEntry`. */
    public get shadowFragmentEntry(): string | undefined { return this.options.shadowFragmentEntry; }

    /** See `MaterialOptions.voxelFragmentEntry`. */
    public get voxelFragmentEntry(): string | undefined { return this.options.voxelFragmentEntry; }

    /**
     * Creates a shader module from the provided shader code.
     *
     * @param gpuDevice - The GPU device used to create the shader module.
     */
    private createShaderModule(gpuDevice: GPUDevice) {
        this.shaderRenderModule = gpuDevice.createShaderModule({
            label: `${this.label}/Shader`,
            code: parseIncludes(this.shaderCode)
        });
    }

    /**
     * Ensures the shader module and shared bind group layouts are created exactly once.
     */
    private _ensureSharedResources(gpuDevice: GPUDevice): void {
        if (!this.shaderRenderModule) {
            this.createShaderModule(gpuDevice);
        }

        // Keyed on the depth layout: getBindGroup may have made the group 0 layout (and a
        // group-0-only pipeline layout) first, when a shadow pass binds the material before any
        // pipeline of it exists.
        if (!this._depthPipelineLayout) {
            this.bindableGroup.createRenderingBindGroupLayout(gpuDevice);
            if (!this.bindableGroup.bindGroupLayout) this.bindableGroup.createBindGroupLayout(gpuDevice);

            this.bindableGroup.pipelineBindGroupLayout = gpuDevice.createPipelineLayout({
                label: `${this.label}/PipelineLayout`,
                bindGroupLayouts: [
                    this.bindableGroup.bindGroupLayout!,
                    this.bindableGroup.cameraBindablesGroupLayout!,
                    this.bindableGroup.meshBindablesGroupLayout!,
                    this.bindableGroup.shadowBindablesGroupLayout!,
                ]
            });
            this._depthPipelineLayout = gpuDevice.createPipelineLayout({
                label: `${this.label}/DepthPipelineLayout`,
                bindGroupLayouts: [
                    this.bindableGroup.bindGroupLayout!,
                    this.bindableGroup.cameraBindablesGroupLayout!,
                    this.bindableGroup.meshBindablesGroupLayout!,
                ]
            });
        }
    }

    /**
     * Builds the colour pipeline for `colorFormats`: blending on the first target when
     * transparent, and empty write masks past `mrtOutputCount`.
     */
    private _buildPipeline(
        gpuDevice: GPUDevice,
        vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null>,
        colorFormats: GPUTextureFormat[],
        sampleCount: number,
        depthFormat: GPUTextureFormat,
    ): GPURenderPipeline {
        const outputCount = this.options.mrtOutputCount ?? colorFormats.length;
        const targets: GPUColorTargetState[] = colorFormats.map((format, i) => {
            const target: GPUColorTargetState = {
                format,
                // Targets the shader doesn't write must have an empty write mask, or WebGPU
                // validation fails ("no corresponding fragment stage output").
                writeMask: i < outputCount ? GPUColorWrite.ALL : 0,
            };
            if (i === 0 && this.transparent) {
                target.blend = {
                    color: {
                        operation: 'add',
                        srcFactor: 'src-alpha',
                        dstFactor: 'one-minus-src-alpha'
                    },
                    alpha: {
                        operation: 'add',
                        srcFactor: 'one',
                        dstFactor: 'one-minus-src-alpha'
                    }
                };
            }
            return target;
        });

        const renderPipelineDescriptor: GPURenderPipelineDescriptor = {
            layout: this.bindableGroup.pipelineBindGroupLayout!,
            label: `${this.label}/Pipeline`,
            multisample: { count: sampleCount },
            vertex: {
                module: this.shaderRenderModule!,
                entryPoint: 'vertex_main',
                buffers: vertexBuffersDescriptors
            } as GPUVertexState,
            fragment: {
                module: this.shaderRenderModule!,
                entryPoint: 'fragment_main',
                targets,
            } as GPUFragmentState,
            primitive: {
                topology: this.topology,
                cullMode: this.transparent ? 'none' : this.cullMode,
            } as GPUPrimitiveState,
            depthStencil: {
                depthWriteEnabled: this.options.depthWriteEnabled ?? (this.transparent ? false : this.depthWriteEnabled),
                depthCompare: this.depthCompare,
                format: depthFormat,
            },
        };
        return gpuDevice.createRenderPipeline(renderPipelineDescriptor);
    }

    /**
     * Returns a cached pipeline for the given configuration, creating it if it does not exist.
     *
     * This allows the same material to be used in multiple render passes that differ in
     * color format, sample count, or depth format — e.g. the canvas MSAA pass and an
     * off-screen GBuffer pass — and with geometries of different vertex layouts, without
     * re-compiling the shader.
     *
     * @param gpuDevice - The GPU device.
     * @param vertexBuffersDescriptors - Vertex buffer layouts of the geometry drawn.
     * @param colorFormat - Target color attachment format.
     * @param sampleCount - MSAA sample count of the render pass.
     * @param depthFormat - Depth-stencil attachment format. Defaults to 'depth24plus'.
     * @param colorTargetCount - Number of targets of `colorFormat`, when `colorFormats` is not given.
     * @param colorFormats - The format of each colour target (MRT).
     */
    public getPipelineForConfig(
        gpuDevice: GPUDevice,
        vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null>,
        colorFormat: GPUTextureFormat,
        sampleCount: number,
        depthFormat: GPUTextureFormat = 'depth24plus',
        colorTargetCount: number = 1,
        colorFormats?: GPUTextureFormat[]
    ): GPURenderPipeline {
        this._ensureSharedResources(gpuDevice);
        const formats = colorFormats && colorFormats.length > 0
            ? colorFormats
            : new Array<GPUTextureFormat>(colorTargetCount).fill(colorFormat);
        const key = `${formats.join(',')}:${sampleCount}:${depthFormat}:${vertexLayoutKey(vertexBuffersDescriptors)}`;
        let pipeline = this._pipelineCache.get(key);
        if (!pipeline) {
            pipeline = this._buildPipeline(gpuDevice, vertexBuffersDescriptors, formats, sampleCount, depthFormat);
            this._pipelineCache.set(key, pipeline);
        }
        return pipeline;
    }

    /**
     * Returns the depth-only pipeline shadow passes draw this material with (Rust
     * `get_depth_pipeline`): its own `vertex_main`, so instancing and vertex animation cast
     * matching shadows, plus `shadowFragmentEntry` if set. Groups 0-2 only: bind the material's
     * group, a camera group (group 1 layout) holding the light's view and projection, and the
     * mesh group. Depth is written and tested with 'less-equal'.
     *
     * @param gpuDevice - The GPU device.
     * @param vertexBuffersDescriptors - Vertex buffer layouts of the geometry drawn.
     * @param depthFormat - Format of the shadow depth target.
     * @param bias - Depth bias of the pass (none by default; ignored for point and line topologies).
     */
    public getDepthPipeline(
        gpuDevice: GPUDevice,
        vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null>,
        depthFormat: GPUTextureFormat,
        bias: DepthBias = {},
    ): GPURenderPipeline {
        this._ensureSharedResources(gpuDevice);
        // WebGPU allows a depth bias on triangle topologies only.
        const triangles = this.topology === 'triangle-list' || this.topology === 'triangle-strip';
        const constant = triangles ? bias.constant ?? 0 : 0;
        const slopeScale = triangles ? bias.slopeScale ?? 0 : 0;
        const clamp = triangles ? bias.clamp ?? 0 : 0;
        const key = `${depthFormat}:${constant}:${slopeScale}:${clamp}:${vertexLayoutKey(vertexBuffersDescriptors)}`;
        let pipeline = this._depthPipelineCache.get(key);
        if (!pipeline) {
            const entry = this.options.shadowFragmentEntry;
            pipeline = gpuDevice.createRenderPipeline({
                label: `${this.label}/DepthPipeline`,
                layout: this._depthPipelineLayout!,
                vertex: {
                    module: this.shaderRenderModule!,
                    entryPoint: 'vertex_main',
                    buffers: vertexBuffersDescriptors,
                },
                fragment: entry ? { module: this.shaderRenderModule!, entryPoint: entry, targets: [] } : undefined,
                primitive: {
                    topology: this.topology,
                    cullMode: this.cullMode,
                },
                depthStencil: {
                    format: depthFormat,
                    depthWriteEnabled: true,
                    depthCompare: 'less-equal',
                    depthBias: constant,
                    depthBiasSlopeScale: slopeScale,
                    depthBiasClamp: clamp,
                },
            });
            this._depthPipelineCache.set(key, pipeline);
        }
        return pipeline;
    }

    /**
     * Returns the pipeline of a velocity pass (Rust `get_velocity_pipeline`): this material's
     * shader with only its @location(4) output kept (a `GBuffer.VELOCITY_FORMAT` target at
     * `GBuffer.VELOCITY_TARGET`, none before it), depth-tested 'less-equal' against the GBuffer's
     * depth without writing it. Mark the position output `@invariant` so both passes produce the
     * same depths; a small bias toward the camera covers compilers that differ.
     *
     * @param gpuDevice - The GPU device.
     * @param vertexBuffersDescriptors - Vertex buffer layouts of the geometry drawn.
     * @param sampleCount - Sample count of the GBuffer depth it tests against. Default 1.
     */
    public getVelocityPipeline(
        gpuDevice: GPUDevice,
        vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null>,
        sampleCount: number = 1,
    ): GPURenderPipeline {
        this._ensureSharedResources(gpuDevice);
        const key = `${sampleCount}:${vertexLayoutKey(vertexBuffersDescriptors)}`;
        let pipeline = this._velocityPipelineCache.get(key);
        if (!pipeline) {
            const targets: (GPUColorTargetState | null)[] = new Array(GBuffer.VELOCITY_TARGET).fill(null);
            targets.push({ format: GBuffer.VELOCITY_FORMAT, writeMask: GPUColorWrite.ALL });
            pipeline = gpuDevice.createRenderPipeline({
                label: `${this.label}/VelocityPipeline`,
                layout: this.bindableGroup.pipelineBindGroupLayout!,
                multisample: { count: sampleCount },
                vertex: {
                    module: this.shaderRenderModule!,
                    entryPoint: 'vertex_main',
                    buffers: vertexBuffersDescriptors,
                },
                fragment: {
                    module: this.shaderRenderModule!,
                    entryPoint: 'fragment_main',
                    targets,
                },
                primitive: {
                    topology: this.topology,
                    cullMode: this.cullMode,
                },
                depthStencil: {
                    format: GBuffer.DEPTH_FORMAT,
                    depthWriteEnabled: false,
                    depthCompare: 'less-equal',
                    depthBias: -4,
                    depthBiasSlopeScale: -1,
                },
            });
            this._velocityPipelineCache.set(key, pipeline);
        }
        return pipeline;
    }

    /**
     * Returns the pipeline voxel GI's mesh voxelizer draws this material with (Rust
     * `get_voxel_pipeline`): its own `vertex_main`, so instancing and vertex animation voxelize as
     * they draw, and its `voxelFragmentEntry` or the voxelizer's `engineFragment`; group 3 is the
     * voxelizer's `voxelBGL`. Every face from every axis (no culling) and no depth, into the
     * voxelizer's masked-off `target` of `sampleCount` samples.
     *
     * @param gpuDevice - The GPU device.
     * @param voxelizer - The voxelizer's id (`MeshVoxelizer.id`): its pipelines are kept apart.
     * @param vertexBuffersDescriptors - Vertex buffer layouts of the geometry drawn.
     * @param voxelBGL - The voxelizer's group 3 layout.
     * @param engineFragment - The voxelizer's own fragment stage, for materials without an entry.
     * @param target - The format of the voxelizer's target.
     * @param sampleCount - Its samples.
     */
    public getVoxelPipeline(
        gpuDevice: GPUDevice,
        voxelizer: number,
        vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null>,
        voxelBGL: GPUBindGroupLayout,
        engineFragment: { module: GPUShaderModule, entryPoint: string },
        target: GPUTextureFormat,
        sampleCount: number,
    ): GPURenderPipeline {
        this._ensureSharedResources(gpuDevice);
        const key = `${voxelizer}:${vertexLayoutKey(vertexBuffersDescriptors)}`;
        let pipeline = this._voxelPipelineCache.get(key);
        if (!pipeline) {
            let layout = this._voxelPipelineLayouts.get(voxelizer);
            if (!layout) {
                layout = gpuDevice.createPipelineLayout({
                    label: `${this.label}/VoxelPipelineLayout`,
                    bindGroupLayouts: [
                        this.bindableGroup.bindGroupLayout!,
                        this.bindableGroup.cameraBindablesGroupLayout!,
                        this.bindableGroup.meshBindablesGroupLayout!,
                        voxelBGL,
                    ],
                });
                this._voxelPipelineLayouts.set(voxelizer, layout);
            }
            const entry = this.options.voxelFragmentEntry;
            const fragment = entry ? { module: this.shaderRenderModule!, entryPoint: entry } : engineFragment;
            pipeline = gpuDevice.createRenderPipeline({
                label: `${this.label}/VoxelPipeline`,
                layout,
                vertex: {
                    module: this.shaderRenderModule!,
                    entryPoint: 'vertex_main',
                    buffers: vertexBuffersDescriptors,
                },
                fragment: { ...fragment, targets: [{ format: target, writeMask: 0 }] },
                // every face from every axis: the far side of a closed mesh is a surface too
                primitive: { topology: this.topology, cullMode: 'none' },
                multisample: { count: sampleCount },
            });
            this._voxelPipelineCache.set(key, pipeline);
        }
        return pipeline;
    }

    /**
     * Initializes the material by creating the shader module, bind group layouts, and render pipeline.
     *
     * @param gpuDevice - The GPU device used for initialization.
     * @param vertexBuffersDescriptors - Descriptors for the vertex buffers.
     * @param presentationFormat - The format of the presentation surface.
     * @param sampleCount - MSAA sample count for the render pass.
     */
    public initialize(gpuDevice: GPUDevice, vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null>, presentationFormat: GPUTextureFormat, sampleCount: number) {
        this._ensureSharedResources(gpuDevice);
        this.pipeline = this.getPipelineForConfig(
            gpuDevice, vertexBuffersDescriptors, presentationFormat, sampleCount, this.depthStencilFormat
        );
        this.initialized = true;
    }

    /**
     * Retrieves the bind group for the material.
     *
     * @param gpuDevice - The GPU device used to get the bind group.
     * @returns The bind group associated with this material.
     */
    public getBindGroup(gpuDevice: GPUDevice): GPUBindGroup {
        this.bindableGroup.getBindGroup(gpuDevice);
        return this.bindableGroup.bindGroup!;
    }

    /**
     * The bind group `getBindGroup` last returned, without updating its resources (a material
     * with an external texture gets a new one each time `getBindGroup` is called).
     */
    public get currentBindGroup(): GPUBindGroup | undefined {
        return this.bindableGroup.bindGroup;
    }
}

export { Material }
