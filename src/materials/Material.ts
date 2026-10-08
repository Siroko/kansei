import { BindGroupDescriptor, BindableGroup } from "./BindableGroup";
import type { IBindable } from "../buffers/IBindable";
import { parseIncludes } from "./shaders/ShaderUtils";
import { GBuffer } from "../postprocessing/GBuffer";
// the stock materials construct Materials only when called, so this cycle is safe
import { GradientSkyOptions, StandardLitOptions, emissive, gradientSky, standardLit } from "./StandardLit";
import { basicInstanced, basicLit } from "./Stock";
import { CLUSTER_VERTEX_ENTRY, InstanceLayout, clusterVertexFunction, clusterVertexStage } from "../clusters/vertexStage";
import { CLUSTER_DEBUG_FRAGMENT_ENTRY, CLUSTER_DEBUG_VERTEX_ENTRY, clusterDebugBindGroupLayoutEntries, clusterDebugWgsl } from "../clusters/ClusterDebug";
import { clusterMeshBindGroupLayoutEntries } from "../renderers/SharedLayouts";

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
    // Cluster LOD (`Renderable.clusters`): the generated vertex stage for an instance layout (or
    // why there is none), its pipelines by pass, and their layouts (group 2 the cluster mesh's).
    private _clusterStage: { key: string; module: GPUShaderModule | null; error: string | null } | null = null;
    private _clusterPipelineCache: Map<string, GPURenderPipeline> = new Map();
    private _clusterDepthPipelineCache: Map<string, GPURenderPipeline> = new Map();
    private _clusterVelocityPipelineCache: Map<number, GPURenderPipeline> = new Map();
    private _clusterVoxelPipelineCache: Map<number, GPURenderPipeline> = new Map();
    // The cluster debug view's pipelines (`Renderer.setClusterDebug`), by instance layout and targets.
    private _clusterDebugPipelineCache: Map<string, GPURenderPipeline> = new Map();
    private _clusterPipelineLayout?: GPUPipelineLayout;
    private _clusterDepthPipelineLayout?: GPUPipelineLayout;

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

    /** The faces its pipelines cull (`MaterialOptions.cullMode`; transparent materials cull none). */
    public get culledFaces(): GPUCullMode { return this.transparent ? 'none' : this.cullMode; }

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
        stage: { layout: GPUPipelineLayout, module: GPUShaderModule, entryPoint: string, label: string } | null = null,
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

        const module = stage?.module ?? this.shaderRenderModule!;
        const renderPipelineDescriptor: GPURenderPipelineDescriptor = {
            layout: stage?.layout ?? this.bindableGroup.pipelineBindGroupLayout!,
            label: `${this.label}/${stage?.label ?? 'Pipeline'}`,
            multisample: { count: sampleCount },
            vertex: {
                module,
                entryPoint: stage?.entryPoint ?? 'vertex_main',
                buffers: vertexBuffersDescriptors
            } as GPUVertexState,
            fragment: {
                module,
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
            pipeline = this._buildDepthPipeline(gpuDevice, this._depthPipelineLayout!, this.shaderRenderModule!, 'vertex_main', vertexBuffersDescriptors, depthFormat, bias, 'DepthPipeline');
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
            pipeline = this._buildVelocityPipeline(gpuDevice, this.bindableGroup.pipelineBindGroupLayout!, this.shaderRenderModule!, 'vertex_main', vertexBuffersDescriptors, sampleCount, 'VelocityPipeline');
            this._velocityPipelineCache.set(key, pipeline);
        }
        return pipeline;
    }

    /** A depth-only pipeline: `entryPoint` of `module` over `vertexBuffersDescriptors`, with `shadowFragmentEntry` if set. */
    private _buildDepthPipeline(
        gpuDevice: GPUDevice,
        layout: GPUPipelineLayout,
        module: GPUShaderModule,
        entryPoint: string,
        vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null>,
        depthFormat: GPUTextureFormat,
        bias: DepthBias,
        label: string,
    ): GPURenderPipeline {
        // WebGPU allows a depth bias on triangle topologies only.
        const triangles = this.topology === 'triangle-list' || this.topology === 'triangle-strip';
        const entry = this.options.shadowFragmentEntry;
        return gpuDevice.createRenderPipeline({
            label: `${this.label}/${label}`,
            layout,
            vertex: { module, entryPoint, buffers: vertexBuffersDescriptors },
            fragment: entry ? { module, entryPoint: entry, targets: [] } : undefined,
            primitive: {
                topology: this.topology,
                cullMode: this.cullMode,
            },
            depthStencil: {
                format: depthFormat,
                depthWriteEnabled: true,
                depthCompare: 'less-equal',
                depthBias: triangles ? bias.constant ?? 0 : 0,
                depthBiasSlopeScale: triangles ? bias.slopeScale ?? 0 : 0,
                depthBiasClamp: triangles ? bias.clamp ?? 0 : 0,
            },
        });
    }

    /** A velocity-pass pipeline: `entryPoint` of `module`, its `fragment_main` into the velocity target only. */
    private _buildVelocityPipeline(
        gpuDevice: GPUDevice,
        layout: GPUPipelineLayout,
        module: GPUShaderModule,
        entryPoint: string,
        vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null>,
        sampleCount: number,
        label: string,
    ): GPURenderPipeline {
        const targets: (GPUColorTargetState | null)[] = new Array(GBuffer.VELOCITY_TARGET).fill(null);
        targets.push({ format: GBuffer.VELOCITY_FORMAT, writeMask: GPUColorWrite.ALL });
        return gpuDevice.createRenderPipeline({
            label: `${this.label}/${label}`,
            layout,
            multisample: { count: sampleCount },
            vertex: { module, entryPoint, buffers: vertexBuffersDescriptors },
            fragment: { module, entryPoint: 'fragment_main', targets },
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
    }

    /**
     * The generated cluster vertex stage's module for `instances`' records (remade, with every
     * cluster pipeline, when the layout changes). Throws, saying why it can't be generated.
     */
    private _clusterModule(gpuDevice: GPUDevice, instances: InstanceLayout | null): GPUShaderModule {
        this._ensureSharedResources(gpuDevice);
        const key = instances ? vertexLayoutKey([instances as GPUVertexBufferLayout]) : '-';
        if (this._clusterStage?.key !== key) {
            let module: GPUShaderModule | null = null, error: string | null = null;
            try {
                const code = clusterVertexStage(parseIncludes(this.shaderCode), instances);
                module = gpuDevice.createShaderModule({ label: `${this.label}/ClusterShader`, code });
            } catch (e) {
                error = (e as Error).message;
            }
            this._clusterStage = { key, module, error };
            this._clusterPipelineCache.clear();
            this._clusterDepthPipelineCache.clear();
            this._clusterVelocityPipelineCache.clear();
            this._clusterVoxelPipelineCache.clear();
        }
        if (!this._clusterStage.module) throw new Error(this._clusterStage.error!);
        return this._clusterStage.module;
    }

    /** The cluster pipelines' layout: the material's groups with the cluster mesh group as group 2. */
    private _clusterLayout(gpuDevice: GPUDevice): GPUPipelineLayout {
        if (!this._clusterPipelineLayout) {
            this._clusterPipelineLayout = gpuDevice.createPipelineLayout({
                label: `${this.label}/ClusterPipelineLayout`,
                bindGroupLayouts: [
                    this.bindableGroup.bindGroupLayout!,
                    this.bindableGroup.cameraBindablesGroupLayout!,
                    gpuDevice.createBindGroupLayout({ label: 'ClusterMesh BindGroupLayout', entries: clusterMeshBindGroupLayoutEntries() }),
                    this.bindableGroup.shadowBindablesGroupLayout!,
                ],
            });
        }
        return this._clusterPipelineLayout;
    }

    /**
     * Returns the pipeline that draws this material over a cluster draw (the camera's cut of
     * `Renderable.clusters`; Rust `get_cluster_pipeline`): its WGSL with the generated vertex
     * stage (`clusterVertexStage`) for `instances`' records, no vertex buffers, group 2 the
     * cluster mesh group (`clusterMeshBindGroupLayoutEntries`). Throws, saying why the stage
     * can't be generated; the renderable then keeps the ordinary path.
     */
    public getClusterPipeline(
        gpuDevice: GPUDevice,
        instances: InstanceLayout | null,
        colorFormats: GPUTextureFormat[],
        sampleCount: number,
        depthFormat: GPUTextureFormat,
    ): GPURenderPipeline {
        const module = this._clusterModule(gpuDevice, instances);
        const key = `${colorFormats.join(',')}:${sampleCount}:${depthFormat}`;
        let pipeline = this._clusterPipelineCache.get(key);
        if (!pipeline) {
            pipeline = this._buildPipeline(gpuDevice, [], colorFormats, sampleCount, depthFormat,
                { layout: this._clusterLayout(gpuDevice), module, entryPoint: CLUSTER_VERTEX_ENTRY, label: 'ClusterPipeline' });
            this._clusterPipelineCache.set(key, pipeline);
        }
        return pipeline;
    }

    /** The cluster pipeline made for a pass of these targets (`getClusterPipeline`), if any. */
    public clusterPipeline(colorFormats: GPUTextureFormat[], sampleCount: number, depthFormat: GPUTextureFormat): GPURenderPipeline | null {
        return this._clusterPipelineCache.get(`${colorFormats.join(',')}:${sampleCount}:${depthFormat}`) ?? null;
    }

    /**
     * Returns the shadow passes' pipeline over a cluster draw (a shadow view's cut of
     * `Renderable.clusters`; Rust `get_cluster_depth_pipeline`): `getDepthPipeline`'s, with the
     * generated vertex stage. Throws as `getClusterPipeline` does.
     */
    public getClusterDepthPipeline(
        gpuDevice: GPUDevice,
        instances: InstanceLayout | null,
        depthFormat: GPUTextureFormat,
        bias: DepthBias = {},
    ): GPURenderPipeline {
        const module = this._clusterModule(gpuDevice, instances);
        const key = `${depthFormat}:${bias.constant ?? 0}:${bias.slopeScale ?? 0}:${bias.clamp ?? 0}`;
        let pipeline = this._clusterDepthPipelineCache.get(key);
        if (!pipeline) {
            if (!this._clusterDepthPipelineLayout) {
                this._clusterDepthPipelineLayout = gpuDevice.createPipelineLayout({
                    label: `${this.label}/ClusterDepthPipelineLayout`,
                    bindGroupLayouts: [
                        this.bindableGroup.bindGroupLayout!,
                        this.bindableGroup.cameraBindablesGroupLayout!,
                        gpuDevice.createBindGroupLayout({ label: 'ClusterMesh BindGroupLayout', entries: clusterMeshBindGroupLayoutEntries() }),
                    ],
                });
            }
            pipeline = this._buildDepthPipeline(gpuDevice, this._clusterDepthPipelineLayout, module, CLUSTER_VERTEX_ENTRY, [], depthFormat, bias, 'ClusterDepthPipeline');
            this._clusterDepthPipelineCache.set(key, pipeline);
        }
        return pipeline;
    }

    /**
     * Returns the velocity pass's pipeline over a cluster draw (the camera's cut of
     * `Renderable.clusters`; Rust `get_cluster_velocity_pipeline`): `getVelocityPipeline`'s, with
     * the generated vertex stage. Throws as `getClusterPipeline` does.
     */
    public getClusterVelocityPipeline(gpuDevice: GPUDevice, instances: InstanceLayout | null, sampleCount: number = 1): GPURenderPipeline {
        const module = this._clusterModule(gpuDevice, instances);
        let pipeline = this._clusterVelocityPipelineCache.get(sampleCount);
        if (!pipeline) {
            pipeline = this._buildVelocityPipeline(gpuDevice, this._clusterLayout(gpuDevice), module, CLUSTER_VERTEX_ENTRY, [], sampleCount, 'ClusterVelocityPipeline');
            this._clusterVelocityPipelineCache.set(sampleCount, pipeline);
        }
        return pipeline;
    }

    /**
     * Returns the cluster debug view's pipeline (`Renderer.setClusterDebug`, `ClusterDebug`) for a
     * pass of these targets: the generated cluster vertex stage as a function
     * (`clusterVertexFunction`) for `instances`' records, under the debug view's own stages
     * (`clusterDebugWgsl`), group 2 the debug group (`clusterDebugBindGroupLayoutEntries`), drawn
     * without an index buffer. Throws as `getClusterPipeline` does.
     */
    public getClusterDebugPipeline(
        gpuDevice: GPUDevice,
        instances: InstanceLayout | null,
        colorFormats: GPUTextureFormat[],
        sampleCount: number,
        depthFormat: GPUTextureFormat,
    ): GPURenderPipeline {
        this._ensureSharedResources(gpuDevice);
        const layoutKey = instances ? vertexLayoutKey([instances as GPUVertexBufferLayout]) : '-';
        const key = `${layoutKey}|${colorFormats.join(',')}:${sampleCount}:${depthFormat}`;
        let pipeline = this._clusterDebugPipelineCache.get(key);
        if (!pipeline) {
            const stage = clusterVertexFunction(parseIncludes(this.shaderCode), instances);
            const module = gpuDevice.createShaderModule({
                label: `${this.label}/ClusterDebugShader`,
                code: `${stage.code}\n${clusterDebugWgsl(stage.position, colorFormats.length)}`,
            });
            pipeline = gpuDevice.createRenderPipeline({
                label: `${this.label}/ClusterDebugPipeline`,
                layout: gpuDevice.createPipelineLayout({
                    label: `${this.label}/ClusterDebugPipelineLayout`,
                    bindGroupLayouts: [
                        this.bindableGroup.bindGroupLayout!,
                        this.bindableGroup.cameraBindablesGroupLayout!,
                        gpuDevice.createBindGroupLayout({ label: 'ClusterDebug BindGroupLayout', entries: clusterDebugBindGroupLayoutEntries() }),
                    ],
                }),
                multisample: { count: sampleCount },
                vertex: { module, entryPoint: CLUSTER_DEBUG_VERTEX_ENTRY, buffers: [] },
                fragment: { module, entryPoint: CLUSTER_DEBUG_FRAGMENT_ENTRY, targets: colorFormats.map((format) => ({ format })) },
                primitive: { topology: 'triangle-list', cullMode: this.transparent ? 'none' : this.cullMode },
                depthStencil: { depthWriteEnabled: true, depthCompare: this.depthCompare, format: depthFormat },
            });
            this._clusterDebugPipelineCache.set(key, pipeline);
        }
        return pipeline;
    }

    /**
     * Returns the pipeline the voxel clipmap's voxelizer draws this material's cluster cuts with
     * (a voxel GI view's cut of `Renderable.clusters`; Rust `get_cluster_voxel_pipeline`):
     * `getVoxelPipeline`'s, with the generated vertex stage and the cluster mesh group as group 2.
     * Throws as `getClusterPipeline` does.
     */
    public getClusterVoxelPipeline(
        gpuDevice: GPUDevice,
        voxelizer: number,
        instances: InstanceLayout | null,
        voxelBGL: GPUBindGroupLayout,
        engineFragment: { module: GPUShaderModule, entryPoint: string },
        target: GPUTextureFormat,
        sampleCount: number,
    ): GPURenderPipeline {
        const module = this._clusterModule(gpuDevice, instances);
        let pipeline = this._clusterVoxelPipelineCache.get(voxelizer);
        if (!pipeline) {
            const layout = gpuDevice.createPipelineLayout({
                label: `${this.label}/ClusterVoxelPipelineLayout`,
                bindGroupLayouts: [
                    this.bindableGroup.bindGroupLayout!,
                    this.bindableGroup.cameraBindablesGroupLayout!,
                    gpuDevice.createBindGroupLayout({ label: 'ClusterMesh BindGroupLayout', entries: clusterMeshBindGroupLayoutEntries() }),
                    voxelBGL,
                ],
            });
            const entry = this.options.voxelFragmentEntry;
            const fragment = entry ? { module, entryPoint: entry } : engineFragment;
            pipeline = gpuDevice.createRenderPipeline({
                label: `${this.label}/ClusterVoxelPipeline`,
                layout,
                vertex: { module, entryPoint: CLUSTER_VERTEX_ENTRY, buffers: [] },
                fragment: { ...fragment, targets: [{ format: target, writeMask: 0 }] },
                primitive: { topology: this.topology, cullMode: 'none' },
                multisample: { count: sampleCount },
            });
            this._clusterVoxelPipelineCache.set(voxelizer, pipeline);
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
     * Bind `value` at `binding` of group 0 in place of what was bound there (a resized render
     * target, a planar reflection remade at a new size), keeping the layout: the bind group is
     * rebuilt on its next use. Rust: `Material::set_bindable`.
     */
    public setBindable(binding: number, value: IBindable): void {
        const bindable = this.bindableGroup.bindables.find((b) => b.binding === binding);
        if (!bindable) throw new Error(`${this.label}: no binding ${binding} to set`);
        bindable.value = value;
        this.bindableGroup.bindGroup = undefined;
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
