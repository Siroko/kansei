import { Geometry } from "../buffers/Geometry";
import { ComputeBuffer } from "../buffers/ComputeBuffer";
import { Material } from "../materials/Material";
import { BindableGroup } from "../materials/BindableGroup";
import { Object3D } from "./Object3D";
import { PathTracerMaterial } from "../pathtracer/PathTracerMaterial";

/**
 * Represents a 3D renderable object that extends Object3D.
 * A renderable is composed of geometry and material.
 */
class Renderable extends Object3D {
    /** Indicates if the object is a mesh. */
    public isRenderable: boolean = true;

    /** Whether this object casts shadows into shadow maps. */
    public castShadow: boolean = true;

    /** Whether this object receives shadows (shader must use #include <shadows>). */
    public receiveShadow: boolean = true;

    /** Controls draw order within the same transparency group. Higher values draw later (on top). */
    public renderOrder: number = 0;

    /** Custom WGSL snippet for shadow vertex transform.
     *  Must declare: fn shadowWorldPos(position: vec4f, instanceIdx: u32) -> vec4f
     *  returning the world-space position.  May include @group(2) bindings. */
    public shadowVertexCode: string | null = null;

    /** Bind group layout for extra resources used by shadowVertexCode (group 2). */
    public shadowExtraBGL: GPUBindGroupLayout | null = null;

    /** Bind group for extra resources used by shadowVertexCode (group 2). */
    public shadowExtraBG: GPUBindGroup | null = null;

    /** Path tracer material properties. If null, defaults are derived at BVH build time. */
    public pathTracerMaterial: PathTracerMaterial | null = null;

    /** GPU-side per-instance data for BVH TLAS expansion.
     *  When set on an InstancedGeometry renderable, BVHBuilder dispatches a compute pass
     *  that reads this buffer and writes N TLAS instance entries (transform + inverse + metadata).
     *  Buffer format depends on gpuInstanceFullTransform:
     *    false (default) — vec4f per instance: xyz = local position offset
     *    true            — 3×vec4f per instance: row-major 4×3 affine transform */
    public gpuInstanceBuffer: ComputeBuffer | null = null;

    /** When true, gpuInstanceBuffer contains 3×vec4f full affine transforms per instance
     *  instead of vec4f position offsets. Default: false (position-only mode). */
    public gpuInstanceFullTransform: boolean = false;

    /** When true, BVHBuilder creates per-instance BLAS copies that can be deformed
     *  each frame via a GPU compute pass (deform + refit). The TLAS instances get
     *  identity transforms since deformed triangles are in world space. */
    public dynamicBLAS: boolean = false;

    /** Override the number of TLAS instances for path tracing.
     *  When set (> 0), BVHBuilder uses this instead of geometry.instanceCount,
     *  allowing the path tracer to see more instances than the rasterizer draws. */
    public gpuInstanceCount: number = 0;

    /** The layout of the bind group for GPU resources. */
    public bindGroupLayout?: GPUBindGroupLayout;

    /** The bind group for GPU resources. */
    public bindGroup?: GPUBindGroup;

    /** Indicates if the renderable has been initialized. */
    public initialized: boolean = false;

    /** The number of instances of the renderable. */
    public instanceCount: number = 1;

    private _material: Material;
    public materialDirty: boolean = false;

    get material(): Material { return this._material; }
    set material(value: Material) {
        this._material = value;
        this.materialDirty = true;
    }

    /**
     * Constructs a new Renderable object.
     *
     * @param geometry - The geometry of the renderable.
     * @param material - The material of the renderable.
     */
    constructor(
        public geometry: Geometry,
        material: Material
    ) {
        super();
        this._material = material; // bypass setter — initial assignment is not a swap
    }

    /**
     * Sets the bindings for the renderable.
     * This method initializes the uniform group with the normal and world matrices.
     */
    protected setUniforms() {
        super.setUniforms();
        this.bindableGroup = new BindableGroup([
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                value: this.normalMatrix
            },
            {
                binding: 1,
                visibility: GPUShaderStage.VERTEX,
                value: this.worldMatrix
            }
        ]);
    }
}

export { Renderable }
