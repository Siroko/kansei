import { Geometry } from "../buffers/Geometry";
import { Material } from "../materials/Material";
import { BindableGroup } from "../materials/BindableGroup";
import { Object3D } from "./Object3D";

/**
 * Represents a 3D renderable object that extends Object3D.
 * A renderable is composed of geometry and material.
 */
class Renderable extends Object3D {
    /** Indicates if the object is a mesh. */
    public isRenderable: boolean = true;

    /** The layout of the bind group for GPU resources. */
    public bindGroupLayout?: GPUBindGroupLayout;

    /** The bind group for GPU resources. */
    public bindGroup?: GPUBindGroup;

    /** Indicates if the renderable has been initialized. */
    public initialized: boolean = false;

    /** The number of instances of the renderable. */
    public instanceCount: number = 1;

    /**
     * Constructs a new Renderable object.
     * 
     * @param geometry - The geometry of the renderable.
     * @param material - The material of the renderable.
     */
    constructor(
        public geometry: Geometry,
        public material: Material
    ) {
        super();
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
