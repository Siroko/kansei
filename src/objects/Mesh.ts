import { Geometry } from "../buffers/Geometry";
import { Material } from "../materials/Material";
import { UniformGroup } from "../materials/UniformGroup";
import { Object3D } from "./Object3D";

/**
 * Represents a 3D mesh object that extends Object3D.
 * A mesh is composed of geometry and material.
 */
class Mesh extends Object3D {
    /** Indicates if the object is a mesh. */
    public isMesh: boolean = true;

    /** The layout of the bind group for GPU resources. */
    public bindGroupLayout?: GPUBindGroupLayout;

    /** The bind group for GPU resources. */
    public bindGroup?: GPUBindGroup;

    /** Indicates if the mesh has been initialized. */
    public initialized: boolean = false;

    /** The number of instances of the mesh. */
    public instanceCount: number = 1;

    /**
     * Constructs a new Mesh object.
     * 
     * @param geometry - The geometry of the mesh.
     * @param material - The material of the mesh.
     */
    constructor(
        public geometry: Geometry,
        public material: Material
    ) {
        super();
    }

    /**
     * Sets the uniforms for the mesh.
     * This method initializes the uniform group with the normal and world matrices.
     */
    protected setUniforms() {
        super.setUniforms();
        this.uniformGroup = new UniformGroup([
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

export { Mesh }
