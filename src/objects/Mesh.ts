import { Geometry } from "../buffers/Geometry";
import { Material } from "../materials/Material";
import { UniformGroup } from "../materials/UniformGroup";
import { Object3D } from "./Object3D";

class Mesh extends Object3D {
    public isMesh: boolean = true;
    public bindGroupLayout?: GPUBindGroupLayout;
    public bindGroup?: GPUBindGroup;
    public initialized: boolean = false;
    public instanceCount: number = 1;

    constructor(
        public geometry: Geometry,
        public material: Material
    ) {
        super();
    }

    protected setUniforms() {
        super.setUniforms();
        this.uniformGroup = new UniformGroup([
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                value: this.modelMatrix
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