import { Vector3 } from "../math/Vector3";
import { Matrix4 } from "../math/Matrix4";
import { UniformGroup } from "../materials/UniformGroup";

class Object3D {
    public modelMatrix: Matrix4;
    public worldMatrix: Matrix4;
    public children: Object3D[] = [];
    public isMesh: boolean = false;
    public parent?: Object3D;
    public position: Vector3 = new Vector3();
    public rotation: Vector3 = new Vector3();
    public scale: Vector3 = new Vector3(1, 1, 1);

    protected uniformGroup?: UniformGroup;

    constructor() {
        this.modelMatrix = new Matrix4();
        this.worldMatrix = new Matrix4();
        this.setUniforms();
    }

    protected setUniforms() {

    }

    public add(object: Object3D) {
        this.children.push(object);
        object.parent = this;
    }

    public updateModelMatrix() {
        this.modelMatrix.fromTranslation(this.position.toVec());
        this.modelMatrix.rotateX(this.rotation.x);
        this.modelMatrix.rotateY(this.rotation.y);
        this.modelMatrix.rotateZ(this.rotation.z);
        this.modelMatrix.scale(this.scale);

        this.updateWorldMatrix();
    }

    public updateWorldMatrix() {
        if (this.parent) {
            this.worldMatrix.multiply(this.parent.worldMatrix, this.modelMatrix);
        } else {
            this.worldMatrix.copy(this.modelMatrix);
        }
    }

    public async getBindGroup(gpuDevice: GPUDevice, pipeline: GPURenderPipeline, bindingGroupLayoutPosition: number = 0): Promise<GPUBindGroup> {
        await this.uniformGroup!.getBindGroup(gpuDevice, pipeline!, bindingGroupLayoutPosition);
        return new Promise((resolve) => {
            resolve(this.uniformGroup!.bindGroup!);
        });
    }
}

export { Object3D }