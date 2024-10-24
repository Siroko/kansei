import { Vector3 } from "../math/Vector3";
import { Matrix4 } from "../math/Matrix4";
import { UniformGroup } from "../materials/UniformGroup";
import { mat4 } from "gl-matrix";

class Object3D {
    public modelMatrix: Matrix4;
    public worldMatrix: Matrix4;

    protected rotationMatrix: Matrix4 = new Matrix4();
    protected translationMatrix: Matrix4 = new Matrix4();
    protected scaleMatrix: Matrix4 = new Matrix4();
    protected lookAtMatrix: Matrix4 = new Matrix4();

    public children: Object3D[] = [];
    public isMesh: boolean = false;
    public parent?: Object3D;
    public position: Vector3 = new Vector3();
    public rotation: Vector3 = new Vector3();
    public scale: Vector3 = new Vector3(1, 1, 1);

    public up: Vector3 = new Vector3(0, 1, 0);

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
        this.scaleMatrix.identity(); // Reset the matrix
        this.scaleMatrix.scale(this.scale);

        this.rotationMatrix.identity(); // Reset the matrix
        this.rotationMatrix.rotate(this.rotation);

        this.translationMatrix.identity(); // Reset the matrix
        this.translationMatrix.translate(this.position);

        this.modelMatrix.identity(); // Reset the matrix

        this.modelMatrix.multiply(this.modelMatrix, this.translationMatrix);
        this.modelMatrix.multiply(this.modelMatrix, this.rotationMatrix);
        this.modelMatrix.multiply(this.modelMatrix, this.scaleMatrix);

        this.updateWorldMatrix();
        this.modelMatrix.needsUpdate = true;
        this.worldMatrix.needsUpdate = true;
    }

    public updateWorldMatrix() {
        if (this.parent) {
            this.worldMatrix.multiply(this.parent.worldMatrix, this.modelMatrix);
        } else {
            this.worldMatrix.copy(this.modelMatrix);
        }
    }

    /**
     * Method to calculate the lookAt rotation matrix
     * @param target
     */
    lookAt(target: Vector3) {
        this.lookAtMatrix.identity();
        mat4.lookAt(this.lookAtMatrix.internalMat4, this.position.toVec(), target.toVec(), this.up.toVec());
        this.lookAtMatrix.invert(this.lookAtMatrix);
        // Extract Euler angles from the world matrix
        const [rotationX, rotationY, rotationZ] = this.lookAtMatrix.extractEulerAngles();

        this.rotation.x = rotationX;
        this.rotation.y = rotationY;
        this.rotation.z = rotationZ;

        this.updateModelMatrix();
    }

    public getBindGroup(gpuDevice: GPUDevice): GPUBindGroup {
        this.uniformGroup!.getBindGroup(gpuDevice);
        return this.uniformGroup!.bindGroup!;
    }
}

export { Object3D }
