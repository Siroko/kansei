import { Vector3 } from "../math/Vector3";
import { Matrix4 } from "../math/Matrix4";
import { UniformGroup } from "../materials/UniformGroup";
import { mat4, vec3 } from "gl-matrix";

/**
 * Represents a 3D object in a scene with transformation matrices and hierarchical relationships.
 */
class Object3D {
    public modelMatrix: Matrix4;
    public worldMatrix: Matrix4;
    public normalMatrix: Matrix4;

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

    /**
     * Constructs a new Object3D instance with default transformation matrices.
     */
    constructor() {
        this.modelMatrix = new Matrix4();
        this.worldMatrix = new Matrix4();
        this.normalMatrix = new Matrix4();
        this.setUniforms();
    }

    /**
     * Sets the uniform variables for the object.
     * This method is intended to be overridden by subclasses.
     */
    protected setUniforms() {
        // Implementation for setting uniforms
    }

    /**
     * Adds a child Object3D to this object.
     * @param object The Object3D to add as a child.
     */
    public add(object: Object3D) {
        this.children.push(object);
        object.parent = this;
    }

    /**
     * Updates the model matrix based on the object's position, rotation, and scale.
     */
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

    /**
     * Updates the normal matrix using the provided view matrix.
     * @param viewMatrix The view matrix to use for updating the normal matrix.
     */
    public updateNormalMatrix(viewMatrix: Matrix4) {
        this.normalMatrix.identity();
        this.normalMatrix.multiply(viewMatrix, this.worldMatrix);
        // Calculate inverse transpose
        this.normalMatrix.invert(this.normalMatrix).transpose();
        this.normalMatrix.needsUpdate = true;
    }

    /**
     * Updates the world matrix based on the parent's world matrix.
     */
    public updateWorldMatrix() {
        if (this.parent) {
            this.worldMatrix.multiply(this.parent.worldMatrix, this.modelMatrix);
        } else {
            this.worldMatrix.copy(this.modelMatrix);
        }
    }

    /**
     * Calculates the lookAt rotation matrix to orient the object towards a target.
     * @param target The target position to look at.
     */
    lookAt(target: Vector3) {
        this.lookAtMatrix.identity();
        mat4.lookAt(this.lookAtMatrix.internalMat4, this.position.getVec() as vec3, target.toVec() as vec3, this.up.toVec() as vec3);
        this.lookAtMatrix.invert(this.lookAtMatrix);
        // Extract Euler angles from the world matrix
        const [rotationX, rotationY, rotationZ] = this.lookAtMatrix.extractEulerAngles();

        this.rotation.x = rotationX;
        this.rotation.y = rotationY;
        this.rotation.z = rotationZ;

        this.updateModelMatrix();
    }

    /**
     * Retrieves the bind group for the GPU device.
     * @param gpuDevice The GPU device to use for retrieving the bind group.
     * @returns The GPUBindGroup associated with this object.
     */
    public getBindGroup(gpuDevice: GPUDevice): GPUBindGroup {
        this.uniformGroup!.getBindGroup(gpuDevice);
        return this.uniformGroup!.bindGroup!;
    }
}

export { Object3D }
