import { Object3D } from "../objects/Object3D";
import { UniformGroup } from "../materials/UniformGroup";
import { Matrix4 } from "../math/Matrix4";
import { Vector3 } from "../math/Vector3";

class Camera extends Object3D {
    public viewMatrix: Matrix4;
    public projectionMatrix: Matrix4;

    constructor(
        public fov: number = 75,
        public near: number = 0.1,
        public far: number = 100,
        public aspect: number = 1
    ) {
        super();
        this.viewMatrix = new Matrix4();
        this.projectionMatrix = new Matrix4().perspective(this.fov, this.aspect, this.near, this.far);
        this.setUniforms();
    }

    public updateProjectionMatrix() {
        this.projectionMatrix.perspective(this.fov, this.aspect, this.near, this.far);
    }

    public updateViewMatrix() {
        this.updateModelMatrix();
        this.viewMatrix.invert(this.worldMatrix);
    }

    /**
     * Method to calculate the lookAt rotation matrix
     * @param target
     */
    lookAt(target: Vector3) {

        this.worldMatrix.lookAt(this.position, target, this.up);
        // const eye = vec3.create();
        // vec3.subtract(eye, this.position.toVec(), target.toVec());

        // mat4.identity(this.modelMatrix.internalMat4);
        // mat4.lookAt(this.modelMatrix.internalMat4, this.position.toVec(), target.toVec(), this.up.toVec());

        // mat4.identity(this.viewMatrix.internalMat4);
        // mat4.copy(this.viewMatrix.internalMat4, this.modelMatrix.internalMat4);

        // this.updateWorldMatrix();
        this.updateViewMatrix();

    }

    protected setUniforms() {
        super.setUniforms();

        this.uniformGroup = new UniformGroup([
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                value: this.viewMatrix
            },
            {
                binding: 1,
                visibility: GPUShaderStage.VERTEX,
                value: this.projectionMatrix
            }
        ]);
    }
}

export { Camera }