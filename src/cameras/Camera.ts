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
        this.viewMatrix.needsUpdate = true;
    }

    lookAt(target: Vector3) {
        super.lookAt(target);
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
