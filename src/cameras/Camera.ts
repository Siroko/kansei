import { Object3D } from "../objects/Object3D";
import { UniformGroup } from "../materials/UniformGroup";
import { Matrix4 } from "../math/Matrix4";
import { Vector3 } from "../math/Vector3";

/**
 * Represents a camera in 3D space, extending the Object3D class.
 * Handles view and projection matrices for rendering.
 */
class Camera extends Object3D {
    public viewMatrix: Matrix4;
    public inverseViewMatrix: Matrix4;
    public projectionMatrix: Matrix4;

    /**
     * Constructs a new Camera instance.
     * 
     * @param fov - Field of view in degrees.
     * @param near - Near clipping plane distance.
     * @param far - Far clipping plane distance.
     * @param aspect - Aspect ratio of the camera.
     */
    constructor(
        public fov: number = 75,
        public near: number = 0.1,
        public far: number = 100,
        public aspect: number = 1
    ) {
        super();
        this.viewMatrix = new Matrix4();
        this.inverseViewMatrix = new Matrix4();
        this.projectionMatrix = new Matrix4().perspective(this.fov, this.aspect, this.near, this.far);
        this.setUniforms();
    }

    /**
     * Updates the projection matrix based on the current camera parameters.
     */
    public updateProjectionMatrix() {
        this.projectionMatrix.perspective(this.fov, this.aspect, this.near, this.far);
    }

    /**
     * Updates the view matrix by inverting the world matrix.
     */
    public updateViewMatrix() {
        this.updateModelMatrix();
        this.viewMatrix.invert(this.worldMatrix);
        this.inverseViewMatrix.invert(this.viewMatrix);
        this.viewMatrix.needsUpdate = true;
        this.inverseViewMatrix.needsUpdate = true;
    }

    /**
     * Adjusts the camera to look at a specific target in 3D space.
     * 
     * @param target - The target position to look at.
     */
    lookAt(target: Vector3) {
        super.lookAt(target);
        this.updateViewMatrix();
    }

    /**
     * Sets the uniform variables for the shader, including view and projection matrices.
     */
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
