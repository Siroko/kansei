import { Object3D } from "../objects/Object3D";
import { BindableGroup } from "../materials/BindableGroup";
import { Matrix4 } from "../math/Matrix4";
import { Vector3 } from "../math/Vector3";
import { ComputeBuffer } from "../buffers/ComputeBuffer";
import { BufferBase } from "../buffers/BufferBase";
import { CAMERA_TEMPORAL_BYTES, LIGHT_UNIFORM_BYTES } from "../renderers/SharedLayouts";

/**
 * Represents a camera in 3D space, extending the Object3D class.
 * Handles view and projection matrices for rendering.
 */
class Camera extends Object3D {
    public viewMatrix: Matrix4;
    public inverseViewMatrix: Matrix4;
    public projectionMatrix: Matrix4;
    /** Group 1 binding 2: the scene lights (`KanseiLights`). All zeros (no lights) until the renderer uploads lights. */
    public lightUniforms!: ComputeBuffer;
    /** Group 1 binding 3: unjittered and previous view-projection, jitter, frame (`KanseiCameraTemporal`). All zeros until the camera tracks them. */
    public temporalUniforms!: ComputeBuffer;

    private _lastViewWorldVersion: number = -1;

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
        this.projectionMatrix = new Matrix4().perspective(this.fov * Math.PI / 180, this.aspect, this.near, this.far);
        this.setUniforms();
    }

    /**
     * Updates the projection matrix based on the current camera parameters.
     */
    public updateProjectionMatrix() {
        this.projectionMatrix.perspective(this.fov * Math.PI / 180, this.aspect, this.near, this.far);
    }

    /**
     * Updates the view matrix by inverting the world matrix.
     */
    public updateViewMatrix() {
        this.updateModelMatrix();
        if (this.worldMatrix.version === this._lastViewWorldVersion) return;
        this.viewMatrix.invert(this.worldMatrix);
        this.inverseViewMatrix.invert(this.viewMatrix);
        this.viewMatrix.needsUpdate = true;
        this.inverseViewMatrix.needsUpdate = true;
        this._lastViewWorldVersion = this.worldMatrix.version;
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
     * Sets the camera's bind group (group 1): view and projection matrices,
     * scene lights and temporal data, laid out like the Rust engine's camera group.
     */
    protected setUniforms() {
        super.setUniforms();

        const uniform = (bytes: number) => new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_UNIFORM,
            usage: BufferBase.BUFFER_USAGE_UNIFORM | BufferBase.BUFFER_USAGE_COPY_DST,
            buffer: new Float32Array(bytes / 4),
        });
        this.lightUniforms = uniform(LIGHT_UNIFORM_BYTES);
        this.temporalUniforms = uniform(CAMERA_TEMPORAL_BYTES);

        const vertexFragment = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
        this.bindableGroup = new BindableGroup([
            { binding: 0, visibility: vertexFragment, value: this.viewMatrix },
            { binding: 1, visibility: vertexFragment, value: this.projectionMatrix },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, value: this.lightUniforms },
            { binding: 3, visibility: vertexFragment, value: this.temporalUniforms },
        ]);
    }
}

export { Camera }
