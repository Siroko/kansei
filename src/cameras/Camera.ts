import { mat4 } from "gl-matrix";
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
 *
 * It also keeps what temporal effects need (Rust `cameras::Camera`): a sub-pixel `jitter` the
 * GPU projection is offset by, last frame's view-projection and a frame counter, at group 1
 * binding 3 (`KanseiCameraTemporal`, `MOTION_VECTORS_WGSL`). The renderer uploads them with
 * `uploadTemporal` and calls `endFrame` after each frame.
 */
class Camera extends Object3D {
    public viewMatrix: Matrix4;
    public inverseViewMatrix: Matrix4;
    /** The projection without jitter; the GPU gets it offset by `jitter` (`jitteredProjectionMatrix`). */
    public projectionMatrix: Matrix4;
    /** Group 1 binding 1: `projectionMatrix` offset by `jitter`, as `uploadTemporal` last computed it. */
    public jitteredProjectionMatrix: Matrix4;
    /**
     * Sub-pixel offset of the projection, in NDC (set when an effect such as TAA asks for it;
     * zero otherwise).
     */
    public jitter: [number, number] = [0, 0];
    /**
     * Group 1 binding 2: the scene lights (`KanseiLights`, `LIGHTS_WGSL`). The renderer swaps in
     * its own buffer (`useLightUniforms`), so every camera it draws with sees the scene's lights;
     * all zeros (no lights) before that.
     */
    public lightUniforms!: ComputeBuffer;
    /** Group 1 binding 3: unjittered and previous view-projection, jitter, frame (`KanseiCameraTemporal`). */
    public temporalUniforms!: ComputeBuffer;

    private _lastViewWorldVersion: number = -1;
    // last frame's unjittered view-projection and jitter, for motion vectors
    private _prevViewProj: mat4 | null = null;
    private _prevJitter: [number, number] = [0, 0];
    private _frame: number = 0;
    // temporalUniforms' data (KanseiCameraTemporal: 40 floats, `frame` a u32 at 36)
    private _temporalData = new Float32Array(CAMERA_TEMPORAL_BYTES / 4);
    // what jitteredProjectionMatrix was last computed from
    private _jitteredFrom: [number, number, number] = [-1, NaN, NaN];

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
        this.jitteredProjectionMatrix = new Matrix4().copy(this.projectionMatrix);
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

    /** The projection with the sub-pixel `jitter` applied (what the GPU renders with), into `out`. */
    public jitteredProjection(out: mat4 = mat4.create()): mat4 {
        // translation(jitter) * projection: the jitter times the w row, added to the x and y rows
        const p = this.projectionMatrix.internalMat4;
        const [jx, jy] = this.jitter;
        mat4.copy(out, p);
        for (let c = 0; c < 4; c++) {
            out[c * 4] += jx * p[c * 4 + 3];
            out[c * 4 + 1] += jy * p[c * 4 + 3];
        }
        return out;
    }

    /** Unjittered projection times view, into `out`. */
    public viewProjection(out: mat4 = mat4.create()): mat4 {
        return mat4.multiply(out, this.projectionMatrix.internalMat4, this.viewMatrix.internalMat4);
    }

    /** Last frame's unjittered view-projection, or null if there was no last frame since `resetMotion`. */
    public previousViewProjection(): mat4 | null {
        return this._prevViewProj ? mat4.clone(this._prevViewProj) : null;
    }

    /**
     * Last frame's view-projection as it drew, jitter included: what reconstructs world positions
     * from last frame's depth (its pixels were rendered jittered). Null as `previousViewProjection`.
     */
    public previousJitteredViewProjection(): mat4 | null {
        if (!this._prevViewProj) return null;
        const out = mat4.fromTranslation(mat4.create(), [this._prevJitter[0], this._prevJitter[1], 0]);
        return mat4.multiply(out, out, this._prevViewProj);
    }

    /** Frames rendered since creation (wraps at 2^32). */
    public get frame(): number {
        return this._frame;
    }

    /** Forget the previous frame's view, so the next frame has no camera motion (camera cuts). */
    public resetMotion(): void {
        this._prevViewProj = null;
    }

    /**
     * Called by the renderer after a frame: this frame's view becomes the previous one. Call it
     * yourself when driving the post-processing effects without the renderer.
     */
    public endFrame(): void {
        this._prevViewProj = this.viewProjection(this._prevViewProj ?? mat4.create());
        this._prevJitter = [this.jitter[0], this.jitter[1]];
        this._frame = (this._frame + 1) >>> 0;
    }

    /**
     * Writes the jittered projection (binding 1) and the temporal data (binding 3) for the GPU;
     * they upload with the next `getBindGroup`. The renderer calls it before each frame's draws,
     * after `updateViewMatrix`.
     */
    public uploadTemporal(): void {
        const from = this._jitteredFrom;
        if (from[0] !== this.projectionMatrix.version || from[1] !== this.jitter[0] || from[2] !== this.jitter[1]) {
            this.jitteredProjection(this.jitteredProjectionMatrix.internalMat4);
            this.jitteredProjectionMatrix.syncBuffer();
            this._jitteredFrom = [this.projectionMatrix.version, this.jitter[0], this.jitter[1]];
        }

        const data = this._temporalData;
        const viewProj = this.viewProjection(data.subarray(0, 16));
        data.set(this._prevViewProj ?? viewProj, 16);
        data[32] = this.jitter[0];
        data[33] = this.jitter[1];
        const prevJitter = this._prevViewProj ? this._prevJitter : this.jitter;
        data[34] = prevJitter[0];
        data[35] = prevJitter[1];
        new Uint32Array(data.buffer, data.byteOffset, data.length)[36] = this._frame;
        this.temporalUniforms.needsUpdate = true;
    }

    /**
     * Binds `lights` at binding 2 instead of the camera's own buffer (the renderer passes its
     * scene light uniform, `Renderer.lightUniforms`). Rebuilds the bind group when it changes.
     */
    public useLightUniforms(lights: ComputeBuffer): void {
        if (this.lightUniforms === lights) return;
        this.lightUniforms = lights;
        const group = this.bindableGroup!;
        group.bindables.find((b) => b.binding === 2)!.value = lights;
        group.bindGroup = undefined;
    }

    /**
     * Sets the camera's bind group (group 1): view and jittered projection matrices,
     * scene lights and temporal data, laid out like the Rust engine's camera group.
     */
    protected setUniforms() {
        super.setUniforms();

        const uniform = (data: Float32Array) => new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_UNIFORM,
            usage: BufferBase.BUFFER_USAGE_UNIFORM | BufferBase.BUFFER_USAGE_COPY_DST,
            buffer: data,
        });
        this.lightUniforms = uniform(new Float32Array(LIGHT_UNIFORM_BYTES / 4));
        this._temporalData ??= new Float32Array(CAMERA_TEMPORAL_BYTES / 4);
        this.temporalUniforms = uniform(this._temporalData);

        const vertexFragment = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
        this.bindableGroup = new BindableGroup([
            { binding: 0, visibility: vertexFragment, value: this.viewMatrix },
            { binding: 1, visibility: vertexFragment, value: this.jitteredProjectionMatrix },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, value: this.lightUniforms },
            { binding: 3, visibility: vertexFragment, value: this.temporalUniforms },
        ]);
    }
}

export { Camera }
