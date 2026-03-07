import { Vector3 } from "../math/Vector3";
import { Matrix4 } from "../math/Matrix4";
import { BindableGroup } from "../materials/BindableGroup";
import { mat4, vec3 } from "gl-matrix";

/**
 * Represents a 3D object in a scene with transformation matrices and hierarchical relationships.
 */
class Object3D {
    public modelMatrix: Matrix4;
    public worldMatrix: Matrix4;
    public normalMatrix: Matrix4;

    protected lookAtMatrix: Matrix4 = new Matrix4();

    public children: Object3D[] = [];
    public isRenderable: boolean = false;
    public parent?: Object3D;
    public position: Vector3 = new Vector3();
    public rotation: Vector3 = new Vector3();
    public scale: Vector3 = new Vector3(1, 1, 1);

    public up: Vector3 = new Vector3(0, 1, 0);

    public matrixNeedsUpdate: boolean = true;

    private _lastParentWorldVersion: number = -1;
    private _lastNormalViewVersion: number = -1;
    private _lastNormalWorldVersion: number = -1;

    protected bindableGroup?: BindableGroup;

    /**
     * Constructs a new Object3D instance with default transformation matrices.
     */
    constructor() {
        this.modelMatrix = new Matrix4();
        this.worldMatrix = new Matrix4();
        this.normalMatrix = new Matrix4();

        const flagDirty = () => { this.matrixNeedsUpdate = true; };
        this.position.onChange = flagDirty;
        this.rotation.onChange = flagDirty;
        this.scale.onChange = flagDirty;

        this.setUniforms();
    }

    /**
     * Sets the uniform variables for the object.
     * This method is intended to be overridden by subclasses.
     */
    protected setUniforms() {
        // Implementation for setting bindings
    }

    public traverse(object: Object3D, callback: (object: Object3D) => void) {
        callback(object);
        object.children.forEach(child => this.traverse(child, callback));
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
        // Ensure parent chain is up to date first
        if (this.parent) this.parent.updateModelMatrix();

        const parentWorldVersion = this.parent ? this.parent.worldMatrix.version : -1;
        if (!this.matrixNeedsUpdate && parentWorldVersion === this._lastParentWorldVersion) return;

        // Build T*R*S directly on internalMat4 — no intermediate buffer syncs.
        // scaleMatrix/rotationMatrix/translationMatrix wrappers are bypassed entirely
        // since those matrices are never uploaded to the GPU.
        const mm = this.modelMatrix.internalMat4;
        mat4.identity(mm);
        mat4.translate(mm, mm, this.position.getVec() as vec3);
        mat4.rotateZ(mm, mm, this.rotation.z);
        mat4.rotateY(mm, mm, this.rotation.y);
        mat4.rotateX(mm, mm, this.rotation.x);
        mat4.scale(mm, mm, this.scale.getVec() as vec3);

        this.updateWorldMatrix();

        this.matrixNeedsUpdate = false;
        this._lastParentWorldVersion = parentWorldVersion;
    }

    /**
     * Updates the normal matrix from the world matrix (inverse-transpose).
     * @param viewMatrix The view matrix (kept for API compatibility / cache invalidation).
     */
    public updateNormalMatrix(viewMatrix: Matrix4) {
        if (viewMatrix.version === this._lastNormalViewVersion &&
            this.worldMatrix.version === this._lastNormalWorldVersion) return;

        // Normal matrix = transpose(inverse(worldMatrix)).
        // World-space normals stay fixed regardless of camera orientation.
        const nm = this.normalMatrix.internalMat4;
        mat4.copy(nm, this.worldMatrix.internalMat4);
        mat4.invert(nm, nm);
        mat4.transpose(nm, nm);
        this.normalMatrix.syncBuffer();

        this._lastNormalViewVersion = viewMatrix.version;
        this._lastNormalWorldVersion = this.worldMatrix.version;
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
        const lm = this.lookAtMatrix.internalMat4;
        mat4.lookAt(lm, this.position.getVec() as vec3, target.toVec() as vec3, this.up.toVec() as vec3);
        mat4.invert(lm, lm);
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
        this.bindableGroup!.getBindGroup(gpuDevice);
        return this.bindableGroup!.bindGroup!;
    }
}

export { Object3D }
