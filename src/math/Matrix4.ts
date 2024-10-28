import { mat4, vec3 } from "gl-matrix";
import { BufferBase } from "../buffers/BufferBase";
import { Vector3 } from "./Vector3";

/**
 * A class representing a 4x4 matrix with various transformation operations.
 */
class Matrix4 extends BufferBase {

    /** The type of buffer, set to uniform. */
    public type: string = BufferBase.BUFFER_TYPE_UNIFORM;

    /** The internal 4x4 matrix. */
    public readonly internalMat4: mat4;

    private right = vec3.fromValues(1, 0, 0);
    private up = vec3.fromValues(0, 1, 0);
    private forward = vec3.fromValues(0, 0, 1);

    /**
     * Constructs a new Matrix4 instance and initializes the internal matrix.
     */
    constructor() {
        super();
        this.internalMat4 = mat4.create();
        this.buffer = new Float32Array(this.internalMat4);
    }

    /**
     * Updates the buffer with the current matrix values and marks it for update.
     */
    private updateBuffer() {
        this.buffer!.set(this.internalMat4);
        this.needsUpdate = true;
    }

    /**
     * Inverts the given world matrix and updates this matrix with the result.
     * @param worldMatrix The matrix to invert.
     * @returns This matrix after inversion.
     */
    invert(worldMatrix: Matrix4) {
        mat4.invert(this.internalMat4, worldMatrix.internalMat4);
        this.updateBuffer();
        return this;
    }

    /**
     * Transposes this matrix.
     * @returns This matrix after transposition.
     */
    transpose() {
        mat4.transpose(this.internalMat4, this.internalMat4);
        this.updateBuffer();
        return this;
    }

    /**
     * Sets this matrix to a perspective projection matrix.
     * @param fov Field of view in radians.
     * @param aspect Aspect ratio.
     * @param near Near clipping plane.
     * @param far Far clipping plane.
     * @returns This matrix after setting perspective.
     */
    perspective(fov: number, aspect: number, near: number, far: number) {
        mat4.perspective(this.internalMat4, fov, aspect, near, far);
        this.updateBuffer();
        return this;
    }

    /**
     * Multiplies two matrices and stores the result in this matrix.
     * @param a The first matrix.
     * @param b The second matrix.
     * @returns This matrix after multiplication.
     */
    multiply(a: Matrix4, b: Matrix4): Matrix4 {
        mat4.multiply(this.internalMat4, a.internalMat4, b.internalMat4);
        this.updateBuffer();
        return this;
    }

    /**
     * Rotates this matrix around the X axis.
     * @param angle The angle in radians.
     * @returns This matrix after rotation.
     */
    rotateX(angle: number): Matrix4 {
        mat4.rotateX(this.internalMat4, this.internalMat4, angle);
        this.updateBuffer();
        return this;
    }

    /**
     * Rotates this matrix around the Y axis.
     * @param angle The angle in radians.
     * @returns This matrix after rotation.
     */
    rotateY(angle: number): Matrix4 {
        mat4.rotateY(this.internalMat4, this.internalMat4, angle);
        this.updateBuffer();
        return this;
    }

    /**
     * Rotates this matrix around the Z axis.
     * @param angle The angle in radians.
     * @returns This matrix after rotation.
     */
    rotateZ(angle: number): Matrix4 {
        mat4.rotateZ(this.internalMat4, this.internalMat4, angle);
        this.updateBuffer();
        return this;
    }

    /**
     * Scales this matrix by a given vector.
     * @param v The vector to scale by.
     * @returns This matrix after scaling.
     */
    scale(v: Vector3): Matrix4 {
        mat4.scale(this.internalMat4, this.internalMat4, v.getVec() as Float32Array);
        this.updateBuffer();
        return this;
    }

    /**
     * Translates this matrix by a given vector.
     * @param v The vector to translate by.
     * @returns This matrix after translation.
     */
    translate(v: Vector3): Matrix4 {
        mat4.translate(this.internalMat4, this.internalMat4, v.getVec() as Float32Array);
        this.updateBuffer();
        return this;
    }

    /**
     * Rotates this matrix by a given vector of angles.
     * @param v The vector containing rotation angles for x, y, and z axes.
     * @returns This matrix after rotation.
     */
    rotate(v: Vector3): Matrix4 {
        mat4.rotate(this.internalMat4, this.internalMat4, v.z, this.forward);
        mat4.rotate(this.internalMat4, this.internalMat4, v.y, this.up);
        mat4.rotate(this.internalMat4, this.internalMat4, v.x, this.right);
        this.updateBuffer();
        return this;
    }

    /**
     * Resets this matrix to the identity matrix.
     * @returns This matrix after reset.
     */
    identity(): Matrix4 {
        mat4.identity(this.internalMat4);
        this.updateBuffer();
        return this;
    }

    /**
     * Copies the values from another matrix into this matrix.
     * @param a The matrix to copy from.
     * @returns This matrix after copying.
     */
    copy(a: Matrix4): Matrix4 {
        mat4.copy(this.internalMat4, a.internalMat4);
        this.updateBuffer();
        return this;
    }

    /**
     * Extracts Euler angles from this matrix.
     * @returns An array containing the Euler angles [rotationX, rotationY, rotationZ].
     */
    extractEulerAngles(): [number, number, number] {
        const m = this.internalMat4;
        let rotationX, rotationY, rotationZ;

        if (m[2] < 1) {
            if (m[2] > -1) {
                rotationY = Math.asin(-m[2]);
                rotationX = Math.atan2(m[6], m[10]);
                rotationZ = Math.atan2(m[1], m[0]);
            } else {
                // m[2] = -1
                rotationY = Math.PI / 2;
                rotationX = -Math.atan2(-m[9], m[5]);
                rotationZ = 0;
            }
        } else {
            // m[2] = 1
            rotationY = -Math.PI / 2;
            rotationX = Math.atan2(-m[9], m[5]);
            rotationZ = 0;
        }

        return [rotationX, rotationY, rotationZ];
    }
}

export { Matrix4 }
