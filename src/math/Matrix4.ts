import { mat4, vec3 } from "gl-matrix";
import { BufferBase } from "../buffers/BufferBase";
import { Vector3 } from "./Vector3";

class Matrix4 extends BufferBase {

    public type: string = BufferBase.BUFFER_TYPE_UNIFORM;
    public readonly internalMat4: mat4;

    private right = vec3.fromValues(1, 0, 0);
    private up = vec3.fromValues(0, 1, 0);
    private forward = vec3.fromValues(0, 0, 1);

    constructor() {
        super();
        this.internalMat4 = mat4.create();
        this.buffer = new Float32Array(this.internalMat4);
    }

    private updateBuffer() {
        this.buffer!.set(this.internalMat4);
        this.needsUpdate = true;
    }

    invert(worldMatrix: Matrix4) {
        mat4.invert(this.internalMat4, worldMatrix.internalMat4);
        this.updateBuffer();
        return this;
    }

    transpose() {
        mat4.transpose(this.internalMat4, this.internalMat4);
        this.updateBuffer();
        return this;
    }

    perspective(fov: number, aspect: number, near: number, far: number) {
        mat4.perspective(this.internalMat4, fov, aspect, near, far);
        this.updateBuffer();
        return this;
    }

    multiply(a: Matrix4, b: Matrix4): Matrix4 {
        mat4.multiply(this.internalMat4, a.internalMat4, b.internalMat4);
        this.updateBuffer();
        return this;
    }

    rotateX(angle: number): Matrix4 {
        mat4.rotateX(this.internalMat4, this.internalMat4, angle);
        this.updateBuffer();
        return this;
    }

    rotateY(angle: number): Matrix4 {
        mat4.rotateY(this.internalMat4, this.internalMat4, angle);
        this.updateBuffer();
        return this;
    }

    rotateZ(angle: number): Matrix4 {
        mat4.rotateZ(this.internalMat4, this.internalMat4, angle);
        this.updateBuffer();
        return this;
    }

    scale(v: Vector3): Matrix4 {
        mat4.scale(this.internalMat4, this.internalMat4, v.internalVec3);
        this.updateBuffer();
        return this;
    }

    translate(v: Vector3): Matrix4 {
        mat4.translate(this.internalMat4, this.internalMat4, v.internalVec3);
        this.updateBuffer();
        return this;
    }

    rotate(v: Vector3): Matrix4 {


        mat4.rotate(this.internalMat4, this.internalMat4, v.z, this.forward);
        mat4.rotate(this.internalMat4, this.internalMat4, v.y, this.up);
        mat4.rotate(this.internalMat4, this.internalMat4, v.x, this.right);
        this.updateBuffer();
        return this;
    }

    identity(): Matrix4 {
        mat4.identity(this.internalMat4);
        this.updateBuffer();
        return this;
    }
    copy(a: Matrix4): Matrix4 {
        mat4.copy(this.internalMat4, a.internalMat4);
        this.updateBuffer();
        return this;
    }

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
