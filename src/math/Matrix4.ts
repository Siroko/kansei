import { mat4, vec3 } from "gl-matrix";
import { BufferBase } from "../buffers/BufferBase";
import { Vector3 } from "./Vector3";

class Matrix4 extends BufferBase {
    public type: string = BufferBase.BUFFER_TYPE_UNIFORM;
    public readonly internalMat4: mat4;

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

    fromTranslation(v: vec3): Matrix4 {
        mat4.fromTranslation(this.internalMat4, v);
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

    copy(a: Matrix4): Matrix4 {
        mat4.copy(this.internalMat4, a.internalMat4);
        this.updateBuffer();
        return this;
    }

    lookAt(position: Vector3, target: Vector3, up: Vector3) {
        mat4.lookAt(this.internalMat4, position.toVec(), target.toVec(), up.toVec());
        this.updateBuffer();
        return this;
    }
}

export { Matrix4 }
