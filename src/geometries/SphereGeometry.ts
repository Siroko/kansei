import { mat4, vec3 } from "gl-matrix";
import { Geometry } from "../buffers/Geometry";

class SphereGeometry extends Geometry {
    // buffers
    private _indices: number[] = [];
    private _vertices: number[] = [];
    private numberOfVertices: number = 0;

    public vertexCount: number = 0;

    constructor(radius: number = 1, segments: number = 5) {
        super();

        const matRotZ = mat4.create();
        const matRotY = mat4.create();
        const tmpVec3 = vec3.create();
        const up = vec3.fromValues(0, 1, 0);

        const totalXRotationSteps = 2 + segments
        const totalYRotationSteps = 2 * totalXRotationSteps

        for (let zRotationStep = 0; zRotationStep <= totalXRotationSteps; zRotationStep++) {
            const normalizedZ: number = zRotationStep / totalXRotationSteps;
            const angleZ: number = (normalizedZ * Math.PI);

            for (let yRotationStep = 0; yRotationStep <= totalYRotationSteps; yRotationStep++) {
                const normalizedY: number = yRotationStep / totalYRotationSteps;
                const angleY: number = normalizedY * Math.PI * 2;

                mat4.identity(matRotZ)
                mat4.rotateZ(matRotZ, matRotZ, -angleZ)

                mat4.identity(matRotY)
                mat4.rotateY(matRotY, matRotY, angleY)

                vec3.transformMat4(tmpVec3, up, matRotZ)
                vec3.transformMat4(tmpVec3, tmpVec3, matRotY)

                vec3.scale(tmpVec3, tmpVec3, -radius)
                this._vertices.push(tmpVec3[0], tmpVec3[1], tmpVec3[2], 1)

                vec3.normalize(tmpVec3, tmpVec3)
                this._vertices.push(tmpVec3[0], tmpVec3[1], tmpVec3[2])

                this._vertices.push(normalizedY, normalizedZ)

                this.numberOfVertices += 1;
            }

            if (zRotationStep > 0) {
                const verticesCount = this.numberOfVertices;
                let firstIndex = verticesCount - 2 * (totalYRotationSteps + 1)
                for (; (firstIndex + totalYRotationSteps + 2) < verticesCount; firstIndex++) {
                    this._indices.push(
                        firstIndex,
                        firstIndex + 1,
                        firstIndex + totalYRotationSteps + 1,
                        firstIndex + totalYRotationSteps + 1,
                        firstIndex + 1,
                        firstIndex + totalYRotationSteps + 2
                    );
                }
            }
        }

        this.vertexCount = this._indices.length;

        this.vertices = new Float32Array(this._vertices);
        this.indices = new Uint16Array(this._indices);
    }
}

export { SphereGeometry };
