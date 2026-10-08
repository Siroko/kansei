import { Geometry } from "../buffers/Geometry";

/**
 * A terrain from a height function: a grid of `cells[0]` x `cells[1]` quads over the rectangle
 * `min`..`max` (x, z), each vertex at `height(x, z)`, with normals from the function's slope
 * and uvs spanning 0..1. Front faces point up.
 *
 * The same mesh as the Rust engine's `HeightfieldGeometry` (`rust/kansei-core/src/geometries/heightfield.rs`).
 */
class HeightfieldGeometry extends Geometry {
    constructor(min: [number, number], max: [number, number], cells: [number, number], height: (x: number, z: number) => number) {
        super();
        const nx = Math.max(1, Math.floor(cells[0]));
        const nz = Math.max(1, Math.floor(cells[1]));
        const step = [(max[0] - min[0]) / nx, (max[1] - min[1]) / nz];
        // the slope over a fraction of a cell
        const ex = step[0] * 0.25;
        const ez = step[1] * 0.25;
        const vertices = new Float32Array((nx + 1) * (nz + 1) * 9);
        let o = 0;
        for (let j = 0; j <= nz; j++) {
            for (let i = 0; i <= nx; i++) {
                const x = min[0] + i * step[0];
                const z = min[1] + j * step[1];
                const dx = (height(x + ex, z) - height(x - ex, z)) / (2 * ex);
                const dz = (height(x, z + ez) - height(x, z - ez)) / (2 * ez);
                const length = Math.hypot(dx, 1, dz);
                vertices.set([x, height(x, z), z, 1, -dx / length, 1 / length, -dz / length, i / nx, j / nz], o);
                o += 9;
            }
        }
        const at = (i: number, j: number) => j * (nx + 1) + i;
        const indices = new Uint32Array(nx * nz * 6);
        o = 0;
        for (let j = 0; j < nz; j++) {
            for (let i = 0; i < nx; i++) {
                const [p00, p10, p01, p11] = [at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1)];
                indices.set([p00, p01, p10, p10, p01, p11], o);
                o += 6;
            }
        }
        this.setArrays('HeightfieldGeometry', vertices, indices);
    }
}

export { HeightfieldGeometry };
