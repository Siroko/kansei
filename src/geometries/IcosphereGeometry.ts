import { Geometry } from "../buffers/Geometry";

/**
 * A sphere of near-equal triangles: an icosahedron whose faces are split in four
 * `subdivisions` times (20 · 4ⁿ triangles), its vertices pushed out to `radius`. Unlike
 * `SphereGeometry` it has no poles, so it suits displacement and cluster LOD.
 *
 * The same mesh as the Rust engine's `IcosphereGeometry` (`rust/kansei-core/src/geometries/icosphere.rs`).
 */
class IcosphereGeometry extends Geometry {
    constructor(radius: number = 1, subdivisions: number = 2) {
        super();
        const t = (1 + Math.sqrt(5)) / 2;
        const points: [number, number, number][] = ([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ] as [number, number, number][]).map(normalize);
        let faces: [number, number, number][] = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ];
        for (let n = 0; n < subdivisions; n++) {
            const midpoints = new Map<number, number>();
            const mid = (x: number, y: number) => {
                const key = Math.min(x, y) * 2 ** 26 + Math.max(x, y);
                let index = midpoints.get(key);
                if (index === undefined) {
                    const [a, b] = [points[x], points[y]];
                    points.push(normalize([(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5]));
                    index = points.length - 1;
                    midpoints.set(key, index);
                }
                return index;
            };
            const next: [number, number, number][] = [];
            for (const [a, b, c] of faces) {
                const [ab, bc, ca] = [mid(a, b), mid(b, c), mid(c, a)];
                next.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
            }
            faces = next;
        }
        const vertices = new Float32Array(points.length * 9);
        points.forEach(([x, y, z], i) => {
            vertices.set([x * radius, y * radius, z * radius, 1, x, y, z, x * 0.5 + 0.5, y * 0.5 + 0.5], i * 9);
        });
        this.setArrays('IcosphereGeometry', vertices, new Uint32Array(faces.flat()));
    }
}

function normalize([x, y, z]: [number, number, number]): [number, number, number] {
    const length = Math.hypot(x, y, z);
    return [x / length, y / length, z / length];
}

export { IcosphereGeometry };
