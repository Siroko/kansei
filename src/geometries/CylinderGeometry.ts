import { Geometry } from "../buffers/Geometry";

/**
 * A (truncated) cone or cylinder round the y axis, standing on y = 0: radius `radiusBottom` at
 * the bottom, `radiusTop` at `height`, in `rings` bands of `segments` quads, with a cap on each
 * end whose radius is not zero. A `radiusTop` of 0 makes a cone.
 *
 * The same mesh as the Rust engine's `CylinderGeometry` (`rust/kansei-core/src/geometries/cylinder.rs`).
 */
class CylinderGeometry extends Geometry {
    constructor(radiusBottom: number, radiusTop: number, height: number, segments: number = 16, rings: number = 1) {
        super();
        segments = Math.max(3, Math.floor(segments));
        rings = Math.max(1, Math.floor(rings));
        const vertices: number[] = [];
        const indices: number[] = [];
        // the side's normal leans by the slope of its radius
        const slope = radiusBottom - radiusTop;
        for (let k = 0; k <= rings; k++) {
            const f = k / rings;
            const y = height * f;
            const r = radiusBottom + (radiusTop - radiusBottom) * f;
            for (let s = 0; s <= segments; s++) {
                const a = (s / segments) * Math.PI * 2;
                const [nx, ny, nz] = [Math.cos(a) * height, slope, Math.sin(a) * height];
                const length = Math.hypot(nx, ny, nz) || 1;
                vertices.push(r * Math.cos(a), y, r * Math.sin(a), 1, nx / length, ny / length, nz / length, s / segments, f);
            }
        }
        const row = segments + 1;
        for (let k = 0; k < rings; k++) {
            for (let s = 0; s < segments; s++) {
                const [a, b, c, d] = [k * row + s, k * row + s + 1, (k + 1) * row + s, (k + 1) * row + s + 1];
                indices.push(a, c, b, b, c, d);
            }
        }
        for (const [y, r, up] of [[0, radiusBottom, false], [height, radiusTop, true]] as [number, number, boolean][]) {
            if (r <= 0) continue;
            const ny = up ? 1 : -1;
            const centre = vertices.length / 9;
            vertices.push(0, y, 0, 1, 0, ny, 0, 0.5, 0.5);
            for (let s = 0; s <= segments; s++) {
                const a = (s / segments) * Math.PI * 2;
                vertices.push(r * Math.cos(a), y, r * Math.sin(a), 1, 0, ny, 0, 0.5 + 0.5 * Math.cos(a), 0.5 + 0.5 * Math.sin(a));
            }
            for (let s = 0; s < segments; s++) {
                const [rim, next] = [centre + 1 + s, centre + 2 + s];
                if (up) indices.push(centre, next, rim);
                else indices.push(centre, rim, next);
            }
        }
        this.setArrays('CylinderGeometry', new Float32Array(vertices), new Uint32Array(indices));
    }
}

export { CylinderGeometry };
