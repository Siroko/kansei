import { mat4 } from "gl-matrix";
import { Geometry } from "../buffers/Geometry";
import { CylinderGeometry } from "./CylinderGeometry";

/**
 * A spruce 1 high standing on y = 0, for forests (thousands placed and scaled per instance): a
 * trunk under `cones` stacked cones of `segments` x `rings` quads, each narrower and shorter than
 * the one below, as one merged mesh. The trunk is within 0.04 of the axis and the crowns outside
 * it, so a shader can tell bark from needles by `length(position.xz)`. Fewer segments, rings and
 * cones make its mesh LODs.
 *
 * The same mesh as the Rust engine's `SpruceGeometry` (`rust/kansei-core/src/geometries/spruce.rs`).
 */
class SpruceGeometry extends Geometry {
    constructor(segments: number = 12, rings: number = 2, cones: number = 4) {
        super();
        const parts: [Geometry, mat4][] = [[new CylinderGeometry(0.035, 0.025, 0.3, Math.min(segments, 8), 1), mat4.create()]];
        for (let k = 0; k < cones; k++) {
            const f = k / cones;
            const y0 = 0.15 + 0.62 * f;
            const y1 = k + 1 === cones ? 1 : y0 + 0.42 - 0.12 * f;
            const cone = new CylinderGeometry(0.24 * (1 - 0.55 * f), 0, y1 - y0, segments, rings);
            parts.push([cone, mat4.fromTranslation(mat4.create(), [0, y0, 0])]);
        }
        const merged = Geometry.merged('Spruce', parts);
        this.setArrays('Spruce', merged.vertices as Float32Array, merged.indices as Uint32Array);
    }
}

export { SpruceGeometry };
