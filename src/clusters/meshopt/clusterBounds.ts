/**
 * A cluster's bounding sphere and backface-culling normal cone: a TypeScript port of
 * `optimesh` 1.1's `meshletutils::compute_cluster_bounds` (meshoptimizer v1.1, MIT). The sphere
 * uses the extremum-pair seeding of Larsson, "Fast and Tight Fitting Bounding Spheres" (2008),
 * refined by Ritter's pass; the cone follows Chajdas, "GeometryFX 1.2 Cluster Culling" (2016).
 * Every float operation is rounded to f32, as Rust computes it.
 */

const f = Math.fround;
const F32_MAX = 3.4028234663852886e38;

/** A cluster's bounding sphere and normal cone. The 8-bit quantized cone of Rust's `Bounds` is left out. */
export interface ClusterBounds {
    center: [number, number, number];
    radius: number;
    coneApex: [number, number, number];
    /** Unit length, or zero when the cone is too wide to cull. */
    coneAxis: [number, number, number];
    /** The sine of the cone's half-angle (its cutoff widened by 90°); 1 when it culls nothing. */
    coneCutoff: number;
}

const D = f(0.57735026);
/** The three coordinate axes and the four cube diagonals, normalized. */
const AXES = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [D, D, D], [-D, D, D], [D, -D, D], [D, D, -D]];

/**
 * A bounding sphere `[cx, cy, cz, radius]` of `count` points of `data` (`stride` floats apart),
 * through `indices` when given: the widest extremum pair along `axisCount` axes seeds it, and a
 * pass grows it to enclose every point. Rust's points carry radii; the cluster bounds' are all 0,
 * which leaves every expression here as Rust rounds it.
 */
function boundingSphere(data: Float32Array, stride: number, count: number, axisCount: number, indices: Uint32Array | null): [number, number, number, number] {
    const vertex = (i: number) => indices ? indices[i] : i;
    const pointMin = new Array<number>(7).fill(0), pointMax = new Array<number>(7).fill(0);
    const tMin = new Array<number>(7).fill(F32_MAX), tMax = new Array<number>(7).fill(-F32_MAX);
    for (let i = 0; i < count; i++) {
        const v = vertex(i), o = v * stride;
        for (let axis = 0; axis < axisCount; axis++) {
            const ax = AXES[axis];
            const projection = f(f(f(ax[0] * data[o]) + f(ax[1] * data[o + 1])) + f(ax[2] * data[o + 2]));
            // (radius 0: low and high are the projection)
            if (projection < tMin[axis]) {
                pointMin[axis] = v;
                tMin[axis] = projection;
            }
            if (projection > tMax[axis]) {
                pointMax[axis] = v;
                tMax[axis] = projection;
            }
        }
    }
    const span2 = (a: number, b: number) => {
        const dx = f(data[b * stride] - data[a * stride]), dy = f(data[b * stride + 1] - data[a * stride + 1]), dz = f(data[b * stride + 2] - data[a * stride + 2]);
        return f(f(f(dx * dx) + f(dy * dy)) + f(dz * dz));
    };
    // the axis whose extremum pair is farthest apart
    let bestAxis = 0, bestSpan = 0;
    for (let axis = 0; axis < axisCount; axis++) {
        const span = f(Math.sqrt(span2(pointMin[axis], pointMax[axis])));
        if (span > bestSpan) {
            bestSpan = span;
            bestAxis = axis;
        }
    }
    const p1 = pointMin[bestAxis] * stride, p2 = pointMax[bestAxis] * stride;
    const segment = f(Math.sqrt(span2(pointMin[bestAxis], pointMax[bestAxis])));
    const along = segment > 0 ? f(segment / f(2 * segment)) : 0;
    const center = [0, 1, 2].map((k) => f(data[p1 + k] + f(f(data[p2 + k] - data[p1 + k]) * along)));
    let radius = f(bestSpan / 2);
    for (let i = 0; i < count; i++) {
        const o = vertex(i) * stride;
        const dx = f(data[o] - center[0]), dy = f(data[o + 1] - center[1]), dz = f(data[o + 2] - center[2]);
        const d = f(Math.sqrt(f(f(f(dx * dx) + f(dy * dy)) + f(dz * dz))));
        if (d > radius) {
            const k = d > 0 ? f(f(d - radius) / f(2 * d)) : 0;
            for (let a = 0; a < 3; a++) center[a] = f(center[a] + f(k * f(data[o + a] - center[a])));
            radius = f(f(radius + d) / 2);
        }
    }
    return [center[0], center[1], center[2], radius];
}

/**
 * The bounds of a cluster of triangles, `indices` (3 per triangle, at most 512 triangles)
 * into `positions` (`stride` floats apart). Rust: `meshletutils::compute_cluster_bounds`.
 */
export function computeClusterBounds(indices: Uint32Array, positions: Float32Array, stride: number): ClusterBounds {
    // a small direct-mapped cache collects the cluster's distinct vertices
    const cache = new Uint32Array(512).fill(0xffffffff);
    const corners: number[] = [];
    for (const v of indices) {
        const slot = v & 511;
        if (cache[slot] !== v) corners.push(v);
        cache[slot] = v;
    }

    const normals = new Float32Array((indices.length / 3) * 4);
    let triangleCount = 0;
    for (let t = 0; t < indices.length / 3; t++) {
        const a = indices[t * 3] * stride, b = indices[t * 3 + 1] * stride, c = indices[t * 3 + 2] * stride;
        const e1x = f(positions[b] - positions[a]), e1y = f(positions[b + 1] - positions[a + 1]), e1z = f(positions[b + 2] - positions[a + 2]);
        const e2x = f(positions[c] - positions[a]), e2y = f(positions[c + 1] - positions[a + 1]), e2z = f(positions[c + 2] - positions[a + 2]);
        let nx = f(f(e1y * e2z) - f(e1z * e2y));
        let ny = f(f(e1z * e2x) - f(e1x * e2z));
        let nz = f(f(e1x * e2y) - f(e1y * e2x));
        const area = f(Math.sqrt(f(f(f(nx * nx) + f(ny * ny)) + f(nz * nz))));
        // degenerate triangles are invisible, so they do not constrain the cone
        if (area === 0) continue;
        nx = f(nx / area);
        ny = f(ny / area);
        nz = f(nz / area);
        const o = triangleCount * 4;
        normals[o] = nx;
        normals[o + 1] = ny;
        normals[o + 2] = nz;
        normals[o + 3] = -f(f(f(nx * positions[a]) + f(ny * positions[a + 1])) + f(nz * positions[a + 2]));
        triangleCount++;
    }

    const bounds: ClusterBounds = { center: [0, 0, 0], radius: 0, coneApex: [0, 0, 0], coneAxis: [0, 0, 0], coneCutoff: 0 };
    if (triangleCount === 0) return bounds;

    const sphere = boundingSphere(positions, stride, corners.length, 7, Uint32Array.from(corners));
    const center: [number, number, number] = [sphere[0], sphere[1], sphere[2]];

    // fitting a sphere to the triangle normals gives the best cone axis
    const normalSphere = boundingSphere(normals, 4, triangleCount, 3, null);
    const axis: [number, number, number] = [normalSphere[0], normalSphere[1], normalSphere[2]];
    const axisLength = f(Math.sqrt(f(f(f(axis[0] * axis[0]) + f(axis[1] * axis[1])) + f(axis[2] * axis[2]))));
    const inv = axisLength === 0 ? 0 : f(1 / axisLength);
    for (let k = 0; k < 3; k++) axis[k] = f(axis[k] * inv);

    // tightest cosine of any normal against the axis (half the cone opening)
    let minDot = 1;
    for (let i = 0; i < triangleCount; i++) {
        const o = i * 4;
        const dot = f(f(f(normals[o] * axis[0]) + f(normals[o + 1] * axis[1])) + f(normals[o + 2] * axis[2]));
        if (dot < minDot) minDot = dot;
    }

    bounds.center = center;
    bounds.radius = sphere[3];
    // a cone wider than ~168 degrees is not worth testing, so accept everything
    if (minDot <= f(0.1)) {
        bounds.coneCutoff = 1;
        return bounds;
    }

    // push the apex back along the axis until it is behind every triangle plane
    let maxT = 0;
    for (let i = 0; i < triangleCount; i++) {
        const o = i * 4;
        const toCenter = f(f(f(f(center[0] * normals[o]) + f(center[1] * normals[o + 1])) + f(center[2] * normals[o + 2])) + normals[o + 3]);
        const alongAxis = f(f(f(axis[0] * normals[o]) + f(axis[1] * normals[o + 1])) + f(axis[2] * normals[o + 2]));
        const t = f(toCenter / alongAxis);
        if (t > maxT) maxT = t;
    }
    bounds.coneApex = [f(center[0] - f(axis[0] * maxT)), f(center[1] - f(axis[1] * maxT)), f(center[2] - f(axis[2] * maxT))];
    bounds.coneAxis = axis;
    // widen by 90 degrees on each side and invert: -cos(a + 90) = sin(a)
    bounds.coneCutoff = f(Math.sqrt(f(1 - f(minDot * minDot))));
    return bounds;
}
