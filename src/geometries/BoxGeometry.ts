import { Geometry } from "../buffers/Geometry";
import { Vector4 } from "../math/Vector4";

/**
 * Class representing a box geometry.
 * @extends Geometry
 */
class BoxGeometry extends Geometry {
    // buffers

    private _indices: number[] = [];
    private _vertices: number[] = [];

    // helper variables

    private numberOfVertices: number = 0;

    public vertexCount: number = 0;

    /**
     * Creates an instance of BoxGeometry.
     * @param {number} [width=1] - The width of the box.
     * @param {number} [height=1] - The height of the box.
     * @param {number} [depth=1] - The depth of the box.
     * @param {number} [widthSegments=1] - Number of segmented faces along the width of the box.
     * @param {number} [heightSegments=1] - Number of segmented faces along the height of the box.
     * @param {number} [depthSegments=1] - Number of segmented faces along the depth of the box.
     */
    constructor(width = 1, height = 1, depth = 1, widthSegments = 1, heightSegments = 1, depthSegments = 1) {

        super();

        // segments

        widthSegments = Math.floor(widthSegments);
        heightSegments = Math.floor(heightSegments);
        depthSegments = Math.floor(depthSegments);

        // build each side of the box geometry

        this.buildPlane(2, 1, 0, - 1, - 1, depth, height, width, depthSegments, heightSegments); // px
        this.buildPlane(2, 1, 0, 1, - 1, depth, height, - width, depthSegments, heightSegments); // nx
        this.buildPlane(0, 2, 1, 1, 1, width, depth, height, widthSegments, depthSegments); // py
        this.buildPlane(0, 2, 1, 1, - 1, width, depth, - height, widthSegments, depthSegments); // ny
        this.buildPlane(0, 1, 2, 1, - 1, width, height, depth, widthSegments, heightSegments); // pz
        this.buildPlane(0, 1, 2, - 1, - 1, width, height, - depth, widthSegments, heightSegments); // nz

        this.vertexCount = this._indices.length;

        this.vertices = new Float32Array(this._vertices);
        this.indices = new Uint16Array(this._indices);
    }

    /**
     * Builds a plane for the box geometry.
     * @private
     * @param {number} u - The u-axis.
     * @param {number} v - The v-axis.
     * @param {number} w - The w-axis.
     * @param {number} udir - The direction of the u-axis.
     * @param {number} vdir - The direction of the v-axis.
     * @param {number} width - The width of the plane.
     * @param {number} height - The height of the plane.
     * @param {number} depth - The depth of the plane.
     * @param {number} gridX - Number of grid segments along the x-axis.
     * @param {number} gridY - Number of grid segments along the y-axis.
     */
    private buildPlane(u: number, v: number, w: number, udir: number, vdir: number, width: number, height: number, depth: number, gridX: number, gridY: number) {

        const segmentWidth = width / gridX;
        const segmentHeight = height / gridY;

        const widthHalf = width / 2;
        const heightHalf = height / 2;
        const depthHalf = depth / 2;

        const gridX1 = gridX + 1;
        const gridY1 = gridY + 1;

        let vertexCounter = 0;

        const vector = new Vector4();

        // generate vertices, normals and uvs

        for (let iy = 0; iy < gridY1; iy++) {

            const y = iy * segmentHeight - heightHalf;

            for (let ix = 0; ix < gridX1; ix++) {

                const x = ix * segmentWidth - widthHalf;

                // Use type assertions to avoid errors
                vector.setComponent(u, x * udir);
                vector.setComponent(v, y * vdir);
                vector.setComponent(w, depthHalf);

                // now apply vector to vertex buffer
                this._vertices.push(vector.x, vector.y, vector.z, 1);

                // set values to correct vector component
                vector.setComponent(u, 0);
                vector.setComponent(v, 0);
                vector.setComponent(w, depth > 0 ? 1 : -1);

                // now apply vector to normal buffer
                this._vertices.push(vector.x, vector.y, vector.z);

                // uvs
                this._vertices.push(ix / gridX);
                this._vertices.push(1 - (iy / gridY));

                // counters
                vertexCounter += 1;

            }

        }

        // indices

        // 1. you need three indices to draw a single face
        // 2. a single segment consists of two faces
        // 3. so we need to generate six (2*3) indices per segment

        for (let iy = 0; iy < gridY; iy++) {

            for (let ix = 0; ix < gridX; ix++) {

                const a = this.numberOfVertices + ix + gridX1 * iy;
                const b = this.numberOfVertices + ix + gridX1 * (iy + 1);
                const c = this.numberOfVertices + (ix + 1) + gridX1 * (iy + 1);
                const d = this.numberOfVertices + (ix + 1) + gridX1 * iy;

                // faces

                this._indices.push(a, b, d);
                this._indices.push(b, c, d);
            }

        }

        this.numberOfVertices += vertexCounter;
    }
}

export { BoxGeometry };
