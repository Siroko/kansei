import { Geometry } from "../buffers/Geometry";

export class PlaneGeometry extends Geometry {
    // buffers

    private _indices: number[] = [];
    private _vertices: number[] = [];

    // helper variables

    private numberOfVertices: number = 0;

    public vertexCount: number = 0;

    constructor(width: number, height: number, segmentsX: number = 1, segmentsY: number = 1) {
        super();

        // Calculate vertex spacing
        const segmentWidth = width / segmentsX;
        const segmentHeight = height / segmentsY;

        // Generate vertices, normals, and UVs
        for (let y = 0; y <= segmentsY; y++) {
            const yPos = (y * segmentHeight) - (height / 2);

            for (let x = 0; x <= segmentsX; x++) {
                const xPos = (x * segmentWidth) - (width / 2);
                // Add vertex position (x, y, 0)
                this._vertices.push(xPos, yPos, 0, 1);
                // Add normal (0, 0, 1) for front face
                this._vertices.push(0, 0, 1);
                // Add UV coordinates
                this._vertices.push(x / segmentsX, y / segmentsY);

                this.numberOfVertices++;
            }
        }

        // Generate indices for triangles
        for (let y = 0; y < segmentsY; y++) {
            for (let x = 0; x < segmentsX; x++) {
                const a = x + (y * (segmentsX + 1));
                const b = x + ((y + 1) * (segmentsX + 1));
                const c = (x + 1) + (y * (segmentsX + 1));
                const d = (x + 1) + ((y + 1) * (segmentsX + 1));

                // First triangle
                this._indices.push(a, b, c);
                // Second triangle
                this._indices.push(c, b, d);
            }
        }

        this.vertexCount = this._indices.length;

        this.vertexCount = this._indices.length;

        this.vertices = new Float32Array(this._vertices);
        this.indices = new Uint16Array(this._indices);
    }
}