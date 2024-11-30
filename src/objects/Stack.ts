import { Object3D } from "./Object3D";
import { Mesh } from "./Mesh";
import { Camera } from "../cameras/Camera";

/**
 * Represents a 3D stack which can contain multiple objects.
 * Extends the Object3D class to inherit transformation properties.
 */
class Stack extends Object3D {
    private opaqueObjects: Mesh[] = [];
    private transparentObjects: Mesh[] = [];

    /**
     * Constructs a new Stack object.
     */
    constructor() {
        super();
    }

    public prepare(camera: Camera) {
        // Clear arrays
        this.opaqueObjects = [];
        this.transparentObjects = [];
        // Sort objects into opaque and transparent
        this.traverse(this, (object: Object3D) => {
            if (object.isMesh) {
                const mesh = object as Mesh;
                if (mesh.material.transparent) {
                    this.transparentObjects.push(mesh);
                } else {
                    this.opaqueObjects.push(mesh);
                }
            }
        });

        // Sort transparent objects back-to-front
        const cameraPosition = camera.position;
        this.transparentObjects.sort((a, b) => {
            const distA = a.position.distanceTo(cameraPosition);
            const distB = b.position.distanceTo(cameraPosition);
            return distB - distA;
        });
    }

    public getOrderedObjects(): Mesh[] {
        return [...this.opaqueObjects, ...this.transparentObjects];
    }
}

export { Stack }
