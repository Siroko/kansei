import { Object3D } from "./Object3D";
import { Camera } from "../cameras/Camera";
import { Renderable } from "./Renderable";

/**
 * Represents a 3D scene which can contain multiple objects.
 * Extends the Object3D class to inherit transformation properties.
 */
class Scene extends Object3D {
    private opaqueObjects: Renderable[] = [];
    private transparentObjects: Renderable[] = [];

    /**
     * Constructs a new Scene object.
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
            if (object.isRenderable) {
                const renderable = object as Renderable;
                if (renderable.material.transparent) {
                    this.transparentObjects.push(renderable);
                } else {
                    this.opaqueObjects.push(renderable);
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

    public getOrderedObjects(): Renderable[] {
        return [...this.opaqueObjects, ...this.transparentObjects];
    }
}

export { Scene }
