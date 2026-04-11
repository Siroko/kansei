import { Object3D } from "./Object3D";
import { Camera } from "../cameras/Camera";
import { Renderable } from "./Renderable";
import { Light } from "../lights/Light";
import { DirectionalLight } from "../lights/DirectionalLight";
import { PointLight } from "../lights/PointLight";
import { AreaLight } from "../lights/AreaLight";

/**
 * Represents a 3D scene which can contain multiple objects.
 * Extends the Object3D class to inherit transformation properties.
 */
class Scene extends Object3D {
    private opaqueObjects: Renderable[] = [];
    private transmissiveObjects: Renderable[] = [];
    private transparentObjects: Renderable[] = [];
    private orderedObjects: Renderable[] = [];
    /** Number of opaque objects at the front of orderedObjects. The renderer
     *  draws `[0, opaqueCount)` into the GBuffer, copies colour to background,
     *  then draws `[opaqueCount, opaqueCount + transmissiveCount)`. */
    public opaqueCount: number = 0;
    public transmissiveCount: number = 0;
    private _directionalLights: DirectionalLight[] = [];
    private _pointLights: PointLight[] = [];
    private _areaLights: AreaLight[] = [];

    public get directionalLights(): readonly DirectionalLight[] { return this._directionalLights; }
    public get pointLights(): readonly PointLight[] { return this._pointLights; }
    public get areaLights(): readonly AreaLight[] { return this._areaLights; }

    /**
     * Constructs a new Scene object.
     */
    constructor() {
        super();
    }

    public prepare(camera: Camera) {
        // Clear arrays in-place to avoid allocation
        this.opaqueObjects.length = 0;
        this.transmissiveObjects.length = 0;
        this.transparentObjects.length = 0;
        this._directionalLights.length = 0;
        this._pointLights.length = 0;
        this._areaLights.length = 0;
        // Sort objects into opaque, transmissive, and transparent; collect lights
        this.traverse(this, (object: Object3D) => {
            if ((object as any).isLight) {
                const light = object as Light;
                if (light.lightType === 'directional') this._directionalLights.push(light as DirectionalLight);
                else if (light.lightType === 'point') this._pointLights.push(light as PointLight);
                else if (light.lightType === 'area') this._areaLights.push(light as AreaLight);
            }
            if (object.isRenderable) {
                const renderable = object as Renderable;
                if (renderable.material.transparent) {
                    this.transparentObjects.push(renderable);
                } else if (renderable.material.transmissive) {
                    this.transmissiveObjects.push(renderable);
                } else {
                    this.opaqueObjects.push(renderable);
                }
            }
        });

        // Sort transparent objects by renderOrder first, then back-to-front
        const cameraPosition = camera.position;
        this.transparentObjects.sort((a, b) => {
            if (a.renderOrder !== b.renderOrder) return a.renderOrder - b.renderOrder;
            const distA = a.position.distanceToSquared(cameraPosition);
            const distB = b.position.distanceToSquared(cameraPosition);
            return distB - distA;
        });

        // Build ordered list in-place: opaque → transmissive → transparent.
        // The renderer relies on this ordering to split the GBuffer render pass
        // at the transmissive boundary (background capture happens between them).
        this.orderedObjects.length = 0;
        for (let i = 0; i < this.opaqueObjects.length; i++) this.orderedObjects.push(this.opaqueObjects[i]);
        for (let i = 0; i < this.transmissiveObjects.length; i++) this.orderedObjects.push(this.transmissiveObjects[i]);
        for (let i = 0; i < this.transparentObjects.length; i++) this.orderedObjects.push(this.transparentObjects[i]);

        this.opaqueCount = this.opaqueObjects.length;
        this.transmissiveCount = this.transmissiveObjects.length;
    }

    public getOrderedObjects(): Renderable[] {
        return this.orderedObjects;
    }
}

export { Scene }
