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
 *
 * Every renderable in the tree holds a matrix slot (`slotOf`) from the time `prepare` first
 * finds it until a `prepare` no longer does: hiding, showing or re-sorting renderables never
 * moves another one to a different slot, so cached render bundles keep pairing each draw with
 * its own matrices. This is the Rust engine's scene index, for a tree.
 */
class Scene extends Object3D {
    private opaqueObjects: Renderable[] = [];
    private transmissiveObjects: Renderable[] = [];
    private transparentObjects: Renderable[] = [];
    private orderedObjects: Renderable[] = [];
    /** Number of visible opaque renderables (`opaque.length`), at the front of `getOrderedObjects()`. */
    public opaqueCount: number = 0;
    /** Number of visible transmissive renderables (`transmissive.length`), after the opaque ones. */
    public transmissiveCount: number = 0;
    private _directionalLights: DirectionalLight[] = [];
    private _pointLights: PointLight[] = [];
    private _areaLights: AreaLight[] = [];

    // Slot allocator: each renderable's slot and the `prepare` that last found it.
    private _slots = new Map<Renderable, { slot: number, seen: number }>();
    private _freeSlots: number[] = [];
    private _slotCapacity: number = 0;
    private _prepareCount: number = 0;

    public get directionalLights(): readonly DirectionalLight[] { return this._directionalLights; }
    public get pointLights(): readonly PointLight[] { return this._pointLights; }
    public get areaLights(): readonly AreaLight[] { return this._areaLights; }

    /** The visible opaque renderables, in tree order. */
    public get opaque(): readonly Renderable[] { return this.opaqueObjects; }
    /** The visible transmissive renderables, in tree order. */
    public get transmissive(): readonly Renderable[] { return this.transmissiveObjects; }
    /** The visible transparent renderables, by `renderOrder` then back to front. */
    public get transparent(): readonly Renderable[] { return this.transparentObjects; }

    /** One more than the highest slot ever handed out: the slots the mesh buffers must hold. */
    public get slotCapacity(): number { return this._slotCapacity; }

    /** The matrix slot of `renderable` (visible or not), or -1 if the last `prepare` did not find it. */
    public slotOf(renderable: Renderable): number {
        return this._slots.get(renderable)?.slot ?? -1;
    }

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
        const stamp = ++this._prepareCount;
        let found = 0;
        // Give every renderable a slot; sort the visible ones into opaque, transmissive and
        // transparent; collect lights
        this.traverse(this, (object: Object3D) => {
            if ((object as any).isLight) {
                const light = object as Light;
                if (light.lightType === 'directional') this._directionalLights.push(light as DirectionalLight);
                else if (light.lightType === 'point') this._pointLights.push(light as PointLight);
                else if (light.lightType === 'area') this._areaLights.push(light as AreaLight);
            }
            if (object.isRenderable) {
                const renderable = object as Renderable;
                const entry = this._slots.get(renderable);
                if (entry === undefined) {
                    const slot = this._freeSlots.length > 0 ? this._freeSlots.pop()! : this._slotCapacity++;
                    this._slots.set(renderable, { slot, seen: stamp });
                    found++;
                } else if (entry.seen !== stamp) {
                    entry.seen = stamp;
                    found++;
                }
                if (!renderable.visible) return;
                if (renderable.material.transparent) {
                    this.transparentObjects.push(renderable);
                } else if (renderable.material.transmissive) {
                    this.transmissiveObjects.push(renderable);
                } else {
                    this.opaqueObjects.push(renderable);
                }
            }
        });

        // Free the slots of renderables that left the tree
        if (this._slots.size > found) {
            for (const [renderable, entry] of this._slots) {
                if (entry.seen !== stamp) {
                    this._slots.delete(renderable);
                    this._freeSlots.push(entry.slot);
                }
            }
        }

        // Sort transparent objects by renderOrder first, then back-to-front
        const cameraPosition = camera.position;
        this.transparentObjects.sort((a, b) => {
            if (a.renderOrder !== b.renderOrder) return a.renderOrder - b.renderOrder;
            const distA = a.position.distanceToSquared(cameraPosition);
            const distB = b.position.distanceToSquared(cameraPosition);
            return distB - distA;
        });

        // Build ordered list in-place: opaque → transmissive → transparent.
        this.orderedObjects.length = 0;
        for (let i = 0; i < this.opaqueObjects.length; i++) this.orderedObjects.push(this.opaqueObjects[i]);
        for (let i = 0; i < this.transmissiveObjects.length; i++) this.orderedObjects.push(this.transmissiveObjects[i]);
        for (let i = 0; i < this.transparentObjects.length; i++) this.orderedObjects.push(this.transparentObjects[i]);

        this.opaqueCount = this.opaqueObjects.length;
        this.transmissiveCount = this.transmissiveObjects.length;
    }

    /** The visible renderables: opaque, then transmissive, then transparent. */
    public getOrderedObjects(): Renderable[] {
        return this.orderedObjects;
    }
}

export { Scene }
