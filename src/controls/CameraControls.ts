/**
 * Created by felixmartinez on 16/02/14.
 * Ported to ES6 on 16/10/2016
 * 
 * * Ported to Typescript on 22/10/2024
 */

import { Vector3 } from '../math/Vector3';
import { Camera } from '../cameras/Camera';

/**
 * Controls the camera movement and interaction with mouse and touch events.
 */
class CameraControls {
    private PI: number = 3.14159265359;
    private camera: Camera;
    private target: Vector3;
    private displacement: { x: number; y: number };
    private prevAngles: { x: number; y: number };
    private currentAngles: { x: number; y: number };
    private finalRadians: { x: number; y: number };
    private downPoint: { x: number; y: number };
    private down: boolean;
    private radius: number;
    private wheelDelta: number;
    private wheelDeltaEase: number;
    private limits: { up: number; down: number };
    private mouseX: number;
    private mouseY: number;
    private _mouseX: number;
    private _mouseY: number;
    private onMove: (() => void) | null;
    private enabled: boolean;
    private offset: { x: number; y: number; z: number };
    private offsetEase: { x: number; y: number; z: number };
    /** Pan from mouse drags (`withMousePan`), added to the target. */
    private pan: { x: number; y: number; z: number } = { x: 0, y: 0, z: 0 };
    private mousePan: boolean = false;
    private panning: boolean = false;
    private panPoint: { x: number; y: number } = { x: 0, y: 0 };
    private readonly lookPoint: Vector3 = new Vector3();
    private contextMenuHandler?: (e: Event) => void;

    private time: number;
    private domElement: HTMLElement | Window;
    /** When true, scroll-wheel zoom direction is inverted
     *  (scroll up → zoom in instead of zoom out). */
    public invertWheel: boolean = false;

    private mouseWheelHandler?: (e: WheelEvent) => void;
    private mouseDownHandler?: (e: MouseEvent) => void;
    private mouseUpHandler?: (e: MouseEvent) => void;
    private mouseMoveHandler?: (e: MouseEvent) => void;
    private touchStartHandler?: (e: TouchEvent) => void;
    private touchEndHandler?: (e: TouchEvent) => void;
    private touchMoveHandler?: (e: TouchEvent) => void;

    /**
     * Creates an instance of CameraControls.
     * @param camera      - The camera to control.
     * @param target      - The target vector for the camera to look at.
     * @param domElement  - The DOM element to attach event listeners to.
     * @param radius      - Initial orbit radius. Defaults to 20.
     * @param options     - Optional behaviour flags.
     * @param options.invertWheel - Invert the scroll-wheel zoom direction.
     */
    constructor(
        camera: Camera,
        target: Vector3,
        domElement: HTMLElement | Window,
        radius: number = 20,
        options: { invertWheel?: boolean } = {},
    ) {
        this.invertWheel = options.invertWheel ?? false;

        this.camera = camera;
        this.target = target;
        this.domElement = domElement;

        this.displacement = { x: 0, y: 0 };
        this.prevAngles = { x: 0.04, y: 0.05 };
        this.currentAngles = { x: this.prevAngles.x, y: this.prevAngles.y };
        this.finalRadians = {
            x: this.prevAngles.x * (this.PI * 2),
            y: this.prevAngles.y * (this.PI * 2)
        };
        this.downPoint = { x: 0, y: 0 };
        this.down = false;

        this.radius = radius;
        this.wheelDelta = this.radius;
        this.wheelDeltaEase = this.radius;

        this.limits = { up: 0.2, down: -0.2 };
        this.mouseX = -1;
        this.mouseY = -1;
        this._mouseX = -1;
        this._mouseY = -1;
        this.onMove = null;
        this.enabled = true;

        this.offset = { x: 0, y: 0, z: 0 };
        this.offsetEase = { x: 0, y: 0, z: 0 };

        this.time = 0;

        this.events();
    }

    /**
     * Initializes event listeners for mouse and touch interactions.
     */
    private events() {
        const domElement = this.domElement || window;

        this.mouseWheelHandler = (e) => this.onMouseWheel(e);
        this.mouseDownHandler = (e) => this.onMouseDown(e);
        this.mouseUpHandler = (e) => this.onMouseUp(e);
        this.mouseMoveHandler = (e) => this.onMouseMove(e);

        this.touchStartHandler = this.onTouchStart.bind(this);
        this.touchEndHandler = this.onTouchEnd.bind(this);
        this.touchMoveHandler = this.onTouchMove.bind(this);

        document.addEventListener('wheel', this.mouseWheelHandler, { passive: false });
        domElement.addEventListener('mousedown', this.mouseDownHandler as EventListener);
        domElement.addEventListener('mouseup', this.mouseUpHandler as EventListener);
        domElement.addEventListener('mousemove', this.mouseMoveHandler as EventListener);

        domElement.addEventListener('touchstart', this.touchStartHandler as EventListener);
        domElement.addEventListener('touchend', this.touchEndHandler as EventListener);
        domElement.addEventListener('touchmove', this.touchMoveHandler as EventListener);
    }

    /**
     * Removes event listeners for mouse and touch interactions.
     */
    private removeEvents() {
        const domElement = this.domElement;

        document.removeEventListener('wheel', this.mouseWheelHandler as EventListener);
        domElement.removeEventListener('mousedown', this.mouseDownHandler as EventListener);
        domElement.removeEventListener('mouseup', this.mouseUpHandler as EventListener);
        domElement.removeEventListener('mousemove', this.mouseMoveHandler as EventListener);

        domElement.removeEventListener('touchstart', this.touchStartHandler as EventListener);
        domElement.removeEventListener('touchend', this.touchEndHandler as EventListener);
        domElement.removeEventListener('touchmove', this.touchMoveHandler as EventListener);
        if (this.contextMenuHandler) domElement.removeEventListener('contextmenu', this.contextMenuHandler);

    }

    /**
     * Handles touch start events and simulates mouse down.
     * @param e - The touch event.
     */
    private onTouchStart(e: TouchEvent) {
        const ev = { pageX: e.changedTouches[0].pageX, pageY: e.changedTouches[0].pageY, preventDefault: () => { } };
        this.onMouseDown(ev as unknown as MouseEvent);

    }

    /**
     * Handles touch end events and simulates mouse up.
     * @param e - The touch event.
     */
    private onTouchEnd(e: TouchEvent) {
        const ev = { pageX: e.changedTouches[0].pageX, pageY: e.changedTouches[0].pageY, preventDefault: () => { } };
        this.onMouseUp(ev as unknown as MouseEvent);

    }

    /**
     * Handles touch move events and simulates mouse move.
     * @param e - The touch event.
     */
    private onTouchMove(e: TouchEvent) {
        const ev = { pageX: e.changedTouches[0].pageX, pageY: e.changedTouches[0].pageY };
        this.onMouseMove(ev as unknown as MouseEvent);
    }

    /**
     * Handles mouse wheel events to zoom in and out.
     * @param e - The wheel event.
     */
    private onMouseWheel(e: WheelEvent): void {
        if (this.enabled) {
            e.preventDefault();
        }
        const delta = e.deltaY;
        const sign = this.invertWheel ? 1 : -1;
        this.wheelDelta += sign * delta * 0.1;

        this._mouseX = e.pageX;
        this._mouseY = e.pageY;
        this.mouseX = e.pageX;
        this.mouseY = e.pageY;
    }

    /**
     * Handles mouse down events to start camera movement.
     * @param e - The mouse event.
     */
    private onMouseDown(e: MouseEvent): void {
        if (this.enabled) {
            // e.preventDefault();
        }
        // with mouse pan: the right button, or the left with shift, pans
        this.panning = this.mousePan && (e.button === 2 || (e.button === 0 && e.shiftKey));
        if (this.panning) {
            this.panPoint.x = e.pageX;
            this.panPoint.y = e.pageY;
            return;
        }
        this.down = true;

        this.downPoint.x = e.pageX;
        this.downPoint.y = e.pageY;
    }

    /**
     * Handles mouse up events to stop camera movement.
     * @param e - The mouse event.
     */
    private onMouseUp(e: MouseEvent): void {
        if (this.enabled) {
            // e.preventDefault();
        }
        this.panning = false;
        this.down = false;

        this.prevAngles.x = this.currentAngles.x;
        this.prevAngles.y = this.currentAngles.y;

        this._mouseX = e.pageX;
        this._mouseY = e.pageY;
        this.mouseX = e.pageX;
        this.mouseY = e.pageY;
    }

    /**
     * Handles mouse move events to update camera angles.
     * @param e - The mouse event.
     */
    private onMouseMove(e: MouseEvent): void {
        if (this.enabled) {
            // e.preventDefault();
        }
        const normalizedX = e.pageX / window.innerWidth - 0.5;
        const normalizedY = e.pageY / window.innerHeight - 0.5;
        const scaleOffset = -30;

        this.offset.x = normalizedX * scaleOffset;
        this.offset.y = normalizedY * scaleOffset;

        if (this.panning) {
            const scale = this.radius * 0.0015;
            this.panBy((e.pageX - this.panPoint.x) * -scale, (e.pageY - this.panPoint.y) * scale);
            this.panPoint.x = e.pageX;
            this.panPoint.y = e.pageY;
        } else if (this.down) {
            this.displacement.x = (this.downPoint.x - e.pageX) / window.innerWidth;
            this.displacement.y = (this.downPoint.y - e.pageY) / window.innerHeight;

            this.currentAngles.x = (this.prevAngles.x + this.displacement.x);
            this.currentAngles.y = (this.prevAngles.y - this.displacement.y);

            //Check if outside limits
            if (this.currentAngles.y > this.limits.up) {
                this.currentAngles.y = this.prevAngles.y = this.limits.up;
                this.downPoint.y = e.pageY;
            }

            if (this.currentAngles.y < this.limits.down) {
                this.currentAngles.y = this.prevAngles.y = this.limits.down;
                this.downPoint.y = e.pageY;
            }

        } else {
            this._mouseX = e.pageX;
            this._mouseY = e.pageY;
        }

        if (this.onMove) this.onMove();

    }

    /**
     * Horizontal orbit angle in radians. Get/set the camera's azimuth around
     * the target on the XZ plane. 0 = +Z side, π = -Z side, π/2 = +X side.
     * Setting this snaps the camera without easing.
     */
    public get azimuth(): number {
        return this.currentAngles.x * this.PI * 2;
    }
    public set azimuth(radians: number) {
        const fraction = radians / (this.PI * 2);
        this.currentAngles.x = fraction;
        this.prevAngles.x    = fraction;
        this.finalRadians.x  = radians;
    }

    /**
     * Vertical orbit angle in radians, above the target's horizon (within ±0.4π). Setting this
     * snaps the camera without easing.
     */
    public get elevation(): number {
        return this.currentAngles.y * this.PI * 2;
    }
    public set elevation(radians: number) {
        const fraction = Math.min(this.limits.up, Math.max(this.limits.down, radians / (this.PI * 2)));
        this.currentAngles.y = fraction;
        this.prevAngles.y    = fraction;
        this.finalRadians.y  = fraction * this.PI * 2;
    }

    /**
     * Move the target toward `target` over `dt` seconds, closing `1 - e^(-rate dt)` of the gap
     * (about `rate` per second at small steps, at any frame rate): a camera that trails a
     * moving character. The target vector given to the constructor moves.
     */
    public follow(target: Vector3, dt: number, rate: number): void {
        const t = 1 - Math.exp(-dt * rate);
        this.target.set(
            this.target.x + (target.x - this.target.x) * t,
            this.target.y + (target.y - this.target.y) * t,
            this.target.z + (target.z - this.target.z) * t,
        );
    }

    /**
     * Look at `target` from `radius` away, at `azimuth` and `elevation` (radians), dropping any
     * pan: a camera preset. The camera snaps there on the next `update`.
     */
    public setView(target: Vector3, radius: number, azimuth: number, elevation: number): void {
        this.target.set(target.x, target.y, target.z);
        this.radius = this.wheelDelta = this.wheelDeltaEase = radius;
        this.azimuth = azimuth;
        this.elevation = elevation;
        this.pan = { x: 0, y: 0, z: 0 };
    }

    /** The point looked at: the target plus any pan. */
    public lookTarget(): Vector3 {
        return new Vector3(this.target.x + this.pan.x, this.target.y + this.pan.y, this.target.z + this.pan.z);
    }

    /**
     * Also pan with the mouse: drag with the right button, or with the left while holding shift
     * (the element's context menu is suppressed for the right drag). Off by default.
     */
    public withMousePan(): this {
        if (!this.mousePan) {
            this.mousePan = true;
            this.contextMenuHandler = (e: Event) => e.preventDefault();
            this.domElement.addEventListener('contextmenu', this.contextMenuHandler);
        }
        return this;
    }

    /** Move the pan by `dx`, `dy` along the camera's right and up axes. */
    private panBy(dx: number, dy: number): void {
        const [sa, ca] = [Math.sin(this.finalRadians.x), Math.cos(this.finalRadians.x)];
        const [se, ce] = [Math.sin(this.finalRadians.y), Math.cos(this.finalRadians.y)];
        // right = (cos az, 0, -sin az); up = right x forward
        this.pan.x += dx * ca + dy * -sa * se;
        this.pan.y += dy * ce;
        this.pan.z += dx * -sa + dy * -ca * se;
    }

    /**
     * Updates the camera position and orientation based on time and input.
     * @param t - The time delta for the update.
     */
    public update(t: number): void {
        this.time += t * 0.1;
        // Interpolamos los radianes en x y en y
        this.finalRadians.x += (this.currentAngles.x * this.PI * 2 - this.finalRadians.x) / 20;
        this.finalRadians.y += (this.currentAngles.y * this.PI * 2 - this.finalRadians.y) / 50;

        this.wheelDeltaEase += (this.wheelDelta - this.wheelDeltaEase) / 10;
        this.radius += (this.wheelDelta - this.radius) / 20;

        const look = this.lookPoint;
        look.set(this.target.x + this.pan.x, this.target.y + this.pan.y, this.target.z + this.pan.z);
        this.camera.position.x = (look.x + this.offsetEase.x) + (Math.sin(this.finalRadians.x) * Math.cos(this.finalRadians.y) * this.radius);
        this.camera.position.y = (look.y + this.offsetEase.y) + (Math.sin(this.finalRadians.y) * this.radius);
        this.camera.position.z = (look.z + this.offsetEase.z) + (Math.cos(this.finalRadians.x) * Math.cos(this.finalRadians.y) * this.radius);

        this.camera.lookAt(look);

        this.mouseX += (this._mouseX - this.mouseX) / 10;
        this.mouseY += (this._mouseY - this.mouseY) / 10;
    }

    /**
     * Disposes of the camera controls by removing event listeners.
     */
    public dispose(): void {
        this.removeEvents();
    }
}

export { CameraControls };
