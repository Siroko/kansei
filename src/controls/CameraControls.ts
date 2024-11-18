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

    private time: number;
    private domElement: HTMLElement | Window;

    private mouseWheelHandler?: (e: WheelEvent) => void;
    private mouseDownHandler?: (e: MouseEvent) => void;
    private mouseUpHandler?: (e: MouseEvent) => void;
    private mouseMoveHandler?: (e: MouseEvent) => void;
    private touchStartHandler?: (e: TouchEvent) => void;
    private touchEndHandler?: (e: TouchEvent) => void;
    private touchMoveHandler?: (e: TouchEvent) => void;

    /**
     * Creates an instance of CameraControls.
     * @param camera - The camera to control.
     * @param target - The target vector for the camera to look at.
     * @param domElement - The DOM element to attach event listeners to.
     */
    constructor(camera: Camera, target: Vector3, domElement: HTMLElement | Window, radius?: number) {

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

        this.radius = radius || window.innerWidth < 768 ? 35 : 22;
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
        const domElement = document.getElementById('scrollHandler') || window;

        document.removeEventListener('wheel', this.mouseWheelHandler as EventListener);
        domElement.removeEventListener('mousedown', this.mouseDownHandler as EventListener);
        domElement.removeEventListener('mouseup', this.mouseUpHandler as EventListener);
        domElement.removeEventListener('mousemove', this.mouseMoveHandler as EventListener);

        domElement.removeEventListener('touchstart', this.touchStartHandler as EventListener);
        domElement.removeEventListener('touchend', this.touchEndHandler as EventListener);
        domElement.removeEventListener('touchmove', this.touchMoveHandler as EventListener);

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
        this.wheelDelta -= delta * 0.1;

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

        if (this.down) {
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

        this.camera.position.x = (this.target.x + this.offsetEase.x) + (Math.sin(this.finalRadians.x) * Math.cos(this.finalRadians.y) * this.radius);
        this.camera.position.y = (this.target.y + this.offsetEase.y) + (Math.sin(this.finalRadians.y) * this.radius);
        this.camera.position.z = (this.target.z + this.offsetEase.z) + (Math.cos(this.finalRadians.x) * Math.cos(this.finalRadians.y) * this.radius);

        this.camera.lookAt(this.target);

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
