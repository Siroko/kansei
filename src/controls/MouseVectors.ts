import { Vector2 } from "../math/Vector2";

/**
 * Class representing mouse vector controls.
 */
class MouseVectors {

    private onMouseMoveHandler: (e: Event) => void;
    private onMouseEnterHandler: (e: Event) => void;
    private isMobile: boolean = navigator.userAgent.match(/(iPad)|(iPhone)|(iPod)|(android)|(webOS)/i) ? true : false;

    private tmpVector: Vector2 = new Vector2();
    private mouseEnd: Vector2 = new Vector2();

    public speed: number = 8;
    public mousePosition: Vector2 = new Vector2();
    public mouseDirection: Vector2 = new Vector2();
    public mouseStrength: number = 0;

    /**
     * Creates an instance of MouseVectors.
     * @param {HTMLElement} domElement - The DOM element to attach mouse events to. Defaults to document.body.
     */
    constructor(
        private domElement: HTMLElement = document.body
    ) {
        this.onMouseMoveHandler = this.onMouseMove.bind(this);
        this.onMouseEnterHandler = this.onMouseEnter.bind(this);

        this.addEvents();
    }

    /**
     * Adds mouse event listeners to the DOM element.
     * @private
     */
    private addEvents() {
        // TODO: Create an event manager so we don't duplicate addEventListeners for the same domElement
        this.domElement.addEventListener(this.isMobile ? 'touchmove' : 'mousemove', this.onMouseMoveHandler);
        this.domElement.addEventListener('mouseenter', this.onMouseEnterHandler);
    }

    /**
     * Removes mouse event listeners from the DOM element.
     * @private
     */
    private removeEvents() {
        this.domElement.removeEventListener(this.isMobile ? 'touchmove' : 'mousemove', this.onMouseMoveHandler);
        this.domElement.removeEventListener('mouseenter', this.onMouseEnterHandler);
    }

    private onMouseEnter(e: Event) {
        const mouseX = this.isMobile ? (e as TouchEvent).touches[0].clientX : (e as MouseEvent).clientX;
        const mouseY = this.isMobile ? (e as TouchEvent).touches[0].clientY : (e as MouseEvent).clientY;

        this.mouseEnd.set((mouseX / innerWidth - 0.5) * 2, (mouseY / innerHeight - 0.5) * 2);
        this.mousePosition.copy(this.mouseEnd);

    }

    /**
     * Handles mouse movement events.
     * @param {Event} e - The mouse or touch event.
     * @private
     */
    private onMouseMove(e: Event) {
        const mouseX = this.isMobile ? (e as TouchEvent).touches[0].clientX : (e as MouseEvent).clientX;
        const mouseY = this.isMobile ? (e as TouchEvent).touches[0].clientY : (e as MouseEvent).clientY;

        this.mouseEnd.set((mouseX / innerWidth - 0.5) * 2, (mouseY / innerHeight - 0.5) * 2);
    }

    /**
     * Updates the mouse position and direction based on the elapsed time.
     * @param {number} dt - The delta time since the last update.
     */
    public update(dt: number) {
        this.tmpVector.copy(this.mousePosition);
        this.mousePosition.set(
            this.mousePosition.x + (this.mouseEnd.x - this.mousePosition.x) * this.speed * dt,
            this.mousePosition.y + (this.mouseEnd.y - this.mousePosition.y) * this.speed * dt
        );

        this.tmpVector.sub(this.mousePosition);
        this.mouseDirection.copy(this.tmpVector);

        this.mouseStrength = this.mouseDirection.length();
    }

    /**
     * Cleans up event listeners and resources.
     */
    public dispose() {
        this.removeEvents();
    }
}

export { MouseVectors }
