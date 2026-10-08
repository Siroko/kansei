/**
 * Keyboard and gamepad input for pages that play: the keys held and pressed, and the first
 * gamepad's sticks and buttons. The TS side of `rust/kansei-wasm/src/input.rs`.
 */

/**
 * The keyboard, from the window's key events: which keys are held, and which went down since
 * the last `takePressed`. Keys are `KeyboardEvent.key` in lower case: `"w"`, `"arrowup"`,
 * `" "`, `"shift"`.
 *
 * Keys typed into a text field (a panel's number box) are left to it. Otherwise the arrows and
 * space don't scroll the page. A key released while the page is in the background never sends
 * its keyup, so leaving the page releases every key.
 */
export class Keys {
    private readonly heldKeys = new Set<string>();
    private pressed: string[] = [];
    private readonly listeners: [string, EventListener][] = [];

    private constructor() { }

    /** Listen to the window's key events from now on. */
    public static listen(): Keys {
        const keys = new Keys();
        const onDown = (e: KeyboardEvent) => {
            if (typedIntoAField(e)) return;
            const key = e.key.toLowerCase();
            if (key.startsWith('arrow') || key === ' ') e.preventDefault();
            if (!e.repeat) keys.pressed.push(key);
            keys.heldKeys.add(key);
        };
        const onUp = (e: KeyboardEvent) => { keys.heldKeys.delete(e.key.toLowerCase()); };
        const onBlur = () => keys.heldKeys.clear();
        keys.listeners.push(['keydown', onDown as EventListener], ['keyup', onUp as EventListener], ['blur', onBlur]);
        for (const [type, listener] of keys.listeners) window.addEventListener(type, listener);
        return keys;
    }

    /** Whether `key` is held down. */
    public held(key: string): boolean {
        return this.heldKeys.has(key);
    }

    /** Whether any of `keys` is held down. */
    public anyHeld(keys: string[]): boolean {
        return keys.some((k) => this.heldKeys.has(k));
    }

    /**
     * 1 while one of `positive` is held, -1 for `negative`, 0 for both or neither: an axis such
     * as `axis(['a', 'arrowleft'], ['d', 'arrowright'])`.
     */
    public axis(negative: string[], positive: string[]): number {
        return Number(this.anyHeld(positive)) - Number(this.anyHeld(negative));
    }

    /** The keys that went down since the last call, in order (a held key's auto-repeat is not a press). */
    public takePressed(): string[] {
        const pressed = this.pressed;
        this.pressed = [];
        return pressed;
    }

    /** Stop listening. */
    public dispose(): void {
        for (const [type, listener] of this.listeners) window.removeEventListener(type, listener);
        this.listeners.length = 0;
    }
}

/** Whether `e` is typing into a text field, which keeps its keys. */
function typedIntoAField(e: KeyboardEvent): boolean {
    const element = e.target;
    if (!(element instanceof HTMLElement)) return false;
    if (element.isContentEditable) return true;
    switch (element.tagName) {
        case 'TEXTAREA':
        case 'SELECT':
            return true;
        case 'INPUT':
            return !['checkbox', 'radio', 'button', 'submit', 'reset', 'range', 'color', 'file']
                .includes((element.getAttribute('type') ?? '').toLowerCase());
        default:
            return false;
    }
}

/**
 * The first connected gamepad in the standard mapping, read once a frame with `poll`: its
 * sticks past a dead zone, its buttons held, and those pressed since the poll before.
 */
export class Gamepad {
    /** Bottom face button (A, cross). */
    public static readonly A = 0;
    /** Right face button (B, circle). */
    public static readonly B = 1;
    /** Left face button (X, square). */
    public static readonly X = 2;
    /** Top face button (Y, triangle). */
    public static readonly Y = 3;
    public static readonly LEFT_BUMPER = 4;
    public static readonly RIGHT_BUMPER = 5;
    public static readonly LEFT_TRIGGER = 6;
    public static readonly RIGHT_TRIGGER = 7;

    private isConnected = false;
    private left: [number, number] = [0, 0];
    private right: [number, number] = [0, 0];
    private buttons: [boolean, number][] = [];
    private was: boolean[] = [];

    /**
     * Read the first connected gamepad; false (sticks at rest, no button held) when there is
     * none.
     */
    public poll(): boolean {
        this.was = this.buttons.map((b) => b[0]);
        const pad = (navigator.getGamepads?.() ?? []).find((p): p is globalThis.Gamepad => p !== null);
        if (!pad) {
            this.isConnected = false;
            this.left = [0, 0];
            this.right = [0, 0];
            this.buttons = [];
            return false;
        }
        const axis = (i: number) => pad.axes[i] ?? 0;
        this.left = deadZone(axis(0), axis(1));
        this.right = deadZone(axis(2), axis(3));
        this.buttons = pad.buttons.map((b) => [b.pressed, b.value]);
        this.isConnected = true;
        return true;
    }

    /** Whether the last poll found a gamepad. */
    public get connected(): boolean {
        return this.isConnected;
    }

    /** The left stick, x right and y down, each -1 to 1, zero within the dead zone. */
    public get leftStick(): [number, number] {
        return this.left;
    }

    /** The right stick, as `leftStick`. */
    public get rightStick(): [number, number] {
        return this.right;
    }

    /** Whether `button` is held. */
    public held(button: number): boolean {
        return this.buttons[button]?.[0] ?? false;
    }

    /** How far `button` is pressed, 0 to 1 (a trigger's travel). */
    public value(button: number): number {
        return this.buttons[button]?.[1] ?? 0;
    }

    /** Whether `button` went down between the last two polls. */
    public pressed(button: number): boolean {
        return this.held(button) && !(this.was[button] ?? false);
    }
}

/** A stick past a radial dead zone of 0.15, rescaled so the zone's edge reads 0 and full tilt 1. */
export function deadZone(x: number, y: number): [number, number] {
    const DEAD = 0.15;
    const m = Math.hypot(x, y);
    if (m < DEAD) return [0, 0];
    const s = Math.min((m - DEAD) / (1 - DEAD), 1) / m;
    return [x * s, y * s];
}
