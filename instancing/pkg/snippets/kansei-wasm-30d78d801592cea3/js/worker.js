// The worker side of kansei_wasm's `launch`: a dedicated worker's entry (`?worker=1`), which loads
// the example's wasm and runs its `start` there on the page's canvas, transferred as an
// OffscreenCanvas. kansei_wasm's Rust reads the canvas, the page's query string and base URL, and
// the frame clock through the exports below (src/worker.rs); host.js is the page's side.
//
// The page's input arrives as messages and is dispatched as events on stand-ins, so listeners
// written for the page work unchanged: the canvas's (CameraControls, MouseVectors) on a stand-in
// element that carries the canvas's CSS size, and the window's (Keys, blur and focus) on the
// worker's own global scope. Only the event types something listens to are forwarded.
//
// The page's bundle imports this module too (kansei_wasm imports it in either realm); on the page
// it does nothing.

const inWorker = typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope;

// the page's query string and base URL, and its canvas (one per worker)
let page = { search: '', base: '' };
const canvases = new Map();
// the example's wasm bindings and the extensions' exports, for calls from the page
let bindings = null;
let extensions = [];
let probe = null;
let pageTicks = false;
const pendingFrames = [];
let framesRequested = false;

// The window's events kansei listens to on the global scope.
const WINDOW_EVENTS = new Set(['keydown', 'keyup', 'blur', 'focus']);

function post(message) {
    self.postMessage(message);
}

// The page's canvas element as listeners see it: an event target with the canvas's CSS size
// (clientWidth, clientHeight) and the page's device pixel ratio. Adding a listener asks the page
// to forward that event type.
class CanvasStandIn extends EventTarget {
    constructor(id, width, height, ratio) {
        super();
        this.id = id;
        this.clientWidth = width;
        this.clientHeight = height;
        this.devicePixelRatio = ratio;
        this.style = {};
    }

    addEventListener(type, listener, options) {
        super.addEventListener(type, listener, options);
        if (type !== 'kansei-resize') post({ type: 'listen', target: 'canvas', eventType: type });
    }

    getBoundingClientRect() {
        const [width, height] = [this.clientWidth, this.clientHeight];
        return { x: 0, y: 0, left: 0, top: 0, right: width, bottom: height, width, height };
    }

    focus() {}
    setPointerCapture() {}
    releasePointerCapture() {}
}

// The page's event as a listener reads it: an Event of its type with the page event's fields.
function dispatch(target, data) {
    probe?.input(data.time);
    const event = new Event(data.type, { cancelable: true });
    for (const [key, value] of Object.entries(data)) {
        if (key !== 'type' && key !== 'time') event[key] = value;
    }
    (target === 'window' ? self : canvases.values().next().value?.element)?.dispatchEvent(event);
}

async function boot(message) {
    page = { search: message.search, base: message.base };
    pageTicks = message.ticks;
    const element = new CanvasStandIn(message.canvasId, message.width, message.height, message.ratio);
    canvases.set(message.canvasId, { canvas: message.canvas, element });
    if (message.probe) probe = frameProbe();
    const wasm = await import(message.module);
    // the module the page compiled, so the worker neither fetches nor compiles it again
    await wasm.default({ module_or_path: message.wasm });
    bindings = wasm;
    extensions = await Promise.all(message.extensions.map((url) => import(url)));
    for (const extension of extensions) await extension.install?.(wasm);
    await wasm.start(message.canvasId);
}

async function call({ id, name, args }) {
    try {
        let fn = name === '__kansei_probe' ? () => probe?.take() ?? null : null;
        fn ??= extensions.find((e) => typeof e[name] === 'function')?.[name] ?? bindings?.[name];
        if (typeof fn !== 'function') throw new Error(`no function ${name}`);
        post({ type: 'result', id, value: await fn(...args) });
    } catch (error) {
        post({ type: 'result', id, error: String(error?.message ?? error) });
    }
}

function onMessage({ data }) {
    switch (data.type) {
        case 'boot':
            boot(data).then(
                () => post({ type: 'started' }),
                (error) => post({ type: 'error', message: String(error?.stack ?? error) }),
            );
            break;
        case 'resize': {
            const element = canvases.values().next().value?.element;
            if (!element) break;
            element.clientWidth = data.width;
            element.clientHeight = data.height;
            element.devicePixelRatio = data.ratio;
            element.dispatchEvent(new Event('kansei-resize'));
            break;
        }
        case 'event':
            dispatch(data.target, data.event);
            break;
        case 'call':
            call(data);
            break;
        case 'frame':
            for (const callback of pendingFrames.splice(0)) callback(performance.now());
            break;
    }
}

if (inWorker) {
    self.addEventListener('message', onMessage);
    self.addEventListener('error', (e) => post({ type: 'error', message: String(e.error?.stack ?? e.message) }));
    self.addEventListener('unhandledrejection', (e) => post({ type: 'error', message: String(e.reason?.stack ?? e.reason) }));
    // the window's listeners go on the worker's scope: ask the page for those events
    const add = self.addEventListener.bind(self);
    self.addEventListener = (type, listener, options) => {
        add(type, listener, options);
        if (WINDOW_EVENTS.has(type)) post({ type: 'listen', target: 'window', eventType: type });
    };
}

// ── for src/worker.rs ──

// The page's canvas with this id, in the worker: { canvas: OffscreenCanvas, element: stand-in }.
export function workerCanvas(id) {
    return canvases.get(id);
}

// The page's query string (`?a=1`).
export function pageSearch() {
    return page.search;
}

// `url` relative to the page, as the page would fetch it (a worker's own URL is its script's).
export function pageUrl(url) {
    return new URL(url, page.base).href;
}

// Show `text` in the page's element with this id.
export function postText(id, text) {
    post({ type: 'text', id, text });
}

// The next frame: the worker's requestAnimationFrame where the browser has one (and `?worker=1`
// rather than `?worker=ticks`); else the page's, each of its frames posted as a tick.
export function requestFrame(callback) {
    if (!pageTicks && typeof self.requestAnimationFrame === 'function') {
        self.requestAnimationFrame(callback);
        return;
    }
    pendingFrames.push(callback);
    if (!framesRequested) {
        framesRequested = true;
        post({ type: 'frames' });
    }
}

// ── for host.js ──

// `?probe=1`: the time of each frame (its getCurrentTexture) and of each input event's arrival
// where the engine runs, on the page's clock in ms since the epoch (performance.timeOrigin + now),
// so the page's and the worker's times compare. `take()` hands over and clears what was recorded.
export function frameProbe() {
    const LIMIT = 100_000;
    const frames = [];
    const inputs = []; // [the event's time, its arrival]
    const now = () => performance.timeOrigin + performance.now();
    const getCurrentTexture = GPUCanvasContext.prototype.getCurrentTexture;
    GPUCanvasContext.prototype.getCurrentTexture = function () {
        if (frames.length < LIMIT) frames.push(now());
        return getCurrentTexture.call(this);
    };
    return {
        input(time) {
            if (inputs.length < LIMIT) inputs.push([time, now()]);
        },
        take() {
            return { frames: frames.splice(0), inputs: inputs.splice(0) };
        },
    };
}
