// The page side of kansei_wasm's `launch`: run an example's `start` on the page, or (`?worker=1`)
// in a dedicated worker on its canvas transferred as an OffscreenCanvas (worker.js), and hand the
// page one API for both. The page keeps its UI (tweakpane, the HUD) and calls the example through
// the API: `await kansei.set_pressure(1)` runs `set_pressure` where the example runs.
//
// In a worker the page also: posts the canvas's CSS size and device pixel ratio when they change;
// forwards the input events the worker listens to (on the canvas, or keys, blur and focus on the
// window), cancelling those whose default it must stop here (touches, the context menu, arrows
// and space); and shows the worker's `set_text`.

import { frameProbe } from './worker.js';

const EVENT_FIELDS = [
    'offsetX', 'offsetY', 'clientX', 'clientY', 'pageX', 'pageY', 'screenX', 'screenY', 'movementX', 'movementY',
    'button', 'buttons', 'ctrlKey', 'shiftKey', 'altKey', 'metaKey', 'deltaX', 'deltaY', 'deltaZ', 'deltaMode',
    'key', 'code', 'repeat', 'pointerId', 'pointerType', 'pressure', 'isPrimary',
];
const TOUCH_FIELDS = ['identifier', 'clientX', 'clientY', 'pageX', 'pageY', 'screenX', 'screenY', 'radiusX', 'radiusY', 'force'];
// a canvas listener in the worker cancels these (CameraControls' touches, with_mouse_pan's menu):
// only the page can, and only while the event is dispatched
const CANCELLED_ON_CANVAS = new Set(['touchstart', 'touchmove', 'touchend', 'touchcancel', 'contextmenu']);
// input events the probe times on the page (in a worker, every forwarded event)
const PROBED = ['mousemove', 'mousedown', 'pointermove', 'pointerdown', 'wheel', 'touchmove'];

const epoch = (pageTime) => performance.timeOrigin + pageTime;

// An event as the worker gets it: its fields, and its time on the page's clock (ms since epoch).
function serialize(e) {
    const out = { type: e.type, time: epoch(e.timeStamp) };
    for (const key of EVENT_FIELDS) if (key in e) out[key] = e[key];
    for (const list of ['touches', 'changedTouches', 'targetTouches']) {
        if (e[list]) out[list] = Array.from(e[list], (t) => Object.fromEntries(TOUCH_FIELDS.map((k) => [k, t[k]])));
    }
    return out;
}

// Whether a key event types into a text field, which keeps its keys (kansei_wasm's Keys rule).
function typedIntoAField(e) {
    const el = e.target;
    if (!(el instanceof HTMLElement)) return false;
    if (el.isContentEditable) return true;
    if (el.tagName === 'TEXTAREA' || el.tagName === 'SELECT') return true;
    return el.tagName === 'INPUT'
        && !['checkbox', 'radio', 'button', 'submit', 'reset', 'range', 'color', 'file'].includes((el.getAttribute('type') || '').toLowerCase());
}

// The API: `kansei.<export>(...args)` calls the example's export (or an extension's) where it
// runs and resolves to its result; `kansei.worker` says where that is; `kansei.probe()` hands over
// the `?probe=1` record (see worker.js `frameProbe`).
function api(base) {
    return new Proxy(base, {
        get(target, key) {
            if (key in target) return target[key];
            // not a thenable: `await launch(...)` resolves to the API itself
            if (typeof key !== 'string' || key === 'then') return undefined;
            return (...args) => target.call(key, args);
        },
    });
}

// `wasm`: the example's bindings (`import * as wasm from '../pkg/<name>.js'`, initialised);
// `module`: its compiled WebAssembly.Module. Options:
// - `module`: the bindings' URL, which the worker imports (needed for a worker);
// - `canvas`: the canvas's id ('kansei');
// - `worker`: false, true, or 'ticks' (the page's requestAnimationFrame paces the worker's
//   frames); by default `?worker=` on the page's URL (1 or ticks);
// - `extensions`: module URLs loaded where the example runs, before `start`: each one's
//   `install(wasm)` is called and its other exports join the API;
// - `probe`: record frame and input times (`?probe=1`).
export async function launchExample(wasm, module, options) {
    const query = new URLSearchParams(location.search);
    const workerParam = query.get('worker');
    const {
        module: moduleUrl,
        canvas: canvasId = 'kansei',
        worker = workerParam === 'ticks' ? 'ticks' : workerParam === '1',
        extensions = [],
        probe = query.get('probe') === '1',
    } = options ?? {};
    const canvas = document.getElementById(canvasId);
    if (!canvas) throw new Error(`no element #${canvasId}`);
    if (worker && !(moduleUrl && 'transferControlToOffscreen' in canvas && typeof Worker === 'function')) {
        console.warn('kansei: running on the page (a worker needs `module` and OffscreenCanvas)');
    } else if (worker) {
        return remote({ module, canvas, canvasId, moduleUrl, extensions, probe, ticks: worker === 'ticks' });
    }
    return local({ wasm, canvas, canvasId, extensions, probe });
}

async function local({ wasm, canvas, canvasId, extensions, probe }) {
    const loaded = await Promise.all(extensions.map((url) => import(url)));
    for (const extension of loaded) await extension.install?.(wasm);
    const recorder = probe ? frameProbe() : null;
    if (recorder) {
        for (const type of PROBED) canvas.addEventListener(type, (e) => recorder.input(epoch(e.timeStamp)), { capture: true, passive: true });
    }
    await wasm.start(canvasId);
    return api({
        worker: false,
        async call(name, args) {
            const fn = loaded.find((e) => typeof e[name] === 'function')?.[name] ?? wasm[name];
            if (typeof fn !== 'function') throw new Error(`no function ${name}`);
            return fn(...args);
        },
        async probe() {
            return recorder?.take() ?? null;
        },
    });
}

async function remote({ module, canvas, canvasId, moduleUrl, extensions, probe, ticks }) {
    const offscreen = canvas.transferControlToOffscreen();
    // worker.js beside this file in pkg/snippets: the same module the bindings import, so the
    // worker's entry and the bindings share its state
    const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module', name: `kansei ${canvasId}` });
    const post = (message, transfer = []) => worker.postMessage(message, transfer);
    const box = () => ({ width: canvas.clientWidth, height: canvas.clientHeight, ratio: devicePixelRatio });

    let started;
    const ready = new Promise((resolve, reject) => (started = { resolve, reject }));
    let nextCall = 1;
    const calls = new Map();
    const forwarded = new Set();

    const forward = (target, type) => {
        const key = `${target} ${type}`;
        if (forwarded.has(key)) return;
        forwarded.add(key);
        const cancel = target === 'canvas' && CANCELLED_ON_CANVAS.has(type);
        const listener = (e) => {
            if (type === 'keydown') {
                if (typedIntoAField(e)) return;
                if (e.key.startsWith('Arrow') || e.key === ' ') e.preventDefault();
            }
            if (cancel) e.preventDefault();
            post({ type: 'event', target, event: serialize(e) });
        };
        (target === 'window' ? window : canvas).addEventListener(type, listener, { passive: !cancel });
    };

    const tick = (time) => {
        post({ type: 'frame', time: epoch(time) });
        requestAnimationFrame(tick);
    };

    worker.addEventListener('message', ({ data }) => {
        switch (data.type) {
            case 'started':
                started.resolve();
                break;
            case 'error':
                console.error(`kansei worker: ${data.message}`);
                started.reject(new Error(data.message));
                break;
            case 'result': {
                const call = calls.get(data.id);
                calls.delete(data.id);
                if ('error' in data) call?.reject(new Error(data.error));
                else call?.resolve(data.value);
                break;
            }
            case 'listen':
                forward(data.target, data.eventType);
                break;
            case 'text': {
                const element = document.getElementById(data.id);
                if (element) element.textContent = data.text;
                break;
            }
            case 'frames':
                requestAnimationFrame(tick);
                break;
        }
    });
    worker.addEventListener('error', (e) => {
        console.error(`kansei worker: ${e.message}`);
        started.reject(new Error(e.message));
    });

    post({
        type: 'boot', module: moduleUrl, wasm: module, canvasId, canvas: offscreen,
        search: location.search, base: document.baseURI, ...box(), extensions, probe, ticks,
    }, [offscreen]);

    // the canvas's CSS box, and the pixel ratio (a move to another screen, a zoom)
    new ResizeObserver(() => post({ type: 'resize', ...box() })).observe(canvas);
    const watchRatio = () => {
        matchMedia(`(resolution: ${devicePixelRatio}dppx)`).addEventListener('change', () => {
            post({ type: 'resize', ...box() });
            watchRatio();
        }, { once: true });
    };
    watchRatio();

    await ready;
    return api({
        worker: true,
        call(name, args) {
            return new Promise((resolve, reject) => {
                const id = nextCall++;
                calls.set(id, { resolve, reject });
                post({ type: 'call', id, name, args });
            });
        },
        probe() {
            return this.call('__kansei_probe', []);
        },
    });
}
