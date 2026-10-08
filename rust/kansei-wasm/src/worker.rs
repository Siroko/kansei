//! Running an example in a dedicated worker, on its page's canvas as an `OffscreenCanvas`.
//!
//! [`launch`] (called by the page) starts the example's `start` on the page or, with
//! `?worker=1`, in a worker (`js/host.js` on the page, `js/worker.js` in the worker, both shipped
//! in the example's `pkg/`). The example's Rust is the same either way: in a worker,
//! [`crate::Canvas::find`] finds the transferred canvas, [`crate::run`] paces frames with the
//! worker's `requestAnimationFrame`, [`crate::param`] reads the page's query string,
//! [`crate::fetch_bytes`] fetches relative to the page, [`crate::set_text`] writes to the page, and
//! input listeners (`CameraControls::from_canvas(canvas.element())`, [`crate::Keys`]) get the
//! page's events, forwarded.

use std::cell::OnceCell;

use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "/js/worker.js")]
extern "C" {
    /// `{ canvas: OffscreenCanvas, element }` for the canvas with this id, in a worker.
    #[wasm_bindgen(js_name = workerCanvas)]
    pub(crate) fn worker_canvas(id: &str) -> JsValue;
    #[wasm_bindgen(js_name = pageSearch)]
    pub(crate) fn page_search() -> String;
    #[wasm_bindgen(js_name = pageUrl)]
    pub(crate) fn page_url(url: &str) -> String;
    #[wasm_bindgen(js_name = postText)]
    pub(crate) fn post_text(id: &str, text: &str);
    #[wasm_bindgen(js_name = requestFrame)]
    pub(crate) fn request_frame(callback: &js_sys::Function);
}

#[wasm_bindgen(module = "/js/host.js")]
extern "C" {
    // (not `launch`: the bindings export `launch`, and an import of that name would shadow it)
    #[wasm_bindgen(js_name = launchExample)]
    fn host_launch(bindings: JsValue, module: JsValue, options: JsValue) -> js_sys::Promise;
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(thread_local_v2, js_name = globalThis)]
    static GLOBAL_THIS: web_sys::EventTarget;
}

thread_local! {
    static IN_WORKER: OnceCell<bool> = const { OnceCell::new() };
}

/// Whether this runs in a worker (no `window`).
pub(crate) fn in_worker() -> bool {
    IN_WORKER.with(|w| *w.get_or_init(|| web_sys::window().is_none()))
}

/// The global scope: the page's `window`, or the worker's scope, where the page's key, blur and
/// focus events arrive.
pub(crate) fn global() -> web_sys::EventTarget {
    GLOBAL_THIS.with(Clone::clone)
}

/// Start the example: the page's module script calls it once its bindings are initialised, in
/// place of `start`, and gets back (a promise of) the API it calls the example through.
///
/// ```js
/// import init, * as wasm from '../pkg/kansei_wasm_fluid.js';
/// await init();
/// const kansei = await wasm.launch(wasm, { module: new URL('../pkg/kansei_wasm_fluid.js', import.meta.url).href });
/// await kansei.set_pressure(40); // every export, as a promise, wherever the example runs
/// ```
///
/// `start(canvas_id)` runs on the page, or with `?worker=1` in a dedicated worker on the canvas
/// transferred as an `OffscreenCanvas` (the module the page compiled is handed over, not fetched
/// again). Options: `module` (the bindings' URL, which the worker imports; without it the
/// example runs on the page), `canvas` (the canvas's id, `"kansei"`), `worker` (`true`, `false`
/// or `"ticks"`; by default `?worker=1` or `?worker=ticks`), `extensions` (module URLs loaded
/// where the example runs: each one's `install(wasm)` runs before `start` and its other exports
/// join the API) and `probe` (`?probe=1`: `kansei.probe()` returns the frames' and the input
/// events' times). See `js/host.js`.
#[wasm_bindgen]
// (`bindings`, not `wasm`: the generated JS calls into the module through a variable `wasm`)
pub fn launch(bindings: JsValue, options: JsValue) -> js_sys::Promise {
    host_launch(bindings, wasm_bindgen::module(), options)
}
