//! Pipelines compiled behind the loading screen (`js/pipelines.js`). The first time a browser
//! shows a page, it compiles every pipeline the engine makes one after another as the engine
//! makes them: a first visit runs at a few frames a second until they are all made. Record what a
//! page makes once (`?record_pipelines=1`, then [`manifest`] or `window.kanseiPipelines.download()`),
//! ship that manifest with it, and hand it to [`warm`] before the renderer asks for its device
//! (before `Canvas::renderer`): the device then comes once the browser has compiled every pipeline
//! of the manifest in parallel (the *Async calls), and the engine's own calls find them made. It
//! changes nothing but when the compiling happens; pipelines missing from the manifest are made
//! as before.
//!
//! ```ignore
//! let manifest = kansei_wasm::fetch_bytes("pipelines.json").await?;
//! kansei_wasm::pipelines::warm(std::str::from_utf8(&manifest).unwrap()).await;
//! let renderer = canvas.renderer(config).await; // waits for the warm-up
//! ```

use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "/js/pipelines.js")]
extern "C" {
    #[wasm_bindgen(js_name = recordPipelines)]
    fn js_record();
    #[wasm_bindgen(js_name = pipelinesManifest)]
    fn js_manifest() -> String;
    #[wasm_bindgen(js_name = warmPipelines)]
    async fn js_warm(json: &str);
    #[wasm_bindgen(js_name = warmStats)]
    fn js_stats() -> String;
}

/// Record the shader modules, layouts and pipelines the page makes from now on (as
/// `?record_pipelines=1` does from the start: start it before the renderer to catch them all).
pub fn record() {
    js_record();
}

/// What the page has made since recording started, as the JSON [`warm`] takes.
pub fn manifest() -> String {
    js_manifest()
}

/// Make a recorded manifest's pipelines on the next device the page asks for (and on any it has),
/// all at once: returns at once; the renderer's device request resolves when they are made.
pub async fn warm(manifest_json: &str) {
    js_warm(manifest_json).await;
}

/// The warm-up's work so far, JSON: `{ replayed, failed, ms }`.
pub fn stats() -> String {
    js_stats()
}
