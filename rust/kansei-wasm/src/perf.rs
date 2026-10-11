//! A performance overlay any example can show, after react-three-fiber's r3f-perf: frames per
//! second, the frame's CPU and GPU time with their graphs, draw calls, triangles, pipelines and
//! the GPU memory the page allocated, and (expanded) each labelled pass's GPU time and the CPU
//! sections. `?perf=1` shows it, the backquote key (`) toggles it, and so does [`set_visible`]
//! (a page's panel).
//!
//! `js/perf.js` (shipped in the example's `pkg/`) draws it, and counts what the page asks of
//! WebGPU itself: buffers and textures as they are created and destroyed (from the start, so the
//! memory is all of it), and while the overlay shows, draws and dispatches as they are encoded
//! (indirect ones counted apart: their triangles are the GPU's to know). [`crate::run`] times
//! every frame for it; an example that forwards its renderer's profile ([`profile`], once a
//! frame) adds the GPU's passes. Hidden, it costs a check a frame.

use std::cell::Cell;

use kansei_core::profiling::FrameProfile;
use kansei_core::renderers::Renderer;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "/js/perf.js")]
extern "C" {
    #[wasm_bindgen(js_name = perfVisible)]
    fn js_visible() -> bool;
    #[wasm_bindgen(js_name = perfSetVisible)]
    fn js_set_visible(visible: bool);
    #[wasm_bindgen(js_name = perfFrame)]
    fn js_frame(cpu_ms: f64);
    #[wasm_bindgen(js_name = perfProfile)]
    fn js_profile(json: &str);
    #[wasm_bindgen(js_name = perfClusters)]
    fn js_clusters(camera: u32, all: u32);
}

/// How often [`profile`] takes the renderer's profile while the overlay shows (ms).
const PROFILE_EVERY_MS: f64 = 500.0;

thread_local! {
    /// When the profile was last taken, and whether [`profile`] turned the renderer's profiling
    /// on (it turns it off again when the overlay hides).
    static LAST_PROFILE: Cell<f64> = const { Cell::new(0.0) };
    static PROFILING: Cell<bool> = const { Cell::new(false) };
}

/// Whether the overlay shows.
pub fn visible() -> bool {
    js_visible()
}

/// Show or hide the overlay.
pub fn set_visible(visible: bool) {
    js_set_visible(visible);
}

/// [`crate::run`]: a frame starts (its time when the overlay shows).
pub(crate) fn frame_begin() -> Option<f64> {
    js_visible().then(crate::now)
}

/// [`crate::run`]: the frame started at `start` ended (its CPU time, the frame's tick).
pub(crate) fn frame_end(start: Option<f64>) {
    if let Some(start) = start {
        js_frame((crate::now() - start) * 1000.0);
    }
}

/// Give the overlay `renderer`'s GPU passes and CPU sections, and its cluster LOD cuts'
/// triangles: call it once a frame, after rendering. While the overlay shows it turns profiling on (`Renderer::set_profiling`) and takes
/// the profile twice a second; when it hides, it turns profiling off again if it turned it on.
/// An example that takes the profile itself (a HUD of its own) shares it: each take averages the
/// frames since the one before.
pub fn profile(renderer: &mut Renderer) {
    if !js_visible() {
        if PROFILING.get() {
            PROFILING.set(false);
            renderer.set_profiling(false);
        }
        return;
    }
    if !renderer.profiling() {
        renderer.set_profiling(true);
        PROFILING.set(true);
    }
    // the cluster LOD cuts' triangles (indirect draws: the overlay can't count them itself)
    let (camera, all) = renderer.cluster_triangles_total();
    js_clusters(camera, all);
    let now = crate::now() * 1000.0;
    if now - LAST_PROFILE.get() < PROFILE_EVERY_MS {
        return;
    }
    LAST_PROFILE.set(now);
    let profile = renderer.take_profile();
    if profile.gpu_frames > 0 || profile.cpu_frames > 0 {
        js_profile(&profile_json(&profile));
    }
}

/// A profile as the overlay reads it: `{gpu_ms, gpu_span_ms, gpu_frames, passes: [[label, ms,
/// count]], cpu: [[label, ms]]}`, the passes and sections most expensive first.
pub fn profile_json(profile: &FrameProfile) -> String {
    let label = |l: &str| l.replace(['"', '\\'], "");
    let mut passes = profile.gpu.clone();
    passes.sort_by(|a, b| b.exclusive_ms.total_cmp(&a.exclusive_ms));
    let passes: Vec<String> = passes.iter().map(|p| format!("[\"{}\",{:.4},{:.2}]", label(p.label), p.exclusive_ms, p.count)).collect();
    let mut cpu = profile.cpu.clone();
    cpu.sort_by(|a, b| b.1.total_cmp(&a.1));
    let cpu: Vec<String> = cpu.iter().map(|(l, ms)| format!("[\"{}\",{:.4}]", label(l), ms)).collect();
    format!(
        "{{\"gpu_ms\":{:.4},\"gpu_span_ms\":{:.4},\"gpu_frames\":{},\"passes\":[{}],\"cpu\":[{}]}}",
        profile.gpu_ms,
        profile.gpu_span_ms,
        profile.gpu_frames,
        passes.join(","),
        cpu.join(",")
    )
}
