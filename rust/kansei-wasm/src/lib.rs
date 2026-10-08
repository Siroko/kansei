//! Kansei on the web: the plumbing every WASM example shares, so an example's code is about the
//! engine feature it shows.
//!
//! ```ignore
//! #[wasm_bindgen]
//! pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
//!     let canvas = kansei_wasm::Canvas::find(canvas_id)?;
//!     let mut renderer = canvas.renderer(RendererConfig { sample_count: 1, ..Default::default() }).await;
//!     let mut camera = Camera::new(45.0, 0.1, 100.0, canvas.aspect());
//!     let mut scene = Scene::new();
//!     // ... build the scene ...
//!     let exposure: f32 = kansei_wasm::param_or("ev", 12.0);
//!     kansei_wasm::run(&canvas, move |frame| {
//!         frame.resize(&mut renderer, &mut camera);
//!         renderer.render(&mut scene, &mut camera);
//!     });
//!     Ok(())
//! }
//! ```
//!
//! - [`Canvas`] sizes the drawing buffer to the canvas's CSS box times `devicePixelRatio`
//!   (capped at 2, or `?dpr=` on the page's URL) and follows it when the page resizes.
//! - [`run`] drives the `requestAnimationFrame` loop and hands each frame its [`Frame`]: time,
//!   delta time and any new canvas size. It skips a refresh while the GPU still has a frame
//!   to finish ([`RunOptions::max_frames_in_flight`], `?inflight=`), so frames do not queue.
//! - [`param`], [`param_or`] and [`flag`] read the page's query string, percent-decoded.
//! - [`now`], [`fetch_bytes`] and [`is_phone`] cover timing, loading and picking a tier;
//!   [`set_text`], [`checkbox`] and [`thousands`] serve a HUD.
//! - [`Keys`] and [`Gamepad`] are the input of pages that play: keys held and pressed, sticks
//!   and buttons.
//! - [`launch`] runs the example's `start` on the page or, with `?worker=1`, in a dedicated
//!   worker on the canvas transferred as an `OffscreenCanvas`: all of the above works there
//!   unchanged (see below).
//!
//! # In a worker
//!
//! A page that starts its example with [`launch`] instead of calling `start` can run it in a
//! worker, so the frame's CPU work (simulation, culling, encoding) leaves the page's main thread
//! and the page's own work (its panels, layout, garbage collection) no longer delays frames. The
//! GPU work is the same: it runs in the browser's GPU process either way.
//!
//! - The page keeps its UI and calls the example through the API [`launch`] returns: every
//!   export, as a promise of its result.
//! - [`Canvas::find`] finds the transferred canvas; the page posts its CSS size and device pixel
//!   ratio when they change, and [`run`] reports the resize as usual.
//! - [`run`] paces frames with the worker's `requestAnimationFrame`. Where a browser has none
//!   (or `?worker=ticks`), the page's `requestAnimationFrame` posts each frame to the worker,
//!   which brings back the page's main thread as the frame's clock (not as its work). The
//!   frames-in-flight cap and `pacing::FixedStep` work as on the page: they read only the GPU
//!   queue and the frame's time.
//! - Input: [`Canvas::element`] is a stand-in that receives the canvas's events, forwarded by the
//!   page for each type something listens to, so `CameraControls::from_canvas` and
//!   `MouseVectors::from_canvas` work unchanged; [`Keys`] and the window's blur and focus arrive
//!   on the worker's global scope. Workers have no gamepads.
//! - [`param`] reads the page's query string, [`fetch_bytes`] fetches relative to the page,
//!   [`set_text`] writes to the page's element; [`checkbox`] finds none.
//!
//! It is for the browser: it compiles on any target (so an example's native tests build), but
//! only does anything in `wasm32`.

mod canvas;
mod frame_gate;
mod frame_loop;
mod input;
mod page;
mod worker;

pub use canvas::Canvas;
pub use frame_gate::FramesInFlight;
pub use frame_loop::{run, run_with, Frame, RunOptions, DEFAULT_MAX_FRAMES_IN_FLIGHT};
pub use input::{Gamepad, Keys};
pub use page::{checkbox, fetch_bytes, flag, init, is_phone, now, param, param_or, set_text, thousands};
pub use worker::launch;
