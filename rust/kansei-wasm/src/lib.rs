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
//!   delta time and any new canvas size.
//! - [`param`], [`param_or`] and [`flag`] read the page's query string, percent-decoded.
//! - [`now`], [`fetch_bytes`] and [`is_phone`] cover timing, loading and picking a tier.
//!
//! It is for the browser: it compiles on any target (so an example's native tests build), but
//! only does anything in `wasm32`.

mod canvas;
mod frame_loop;
mod page;

pub use canvas::Canvas;
pub use frame_loop::{run, Frame};
pub use page::{fetch_bytes, flag, init, is_phone, now, param, param_or};
