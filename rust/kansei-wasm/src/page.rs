//! The page around the canvas: logging, the clock, the query string and fetching files.

use std::str::FromStr;
use std::sync::Once;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

/// Route Rust panics and `log` output to the browser console. [`crate::Canvas::find`] calls it,
/// so an example only needs it to log before that; calling it again does nothing.
pub fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Info).ok();
    });
}

/// Seconds since the page started (`performance.now()`), at sub-millisecond resolution.
pub fn now() -> f64 {
    web_sys::window().and_then(|w| w.performance()).map_or(0.0, |p| p.now() / 1000.0)
}

/// The query string's value for `name`, percent-decoded (`?gi=voxel%2Bssgi` reads
/// `voxel+ssgi`; a literal `+` reads as a space, as in any form-encoded URL). `None` when the
/// page's URL has no `name`.
pub fn param(name: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    web_sys::UrlSearchParams::new_with_str(&search).ok()?.get(name)
}

/// The query string's value for `name` parsed as `T`, or `default` when it is missing or does
/// not parse: `let ev: f32 = param_or("ev", 12.0);`.
pub fn param_or<T: FromStr>(name: &str, default: T) -> T {
    param(name).and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

/// A switch in the query string: `1`, `true`, `on` or `yes` turn it on, `0`, `false`, `off` or
/// `no` off; anything else (or no `name`) leaves `default`.
pub fn flag(name: &str, default: bool) -> bool {
    match param(name).as_deref().map(str::trim) {
        Some("1" | "true" | "on" | "yes") => true,
        Some("0" | "false" | "off" | "no") => false,
        _ => default,
    }
}

/// Whether the browser says it is a phone or tablet (by its user agent), for examples that pick
/// a lighter quality tier there.
pub fn is_phone() -> bool {
    let agent = web_sys::window().and_then(|w| w.navigator().user_agent().ok()).unwrap_or_default();
    ["Mobi", "Android", "iPhone", "iPad"].iter().any(|k| agent.contains(k))
}

/// Fetch `url` (relative to the page) as bytes; an HTTP error status is an error too.
pub async fn fetch_bytes(url: &str) -> Result<Vec<u8>, JsValue> {
    let window = web_sys::window().ok_or("no window")?;
    let response: web_sys::Response = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(url)).await?.dyn_into()?;
    if !response.ok() {
        return Err(format!("{url}: HTTP {}", response.status()).into());
    }
    let buffer = wasm_bindgen_futures::JsFuture::from(response.array_buffer()?).await?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}
