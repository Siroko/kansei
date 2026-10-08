//! The page around the canvas: logging, the clock, the query string and fetching files. In a
//! worker ([`crate::launch`]) each reads the page it was launched from.

use std::str::FromStr;
use std::sync::Once;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::worker;

#[wasm_bindgen]
extern "C" {
    // the global `performance` and `fetch`, a page's or a worker's
    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn performance_now() -> f64;
    #[wasm_bindgen(js_name = fetch)]
    fn global_fetch(url: &str) -> js_sys::Promise;
}

/// Route Rust panics and `log` output to the browser console. [`crate::Canvas::find`] calls it,
/// so an example only needs it to log before that; calling it again does nothing.
pub fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Info).ok();
    });
}

/// Seconds since the page started (`performance.now()`; in a worker, since the worker
/// started), at sub-millisecond resolution.
pub fn now() -> f64 {
    performance_now() / 1000.0
}

/// The query string's value for `name`, percent-decoded (`?gi=voxel%2Bssgi` reads
/// `voxel+ssgi`; a literal `+` reads as a space, as in any form-encoded URL). `None` when the
/// page's URL has no `name`.
pub fn param(name: &str) -> Option<String> {
    let search = if worker::in_worker() { worker::page_search() } else { web_sys::window()?.location().search().ok()? };
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
    // the global navigator: a page's or a worker's
    let agent = js_sys::Reflect::get(&js_sys::global(), &"navigator".into())
        .and_then(|n| js_sys::Reflect::get(&n, &"userAgent".into()))
        .ok()
        .and_then(|a| a.as_string())
        .unwrap_or_default();
    ["Mobi", "Android", "iPhone", "iPad"].iter().any(|k| agent.contains(k))
}

/// Show `text` in the page element with id `id` (a HUD), if there is one. From a worker the
/// text is posted to the page.
pub fn set_text(id: &str, text: &str) {
    if worker::in_worker() {
        worker::post_text(id, text);
    } else if let Some(element) = web_sys::window().and_then(|w| w.document()).and_then(|d| d.get_element_by_id(id)) {
        element.set_text_content(Some(text));
    }
}

/// The page's checkbox with id `id`, if there is one (a HUD's toggles). `None` in a worker: a
/// page that runs its example in one sends its toggles as calls.
pub fn checkbox(id: &str) -> Option<web_sys::HtmlInputElement> {
    web_sys::window()?.document()?.get_element_by_id(id)?.dyn_into().ok()
}

/// `n` with its thousands apart, for a HUD: `40000` reads `40 000`.
pub fn thousands(n: impl Into<u64>) -> String {
    let s = n.into().to_string();
    let mut out = String::new();
    for (k, c) in s.chars().enumerate() {
        if k > 0 && (s.len() - k).is_multiple_of(3) {
            out.push(' ');
        }
        out.push(c);
    }
    out
}

/// Fetch `url` (relative to the page, in a worker too) as bytes; an HTTP error status is an
/// error too.
pub async fn fetch_bytes(url: &str) -> Result<Vec<u8>, JsValue> {
    let resolved = if worker::in_worker() { worker::page_url(url) } else { url.to_string() };
    let response: web_sys::Response = wasm_bindgen_futures::JsFuture::from(global_fetch(&resolved)).await?.dyn_into()?;
    if !response.ok() {
        return Err(format!("{url}: HTTP {}", response.status()).into());
    }
    let buffer = wasm_bindgen_futures::JsFuture::from(response.array_buffer()?).await?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

#[cfg(test)]
mod tests {
    #[test]
    fn thousands_group_by_three() {
        assert_eq!(super::thousands(7u32), "7");
        assert_eq!(super::thousands(40_000u32), "40 000");
        assert_eq!(super::thousands(1_234_567u64), "1 234 567");
    }
}
