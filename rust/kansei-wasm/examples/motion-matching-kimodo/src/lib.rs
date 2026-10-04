//! The motion-matching example on generated animation: NVIDIA Kimodo clips baked into packs by
//! `kansei-anim-bake/genanim`, to feel how they play under the stick.
//!
//! Everything but the start is the motion-matching example's (`kansei_wasm_motion_matching`): the
//! scene, the controller, the course and the lake, the HUD, and the exports the page calls
//! (`clip_names`, `play_clips`, `set_drive`, `drive_restart`, the lake's panel). The page picks
//! the pack (`pack=`, `walk=`, `run=`, `hero=`); see this example's README.

use wasm_bindgen::prelude::*;

use kansei_wasm_motion_matching::{fetch_bytes, query_param, start_with_loader};

/// Start the example on canvas `canvas_id`. Unlike the motion-matching example, it loads a
/// character pack only when the page names one (`hero=<url>`): the generated packs carry their
/// own SOMA body, and no `pack/hero.kmm` sits next to them.
#[wasm_bindgen]
pub async fn start_kimodo(canvas_id: &str) -> Result<(), JsValue> {
    let hero = query_param("hero");
    start_with_loader(canvas_id, move |url: String| {
        let skip = url == "pack/hero.kmm" && hero.is_none();
        async move {
            if skip {
                return Err("none asked for (hero=<url>)".to_string());
            }
            fetch_bytes(&url).await
        }
    })
    .await
}
