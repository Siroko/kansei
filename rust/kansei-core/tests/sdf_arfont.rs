use kansei_core::sdf::FontAtlas;

const FONT: &[u8] = include_bytes!("fixtures/L10-medium.arfont");

#[test]
fn parses_all_clock_glyphs() {
    let atlas = FontAtlas::parse(FONT).expect("parse .arfont");

    // Every clock glyph must be present: '0'..'9' and ':'.
    for cp in ('0'..='9').chain([':'].into_iter()) {
        let g = atlas
            .glyphs
            .iter()
            .find(|g| g.codepoint == cp as u32)
            .unwrap_or_else(|| panic!("missing glyph for {cp:?}"));

        // image_bounds must be a sane sub-rect inside the 448×448 atlas.
        let [l, b, r, t] = g.image_bounds;
        assert!(r > l && t >= b, "glyph {cp:?} bounds not ordered: {:?}", g.image_bounds);
        assert!(l >= 0.0 && r <= atlas.width as f32, "glyph {cp:?} x out of range");
        assert!(b >= 0.0 && t <= atlas.height as f32, "glyph {cp:?} y out of range");
    }
}

#[test]
fn reports_atlas_dimensions_and_metrics() {
    let atlas = FontAtlas::parse(FONT).expect("parse .arfont");
    assert_eq!(atlas.width, 448);
    assert_eq!(atlas.height, 448);
    assert_eq!(atlas.rgba.len(), (448 * 448 * 4) as usize);
    assert!(atlas.distance_range > 0.0);
    assert!(atlas.em_size > 0.0);
}

#[test]
fn atlas_alpha_channel_is_populated_sdf() {
    let atlas = FontAtlas::parse(FONT).expect("parse .arfont");
    // A real MTSDF alpha channel spans the full range across the atlas.
    let mut min_a = 255u8;
    let mut max_a = 0u8;
    for px in atlas.rgba.chunks_exact(4) {
        min_a = min_a.min(px[3]);
        max_a = max_a.max(px[3]);
    }
    assert!(max_a > min_a, "alpha channel is flat — PNG not decoded");
    assert!(max_a > 200 && min_a < 55, "alpha does not span SDF range: {min_a}..{max_a}");
}
