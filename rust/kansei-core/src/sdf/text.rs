//! MSDF text: glyphs from an `.arfont` atlas drawn as instanced quads ([`Material::msdf_text`]),
//! laid out from the atlas's metrics ([`layout_line`]).
//!
//! ```ignore
//! let atlas = FontAtlas::parse(&bytes)?;
//! let glyphs = layout_line(&atlas, "Kansei", 2.0, 1.0);
//! // one instance per glyph: position (location 3), atlas rect (4), plane rect (5), colour (6)
//! let material = Material::msdf_text("Title", &atlas, MsdfTextOptions::default());
//! ```

use super::{FontAtlas, GlyphMetrics};
use crate::buffers::{Sampler, Texture};
use crate::materials::{Binding, CullMode, Material, MaterialOptions};

/// The glyph material's shader; see [`Material::msdf_text`].
pub const MSDF_TEXT_WGSL: &str = include_str!("../shaders/msdf_text.wgsl");

/// Where a glyph's quad samples the atlas and where it sits around its instance's position,
/// as the [`MSDF_TEXT_WGSL`] instance attributes take them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphRects {
    /// Left, top, right, bottom in texture uv (v runs down the atlas image).
    pub atlas: [f32; 4],
    /// Left, top, right, bottom around the glyph's origin, in world units (y up).
    pub plane: [f32; 4],
}

/// A glyph placed on a line by [`layout_line`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlacedGlyph {
    pub character: char,
    /// The glyph's origin along the line, from the line's start, world units.
    pub x: f32,
    pub rects: GlyphRects,
}

/// How [`Material::msdf_text`] places a glyph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MsdfTextOptions {
    /// Turn each glyph about its own x axis by its position's w (radians); otherwise w is
    /// ignored.
    pub rotate_x_by_w: bool,
}

impl FontAtlas {
    /// The metrics of `character`, if the atlas has it.
    pub fn glyph(&self, character: char) -> Option<&GlyphMetrics> {
        self.glyphs.iter().find(|g| g.codepoint == character as u32)
    }

    /// `glyph`'s rects for text `size` world units per em.
    pub fn glyph_rects(&self, glyph: &GlyphMetrics, size: f32) -> GlyphRects {
        let [l, b, r, t] = glyph.image_bounds;
        let (w, h) = (self.width as f32, self.height as f32);
        let [pl, pb, pr, pt] = glyph.plane_bounds;
        // image bounds count y up from the atlas's bottom; its rows run down
        GlyphRects { atlas: [l / w, 1.0 - t / h, r / w, 1.0 - b / h], plane: [pl * size, pt * size, pr * size, pb * size] }
    }

    /// The atlas image as a texture (RGBA8, linear: the channels are distances, not colour).
    pub fn texture(&self, label: &str) -> Texture {
        Texture::from_rgba(label, self.width, self.height, &self.rgba)
    }
}

/// `text` on one line from x = 0, `size` world units per em, each advance times
/// `letter_spacing`; characters the atlas lacks advance by half an em and draw nothing, spaces
/// advance by the atlas's space (or half an em).
pub fn layout_line(atlas: &FontAtlas, text: &str, size: f32, letter_spacing: f32) -> Vec<PlacedGlyph> {
    let mut x = 0.0;
    let mut placed = Vec::new();
    for character in text.chars() {
        let glyph = atlas.glyph(character);
        if let (Some(glyph), false) = (glyph, character.is_whitespace()) {
            placed.push(PlacedGlyph { character, x, rects: atlas.glyph_rects(glyph, size) });
        }
        x += glyph.map_or(0.5, |g| g.advance) * size * letter_spacing;
    }
    placed
}

impl Material {
    /// A material that draws MSDF glyphs from `atlas`: put it on an `InstancedGeometry` of a
    /// `PlaneGeometry(1, 1)` quad with four vec4 instance attributes at locations 3-6 (position,
    /// atlas rect, plane rect, colour; see [`MSDF_TEXT_WGSL`] and [`FontAtlas::glyph_rects`]).
    /// Double-sided, alpha-blended yet writing depth (the cut-out discards the faint edge, so
    /// glyphs hide what is behind them, whatever the draw order), sampled anisotropically for
    /// text seen at an angle.
    pub fn msdf_text(label: &str, atlas: &FontAtlas, options: MsdfTextOptions) -> Material {
        let shader = MSDF_TEXT_WGSL.replace("KANSEI_ROTATE_X", if options.rotate_x_by_w { "true" } else { "false" });
        let material_options = MaterialOptions { cull_mode: CullMode::None, transparent: true, depth_write: Some(true), ..Default::default() };
        let mut material = Material::new(
            label,
            &shader,
            vec![Binding::texture_2d(0, wgpu::ShaderStages::FRAGMENT), Binding::sampler(1, wgpu::ShaderStages::FRAGMENT)],
            material_options,
        );
        material.set_bindable(0, atlas.texture(&format!("{label}/Atlas")));
        material.set_bindable(1, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_anisotropy(8));
        material
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atlas() -> FontAtlas {
        let glyph = |codepoint: char, advance: f32| GlyphMetrics { codepoint: codepoint as u32, advance, image_bounds: [10.0, 20.0, 30.0, 60.0], plane_bounds: [0.05, -0.1, 0.55, 0.7] };
        FontAtlas { width: 100, height: 200, rgba: vec![0; 100 * 200 * 4], glyphs: vec![glyph('A', 0.6), glyph(' ', 0.25), glyph('B', 0.5)], distance_range: 4.0, em_size: 32.0 }
    }

    #[test]
    fn glyph_rects_flip_the_atlas_rows_and_scale_the_plane() {
        let a = atlas();
        let rects = a.glyph_rects(a.glyph('A').unwrap(), 2.0);
        assert_eq!(rects.atlas, [0.1, 1.0 - 60.0 / 200.0, 0.3, 1.0 - 20.0 / 200.0]);
        assert_eq!(rects.plane, [0.1, 1.4, 1.1, -0.2]);
    }

    #[test]
    fn a_line_advances_by_each_glyph_and_skips_spaces() {
        let placed = layout_line(&atlas(), "A B?", 2.0, 1.0);
        assert_eq!(placed.iter().map(|g| (g.character, g.x)).collect::<Vec<_>>(), vec![('A', 0.0), ('B', 1.7)]);
    }

    #[test]
    fn both_variants_of_the_shader_validate() {
        for rotate in ["true", "false"] {
            let code = MSDF_TEXT_WGSL.replace("KANSEI_ROTATE_X", rotate);
            let module = naga::front::wgsl::parse_str(&code).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&code)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
        }
    }
}
