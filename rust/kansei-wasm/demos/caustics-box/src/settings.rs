//! Every setting of the page, in one table (as the motion-matching room's): its URL parameter (the
//! initial value), its JSON for the page's panel (`caustics_settings()`), and the live change the
//! panel makes (`caustics_set`).

use kansei_wasm::param;

/// A setting's value: a number, a switch or a name.
pub trait Value: Sized {
    fn parse(text: &str) -> Option<Self>;
    fn json(&self) -> String;
}

impl Value for f32 {
    fn parse(text: &str) -> Option<Self> {
        text.trim().parse().ok().filter(|v: &f32| v.is_finite())
    }
    fn json(&self) -> String {
        format!("{}", self)
    }
}

impl Value for bool {
    fn parse(text: &str) -> Option<Self> {
        match text.trim() {
            "1" | "true" | "on" | "yes" => Some(true),
            "0" | "false" | "off" | "no" => Some(false),
            _ => None,
        }
    }
    fn json(&self) -> String {
        format!("{}", self)
    }
}

impl Value for String {
    fn parse(text: &str) -> Option<Self> {
        Some(text.trim().to_string())
    }
    fn json(&self) -> String {
        format!("\"{}\"", self.replace('"', ""))
    }
}

macro_rules! settings {
    ($($name:ident: $ty:ty = $default:expr;)*) => {
        /// The page's settings (see the module's header).
        #[derive(Debug, Clone, PartialEq)]
        pub struct Settings {
            $(pub $name: $ty,)*
        }

        impl Default for Settings {
            fn default() -> Self {
                Self { $($name: $default,)* }
            }
        }

        impl Settings {
            /// The defaults for the scene the URL names (`scene=box`, the default, or `dragon`),
            /// the quality preset it names (`quality=game`, the default, or `high`), then any
            /// setting it names.
            pub fn from_url() -> Self {
                let mut s = Self::default();
                if let Some(v) = param("scene") {
                    s.scene = if v == "dragon" { "dragon".into() } else { "box".into() };
                }
                s.scene_defaults();
                if let Some(v) = param("quality") {
                    s.apply_quality(&v);
                }
                $(
                    if let Some(v) = param(stringify!($name)) {
                        s.set(stringify!($name), &v);
                    }
                )*
                s
            }

            /// Set `key` from its text; false for an unknown key or a value that doesn't parse.
            pub fn set(&mut self, key: &str, value: &str) -> bool {
                match key {
                    $(stringify!($name) => match <$ty as Value>::parse(value) {
                        Some(v) => {
                            self.$name = v;
                            true
                        }
                        None => false,
                    },)*
                    _ => false,
                }
            }

            /// All of them as a JSON object.
            pub fn json(&self) -> String {
                let fields: Vec<String> = vec![$(format!("\"{}\":{}", stringify!($name), Value::json(&self.$name)),)*];
                format!("{{{}}}", fields.join(","))
            }
        }
    };
}

settings! {
    // the scene: the box of water or the glass dragon (a reload: `scene=`)
    scene: String = "box".into();
    // what a left drag does: push the water along the mouse's motion, orbit, or move the light
    // (a right drag always orbits, a shift drag always moves the light, the wheel zooms)
    drag: String = "push".into();
    // the push: how fast the water under the cursor is made to follow the mouse (the share of
    // the difference a substep), the column's radius (m), its speed's cap (m/s)
    push_strength: f32 = 0.35;
    push_radius: f32 = 0.22;
    push_max_speed: f32 = 4.0;
    // the studio: the floor white or grey, its roughness (under 0.6 it reflects, ray traced),
    // the background as bright as the lit floor far off (1: seamless) or always white
    floor: String = "white".into();
    floor_rough: f32 = 0.9;
    // `floor=checker`: a lab's checkerboard, its squares (m: 0.05, 0.1, 0.25, 0.5 or 1), its
    // metre lines, its axes (x = 0 blue, z = 0 red)
    checker_size: f32 = 0.5;
    checker_lines: bool = true;
    checker_axes: bool = true;
    bg_white: bool = true;
    bg: f32 = 1.0;
    // the far floor faded into the backdrop (no horizon line)
    horizon: bool = true;
    // the key light: a sun (direction) or a spot (direction and distance from the subject);
    // illuminance at the subject (lux), colour (K), its size (degrees across, seen from the
    // subject: the shadows' softness), the white sky's radiance (cd/m²)
    light_type: String = "sun".into();
    light_azim: f32 = -50.0;
    light_elev: f32 = 58.0;
    light_dist: f32 = 6.0;
    light_lux: f32 = 30000.0;
    light_temp: f32 = 6500.0;
    light_size: f32 = 2.0;
    sky_radiance: f32 = 1800.0;
    // the ambient: the hybrid RT GI under the white sky (its occlusion and the floor's bounce) or
    // the flat sky alone; its rays half or full resolution
    gi: String = "rt".into();
    gi_res: String = "half".into();
    // ambient occlusion (`ao`): a game's contact darkening over the lit image, its reach (m),
    // strength (an exponent on the unoccluded share), tightness round contacts (an exponent on
    // the near samples' weight), samples a pixel, the occlusion alone
    ao: bool = true;
    ao_radius: f32 = 0.8;
    ao_intensity: f32 = 3.0;
    ao_falloff: f32 = 1.0;
    ao_samples: f32 = 12.0;
    ao_view: bool = false;
    contact: bool = true;
    contact_length: f32 = 0.2;
    shadow_res: String = "half".into();
    // the caustics (`RtCausticsEffect`): photons a light (a side), the weight of a frame, frames
    // between updates, intensity, the map alone, the water's by photons through its surface (else
    // a net of lines), from the light's size, the glass's gain and splat width (1, 1: physical)
    caustics: bool = true;
    caustics_photons: f32 = 192.0;
    caustics_blend: f32 = 0.08;
    caustics_every: f32 = 1.0;
    caustics_intensity: f32 = 1.0;
    caustics_view: bool = false;
    caustics_water_photons: bool = true;
    // the net of caustic lines on the bed (the shimmer; the photons bring the waves' focus)
    caustics_lines: f32 = 0.7;
    // the texels of the water's height field its normal is taken over for the photons (wider:
    // the particles' bumps smoothed away, the waves kept)
    caustics_smooth: f32 = 3.0;
    caustics_area: bool = false;
    caustics_glass_gain: f32 = 2.0;
    caustics_sharpen: f32 = 0.6;
    // the glass (the dragon, the box's sphere): index of refraction, frost, a tint toward aqua
    glass_ior: f32 = 1.5;
    glass_rough: f32 = 0.0;
    glass_tint: f32 = 0.15;
    glass_checker: bool = true;
    glass_refl_shared: bool = true;
    // the box's glass sphere; the dragon's mesh (the full scan or its decimation, a reload) and
    // its cluster LOD (off: the whole mesh everywhere, a reload)
    sphere: bool = true;
    dragon: String = "full".into();
    dragon_lod: bool = false;
    // the error (mm) of the dragon's copy in the ray tracing grid (a reload: 40 mm, about 45k
    // triangles of the full scan's 871k)
    dragon_rt_mm: f32 = 40.0;
    // the water (see `Water::apply`): the box's inside (m: width x, depth z, wall height) and the
    // water's depth (m), all refilling it
    box_w: f32 = 3.0;
    box_d: f32 = 2.0;
    box_h: f32 = 0.55;
    fluid_depth: f32 = 0.28;
    fluid: bool = true;
    fluid_show: bool = true;
    fluid_rest: bool = true;
    fluid_solver: String = "sph".into();
    fluid_speed: f32 = 1.0;
    fluid_substeps: f32 = 4.0;
    fluid_viscosity: f32 = 0.15;
    fluid_pressure: f32 = 46.5;
    fluid_near: f32 = 20.0;
    fluid_cohesion: f32 = 0.6;
    fluid_damping: f32 = 1.0;
    fluid_gravity: f32 = 9.8;
    fluid_pbf_iter: f32 = 3.0;
    fluid_xsph: f32 = 0.1;
    // drawn as ray-traced glass ("rt") or the screen-space composite ("screen")
    fluid_render: String = "rt".into();
    fluid_ior: f32 = 1.33;
    fluid_tint: f32 = 1.0;
    fluid_rough: f32 = 0.02;
    fluid_ripples: f32 = 0.35;
    // the surface: the field's voxels along the box's longest side, the marching cubes' cap, a
    // lone particle's droplet radius and the splat radius (m), the ray tracing grid's mesh
    // spacing (m); the box drawn as a wireframe
    fluid_res: f32 = 160.0;
    fluid_mc_max: f32 = 400000.0;
    fluid_drop: f32 = 0.03;
    fluid_splat: f32 = 0.08;
    fluid_rt_spacing: f32 = 0.08;
    fluid_bounds_show: bool = false;
    // the post-processing
    tonemap: String = "neutral".into();
    // the exposure: the lit white floor at display white (`auto_ev`), or `ev` (EV100)
    auto_ev: bool = true;
    ev: f32 = 12.6;
    exposure_comp: f32 = 0.0;
    bloom: bool = true;
    bloom_intensity: f32 = 0.04;
    vignette: f32 = 0.12;
    taa: bool = true;
    // the page: the canvas's height in pixels ("native": the window's times the pixel ratio),
    // the pixel ratio (0: the screen's, at most 2), the scene's share of the canvas
    resolution: String = "native".into();
    dpr: f32 = 0.0;
    // (0: automatic, the scene at about `AUTO_PIXELS` whatever the canvas, TAA upscaling)
    render_scale: f32 = 0.0;
    stats: bool = false;
    // the performance overlay (kansei_wasm::perf; also the ` key)
    perf: bool = false;
}

impl Settings {
    /// What the scene changes of the defaults: the dragon's drag orbits, its light higher and
    /// to the side (its shadow and caustics fall toward the camera), its glass clear.
    fn scene_defaults(&mut self) {
        if self.scene == "dragon" {
            self.drag = "orbit".into();
            self.light_elev = 58.0;
            self.light_azim = 125.0;
            self.glass_tint = 0.0;
        }
        self.apply_quality("game");
    }

    /// The quality preset (the panel's): `game` (the default) traces half the glass's pixels a
    /// frame and its reflections with the other surfaces', the caustics' photons from the light's
    /// centre, the glass's light brighter and narrower, the GI's and shadows' rays one per 2 x 2
    /// pixels; `high` traces every glass pixel and its reflection, every pixel's GI and shadow
    /// ray, the caustics from the light's size, physical, with more photons. The dragon scatters
    /// its light far more than the water focuses it: twice the photons a side, a wider splat
    /// and a slower blend there, so the floor doesn't speckle, fed every other frame in `game`.
    pub fn apply_quality(&mut self, quality: &str) {
        let high = quality == "high";
        let dragon = self.scene == "dragon";
        self.glass_checker = !high;
        self.glass_refl_shared = !high;
        self.caustics_area = high;
        self.caustics_photons = match (dragon, high) {
            (false, false) => 192.0,
            (false, true) => 320.0,
            (true, false) => 384.0,
            (true, true) => 640.0,
        };
        self.caustics_blend = if dragon { 0.04 } else { 0.08 };
        // (the still dragon's caustics fed every other frame; the water's, which move, every frame)
        self.caustics_every = if dragon && !high { 2.0 } else { 1.0 };
        self.caustics_glass_gain = if high { 1.0 } else if dragon { 1.5 } else { 2.0 };
        self.caustics_sharpen = if high || dragon { 1.0 } else { 0.6 };
        self.gi_res = if high { "full".into() } else { "half".into() };
        self.shadow_res = if high { "full".into() } else { "half".into() };
    }

    /// The settings `apply_quality` sets.
    pub const QUALITY_KEYS: [&'static str; 10] = ["glass_checker", "glass_refl_shared", "caustics_area", "caustics_photons", "caustics_blend", "caustics_every", "caustics_glass_gain", "caustics_sharpen", "gi_res", "shadow_res"];
}

/// A colour temperature (K) as linear rgb with its largest channel 1 (Tanner Helland's fit, then
/// to linear).
pub fn kelvin(k: f32) -> [f32; 3] {
    let t = (k / 100.0).clamp(10.0, 400.0);
    let r = if t <= 66.0 { 255.0 } else { 329.698_73 * (t - 60.0).powf(-0.133_204_76) };
    let g = if t <= 66.0 { 99.470_8 * t.ln() - 161.119_57 } else { 288.122_16 * (t - 60.0).powf(-0.075_514_85) };
    let b = if t >= 66.0 { 255.0 } else if t <= 19.0 { 0.0 } else { 138.517_73 * (t - 10.0).ln() - 305.044_8 };
    let lin = |c: f32| {
        let c = (c / 255.0).clamp(0.0, 1.0);
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let c = [lin(r), lin(g), lin(b)];
    let m = c[0].max(c[1]).max(c[2]).max(1e-6);
    c.map(|v| v / m)
}
