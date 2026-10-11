//! Every setting of the room page, in one table: its URL parameter (the initial value), its JSON for
//! the page's panel (`room_settings()`), and the live change the panel makes (`room_set`).

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
            /// The defaults, with any setting the URL names.
            pub fn from_url() -> Self {
                let mut s = Self::default();
                // the old name of `fluid_rest`
                if let Some(v) = param("rest") {
                    s.set("fluid_rest", &v);
                }
                // `quality=high`: the physical paths (`game`, the default: the approximations)
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
    // the scene
    floor: String = "marble".into();
    floor_rough: f32 = 1.0;
    cam: String = "follow".into();
    // global illumination, shadows, reflections
    gi: String = "rt".into();
    rtgi_res: String = "half".into();
    gi_shadows: String = "maps".into();
    probe_spacing: f32 = 6.0;
    probes_per_frame: f32 = 0.0;
    gi_near: f32 = 0.0;
    shadows: String = "rt".into();
    shadow_res: String = "half".into();
    contact: bool = true;
    contact_length: f32 = 0.25;
    sun_soft: f32 = 1.2;
    shadow_steps: f32 = 2.0;
    reflections: bool = true;
    refl_indirect: f32 = 1.0;
    refl_intensity: f32 = 1.0;
    refl_screen: bool = true;
    rt_view: String = "lit".into();
    debug_light: f32 = 0.0;
    // the lights
    sun: bool = true;
    sun_lux: f32 = 9000.0;
    sun_elev: f32 = 24.0;
    sun_azim: f32 = 20.0;
    sun_temp: f32 = 3600.0;
    sky: bool = true;
    sky_cd: f32 = 32000.0;
    sky_temp: f32 = 7500.0;
    sky_radiance: f32 = 2500.0;
    lamps: bool = true;
    lamp_cd: f32 = 700.0;
    lamp_temp: f32 = 2700.0;
    // the air
    fog: f32 = 0.002;
    fog_aniso: f32 = 0.55;
    dust: bool = true;
    dust_amount: f32 = 1.0;
    dust_size: f32 = 0.008;
    dust_bright: f32 = 2.5;
    dust_opacity: f32 = 0.7;
    dust_mix: f32 = 0.3;
    dust_speed: f32 = 0.06;
    // the pond's water (see `Pond::apply`; `fluid_fill` takes a reset)
    fluid: bool = true;
    fluid_show: bool = true;
    fluid_rest: bool = true;
    fluid_solver: String = "sph".into();
    fluid_fill: f32 = 1.0;
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
    fluid_push: f32 = 0.6;
    fluid_bounce: f32 = 0.3;
    fluid_splash: f32 = 1.2;
    fluid_mesh: String = "smooth".into();
    fluid_ior: f32 = 1.33;
    fluid_tint: f32 = 1.0;
    fluid_rough: f32 = 0.08;
    fluid_thickness: f32 = 1.2;
    fluid_reflect: f32 = 0.6;
    fluid_chromatic: f32 = 0.02;
    // drawn as ray-traced glass ("rt": refracted and reflected through the grid, in it for the
    // mirrors) or the screen-space composite ("screen"); the small wind ripples on it
    fluid_render: String = "rt".into();
    fluid_bed: String = "marble".into();
    fluid_ripples: f32 = 1.0;
    // the caustics of the glass and the water (`RtCausticsEffect`): photons a light (a side), the
    // weight of a frame, the light's scale; the display spot over the dragon
    caustics: bool = true;
    caustics_photons: f32 = 128.0;
    caustics_blend: f32 = 0.1;
    // frames between the caustics' updates (four times that with the pond out of view)
    caustics_every: f32 = 2.0;
    caustics_intensity: f32 = 1.0;
    caustics_view: bool = false;
    // the water's caustics from photons (else its ripples' focus), the lights' emitters' size
    caustics_water_photons: bool = false;
    caustics_area: bool = false;
    display_spot: bool = true;
    display_cd: f32 = 60000.0;
    // materials
    mirror_rough: f32 = 0.0;
    mirror2_rough: f32 = 0.1;
    glass_ior: f32 = 1.5;
    glass_rough: f32 = 0.0;
    // the glass pass (the dragon, the water) traces half its pixels a frame; the glass's
    // reflections traced with the other surfaces' (half resolution, accumulated)
    glass_checker: bool = true;
    glass_refl_shared: bool = true;
    // the post-processing
    tonemap: String = "agx".into();
    ev: f32 = 9.0;
    exposure_comp: f32 = 0.0;
    local_exposure: bool = false;
    white_temp: f32 = 6500.0;
    tint: f32 = 0.0;
    contrast: f32 = 1.05;
    saturation: f32 = 1.05;
    lift: f32 = 1.0;
    gain: f32 = 1.0;
    highlights: f32 = 1.0;
    bloom: bool = true;
    bloom_intensity: f32 = 0.06;
    bloom_threshold: f32 = 1.0;
    bloom_radius: f32 = 1.0;
    vignette: f32 = 0.28;
    grain: f32 = 0.06;
    chromatic: f32 = 0.08;
    taa: bool = true;
    motion_blur: bool = false;
    motion_amount: f32 = 0.5;
    // the depth of field
    dof: bool = true;
    dof_focus: String = "character".into();
    dof_distance: f32 = 5.0;
    f_stop: f32 = 5.6;
    focal_mm: f32 = 0.0;
    blades: f32 = 6.0;
    blade_rot: f32 = 0.0;
    max_coc: f32 = 0.02;
    // the page: the canvas's height in pixels ("native": the window's times the pixel ratio),
    // the pixel ratio (0: the screen's, at most 2) and the scene's share of the canvas
    resolution: String = "native".into();
    dpr: f32 = 0.0;
    render_scale: f32 = 1.0;
    stats: bool = false;
}

impl Settings {
    /// The quality preset (the panel's): `game` (the default) traces half the glass's pixels a frame
    /// and its reflections with the other surfaces', the water's caustics as its ripples' focus,
    /// the photons from the lights' centres, the map fed every other frame; `high` traces every glass pixel and its reflection,
    /// the water's caustics by photons through it, from the lights' emitters, 4x the photons.
    pub fn apply_quality(&mut self, quality: &str) {
        let high = quality == "high";
        self.glass_checker = !high;
        self.glass_refl_shared = !high;
        self.caustics_water_photons = high;
        self.caustics_area = high;
        self.caustics_photons = if high { 256.0 } else { 128.0 };
        self.caustics_every = if high { 1.0 } else { 2.0 };
    }
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
