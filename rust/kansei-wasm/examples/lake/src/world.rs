//! The world the lake page and the motion-matching example share: the sky, the ground, the course
//! of boxes, the lake with its cannon and mill, the sun, and the post-processing chain the lake's
//! surface needs. Built once ([`World::new`], then [`World::post_processing`]) and stepped every
//! frame ([`World::update`]) with whatever stands in the water.

use glam::{Mat4, Vec3 as GVec3};

use kansei_core::collision::{CollisionWorld, Obb};
use kansei_core::geometries::{PlaneGeometry, SphereGeometry};
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    effects::{exposure_from_ev100, FluidSurfaceEffect, TemporalAAEffect, TemporalAAOptions, ToneMapEffect, ToneMapOptions},
    PostProcessingEffect, PostProcessingVolume,
};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::shadows::{CascadedShadowOptions, CASCADED_SHADOWS_WGSL};
use kansei_wasm::{flag, Canvas};

use crate::cannon::Cannon;
use crate::course::{build_course, surface_params, GROUND_WGSL, SKY_WGSL};
use crate::lake::{self, Lake};
use crate::mill::Mill;
use crate::SUN_DIR;

/// What the world is built with.
#[derive(Debug, Clone, Copy)]
pub struct WorldOptions {
    /// The course of boxes.
    pub course: bool,
    /// The lake, its cannon and its mill (else a flat ground plane).
    pub lake: bool,
    /// Whether the lake's water may rest (culled out of view, asleep when settled).
    pub rest: bool,
    /// Whether the mill starts turning.
    pub mill: bool,
    /// Temporal anti-aliasing in the post-processing chain.
    pub taa: bool,
    /// The cannon fires only with a character within its reach ([`WorldInput::at`]); else from
    /// anywhere, as it stands (the lake page, which has no character).
    pub cannon_reach: bool,
}

impl WorldOptions {
    /// From the page's URL: `course=0`, `lake=0`, `rest=0`, `mill=0` and `taa=0` turn those off;
    /// the cannon needs a character within reach.
    pub fn from_url() -> Self {
        Self { course: flag("course", true), lake: flag("lake", true), rest: flag("rest", true), mill: flag("mill", true), taa: flag("taa", true), cannon_reach: true }
    }
}

/// What acts on the world this frame.
#[derive(Debug, Default, Clone, Copy)]
pub struct WorldInput<'a> {
    /// Capsules in the water (world ends and radius): a character's legs.
    pub legs: &'a [(GVec3, GVec3, f32)],
    /// A landing: where the feet came down and how fast (m/s).
    pub landing: Option<(GVec3, f32)>,
    /// Where the character stands, for the cannon's reach.
    pub at: Option<GVec3>,
    /// The cannon's trigger from the keyboard or a gamepad: pressed this frame, held. A click on
    /// the page's prompt ([`World::fire`]) adds to it.
    pub fire: (bool, bool),
    /// Drain the lake back to its start (by the cannon).
    pub drain: bool,
}

pub struct World {
    /// The floor, the course, the lake's terrain and the props, for a character to walk on.
    pub collision: CollisionWorld,
    pub lake: Option<Lake>,
    /// The props by the lake: the water cannon and the mill (with the lake only).
    pub cannon: Option<Cannon>,
    pub mill: Option<Mill>,
    /// The lake's surface effect, until [`World::post_processing`] puts it in the chain.
    surface: Option<FluidSurfaceEffect>,
    taa: bool,
    cannon_reach: bool,
    /// The cannon's trigger from the page (a click on its prompt): pressed since last frame, held.
    pointer_fire: (bool, bool),
    /// The prompt shown and the still water's height drawn, as last set.
    prompt: String,
    level: f32,
}

/// A renderer on `canvas` for the world: one sample per pixel, cascaded shadows to 60 m.
pub async fn renderer(canvas: &Canvas) -> Renderer {
    let mut renderer = canvas.renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() }).await;
    renderer.enable_cascaded_shadows(CascadedShadowOptions { max_distance: 60.0, ..Default::default() });
    renderer
}

/// Show `text` in the page's `#prompt` element (hidden when empty), if it has one.
fn set_prompt(text: &str) {
    if let Some(prompt) = web_sys::window().and_then(|w| w.document()).and_then(|d| d.get_element_by_id("prompt")) {
        prompt.set_text_content(Some(text));
        let _ = prompt.set_attribute("data-visible", if text.is_empty() { "0" } else { "1" });
    }
}

impl World {
    /// Build the world into `scene`: the course, the sky, the ground (with the lake, a hole its
    /// terrain fills), the cannon and the mill, and the sun.
    pub fn new(renderer: &Renderer, scene: &mut Scene, options: &WorldOptions) -> Self {
        // the floor and the course, for collision
        let mut collision = CollisionWorld::new();
        if !options.lake {
            collision.add_box(Obb::from_min_max(GVec3::new(-200.0, -1.0, -200.0), GVec3::new(200.0, 0.0, 200.0)));
        }
        if options.course {
            build_course(scene, &mut collision);
        }
        let mut sky = Material::new("Sky", SKY_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { cull_mode: CullMode::None, ..Default::default() });
        sky.set_uniform_bindable(0, "Sky", &[0.0f32; 4]);
        let mut sky = Renderable::new(SphereGeometry::new(900.0, 32, 16), sky);
        sky.cast_shadow = false;
        scene.add(SceneNode::Renderable(sky));
        let mut ground_material = Material::new("Ground", &format!("{CASCADED_SHADOWS_WGSL}\n{GROUND_WGSL}"), vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions::default());
        ground_material.set_uniform_bindable(0, "Ground", &surface_params([0.32, 0.32, 0.3]));
        // with the lake, the ground has a hole the lake's terrain fills
        let (mut lake, mut surface, mut cannon, mut mill) = (None, None, None, None);
        if options.lake {
            let (mut l, s) = Lake::new(renderer, scene, &mut collision, ground_material);
            l.rest = options.rest;
            cannon = Some(Cannon::new(scene, &mut collision, &l));
            let mut m = Mill::new(scene, &mut collision, &l);
            m.on = options.mill;
            mill = Some(m);
            lake = Some(l);
            surface = Some(s);
        } else {
            let mut ground = Renderable::new(PlaneGeometry::new(400.0, 400.0), ground_material);
            ground.object.rotation.x = -std::f32::consts::FRAC_PI_2;
            ground.cast_shadow = false;
            scene.add(SceneNode::Renderable(ground));
        }
        let mut sun = DirectionalLight::new(Vec3::new(SUN_DIR[0], SUN_DIR[1], SUN_DIR[2]), Vec3::new(1.0, 0.9, 0.75), 80000.0);
        sun.cast_shadow = true;
        scene.add(SceneNode::Light(Light::Directional(sun)));
        let level = lake.as_ref().map_or(0.0, Lake::level);
        Self { collision, lake, cannon, mill, surface, taa: options.taa, cannon_reach: options.cannon_reach, pointer_fire: (false, false), prompt: String::new(), level }
    }

    /// The post-processing chain: the lake's refraction and reflection first, on the lit scene (its
    /// surface renderable goes into `scene` now), then TAA and the tone map.
    pub fn post_processing(&mut self, renderer: &Renderer, scene: &mut Scene) -> PostProcessingVolume {
        let tonemap = {
            let mut o = ToneMapOptions::for_surface(renderer.presentation_format());
            o.exposure = exposure_from_ev100(14.0);
            o.vignette = 0.3;
            ToneMapEffect::new(o)
        };
        let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
        if let Some(surface) = self.surface.take() {
            Lake::add_surface(scene, &surface);
            effects.push(Box::new(surface));
        }
        if self.taa {
            effects.push(Box::new(TemporalAAEffect::new(TemporalAAOptions { exposure: tonemap.total_exposure(), ..Default::default() })));
        }
        effects.push(Box::new(tonemap));
        PostProcessingVolume::new(renderer, effects)
    }

    /// The cannon's trigger from the page (a click or touch on the prompt): `down` fires (a burst,
    /// and it pours while held), `false` releases it.
    pub fn fire(&mut self, down: bool) {
        self.pointer_fire = (self.pointer_fire.0 || down, down);
    }

    /// What the page prompts near the cannon, or "" away from it.
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// Step the world by `dt`: the mill turns, the cannon fires (within reach of `input.at` when
    /// the world was built with `cannon_reach`), and the lake's water steps with `input`'s legs, a
    /// landing, the mill's paddles and the cannon's stream in it, resting when out of the view
    /// `view_proj` (world to clip) or settled. The bed's wet line and the page's `#prompt` follow.
    #[allow(clippy::too_many_arguments)]
    pub fn update(&mut self, input: WorldInput, dt: f32, scene: &mut Scene, volume: &mut PostProcessingVolume, renderer: &Renderer, view_proj: Mat4) {
        let fire_pressed = input.fire.0 || self.pointer_fire.0;
        let fire_held = input.fire.1 || self.pointer_fire.1;
        self.pointer_fire.0 = false;
        let Some(lake) = &mut self.lake else { return };
        if let Some(surface) = volume.effect_mut::<FluidSurfaceEffect>() {
            // the props: the mill turns, the cannon fires when the character stands by it
            let (bodies, stirring) = match &mut self.mill {
                Some(mill) => {
                    mill.update(dt, scene);
                    (mill.capsules(), mill.turning())
                }
                None => (Vec::new(), false),
            };
            let stream = match &mut self.cannon {
                Some(cannon) => {
                    let near = !self.cannon_reach || input.at.is_some_and(|at| cannon.in_reach(at));
                    if input.drain && cannon.near {
                        lake.reset(surface);
                    }
                    cannon.update(dt, near, fire_pressed, fire_held, lake, scene)
                }
                None => None,
            };
            let poured = lake.update(surface, input.legs, input.landing, &bodies, stirring, stream, dt, view_proj);
            if let Some(cannon) = &mut self.cannon {
                cannon.poured += poured as u64;
            }
        }
        // the wet line on the bed follows the still water's height
        let level = lake.level();
        if (level - self.level).abs() > 0.002 {
            if let Some(buffer) = scene.get_renderable_mut(lake.terrain).and_then(|r| r.material.bindable_buffer(0)) {
                renderer.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(&lake.terrain_uniform()));
                self.level = level;
            }
        }
        let prompt = self.cannon.as_ref().map_or(String::new(), |c| c.prompt(lake));
        if prompt != self.prompt {
            set_prompt(&prompt);
            self.prompt = prompt;
        }
    }

    /// The HUD's lines about the lake, the cannon and the mill ("" without a lake), with `place`
    /// (a hint where to find it, ending in ", ") on the first.
    pub fn status(&self, place: &str) -> String {
        self.lake.as_ref().map_or(String::new(), |l| {
            format!(
                "lake   {} / {} particles, {:.0}% full (level {:+.3} m), {place}P tweaks\nwater  {}, fastest {:.2} m/s, {} over {} m/s{}{}\n",
                l.particles(),
                l.capacity(),
                l.fill() * 100.0,
                l.level(),
                l.state().name(),
                l.speed().0,
                l.speed().1,
                lake::SETTLE_SPEED,
                self.cannon.as_ref().map_or(String::new(), |c| format!("\ncannon {}{} poured", if c.firing() { "firing, " } else if c.near { "ready, " } else { "" }, c.poured)),
                self.mill.as_ref().map_or(String::new(), |m| format!("\nmill   {}", if m.turning() { format!("{:.0} rpm", m.rpm) } else { "stopped".to_string() })),
            )
        })
    }
}
