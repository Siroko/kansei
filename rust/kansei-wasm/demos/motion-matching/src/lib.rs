//! Motion matching: a skinned character walking, running, stopping and turning under keyboard
//! or gamepad control, animated by searching a database of animation frames
//! (`kansei_core::animation::motion_matching`), on a sunlit ground plane with cascaded shadows
//! and TAA (the character writes motion vectors from last frame's bone palette).
//!
//! The animation comes from a motion-matching pack (`.kmm`) baked with `kansei-anim-bake`. None
//! ships with Kansei: the page loads `pack/locomotion.kmm` next to `index.html` (or `pack=<url>`)
//! and says how to make one when it is missing. See this example's README.
//!
//! A character pack (`pack/hero.kmm`, or `hero=<url>`), when there is one, is a second body for
//! the same animation: a mesh rigged to the same skeleton with its own proportions and textures,
//! the pose retargeted onto it (`animation::retarget`). It is shown by default; C switches
//! between it and the motion pack's own mesh (`char=hero` or `char=mannequin` to choose).
//!
//! A course of boxes stands around the start: low rails to hurdle, boxes to vault, blocks to mantle
//! onto, walls to climb, long narrow beams and stacked blocks, some turned. Space in front of one
//! traverses it (`motion_matching::traversal`): the kind from its shape, the clip from the pace,
//! the clip's root motion warped onto its ledge. With nothing to traverse ahead, Space jumps: a
//! jump clip for the pace up to its take-off, then a ballistic flight that lands wherever it
//! comes down, box tops included. Walking off a top falls and lands. The pack needs action clips
//! for that (`kansei-anim-bake`'s `actions`); without them Space does nothing.
//!
//! Controls: WASD or arrows move relative to the camera, Shift runs, Space jumps or traverses, Q
//! toggles strafing (face the camera's direction), mouse drag orbits and the wheel zooms. Gamepad:
//! left stick moves (tilt sets the pace), right stick orbits, A jumps or traverses, B or the right
//! trigger runs, the left bumper toggles strafing. Keys B, K, M, L and C toggle the trajectory
//! overlay and HUD, the skeleton, the mesh, foot locking and the character.
//!
//! A small lake lies east of the course: SPH water in a container shaped like the lake, which the
//! character wades into, its legs pushing the water (wakes, splashes and ripples). A water cannon
//! on its west bank pours more water in when the character stands by it and presses E (X on a
//! gamepad, or a click on its prompt), raising the lake's level; R (Y) by it drains the lake back.
//! A water mill in the lake turns its paddles through the water. The course, the lake and its
//! props are the lake example's world (`kansei_wasm_lake::World`, with its P panel's exports);
//! this crate is the character on it.
//!
//! URL parameters: `pack=<url>`, `hero=<url>` (`hero=none`: no character pack), `gait=0`
//! (search every clip whatever the gait, instead of idle + walk or idle + run by the pack's tags),
//! `taa=0`, `walk=<m/s>`, `run=<m/s>` (forward paces; sideways and backward scale with them),
//! `course=0` (no boxes), `lake=0` (no lake), `rest=0` (the lake's water never rests: always
//! stepped and drawn), `mill=0` (the mill stands still), `profile=1` (log the renderer's GPU/CPU
//! profile every 3 s), `debug=1` (allows `lake_regions()`, a GPU readback),
//! `at=<x>,<z>,<heading in degrees>` (where the character starts; `at=14,-1,90` at the lake),
//! `drive=1` (a fixed route instead of the player, for side-by-side captures; `demo`),
//! `circle=<radius>` (round a circle of that radius at a run instead, negative turning right),
//! `play=<pattern>` (the pack's clips whose names start with it, `*` any run, one after another),
//! `view=<degrees>` (the camera turned round the character from behind it).

//!
//! `www/room.html` is a second page on this module (`start_room`, the [`room`] module): the same
//! character in a 40 m room lit by ray-traced global illumination, with mirrors, a glass dragon on
//! an island in a pond, furniture to vault and climb, fog and dust. The character and its
//! controls are shared ([`character`], [`player`]).

pub mod character;
mod demo;
pub mod player;
pub mod room;
mod timing;

use std::cell::RefCell;
use std::rc::Rc;

use glam::Vec3 as GVec3;
use wasm_bindgen::prelude::*;

use kansei_core::animation::motion_matching::pack::{CharacterPack, MotionPack};
use kansei_core::animation::motion_matching::yaw_of;
use kansei_core::cameras::Camera;
use kansei_core::collision::CollisionWorld;
use kansei_core::controls::CameraControls;
use kansei_core::math::Vec3;
use kansei_core::objects::Scene;
use kansei_core::postprocessing::PostProcessingVolume;
use kansei_core::renderers::Renderer;
use kansei_wasm::{flag, param, param_or, set_text, Canvas};
use kansei_wasm_lake::{World, WorldInput, WorldOptions};

use character::{BodyLook, Character, SunLit};
use player::Player;

/// The page URL's parameter `name`, percent-decoded ([`kansei_wasm::param`]).
pub fn query_param(name: &str) -> Option<String> {
    param(name)
}

/// Fetch `url` as bytes; the error says what went wrong in words for the page.
pub async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
    kansei_wasm::fetch_bytes(url).await.map_err(|e| e.as_string().unwrap_or_else(|| format!("could not fetch {url}")))
}

struct State {
    renderer: Renderer,
    scene: Scene,
    /// The course, the lake and its props, and their collision.
    world: World,
    camera: Camera,
    controls: CameraControls,
    volume: PostProcessingVolume,
    player: Player,
    /// `profile=1`: when the profile was last logged (0: not profiling).
    profile_since: f64,
}

impl State {
    fn frame(&mut self) {
        let now = kansei_wasm::now();
        let dt = ((now - self.player.last_frame(now)) as f32).clamp(1e-4, 1.0 / 15.0);
        let State { renderer, scene, world, controls, player, .. } = self;
        let out = player.update(now, dt, scene, renderer, controls, &world.collision);
        // the cannon's trigger: E, the gamepad's X (a click on the prompt adds to it in the world);
        // R (Y) drains the lake
        let fire_pressed = out.keys.iter().any(|k| k == "e");
        let drain = out.keys.iter().any(|k| k == "r");
        self.controls.update(&mut self.camera, dt);
        if self.world.lake.is_some() {
            let view_proj = self.camera.view_projection().to_glam();
            let State { world, scene, volume, renderer, .. } = &mut *self;
            world.update(WorldInput { legs: &out.legs, landing: out.landing, at: out.at, fire: (fire_pressed, out.e_held), drain }, dt, scene, volume, renderer, view_proj);
        }

        // HUD, a few times a second
        if self.player.frame.is_multiple_of(10) {
            if let Some(text) = self.player.hud(&self.world.status("east of the course (at=14,-1,90), "), " · E / X fire the cannon (by it)") {
                set_text("hud", &text);
            }
        }

        self.renderer.render_with_postprocessing(&mut self.scene, &mut self.camera, &mut self.volume);
        if self.profile_since > 0.0 && now - self.profile_since > 3.0 {
            self.profile_since = now;
            log::info!("profile ({:.1} ms/frame)\n{}", 1000.0 / self.player.fps, self.renderer.take_profile().report());
        }
    }
}

/// The motion pack (`pack=<url>`, else `pack/locomotion.kmm`) and the character pack (`hero=<url>`,
/// else `pack/hero.kmm`; `hero=none`: none) loaded by `load`, as a character drawn by `look`
/// (`char=hero|mannequin` picks the body shown, `at=<x>,<z>,<degrees>` where it starts, on
/// whatever `collision` has there). Without a pack, the HUD says how to make one (`page`: where
/// to put it).
pub(crate) async fn load_character<L, F>(renderer: &Renderer, scene: &mut Scene, collision: &CollisionWorld, look: &dyn BodyLook, load: &L, page: &str) -> Option<Character>
where
    L: Fn(String) -> F,
    F: std::future::Future<Output = Result<Vec<u8>, String>>,
{
    // the character, from a pack outside the repository
    let url = param("pack").unwrap_or_else(|| "pack/locomotion.kmm".to_string());
    set_text("hud", &format!("Loading motion pack {url} …"));
    let gait = flag("gait", true);
    let motion = load(url.clone()).await.and_then(|bytes| MotionPack::from_bytes(&bytes));
    // a second body, optional
    let hero_url = param("hero").unwrap_or_else(|| "pack/hero.kmm".to_string());
    let hero = match &motion {
        Ok(_) if hero_url == "none" => {
            log::info!("no character pack: none asked for (hero=none)");
            None
        }
        Ok(_) => match load(hero_url).await.and_then(|bytes| CharacterPack::from_bytes(&bytes)) {
            Ok(h) => Some(h),
            Err(e) => {
                log::info!("no character pack: {e}");
                None
            }
        },
        Err(_) => None,
    };
    match motion {
        Ok(pack) => match Character::new(renderer, scene, pack, hero, gait, look) {
            Ok(mut c) => {
                let wanted = match param("char").as_deref() {
                    Some("mannequin") => 0,
                    _ => c.bodies.len() - 1,
                };
                c.show(scene, wanted);
                if let Some(at) = param("at") {
                    let v: Vec<f32> = at.split(',').filter_map(|x| x.parse().ok()).collect();
                    if v.len() >= 2 {
                        // on whatever is there (a box top)
                        let y = collision.ground_height(GVec3::new(v[0], 0.0, v[1]), 50.0, 50.0, u32::MAX).unwrap_or(0.0);
                        c.controller.matcher.teleport(GVec3::new(v[0], y, v[1]), v.get(2).copied().unwrap_or(0.0).to_radians());
                    }
                }
                Some(c)
            }
            Err(e) => {
                set_text("hud", &format!("The motion pack {url} can't be used: {e}."));
                None
            }
        },
        Err(e) => {
            log::warn!("no motion pack: {e}");
            set_text("hud", &format!(
                "No motion pack ({e}).\n\nThis example animates a character from a motion-matching pack (.kmm), and Kansei ships none:\n\
                 bake one from your own glTF clips with kansei-anim-bake, then put it at\n\
                 rust/kansei-wasm/demos/motion-matching/www/pack/locomotion.kmm\n\
                 (or open this page with ?pack=<url>). See this example's README.{page}"
            ));
            None
        }
    }
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    start_with_loader(canvas_id, |url: String| async move { fetch_bytes(&url).await }).await
}

/// `start`, with the packs' bytes from `load` instead of a plain fetch: it gets each pack's URL
/// (`pack/locomotion.kmm`, `pack/hero.kmm`, or `pack=`/`hero=`; never asked for the character
/// pack with `hero=none`) and returns the `.kmm` bytes, or why there are none. For an app that
/// stores its packs another way, e.g. encrypted.
pub async fn start_with_loader<L, F>(canvas_id: &str, load: L) -> Result<(), JsValue>
where
    L: Fn(String) -> F,
    F: std::future::Future<Output = Result<Vec<u8>, String>>,
{
    let canvas = Canvas::find(canvas_id)?;
    let renderer = kansei_wasm_lake::renderer(&canvas).await;

    let mut scene = Scene::new();
    // the course, the lake and its props (`course=0`, `lake=0`, `rest=0`, `mill=0`, `taa=0`)
    let mut world = World::new(&renderer, &mut scene, &WorldOptions::from_url());
    let character = load_character(&renderer, &mut scene, &world.collision, &SunLit, &load, "").await;

    let volume = world.post_processing(&renderer, &mut scene);
    let mut camera = Camera::new(45.0, 0.1, 1200.0, canvas.aspect());
    camera.update_projection_matrix();
    let start = character.as_ref().map(|c| c.controller.matcher.character()).unwrap_or_default();
    let mut controls = CameraControls::from_canvas(canvas.element(), Vec3::new(start.translation.x, 0.9, start.translation.z), 4.5);
    controls.set_elevation(0.25);
    // behind the character, or turned round it by `view=<degrees>` (90: its left side)
    let view = param_or("view", 0.0f32).to_radians();
    controls.set_azimuth(std::f32::consts::PI + yaw_of(start.rotation) + view);

    log::info!("Kansei — Motion Matching (WASM) ready: character {}", character.is_some());
    let state = Rc::new(RefCell::new(State { renderer, world, scene, camera, controls, volume, player: Player::new(character), profile_since: 0.0 }));
    if flag("profile", false) {
        let mut s = state.borrow_mut();
        s.renderer.set_profiling(true);
        s.profile_since = kansei_wasm::now();
    }
    set_page(state.clone());
    // the lake's panel and the cannon's prompt act on this world
    kansei_wasm_lake::register(state.clone());
    kansei_wasm::run(&canvas, move |frame| {
        let mut s = state.borrow_mut();
        let State { renderer, camera, .. } = &mut *s;
        frame.resize(renderer, camera);
        s.frame();
    });
    Ok(())
}

impl kansei_wasm_lake::Host for State {
    fn parts(&mut self) -> (&mut World, &mut PostProcessingVolume, &Renderer) {
        (&mut self.world, &mut self.volume, &self.renderer)
    }
}

/// A page of this module, for the exports that act on its player between frames.
pub(crate) trait Page {
    fn player(&mut self) -> &mut Player;
}

impl Page for State {
    fn player(&mut self) -> &mut Player {
        &mut self.player
    }
}

thread_local! {
    /// The page's state, for the exports the clip tools call between frames.
    static STATE: RefCell<Option<Rc<RefCell<dyn Page>>>> = const { RefCell::new(None) };
}

/// Make `page` the one the exports act on.
pub(crate) fn set_page(page: Rc<RefCell<dyn Page>>) {
    STATE.with(|s| *s.borrow_mut() = Some(page));
}

/// Run `f` on the page's player, once it has started.
fn with_player<R>(f: impl FnOnce(&mut Player) -> R) -> Option<R> {
    STATE.with(|s| {
        let state = s.borrow().clone()?;
        let mut state = state.borrow_mut();
        Some(f(state.player()))
    })
}

/// Start the `drive=1` route (or the `play=` clips) over, the character back where it started:
/// to line up recordings of different packs.
#[wasm_bindgen]
pub fn drive_restart() {
    with_player(Player::drive_restart);
}

/// The motion pack's clip names, in its order (empty before it has loaded): for a page's clip
/// browser.
#[wasm_bindgen]
pub fn clip_names() -> Vec<String> {
    with_player(|p| p.character.as_ref().map(|c| c.db.clips.iter().map(|c| c.name.clone()).collect())).flatten().unwrap_or_default()
}

/// Play the clips whose names start with `pattern` one after another, as `play=` does, from where
/// the character stands now; `""` hands it back to the player (the clip playing finishes first).
/// Returns how many clips match.
#[wasm_bindgen]
pub fn play_clips(pattern: &str) -> usize {
    with_player(|p| p.play_clips(pattern)).unwrap_or(0)
}

/// Drive the `drive=1` route from where the character stands now, or hand it back to the player.
#[wasm_bindgen]
pub fn set_drive(on: bool) {
    with_player(|p| p.set_drive(on));
}

/// The character's update times (ms per frame) over the last frames, as JSON (`timing::Window`).
#[wasm_bindgen]
pub fn motion_timings() -> String {
    with_player(|p| p.update_ms.json()).unwrap_or_default()
}

/// Milliseconds per `Database::search` over the pack's own frames as queries (every `step`th,
/// nudged), the default filter: the same queries as the TS page's `benchSearch`.
#[wasm_bindgen]
pub fn bench_search(step: usize) -> f64 {
    with_player(|p| p.character.as_ref().map(|c| timing::bench_search(&c.db, step))).flatten().unwrap_or(-1.0)
}

/// Milliseconds to sample a pose between two frames, run its forward kinematics and fill the
/// first body's bone palette, `count` times over the pack: the same as the TS page's `benchPose`.
#[wasm_bindgen]
pub fn bench_pose(count: usize) -> f64 {
    with_player(|p| {
        p.character.as_mut().map(|c| {
            let body = &mut c.bodies[0];
            let (mesh, palette) = (&body.mesh, &mut body.palette);
            let skeleton = body.display.as_ref().map_or(&c.db.skeleton, |d| &d.0).clone();
            let retarget = body.display.as_ref().map(|d| d.1.clone());
            timing::bench_pose(&c.db, count, |pose, model| {
                match &retarget {
                    Some(r) => {
                        let mut out = kansei_core::animation::Pose { local: Vec::new() };
                        r.apply(pose, &mut out);
                        out.to_model(&skeleton, model);
                    }
                    None => pose.to_model(&skeleton, model),
                }
                palette.update(mesh, model);
            })
        })
    })
    .flatten()
    .unwrap_or(-1.0)
}
