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

mod demo;
mod timing;

use std::cell::RefCell;
use std::rc::Rc;

use glam::{Mat4, Quat, Vec3 as GVec3};
use wasm_bindgen::prelude::*;

use kansei_core::animation::motion_matching::pack::{CharacterPack, MotionPack};
use kansei_core::animation::retarget::Retarget;
use kansei_core::animation::motion_matching::traversal::{CharacterController, CharacterState};
use kansei_core::animation::motion_matching::{yaw_of, Database, MotionInput, MotionMatcher, MotionMatchingSettings, ACTION_TAG};
use kansei_core::animation::{skinned_lit_material, skinned_lit_textured_material, BonePalette, Skeleton, SkinTextures, SkinnedLitParams, SkinnedMesh, PALETTE_BINDING};
use kansei_core::buffers::Texture;
use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::debug::{segment, DebugBoxes};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::PostProcessingVolume;
use kansei_core::renderers::Renderer;
use kansei_wasm::{flag, param, param_or, set_text, Canvas, Gamepad, Keys};
use kansei_wasm_lake::{World, WorldInput, WorldOptions, SKY, SUN, SUN_DIR};

/// Paces of the walk and run loops (m/s) forward, sideways and backward: strafing moves at the
/// pace of the loop for its direction relative to the facing. `walk=` and `run=` scale them.
const WALK: [f32; 3] = [2.0, 1.8, 1.5];
const RUN: [f32; 3] = [5.0, 3.5, 3.0];

/// The pace for moving along `direction` (unit, local to the facing: x sideways, y forward) on
/// an ellipse through the forward, sideways and backward paces.
fn pace(paces: [f32; 3], direction: [f32; 2]) -> f32 {
    let along = if direction[1] >= 0.0 { paces[0] } else { paces[2] };
    1.0 / ((direction[1] / along).powi(2) + (direction[0] / paces[1]).powi(2)).sqrt().max(1e-6)
}

/// The page URL's parameter `name`, percent-decoded ([`kansei_wasm::param`]).
pub fn query_param(name: &str) -> Option<String> {
    param(name)
}

/// Fetch `url` as bytes; the error says what went wrong in words for the page.
pub async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
    kansei_wasm::fetch_bytes(url).await.map_err(|e| e.as_string().unwrap_or_else(|| format!("could not fetch {url}")))
}

/// A mesh the character can be shown as: the motion pack's own, on the database's skeleton, or a
/// character pack's, the pose retargeted onto its skeleton.
struct Body {
    name: &'static str,
    mesh: SkinnedMesh,
    palette: BonePalette,
    index: usize,
    display: Option<(Skeleton, Retarget)>,
}

/// A character pack as a body: its textured mesh, hidden until shown.
fn hero_body(scene: &mut Scene, pack: CharacterPack, db: &Database) -> Result<Body, String> {
    let retarget = Retarget::new(&db.skeleton, &pack.skeleton, &Retarget::UNREAL_KEEP);
    let texture = |name: &str, srgb: bool, fallback: [u8; 4]| -> Result<Texture, String> {
        let image = match pack.image(name) {
            Some(i) => image::load_from_memory(&i.bytes).map_err(|e| format!("texture {name}: {e}"))?.to_rgba8(),
            None => image::RgbaImage::from_pixel(1, 1, image::Rgba(fallback)),
        };
        Ok(Texture::from_image(name, &image, srgb))
    };
    let textures = SkinTextures { base_color: texture("base_color", true, [200, 200, 200, 255])?, normal: texture("normal", false, [128, 128, 255, 255])?, orm: texture("orm", false, [255, 160, 0, 255])? };
    let first = pack.meshes.into_iter().next().ok_or("the character pack has no mesh")?;
    let mesh = first.mesh;
    let mut palette = BonePalette::new(mesh.skin_joints.len());
    palette.update(&mesh, &pack.skeleton.rest_model());
    let params = SkinnedLitParams { base_color: first.color, sun_direction: [SUN_DIR[0], SUN_DIR[1], SUN_DIR[2], 0.0], sun: [SUN[0], SUN[1], SUN[2], 0.0], sky: [SKY[0], SKY[1], SKY[2], 0.0] };
    let material = skinned_lit_textured_material("Hero", params, &mesh, &palette, textures);
    let mut r = Renderable::new(mesh.geometry(), material);
    r.dynamic = true;
    r.visible = false;
    let index = scene.add(SceneNode::Renderable(r));
    log::info!("character pack: {} joints, {} vertices, {} triangles", pack.skeleton.len(), mesh.vertices.len(), mesh.indices.len() / 3);
    Ok(Body { name: "hero", mesh, palette, index, display: Some((pack.skeleton, retarget)) })
}

/// The character: its database, matcher, bodies, and the debug markers.
struct Character {
    db: Database,
    controller: CharacterController,
    bodies: Vec<Body>,
    /// The last obstacle found, marked at its ledge.
    ledge: DebugBoxes,
    showing: usize,
    bones: DebugBoxes,
    trajectory: DebugBoxes,
    /// Tag bits the search may use while walking and while running (all when the pack has no
    /// gait tags or `gait=0`).
    walk_tags: u32,
    run_tags: u32,
}

impl Character {
    fn new(renderer: &Renderer, scene: &mut Scene, pack: MotionPack, hero: Option<CharacterPack>, gait: bool) -> Result<Self, String> {
        let tags: Vec<String> = pack.meta("tags").unwrap_or("").split(',').map(str::to_string).collect();
        let MotionPack { database: db, meshes, actions, .. } = pack;
        // the pack's own mesh, when it has one (a pack may ship without, for a character pack's body)
        let mut bodies = Vec::new();
        if let Some(first) = meshes.into_iter().next() {
            let mesh = first.mesh;
            let mut palette = BonePalette::new(mesh.skin_joints.len());
            palette.update(&mesh, &db.skeleton.rest_model());
            let params = SkinnedLitParams {
                base_color: first.color,
                sun_direction: [SUN_DIR[0], SUN_DIR[1], SUN_DIR[2], 0.0],
                sun: [SUN[0], SUN[1], SUN[2], 0.0],
                sky: [SKY[0], SKY[1], SKY[2], 0.0],
            };
            let mut r = Renderable::new(mesh.geometry(), skinned_lit_material("Character", params, &mesh, &palette));
            r.dynamic = true;
            let index = scene.add(SceneNode::Renderable(r));
            bodies.push(Body { name: "the pack's mesh", mesh, palette, index, display: None });
        }
        if let Some(pack) = hero {
            match hero_body(scene, pack, &db) {
                Ok(b) => bodies.push(b),
                Err(e) => log::warn!("character pack left out: {e}"),
            }
        }
        if bodies.is_empty() {
            return Err("the pack has no mesh, and there is no character pack to show it on".into());
        }
        let joints = bodies.iter().map(|b| b.display.as_ref().map_or(db.joint_count(), |d| d.0.len())).max().unwrap_or(0);
        let bones = DebugBoxes::new(renderer, scene, "Bones", joints, [30000.0, 20000.0, 4000.0], true);
        bones.set_visible(scene, false);
        // the simulation now and its 3 predicted samples, and each foot's target
        let trajectory = DebugBoxes::new(renderer, scene, "Trajectory", 6, [300.0, 1600.0, 3000.0], false);
        let bit = |name: &str| tags.iter().position(|t| t == name).map_or(0, |b| 1u32 << b);
        let (idle, walk, run) = (bit("idle"), bit("walk"), bit("run"));
        let (walk_tags, run_tags) = if gait && walk != 0 && run != 0 { (idle | walk, idle | run) } else { (!ACTION_TAG, !ACTION_TAG) };
        let matcher = MotionMatcher::new(&db, MotionMatchingSettings::default(), GVec3::ZERO, 0.0);
        log::info!("{} action clips", actions.len());
        let controller = CharacterController::new(matcher, actions);
        let ledge = DebugBoxes::new(renderer, scene, "Ledge", 2, [3000.0, 400.0, 200.0], true);
        log::info!("motion pack: {} clips, {} frames, {} joints", db.clips.len(), db.frame_count(), db.joint_count());
        Ok(Self { db, controller, bodies, ledge, showing: 0, bones, trajectory, walk_tags, run_tags })
    }

    /// Show body `which` (its mesh, the pose on its skeleton).
    fn show(&mut self, scene: &mut Scene, which: usize) {
        self.showing = which % self.bodies.len();
        for (i, b) in self.bodies.iter_mut().enumerate() {
            if let Some(r) = scene.get_renderable_mut(b.index) {
                r.visible = i == self.showing;
                r.reset_motion();
            }
            b.palette.reset_motion();
        }
        self.controller.matcher.set_display(&self.db, self.bodies[self.showing].display.clone());
    }

    /// The output skeleton's joint named as database joint `j`.
    fn joint(&self, j: usize) -> usize {
        self.controller.matcher.output_skeleton(&self.db).find(&self.db.skeleton.names[j]).unwrap_or(0)
    }

    /// The legs as capsules (world ends, radius): each thigh, shin and foot (hip to knee to ankle
    /// to toe, up the foot's parents and down to its first child), and the hips.
    fn leg_capsules(&self) -> Vec<(GVec3, GVec3, f32)> {
        let matcher = &self.controller.matcher;
        let (character, model, skeleton) = (matcher.character(), matcher.model(), matcher.output_skeleton(&self.db));
        let at = |j: usize| character.transform_point(model[j].translation);
        let mut legs = Vec::new();
        for side in 0..2 {
            let foot = self.joint(self.db.roles.feet[side]);
            let Some(knee) = skeleton.parents[foot] else { continue };
            let Some(hip) = skeleton.parents[knee] else { continue };
            let toe = skeleton.parents.iter().position(|p| *p == Some(foot)).map_or(at(foot) + character.rotation * GVec3::Z * 0.15, at);
            legs.extend([(at(hip), at(knee), 0.085), (at(knee), at(foot), 0.06), (at(foot), toe, 0.05)]);
        }
        let hips = at(self.joint(self.db.roles.hips));
        legs.push((hips, hips, 0.14));
        legs
    }
}

struct State {
    renderer: Renderer,
    scene: Scene,
    /// The course, the lake and its props, and their collision.
    world: World,
    camera: Camera,
    controls: CameraControls,
    volume: PostProcessingVolume,
    character: Option<Character>,
    keys: Keys,
    pad: Gamepad,
    last: f64,
    frame: u32,
    strafe: bool,
    overlay: bool,
    /// Walk and run paces: forward, sideways, backward.
    speeds: ([f32; 3], [f32; 3]),
    fps: f32,
    searches: u32,
    switches: u32,
    counted_since: f64,
    rates: (f32, f32),
    /// Whether the character was in the air last frame, and the fastest it has fallen since
    /// (m/s); its height last frame.
    air: (bool, f32),
    last_y: f32,
    /// `profile=1`: when the profile was last logged (0: not profiling).
    profile_since: f64,
    /// `drive=1`: the scripted route; `play=<pattern>`: the clips played in turn.
    drive: Option<demo::Drive>,
    player: Option<demo::ClipPlayer>,
    /// The character's update (search, pose, IK) per frame, in ms: the last `timing::WINDOW` frames.
    update_ms: timing::Window,
}

impl State {
    fn frame(&mut self) {
        let now = kansei_wasm::now();
        let dt = ((now - self.last) as f32).clamp(1e-4, 1.0 / 15.0);
        self.last = now;
        self.frame += 1;
        self.fps = self.fps * 0.95 + 0.05 / dt;

        // input: keyboard, then the gamepad on top
        let keys = &self.keys;
        let mut stick = [keys.axis(&["a", "arrowleft"], &["d", "arrowright"]), keys.axis(&["s", "arrowdown"], &["w", "arrowup"])];
        let mut run = keys.held("shift");
        let mut toggles = keys.take_pressed();
        // the cannon's trigger: E, the gamepad's X (a click on the prompt adds to it in the world)
        let (mut fire_pressed, mut fire_held) = (false, keys.held("e"));
        let length = (stick[0] * stick[0] + stick[1] * stick[1]).sqrt();
        if length > 1.0 {
            stick = [stick[0] / length, stick[1] / length];
        }
        let pad = &mut self.pad;
        if pad.poll() {
            let [x, y] = pad.left_stick();
            if [x, y] != [0.0, 0.0] {
                stick = [x, -y];
            }
            let [x, y] = pad.right_stick();
            self.controls.rotate(-x * 2.5 * dt, y * 1.5 * dt);
            run |= pad.held(Gamepad::B) || pad.value(Gamepad::RIGHT_TRIGGER) > 0.3;
            for (button, key) in [(Gamepad::A, " "), (Gamepad::LEFT_BUMPER, "q"), (Gamepad::X, "e"), (Gamepad::Y, "r")] {
                if pad.pressed(button) {
                    toggles.push(key.into());
                }
            }
            fire_held |= pad.held(Gamepad::X);
        }
        let (mut traverse, mut drain) = (false, false);
        for key in toggles {
            match key.as_str() {
                " " => traverse = true,
                "e" => fire_pressed = true,
                "r" => drain = true,
                "q" => self.strafe = !self.strafe,
                "b" => self.overlay = !self.overlay,
                "c" => {
                    if let Some(c) = &mut self.character {
                        let next = c.showing + 1;
                        c.show(&mut self.scene, next);
                    }
                }
                "k" | "m" | "l" => {
                    if let Some(c) = &mut self.character {
                        match key.as_str() {
                            "k" => {
                                let visible = c.bones.visible(&self.scene);
                                c.bones.set_visible(&mut self.scene, !visible);
                            }
                            "m" => {
                                if let Some(r) = self.scene.get_renderable_mut(c.bodies[c.showing].index) {
                                    r.visible = !r.visible;
                                }
                            }
                            _ => c.controller.matcher.settings.foot_lock = !c.controller.matcher.settings.foot_lock,
                        }
                    }
                }
                _ => {}
            }
        }

        // the camera's heading on the ground: the stick moves relative to it
        let azimuth = self.controls.azimuth();
        let forward = GVec3::new(-azimuth.sin(), 0.0, -azimuth.cos());
        let right = GVec3::new(azimuth.cos(), 0.0, -azimuth.sin());
        let step = self.drive.as_ref().map(|d| d.step(now));
        if let Some(step) = &step {
            run = step.run;
            self.strafe = step.facing.is_some();
        }
        if let Some(c) = &mut self.character {
            let paces = if run { self.speeds.1 } else { self.speeds.0 };
            // facing the way it moves, the character walks its loops forward; strafing, the pace
            // follows the direction
            let tilt = (stick[0] * stick[0] + stick[1] * stick[1]).sqrt();
            let speed = if self.strafe && tilt > 1e-3 { pace(paces, [stick[0] / tilt, stick[1] / tilt]) } else { paces[0] };
            let mut velocity = (forward * stick[1] + right * stick[0]) * speed;
            let mut facing = self.strafe.then(|| forward.x.atan2(forward.z));
            if let Some(step) = &step {
                // the route's direction, at the pace for it relative to the facing held
                velocity = step.direction.map_or(GVec3::ZERO, |d| {
                    let f = step.facing.unwrap_or(d.x.atan2(d.z));
                    let (front, side) = (GVec3::new(f.sin(), 0.0, f.cos()), GVec3::new(-f.cos(), 0.0, f.sin()));
                    d * pace(paces, [d.dot(side), d.dot(front)])
                });
                facing = step.facing;
            }
            c.controller.matcher.settings.filter.tags = if run { c.run_tags } else { c.walk_tags };
            if let Some(player) = &mut self.player {
                // clips as they are, one after another, from the start point
                if let Some(action) = player.due(&c.db, now) {
                    c.controller.matcher.teleport(player.home.0, player.home.1);
                    c.controller.matcher.start_action(&c.db, action);
                }
                c.controller.matcher.update(&c.db, &MotionInput { velocity: GVec3::ZERO, facing: None }, dt);
            } else {
                let t0 = kansei_wasm::now();
                c.controller.update(&c.db, &self.world.collision, &MotionInput { velocity, facing }, dt);
                self.update_ms.push(((kansei_wasm::now() - t0) * 1000.0) as f32);
            }
            if traverse {
                // an obstacle ahead: traverse it (pressed a little early, once in reach); else jump
                let _ = c.controller.request_traverse_or_jump(&c.db, &self.world.collision, 1.0);
            }
            // the obstacle last looked at: its ledge, and a post down to the floor
            c.ledge.matrices.fill(Mat4::ZERO);
            if let (true, Some(o)) = (self.overlay, c.controller.last_obstacle) {
                let side = o.normal.cross(GVec3::Y);
                c.ledge.matrices[0] = segment(o.ledge - side * 0.4, o.ledge + side * 0.4, 0.04);
                c.ledge.matrices[1] = segment(o.ledge, o.ledge - GVec3::Y * o.height, 0.02);
            }
            c.ledge.upload(&self.renderer);
            let search = c.controller.matcher.last_search();
            self.searches += search.searched as u32;
            self.switches += search.switched as u32;

            let character = c.controller.matcher.character();
            let body = &mut c.bodies[c.showing];
            if let Some(r) = self.scene.get_renderable_mut(body.index) {
                r.object.set_position(character.translation.x, character.translation.y, character.translation.z);
                r.object.rotation.y = yaw_of(character.rotation);
                body.palette.update(&body.mesh, c.controller.matcher.model());
                if let Some(buffer) = r.material.bindable_buffer(PALETTE_BINDING) {
                    body.palette.upload(self.renderer.queue(), &buffer);
                }
            }
            // follow the character's hips
            let hips = character.transform_point(c.controller.matcher.model()[c.joint(c.db.roles.hips)].translation);
            let target = GVec3::new(character.translation.x, hips.y * 0.9, character.translation.z);
            self.controls.follow(Vec3::new(target.x, target.y, target.z), dt, 8.0);

            if self.overlay {
                // the simulation and its predicted samples: flat boxes pointing where they face
                let s = c.controller.matcher.simulation();
                let mut k = 0;
                for (p, q, size) in std::iter::once((s.position, s.rotation, 0.16)).chain(c.controller.matcher.trajectory().iter().map(|t| (t.translation, t.rotation, 0.1))) {
                    c.trajectory.matrices[k] = Mat4::from_scale_rotation_translation(GVec3::new(size, 0.02, size * 2.0), q, p + GVec3::Y * 0.01);
                    k += 1;
                }
                // each foot's target, raised when not planted
                let model = c.controller.matcher.model();
                for (side, locked) in c.controller.matcher.feet_locked().iter().enumerate() {
                    let p = character.transform_point(model[c.joint(c.db.roles.feet[side])].translation);
                    let size = if *locked { 0.09 } else { 0.04 };
                    c.trajectory.matrices[k] = Mat4::from_scale_rotation_translation(GVec3::splat(size), Quat::IDENTITY, GVec3::new(p.x, character.translation.y + 0.02, p.z));
                    k += 1;
                }
            } else {
                c.trajectory.matrices.fill(Mat4::ZERO);
            }
            c.trajectory.upload(&self.renderer);
            if c.bones.visible(&self.scene) {
                let model = c.controller.matcher.model();
                let root = c.joint(c.db.roles.root);
                let parents = c.controller.matcher.output_skeleton(&c.db).parents.clone();
                c.bones.matrices.fill(Mat4::ZERO);
                for (j, parent) in parents.iter().enumerate() {
                    if let Some(p) = parent.filter(|p| *p != root) {
                        c.bones.matrices[j] = segment(character.transform_point(model[p].translation), character.transform_point(model[j].translation), 0.015);
                    }
                }
                c.bones.upload(&self.renderer);
            }
        }
        self.controls.update(&mut self.camera, dt);
        if self.world.lake.is_some() {
            let legs = self.character.as_ref().map(Character::leg_capsules).unwrap_or_default();
            // a landing: back on its feet after being in the air, at the fastest it came down
            let mut landing = None;
            if let Some(c) = &self.character {
                let y = c.controller.matcher.character().translation.y;
                let airborne = matches!(c.controller.state(), CharacterState::Jumping | CharacterState::Falling(_));
                let (was, fall) = &mut self.air;
                if airborne {
                    *fall = fall.max((self.last_y - y) / dt);
                } else if *was {
                    landing = Some((c.controller.matcher.character().translation, *fall));
                    *fall = 0.0;
                }
                *was = airborne;
                self.last_y = y;
            }
            // the props: the mill turns, the cannon fires when the character stands by it
            let at = self.character.as_ref().map(|c| c.controller.matcher.character().translation);
            let view_proj = self.camera.view_projection().to_glam();
            let State { world, scene, volume, renderer, .. } = &mut *self;
            world.update(WorldInput { legs: &legs, landing, at, fire: (fire_pressed, fire_held), drain }, dt, scene, volume, renderer, view_proj);
        }

        // HUD, a few times a second
        if now - self.counted_since > 1.0 {
            let span = (now - self.counted_since) as f32;
            self.rates = (self.searches as f32 / span, self.switches as f32 / span);
            self.searches = 0;
            self.switches = 0;
            self.counted_since = now;
        }
        if self.frame % 10 == 0 {
            match &self.character {
                Some(c) if self.overlay => {
                    let (clip, frame) = c.controller.matcher.playing();
                    let info = &c.db.clips[clip];
                    let s = c.controller.matcher.last_search();
                    let feet = c.controller.matcher.feet_locked();
                    let speed = c.controller.matcher.simulation().velocity.length();
                    set_text("hud", &format!(
                        "{:.0} fps   {} {}{}\nclip   {}\nframe  {:.0} / {}{}\nsearch {:.0}/s, switch {:.1}/s, cost {:.3}\nupdate {}\nfeet   {} {}  (lock {})\n{}\n\n{}\n\nWASD / left stick move · Shift / B run · Space / A jump, traverse · Q / LB strafe\ndrag / right stick orbit · B overlay · K skeleton · M mesh · L foot lock · E / X fire the cannon (by it)",
                        self.fps,
                        if run { "run" } else { "walk" },
                        format_args!("{speed:.1} m/s"),
                        if self.strafe { "  strafe" } else { "" },
                        info.name,
                        frame,
                        info.playable(),
                        if info.looping { " (loop)" } else { "" },
                        self.rates.0,
                        self.rates.1,
                        s.cost,
                        self.update_ms.summary(),
                        if feet[0] { "L planted" } else { "L free" },
                        if feet[1] { "R planted" } else { "R free" },
                        if c.controller.matcher.settings.foot_lock { "on" } else { "off" },
                        format_args!(
                            "state  {}{}",
                            match c.controller.state() {
                                CharacterState::Grounded => "on the ground".to_string(),
                                CharacterState::Traversing(k) => ["hurdling", "vaulting", "mantling", "climbing", "falling", "landing", "jumping"][k as usize].to_string(),
                                CharacterState::Jumping => "jumping".to_string(),
                                CharacterState::Falling(t) => format!("in the air {t:.1} s"),
                                CharacterState::Landing => "landing".to_string(),
                            },
                            match c.controller.last_result {
                                Some(Ok(k)) => format!("   last Space: {}", k.name()),
                                Some(Err(e)) => format!("   last Space: {}", e.describe()),
                                None => String::new(),
                            }
                        ),
                        format_args!(
                            "{}{}",
                            self.world.status("east of the course (at=14,-1,90), "),
                            if c.bodies.len() > 1 { format!("character: {} (C to switch)", c.bodies[c.showing].name) } else { String::new() },
                        ),
                    ));
                }
                Some(_) => set_text("hud", ""),
                None => {}
            }
        }

        self.renderer.render_with_postprocessing(&mut self.scene, &mut self.camera, &mut self.volume);
        if self.profile_since > 0.0 && now - self.profile_since > 3.0 {
            self.profile_since = now;
            log::info!("profile ({:.1} ms/frame)\n{}", 1000.0 / self.fps, self.renderer.take_profile().report());
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
    let character = match motion {
        Ok(pack) => match Character::new(&renderer, &mut scene, pack, hero, gait) {
            Ok(mut c) => {
                let wanted = match param("char").as_deref() {
                    Some("mannequin") => 0,
                    _ => c.bodies.len() - 1,
                };
                c.show(&mut scene, wanted);
                if let Some(at) = param("at") {
                    let v: Vec<f32> = at.split(',').filter_map(|x| x.parse().ok()).collect();
                    if v.len() >= 2 {
                        // on whatever is there (a box top)
                        let y = world.collision.ground_height(GVec3::new(v[0], 0.0, v[1]), 50.0, 50.0, u32::MAX).unwrap_or(0.0);
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
                 rust/kansei-wasm/examples/motion-matching/www/pack/locomotion.kmm\n\
                 (or open this page with ?pack=<url>). See this example's README."
            ));
            None
        }
    };

    let volume = world.post_processing(&renderer, &mut scene);
    let mut camera = Camera::new(45.0, 0.1, 1200.0, canvas.aspect());
    camera.update_projection_matrix();
    let start = character.as_ref().map(|c| c.controller.matcher.character()).unwrap_or_default();
    let mut controls = CameraControls::from_canvas(canvas.element(), Vec3::new(start.translation.x, 0.9, start.translation.z), 4.5);
    controls.set_elevation(0.25);
    // behind the character, or turned round it by `view=<degrees>` (90: its left side)
    let view = param_or("view", 0.0f32).to_radians();
    controls.set_azimuth(std::f32::consts::PI + yaw_of(start.rotation) + view);

    let scaled = |paces: [f32; 3], name: &str| param(name).and_then(|v| v.parse::<f32>().ok()).map_or(paces, |forward| paces.map(|p| p * forward / paces[0]));
    let speeds = (scaled(WALK, "walk"), scaled(RUN, "run"));
    log::info!("Kansei — Motion Matching (WASM) ready: character {}", character.is_some());
    let state = Rc::new(RefCell::new(State {
        renderer,
        world,
        scene,
        camera,
        controls,
        volume,
        character,
        keys: Keys::listen(),
        pad: Gamepad::new(),
        last: kansei_wasm::now(),
        frame: 0,
        strafe: false,
        overlay: true,
        speeds,
        fps: 60.0,
        searches: 0,
        switches: 0,
        counted_since: kansei_wasm::now(),
        rates: (0.0, 0.0),
        profile_since: 0.0,
        air: (false, 0.0),
        last_y: 0.0,
        drive: None,
        player: None,
        update_ms: timing::Window::default(),
    }));
    if flag("profile", false) {
        let mut s = state.borrow_mut();
        s.renderer.set_profiling(true);
        s.profile_since = kansei_wasm::now();
    }
    {
        let mut s = state.borrow_mut();
        let home = s.character.as_ref().map(|c| {
            let at = c.controller.matcher.character();
            (at.translation, yaw_of(at.rotation))
        });
        if let Some(home) = home {
            // circle=<radius>: round a circle at the run pace (negative radius: turning right)
            let circle = param("circle").and_then(|r| r.parse::<f32>().ok()).filter(|r| r.abs() > 0.1).map(|r| s.speeds.1[0] / r);
            if flag("drive", false) || circle.is_some() {
                s.drive = Some(demo::Drive { start: kansei_wasm::now(), home, circle });
            }
            if let Some(prefix) = param("play") {
                let player = s.character.as_ref().map(|c| demo::ClipPlayer::new(&c.db, &prefix, home));
                s.player = player;
            }
        }
    }
    STATE.with(|s| *s.borrow_mut() = Some(state.clone()));
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

thread_local! {
    /// The page's state, for the exports the clip tools call between frames.
    static STATE: RefCell<Option<Rc<RefCell<State>>>> = const { RefCell::new(None) };
}

/// Run `f` on the page's state, once it has started.
fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    STATE.with(|s| {
        let state = s.borrow().clone()?;
        let mut state = state.borrow_mut();
        Some(f(&mut state))
    })
}

/// Start the `drive=1` route (or the `play=` clips) over, the character back where it started:
/// to line up recordings of different packs.
#[wasm_bindgen]
pub fn drive_restart() {
    with_state(|s| {
        if let (Some(d), Some(c)) = (&mut s.drive, &mut s.character) {
            d.start = kansei_wasm::now();
            c.controller.matcher.teleport(d.home.0, d.home.1);
        }
        if let Some(p) = &mut s.player {
            p.restart();
        }
    });
}

/// The motion pack's clip names, in its order (empty before it has loaded): for a page's clip
/// browser.
#[wasm_bindgen]
pub fn clip_names() -> Vec<String> {
    with_state(|s| s.character.as_ref().map(|c| c.db.clips.iter().map(|c| c.name.clone()).collect())).flatten().unwrap_or_default()
}

/// Play the clips whose names start with `pattern` one after another, as `play=` does, from where
/// the character stands now; `""` hands it back to the player (the clip playing finishes first).
/// Returns how many clips match.
#[wasm_bindgen]
pub fn play_clips(pattern: &str) -> usize {
    with_state(|s| {
        if pattern.is_empty() {
            s.player = None;
            return 0;
        }
        let Some(c) = &s.character else { return 0 };
        let at = c.controller.matcher.character();
        let player = demo::ClipPlayer::new(&c.db, pattern, (at.translation, yaw_of(at.rotation)));
        let n = player.clips.len();
        s.player = (n > 0).then_some(player);
        n
    })
    .unwrap_or(0)
}

/// Drive the `drive=1` route from where the character stands now, or hand it back to the player.
#[wasm_bindgen]
pub fn set_drive(on: bool) {
    with_state(|s| {
        s.drive = on.then(|| {
            let at = s.character.as_ref().map(|c| c.controller.matcher.character()).unwrap_or_default();
            demo::Drive { start: kansei_wasm::now(), home: (at.translation, yaw_of(at.rotation)), circle: None }
        });
        if !on {
            s.strafe = false;
        }
    });
}

/// The character's update times (ms per frame) over the last frames, as JSON (`timing::Window`).
#[wasm_bindgen]
pub fn motion_timings() -> String {
    with_state(|s| s.update_ms.json()).unwrap_or_default()
}

/// Milliseconds per `Database::search` over the pack's own frames as queries (every `step`th,
/// nudged), the default filter: the same queries as the TS page's `benchSearch`.
#[wasm_bindgen]
pub fn bench_search(step: usize) -> f64 {
    with_state(|s| s.character.as_ref().map(|c| timing::bench_search(&c.db, step))).flatten().unwrap_or(-1.0)
}

/// Milliseconds to sample a pose between two frames, run its forward kinematics and fill the
/// first body's bone palette, `count` times over the pack: the same as the TS page's `benchPose`.
#[wasm_bindgen]
pub fn bench_pose(count: usize) -> f64 {
    with_state(|s| {
        s.character.as_mut().map(|c| {
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
