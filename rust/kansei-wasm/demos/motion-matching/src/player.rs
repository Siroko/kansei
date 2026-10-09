//! The player both pages share: keyboard and gamepad input, the character's controller stepped
//! against the page's collision world, the overlay (trajectory, feet, ledge, skeleton), the follow
//! camera and the HUD. A page steps it each frame ([`Player::update`]), acts on the keys it leaves
//! ([`PlayerFrame::keys`]) and on what the character does in its world (legs, landing), and shows
//! [`Player::hud`] with its own lines.

use glam::{Mat4, Quat, Vec3 as GVec3};

use kansei_core::animation::motion_matching::traversal::CharacterState;
use kansei_core::animation::motion_matching::{yaw_of, MotionInput};
use kansei_core::animation::PALETTE_BINDING;
use kansei_core::collision::CollisionWorld;
use kansei_core::controls::CameraControls;
use kansei_core::debug::segment;
use kansei_core::math::Vec3;
use kansei_core::objects::Scene;
use kansei_core::renderers::Renderer;
use kansei_wasm::{param, Gamepad, Keys};

use crate::character::{pace, Character, RUN, WALK};
use crate::{demo, timing};

/// What the character did this frame, for the page's world, and the keys the player left to it.
#[derive(Default)]
pub struct PlayerFrame {
    /// Keys pressed this frame that the player does not use (the gamepad's X and Y come as "e" and
    /// "r").
    pub keys: Vec<String>,
    /// E or the gamepad's X held.
    pub e_held: bool,
    /// The legs as capsules (world ends, radius), empty without a character.
    pub legs: Vec<(GVec3, GVec3, f32)>,
    /// A landing: back on its feet after being in the air, and the fastest it came down (m/s).
    pub landing: Option<(GVec3, f32)>,
    /// Where the character stands.
    pub at: Option<GVec3>,
    /// Running (Shift, B, the right trigger, or the route).
    pub run: bool,
}

pub struct Player {
    pub character: Option<Character>,
    pub keys: Keys,
    pub pad: Gamepad,
    pub frame: u32,
    pub strafe: bool,
    pub overlay: bool,
    /// The camera follows the character's hips (on by default).
    pub follow: bool,
    /// Walk and run paces: forward, sideways, backward.
    pub speeds: ([f32; 3], [f32; 3]),
    pub fps: f32,
    searches: u32,
    switches: u32,
    counted_since: f64,
    rates: (f32, f32),
    /// Whether the character was in the air last frame, and the fastest it has fallen since
    /// (m/s); its height last frame.
    air: (bool, f32),
    last_y: f32,
    /// `drive=1`: the scripted route; `play=<pattern>`: the clips played in turn.
    pub drive: Option<demo::Drive>,
    pub player: Option<demo::ClipPlayer>,
    /// The character's update (search, pose, IK) per frame, in ms: the last `timing::WINDOW` frames.
    pub update_ms: timing::Window,
    run: bool,
    last: f64,
}

/// The walk and run paces, scaled by `walk=` and `run=` (forward m/s).
pub fn speeds_from_url() -> ([f32; 3], [f32; 3]) {
    let scaled = |paces: [f32; 3], name: &str| param(name).and_then(|v| v.parse::<f32>().ok()).map_or(paces, |forward| paces.map(|p| p * forward / paces[0]));
    (scaled(WALK, "walk"), scaled(RUN, "run"))
}

impl Player {
    pub fn new(character: Option<Character>) -> Self {
        let now = kansei_wasm::now();
        let mut player = Self {
            character,
            keys: Keys::listen(),
            pad: Gamepad::new(),
            frame: 0,
            strafe: false,
            overlay: true,
            follow: true,
            speeds: speeds_from_url(),
            fps: 60.0,
            searches: 0,
            switches: 0,
            counted_since: now,
            rates: (0.0, 0.0),
            air: (false, 0.0),
            last_y: 0.0,
            drive: None,
            player: None,
            update_ms: timing::Window::default(),
            run: false,
            last: now,
        };
        // `drive=1`, `play=<pattern>`: from where the character starts
        let home = player.character.as_ref().map(|c| {
            let at = c.controller.matcher.character();
            (at.translation, yaw_of(at.rotation))
        });
        if let Some(home) = home {
            // circle=<radius>: round a circle at the run pace (negative radius: turning right)
            let circle = param("circle").and_then(|r| r.parse::<f32>().ok()).filter(|r| r.abs() > 0.1).map(|r| player.speeds.1[0] / r);
            if kansei_wasm::flag("drive", false) || circle.is_some() {
                player.drive = Some(demo::Drive { start: now, home, circle });
            }
            if let Some(prefix) = param("play") {
                player.player = player.character.as_ref().map(|c| demo::ClipPlayer::new(&c.db, &prefix, home));
            }
        }
        player
    }

    /// When the last frame was (`kansei_wasm::now`), and `now` from here on: a page's frame time.
    pub fn last_frame(&mut self, now: f64) -> f64 {
        std::mem::replace(&mut self.last, now)
    }

    /// Step the player by `dt` at `now`: input, the character against `collision`, its body and
    /// overlay into `scene`, the camera following its hips.
    pub fn update(&mut self, now: f64, dt: f32, scene: &mut Scene, renderer: &Renderer, controls: &mut CameraControls, collision: &CollisionWorld) -> PlayerFrame {
        self.frame += 1;
        self.fps = self.fps * 0.95 + 0.05 / dt;
        let mut out = PlayerFrame::default();

        // input: keyboard, then the gamepad on top
        let keys = &self.keys;
        let mut stick = [keys.axis(&["a", "arrowleft"], &["d", "arrowright"]), keys.axis(&["s", "arrowdown"], &["w", "arrowup"])];
        let mut run = keys.held("shift");
        let mut toggles = keys.take_pressed();
        out.e_held = keys.held("e");
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
            controls.rotate(-x * 2.5 * dt, y * 1.5 * dt);
            run |= pad.held(Gamepad::B) || pad.value(Gamepad::RIGHT_TRIGGER) > 0.3;
            for (button, key) in [(Gamepad::A, " "), (Gamepad::LEFT_BUMPER, "q"), (Gamepad::X, "e"), (Gamepad::Y, "r")] {
                if pad.pressed(button) {
                    toggles.push(key.into());
                }
            }
            out.e_held |= pad.held(Gamepad::X);
        }
        let mut traverse = false;
        for key in toggles {
            match key.as_str() {
                " " => traverse = true,
                "q" => self.strafe = !self.strafe,
                "b" => self.overlay = !self.overlay,
                "c" => {
                    if let Some(c) = &mut self.character {
                        let next = c.showing + 1;
                        c.show(scene, next);
                    }
                }
                "k" | "m" | "l" => {
                    if let Some(c) = &mut self.character {
                        match key.as_str() {
                            "k" => {
                                let visible = c.bones.visible(scene);
                                c.bones.set_visible(scene, !visible);
                            }
                            "m" => {
                                if let Some(r) = scene.get_renderable_mut(c.bodies[c.showing].index) {
                                    r.visible = !r.visible;
                                }
                            }
                            _ => c.controller.matcher.settings.foot_lock = !c.controller.matcher.settings.foot_lock,
                        }
                    }
                }
                _ => out.keys.push(key),
            }
        }

        // the camera's heading on the ground: the stick moves relative to it
        let azimuth = controls.azimuth();
        let forward = GVec3::new(-azimuth.sin(), 0.0, -azimuth.cos());
        let right = GVec3::new(azimuth.cos(), 0.0, -azimuth.sin());
        let step = self.drive.as_ref().map(|d| d.step(now));
        if let Some(step) = &step {
            run = step.run;
            self.strafe = step.facing.is_some();
        }
        self.run = run;
        out.run = run;
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
                c.controller.update(&c.db, collision, &MotionInput { velocity, facing }, dt);
                self.update_ms.push(((kansei_wasm::now() - t0) * 1000.0) as f32);
            }
            if traverse {
                // an obstacle ahead: traverse it (pressed a little early, once in reach); else jump
                let _ = c.controller.request_traverse_or_jump(&c.db, collision, 1.0);
            }
            // the obstacle last looked at: its ledge, and a post down to the floor
            c.ledge.matrices.fill(Mat4::ZERO);
            if let (true, Some(o)) = (self.overlay, c.controller.last_obstacle) {
                let side = o.normal.cross(GVec3::Y);
                c.ledge.matrices[0] = segment(o.ledge - side * 0.4, o.ledge + side * 0.4, 0.04);
                c.ledge.matrices[1] = segment(o.ledge, o.ledge - GVec3::Y * o.height, 0.02);
            }
            c.ledge.upload(renderer);
            let search = c.controller.matcher.last_search();
            self.searches += search.searched as u32;
            self.switches += search.switched as u32;

            let character = c.controller.matcher.character();
            let body = &mut c.bodies[c.showing];
            if let Some(r) = scene.get_renderable_mut(body.index) {
                r.object.set_position(character.translation.x, character.translation.y, character.translation.z);
                r.object.rotation.y = yaw_of(character.rotation);
                body.palette.update(&body.mesh, c.controller.matcher.model());
                if let Some(buffer) = r.material.bindable_buffer(PALETTE_BINDING) {
                    body.palette.upload(renderer.queue(), &buffer);
                }
            }
            // follow the character's hips
            let hips = character.transform_point(c.controller.matcher.model()[c.joint(c.db.roles.hips)].translation);
            let target = GVec3::new(character.translation.x, hips.y * 0.9, character.translation.z);
            if self.follow {
                controls.follow(Vec3::new(target.x, target.y, target.z), dt, 8.0);
            }

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
            c.trajectory.upload(renderer);
            if c.bones.visible(scene) {
                c.bones.matrices.fill(Mat4::ZERO);
                for (j, a, b) in c.bones_world() {
                    c.bones.matrices[j] = segment(a, b, 0.015);
                }
                c.bones.upload(renderer);
            }

            // a landing: back on its feet after being in the air, at the fastest it came down
            let y = character.translation.y;
            let airborne = matches!(c.controller.state(), CharacterState::Jumping | CharacterState::Falling(_));
            let (was, fall) = &mut self.air;
            if airborne {
                *fall = fall.max((self.last_y - y) / dt);
            } else if *was {
                out.landing = Some((character.translation, *fall));
                *fall = 0.0;
            }
            *was = airborne;
            self.last_y = y;
            out.at = Some(character.translation);
            out.legs = c.leg_capsules();
        }

        if now - self.counted_since > 1.0 {
            let span = (now - self.counted_since) as f32;
            self.rates = (self.searches as f32 / span, self.switches as f32 / span);
            self.searches = 0;
            self.switches = 0;
            self.counted_since = now;
        }
        out
    }

    /// The HUD's text with the overlay on (`world`: the page's lines, ending in a newline or empty;
    /// `hint`: its keys, after the shared ones), "" with it off; None without a character (the
    /// page's own message stays).
    pub fn hud(&self, world: &str, hint: &str) -> Option<String> {
        let c = self.character.as_ref()?;
        if !self.overlay {
            return Some(String::new());
        }
        let (clip, frame) = c.controller.matcher.playing();
        let info = &c.db.clips[clip];
        let s = c.controller.matcher.last_search();
        let feet = c.controller.matcher.feet_locked();
        let speed = c.controller.matcher.simulation().velocity.length();
        Some(format!(
            "{:.0} fps   {} {}{}\nclip   {}\nframe  {:.0} / {}{}\nsearch {:.0}/s, switch {:.1}/s, cost {:.3}\nupdate {}\nfeet   {} {}  (lock {})\n{}\n\n{}\n\nWASD / left stick move · Shift / B run · Space / A jump, traverse · Q / LB strafe\ndrag / right stick orbit · B overlay · K skeleton · M mesh · L foot lock{}",
            self.fps,
            if self.run { "run" } else { "walk" },
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
            format_args!("{}{}", world, if c.bodies.len() > 1 { format!("character: {} (C to switch)", c.bodies[c.showing].name) } else { String::new() }),
            hint,
        ))
    }

    /// Start the `drive=1` route (or the `play=` clips) over, the character back where it started.
    pub fn drive_restart(&mut self) {
        if let (Some(d), Some(c)) = (&mut self.drive, &mut self.character) {
            d.start = kansei_wasm::now();
            c.controller.matcher.teleport(d.home.0, d.home.1);
        }
        if let Some(p) = &mut self.player {
            p.restart();
        }
    }

    /// Play the clips matching `pattern` from where the character stands (`""`: back to the
    /// player); how many match.
    pub fn play_clips(&mut self, pattern: &str) -> usize {
        if pattern.is_empty() {
            self.player = None;
            return 0;
        }
        let Some(c) = &self.character else { return 0 };
        let at = c.controller.matcher.character();
        let player = demo::ClipPlayer::new(&c.db, pattern, (at.translation, yaw_of(at.rotation)));
        let n = player.clips.len();
        self.player = (n > 0).then_some(player);
        n
    }

    /// Drive the route from where the character stands, or hand it back to the player.
    pub fn set_drive(&mut self, on: bool) {
        self.drive = on.then(|| {
            let at = self.character.as_ref().map(|c| c.controller.matcher.character()).unwrap_or_default();
            demo::Drive { start: kansei_wasm::now(), home: (at.translation, yaw_of(at.rotation)), circle: None }
        });
        if !on {
            self.strafe = false;
        }
    }
}
