//! A little water cannon on the lake's west bank, aimed in an arc into the lake. Walk up to it
//! (within `REACH`; on the lake page, which has no character, it is always ready) and it offers to
//! fire: E (keyboard), X (gamepad) or a click on the prompt. A press fires a short burst; holding keeps pouring. The water comes out of the muzzle as new
//! particles (`FluidNozzle` into the lake's spare capacity), flies its arc, splashes in, and the
//! lake's level rises, up to the lake's highest (`Lake::full`); then the prompt says so and R (Y)
//! drains it back to the start, as the P panel's reset does.
//!
//! The rate is the nozzle's (a disc 20 cm across at 5 m/s, about 2,000 particles a second), so
//! filling the lake takes about half a minute and never more than a step's worth arrives at once.

use glam::{Mat4, Quat, Vec3 as GVec3};

use kansei_core::collision::{CollisionWorld, Obb};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::simulations::fluid::FluidNozzle;

use crate::lake::Lake;
use crate::props::{self, Mesh, Paint};

/// Where it stands: the angle from the lake's middle (radians from +x toward +z) and how far
/// outside the waterline (m), on the bank.
const ANGLE: f32 = 3.25;
const FROM_WATERLINE: f32 = 0.85;
/// The barrel: its pivot's height over the ground, its length, its elevation, and the muzzle's
/// inner radius.
const PIVOT: f32 = 0.5;
const BARREL: f32 = 0.8;
const ELEVATION: f32 = 0.42;
const BORE: f32 = 0.1;
/// The stream's speed (m/s) and the shortest burst (s).
const SPEED: f32 = 5.0;
const BURST: f32 = 0.6;
/// How near (m, on the ground) the character must be to fire.
pub const REACH: f32 = 2.5;

pub struct Cannon {
    /// The barrel's renderable (it recoils while firing).
    barrel: usize,
    /// The pivot (world), the way it fires (the barrel's axis) and its heading.
    pivot: GVec3,
    aim: GVec3,
    yaw: f32,
    nozzle: FluidNozzle,
    /// Seconds left of the burst; seconds it has been firing.
    burst: f32,
    /// Held since it was fired: it keeps pouring.
    holding: bool,
    firing_for: f32,
    /// Whether it may fire: the character near enough (or no character needed), as of the last
    /// update.
    pub near: bool,
    /// Particles poured in since the page opened.
    pub poured: u64,
}

impl Cannon {
    /// Build it on the bank, a blockout like the course's boxes: a base and a barrel on its
    /// trunnion, in `scene`, and its base as a box in `world` (the character walks round it).
    pub fn new(scene: &mut Scene, world: &mut CollisionWorld, lake: &Lake) -> Self {
        let [x, z] = lake.from_waterline(ANGLE, FROM_WATERLINE);
        let ground = lake.ground(x, z);
        let base = GVec3::new(x, ground, z);
        // facing the lake's middle
        let to_lake = (GVec3::new(lake.from_waterline(ANGLE, -3.0)[0], 0.0, lake.from_waterline(ANGLE, -3.0)[1]) - GVec3::new(x, 0.0, z)).normalize();
        // the props' local +z is forward: a yaw `a` turns +z to (sin a, 0, cos a)
        let yaw = to_lake.x.atan2(to_lake.z);
        let pivot = base + GVec3::Y * PIVOT;
        let aim = Quat::from_rotation_y(yaw) * GVec3::new(0.0, ELEVATION.sin(), ELEVATION.cos());

        // the base: a slab on the ground and two cheeks holding the barrel's trunnion
        let mut stand = Mesh::default();
        stand.cuboid(Mat4::from_translation(GVec3::new(0.0, 0.1, -0.1)), GVec3::new(0.5, 0.2, 0.9), Paint::Grey);
        for side in [-1.0f32, 1.0] {
            let (bottom, top) = (0.2, PIVOT + 0.08);
            stand.cuboid(Mat4::from_translation(GVec3::new(side * 0.18, (bottom + top) * 0.5, 0.0)), GVec3::new(0.08, top - bottom, 0.3), Paint::Grey);
        }
        let place = |r: &mut Renderable, at: GVec3| {
            r.object.set_position(at.x, at.y, at.z);
            r.object.rotation.y = yaw;
        };
        let mut r = Renderable::new(stand.geometry("Cannon/Base"), props::material("Cannon/Base"));
        place(&mut r, base);
        scene.add(SceneNode::Renderable(r));

        // the barrel, along local +z from behind its pivot to the muzzle, pitched up
        let mut barrel = Mesh::default();
        barrel.rod(GVec3::Z * -0.3, GVec3::Z * (BARREL - 0.1), 0.13, 16, Paint::Orange);
        barrel.rod(GVec3::Z * (BARREL - 0.1), GVec3::Z * BARREL, 0.15, 16, Paint::Dark);
        barrel.beam(GVec3::new(-0.22, 0.0, 0.0), GVec3::new(0.22, 0.0, 0.0), 0.07, Paint::Dark);
        // the bore: a black disc just inside the muzzle
        barrel.cylinder(Mat4::from_translation(GVec3::Z * (BARREL - 0.005)) * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2), BORE, 0.01, 16, Paint::Black);
        let mut r = Renderable::new(barrel.geometry("Cannon/Barrel"), props::material("Cannon/Barrel"));
        place(&mut r, pivot);
        r.object.rotation.x = -ELEVATION;
        let barrel = scene.add(SceneNode::Renderable(r));

        world.add_box(Obb::new(base + GVec3::new(0.0, 0.4, 0.0), GVec3::new(0.32, 0.4, 0.55), Quat::from_rotation_y(yaw)));

        let muzzle = pivot + aim * BARREL;
        log::info!("cannon at {:.2}, {:.2}, heading {:.0}°", base.x, base.z, yaw.to_degrees());
        let mut nozzle = FluidNozzle::new(lake.to_sim(muzzle), aim.to_array(), lake.sim_length(BORE), 0.0, lake.sim_length(lake.spacing()));
        nozzle.jitter = 0.12;
        nozzle.spread = 0.03;
        Self { barrel, pivot, aim, yaw, nozzle, burst: 0.0, holding: false, firing_for: 0.0, near: false, poured: 0 }
    }

    /// Whether a character at `at` (world) is near enough to fire it.
    pub fn in_reach(&self, at: GVec3) -> bool {
        let d = at - self.pivot;
        (d.x * d.x + d.z * d.z).sqrt() < REACH
    }

    /// Advance by `dt`: whether it may fire (`near`: a character within reach, or always on the
    /// lake page), and whether it fires (`trigger` pressed this frame, `held`). Returns the stream
    /// to pour this frame, if firing.
    pub fn update(&mut self, dt: f32, near: bool, trigger: bool, held: bool, lake: &Lake, scene: &mut Scene) -> Option<&mut FluidNozzle> {
        self.near = near;
        if self.near && trigger && !lake.full() {
            if !self.firing() {
                self.nozzle.restart();
            }
            self.burst = BURST;
            self.holding = true;
        }
        self.holding &= held && self.near;
        let firing = (self.burst > 0.0 || self.holding) && !lake.full();
        self.burst = (self.burst - dt).max(0.0);
        self.firing_for = if firing { self.firing_for + dt } else { 0.0 };

        // the barrel kicks back as it starts, and trembles while it pours
        if let Some(r) = scene.get_renderable_mut(self.barrel) {
            let kick = if firing { 0.06 * (-self.firing_for * 12.0).exp() + 0.006 * (self.firing_for * 90.0).sin() } else { 0.0 };
            let at = self.pivot - self.aim * kick;
            r.object.set_position(at.x, at.y, at.z);
            r.object.rotation.y = self.yaw;
        }
        if !firing {
            return None;
        }
        // the stream's speed in the simulation's units (the lake's time scale can change)
        self.nozzle.speed = lake.velocity_to_sim(GVec3::splat(SPEED)).x;
        Some(&mut self.nozzle)
    }

    pub fn firing(&self) -> bool {
        self.firing_for > 0.0
    }

    /// What the page shows near the cannon (empty when it may not fire).
    pub fn prompt(&self, lake: &Lake) -> String {
        if !self.near {
            String::new()
        } else if lake.full() {
            "The lake is full · R / Y — drain it (or reset water, P)".to_string()
        } else {
            format!("E / X — fire water · hold to pour · lake {:.0}% full", lake.fill() * 100.0)
        }
    }
}
