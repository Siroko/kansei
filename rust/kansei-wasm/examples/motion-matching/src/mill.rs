//! A water mill in the lake: a paddle wheel on an axle, its lower paddles in the water, turning
//! steadily on a frame standing on the bed. Its paddles are the fluid's colliders, moving with
//! the wheel (`FluidCapsule::rigid`: each end at the speed its point of the wheel moves), so they
//! scoop the water along, lift it, throw it off at the top of their dip and leave a current
//! and a wake along the shore. The frame's legs stand in the water as still colliders.
//!
//! Turning, it keeps the lake awake (a disturbance for `FluidSleep`) while it is in view; out of
//! view the lake is culled as usual. The P panel turns it on and off and sets its speed.

use glam::{Mat4, Quat, Vec3 as GVec3};

use kansei_core::collision::{CollisionWorld, Obb};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::simulations::fluid::FluidCapsule;

use crate::lake::Lake;
use crate::props::{self, Mesh, Paint};

/// Where it stands: the angle from the lake's middle (radians from +x toward +z) and how far in
/// from the waterline (m).
const ANGLE: f32 = 1.9;
const INSIDE: f32 = 1.5;
/// The axle's height (m), the wheel's radius to the paddles' tips, its width, and its paddles.
/// The lowest paddle's collider clears the bed by 10 cm or more: water squeezed between a
/// collider and the floor jitters and never settles (the lake could not sleep with the mill
/// stopped).
const AXLE: f32 = 0.22;
const RADIUS: f32 = 0.58;
const WIDTH: f32 = 0.46;
const PADDLES: usize = 8;
/// The paddles' colliders: two capsules across each, at these radii, this thick.
const PADDLE_RINGS: [f32; 2] = [0.34, 0.5];
const PADDLE_CAPSULE: f32 = 0.08;
/// The frame's legs: half the spread along the axle and across it.
const LEGS: [f32; 2] = [0.42, 0.38];

pub struct Mill {
    wheel: usize,
    /// The axle's middle (world) and direction.
    center: GVec3,
    axle: GVec3,
    yaw: f32,
    angle: f32,
    /// The frame's legs (world ends).
    legs: Vec<[GVec3; 2]>,
    pub on: bool,
    /// Turns a minute.
    pub rpm: f32,
}

impl Mill {
    pub fn new(scene: &mut Scene, world: &mut CollisionWorld, lake: &Lake) -> Self {
        let [x, z] = lake.from_waterline(ANGLE, -INSIDE);
        let center = GVec3::new(x, AXLE, z);
        // the axle points out from the lake's middle, so the paddles push the water along the shore
        let [ox, oz] = lake.from_waterline(ANGLE, 0.0);
        let axle = GVec3::new(ox - x, 0.0, oz - z).normalize();
        // local +x is the axle: a yaw `a` turns +x to (cos a, 0, -sin a)
        let yaw = (-axle.z).atan2(axle.x);

        // the wheel, about local +x: a hub, two rims, spokes and the paddles
        let mut wheel = Mesh::default();
        wheel.rod(GVec3::X * -(LEGS[0] + 0.05), GVec3::X * (LEGS[0] + 0.05), 0.04, 12, Paint::Iron);
        for side in [-1.0f32, 1.0] {
            let x = side * WIDTH * 0.5;
            wheel.rod(GVec3::new(x - 0.03, 0.0, 0.0), GVec3::new(x + 0.03, 0.0, 0.0), 0.12, 12, Paint::DarkWood);
            let rim = RADIUS - 0.05;
            for k in 0..16 {
                let (a, b) = (k as f32 / 16.0 * std::f32::consts::TAU, (k + 1) as f32 / 16.0 * std::f32::consts::TAU);
                let p = |t: f32| GVec3::new(x, t.cos() * rim, t.sin() * rim);
                wheel.rod(p(a), p(b), 0.025, 6, Paint::DarkWood);
            }
            for k in 0..PADDLES {
                let a = (k as f32 + 0.5) / PADDLES as f32 * std::f32::consts::TAU;
                wheel.rod(GVec3::new(x, 0.0, 0.0), GVec3::new(x, a.cos() * rim, a.sin() * rim), 0.02, 6, Paint::Wood);
            }
        }
        for k in 0..PADDLES {
            let a = k as f32 / PADDLES as f32 * std::f32::consts::TAU;
            let at = Mat4::from_rotation_x(a) * Mat4::from_translation(GVec3::new(0.0, (PADDLE_RINGS[0] + RADIUS) * 0.5 - 0.04, 0.0));
            wheel.cuboid(at, GVec3::new(WIDTH + 0.04, RADIUS - PADDLE_RINGS[0] + 0.12, 0.035), Paint::Wood);
        }
        let mut r = Renderable::new(wheel.geometry("Mill/Wheel"), props::material("Mill/Wheel"));
        r.object.set_position(center.x, center.y, center.z);
        r.object.rotation.y = yaw;
        let wheel = scene.add(SceneNode::Renderable(r));

        // the frame: two A-frames on the bed either end of the axle, and a beam across their tops
        let local = |p: GVec3| center + Quat::from_rotation_y(yaw) * p;
        let mut legs = Vec::new();
        let mut frame = Mesh::default();
        for side in [-1.0f32, 1.0] {
            let top = GVec3::new(side * LEGS[0], 0.06, 0.0);
            for across in [-1.0f32, 1.0] {
                let foot = local(GVec3::new(side * LEGS[0], 0.0, across * LEGS[1]));
                let foot = GVec3::new(foot.x, lake.ground(foot.x, foot.z) - 0.05, foot.z);
                let foot_local = Quat::from_rotation_y(-yaw) * (foot - center);
                frame.rod(foot_local, top, 0.035, 8, Paint::Wood);
                // the collider stops short of the bed (see `AXLE`)
                legs.push([foot.lerp(local(top), 0.35), local(top)]);
            }
            frame.cuboid(Mat4::from_translation(GVec3::new(side * LEGS[0], 0.0, 0.0)), GVec3::new(0.09, 0.1, 0.12), Paint::Iron);
        }
        let mut r = Renderable::new(frame.geometry("Mill/Frame"), props::material("Mill/Frame"));
        r.object.set_position(center.x, center.y, center.z);
        r.object.rotation.y = yaw;
        scene.add(SceneNode::Renderable(r));

        // the character walks round it
        let bottom = center.y - RADIUS - 0.3;
        world.add_box(Obb::new(GVec3::new(center.x, (center.y + RADIUS + bottom) * 0.5, center.z), GVec3::new(LEGS[0] + 0.05, (center.y + RADIUS - bottom) * 0.5, RADIUS.max(LEGS[1]) + 0.05), Quat::from_rotation_y(yaw)));

        log::info!("mill at {:.2}, {:.2}", center.x, center.z);
        Self { wheel, center, axle, yaw, angle: 0.0, legs, on: true, rpm: 16.0 }
    }

    /// Radians a second it turns at (0 when off).
    fn omega(&self) -> f32 {
        if self.on { self.rpm * std::f32::consts::TAU / 60.0 } else { 0.0 }
    }

    /// Whether it is turning (stirring the water).
    pub fn turning(&self) -> bool {
        self.omega() > 0.0
    }

    /// Turn it by `dt` seconds.
    pub fn update(&mut self, dt: f32, scene: &mut Scene) {
        // the bottom paddles move toward -axle × up: along the shore
        self.angle = (self.angle + self.omega() * dt) % std::f32::consts::TAU;
        if let Some(r) = scene.get_renderable_mut(self.wheel) {
            r.object.rotation.x = self.angle;
            r.object.rotation.y = self.yaw;
        }
    }

    /// Its colliders in the world: the paddles, moving with the wheel, and the frame's legs.
    pub fn capsules(&self) -> Vec<FluidCapsule> {
        let to_world = Mat4::from_translation(self.center) * Mat4::from_rotation_y(self.yaw) * Mat4::from_rotation_x(self.angle);
        let spin = (self.axle * self.omega()).to_array();
        let mut capsules = Vec::with_capacity(PADDLES * 2 + self.legs.len());
        for k in 0..PADDLES {
            let a = k as f32 / PADDLES as f32 * std::f32::consts::TAU;
            for r in PADDLE_RINGS {
                // the paddle's radius points along local +y turned by `a` about +x
                let p = |x: f32| to_world.transform_point3(Mat4::from_rotation_x(a).transform_point3(GVec3::new(x, r, 0.0))).to_array();
                capsules.push(FluidCapsule::rigid(p(-WIDTH * 0.5), p(WIDTH * 0.5), PADDLE_CAPSULE, self.center.to_array(), [0.0; 3], spin));
            }
        }
        capsules.extend(self.legs.iter().map(|[a, b]| FluidCapsule::new(a.to_array(), b.to_array(), 0.045, [0.0; 3], [0.0; 3])));
        capsules
    }
}
