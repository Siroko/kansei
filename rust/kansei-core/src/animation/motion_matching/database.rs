use glam::{Quat, Vec3, Vec4};

use crate::animation::{angular_velocity, nlerp, Clip, Pose, Skeleton, Transform};

/// Features per frame (see `FeatureWeights` for the layout).
pub const FEATURES: usize = 27;
/// Floats per frame in the feature table: `FEATURES` padded to whole `Vec4`s.
pub const STRIDE: usize = 28;
/// Frames per small and large bounding box of the search's acceleration structure.
pub const BOUND_SMALL: usize = 16;
pub const BOUND_LARGE: usize = 64;
/// Seconds ahead of each trajectory sample.
pub const TRAJECTORY_TIMES: [f32; 3] = [1.0 / 3.0, 2.0 / 3.0, 1.0];

/// The character's forward axis in its root's space (glTF: models face +Z).
pub const FORWARD: Vec3 = Vec3::Z;

/// The joints the database reads: the root (whose motion moves the character, and whose facing
/// the features are relative to), the hips and the feet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JointRoles {
    pub root: usize,
    pub hips: usize,
    /// Left, right.
    pub feet: [usize; 2],
}

impl JointRoles {
    /// The joints with these names; the root must be a top joint (no parent).
    pub fn find(skeleton: &Skeleton, root: &str, hips: &str, left_foot: &str, right_foot: &str) -> Result<Self, String> {
        let find = |name: &str| skeleton.find(name).ok_or_else(|| format!("the skeleton has no joint '{name}'"));
        let roles = Self { root: find(root)?, hips: find(hips)?, feet: [find(left_foot)?, find(right_foot)?] };
        if skeleton.parents[roles.root].is_some() {
            return Err(format!("the root joint '{root}' has a parent: the character root must be a top joint"));
        }
        Ok(roles)
    }
}

/// Weights of the feature groups; the layout of a frame's features is:
///
/// | floats | feature (in the root's frame, at the character's facing) |
/// |---|---|
/// | 0-5 | left and right foot positions |
/// | 6-11 | left and right foot velocities |
/// | 12-14 | hips velocity |
/// | 15-20 | the root's future positions (x, z) at `TRAJECTORY_TIMES` |
/// | 21-26 | the root's future facing directions (x, z) at `TRAJECTORY_TIMES` |
///
/// Each group is normalized by its spread over the database, then scaled by its weight: a larger
/// weight makes that group matter more in the search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureWeights {
    pub foot_position: f32,
    pub foot_velocity: f32,
    pub hips_velocity: f32,
    pub trajectory_position: f32,
    pub trajectory_direction: f32,
}

impl Default for FeatureWeights {
    fn default() -> Self {
        Self { foot_position: 0.75, foot_velocity: 1.0, hips_velocity: 1.0, trajectory_position: 1.0, trajectory_direction: 1.5 }
    }
}

impl FeatureWeights {
    /// (first float, floats, weight) of each group.
    fn groups(&self) -> [(usize, usize, f32); 5] {
        [(0, 6, self.foot_position), (6, 6, self.foot_velocity), (12, 3, self.hips_velocity), (15, 6, self.trajectory_position), (21, 6, self.trajectory_direction)]
    }
}

/// When a foot is planted: its ankle or its ball (the foot joint's first child) below `height`
/// metres above where it is at rest and slower than `speed` metres per second, for at least
/// `min_frames` frames in a row. A running foot lands on its ball and rolls off it: its ankle is
/// never still long enough on its own.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactThresholds {
    pub height: f32,
    pub speed: f32,
    pub min_frames: usize,
}

impl Default for ContactThresholds {
    fn default() -> Self {
        Self { height: 0.15, speed: 1.0, min_frames: 3 }
    }
}

/// Tag bit (in `ClipInfo::tags`) of clips only played on command (traversals, falls, landings):
/// searches leave them out unless their filter asks for this bit.
pub const ACTION_TAG: u32 = 1 << 31;

/// One clip's frames in the database.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipInfo {
    pub name: String,
    /// First frame (a database index) and frame count.
    pub start: usize,
    pub frames: usize,
    /// Plays around: the last frame is the first again, and play wraps from it to frame 1.
    pub looping: bool,
    /// A bit mask the search can filter clips with (e.g. one bit per gait).
    pub tags: u32,
}

impl ClipInfo {
    /// Frames the playhead can be on: a loop's last frame is its first.
    pub fn playable(&self) -> usize {
        if self.looping { self.frames - 1 } else { self.frames }
    }

    pub fn contains(&self, frame: usize) -> bool {
        frame >= self.start && frame < self.start + self.frames
    }
}

/// A Vec3 per joint per frame: stored once for joints that never change, else quantized to 16
/// bits per axis over the joint's range.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Vec3Tracks {
    /// Per joint: its constant value, or `None` when it is animated.
    pub(crate) constant: Vec<Option<Vec3>>,
    /// Per animated joint, its slot in each frame's row of `animated`.
    pub(crate) slot: Vec<u32>,
    pub(crate) animated_joints: usize,
    /// Per slot: the middle of the joint's range and half its extent.
    pub(crate) center: Vec<Vec3>,
    pub(crate) extent: Vec<Vec3>,
    /// `frame * animated_joints + slot`: (value - center) / extent in signed 16 bits.
    pub(crate) animated: Vec<[i16; 3]>,
}

impl Vec3Tracks {
    /// Tracks of `values[frame][joint]`, keeping joints that vary by more than `tolerance` animated.
    fn new(values: &[Vec<Vec3>], joints: usize, tolerance: f32) -> Self {
        let mut constant = vec![None; joints];
        let mut slot = vec![u32::MAX; joints];
        let (mut center, mut extent) = (Vec::new(), Vec::new());
        for j in 0..joints {
            let (lo, hi) = values.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), f| (lo.min(f[j]), hi.max(f[j])));
            if values.is_empty() || (hi - lo).max_element() <= 2.0 * tolerance {
                constant[j] = Some(values.first().map_or(Vec3::ZERO, |_| (lo + hi) * 0.5));
            } else {
                slot[j] = center.len() as u32;
                center.push((lo + hi) * 0.5);
                extent.push(((hi - lo) * 0.5).max(Vec3::splat(1e-9)));
            }
        }
        let animated_joints = center.len();
        let mut animated = Vec::with_capacity(values.len() * animated_joints);
        for f in values {
            for j in (0..joints).filter(|&j| constant[j].is_none()) {
                let k = slot[j] as usize;
                let q = ((f[j] - center[k]) / extent[k]).clamp(Vec3::NEG_ONE, Vec3::ONE) * 32767.0;
                animated.push(q.round().to_array().map(|x| x as i16));
            }
        }
        Self { constant, slot, animated_joints, center, extent, animated }
    }

    fn get(&self, frame: usize, joint: usize) -> Vec3 {
        self.constant[joint].unwrap_or_else(|| {
            let k = self.slot[joint] as usize;
            let q = self.animated[frame * self.animated_joints + k];
            self.center[k] + self.extent[k] * Vec3::new(q[0] as f32, q[1] as f32, q[2] as f32) / 32767.0
        })
    }
}

/// A quaternion quantized to four signed 16-bit components (w kept non-negative).
pub(crate) fn quantize(q: Quat) -> [i16; 4] {
    let q = if q.w < 0.0 { -q } else { q };
    q.to_array().map(|c| (c.clamp(-1.0, 1.0) * 32767.0).round() as i16)
}

pub(crate) fn dequantize(q: [i16; 4]) -> Quat {
    Quat::from_array(q.map(|c| c as f32 / 32767.0)).normalize()
}

/// The yaw of a rotation: the heading its forward axis points to, about +Y.
pub fn yaw_of(rotation: Quat) -> f32 {
    let f = rotation * FORWARD;
    f.x.atan2(f.z)
}

/// A rotation of `yaw` radians about +Y.
pub fn yaw_rotation(yaw: f32) -> Quat {
    Quat::from_rotation_y(yaw)
}

/// Wrap an angle to (-pi, pi].
pub fn wrap_angle(a: f32) -> f32 {
    let a = (a + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
    if a <= -std::f32::consts::PI { a + std::f32::consts::TAU } else { a }
}

/// The joints foot contacts are read from: left ankle, left ball, right ankle, right ball (a foot
/// with no child is its own ball).
fn contact_joints(skeleton: &Skeleton, roles: &JointRoles) -> [usize; 4] {
    let ball = |foot: usize| skeleton.parents.iter().position(|p| *p == Some(foot)).unwrap_or(foot);
    [roles.feet[0], ball(roles.feet[0]), roles.feet[1], ball(roles.feet[1])]
}

/// The velocity of `positions` (per frame of a clip of `n` frames) at frame `f`, by central
/// differences: one-sided at a clip's ends, around the seam of a loop (the frame before its first
/// is its second-to-last a cycle back, the frame after its last is its second a cycle on; `cycle`
/// is the root's move over one).
fn clip_velocity(f: usize, n: usize, looping: bool, cycle: &Transform, rate: f32, positions: &dyn Fn(usize) -> Vec3) -> Vec3 {
    let (before, after, span) = match (looping, f) {
        (true, 0) => (cycle.inverse().transform_point(positions(n - 2)), positions(1), 2.0),
        (true, f) if f == n - 1 => (positions(n - 2), cycle.transform_point(positions(1)), 2.0),
        (false, 0) => (positions(0), positions(1), 1.0),
        (false, f) if f == n - 1 => (positions(n - 2), positions(n - 1), 1.0),
        (_, f) => (positions(f - 1), positions(f + 1), 2.0),
    };
    (after - before) * rate / span
}

/// A clip's foot contacts (per frame, bit 0 left planted, bit 1 right) from its `contact_joints`
/// positions and its root's height per frame (`ContactThresholds`).
fn plant(points: &[[Vec3; 4]], ground: &[f32], rest: [f32; 4], looping: bool, cycle: &Transform, rate: f32, thresholds: &ContactThresholds) -> Vec<u8> {
    let n = points.len();
    let mut planted = vec![[false; 2]; n];
    for side in 0..2 {
        for f in 0..n {
            planted[f][side] = (2 * side..2 * side + 2).any(|k| {
                let v = clip_velocity(f, n, looping, cycle, rate, &|i| points[i][k]);
                points[f][k].y - ground[f] < rest[k] + thresholds.height && v.length() < thresholds.speed
            });
        }
        // contacts: in runs of at least min_frames
        let mut f = 0;
        while f < n {
            if planted[f][side] {
                let run = (f..n).take_while(|&k| planted[k][side]).count();
                if run < thresholds.min_frames {
                    for k in f..f + run {
                        planted[k][side] = false;
                    }
                }
                f += run;
            } else {
                f += 1;
            }
        }
    }
    planted.iter().map(|p| p[0] as u8 | (p[1] as u8) << 1).collect()
}

/// Collects clips on one skeleton and builds their `Database`.
pub struct DatabaseBuilder {
    skeleton: Skeleton,
    roles: JointRoles,
    sample_rate: f32,
    weights: FeatureWeights,
    contacts: ContactThresholds,
    clips: Vec<(Clip, bool, u32)>,
}

impl DatabaseBuilder {
    pub fn new(skeleton: Skeleton, roles: JointRoles, sample_rate: f32) -> Self {
        Self { skeleton, roles, sample_rate, weights: FeatureWeights::default(), contacts: ContactThresholds::default(), clips: Vec::new() }
    }

    pub fn with_weights(mut self, weights: FeatureWeights) -> Self {
        self.weights = weights;
        self
    }

    pub fn with_contacts(mut self, contacts: ContactThresholds) -> Self {
        self.contacts = contacts;
        self
    }

    /// Add a clip on the builder's skeleton at its sample rate.
    pub fn add_clip(&mut self, clip: &Clip, looping: bool, tags: u32) -> Result<(), String> {
        if clip.joint_count() != self.skeleton.len() {
            return Err(format!("clip '{}' has {} joints, the skeleton {}", clip.name, clip.joint_count(), self.skeleton.len()));
        }
        if (clip.sample_rate - self.sample_rate).abs() > 1e-3 {
            return Err(format!("clip '{}' is at {} fps, the database at {}", clip.name, clip.sample_rate, self.sample_rate));
        }
        if clip.frame_count() < 2 || (looping && clip.frame_count() < 3) {
            return Err(format!("clip '{}' is too short ({} frames)", clip.name, clip.frame_count()));
        }
        self.clips.push((clip.clone(), looping, tags));
        Ok(())
    }

    pub fn build(self) -> Database {
        let joints = self.skeleton.len();
        let rate = self.sample_rate;
        let roles = self.roles;
        let mut clips = Vec::new();
        let mut rotations = Vec::new();
        let mut translations = Vec::new();
        let mut scales = Vec::new();
        let mut roots = Vec::new();
        let mut contacts = Vec::new();
        let mut raw: Vec<[f32; FEATURES]> = Vec::new();
        let contact_joints = contact_joints(&self.skeleton, &roles);
        let rest_points = {
            let rest = self.skeleton.rest_model();
            contact_joints.map(|j| rest[j].translation.y)
        };

        for (clip, looping, tags) in &self.clips {
            let n = clip.frame_count();
            let start = rotations.len() / joints;
            clips.push(ClipInfo { name: clip.name.clone(), start, frames: n, looping: *looping, tags: *tags });

            // the character root per frame: the root joint's position and heading; the pose keeps
            // what is left of the root joint under it (nothing, for a root on the ground)
            let mut pose = Pose { local: Vec::new() };
            let mut model = Vec::new();
            let mut root = Vec::with_capacity(n);
            let mut feet = Vec::with_capacity(n);
            let mut points = Vec::with_capacity(n);
            let mut hips = Vec::with_capacity(n);
            for f in 0..n {
                clip.frame_pose(f, &mut pose);
                pose.to_model(&self.skeleton, &mut model);
                let r = model[roles.root];
                let character = Transform::from_translation_rotation(r.translation, yaw_rotation(yaw_of(r.rotation)));
                root.push(character);
                pose.local[roles.root] = character.inverse().mul(&pose.local[roles.root]);
                for t in &pose.local {
                    rotations.push(quantize(t.rotation));
                }
                translations.push(pose.local.iter().map(|t| t.translation).collect::<Vec<_>>());
                scales.push(pose.local.iter().map(|t| t.scale).collect::<Vec<_>>());
                feet.push([model[roles.feet[0]].translation, model[roles.feet[1]].translation]);
                points.push(contact_joints.map(|j| model[j].translation));
                hips.push(model[roles.hips].translation);
            }

            let period = if *looping { n - 1 } else { n };
            // velocities by central differences (see `clip_velocity`)
            let cycle = root[n - 1].mul(&root[0].inverse());
            let velocity = |f: usize, positions: &dyn Fn(usize) -> Vec3| -> Vec3 { clip_velocity(f, n, *looping, &cycle, rate, positions) };
            let ground: Vec<f32> = root.iter().map(|r| r.translation.y).collect();
            contacts.extend(plant(&points, &ground, rest_points, *looping, &cycle, rate, &self.contacts));

            // the root at any frame ahead: loops continue cycle after cycle, other clips go on
            // at their last frame's velocity and turn rate
            let last_velocity = root[n - 2].inverse().transform_point(root[n - 1].translation) * rate;
            let last_turn = wrap_angle(yaw_of(root[n - 1].rotation) - yaw_of(root[n - 2].rotation)) * rate;
            let root_ahead = |f: usize| -> Transform {
                if f < n {
                    return root[f];
                }
                if *looping {
                    let (cycles, rest) = (f / period, f % period);
                    let mut t = root[rest];
                    for _ in 0..cycles {
                        t = cycle.mul(&t);
                    }
                    return t;
                }
                let dt = (f - (n - 1)) as f32 / rate;
                let yaw = yaw_of(root[n - 1].rotation);
                let heading = yaw_rotation(yaw + 0.5 * last_turn * dt);
                Transform::from_translation_rotation(root[n - 1].translation + heading * last_velocity * dt, yaw_rotation(yaw + last_turn * dt))
            };

            for f in 0..n {
                let here = root[f];
                let to_local = here.rotation.conjugate();
                let mut x = [0.0; FEATURES];
                for side in 0..2 {
                    let p = to_local * (feet[f][side] - here.translation);
                    x[3 * side..3 * side + 3].copy_from_slice(&p.to_array());
                    let v = to_local * velocity(f, &|k| feet[k][side]);
                    x[6 + 3 * side..9 + 3 * side].copy_from_slice(&v.to_array());
                }
                let v = to_local * velocity(f, &|k| hips[k]);
                x[12..15].copy_from_slice(&v.to_array());
                for (k, t) in TRAJECTORY_TIMES.iter().enumerate() {
                    let ahead = root_ahead(f + (t * rate).round() as usize);
                    let p = to_local * (ahead.translation - here.translation);
                    let d = to_local * (ahead.rotation * FORWARD);
                    x[15 + 2 * k] = p.x;
                    x[16 + 2 * k] = p.z;
                    x[21 + 2 * k] = d.x;
                    x[22 + 2 * k] = d.z;
                }
                raw.push(x);
            }
            roots.extend(root);
        }

        let rotations_len = rotations.len();
        debug_assert_eq!(rotations_len, raw.len() * joints);
        let mut db = Database {
            skeleton: self.skeleton,
            roles,
            sample_rate: rate,
            weights: self.weights,
            clips,
            rotations,
            translations: Vec3Tracks::new(&translations, joints, 1e-5),
            scales: Vec3Tracks::new(&scales, joints, 1e-5),
            roots,
            contacts,
            feature_offset: [0.0; FEATURES],
            feature_scale: [1.0; FEATURES],
            features: Vec::new(),
            bounds_small: Vec::new(),
            bounds_large: Vec::new(),
        };
        db.normalize(&raw);
        db.build_bounds();
        db
    }
}

/// Animation frames ready for motion matching: every clip's poses (quantized rotations), the
/// character root's motion, foot contacts and the normalized search features, with bounding boxes
/// over runs of frames for the search to skip.
#[derive(Debug, Clone, PartialEq)]
pub struct Database {
    pub skeleton: Skeleton,
    pub roles: JointRoles,
    pub sample_rate: f32,
    pub weights: FeatureWeights,
    pub clips: Vec<ClipInfo>,
    /// `frame * joints + joint`, local, the root joint relative to the character root.
    pub(crate) rotations: Vec<[i16; 4]>,
    pub(crate) translations: Vec3Tracks,
    pub(crate) scales: Vec3Tracks,
    /// The character root per frame, in its clip's space.
    pub(crate) roots: Vec<Transform>,
    /// Per frame: bit 0 left foot planted, bit 1 right.
    pub(crate) contacts: Vec<u8>,
    /// Per feature, subtracted then divided to normalize.
    pub feature_offset: [f32; FEATURES],
    pub feature_scale: [f32; FEATURES],
    /// `frame * STRIDE`, normalized, the padding zero.
    pub(crate) features: Vec<f32>,
    /// (min, max) per `BOUND_SMALL` and `BOUND_LARGE` frames.
    pub(crate) bounds_small: Vec<([f32; STRIDE], [f32; STRIDE])>,
    pub(crate) bounds_large: Vec<([f32; STRIDE], [f32; STRIDE])>,
}

impl Database {
    pub fn frame_count(&self) -> usize {
        self.roots.len()
    }

    pub fn joint_count(&self) -> usize {
        self.skeleton.len()
    }

    /// The clip holding database frame `frame`.
    pub fn clip_of(&self, frame: usize) -> usize {
        self.clips.partition_point(|c| c.start <= frame) - 1
    }

    /// A frame's normalized features.
    pub fn features(&self, frame: usize) -> &[f32] {
        &self.features[frame * STRIDE..(frame + 1) * STRIDE]
    }

    /// Whether each foot is planted at `frame`.
    pub fn contacts(&self, frame: usize) -> [bool; 2] {
        let c = self.contacts[frame];
        [c & 1 != 0, c & 2 != 0]
    }

    /// Find the feet's contacts again from the poses, with `thresholds` (packs baked before
    /// contacts read the balls of the feet as well as the ankles get them that way).
    pub fn detect_contacts(&mut self, thresholds: &ContactThresholds) {
        let joints = contact_joints(&self.skeleton, &self.roles);
        let rest = {
            let rest = self.skeleton.rest_model();
            joints.map(|j| rest[j].translation.y)
        };
        // each contact joint's chain of ancestors, root first
        let chains = joints.map(|j| {
            let mut chain = vec![j];
            while let Some(p) = self.skeleton.parents[*chain.last().unwrap()] {
                chain.push(p);
            }
            chain.reverse();
            chain
        });
        let mut contacts = Vec::with_capacity(self.frame_count());
        for clip in &self.clips {
            let (n, start) = (clip.frames, clip.start);
            let points: Vec<[Vec3; 4]> = (start..start + n)
                .map(|f| chains.each_ref().map(|chain| chain.iter().fold(self.roots[f], |t, &j| t.mul(&self.transform(f, j))).translation))
                .collect();
            let ground: Vec<f32> = self.roots[start..start + n].iter().map(|r| r.translation.y).collect();
            let cycle = self.roots[start + n - 1].mul(&self.roots[start].inverse());
            contacts.extend(plant(&points, &ground, rest, clip.looping, &cycle, self.sample_rate, thresholds));
        }
        self.contacts = contacts;
    }

    /// The character root at `frame`, in its clip's space.
    pub fn root(&self, frame: usize) -> Transform {
        self.roots[frame]
    }

    /// A joint's local transform at a frame.
    pub fn transform(&self, frame: usize, joint: usize) -> Transform {
        let j = self.joint_count();
        Transform {
            translation: self.translations.get(frame, joint),
            rotation: dequantize(self.rotations[frame * j + joint]),
            scale: self.scales.get(frame, joint),
        }
    }

    /// The pose between frames `a` and `b` (`t` of the way), into `out`.
    pub fn pose(&self, a: usize, b: usize, t: f32, out: &mut Pose) {
        out.local.clear();
        for j in 0..self.joint_count() {
            let (x, y) = (self.transform(a, j), self.transform(b, j));
            out.local.push(Transform { translation: x.translation.lerp(y.translation, t), rotation: nlerp(x.rotation, y.rotation, t), scale: x.scale.lerp(y.scale, t) });
        }
    }

    /// Each joint's local linear and angular velocity at `frame` (forward difference, backward at
    /// a clip's last frame).
    pub fn velocities(&self, frame: usize, linear: &mut Vec<Vec3>, angular: &mut Vec<Vec3>) {
        let clip = &self.clips[self.clip_of(frame)];
        let (a, b) = if frame + 1 < clip.start + clip.frames { (frame, frame + 1) } else { (frame - 1, frame) };
        linear.clear();
        angular.clear();
        for j in 0..self.joint_count() {
            let (x, y) = (self.transform(a, j), self.transform(b, j));
            linear.push((y.translation - x.translation) * self.sample_rate);
            angular.push(angular_velocity(x.rotation, y.rotation, 1.0 / self.sample_rate));
        }
    }

    /// The character root (position and heading) at fractional frame `frame` of clip `clip`, in
    /// the clip's space.
    pub fn root_at(&self, clip: usize, frame: f32) -> (Vec3, f32) {
        self.root_in(&self.clips[clip], frame)
    }

    fn root_in(&self, clip: &ClipInfo, f: f32) -> (Vec3, f32) {
        let a = (f.floor() as usize).min(clip.frames - 1);
        let b = (a + 1).min(clip.frames - 1);
        let t = f - a as f32;
        let (ra, rb) = (self.roots[clip.start + a], self.roots[clip.start + b]);
        let ya = yaw_of(ra.rotation);
        (ra.translation.lerp(rb.translation, t), ya + wrap_angle(yaw_of(rb.rotation) - ya) * t)
    }

    /// How the character root moves while clip `clip` plays from fractional frame `from` to `to`
    /// (clip frames; `to` may pass a loop's end): the displacement in the root's frame at `from`,
    /// and the turn in radians about +Y.
    pub fn root_motion(&self, clip: usize, from: f32, to: f32) -> (Vec3, f32) {
        let info = &self.clips[clip];
        if info.looping && to > (info.frames - 1) as f32 {
            // to the loop's end, then on from its start
            let end = (info.frames - 1) as f32;
            let (d0, y0) = self.root_motion(clip, from, end);
            let (d1, y1) = self.root_motion(clip, 0.0, to - end);
            return (d0 + yaw_rotation(y0) * d1, y0 + y1);
        }
        let to = to.min((info.frames - 1) as f32);
        let (pa, ya) = self.root_in(info, from);
        let (pb, yb) = self.root_in(info, to);
        (yaw_rotation(-ya) * (pb - pa), wrap_angle(yb - ya))
    }

    /// Normalized features of raw ones.
    pub fn normalize_query(&self, raw: &[f32; FEATURES]) -> [f32; STRIDE] {
        let mut out = [0.0; STRIDE];
        for i in 0..FEATURES {
            out[i] = (raw[i] - self.feature_offset[i]) / self.feature_scale[i];
        }
        out
    }

    /// Raw features of normalized ones.
    pub fn denormalize(&self, features: &[f32]) -> [f32; FEATURES] {
        std::array::from_fn(|i| features[i] * self.feature_scale[i] + self.feature_offset[i])
    }

    /// Per-feature mean; per-group spread (the root mean square deviation over the group), divided
    /// by the group's weight.
    fn normalize(&mut self, raw: &[[f32; FEATURES]]) {
        let n = raw.len().max(1) as f32;
        for i in 0..FEATURES {
            self.feature_offset[i] = raw.iter().map(|x| x[i]).sum::<f32>() / n;
        }
        for (first, count, weight) in self.weights.groups() {
            let variance = raw.iter().map(|x| (first..first + count).map(|i| (x[i] - self.feature_offset[i]).powi(2)).sum::<f32>()).sum::<f32>() / (n * count as f32);
            let spread = variance.sqrt().max(1e-6);
            for i in first..first + count {
                self.feature_scale[i] = spread / weight.max(1e-6);
            }
        }
        self.features = Vec::with_capacity(raw.len() * STRIDE);
        for x in raw {
            self.features.extend_from_slice(&self.normalize_query(x));
        }
    }

    pub(crate) fn build_bounds(&mut self) {
        let frames = self.frame_count();
        let bounds = |size: usize| {
            (0..frames.div_ceil(size))
                .map(|b| {
                    let mut lo = [f32::MAX; STRIDE];
                    let mut hi = [f32::MIN; STRIDE];
                    for f in b * size..((b + 1) * size).min(frames) {
                        for (i, &v) in self.features(f).iter().enumerate() {
                            lo[i] = lo[i].min(v);
                            hi[i] = hi[i].max(v);
                        }
                    }
                    (lo, hi)
                })
                .collect::<Vec<_>>()
        };
        let (small, large) = (bounds(BOUND_SMALL), bounds(BOUND_LARGE));
        self.bounds_small = small;
        self.bounds_large = large;
    }
}

/// Squared distance between two feature rows, stopping once it passes `limit`.
#[inline]
pub(crate) fn distance(a: &[f32], b: &[f32], limit: f32) -> f32 {
    let mut total = 0.0;
    for k in 0..STRIDE / 4 {
        let d = Vec4::from_slice(&a[4 * k..]) - Vec4::from_slice(&b[4 * k..]);
        total += d.dot(d);
        if total >= limit {
            break;
        }
    }
    total
}

/// Squared distance from `q` to a box, a lower bound of its distance to every row inside.
#[inline]
pub(crate) fn box_distance(q: &[f32; STRIDE], (lo, hi): &([f32; STRIDE], [f32; STRIDE]), limit: f32) -> f32 {
    let mut total = 0.0;
    for k in 0..STRIDE / 4 {
        let v = Vec4::from_slice(&q[4 * k..]);
        let d = v - v.max(Vec4::from_slice(&lo[4 * k..])).min(Vec4::from_slice(&hi[4 * k..]));
        total += d.dot(d);
        if total >= limit {
            break;
        }
    }
    total
}
