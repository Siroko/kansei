//! `.kmm`: a motion-matching database and its skinned meshes in one little-endian binary file;
//! or, as a `CharacterPack`, a character to show that animation on (skeleton, meshes, images).
//!
//! `KMMP`, a u32 version, then sections: a 4-byte tag, a u64 length and the payload. Readers skip
//! tags they don't know. Sections: `SKEL` skeleton, `ROLE` joint roles, rate and feature weights,
//! `CLIP` clips, `ROTS` quantized rotations, `TRAN`/`SCAL` translation and scale tracks, `ROOT`
//! character root per frame, `CONT` foot contacts, `FEAT` feature normalization and rows, `MESH`
//! (repeated) skinned meshes with a colour, `IMAG` (repeated) named images (encoded bytes, e.g.
//! WebP), `ACTS` action clips (traversals, falls, landings: what `traversal::ActionClip::analyze`
//! found), `META` key/value strings (source, licence).

use glam::{Mat4, Quat, Vec3};

use super::database::{ClipInfo, Database, FeatureWeights, JointRoles, Vec3Tracks, FEATURES, STRIDE};
use super::traversal::{ActionClip, ActionKind};
use crate::animation::{Skeleton, SkinnedMesh, Transform, MAX_INFLUENCES};
use crate::geometries::Vertex;

pub const MAGIC: &[u8; 4] = b"KMMP";
pub const VERSION: u32 = 1;

/// A skinned mesh of a pack and the colour to draw it with.
#[derive(Debug, Clone)]
pub struct PackMesh {
    pub mesh: SkinnedMesh,
    pub color: [f32; 4],
}

/// Everything a motion-matched character needs, as one file.
#[derive(Debug, Clone)]
pub struct MotionPack {
    pub database: Database,
    pub meshes: Vec<PackMesh>,
    /// Clips played on command, with their analysis.
    pub actions: Vec<ActionClip>,
    /// Free-form (key, value) notes: where the data comes from and under which licence.
    pub meta: Vec<(String, String)>,
}

impl MotionPack {
    pub fn meta(&self, key: &str) -> Option<&str> {
        self.meta.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let db = &self.database;
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        section(&mut out, b"META", |w| {
            w.u32(self.meta.len() as u32);
            for (k, v) in &self.meta {
                w.str(k);
                w.str(v);
            }
        });
        section(&mut out, b"SKEL", |w| w.skeleton(&db.skeleton));
        section(&mut out, b"ROLE", |w| {
            for j in [db.roles.root, db.roles.hips, db.roles.feet[0], db.roles.feet[1]] {
                w.u32(j as u32);
            }
            w.f32(db.sample_rate);
            let f = &db.weights;
            for x in [f.foot_position, f.foot_velocity, f.hips_velocity, f.trajectory_position, f.trajectory_direction] {
                w.f32(x);
            }
        });
        section(&mut out, b"CLIP", |w| {
            w.u32(db.clips.len() as u32);
            for c in &db.clips {
                w.str(&c.name);
                w.u32(c.start as u32);
                w.u32(c.frames as u32);
                w.u8(c.looping as u8);
                w.u32(c.tags);
            }
        });
        section(&mut out, b"ROTS", |w| {
            w.u32(db.rotations.len() as u32);
            for q in &db.rotations {
                for c in q {
                    w.bytes(&c.to_le_bytes());
                }
            }
        });
        section(&mut out, b"TRAN", |w| w.tracks(&db.translations));
        section(&mut out, b"SCAL", |w| w.tracks(&db.scales));
        section(&mut out, b"ROOT", |w| {
            w.u32(db.roots.len() as u32);
            for r in &db.roots {
                w.vec3(r.translation);
                w.quat(r.rotation);
            }
        });
        section(&mut out, b"CONT", |w| {
            w.u32(db.contacts.len() as u32);
            w.bytes(&db.contacts);
        });
        section(&mut out, b"FEAT", |w| {
            for x in db.feature_offset.iter().chain(&db.feature_scale) {
                w.f32(*x);
            }
            let frames = db.frame_count();
            w.u32(frames as u32);
            for f in 0..frames {
                for x in &db.features(f)[..FEATURES] {
                    w.f32(*x);
                }
            }
        });
        for m in &self.meshes {
            section(&mut out, b"MESH", |w| w.mesh(m));
        }
        if !self.actions.is_empty() {
            section(&mut out, b"ACTS", |w| {
                w.u32(self.actions.len() as u32);
                for a in &self.actions {
                    w.u32(a.clip as u32);
                    w.u8(a.kind as u8);
                    w.f32(a.height);
                    w.vec3(a.ledge);
                    w.vec3(a.forward);
                    for x in [a.rise, a.anchor, a.on_top, a.off_top, a.down, a.exit, a.span, a.last_entry] {
                        w.f32(x);
                    }
                }
            });
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 8 || &bytes[..4] != MAGIC {
            return Err("not a Kansei motion pack (.kmm)".into());
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(format!("motion pack version {version}, this build reads {VERSION}"));
        }
        let mut meta = Vec::new();
        let mut skeleton = None;
        let mut roles = None;
        let mut clips = Vec::new();
        let mut rotations = Vec::new();
        let (mut translations, mut scales) = (None, None);
        let mut roots = Vec::new();
        let mut contacts = Vec::new();
        let mut features = None;
        let mut meshes = Vec::new();
        let mut actions = Vec::new();
        let mut at = 8;
        while at < bytes.len() {
            let mut header = Reader { bytes, at };
            let tag: [u8; 4] = header.take(4)?.try_into().unwrap();
            let length = header.u64()? as usize;
            let start = header.at;
            let payload = bytes.get(start..start + length).ok_or("truncated motion pack")?;
            let mut r = Reader { bytes: payload, at: 0 };
            match &tag {
                b"META" => {
                    for _ in 0..r.u32()? {
                        meta.push((r.str()?, r.str()?));
                    }
                }
                b"SKEL" => skeleton = Some(r.skeleton()?),
                b"ROLE" => {
                    let j = [r.u32()?, r.u32()?, r.u32()?, r.u32()?].map(|x| x as usize);
                    let rate = r.f32()?;
                    let f = [r.f32()?, r.f32()?, r.f32()?, r.f32()?, r.f32()?];
                    let weights = FeatureWeights { foot_position: f[0], foot_velocity: f[1], hips_velocity: f[2], trajectory_position: f[3], trajectory_direction: f[4] };
                    roles = Some((JointRoles { root: j[0], hips: j[1], feet: [j[2], j[3]] }, rate, weights));
                }
                b"CLIP" => {
                    for _ in 0..r.u32()? {
                        clips.push(ClipInfo { name: r.str()?, start: r.u32()? as usize, frames: r.u32()? as usize, looping: r.u8()? != 0, tags: r.u32()? });
                    }
                }
                b"ROTS" => {
                    let n = r.u32()? as usize;
                    let raw = r.take(n * 8)?;
                    rotations = raw.chunks_exact(8).map(|c| std::array::from_fn(|k| i16::from_le_bytes([c[2 * k], c[2 * k + 1]]))).collect();
                }
                b"TRAN" => translations = Some(r.tracks()?),
                b"SCAL" => scales = Some(r.tracks()?),
                b"ROOT" => {
                    for _ in 0..r.u32()? {
                        roots.push(Transform::from_translation_rotation(r.vec3()?, r.quat()?));
                    }
                }
                b"CONT" => {
                    let n = r.u32()? as usize;
                    contacts = r.take(n)?.to_vec();
                }
                b"FEAT" => {
                    let mut offset = [0.0; FEATURES];
                    let mut scale = [0.0; FEATURES];
                    for x in offset.iter_mut().chain(scale.iter_mut()) {
                        *x = r.f32()?;
                    }
                    let frames = r.u32()? as usize;
                    let mut rows = Vec::with_capacity(frames * STRIDE);
                    for _ in 0..frames {
                        for _ in 0..FEATURES {
                            rows.push(r.f32()?);
                        }
                        rows.extend(std::iter::repeat_n(0.0, STRIDE - FEATURES));
                    }
                    features = Some((offset, scale, rows));
                }
                b"MESH" => meshes.push(r.mesh()?),
                b"ACTS" => {
                    for _ in 0..r.u32()? {
                        let clip = r.u32()? as usize;
                        let kind = ActionKind::from_u8(r.u8()?).ok_or("unknown action kind in the motion pack")?;
                        let height = r.f32()?;
                        let (ledge, forward) = (r.vec3()?, r.vec3()?);
                        let [rise, anchor, on_top, off_top, down, exit, span, last_entry] = r.f32s::<8>()?;
                        actions.push(ActionClip { clip, kind, height, ledge, forward, rise, anchor, on_top, off_top, down, exit, span, last_entry });
                    }
                }
                _ => {}
            }
            at = start + length;
        }
        let skeleton = skeleton.ok_or("the motion pack has no skeleton")?;
        let (roles, sample_rate, weights) = roles.ok_or("the motion pack has no joint roles")?;
        let (feature_offset, feature_scale, features) = features.ok_or("the motion pack has no features")?;
        let joints = skeleton.len();
        let frames = roots.len();
        let translations = translations.ok_or("the motion pack has no translations")?;
        let scales = scales.ok_or("the motion pack has no scales")?;
        if rotations.len() != frames * joints || contacts.len() != frames || features.len() != frames * STRIDE || translations.animated.len() != frames * translations.animated_joints {
            return Err("the motion pack's sections disagree on the frame count".into());
        }
        if clips.iter().any(|c| c.start + c.frames > frames) || [roles.root, roles.hips, roles.feet[0], roles.feet[1]].iter().any(|&j| j >= joints) {
            return Err("the motion pack's clips or roles are out of range".into());
        }
        if actions.iter().any(|a| a.clip >= clips.len()) {
            return Err("the motion pack's actions name clips it lacks".into());
        }
        for m in &meshes {
            if m.mesh.skin_joints.iter().any(|&j| j >= joints) {
                return Err(format!("mesh '{}' is skinned to joints the skeleton lacks", m.mesh.name));
            }
        }
        let mut database = Database { skeleton, roles, sample_rate, weights, clips, rotations, translations, scales, roots, contacts, feature_offset, feature_scale, features, bounds_small: Vec::new(), bounds_large: Vec::new() };
        database.build_bounds();
        Ok(Self { database, meshes, actions, meta })
    }
}

/// A named image, encoded (PNG, JPEG, WebP...): a character's textures.
#[derive(Debug, Clone, PartialEq)]
pub struct PackImage {
    pub name: String,
    /// Its media type, e.g. `image/webp`.
    pub mime: String,
    pub bytes: Vec<u8>,
}

/// A character to show a motion pack's animation on: its skeleton (the same joint names and axes
/// as the motion pack's, its own proportions: see `animation::retarget`), skinned meshes and
/// images.
#[derive(Debug, Clone)]
pub struct CharacterPack {
    pub skeleton: Skeleton,
    pub meshes: Vec<PackMesh>,
    pub images: Vec<PackImage>,
    pub meta: Vec<(String, String)>,
}

impl CharacterPack {
    pub fn meta(&self, key: &str) -> Option<&str> {
        self.meta.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    pub fn image(&self, name: &str) -> Option<&PackImage> {
        self.images.iter().find(|i| i.name == name)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        section(&mut out, b"META", |w| {
            w.u32(self.meta.len() as u32);
            for (k, v) in &self.meta {
                w.str(k);
                w.str(v);
            }
        });
        section(&mut out, b"SKEL", |w| w.skeleton(&self.skeleton));
        for m in &self.meshes {
            section(&mut out, b"MESH", |w| w.mesh(m));
        }
        for image in &self.images {
            section(&mut out, b"IMAG", |w| {
                w.str(&image.name);
                w.str(&image.mime);
                w.u32(image.bytes.len() as u32);
                w.bytes(&image.bytes);
            });
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 8 || &bytes[..4] != MAGIC {
            return Err("not a Kansei pack (.kmm)".into());
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(format!("pack version {version}, this build reads {VERSION}"));
        }
        let (mut skeleton, mut meshes, mut images, mut meta) = (None, Vec::new(), Vec::new(), Vec::new());
        let mut at = 8;
        while at < bytes.len() {
            let mut header = Reader { bytes, at };
            let tag: [u8; 4] = header.take(4)?.try_into().unwrap();
            let length = header.u64()? as usize;
            let start = header.at;
            let payload = bytes.get(start..start + length).ok_or("truncated pack")?;
            let mut r = Reader { bytes: payload, at: 0 };
            match &tag {
                b"META" => {
                    for _ in 0..r.u32()? {
                        meta.push((r.str()?, r.str()?));
                    }
                }
                b"SKEL" => skeleton = Some(r.skeleton()?),
                b"MESH" => meshes.push(r.mesh()?),
                b"IMAG" => {
                    let (name, mime) = (r.str()?, r.str()?);
                    let n = r.u32()? as usize;
                    images.push(PackImage { name, mime, bytes: r.take(n)?.to_vec() });
                }
                _ => {}
            }
            at = start + length;
        }
        let skeleton = skeleton.ok_or("the pack has no skeleton")?;
        if meshes.is_empty() {
            return Err("the pack has no mesh".into());
        }
        if meshes.iter().any(|m| m.mesh.skin_joints.iter().any(|&j| j >= skeleton.len())) {
            return Err("a mesh is skinned to joints the skeleton lacks".into());
        }
        Ok(Self { skeleton, meshes, images, meta })
    }
}

fn section(out: &mut Vec<u8>, tag: &[u8; 4], write: impl FnOnce(&mut Writer)) {
    let mut w = Writer(Vec::new());
    write(&mut w);
    out.extend_from_slice(tag);
    out.extend_from_slice(&(w.0.len() as u64).to_le_bytes());
    out.extend_from_slice(&w.0);
}

struct Writer(Vec<u8>);

impl Writer {
    fn bytes(&mut self, b: &[u8]) {
        self.0.extend_from_slice(b);
    }
    fn u8(&mut self, x: u8) {
        self.0.push(x);
    }
    fn u32(&mut self, x: u32) {
        self.bytes(&x.to_le_bytes());
    }
    fn i32(&mut self, x: i32) {
        self.bytes(&x.to_le_bytes());
    }
    fn f32(&mut self, x: f32) {
        self.bytes(&x.to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.bytes(s.as_bytes());
    }
    fn vec3(&mut self, v: Vec3) {
        v.to_array().iter().for_each(|x| self.f32(*x));
    }
    fn quat(&mut self, q: Quat) {
        q.to_array().iter().for_each(|x| self.f32(*x));
    }
    fn transform(&mut self, t: &Transform) {
        self.vec3(t.translation);
        self.quat(t.rotation);
        self.vec3(t.scale);
    }
    fn skeleton(&mut self, skeleton: &Skeleton) {
        self.u32(skeleton.len() as u32);
        for j in 0..skeleton.len() {
            self.str(&skeleton.names[j]);
            self.i32(skeleton.parents[j].map_or(-1, |p| p as i32));
            self.transform(&skeleton.rest[j]);
        }
    }
    fn tracks(&mut self, t: &Vec3Tracks) {
        self.u32(t.constant.len() as u32);
        for c in &t.constant {
            match c {
                Some(v) => {
                    self.u8(0);
                    self.vec3(*v);
                }
                None => self.u8(1),
            }
        }
        for (c, e) in t.center.iter().zip(&t.extent) {
            self.vec3(*c);
            self.vec3(*e);
        }
        self.u32(t.animated.len() as u32);
        for q in &t.animated {
            q.iter().for_each(|c| self.bytes(&c.to_le_bytes()));
        }
    }
    fn mesh(&mut self, m: &PackMesh) {
        let mesh = &m.mesh;
        self.str(&mesh.name);
        m.color.iter().for_each(|x| self.f32(*x));
        self.i32(mesh.material.map_or(-1, |x| x as i32));
        self.u32(mesh.vertices.len() as u32);
        for v in &mesh.vertices {
            v.position[..3].iter().chain(&v.normal).chain(&v.uv).for_each(|x| self.f32(*x));
        }
        // joints and weights as the shader reads them (weights in unorm16)
        for w in mesh.skin_words() {
            w.iter().for_each(|x| self.u32(*x));
        }
        self.u32(mesh.indices.len() as u32);
        mesh.indices.iter().for_each(|i| self.u32(*i));
        self.u32(mesh.skin_joints.len() as u32);
        for (j, m) in mesh.skin_joints.iter().zip(&mesh.inverse_bind) {
            self.u32(*j as u32);
            m.to_cols_array().iter().for_each(|x| self.f32(*x));
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let out = self.bytes.get(self.at..self.at + n).ok_or("truncated motion pack")?;
        self.at += n;
        Ok(out)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32s<const N: usize>(&mut self) -> Result<[f32; N], String> {
        let mut out = [0.0; N];
        for x in &mut out {
            *x = self.f32()?;
        }
        Ok(out)
    }
    fn str(&mut self) -> Result<String, String> {
        let n = self.u32()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| "a motion pack string is not UTF-8".to_string())
    }
    fn vec3(&mut self) -> Result<Vec3, String> {
        Ok(Vec3::new(self.f32()?, self.f32()?, self.f32()?))
    }
    fn quat(&mut self) -> Result<Quat, String> {
        Ok(Quat::from_xyzw(self.f32()?, self.f32()?, self.f32()?, self.f32()?))
    }
    fn transform(&mut self) -> Result<Transform, String> {
        Ok(Transform::new(self.vec3()?, self.quat()?, self.vec3()?))
    }
    fn skeleton(&mut self) -> Result<Skeleton, String> {
        let n = self.u32()? as usize;
        let (mut names, mut parents, mut rest) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            names.push(self.str()?);
            let p = self.i32()?;
            if p >= i as i32 {
                return Err(format!("joint {i} comes before its parent {p}"));
            }
            parents.push((p >= 0).then_some(p as usize));
            rest.push(self.transform()?);
        }
        Ok(Skeleton::new(names, parents, rest))
    }
    fn tracks(&mut self) -> Result<Vec3Tracks, String> {
        let joints = self.u32()? as usize;
        let mut constant = Vec::with_capacity(joints);
        let mut slot = Vec::with_capacity(joints);
        let mut animated_joints = 0;
        for _ in 0..joints {
            if self.u8()? == 0 {
                constant.push(Some(self.vec3()?));
                slot.push(u32::MAX);
            } else {
                constant.push(None);
                slot.push(animated_joints as u32);
                animated_joints += 1;
            }
        }
        let (mut center, mut extent) = (Vec::with_capacity(animated_joints), Vec::with_capacity(animated_joints));
        for _ in 0..animated_joints {
            center.push(self.vec3()?);
            extent.push(self.vec3()?);
        }
        let n = self.u32()? as usize;
        let animated = self.take(n * 6)?.chunks_exact(6).map(|c| [0, 1, 2].map(|k| i16::from_le_bytes([c[2 * k], c[2 * k + 1]]))).collect();
        Ok(Vec3Tracks { constant, slot, animated_joints, center, extent, animated })
    }
    fn mesh(&mut self) -> Result<PackMesh, String> {
        let name = self.str()?;
        let color = self.f32s::<4>()?;
        let material = self.i32()?;
        let n = self.u32()? as usize;
        let mut vertices = Vec::with_capacity(n);
        for _ in 0..n {
            let f: [f32; 8] = self.f32s()?;
            vertices.push(Vertex { position: [f[0], f[1], f[2], 1.0], normal: [f[3], f[4], f[5]], uv: [f[6], f[7]] });
        }
        let mut joints = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        for _ in 0..n {
            let w = [self.u32()?, self.u32()?, self.u32()?, self.u32()?];
            joints.push([w[0] as u16, (w[0] >> 16) as u16, w[1] as u16, (w[1] >> 16) as u16]);
            let q = [w[2] & 0xffff, w[2] >> 16, w[3] & 0xffff, w[3] >> 16];
            weights.push(std::array::from_fn::<f32, MAX_INFLUENCES, _>(|k| q[k] as f32 / 65535.0));
        }
        let count = self.u32()? as usize;
        let indices = (0..count).map(|_| self.u32()).collect::<Result<Vec<_>, _>>()?;
        if indices.iter().any(|&i| i as usize >= n) {
            return Err(format!("mesh '{name}' indexes past its vertices"));
        }
        let skin = self.u32()? as usize;
        let mut skin_joints = Vec::with_capacity(skin);
        let mut inverse_bind = Vec::with_capacity(skin);
        for _ in 0..skin {
            skin_joints.push(self.u32()? as usize);
            inverse_bind.push(Mat4::from_cols_array(&self.f32s::<16>()?));
        }
        if joints.iter().flatten().any(|&j| j as usize >= skin) {
            return Err(format!("mesh '{name}' references a joint beyond its skin"));
        }
        let mesh = SkinnedMesh { name, vertices, indices, joints, weights, skin_joints, inverse_bind, material: (material >= 0).then_some(material as usize) };
        Ok(PackMesh { mesh, color })
    }
}
