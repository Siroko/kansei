//! The character both pages play: its motion pack's database and controller, the bodies it can be
//! shown as (the pack's own mesh, a character pack's), and its debug markers. How a body is drawn
//! is the page's ([`BodyLook`]): the lake lights it by the sun, the room writes the GBuffer its
//! global illumination reads.

use glam::Vec3 as GVec3;

use kansei_core::animation::motion_matching::pack::{CharacterPack, MotionPack};
use kansei_core::animation::motion_matching::traversal::CharacterController;
use kansei_core::animation::motion_matching::{Database, MotionMatcher, MotionMatchingSettings, ACTION_TAG};
use kansei_core::animation::retarget::Retarget;
use kansei_core::animation::{skinned_lit_material, skinned_lit_textured_material, BonePalette, Skeleton, SkinTextures, SkinnedLitParams, SkinnedMesh};
use kansei_core::buffers::Texture;
use kansei_core::debug::DebugBoxes;
use kansei_core::materials::Material;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::Renderer;
use kansei_wasm_lake::{SKY, SUN, SUN_DIR};

/// Paces of the walk and run loops (m/s) forward, sideways and backward: strafing moves at the
/// pace of the loop for its direction relative to the facing. `walk=` and `run=` scale them.
pub const WALK: [f32; 3] = [2.0, 1.8, 1.5];
pub const RUN: [f32; 3] = [5.0, 3.5, 3.0];

/// The pace for moving along `direction` (unit, local to the facing: x sideways, y forward) on
/// an ellipse through the forward, sideways and backward paces.
pub fn pace(paces: [f32; 3], direction: [f32; 2]) -> f32 {
    let along = if direction[1] >= 0.0 { paces[0] } else { paces[2] };
    1.0 / ((direction[1] / along).powi(2) + (direction[0] / paces[1]).powi(2)).sqrt().max(1e-6)
}

/// How a page draws the character's bodies: the motion pack's mesh in its colour, and a character
/// pack's with its textures. Both skin in their own `vertex_main` from the palette at
/// `animation::PALETTE_BINDING` and write motion vectors.
pub trait BodyLook {
    fn plain(&self, label: &str, color: [f32; 4], mesh: &SkinnedMesh, palette: &BonePalette) -> Material;
    fn textured(&self, label: &str, color: [f32; 4], mesh: &SkinnedMesh, palette: &BonePalette, textures: SkinTextures) -> Material;
}

/// The lake's look: lit by its sun (cascade-shadowed) and sky (`animation::skinned_lit_material`).
pub struct SunLit;

impl SunLit {
    fn params(color: [f32; 4]) -> SkinnedLitParams {
        SkinnedLitParams { base_color: color, sun_direction: [SUN_DIR[0], SUN_DIR[1], SUN_DIR[2], 0.0], sun: [SUN[0], SUN[1], SUN[2], 0.0], sky: [SKY[0], SKY[1], SKY[2], 0.0] }
    }
}

impl BodyLook for SunLit {
    fn plain(&self, label: &str, color: [f32; 4], mesh: &SkinnedMesh, palette: &BonePalette) -> Material {
        skinned_lit_material(label, Self::params(color), mesh, palette)
    }

    fn textured(&self, label: &str, color: [f32; 4], mesh: &SkinnedMesh, palette: &BonePalette, textures: SkinTextures) -> Material {
        skinned_lit_textured_material(label, Self::params(color), mesh, palette, textures)
    }
}

/// A mesh the character can be shown as: the motion pack's own, on the database's skeleton, or a
/// character pack's, the pose retargeted onto its skeleton.
pub struct Body {
    pub name: &'static str,
    pub mesh: SkinnedMesh,
    pub palette: BonePalette,
    pub index: usize,
    pub display: Option<(Skeleton, Retarget)>,
}

/// A character pack as a body: its textured mesh, hidden until shown.
fn hero_body(scene: &mut Scene, pack: CharacterPack, db: &Database, look: &dyn BodyLook) -> Result<Body, String> {
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
    let material = look.textured("Hero", first.color, &mesh, &palette, textures);
    let mut r = Renderable::new(mesh.geometry(), material);
    r.dynamic = true;
    r.visible = false;
    let index = scene.add(SceneNode::Renderable(r));
    log::info!("character pack: {} joints, {} vertices, {} triangles", pack.skeleton.len(), mesh.vertices.len(), mesh.indices.len() / 3);
    Ok(Body { name: "hero", mesh, palette, index, display: Some((pack.skeleton, retarget)) })
}

/// The character: its database, matcher, bodies, and the debug markers.
pub struct Character {
    pub db: Database,
    pub controller: CharacterController,
    pub bodies: Vec<Body>,
    /// The last obstacle found, marked at its ledge.
    pub ledge: DebugBoxes,
    pub showing: usize,
    pub bones: DebugBoxes,
    pub trajectory: DebugBoxes,
    /// Tag bits the search may use while walking and while running (all when the pack has no
    /// gait tags or `gait=0`).
    pub walk_tags: u32,
    pub run_tags: u32,
}

impl Character {
    pub fn new(renderer: &Renderer, scene: &mut Scene, pack: MotionPack, hero: Option<CharacterPack>, gait: bool, look: &dyn BodyLook) -> Result<Self, String> {
        let tags: Vec<String> = pack.meta("tags").unwrap_or("").split(',').map(str::to_string).collect();
        let MotionPack { database: db, meshes, actions, .. } = pack;
        // the pack's own mesh, when it has one (a pack may ship without, for a character pack's body)
        let mut bodies = Vec::new();
        if let Some(first) = meshes.into_iter().next() {
            let mesh = first.mesh;
            let mut palette = BonePalette::new(mesh.skin_joints.len());
            palette.update(&mesh, &db.skeleton.rest_model());
            let mut r = Renderable::new(mesh.geometry(), look.plain("Character", first.color, &mesh, &palette));
            r.dynamic = true;
            let index = scene.add(SceneNode::Renderable(r));
            bodies.push(Body { name: "the pack's mesh", mesh, palette, index, display: None });
        }
        if let Some(pack) = hero {
            match hero_body(scene, pack, &db, look) {
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
    pub fn show(&mut self, scene: &mut Scene, which: usize) {
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
    pub fn joint(&self, j: usize) -> usize {
        self.controller.matcher.output_skeleton(&self.db).find(&self.db.skeleton.names[j]).unwrap_or(0)
    }

    /// The legs as capsules (world ends, radius): each thigh, shin and foot (hip to knee to ankle
    /// to toe, up the foot's parents and down to its first child), and the hips.
    pub fn leg_capsules(&self) -> Vec<(GVec3, GVec3, f32)> {
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

    /// The output skeleton's bones in the world: (joint, its parent's position, its own), for each
    /// joint whose parent is not the root.
    pub fn bones_world(&self) -> Vec<(usize, GVec3, GVec3)> {
        let matcher = &self.controller.matcher;
        let (character, model, skeleton) = (matcher.character(), matcher.model(), matcher.output_skeleton(&self.db));
        let root = self.joint(self.db.roles.root);
        skeleton
            .parents
            .iter()
            .enumerate()
            .filter_map(|(j, parent)| parent.filter(|p| *p != root).map(|p| (j, character.transform_point(model[p].translation), character.transform_point(model[j].translation))))
            .collect()
    }

    /// The output skeleton's name for joint `j`.
    pub fn joint_name(&self, j: usize) -> &str {
        &self.controller.matcher.output_skeleton(&self.db).names[j]
    }
}
