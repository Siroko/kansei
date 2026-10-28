//! The level geometry parkour moves use, from pieces a level tags (a glTF node's
//! `kansei_traversal` extra, or a layout built in code): edges to hang from and shimmy along,
//! bars to swing from, wall faces to run along, and platforms to jump between.
//!
//! - A `ledge` box's top edges are grab edges, each with its outward normal, linked round the
//!   box's top so a shimmy turns its (outer) corners. An edge with something solid against its
//!   face is left out, and the loop breaks there.
//! - A `bar` is the segment `kansei_bar_height` above the piece's base along its `kansei_bar_axis`
//!   (`x` or `z`, the piece's own axes), between the posts at its ends.
//! - A `wallrun` box's faces along `kansei_run_axis` can be run along: `kansei_run_faces` is
//!   `both`, or `+` / `-` for the face on that side of the other horizontal axis.
//! - A `platform` keeps its chain (`kansei_chain`, `kansei_chain_index`): a jump aims at the next
//!   one.
//!
//! Every piece is also an ordinary collider (the caller adds its box to the `CollisionWorld`), so
//! today's traversals (vault, mantle, climb) still see it.

use glam::{Quat, Vec3};

use crate::collision::{CollisionWorld, Obb};

/// What a piece is for (`kansei_traversal`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParkourTag {
    Vault,
    Hurdle,
    Mantle,
    Climb,
    Platform,
    Ledge,
    WallRun,
    Bar,
}

impl ParkourTag {
    pub const ALL: [ParkourTag; 8] = [ParkourTag::Vault, ParkourTag::Hurdle, ParkourTag::Mantle, ParkourTag::Climb, ParkourTag::Platform, ParkourTag::Ledge, ParkourTag::WallRun, ParkourTag::Bar];

    pub fn name(self) -> &'static str {
        match self {
            ParkourTag::Vault => "vault",
            ParkourTag::Hurdle => "hurdle",
            ParkourTag::Mantle => "mantle",
            ParkourTag::Climb => "climb",
            ParkourTag::Platform => "platform",
            ParkourTag::Ledge => "ledge",
            ParkourTag::WallRun => "wallrun",
            ParkourTag::Bar => "bar",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.name() == name)
    }
}

/// A piece's optional properties (the `kansei_*` extras besides `kansei_traversal`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PieceProps {
    pub chain: Option<String>,
    pub chain_index: Option<i32>,
    /// A bar's height above the piece's base (m), and the piece axis it runs along.
    pub bar_height: Option<f32>,
    pub bar_axis: Option<char>,
    /// A wall's run axis, and which faces run (`both`, `+`, `-`).
    pub run_axis: Option<char>,
    pub run_faces: Option<String>,
}

impl PieceProps {
    /// Set one property from its extra's name and value (as text); unknown names are ignored.
    pub fn set(&mut self, key: &str, value: &str) {
        let value = value.trim().trim_matches('"');
        let axis = |v: &str| v.chars().next().map(|c| c.to_ascii_lowercase()).filter(|c| matches!(c, 'x' | 'z'));
        match key {
            "kansei_chain" => self.chain = Some(value.to_string()),
            "kansei_chain_index" => self.chain_index = value.parse::<f32>().ok().map(|v| v as i32),
            "kansei_bar_height" => self.bar_height = value.parse().ok(),
            "kansei_bar_axis" => self.bar_axis = axis(value),
            "kansei_run_axis" => self.run_axis = axis(value),
            "kansei_run_faces" => self.run_faces = Some(value.to_string()),
            _ => {}
        }
    }
}

/// A tagged piece of the level: its box (world) and properties.
#[derive(Debug, Clone, PartialEq)]
pub struct ParkourPiece {
    pub name: String,
    pub tag: ParkourTag,
    pub obb: Obb,
    pub props: PieceProps,
}

impl ParkourPiece {
    pub fn new(name: &str, tag: ParkourTag, obb: Obb) -> Self {
        Self { name: name.to_string(), tag, obb, props: PieceProps::default() }
    }

    pub fn with_props(mut self, props: PieceProps) -> Self {
        self.props = props;
        self
    }

    /// A piece from a level object's extras (a glTF node's `extras`, as name and value text):
    /// `kansei_traversal` names its tag, the other `kansei_*` extras its properties. `None` when
    /// it has no tag this knows.
    pub fn from_extras<'a>(name: &str, obb: Obb, extras: impl IntoIterator<Item = (&'a str, &'a str)>) -> Option<Self> {
        let mut tag = None;
        let mut props = PieceProps::default();
        for (key, value) in extras {
            if key == "kansei_traversal" {
                tag = ParkourTag::from_name(value.trim().trim_matches('"'));
            } else {
                props.set(key, value);
            }
        }
        Some(Self::new(name, tag?, obb).with_props(props))
    }
}

/// An edge to hang from: from `a` to `b` (world, at the top), `normal` horizontal and pointing out
/// of the face below it; `prev` and `next` are the edges it turns onto at `a` and at `b`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrabEdge {
    pub a: Vec3,
    pub b: Vec3,
    pub normal: Vec3,
    pub prev: Option<usize>,
    pub next: Option<usize>,
    /// Height of the floor under the edge's face (the box's base).
    pub floor: f32,
}

impl GrabEdge {
    pub fn length(&self) -> f32 {
        self.a.distance(self.b)
    }

    /// Unit direction from `a` to `b`.
    pub fn direction(&self) -> Vec3 {
        (self.b - self.a).normalize_or_zero()
    }

    /// The point `s` metres from `a`.
    pub fn point(&self, s: f32) -> Vec3 {
        self.a + self.direction() * s
    }

    /// Distance from `a` along the edge of the point nearest `p` (clamped to the edge).
    pub fn project(&self, p: Vec3) -> f32 {
        (p - self.a).dot(self.direction()).clamp(0.0, self.length())
    }
}

/// A bar to swing from: the segment `a`-`b` (world), `radius` thick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bar {
    pub a: Vec3,
    pub b: Vec3,
    pub radius: f32,
    /// Height of the floor under it.
    pub floor: f32,
}

impl Bar {
    pub fn direction(&self) -> Vec3 {
        (self.b - self.a).normalize_or_zero()
    }

    pub fn length(&self) -> f32 {
        self.a.distance(self.b)
    }

    /// The point of the bar nearest `p`, and its distance from `a`.
    pub fn nearest(&self, p: Vec3) -> (Vec3, f32) {
        let s = (p - self.a).dot(self.direction()).clamp(0.0, self.length());
        (self.a + self.direction() * s, s)
    }
}

/// A wall face to run along: centred at `center` (on the face, half way up), `normal` out of it,
/// `along` horizontal on it; `half_length` either way of the centre, from `bottom` to `top`
/// (world heights).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunFace {
    pub center: Vec3,
    pub normal: Vec3,
    pub along: Vec3,
    pub half_length: f32,
    pub bottom: f32,
    pub top: f32,
}

impl RunFace {
    /// `p`'s distance out of the face (negative behind it) and along it from the centre.
    pub fn local(&self, p: Vec3) -> (f32, f32) {
        let d = p - self.center;
        (d.dot(self.normal), d.dot(self.along))
    }

    /// The point on the face `along` metres from the centre, at height `y`.
    pub fn point(&self, along: f32, y: f32) -> Vec3 {
        let p = self.center + self.along * along;
        Vec3::new(p.x, y, p.z)
    }
}

/// A platform to jump onto: its box and its chain.
#[derive(Debug, Clone, PartialEq)]
pub struct Platform {
    pub obb: Obb,
    pub chain: Option<String>,
    pub index: i32,
}

impl Platform {
    /// The top's height.
    pub fn top(&self) -> f32 {
        self.obb.bounds().1.y
    }

    /// Whether `p` (horizontally) is over the top, `margin` in from its sides.
    pub fn over(&self, p: Vec3, margin: f32) -> bool {
        let local = self.obb.rotation.conjugate() * (p - self.obb.center);
        local.x.abs() <= self.obb.half_extents.x - margin && local.z.abs() <= self.obb.half_extents.z - margin
    }

    /// The point of the top nearest `p`, `margin` in from its sides.
    pub fn nearest_on_top(&self, p: Vec3, margin: f32) -> Vec3 {
        let h = self.obb.half_extents;
        let local = self.obb.rotation.conjugate() * (p - self.obb.center);
        let m = |e: f32| (e - margin).max(0.0);
        let clamped = Vec3::new(local.x.clamp(-m(h.x), m(h.x)), h.y, local.z.clamp(-m(h.z), m(h.z)));
        self.obb.center + self.obb.rotation * clamped
    }
}

/// The parkour geometry of a level.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParkourFeatures {
    pub edges: Vec<GrabEdge>,
    pub bars: Vec<Bar>,
    pub faces: Vec<RunFace>,
    pub platforms: Vec<Platform>,
}

/// The unit axis of a box's rotation named `x` or `z`, and the other horizontal one.
fn axes(rotation: Quat, axis: char) -> (Vec3, Vec3) {
    let (x, z) = (rotation * Vec3::X, rotation * Vec3::Z);
    if axis == 'z' { (z, x) } else { (x, z) }
}

impl ParkourFeatures {
    /// The features of `pieces`; `world` (which holds their colliders too) tells which ledge
    /// edges have something against their face.
    pub fn from_pieces(pieces: &[ParkourPiece], world: &CollisionWorld) -> Self {
        let mut f = Self::default();
        for piece in pieces {
            let o = &piece.obb;
            let (lo, hi) = o.bounds();
            match piece.tag {
                ParkourTag::Ledge => f.add_ledge(o, lo.y, world),
                ParkourTag::Bar => {
                    let (along, _) = axes(o.rotation, piece.props.bar_axis.unwrap_or('x'));
                    let half = if piece.props.bar_axis == Some('z') { o.half_extents.z } else { o.half_extents.x };
                    let height = piece.props.bar_height.unwrap_or(hi.y - lo.y);
                    let mid = Vec3::new(o.center.x, lo.y + height, o.center.z);
                    // inside the posts at its ends
                    let reach = (half - 0.08).max(0.1);
                    f.bars.push(Bar { a: mid - along * reach, b: mid + along * reach, radius: 0.025, floor: lo.y });
                }
                ParkourTag::WallRun => {
                    let axis = piece.props.run_axis.unwrap_or('x');
                    let (along, across) = axes(o.rotation, axis);
                    let (half_length, half_thick) = if axis == 'z' { (o.half_extents.z, o.half_extents.x) } else { (o.half_extents.x, o.half_extents.z) };
                    let faces = piece.props.run_faces.as_deref().unwrap_or("both");
                    for sign in [1.0f32, -1.0] {
                        let wanted = match faces {
                            "+" | "+z" | "+x" | "positive" => sign > 0.0,
                            "-" | "-z" | "-x" | "negative" => sign < 0.0,
                            _ => true,
                        };
                        if wanted {
                            let normal = Vec3::new(across.x, 0.0, across.z).normalize() * sign;
                            let center = Vec3::new(o.center.x, (lo.y + hi.y) * 0.5, o.center.z) + normal * half_thick;
                            f.faces.push(RunFace { center, normal, along: Vec3::new(along.x, 0.0, along.z).normalize(), half_length, bottom: lo.y, top: hi.y });
                        }
                    }
                }
                ParkourTag::Platform => f.platforms.push(Platform { obb: *o, chain: piece.props.chain.clone(), index: piece.props.chain_index.unwrap_or(0) }),
                _ => {}
            }
        }
        f
    }

    /// A box's four top edges, linked round its top; an edge with something solid against its
    /// face (beyond the box itself) is left out.
    fn add_ledge(&mut self, o: &Obb, floor: f32, world: &CollisionWorld) {
        let h = o.half_extents;
        let top = o.center + o.rotation * Vec3::new(0.0, h.y, 0.0);
        // the corners counter-clockwise seen from above (+x, +z ... in the box's frame)
        let corners: Vec<Vec3> = [(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)].iter().map(|&(x, z)| top + o.rotation * Vec3::new(x * h.x, 0.0, z * h.z)).collect();
        let first = self.edges.len();
        let mut kept = [false; 4];
        for k in 0..4 {
            let (a, b) = (corners[k], corners[(k + 1) % 4]);
            let mid = (a + b) * 0.5;
            let out = Vec3::new(mid.x - top.x, 0.0, mid.z - top.z).normalize_or_zero();
            // room in front of the face, where a hanging body goes
            let probe = mid + out * 0.45 - Vec3::Y * 0.9;
            kept[k] = !world.overlap_sphere(probe, 0.2, u32::MAX);
            self.edges.push(GrabEdge { a, b, normal: out, prev: None, next: None, floor });
        }
        for k in 0..4 {
            let (i, n, p) = (first + k, first + (k + 1) % 4, first + (k + 3) % 4);
            self.edges[i].next = (kept[k] && kept[(k + 1) % 4]).then_some(n);
            self.edges[i].prev = (kept[k] && kept[(k + 3) % 4]).then_some(p);
        }
        // drop the blocked ones (keeping the links' indices right)
        let mut map = [None; 4];
        let mut out = Vec::new();
        for k in 0..4 {
            if kept[k] {
                map[k] = Some(first + out.len());
                out.push(self.edges[first + k]);
            }
        }
        let remap = |i: Option<usize>| i.and_then(|i| map[i - first]);
        for e in &mut out {
            e.next = remap(e.next);
            e.prev = remap(e.prev);
        }
        self.edges.truncate(first);
        self.edges.extend(out);
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty() && self.bars.is_empty() && self.faces.is_empty() && self.platforms.is_empty()
    }

    /// The edge nearest `p` within `max` metres: its index, the distance along it, the point.
    pub fn nearest_edge(&self, p: Vec3, max: f32) -> Option<(usize, f32, Vec3)> {
        self.edges
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let s = e.project(p);
                (i, s, e.point(s))
            })
            .filter(|(_, _, q)| q.distance(p) <= max)
            .min_by(|a, b| a.2.distance(p).total_cmp(&b.2.distance(p)))
    }

    /// Walk `ds` metres along the edges from edge `edge` at `s`, turning onto linked edges at
    /// their ends; stops at an end with no link. Returns the edge and distance reached.
    pub fn advance_edge(&self, mut edge: usize, mut s: f32, mut ds: f32) -> (usize, f32) {
        for _ in 0..8 {
            let e = &self.edges[edge];
            let target = s + ds;
            if target > e.length() {
                match e.next {
                    Some(n) => {
                        ds = target - e.length();
                        edge = n;
                        s = 0.0;
                    }
                    None => return (edge, e.length()),
                }
            } else if target < 0.0 {
                match e.prev {
                    Some(p) => {
                        ds = target;
                        edge = p;
                        s = self.edges[p].length();
                    }
                    None => return (edge, 0.0),
                }
            } else {
                return (edge, target);
            }
        }
        (edge, s)
    }

    /// The point `ds` metres along the edges from `(edge, s)`, and the edge it lies on.
    pub fn edge_point(&self, edge: usize, s: f32, ds: f32) -> (Vec3, usize) {
        let (e, t) = self.advance_edge(edge, s, ds);
        (self.edges[e].point(t), e)
    }

    /// The outward normal at `s` on `edge`, turning smoothly round its corners: within `round`
    /// metres of an end that links to another edge, it blends toward that edge's normal (half
    /// way at the corner).
    pub fn edge_normal(&self, edge: usize, s: f32, round: f32) -> Vec3 {
        let e = &self.edges[edge];
        let blend = |other: usize, d: f32| {
            let x = (1.0 - d / round).clamp(0.0, 1.0);
            let t = 0.5 * x * x * (3.0 - 2.0 * x);
            e.normal.lerp(self.edges[other].normal, t).normalize_or(e.normal)
        };
        match (e.next, e.prev) {
            (Some(n), _) if e.length() - s < round => blend(n, e.length() - s),
            (_, Some(p)) if s < round => blend(p, s),
            _ => e.normal,
        }
    }

    /// The next platform in `platform`'s chain, if any.
    pub fn next_in_chain(&self, platform: usize) -> Option<usize> {
        let p = &self.platforms[platform];
        let chain = p.chain.as_ref()?;
        self.platforms.iter().position(|q| q.chain.as_ref() == Some(chain) && q.index == p.index + 1)
    }

    /// The platform whose top `p` stands on (within `tolerance` of its height).
    pub fn platform_under(&self, p: Vec3, tolerance: f32) -> Option<usize> {
        self.platforms.iter().position(|q| q.over(p, 0.0) && (q.top() - p.y).abs() <= tolerance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tower() -> (ParkourPiece, CollisionWorld) {
        let obb = Obb::new(Vec3::new(14.4, 1.4, 8.0), Vec3::new(1.0, 1.4, 1.0), Quat::IDENTITY);
        let mut world = CollisionWorld::new();
        world.add_box(obb);
        (ParkourPiece::new("Course_Tower", ParkourTag::Ledge, obb), world)
    }

    #[test]
    fn a_ledge_box_gives_four_linked_top_edges() {
        let (piece, world) = tower();
        let f = ParkourFeatures::from_pieces(&[piece], &world);
        assert_eq!(f.edges.len(), 4);
        for (i, e) in f.edges.iter().enumerate() {
            assert!((e.a.y - 2.8).abs() < 1e-5 && (e.b.y - 2.8).abs() < 1e-5);
            assert!((e.length() - 2.0).abs() < 1e-5);
            // the normal points out of the box, across the edge
            let mid = (e.a + e.b) * 0.5;
            assert!((mid - Vec3::new(14.4, 2.8, 8.0)).dot(e.normal) > 0.9);
            assert!(e.direction().dot(e.normal).abs() < 1e-5);
            assert_eq!(f.edges[e.next.unwrap()].prev, Some(i));
            assert_eq!(e.b, f.edges[e.next.unwrap()].a);
        }
    }

    #[test]
    fn walking_an_edge_turns_its_corners_round_the_top() {
        let (piece, world) = tower();
        let f = ParkourFeatures::from_pieces(&[piece], &world);
        // all the way round is back where it started
        let (e, s) = f.advance_edge(0, 0.5, 8.0);
        assert_eq!((e, (s * 1e3).round()), (0, 500.0));
        let (e, s) = f.advance_edge(0, 0.5, -1.0);
        assert_eq!(e, f.edges[0].prev.unwrap());
        assert!((s - 1.5).abs() < 1e-5);
        // the normal turns smoothly: half way at the corner
        let at_corner = f.edge_normal(0, 2.0, 0.3);
        let n = f.edges[f.edges[0].next.unwrap()].normal;
        assert!((at_corner.dot(f.edges[0].normal) - at_corner.dot(n)).abs() < 1e-3);
        assert_eq!(f.edge_normal(0, 1.0, 0.3), f.edges[0].normal);
    }

    #[test]
    fn an_edge_against_a_wall_is_left_out_and_breaks_the_loop() {
        let (piece, mut world) = tower();
        // a wall against the tower's +x face
        world.add_box(Obb::new(Vec3::new(15.6, 1.5, 8.0), Vec3::new(0.2, 1.5, 3.0), Quat::IDENTITY));
        let f = ParkourFeatures::from_pieces(&[piece], &world);
        assert_eq!(f.edges.len(), 3);
        assert!(f.edges.iter().all(|e| e.normal.x < 0.9));
        let ends = f.edges.iter().filter(|e| e.next.is_none() || e.prev.is_none()).count();
        assert_eq!(ends, 2);
        // a shimmy stops at the broken end
        let start = f.edges.iter().position(|e| e.next.is_none()).unwrap();
        let (e, s) = f.advance_edge(start, 0.0, 5.0);
        assert_eq!(e, start);
        assert!((s - f.edges[start].length()).abs() < 1e-5);
    }

    #[test]
    fn pieces_come_from_a_nodes_extras() {
        let obb = Obb::new(Vec3::new(7.6, 1.17, 8.4), Vec3::new(1.14, 1.17, 0.04), Quat::IDENTITY);
        let piece = ParkourPiece::from_extras("Course_Bar_1", obb, [("kansei_traversal", "\"bar\""), ("kansei_bar_height", "2.3"), ("kansei_bar_axis", "x"), ("kansei_source", "made here")]).unwrap();
        assert_eq!(piece.tag, ParkourTag::Bar);
        assert_eq!(piece.props.bar_height, Some(2.3));
        assert_eq!(piece.props.bar_axis, Some('x'));
        let platform = ParkourPiece::from_extras("P", obb, [("kansei_traversal", "platform"), ("kansei_chain", "south"), ("kansei_chain_index", "2")]).unwrap();
        assert_eq!((platform.props.chain.as_deref(), platform.props.chain_index), (Some("south"), Some(2)));
        assert!(ParkourPiece::from_extras("Sofa", obb, [("kansei_source", "polyhaven")]).is_none());
        assert!(ParkourPiece::from_extras("X", obb, [("kansei_traversal", "trampoline")]).is_none());
    }

    #[test]
    fn bars_walls_and_platform_chains() {
        let world = CollisionWorld::new();
        let mut bar = ParkourPiece::new("Course_Bar_1", ParkourTag::Bar, Obb::new(Vec3::new(7.6, 1.17, 8.4), Vec3::new(1.14, 1.17, 0.04), Quat::IDENTITY));
        bar.props.set("kansei_bar_height", "2.3");
        bar.props.set("kansei_bar_axis", "x");
        let mut wall = ParkourPiece::new("Course_Wall", ParkourTag::WallRun, Obb::new(Vec3::new(10.0, 1.6, 3.0), Vec3::new(2.5, 1.6, 0.12), Quat::IDENTITY));
        wall.props.set("kansei_run_faces", "both");
        wall.props.set("kansei_run_axis", "x");
        let platform = |k: i32, x: f32, h: f32| {
            let mut p = ParkourPiece::new("Course_Platform", ParkourTag::Platform, Obb::new(Vec3::new(x, h * 0.5, 11.6), Vec3::new(0.8, h * 0.5, 0.8), Quat::IDENTITY));
            p.props.set("kansei_chain", "south");
            p.props.set("kansei_chain_index", &k.to_string());
            p
        };
        let f = ParkourFeatures::from_pieces(&[bar, wall, platform(0, 5.2, 1.0), platform(1, 7.8, 1.2)], &world);
        assert_eq!(f.bars.len(), 1);
        let b = f.bars[0];
        assert!((b.a.y - 2.3).abs() < 1e-5 && (b.direction() - Vec3::X).length() < 1e-5);
        assert!((b.length() - 2.12).abs() < 1e-3);
        assert_eq!(f.faces.len(), 2);
        assert!(f.faces.iter().any(|w| w.normal.z > 0.99 && (w.center.z - 3.12).abs() < 1e-5));
        assert!(f.faces.iter().all(|w| (w.along.x.abs() - 1.0).abs() < 1e-5 && (w.half_length - 2.5).abs() < 1e-5 && (w.top - 3.2).abs() < 1e-5));
        assert_eq!(f.next_in_chain(0), Some(1));
        assert_eq!(f.next_in_chain(1), None);
        assert_eq!(f.platform_under(Vec3::new(5.3, 1.0, 11.5), 0.1), Some(0));
        assert!((f.platforms[1].top() - 1.2).abs() < 1e-5);
    }
}
