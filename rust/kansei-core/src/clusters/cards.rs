//! Cards: foliage's small, open, flat-ish pieces (sprays, leaves, ribbons), which edge collapse
//! can't reduce (every vertex is on a border). Their coarser levels are pruned instead: see
//! `ClusterMesh::build` and the card rounds below.

use glam::Vec3;
use std::collections::HashMap;

use super::ClusterOptions;

/// A card: its vertices (into the mesh's) and its triangles (into those), its area-weighted
/// centroid, its area, and its radius from the centroid.
#[derive(Clone, Debug)]
pub(super) struct Card {
    pub vertices: Vec<u32>,
    pub triangles: Vec<[u16; 3]>,
    pub centroid: Vec3,
    pub area: f32,
    pub radius: f32,
}

/// The triangles of `indices` split into cards and the rest (as indices), by connected component
/// over `position_ids` (welded positions): a card has at most `card_max_triangles` triangles and
/// `max_vertices` vertices, is open (an edge used once), and keeps `card_flatness` of its area in
/// its area-weighted normal.
pub(super) fn find_cards(indices: &[u32], positions: &[f32], position_ids: &[u32], options: &ClusterOptions) -> (Vec<Card>, Vec<u32>) {
    let p = |v: u32| Vec3::from_slice(&positions[v as usize * 3..v as usize * 3 + 3]);
    // components: union-find over positions
    let mut parent: Vec<u32> = (0..position_ids.len() as u32).collect();
    fn root(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }
    for t in indices.chunks(3) {
        let a = root(&mut parent, position_ids[t[0] as usize]);
        for &v in &t[1..] {
            let b = root(&mut parent, position_ids[v as usize]);
            parent[b as usize] = a;
        }
    }
    let mut components: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, t) in indices.chunks(3).enumerate() {
        components.entry(root(&mut parent, position_ids[t[0] as usize])).or_default().push(i);
    }
    let mut roots: Vec<u32> = components.keys().copied().collect();
    roots.sort_unstable();
    let (mut cards, mut rest) = (Vec::new(), Vec::new());
    for r in roots {
        let triangles = &components[&r];
        let tri = |i: usize| [indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]];
        let mut vertices: Vec<u32> = triangles.iter().flat_map(|&i| tri(i)).collect();
        vertices.sort_unstable();
        vertices.dedup();
        let (mut area, mut normal, mut centroid) = (0.0f32, Vec3::ZERO, Vec3::ZERO);
        let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
        for &i in triangles {
            let [a, b, c] = tri(i);
            let n = (p(b) - p(a)).cross(p(c) - p(a));
            let ta = 0.5 * n.length();
            area += ta;
            normal += n * 0.5;
            centroid += (p(a) + p(b) + p(c)) / 3.0 * ta;
            for (x, y) in [(a, b), (b, c), (c, a)] {
                let (x, y) = (position_ids[x as usize], position_ids[y as usize]);
                *edges.entry((x.min(y), x.max(y))).or_default() += 1;
            }
        }
        let open = edges.values().any(|&n| n == 1);
        let is_card = triangles.len() <= options.card_max_triangles && vertices.len() <= options.max_vertices && open && area > 0.0 && normal.length() >= options.card_flatness * area;
        if !is_card {
            rest.extend(triangles.iter().flat_map(|&i| tri(i)));
            continue;
        }
        let centroid = centroid / area;
        let local: HashMap<u32, u16> = vertices.iter().enumerate().map(|(j, &v)| (v, j as u16)).collect();
        let radius = vertices.iter().map(|&v| p(v).distance(centroid)).fold(0.0, f32::max);
        cards.push(Card { triangles: triangles.iter().map(|&i| tri(i).map(|v| local[&v])).collect(), vertices, centroid, area, radius });
    }
    (cards, rest)
}

use super::{ClusterMesh, Sphere};
use crate::geometries::Vertex;

/// A card at some level: which card, where its vertices are (u32::MAX: the card's own; else the
/// first of its scaled copy's, in the card's vertex order), how many level-0 cards it stands for,
/// the area it covers, and where (their area-weighted centre, where it is drawn), and the box
/// round those level-0 cards (the outline grown cards should keep within).
#[derive(Clone, Copy, Debug)]
pub(super) struct Placed {
    pub card: u32,
    pub first_vertex: u32,
    pub represents: u32,
    pub area: f32,
    pub center: Vec3,
    pub region: CardBox,
}

impl Placed {
    fn vertex(&self, card: &Card, j: usize) -> u32 {
        if self.first_vertex == u32::MAX { card.vertices[j] } else { self.first_vertex + j as u32 }
    }
}

/// A 30-bit Morton code of `p` in `(lo, hi)`.
pub(super) fn morton(p: Vec3, lo: Vec3, hi: Vec3) -> u32 {
    let q = ((p - lo) / (hi - lo).max(Vec3::splat(1e-9)) * 1023.0).clamp(Vec3::ZERO, Vec3::splat(1023.0)).as_uvec3();
    let spread = |mut x: u32| {
        x = (x | (x << 16)) & 0x0300_00ff;
        x = (x | (x << 8)) & 0x0300_f00f;
        x = (x | (x << 4)) & 0x030c_30c3;
        (x | (x << 2)) & 0x0924_9249
    };
    spread(q.x) | spread(q.y) << 1 | spread(q.z) << 2
}

fn hash(a: u32, b: u32) -> u32 {
    let x = a.wrapping_mul(0x9E37_79B1) ^ b.wrapping_mul(0x85EB_CA77);
    (x ^ (x >> 15)).wrapping_mul(0x2C1B_3C6D)
}

/// Whole cards packed into clusters, in the order given (neighbours: Morton order), each up to
/// the options' vertex and triangle limits, with `error`, `lod_bounds` (their own when None) and
/// `level`; per new cluster, its index and its cards.
#[allow(clippy::too_many_arguments)]
pub(super) fn pack(mesh: &mut ClusterMesh, cards: &[Card], placed: &[Placed], positions: &[f32], error: f32, lod_bounds: Option<Sphere>, level: u32, options: &super::ClusterOptions) -> Vec<(usize, Vec<Placed>)> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < placed.len() {
        let (mut v, mut t, mut end) = (0, 0, start);
        while end < placed.len() {
            let card = &cards[placed[end].card as usize];
            if end > start && (v + card.vertices.len() > options.max_vertices || t + card.triangles.len() > options.max_triangles) {
                break;
            }
            v += card.vertices.len();
            t += card.triangles.len();
            end += 1;
        }
        let (mut vertices, mut triangles) = (Vec::with_capacity(v), Vec::with_capacity(t * 3));
        for p in &placed[start..end] {
            let card = &cards[p.card as usize];
            let base = vertices.len() as u8;
            vertices.extend((0..card.vertices.len()).map(|j| p.vertex(card, j)));
            triangles.extend(card.triangles.iter().flat_map(|t| t.map(|j| base + j as u8)));
        }
        let index = mesh.push_cluster(&vertices, &triangles, positions, error, lod_bounds, level, true);
        out.push((index, placed[start..end].to_vec()));
        start = end;
    }
    out
}

/// One round over the card clusters still without a parent (`pending`): grouped in Morton order,
/// each group's cards pruned to one of each neighbouring pair (the kept one scaled to cover its
/// pair's area: new vertices, appended to the mesh and to `positions`), the children given their
/// parent's error and sphere, and the kept cards packed into this level's clusters. Returns the
/// next pending set and whether any group was pruned.
pub(super) fn prune_round(mesh: &mut ClusterMesh, cards: &[Card], pending: Vec<(usize, Vec<Placed>)>, positions: &mut Vec<f32>, level: u32, options: &super::ClusterOptions) -> (Vec<(usize, Vec<Placed>)>, bool) {
    let (lo, hi) = bounds_of(positions);
    let mut pending = pending;
    pending.sort_by_key(|(c, _)| morton(mesh.clusters[*c].bounds.center, lo, hi));
    let mut next = Vec::new();
    let mut progress = false;
    for group in pending.chunks(options.group_size.max(2)) {
        let mut placed: Vec<Placed> = group.iter().flat_map(|(_, p)| p.iter().copied()).collect();
        if placed.len() < 2 {
            next.extend(group.iter().cloned());
            continue;
        }
        placed.sort_by_key(|p| morton(p.center, lo, hi));
        let (area, n0) = placed.iter().fold((0.0, 0), |(a, n), p| (a + p.area, n + p.represents));
        let mut kept = Vec::with_capacity(placed.len().div_ceil(2));
        // how far the grown cards reach past the level-0 cards they stand for
        let mut protrusion = 0.0f32;
        for (a, b) in pairs(&placed) {
            let Some(b) = b else {
                kept.push(a);
                continue;
            };
            // the one kept: either, by a hash (deterministic, uncorrelated with place)
            let (a, b) = if hash(a.card, level) <= hash(b.card, level) { (a, b) } else { (b, a) };
            let card = &cards[a.card as usize];
            let (represents, covered) = (a.represents + b.represents, a.area + b.area);
            let scale = (covered / card.area).sqrt();
            if scale > options.card_max_scale {
                // as far as this card goes: both stay
                kept.extend([a, b]);
                continue;
            }
            // drawn at the centre of the area it stands for, scaled to cover it
            let center = (a.center * a.area + b.center * b.area) / covered;
            let region = a.region.union(&b.region);
            let first_vertex = mesh.vertices.len() as u32;
            for &v in &card.vertices {
                let mut vertex: Vertex = mesh.vertices[v as usize];
                let p = center + (Vec3::from_slice(&vertex.position[..3]) - card.centroid) * scale;
                protrusion = protrusion.max(region.distance(p));
                vertex.position[..3].copy_from_slice(&p.to_array());
                mesh.vertices.push(vertex);
                positions.extend_from_slice(&p.to_array());
            }
            kept.push(Placed { card: a.card, first_vertex, represents, area: covered, center, region });
        }
        if kept.len() == placed.len() {
            // nothing came off: next round, grouped with other neighbours
            next.extend(group.iter().cloned());
            continue;
        }
        progress = true;
        let k = kept.len() as f32;
        // the crown thinned (scaled: how soon a crown may thin is a choice), or its outline moved
        // (in full: a spire rounded off is seen as it is)
        let thinned = options.card_error_scale * ((area / k).sqrt() - (area / n0 as f32).sqrt());
        let own = thinned.max(protrusion).max(0.0);
        let error = group.iter().map(|(c, _)| mesh.clusters[*c].error).fold(own, f32::max);
        let bounds = Sphere::enclosing(group.iter().map(|(c, _)| mesh.clusters[*c].lod_bounds));
        for (c, _) in group {
            mesh.clusters[*c].parent_error = error;
            mesh.clusters[*c].parent_bounds = bounds;
        }
        next.extend(pack(mesh, cards, &kept, positions, error, Some(bounds), level, options));
    }
    (next, progress)
}

/// Neighbouring pairs of `placed` (in Morton order): each card with the nearest still free among
/// the next few; an odd one alone.
fn pairs(placed: &[Placed]) -> Vec<(Placed, Option<Placed>)> {
    const WINDOW: usize = 8;
    let mut taken = vec![false; placed.len()];
    let mut out = Vec::with_capacity(placed.len().div_ceil(2));
    for i in 0..placed.len() {
        if taken[i] {
            continue;
        }
        taken[i] = true;
        let nearest = (i + 1..placed.len().min(i + 1 + WINDOW)).filter(|&j| !taken[j]).min_by(|&j, &k| {
            placed[i].center.distance_squared(placed[j].center).total_cmp(&placed[i].center.distance_squared(placed[k].center))
        });
        match nearest {
            Some(j) => {
                taken[j] = true;
                out.push((placed[i], Some(placed[j])));
            }
            None => out.push((placed[i], None)),
        }
    }
    out
}

/// An axis-aligned box round cards.
#[derive(Clone, Copy, Debug)]
pub(super) struct CardBox {
    lo: Vec3,
    hi: Vec3,
}

impl CardBox {
    fn of(points: impl Iterator<Item = Vec3>) -> Self {
        let (lo, hi) = points.fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), p| (lo.min(p), hi.max(p)));
        Self { lo, hi }
    }

    fn union(&self, other: &Self) -> Self {
        Self { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }

    /// How far `p` lies outside it (0 inside).
    fn distance(&self, p: Vec3) -> f32 {
        (self.lo - p).max(p - self.hi).max(Vec3::ZERO).length()
    }
}

fn bounds_of(positions: &[f32]) -> (Vec3, Vec3) {
    positions.chunks(3).fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), p| (lo.min(Vec3::from_slice(p)), hi.max(Vec3::from_slice(p))))
}

/// The cards at level 0: each on its own, in Morton order of their centroids, packed into
/// clusters.
pub(super) fn level_zero(mesh: &mut ClusterMesh, cards: &[Card], positions: &[f32], options: &super::ClusterOptions) -> Vec<(usize, Vec<Placed>)> {
    if cards.is_empty() {
        return Vec::new();
    }
    let (lo, hi) = bounds_of(positions);
    let mut order: Vec<u32> = (0..cards.len() as u32).collect();
    order.sort_by_key(|&c| morton(cards[c as usize].centroid, lo, hi));
    let placed: Vec<Placed> = order.iter().map(|&c| Placed { card: c, first_vertex: u32::MAX, represents: 1, area: cards[c as usize].area, center: cards[c as usize].centroid, region: CardBox::of(cards[c as usize].vertices.iter().map(|&v| Vec3::from_slice(&positions[v as usize * 3..v as usize * 3 + 3]))) }).collect();
    pack(mesh, cards, &placed, positions, 0.0, None, 0, options)
}
