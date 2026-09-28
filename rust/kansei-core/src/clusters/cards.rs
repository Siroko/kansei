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
