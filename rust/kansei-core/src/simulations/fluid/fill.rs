//! Initial fills for a fluid: particles on a lattice, so it starts at rest density instead of
//! collapsing or bursting on its first steps. `PlanarContainerShape::lattice` fills a shaped
//! container up to a level; [`fill_box`] spreads a count of particles through a box.

use super::container::PlanarContainerShape;

/// `count` particles (4 floats each, w = 1) on a cubic lattice through the box `lo`..`hi`, its
/// spacing chosen so they fill it, each coordinate moved by up to `jitter` / 2 of a spacing
/// (the same pseudo-random offsets every call) so the lattice does not stay a crystal. When the
/// rounding leaves the lattice short of `count`, further layers fill in half a cell higher.
///
/// The rows nearest +z come first: a renderer that draws particles in their order, seen from
/// +z, then draws them roughly front to back, and the depth test spares the shading of those
/// behind (see AGENTS.md on Apple GPUs).
pub fn fill_box(count: usize, lo: [f32; 3], hi: [f32; 3], jitter: f32) -> Vec<f32> {
    let size: [f32; 3] = std::array::from_fn(|i| hi[i] - lo[i]);
    let spacing = (size[0] * size[1] * size[2] / count.max(1) as f32).cbrt();
    let cells: [usize; 3] = std::array::from_fn(|i| ((size[i] / spacing).floor() as usize).max(1));
    let mut rng: u32 = 12345;
    let mut offset = || {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        ((rng >> 8) as f32 / 16777216.0 - 0.5) * spacing * jitter
    };
    let mut positions = Vec::with_capacity(count * 4);
    'fill: for layer in 0.. {
        for z in (0..cells[2]).rev() {
            for y in 0..cells[1] {
                for x in 0..cells[0] {
                    if positions.len() >= count * 4 {
                        break 'fill;
                    }
                    let base = [x, y, z].map(|c| (c as f32 + 0.5) * spacing);
                    let lift = layer as f32 * spacing * 0.5;
                    positions.extend_from_slice(&[lo[0] + base[0] + offset(), lo[1] + base[1] + lift + offset(), lo[2] + base[2] + offset(), 1.0]);
                }
            }
        }
    }
    positions
}

/// The density poly6 (smoothing radius `h`, unit mass) sums to at a particle of a cubic lattice
/// `spacing` apart, itself included: Position Based Fluids' rest density
/// (`PbfOptions::rest_density`) for a fluid filled on that lattice, so it keeps its volume.
/// About `1 / spacing³` once `spacing` is well under `h`.
pub fn lattice_density(spacing: f32, h: f32) -> f32 {
    let n = (h / spacing).ceil() as i32;
    let poly6 = 315.0 / (64.0 * std::f32::consts::PI * h.powi(9));
    let mut rho = 0.0;
    for i in -n..=n {
        for j in -n..=n {
            for k in -n..=n {
                let d2 = ((i * i + j * j + k * k) as f32) * spacing * spacing;
                if d2 < h * h {
                    rho += poly6 * (h * h - d2).powi(3);
                }
            }
        }
    }
    rho
}

impl PlanarContainerShape {
    /// A lattice `spacing` apart in the columns whose signed distance to the outline `inside`
    /// accepts, from half a spacing over the floor up to `top`: the water the container holds
    /// to that level (4 floats a particle, w = 1). Its length / 4 is also how many particles
    /// fill it to there.
    pub fn lattice(&self, spacing: f32, top: f32, inside: impl Fn(f32) -> bool) -> Vec<f32> {
        let (lo, hi) = self.bounds();
        let mut particles = Vec::new();
        let mut z = lo[1];
        while z <= hi[1] {
            let mut x = lo[0];
            while x <= hi[0] {
                let [d, floor] = self.sample(x, z);
                if inside(d) {
                    let mut y = floor + spacing * 0.5;
                    while y < top {
                        particles.extend_from_slice(&[x, y, z, 1.0]);
                        y += spacing;
                    }
                }
                x += spacing;
            }
            z += spacing;
        }
        particles
    }

    /// Blur the floor (a box filter `radius` nodes each way, along x then z, twice). A floor
    /// shaped by the distance to the outline creases outside its concave stretches; this smooths
    /// the creases out.
    pub fn smooth_floor(&mut self, radius: u32) {
        let [w, h] = self.dims.map(|d| d as i32);
        let r = radius as i32;
        for _ in 0..2 {
            for axis in 0..2 {
                let floor: Vec<f32> = self.nodes.iter().map(|n| n[1]).collect();
                for j in 0..h {
                    for i in 0..w {
                        let (mut sum, mut n) = (0.0, 0.0);
                        for k in -r..=r {
                            let (x, z) = if axis == 0 { (i + k, j) } else { (i, j + k) };
                            if x >= 0 && x < w && z >= 0 && z < h {
                                sum += floor[(z * w + x) as usize];
                                n += 1.0;
                            }
                        }
                        self.nodes[(j * w + i) as usize][1] = sum / n;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_box_fill_holds_the_count_inside_the_box() {
        let (lo, hi) = ([-1.0, 0.0, -2.0], [1.0, 3.0, 2.0]);
        for count in [1, 1000, 4097] {
            let p = fill_box(count, lo, hi, 0.3);
            assert_eq!(p.len(), count * 4);
            for q in p.chunks(4) {
                assert!((0..3).all(|i| q[i] > lo[i] && q[i] < hi[i] + 1.0), "{q:?} outside");
                assert_eq!(q[3], 1.0);
            }
        }
        // the first row is at the box's +z end
        assert!(fill_box(1000, lo, hi, 0.0)[2] > 1.5);
    }

    #[test]
    fn lattice_density_tends_to_one_over_the_cell_volume() {
        let fine = lattice_density(0.1, 1.0);
        assert!((fine * 0.001 - 1.0).abs() < 0.01, "{fine}");
        // coarse: the particle's own share dominates
        let coarse = lattice_density(2.0, 1.0);
        assert!((coarse - 315.0 / (64.0 * std::f32::consts::PI)).abs() < 1e-3);
    }

    #[test]
    fn a_shaped_lattice_fills_the_columns_inside_up_to_the_top() {
        let square = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let shape = PlanarContainerShape::from_outline(&square, 0.1, 0.2, |_, _, _| -0.5);
        let p = shape.lattice(0.1, 0.0, |d| d < 0.0);
        assert!(!p.is_empty());
        for q in p.chunks(4) {
            assert!(q[0].abs() < 1.0 && q[2].abs() < 1.0 && q[1] > -0.5 && q[1] < 0.0, "{q:?}");
        }
        // 5 layers (-0.45 .. -0.05) of about 20 x 20 columns
        let columns = p.len() / 4 / 5;
        assert!((360..=441).contains(&columns), "{columns}");
    }
}
