//! A container of any outline in plan for the 3D fluid: vertical walls that follow a closed 2D
//! outline on the XZ plane, and a floor whose height varies across it (a lake bed, a pool with a
//! shallow end, a channel). Both are sampled from one grid of (signed distance to the outline,
//! floor height) nodes, built on the CPU by [`PlanarContainerShape`] and applied to the particles
//! after each substep's integration by [`FluidContainer`] (a [`FluidSubstepPass`]).
//!
//! The solver's own box bounds still apply; keep them around the container.

use bytemuck::{Pod, Zeroable};

use super::simulation::{FluidSimulation, FluidSubstepPass, SIM_PARAMS_WGSL};

/// A closed outline on the XZ plane and a floor under it, sampled on a grid of nodes.
///
/// `distance` is the signed distance to the outline (negative inside); the container's walls
/// stand `wall_offset` outside the outline, so the fluid can be confined to the outline plus a
/// strip around it (a shore) while the floor is shaped by the distance to the outline itself.
#[derive(Debug, Clone)]
pub struct PlanarContainerShape {
    /// The grid's first node (x, z).
    pub origin: [f32; 2],
    /// The spacing of the nodes.
    pub cell: f32,
    /// Nodes along x and z.
    pub dims: [u32; 2],
    /// Where the walls stand: this far outside the outline (negative: inside it).
    pub wall_offset: f32,
    /// Per node, x fastest: (signed distance to the outline, floor height).
    pub nodes: Vec<[f32; 2]>,
}

impl PlanarContainerShape {
    /// Sample `outline` (a closed polygon on XZ, either winding, the last point joined to the
    /// first) every `cell` over its bounds grown by `wall_offset` and `cell` on each side. The
    /// floor at each node is `floor(x, z, distance)`, `distance` the node's signed distance to
    /// the outline (negative inside).
    pub fn from_outline(outline: &[[f32; 2]], cell: f32, wall_offset: f32, floor: impl Fn(f32, f32, f32) -> f32) -> Self {
        assert!(outline.len() >= 3, "an outline needs 3 points or more");
        assert!(cell > 0.0);
        let (mut min, mut max) = ([f32::MAX; 2], [f32::MIN; 2]);
        for p in outline {
            for k in 0..2 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
        let grow = wall_offset.max(0.0) + cell;
        let origin = [min[0] - grow, min[1] - grow];
        let dims = [0, 1].map(|k| ((max[k] + grow - origin[k]) / cell).ceil() as u32 + 1);
        let mut nodes = Vec::with_capacity((dims[0] * dims[1]) as usize);
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let p = [origin[0] + i as f32 * cell, origin[1] + j as f32 * cell];
                let d = signed_distance(outline, p);
                nodes.push([d, floor(p[0], p[1], d)]);
            }
        }
        Self { origin, cell, dims, wall_offset, nodes }
    }

    /// Bilinear (distance, floor) at `(x, z)`, as the GPU pass samples it (clamped to the grid).
    pub fn sample(&self, x: f32, z: f32) -> [f32; 2] {
        let fx = ((x - self.origin[0]) / self.cell).clamp(0.0, (self.dims[0] - 1) as f32);
        let fz = ((z - self.origin[1]) / self.cell).clamp(0.0, (self.dims[1] - 1) as f32);
        let (i0, j0) = (fx.floor() as u32, fz.floor() as u32);
        let (i1, j1) = ((i0 + 1).min(self.dims[0] - 1), (j0 + 1).min(self.dims[1] - 1));
        let (tx, tz) = (fx - i0 as f32, fz - j0 as f32);
        let n = |i: u32, j: u32| self.nodes[(j * self.dims[0] + i) as usize];
        let lerp = |a: [f32; 2], b: [f32; 2], t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        lerp(lerp(n(i0, j0), n(i1, j0), tx), lerp(n(i0, j1), n(i1, j1), tx), tz)
    }

    /// Signed distance to the outline at `(x, z)` (negative inside).
    pub fn distance(&self, x: f32, z: f32) -> f32 {
        self.sample(x, z)[0]
    }

    /// Floor height at `(x, z)`.
    pub fn floor(&self, x: f32, z: f32) -> f32 {
        self.sample(x, z)[1]
    }

    /// Whether `(x, z)` is inside the walls.
    pub fn inside(&self, x: f32, z: f32) -> bool {
        self.distance(x, z) < self.wall_offset
    }

    /// The grid's extent: its first and last node (x, z).
    pub fn bounds(&self) -> ([f32; 2], [f32; 2]) {
        let last = [0, 1].map(|k| self.origin[k] + (self.dims[k] - 1) as f32 * self.cell);
        (self.origin, last)
    }

    /// The same shape in a space scaled by `s` (distances, floor and grid alike), e.g. a
    /// simulation run at a scale of the world.
    pub fn scaled(&self, s: f32) -> Self {
        Self {
            origin: self.origin.map(|v| v * s),
            cell: self.cell * s,
            dims: self.dims,
            wall_offset: self.wall_offset * s,
            nodes: self.nodes.iter().map(|n| [n[0] * s, n[1] * s]).collect(),
        }
    }
}

/// Signed distance from `p` to the closed polygon `outline` (negative inside, even-odd rule).
pub fn signed_distance(outline: &[[f32; 2]], p: [f32; 2]) -> f32 {
    let mut d2 = f32::MAX;
    let mut inside = false;
    let mut j = outline.len() - 1;
    for i in 0..outline.len() {
        let (a, b) = (outline[j], outline[i]);
        let e = [b[0] - a[0], b[1] - a[1]];
        let w = [p[0] - a[0], p[1] - a[1]];
        let t = ((w[0] * e[0] + w[1] * e[1]) / (e[0] * e[0] + e[1] * e[1]).max(1e-12)).clamp(0.0, 1.0);
        let q = [w[0] - e[0] * t, w[1] - e[1] * t];
        d2 = d2.min(q[0] * q[0] + q[1] * q[1]);
        // crossing test on the edge a→b
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]) {
            inside = !inside;
        }
        j = i;
    }
    if inside { -d2.sqrt() } else { d2.sqrt() }
}

/// How the container treats the particles it stops.
#[derive(Debug, Clone, Copy)]
pub struct FluidContainerOptions {
    /// How far inside the walls and above the floor the particles are kept.
    pub margin: f32,
    /// The fraction of the speed into a wall or the floor that bounces back.
    pub restitution: f32,
    /// The fraction of the speed along a wall or the floor lost at each contact.
    pub friction: f32,
}

impl Default for FluidContainerOptions {
    fn default() -> Self {
        Self { margin: 0.05, restitution: 0.1, friction: 0.02 }
    }
}

/// GPU layout of the container's uniform (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuContainer {
    origin: [f32; 2],
    cell: f32,
    wall_offset: f32,
    dims: [u32; 2],
    margin: f32,
    restitution: f32,
    friction: f32,
    _pad: [f32; 3],
}

pub(crate) const CONTAINER_WGSL: &str = r#"
struct Container {
    origin: vec2<f32>,
    cell: f32,
    wall_offset: f32,
    dims: vec2<u32>,
    margin: f32,
    restitution: f32,
    friction: f32,
    _pad0: f32, _pad1: f32, _pad2: f32,
};

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<uniform> container: Container;
// (signed distance to the outline, floor height) per node, x fastest
@group(0) @binding(4) var<storage, read> nodes: array<vec2<f32>>;

fn node(i: i32, j: i32) -> vec2<f32> {
    let d = vec2<i32>(container.dims);
    let c = clamp(vec2<i32>(i, j), vec2<i32>(0), d - vec2<i32>(1));
    return nodes[c.y * d.x + c.x];
}

// Bilinear (distance, floor) at p (x, z), clamped to the grid.
fn sample_field(p: vec2<f32>) -> vec2<f32> {
    let g = clamp((p - container.origin) / container.cell, vec2<f32>(0.0), vec2<f32>(container.dims - vec2<u32>(1u)));
    let i = vec2<i32>(floor(g));
    let t = g - floor(g);
    let a = mix(node(i.x, i.y), node(i.x + 1, i.y), t.x);
    let b = mix(node(i.x, i.y + 1), node(i.x + 1, i.y + 1), t.x);
    return mix(a, b, t.y);
}

// d(distance, floor)/dx and d/dz, by central differences half a cell apart.
fn gradient(p: vec2<f32>) -> mat2x2<f32> {
    let h = container.cell * 0.5;
    let dx = (sample_field(p + vec2<f32>(h, 0.0)) - sample_field(p - vec2<f32>(h, 0.0))) / (2.0 * h);
    let dz = (sample_field(p + vec2<f32>(0.0, h)) - sample_field(p - vec2<f32>(0.0, h))) / (2.0 * h);
    return mat2x2<f32>(dx, dz);
}

// The velocity after a contact on a surface of normal n (toward the fluid): what went into it
// bounces back by the restitution, what slid along it loses the friction.
fn collide(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let vn = dot(v, n);
    if (vn >= 0.0) { return v; }
    let vt = v - n * vn;
    return vt * (1.0 - container.friction) - n * vn * container.restitution;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }
    var pos = positions[idx].xyz;
    let v4 = velocities[idx];
    var vel = v4.xyz;

    // walls: out past them, back along the distance's gradient
    let over = sample_field(pos.xz).x - container.wall_offset + container.margin;
    if (over > 0.0) {
        let g = gradient(pos.xz);
        let outward = vec2<f32>(g[0].x, g[1].x);
        let len = length(outward);
        if (len > 1e-5) {
            let n2 = outward / len;
            pos = vec3<f32>(pos.x - n2.x * over, pos.y, pos.z - n2.y * over);
            vel = collide(vel, vec3<f32>(-n2.x, 0.0, -n2.y));
        }
    }

    // floor: up onto it, and off its slope
    let floor_y = sample_field(pos.xz).y + container.margin;
    if (pos.y < floor_y) {
        let g = gradient(pos.xz);
        let n = normalize(vec3<f32>(-g[0].y, 1.0, -g[1].y));
        pos.y = floor_y;
        vel = collide(vel, n);
    }

    positions[idx] = vec4<f32>(pos, positions[idx].w);
    velocities[idx] = vec4<f32>(vel, v4.w);
}
"#;

/// The container's pass: keeps a [`FluidSimulation`]'s particles inside a
/// [`PlanarContainerShape`] (in the simulation's space), after each substep's integration.
/// Run it with `FluidSimulation::update_batched_with` (after any collider passes, so the walls
/// and floor have the last word).
pub struct FluidContainer {
    pub options: FluidContainerOptions,
    shape: PlanarContainerShape,
    queue: wgpu::Queue,
    uniform: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl FluidContainer {
    pub fn new(sim: &FluidSimulation, shape: PlanarContainerShape, options: FluidContainerOptions) -> Self {
        use wgpu::util::DeviceExt;
        let (device, queue) = sim.gpu();
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FluidContainer/Uniform"),
            size: std::mem::size_of::<GpuContainer>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let nodes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FluidContainer/Nodes"),
            contents: bytemuck::cast_slice(&shape.nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let (pipeline, bind_group) = sim.substep_pipeline("FluidContainer", &format!("{SIM_PARAMS_WGSL}\n{CONTAINER_WGSL}"), &[(&uniform, false), (&nodes, true)]);
        let container = Self { options, shape, queue: queue.clone(), uniform, pipeline, bind_group };
        container.upload();
        container
    }

    pub fn shape(&self) -> &PlanarContainerShape {
        &self.shape
    }

    /// Upload `options` after changing them.
    pub fn upload(&self) {
        let s = &self.shape;
        let gpu = GpuContainer {
            origin: s.origin,
            cell: s.cell,
            wall_offset: s.wall_offset,
            dims: s.dims,
            margin: self.options.margin,
            restitution: self.options.restitution,
            friction: self.options.friction,
            _pad: [0.0; 3],
        };
        self.queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&gpu));
    }
}

impl FluidSubstepPass for FluidContainer {
    fn dispatch(&self, pass: &mut wgpu::ComputePass<'_>, particle_count: u32) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(particle_count.div_ceil(64), 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulations::fluid::colliders::{GpuColliders, COLLIDERS_WGSL};
    use crate::simulations::fluid::FluidCapsule;

    fn validate(name: &str, code: &str) -> naga::Module {
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        module
    }

    fn struct_size(module: &naga::Module, name: &str) -> usize {
        module
            .types
            .iter()
            .find_map(|(_, t)| match (&t.name, &t.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == name => Some(*span as usize),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no struct {name}"))
    }

    #[test]
    fn shaders_validate_and_the_uniforms_match() {
        let container = validate("container", &format!("{SIM_PARAMS_WGSL}\n{CONTAINER_WGSL}"));
        assert_eq!(struct_size(&container, "Container"), std::mem::size_of::<GpuContainer>());
        let colliders = validate("colliders", &format!("{SIM_PARAMS_WGSL}\n{COLLIDERS_WGSL}"));
        assert_eq!(struct_size(&colliders, "Colliders"), std::mem::size_of::<GpuColliders>());
        assert_eq!(struct_size(&colliders, "Capsule"), std::mem::size_of::<FluidCapsule>());
    }

    #[test]
    fn the_signed_distance_is_negative_inside_either_winding() {
        let square = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        let mut reversed = square;
        reversed.reverse();
        for outline in [square, reversed] {
            assert!((signed_distance(&outline, [2.0, 2.0]) + 2.0).abs() < 1e-5);
            assert!((signed_distance(&outline, [1.0, 3.0]) + 1.0).abs() < 1e-5);
            assert!((signed_distance(&outline, [6.0, 2.0]) - 2.0).abs() < 1e-5);
            // past a corner: to the corner
            assert!((signed_distance(&outline, [7.0, 8.0]) - 5.0).abs() < 1e-5);
        }
        // a concave notch: the notch is outside
        let notched = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [2.5, 4.0], [2.0, 1.0], [1.5, 4.0], [0.0, 4.0]];
        assert!(signed_distance(&notched, [2.0, 3.0]) > 0.0);
        assert!(signed_distance(&notched, [0.7, 3.0]) < 0.0);
    }

    #[test]
    fn the_shape_samples_its_distance_and_floor() {
        // a 10 x 6 rectangle whose floor deepens 0.5 per metre inside, down to -1
        let outline = [[-5.0, -3.0], [5.0, -3.0], [5.0, 3.0], [-5.0, 3.0]];
        let shape = PlanarContainerShape::from_outline(&outline, 0.25, 1.0, |_, _, d| (d * 0.5).clamp(-1.0, 0.0));
        let (min, max) = shape.bounds();
        assert!(min[0] <= -6.0 && min[1] <= -4.0 && max[0] >= 6.0 && max[1] >= 4.0, "{min:?} {max:?}");
        assert!((shape.distance(0.0, 0.0) + 3.0).abs() < 1e-4);
        assert!((shape.floor(0.0, 0.0) + 1.0).abs() < 1e-4);
        assert!((shape.floor(4.5, 0.0) + 0.25).abs() < 1e-4);
        assert_eq!(shape.floor(5.5, 0.0), 0.0);
        // the walls stand a metre outside the outline
        assert!(shape.inside(5.9, 0.0) && !shape.inside(6.1, 0.0));
        // scaled by 8, everything is 8 times as far
        let big = shape.scaled(8.0);
        assert!((big.floor(36.0, 0.0) + 2.0).abs() < 1e-3);
        assert!(big.inside(47.0, 0.0) && !big.inside(49.0, 0.0));
    }
}
