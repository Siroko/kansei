//! The ray tracing grid on a real GPU, against a CPU reference: every triangle is listed in every
//! cell it overlaps (boxes whose faces lie on cell faces, where the scout's grid lost hits, a
//! soup of small triangles, a sphere half out of the box, a ground too wide for the cells that
//! goes to the big list, an alpha-tested quad), and rays find the closest hit (and any hit) the
//! brute-force reference finds, with the big list overflowing, without macro-cell skipping, and
//! after the buffers grew from a readback. Skipped (passes) when no adapter is available.

use glam::{DMat4, DVec3, Mat4, Quat, Vec3};
use kansei_core::geometries::{BoxGeometry, Geometry, IcosphereGeometry, Vertex};
use kansei_core::rt::{RtGrid, RtGridOptions, RtInstance, RtMesh, RtScene, RtSurface, RT_GRID_WGSL};

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    Some(pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap())
}

/// A small deterministic generator (xorshift).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }

    fn unit(&mut self) -> DVec3 {
        loop {
            let v = DVec3::new(self.range(-1.0, 1.0), self.range(-1.0, 1.0), self.range(-1.0, 1.0));
            if v.length_squared() > 1e-3 && v.length_squared() <= 1.0 {
                return v.normalize();
            }
        }
    }
}

fn vertex(p: [f32; 3], uv: [f32; 2]) -> Vertex {
    Vertex { position: [p[0], p[1], p[2], 1.0], normal: [0.0, 1.0, 0.0], uv }
}

/// `count` triangles of 5 to 60 cm round the origin, within `reach`.
fn soup(rng: &mut Rng, count: usize, reach: f64) -> Geometry {
    let mut vertices = Vec::new();
    for _ in 0..count {
        let c = DVec3::new(rng.range(-reach, reach), rng.range(-reach, reach), rng.range(-reach, reach));
        let size = rng.range(0.05, 0.6);
        for k in 0..3 {
            let p = c + rng.unit() * size;
            vertices.push(vertex([p.x as f32, p.y as f32, p.z as f32], [k as f32 * 0.5, (k % 2) as f32]));
        }
    }
    let indices = (0..vertices.len() as u32).collect();
    Geometry::new("Soup", vertices, indices)
}

/// A horizontal square of side `size` at the origin, two triangles, facing up.
fn ground(size: f32) -> Geometry {
    let h = size * 0.5;
    let vertices = vec![vertex([-h, 0.0, -h], [0.0, 0.0]), vertex([h, 0.0, -h], [1.0, 0.0]), vertex([h, 0.0, h], [1.0, 1.0]), vertex([-h, 0.0, h], [0.0, 1.0])];
    Geometry::new("Ground", vertices, vec![0, 2, 1, 0, 3, 2])
}

/// A unit quad in the xy plane, uvs 0-1 across it.
fn quad() -> Geometry {
    let vertices = vec![vertex([-0.5, -0.5, 0.0], [0.0, 0.0]), vertex([0.5, -0.5, 0.0], [1.0, 0.0]), vertex([0.5, 0.5, 0.0], [1.0, 1.0]), vertex([-0.5, 0.5, 0.0], [0.0, 1.0])];
    Geometry::new("Quad", vertices, vec![0, 1, 2, 0, 2, 3])
}

/// Layer 1 is a 4 x 4 checkerboard of holes; every other layer is solid. (CPU twin of COVERED.)
const COVERED_WGSL: &str = r#"
fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool {
    if (layer != 1u) { return true; }
    let c = vec2i(floor(uv * 4.0));
    return ((c.x + c.y) & 1) == 0;
}
"#;

fn covered(layer: Option<u8>, uv: [f64; 2]) -> Option<bool> {
    if layer != Some(1) {
        return Some(true);
    }
    // too near a checker line to say (f32 on the GPU)
    let near = |x: f64| ((x * 4.0) - (x * 4.0).round()).abs() < 1e-4;
    if near(uv[0]) || near(uv[1]) {
        return None;
    }
    Some(((uv[0] * 4.0).floor() as i64 + (uv[1] * 4.0).floor() as i64) & 1 == 0)
}

/// A world triangle on the CPU, in f64.
#[derive(Clone, Copy)]
struct Tri {
    v: [DVec3; 3],
    uv: [[f64; 2]; 3],
    layer: Option<u8>,
}

fn world_triangles(scene: &RtScene) -> Vec<Tri> {
    let mut out = Vec::new();
    for i in &scene.instances {
        let m = scene.mesh(i.mesh);
        let t = DMat4::from_cols_array(&i.transform.to_cols_array().map(|x| x as f64));
        for k in m.indices.chunks(3) {
            let p = |v: u32| t.transform_point3(DVec3::from(m.positions[v as usize].map(|x| x as f64)));
            let uv = |v: u32| m.uvs[v as usize].map(|x| x as f64);
            out.push(Tri { v: [p(k[0]), p(k[1]), p(k[2])], uv: [uv(k[0]), uv(k[1]), uv(k[2])], layer: i.surface.alpha_layer });
        }
    }
    out
}

/// The closest hit (t) of a ray among `tris` within (0, t_max), barycentric edges widened by
/// `slack` (negative: narrowed); None for a miss; Err when an alpha test is too close to call.
fn reference_hit(tris: &[Tri], o: DVec3, d: DVec3, t_max: f64, slack: f64, solid: bool) -> Result<Option<f64>, ()> {
    let mut best: Option<f64> = None;
    let mut unsure = false;
    for tri in tris {
        let e1 = tri.v[1] - tri.v[0];
        let e2 = tri.v[2] - tri.v[0];
        let p = d.cross(e2);
        let det = e1.dot(p);
        if det.abs() < 1e-18 {
            continue;
        }
        let s = o - tri.v[0];
        let u = s.dot(p) / det;
        let q = s.cross(e1);
        let v = d.dot(q) / det;
        let t = e2.dot(q) / det;
        if u < -slack || v < -slack || u + v > 1.0 + slack || t <= 1e-9 || t >= t_max || best.is_some_and(|b| t >= b) {
            continue;
        }
        if !solid {
            let w = 1.0 - u - v;
            let uv = [0, 1].map(|c| tri.uv[0][c] * w + tri.uv[1][c] * u + tri.uv[2][c] * v);
            match covered(tri.layer, uv) {
                Some(true) => {}
                Some(false) => continue,
                None => {
                    unsure = true;
                    continue;
                }
            }
        }
        best = Some(t);
    }
    if unsure {
        Err(())
    } else {
        Ok(best)
    }
}

/// The reference's verdict when it doesn't depend on rounding: the same under barycentric edges
/// widened and narrowed by 1e-5, and alpha tests clear of the checker's lines.
fn robust_hit(tris: &[Tri], o: DVec3, d: DVec3, t_max: f64, solid: bool) -> Option<Option<f64>> {
    let wide = reference_hit(tris, o, d, t_max, 1e-5, solid).ok()?;
    let narrow = reference_hit(tris, o, d, t_max, -1e-5, solid).ok()?;
    match (wide, narrow) {
        (None, None) => Some(None),
        (Some(a), Some(b)) if (a - b).abs() <= 1e-7 * (1.0 + a) => Some(Some(a)),
        _ => None,
    }
}

fn exit_distance(o: DVec3, d: DVec3, lo: DVec3, hi: DVec3) -> f64 {
    let inv = d.recip();
    let a = (lo - o) * inv;
    let b = (hi - o) * inv;
    a.max(b).min_element()
}

/// The test scene: unit boxes with faces on cell faces, a soup, a sphere half out of the box, a
/// wide ground, an alpha-tested quad.
fn scene(rng: &mut Rng) -> RtScene {
    let mut scene = RtScene::new();
    let cube = scene.add_mesh(RtMesh::from_geometry(&BoxGeometry::new(1.0, 1.0, 1.0)));
    let soup = scene.add_mesh(RtMesh::from_geometry(&soup(rng, 400, 1.5)));
    let sphere = scene.add_mesh(RtMesh::from_geometry(&IcosphereGeometry::new(1.2, 2)));
    let ground = scene.add_mesh(RtMesh::from_geometry(&ground(40.0)));
    let quad = scene.add_mesh(RtMesh::from_geometry(&quad()));
    let solid = RtSurface::new([0.5, 0.4, 0.3]);
    // cell faces lie on multiples of 0.25: cubes centred on half metres have faces on them
    for (x, y, z) in [(0.5, 0.5, 0.5), (-1.5, 0.5, 2.5), (2.5, 1.5, -1.5), (-2.5, 0.5, -2.5)] {
        scene.add_instance(RtInstance { mesh: cube, transform: Mat4::from_translation(Vec3::new(x, y, z)), surface: solid });
    }
    // a cube whose faces run exactly along a cell face, turned a quarter turn (cos/sin rounding)
    scene.add_instance(RtInstance { mesh: cube, transform: Mat4::from_translation(Vec3::new(1.5, 0.5, 1.5)) * Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2), surface: solid });
    scene.add_instance(RtInstance { mesh: soup, transform: Mat4::IDENTITY, surface: solid });
    scene.add_instance(RtInstance {
        mesh: soup,
        transform: Mat4::from_scale_rotation_translation(Vec3::splat(0.8), Quat::from_euler(glam::EulerRot::XYZ, 0.3, 1.1, -0.4), Vec3::new(1.0, 1.2, -1.0)),
        surface: RtSurface::new([0.9, 0.9, 0.9]),
    });
    scene.add_instance(RtInstance { mesh: sphere, transform: Mat4::from_translation(Vec3::new(3.9, 1.0, 0.0)), surface: solid });
    scene.add_instance(RtInstance { mesh: ground, transform: Mat4::from_translation(Vec3::new(0.0, -0.5, 0.0)), surface: solid });
    scene.add_instance(RtInstance {
        mesh: quad,
        transform: Mat4::from_translation(Vec3::new(-1.0, 1.5, 0.0)) * Mat4::from_rotation_y(0.4) * Mat4::from_scale(Vec3::splat(1.5)),
        surface: RtSurface::new([0.1, 0.6, 0.1]).with_alpha_layer(1),
    });
    // an instance wholly outside the box: never gathered
    scene.add_instance(RtInstance { mesh: cube, transform: Mat4::from_translation(Vec3::new(30.0, 0.0, 0.0)), surface: solid });
    scene
}

/// An 8 x 4 x 8 m room of 25 cm cells; triangles wider than 16 x 16 cells go to the big list.
fn options() -> RtGridOptions {
    RtGridOptions { dims: [32, 16, 32], cell: 0.25, fixed_origin: Some(Vec3::new(-4.0, -1.0, -4.0)), big_triangle_cells: 256, ..Default::default() }
}

fn build(device: &wgpu::Device, queue: &wgpu::Queue, grid: &mut RtGrid, scene: &mut RtScene) {
    let mut encoder = device.create_command_encoder(&Default::default());
    grid.begin(device, queue, &mut encoder);
    scene.gather(device, queue, &mut encoder, grid);
    grid.finish(&mut encoder);
    queue.submit(Some(encoder.finish()));
    grid.read_back(device, queue);
    grid.wait_readback(device);
}

fn read_words(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: buffer.size(), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    queue.submit(Some(encoder.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let words = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    words
}

/// A ray: origin, t_min = 0; direction, t_max.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Ray {
    o: [f32; 4],
    d: [f32; 4],
}

const TRACE_WGSL: &str = r#"
struct Ray { o: vec4f, d: vec4f };
@group(0) @binding(3) var<storage, read> rays: array<Ray>;
@group(0) @binding(4) var<storage, read_write> hits: array<vec4f>;
@group(0) @binding(5) var<uniform> mode: vec4u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= arrayLength(&rays)) { return; }
    let r = rays[i];
    let t_max = min(r.d.w, kansei_rt_exit(r.o.xyz, r.d.xyz));
    let h = kansei_rt_trace(r.o.xyz, r.d.xyz, 0.0, t_max, mode.x);
    hits[i * 2u] = vec4f(h.t, select(0.0, 1.0, h.found), f32(h.cells), f32(h.tests));
    hits[i * 2u + 1u] = vec4f(h.normal, bitcast<f32>(h.triangle));
}
"#;

/// (t, found, cells, tests) and the normal of each ray's hit, traced with `flags`.
fn trace(device: &wgpu::Device, queue: &wgpu::Queue, grid: &RtGrid, rays: &[Ray], flags: u32) -> Vec<[f32; 8]> {
    use wgpu::util::DeviceExt;
    let code = format!("{RT_GRID_WGSL}\n{}\n{COVERED_WGSL}\n{TRACE_WGSL}", RtGrid::bindings_wgsl(0, 0));
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("trace"), source: wgpu::ShaderSource::Wgsl(code.into()) });
    let mut entries = RtGrid::layout_entries(0, wgpu::ShaderStages::COMPUTE).to_vec();
    let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    entries.extend([
        storage(3, true),
        storage(4, false),
        wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
    ]);
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&layout), module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let ray_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(rays), usage: wgpu::BufferUsages::STORAGE });
    let hits = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (rays.len() * 32) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let mode = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&[flags, 0, 0, 0]), usage: wgpu::BufferUsages::UNIFORM });
    let mut bind = grid.bind_group_entries(0).to_vec();
    bind.extend([
        wgpu::BindGroupEntry { binding: 3, resource: ray_buffer.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: hits.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 5, resource: mode.as_entire_binding() },
    ]);
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &bind });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((rays.len() as u32).div_ceil(64), 1, 1);
    }
    queue.submit(Some(encoder.finish()));
    let words = read_words(device, queue, &hits);
    words.chunks(8).map(|c| c.iter().map(|w| f32::from_bits(*w)).collect::<Vec<_>>().try_into().unwrap()).collect()
}

/// Rays from inside the box: aimed at interior points of random triangles, in random
/// directions, and running exactly along cell faces.
fn rays(rng: &mut Rng, tris: &[Tri], lo: DVec3, hi: DVec3) -> Vec<Ray> {
    let inside = |rng: &mut Rng| DVec3::new(rng.range(lo.x + 0.01, hi.x - 0.01), rng.range(lo.y + 0.01, hi.y - 0.01), rng.range(lo.z + 0.01, hi.z - 0.01));
    let ray = |o: DVec3, d: DVec3| Ray { o: [o.x as f32, o.y as f32, o.z as f32, 0.0], d: [d.x as f32, d.y as f32, d.z as f32, 1e4] };
    let mut out = Vec::new();
    for _ in 0..3000 {
        let o = inside(rng);
        let tri = &tris[(rng.next() * tris.len() as f64) as usize % tris.len()];
        let (u, v) = (rng.range(0.05, 0.9), rng.range(0.05, 0.9));
        let (u, v) = if u + v > 0.95 { (1.0 - u, 1.0 - v) } else { (u, v) };
        let target = tri.v[0] + (tri.v[1] - tri.v[0]) * u + (tri.v[2] - tri.v[0]) * v;
        if target.cmpgt(lo).all() && target.cmplt(hi).all() {
            out.push(ray(o, (target - o).normalize()));
        }
    }
    for _ in 0..2000 {
        out.push(ray(inside(rng), rng.unit()));
    }
    // along cell faces: an origin on a face of x (or z), the direction in that face
    for k in 0..1000 {
        let mut o = inside(rng);
        let mut d = rng.unit();
        if k % 2 == 0 {
            o.x = (o.x * 4.0).round() / 4.0;
            d.x = 0.0;
        } else {
            o.z = (o.z * 4.0).round() / 4.0;
            d.z = 0.0;
        }
        out.push(ray(o, d.normalize()));
    }
    out
}

/// Every ray whose reference verdict is robust gets it from the GPU, closest and any hit; returns
/// (rays compared, rays hitting).
fn check_rays(device: &wgpu::Device, queue: &wgpu::Queue, grid: &RtGrid, tris: &[Tri], rays: &[Ray], what: &str) -> (usize, usize) {
    let (lo, hi) = grid.bounds();
    let (lo, hi) = (lo.as_dvec3(), hi.as_dvec3());
    let closest = trace(device, queue, grid, rays, 0);
    let any = trace(device, queue, grid, rays, 1);
    let (mut compared, mut hitting, mut wrong) = (0, 0, Vec::new());
    for (k, r) in rays.iter().enumerate() {
        let o = DVec3::new(r.o[0] as f64, r.o[1] as f64, r.o[2] as f64);
        let d = DVec3::new(r.d[0] as f64, r.d[1] as f64, r.d[2] as f64);
        let t_max = exit_distance(o, d, lo, hi);
        let Some(expect) = robust_hit(tris, o, d, t_max, false) else { continue };
        compared += 1;
        let (got, got_any) = (closest[k], any[k]);
        let ok = match expect {
            None => got[1] == 0.0 && got_any[1] == 0.0,
            Some(t) => {
                hitting += 1;
                // the closest hit's distance, and a unit normal facing the ray; any hit found
                let n = DVec3::new(got[4] as f64, got[5] as f64, got[6] as f64);
                got[1] == 1.0 && (got[0] as f64 - t).abs() <= 1e-4 * (1.0 + t) && (n.length() - 1.0).abs() < 1e-3 && n.dot(d) <= 1e-6 && got_any[1] == 1.0 && (got_any[0] as f64) >= t - 1e-4 * (1.0 + t)
            }
        };
        if !ok {
            wrong.push(format!("ray {k}: o {o} d {d}: expected {expect:?}, got t {} found {} (any: found {} t {})", got[0], got[1], got_any[1], got_any[0]));
        }
    }
    assert!(wrong.is_empty(), "{what}: {} of {compared} rays wrong:\n{}", wrong.len(), wrong.iter().take(12).cloned().collect::<Vec<_>>().join("\n"));
    assert!(compared * 10 >= rays.len() * 9, "{what}: only {compared} of {} rays robust", rays.len());
    (compared, hitting)
}

/// Whether a triangle overlaps an axis-aligned box (centre c, half size h), exactly (f64): the
/// separating axes of Akenine-Moller 2001.
fn tri_box_overlap(v: [DVec3; 3], c: DVec3, h: f64) -> bool {
    let v = v.map(|p| p - c);
    let separated = |axis: DVec3| {
        let p = [axis.dot(v[0]), axis.dot(v[1]), axis.dot(v[2])];
        let r = h * (axis.x.abs() + axis.y.abs() + axis.z.abs());
        p.iter().cloned().fold(f64::MAX, f64::min) > r || p.iter().cloned().fold(f64::MIN, f64::max) < -r
    };
    if [DVec3::X, DVec3::Y, DVec3::Z].into_iter().any(separated) {
        return false;
    }
    let n = (v[1] - v[0]).cross(v[2] - v[0]);
    if separated(n) {
        return false;
    }
    let edges = [v[1] - v[0], v[2] - v[1], v[0] - v[2]];
    !edges.iter().any(|e| [DVec3::X, DVec3::Y, DVec3::Z].iter().any(|a| separated(a.cross(*e))))
}

#[test]
fn every_triangle_is_listed_in_every_cell_it_overlaps() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let mut rng = Rng(0x2545f4914f6cdd1d);
    let mut scene = scene(&mut rng);
    let options = options();
    let mut grid = RtGrid::new(&device, options);
    build(&device, &queue, &mut grid, &mut scene);
    let stats = grid.stats();
    assert_eq!(scene.in_box, scene.instances.len() - 1, "the cube far away is left out");
    // the ground's two triangles are too wide for the cells: the big list
    assert_eq!(stats.big_triangles, 2);
    let triangles = read_words(&device, &queue, grid.triangles_buffer());
    let cells = read_words(&device, &queue, grid.cells_buffer());
    let (_, big_base, refs_base) = grid.cells_layout_words();
    let count = stats.triangles as usize;
    assert!(count > 0 && count <= stats.triangle_capacity as usize);
    // the cells' lists, as sets
    let dims = options.dims.map(|d| d as usize);
    let cell_count = dims.iter().product::<usize>();
    let mut listed = vec![std::collections::HashSet::new(); cell_count];
    let mut start = 0;
    for (c, set) in listed.iter_mut().enumerate() {
        let end = cells[c] as usize;
        for k in start..end {
            set.insert(cells[refs_base as usize + k]);
        }
        start = end;
    }
    assert_eq!(start as u32, stats.references, "the last cell ends at the references needed");
    let big: Vec<u32> = cells[big_base as usize + 1..][..cells[big_base as usize] as usize].to_vec();
    // each gathered triangle, from the GPU's own words: in every cell it overlaps (exactly, in
    // f64), unless it is in the big list
    let origin = options.fixed_origin.unwrap().as_dvec3();
    let cell = options.cell as f64;
    let (mut required, mut extra) = (0usize, 0usize);
    for id in 0..count {
        let w = &triangles[id * 16..id * 16 + 16];
        let f = |k: usize| f32::from_bits(w[k]) as f64;
        let v0 = DVec3::new(f(0), f(1), f(2));
        let v = [v0, v0 + DVec3::new(f(4), f(5), f(6)), v0 + DVec3::new(f(8), f(9), f(10))];
        if big.contains(&(id as u32)) {
            continue;
        }
        let lo = ((v[0].min(v[1]).min(v[2]) - origin) / cell).floor().max(DVec3::ZERO);
        let hi = ((v[0].max(v[1]).max(v[2]) - origin) / cell).floor().min(DVec3::new(dims[0] as f64 - 1.0, dims[1] as f64 - 1.0, dims[2] as f64 - 1.0));
        for z in lo.z as usize..=hi.z as usize {
            for y in lo.y as usize..=hi.y as usize {
                for x in lo.x as usize..=hi.x as usize {
                    let c = (z * dims[1] + y) * dims[0] + x;
                    let centre = origin + (DVec3::new(x as f64, y as f64, z as f64) + 0.5) * cell;
                    let overlaps = tri_box_overlap(v, centre, cell * 0.5);
                    let has = listed[c].contains(&(id as u32));
                    if overlaps {
                        required += 1;
                        assert!(has, "triangle {id} {v:?} overlaps cell ({x}, {y}, {z}) but is not listed there");
                    } else if has {
                        extra += 1;
                    }
                }
            }
        }
    }
    // the epsilon lists a few more than the exact overlaps (triangles touching a cell's face)
    assert!(required > 0 && extra * 4 < required, "{extra} references past the {required} exact ones");
    eprintln!("{count} triangles, {} references: {required} exact overlaps, {extra} more in their boxes, the rest touching cells past them (epsilon); {} big", stats.references, stats.big_triangles);
}

#[test]
fn rays_find_what_the_reference_finds() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let mut scene = scene(&mut rng);
    let tris = world_triangles(&scene);
    let options = options();
    let mut grid = RtGrid::new(&device, options);
    build(&device, &queue, &mut grid, &mut scene);
    let (lo, hi) = grid.bounds();
    let rays = rays(&mut rng, &tris, lo.as_dvec3(), hi.as_dvec3());
    let (compared, hitting) = check_rays(&device, &queue, &grid, &tris, &rays, "default");
    assert!(hitting * 2 > compared, "most rays hit something ({hitting} of {compared})");
    // the alpha test cut holes: some rays aimed at the quad pass through
    let solid = trace(&device, &queue, &grid, &rays, 2);
    let cut = trace(&device, &queue, &grid, &rays, 0);
    assert!(solid.iter().zip(&cut).filter(|(s, c)| s[1] == 1.0 && (c[1] == 0.0 || c[0] > s[0] + 1e-4)).count() > 10, "the quad's holes let rays through");
    // without macro-cell skipping, with the big list too small for the ground (one triangle
    // scattered into the cells), and with a coarser cell: the same hits
    for (what, o) in [
        ("no macro skip", RtGridOptions { macro_skip: false, ..options }),
        ("big list of one", RtGridOptions { big_triangle_capacity: 1, ..options }),
        ("no big list", RtGridOptions { big_triangle_cells: u32::MAX, ..options }),
        ("half-metre cells", RtGridOptions { dims: [16, 8, 16], cell: 0.5, ..options }),
    ] {
        let mut grid = RtGrid::new(&device, o);
        build(&device, &queue, &mut grid, &mut scene);
        check_rays(&device, &queue, &grid, &tris, &rays, what);
    }
}

#[test]
fn the_buffers_grow_to_what_a_build_needed() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let mut rng = Rng(0xd1b54a32d192ed03);
    let mut scene = scene(&mut rng);
    let tris = world_triangles(&scene);
    let options = RtGridOptions { triangle_capacity: 64, reference_capacity: 1024, ..options() };
    let mut grid = RtGrid::new(&device, options);
    build(&device, &queue, &mut grid, &mut scene);
    // the scene reserved its triangles; the references overflowed and were read back
    let first = grid.stats();
    assert!(first.triangles <= first.triangle_capacity);
    assert!(first.references > 1024, "{first:?}");
    build(&device, &queue, &mut grid, &mut scene);
    let second = grid.stats();
    assert!(second.reference_capacity >= first.references, "{second:?}");
    assert_eq!(second.references, first.references);
    let (lo, hi) = grid.bounds();
    let rays = rays(&mut rng, &tris, lo.as_dvec3(), hi.as_dvec3());
    check_rays(&device, &queue, &grid, &tris, &rays, "grown");
}

#[test]
fn a_box_that_follows_the_eye_moves_in_steps() {
    let Some((device, _queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let mut grid = RtGrid::new(&device, RtGridOptions { dims: [16, 8, 16], cell: 0.5, below: 0.25, snap_cells: 4, ..Default::default() });
    assert!(grid.follow(Vec3::new(0.3, 1.7, -0.2)));
    let (lo, hi) = grid.bounds();
    assert_eq!(hi - lo, Vec3::new(8.0, 4.0, 8.0));
    // snapped to 2 m, the eye inside, about a quarter of the height below it
    assert_eq!(lo % 2.0, Vec3::ZERO);
    assert!(Vec3::new(0.3, 1.7, -0.2).cmpgt(lo).all() && Vec3::new(0.3, 1.7, -0.2).cmplt(hi).all());
    assert!(!grid.follow(Vec3::new(0.4, 1.75, -0.15)), "a small move keeps the box");
    assert!(grid.follow(Vec3::new(5.0, 1.7, 0.0)));
}

/// GPU build times on this machine (native, kansei's profiler through a headless renderer), for
/// the PR's numbers: `cargo test --release -p kansei-core --test rt_grid_gpu -- --ignored
/// --nocapture`. A room of 4.6 m walls at 4.75 cm cells (gi-box's grid), with and without the big
/// list, alone and with 20k small triangles.
#[test]
#[ignore]
fn build_times() {
    use kansei_core::renderers::{Renderer, RendererConfig};
    let instance = wgpu::Instance::default();
    let Some(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let mut renderer = Renderer::new(RendererConfig { width: 64, height: 64, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    renderer.set_profiling(true);
    let (device, queue) = (renderer.device().clone(), renderer.queue().clone());
    let mut rng = Rng(7);
    // a room: floor, ceiling and four walls of 4.6 x 4.4 x 4.6 m, 10 cm thick (gi-box's size)
    let room = |soup_triangles: usize, rng: &mut Rng| {
        let mut room = RtScene::new();
        let slab = room.add_mesh(RtMesh::from_geometry(&BoxGeometry::new(1.0, 1.0, 1.0)));
        let solid = RtSurface::new([0.5; 3]);
        for (size, at) in [
            ([4.6, 0.1, 4.6], [0.0, -0.05, 0.0]),
            ([4.6, 0.1, 4.6], [0.0, 4.45, 0.0]),
            ([0.1, 4.4, 4.6], [-2.35, 2.2, 0.0]),
            ([0.1, 4.4, 4.6], [2.35, 2.2, 0.0]),
            ([4.6, 4.4, 0.1], [0.0, 2.2, -2.35]),
            ([4.6, 4.4, 0.1], [0.0, 2.2, 2.35]),
        ] {
            room.add_instance(RtInstance { mesh: slab, transform: Mat4::from_translation(Vec3::from(at)) * Mat4::from_scale(Vec3::from(size)), surface: solid });
        }
        if soup_triangles > 0 {
            // a dense object: triangles of 1-12 cm (a 19k-triangle dragon's are about 2 cm)
            let mut soup = soup(rng, soup_triangles, 0.5);
            for v in &mut soup.vertices {
                for c in 0..3 {
                    v.position[c] *= 0.2;
                }
            }
            let mesh = room.add_mesh(RtMesh::from_geometry(&soup));
            room.add_instance(RtInstance { mesh, transform: Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)), surface: solid });
        }
        room
    };
    let base = RtGridOptions { dims: [100, 96, 100], cell: 0.0475, fixed_origin: Some(Vec3::new(-2.375, -0.1, -2.375)), ..Default::default() };
    for (scene_name, soup_triangles) in [("walls", 0), ("walls and 20k small triangles", 20_000)] {
        let mut scene = room(soup_triangles, &mut rng);
        for (what, options) in [
            ("big list", base),
            ("big list over 256 columns", RtGridOptions { big_triangle_cells: 256, big_triangle_capacity: 128, ..base }),
            ("big list over 64 columns", RtGridOptions { big_triangle_cells: 64, big_triangle_capacity: 128, ..base }),
            ("no big list", RtGridOptions { big_triangle_cells: u32::MAX, ..base }),
        ] {
            let mut grid = RtGrid::new(&device, options);
            build(&device, &queue, &mut grid, &mut scene);
            let _ = renderer.take_profile();
            for _ in 0..20 {
                let mut encoder = device.create_command_encoder(&Default::default());
                grid.begin(&device, &queue, &mut encoder);
                scene.gather(&device, &queue, &mut encoder, &mut grid);
                grid.finish(&mut encoder);
                queue.submit(Some(encoder.finish()));
                renderer.end_profiled_frame();
                device.poll(wgpu::Maintain::Wait);
            }
            let profile = renderer.take_profile();
            let ms = |label: &str| profile.gpu.iter().find(|p| p.label == label).map_or(0.0, |p| p.busy_ms);
            let s = grid.stats();
            eprintln!(
                "{scene_name}, {what}: gather {:.3} + count {:.3} + scan {:.3} + fill {:.3} ms (GPU, {} frames); {} triangles, {} references, {} big",
                ms("Rt/Gather"), ms("Rt/GridCount"), ms("Rt/GridScan"), ms("Rt/GridFill"), profile.gpu_frames, s.triangles, s.references, s.big_triangles
            );
        }
    }
}
