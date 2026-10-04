//! The jump-flood distance field on a real GPU, against a brute-force 3D distance transform:
//! random seeds, the static and dynamic split, and seeds from a volume's opacity.
//! Skipped (passes) when no adapter is available.

use kansei_core::gi::{JumpFloodSdf, SdfSeeds, VolumeLayout};
use wgpu::util::DeviceExt;

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let desc = wgpu::DeviceDescriptor { required_features: wgpu::Features::FLOAT32_FILTERABLE & adapter.features(), ..Default::default() };
    pollster::block_on(adapter.request_device(&desc, None)).ok()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
}

/// A voxelizer-style surface buffer (4 u32 a voxel) with `seeds` occupied.
fn surfaces(device: &wgpu::Device, layout: &VolumeLayout, seeds: &[[u32; 3]]) -> wgpu::Buffer {
    let [w, h, _] = layout.dims;
    let mut words = vec![0u32; layout.voxel_count() as usize * 4];
    for s in seeds {
        let i = ((s[2] * h + s[1]) * w + s[0]) as usize;
        words[4 * i] = 1 << 24;
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&words), usage: wgpu::BufferUsages::STORAGE })
}

/// Every texel of an r32float 3D texture.
fn read_field(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<f32> {
    let size = texture.size();
    let (w, h, d) = (size.width, size.height, size.depth_or_array_layers);
    let row = (w * 4).next_multiple_of(256);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h * d) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        size,
    );
    queue.submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let bytes = buffer.slice(..).get_mapped_range();
    let mut out = Vec::with_capacity((w * h * d) as usize);
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let o = ((z * h + y) * row + x * 4) as usize;
                out.push(f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]));
            }
        }
    }
    out
}

/// The exact field: from each voxel centre to the nearest seed's centre, less half a voxel.
fn brute_force(layout: &VolumeLayout, seeds: &[[u32; 3]]) -> Vec<f32> {
    let [w, h, d] = layout.dims;
    let mut out = Vec::with_capacity(layout.voxel_count() as usize);
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let best = seeds
                    .iter()
                    .map(|s| {
                        let (dx, dy, dz) = (s[0] as f32 - x as f32, s[1] as f32 - y as f32, s[2] as f32 - z as f32);
                        dx * dx + dy * dy + dz * dz
                    })
                    .fold(f32::MAX, f32::min);
                out.push((best.sqrt() - 0.5).max(0.0) * layout.voxel_size);
            }
        }
    }
    out
}

/// (share of voxels within a hundredth of a voxel of the exact distance, largest error in voxels)
fn compare(layout: &VolumeLayout, got: &[f32], want: &[f32]) -> (f32, f32) {
    let mut exact = 0;
    let mut worst = 0.0f32;
    for (g, w) in got.iter().zip(want) {
        let e = (g - w).abs() / layout.voxel_size;
        exact += (e < 0.01) as usize;
        worst = worst.max(e);
    }
    (exact as f32 / got.len() as f32, worst)
}

fn random_seeds(layout: &VolumeLayout, count: usize, seed: u64) -> Vec<[u32; 3]> {
    let mut rng = Rng(seed);
    (0..count).map(|_| std::array::from_fn(|i| rng.next() % layout.dims[i])).collect()
}

/// Random seeds in a 40 x 32 x 24 volume: the flood (JFA+2) finds the exact nearest seed for
/// almost every voxel, and is never more than a fraction of a voxel off.
#[test]
fn the_jump_flood_matches_a_brute_force_distance_transform() {
    let Some((device, queue)) = gpu() else { return eprintln!("no GPU adapter: skipping") };
    let layout = VolumeLayout::new([0.0, 0.0, 0.0], [2.5, 2.0, 1.5], 40);
    assert_eq!(layout.dims, [40, 32, 24]);
    for (count, seed) in [(1, 7), (12, 11), (90, 13)] {
        let seeds = random_seeds(&layout, count, seed);
        let buffer = surfaces(&device, &layout, &seeds);
        let sdf = JumpFloodSdf::new(&device, layout, SdfSeeds::Surfaces, Some(&buffer));
        let mut encoder = device.create_command_encoder(&Default::default());
        sdf.encode_static(&mut encoder);
        sdf.encode_distance(&mut encoder);
        queue.submit(Some(encoder.finish()));
        let got = read_field(&device, &queue, sdf.texture());
        let (exact, worst) = compare(&layout, &got, &brute_force(&layout, &seeds));
        eprintln!("{count} seeds: {:.4} of voxels exact, worst {worst:.3} voxels, {} flood passes", exact, sdf.flood_passes());
        assert!(exact > 0.995 && worst < 0.5, "{count} seeds: {exact} exact, worst {worst}");
    }
}

/// Static and dynamic seeds flood apart; the distance is the nearer of the two, as one flood of
/// both would give, and dropping the dynamic seeds brings the static field back.
#[test]
fn static_and_dynamic_seeds_combine() {
    let Some((device, queue)) = gpu() else { return eprintln!("no GPU adapter: skipping") };
    let layout = VolumeLayout::new([0.0; 3], [1.0; 3], 32);
    let (statics, dynamics) = (random_seeds(&layout, 20, 3), random_seeds(&layout, 6, 5));
    let static_buffer = surfaces(&device, &layout, &statics);
    let dynamic_buffer = surfaces(&device, &layout, &dynamics);
    let mut sdf = JumpFloodSdf::new(&device, layout, SdfSeeds::Surfaces, Some(&static_buffer));
    sdf.set_dynamic_surfaces(&device, Some(&dynamic_buffer));
    let mut encoder = device.create_command_encoder(&Default::default());
    sdf.encode_static(&mut encoder);
    sdf.encode_dynamic(&mut encoder);
    sdf.encode_distance(&mut encoder);
    queue.submit(Some(encoder.finish()));
    let all: Vec<[u32; 3]> = statics.iter().chain(&dynamics).copied().collect();
    let (exact, worst) = compare(&layout, &read_field(&device, &queue, sdf.texture()), &brute_force(&layout, &all));
    eprintln!("static and dynamic: {exact:.4} exact, worst {worst:.3} voxels");
    assert!(exact > 0.995 && worst < 0.5);
    // the dynamic seeds gone: only the static field's distance pass runs again
    sdf.set_dynamic_surfaces(&device, None);
    let mut encoder = device.create_command_encoder(&Default::default());
    sdf.encode_distance(&mut encoder);
    queue.submit(Some(encoder.finish()));
    let (exact, worst) = compare(&layout, &read_field(&device, &queue, sdf.texture()), &brute_force(&layout, &statics));
    assert!(exact > 0.995 && worst < 0.5, "static alone: {exact} exact, worst {worst}");
}

/// Seeds from a volume's opacity: voxels at least as opaque as the threshold.
#[test]
fn opacity_seeds_a_volume_field() {
    let Some((device, queue)) = gpu() else { return eprintln!("no GPU adapter: skipping") };
    let layout = VolumeLayout::new([0.0; 3], [1.0; 3], 16);
    let [w, h, d] = layout.dims;
    // two opaque voxels and a faint one (under the threshold)
    let mut texels = vec![[0u16; 4]; (w * h * d) as usize];
    let half = |v: f32| half_bits(v);
    let at = |x: u32, y: u32, z: u32| ((z * h + y) * w + x) as usize;
    texels[at(3, 4, 5)] = [0, 0, 0, half(1.0)];
    texels[at(12, 10, 2)] = [0, 0, 0, half(0.8)];
    texels[at(8, 8, 8)] = [0, 0, 0, half(0.2)];
    let texture = device.create_texture_with_data(
        &queue,
        &wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: d },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        bytemuck::cast_slice(&texels),
    );
    let view = texture.create_view(&Default::default());
    let sdf = JumpFloodSdf::new(&device, layout, SdfSeeds::Opacity { radiance: &view, threshold: 0.5 }, None);
    let mut encoder = device.create_command_encoder(&Default::default());
    sdf.encode(&mut encoder);
    queue.submit(Some(encoder.finish()));
    let (exact, worst) = compare(&layout, &read_field(&device, &queue, sdf.texture()), &brute_force(&layout, &[[3, 4, 5], [12, 10, 2]]));
    assert!(exact > 0.995 && worst < 0.5, "{exact} exact, worst {worst}");
}

fn half_bits(v: f32) -> u16 {
    if v == 0.0 {
        return 0;
    }
    let bits = v.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    ((bits >> 16) & 0x8000) as u16 | ((exponent as u16) << 10) | ((bits >> 13) & 0x3ff) as u16
}

// ── the field in the renderer's scene voxel GI ────────────────────────────────────────────

mod scene {
    use super::read_field;
    use kansei_core::cameras::Camera;
    use kansei_core::geometries::BoxGeometry;
    use kansei_core::gi::{GiSurface, SceneVoxelGiOptions, SdfShadows, VoxelGIEffect, VoxelGIOptions, VoxelGiQuality};
    use kansei_core::lights::{DirectionalLight, Light};
    use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages};
    use kansei_core::math::Vec3;
    use kansei_core::objects::{Renderable, Scene, SceneNode};
    use kansei_core::postprocessing::PostProcessingEffect;
    use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};

    const W: u32 = 128;
    const H: u32 = 96;

    fn renderer() -> Option<Renderer> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
        let mut renderer = Renderer::new(RendererConfig { width: W, height: H, sample_count: 1, ..Default::default() });
        pollster::block_on(renderer.initialize_headless(&adapter));
        Some(renderer)
    }

    const SURFACE_WGSL: &str = r#"
struct Surface { albedo: vec4<f32> };
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) normal: vec3<f32> };
struct GBufferOut { @location(0) color: vec4<f32>, @location(1) emissive: vec4<f32>, @location(2) normal: vec4<f32>, @location(3) albedo: vec4<f32> };
@vertex fn vertex_main(@location(0) p: vec4<f32>, @location(1) n: vec3<f32>) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * p;
    out.normal = (normal_matrix * vec4<f32>(n, 0.0)).xyz;
    return out;
}
@fragment fn fragment_main(in: VOut) -> GBufferOut {
    var out: GBufferOut;
    out.color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.emissive = vec4<f32>(0.0);
    out.normal = vec4<f32>(normalize(in.normal) * 0.5 + 0.5, 1.0);
    out.albedo = vec4<f32>(surface.albedo.rgb, 1.0);
    return out;
}
"#;

    fn material(albedo: [f32; 3]) -> Material {
        let mut m = Material::new("Surface", SURFACE_WGSL, vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), ..Default::default() });
        m.set_uniform_bindable(0, "Surface", &[albedo[0], albedo[1], albedo[2], 1.0f32]);
        m
    }

    fn block(size: [f32; 3], at: [f32; 3], albedo: [f32; 3]) -> Renderable {
        let mut r = Renderable::new(BoxGeometry::new(size[0], size[1], size[2]), material(albedo)).with_gi(GiSurface::new(albedo));
        r.object.set_position(at[0], at[1], at[2]);
        r
    }

    fn camera() -> Camera {
        let mut camera = Camera::new(50.0, 0.1, 50.0, W as f32 / H as f32);
        camera.set_position(0.0, 1.5, 3.0);
        camera.look_at(&Vec3::new(0.0, 0.0, 0.0));
        camera.update_projection_matrix();
        camera
    }

    fn box_distance(p: glam::Vec3, centre: glam::Vec3, half: glam::Vec3) -> f32 {
        let q = (p - centre).abs() - half;
        q.max(glam::Vec3::ZERO).length() + q.max_element().min(0.0)
    }

    /// The field around a voxelized box is the distance to the box, within a voxel and a half (the
    /// voxels approximate its surface), and follows the box when it moves as a dynamic renderable.
    #[test]
    fn the_scene_field_measures_the_distance_to_the_voxelized_meshes() {
        let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
        renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Low, bounds_min: [-1.0; 3], bounds_max: [1.0; 3], ..Default::default() });
        renderer.voxel_gi_mut().unwrap().enable_sdf();
        let mut scene = Scene::new();
        let (half, mut centre) = (glam::Vec3::new(0.3, 0.2, 0.25), glam::Vec3::new(-0.2, 0.0, 0.1));
        let index = scene.add(SceneNode::Renderable(block((half * 2.0).to_array(), centre.to_array(), [0.5; 3])));
        let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
        let mut camera = camera();
        for moved in [false, true] {
            if moved {
                centre = glam::Vec3::new(0.35, -0.1, -0.2);
                let r = scene.get_renderable_mut(index).unwrap();
                r.dynamic = true;
                r.object.set_position(centre.x, centre.y, centre.z);
            }
            renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
            let gi = renderer.voxel_gi().unwrap();
            let layout = *gi.volume().layout();
            let field = read_field(renderer.device(), renderer.queue(), gi.sdf().unwrap().texture());
            let [w, h, d] = layout.dims;
            let mut worst = 0.0f32;
            for z in 0..d {
                for y in 0..h {
                    for x in 0..w {
                        let p = glam::Vec3::from(layout.origin) + (glam::UVec3::new(x, y, z).as_vec3() + 0.5) * layout.voxel_size;
                        let want = box_distance(p, centre, half).max(0.0);
                        // outside the box, away from where its voxels are
                        if want < 2.0 * layout.voxel_size {
                            continue;
                        }
                        worst = worst.max((field[((z * h + y) * w + x) as usize] - want).abs() / layout.voxel_size);
                    }
                }
            }
            eprintln!("box {}: worst error {worst:.2} voxels", if moved { "moved (dynamic)" } else { "static" });
            assert!(worst < 1.5, "the field is {worst} voxels off");
        }
    }

    /// A floor under a blocker, lit by a sun with no shadow map: with the field as the fallback
    /// the floor's voxels in the blocker's shadow go dark (and those in the light stay as lit),
    /// without it they are all lit.
    #[test]
    fn the_field_shadows_the_injection_where_no_map_reaches() {
        let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
        renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Low, bounds_min: [-1.0; 3], bounds_max: [1.0; 3], ..Default::default() });
        let gi = renderer.voxel_gi_mut().unwrap();
        gi.settings.bounce = 0.0;
        let mut scene = Scene::new();
        scene.add(SceneNode::Renderable(block([1.8, 0.2, 1.8], [0.0, -0.59, 0.0], [0.5; 3])));
        scene.add(SceneNode::Renderable(block([0.6, 0.1, 0.6], [0.0, 0.3, 0.0], [0.5; 3])));
        scene.add(SceneNode::Light(Light::Directional(DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 3.0))));
        let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
        let mut camera = camera();
        let mut floor = |renderer: &mut Renderer, shadows: SdfShadows| -> (f32, f32) {
            let gi = renderer.voxel_gi_mut().unwrap();
            gi.settings.sdf_shadows = shadows;
            renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
            let gi = renderer.voxel_gi().unwrap();
            let layout = *gi.volume().layout();
            let radiance = super::read_mip0(renderer.device(), renderer.queue(), gi.volume().texture());
            let [w, h, _] = layout.dims;
            let y = ((-0.49 - layout.origin[1]) / layout.voxel_size) as u32;
            let at = |x: f32, z: f32| {
                let c = |v: f32, o: f32| ((v - o) / layout.voxel_size) as u32;
                radiance[((c(z, layout.origin[2]) * h + y) * w + c(x, layout.origin[0])) as usize][0]
            };
            (at(0.0, 0.0), at(0.7, 0.0))
        };
        let (under_off, open_off) = floor(&mut renderer, SdfShadows::Off);
        renderer.voxel_gi_mut().unwrap().enable_sdf();
        let (under, open) = floor(&mut renderer, SdfShadows::Fallback);
        eprintln!("under the blocker {under_off} -> {under}, in the open {open_off} -> {open}");
        assert!(under_off > 0.3 && (under_off - open_off).abs() < 0.05 * open_off, "no field: the floor is lit evenly");
        assert!(under < 0.05 * open, "the blocker's shadow keeps {under} of {open}");
        assert!((open - open_off).abs() < 0.02 * open_off, "the open floor changed: {open_off} -> {open}");
    }

    /// Voxel GI's light (the indirect view) inside a closed glowing box with a block on its floor,
    /// with the field's AO at `ao`.
    fn glowing_box_gain(ao: f32) -> Option<Vec<[f32; 4]>> {
        let mut renderer = renderer()?;
        renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Medium, bounds_min: [-1.7; 3], bounds_max: [1.7; 3], ..Default::default() });
        let gi = renderer.voxel_gi_mut().unwrap();
        gi.enable_sdf();
        let mut scene = Scene::new();
        let walls: [([f32; 3], [f32; 3]); 6] = [
            ([3.3, 0.15, 3.3], [0.0, -1.575, 0.0]),
            ([3.3, 0.15, 3.3], [0.0, 1.575, 0.0]),
            ([0.15, 3.3, 3.3], [-1.575, 0.0, 0.0]),
            ([0.15, 3.3, 3.3], [1.575, 0.0, 0.0]),
            ([3.3, 3.3, 0.15], [0.0, 0.0, -1.575]),
            ([3.3, 3.3, 0.15], [0.0, 0.0, 1.575]),
        ];
        for (size, at) in walls {
            let mut wall = block(size, at, [0.5; 3]);
            wall.gi = Some(GiSurface::new([0.5; 3]).with_emission([1.0, 0.5, 0.25]));
            scene.add(SceneNode::Renderable(wall));
        }
        // a block on the floor, for contact occlusion
        scene.add(SceneNode::Renderable(block([0.6, 0.6, 0.6], [0.5, -1.2, -0.3], [0.5; 3])));
        let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
        let mut camera = Camera::new(70.0, 0.05, 20.0, W as f32 / H as f32);
        camera.set_position(0.0, 0.5, 1.2);
        camera.look_at(&Vec3::new(0.0, -1.5, -0.2));
        camera.update_projection_matrix();
        let mut effect = VoxelGIEffect::new(renderer.voxel_gi().unwrap().volume(), VoxelGIOptions { quality: VoxelGiQuality::Medium, sdf_ao: ao, ..Default::default() });
        effect.set_sdf(renderer.voxel_gi().unwrap().sdf());
        effect.show_indirect = true;
        let output = renderer.device().create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = output.create_view(&Default::default());
        for _ in 0..24 {
            renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
            let mut encoder = renderer.device().create_command_encoder(&Default::default());
            effect.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, &gbuffer.color_view, &gbuffer.depth_view, &view, &camera, W, H);
            renderer.queue().submit(Some(encoder.finish()));
        }
        Some(super::read_image(renderer.device(), renderer.queue(), &output))
    }

    /// The field's AO darkens the floor where it meets the block and leaves the open floor alone.
    #[test]
    fn field_ao_darkens_contact_only() {
        let Some(plain) = glowing_box_gain(0.0) else { return eprintln!("no GPU adapter: skipping") };
        let ao = glowing_box_gain(1.0).unwrap();
        let ratio: Vec<f32> = plain.iter().zip(&ao).map(|(p, a)| a[0] / p[0].max(1e-4)).collect();
        let darkest = ratio.iter().cloned().fold(1.0f32, f32::min);
        // most of the image is open floor and walls: unchanged
        let unchanged = ratio.iter().filter(|r| **r > 0.98).count() as f32 / ratio.len() as f32;
        eprintln!("AO: darkest {darkest:.3} of the light, {unchanged:.3} of pixels within 2 %");
        assert!(darkest < 0.85, "no contact occlusion: darkest {darkest}");
        assert!(unchanged > 0.6, "only {unchanged} of the pixels unchanged");
    }
}

/// Mip 0 of an rgba16float 3D texture, rgba per voxel.
fn read_mip0(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<[f32; 4]> {
    let size = texture.size();
    let (w, h, d) = (size.width, size.height, size.depth_or_array_layers);
    let row = (w * 8).next_multiple_of(256);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h * d) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: d },
    );
    queue.submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let bytes = buffer.slice(..).get_mapped_range();
    let mut out = Vec::new();
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let o = ((z * h + y) * row + x * 8) as usize;
                out.push(std::array::from_fn(|c| f16_value(u16::from_le_bytes([bytes[o + 2 * c], bytes[o + 2 * c + 1]]))));
            }
        }
    }
    out
}

/// An rgba16float 2D texture, rgba per pixel.
fn read_image(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<[f32; 4]> {
    let size = texture.size();
    let (w, h) = (size.width, size.height);
    let row = (w * 8).next_multiple_of(256);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        size,
    );
    queue.submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let bytes = buffer.slice(..).get_mapped_range();
    (0..w * h)
        .map(|i| {
            let o = ((i / w) * row + (i % w) * 8) as usize;
            std::array::from_fn(|c| f16_value(u16::from_le_bytes([bytes[o + 2 * c], bytes[o + 2 * c + 1]])))
        })
        .collect()
}

fn f16_value(h: u16) -> f32 {
    let exponent = ((h >> 10) & 0x1f) as i32;
    let mantissa = (h & 0x3ff) as f32;
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

/// Particles' sun through the volume's distance field: a particle four voxels short of an opaque
/// slab is in its shadow when the sun is behind the slab, and sees it when the sun is on its side.
#[test]
fn particles_take_the_sun_from_the_volume_field() {
    use kansei_core::gi::{GiBox, ParticleGi, ParticleGiOptions, VoxelGiQuality};
    let Some((device, queue)) = gpu() else { return eprintln!("no GPU adapter: skipping") };
    let positions = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&[[2.0f32, 2.0, 2.0, 1.0]]), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX });
    let options = ParticleGiOptions { quality: VoxelGiQuality::Low, bounds_min: [0.0; 3], bounds_max: [4.0; 3], capacity: 1, ..Default::default() };
    let mut gi = ParticleGi::new(&device, options, &positions, None);
    gi.set_boxes(&queue, &[GiBox::new([2.5, -1.0, -1.0], [5.0, 5.0, 5.0], [0.0; 3], [-1.0, 0.0, 0.0])]);
    let sky = gi.sky_buffer().clone();
    gi.enable_sdf(&device, &positions, None, &sky);
    gi.settings.splat.density_per_particle = 0.0;
    gi.settings.cones.sdf_sun = true;
    gi.settings.cones.jitter_voxels = 0.0;
    gi.settings.cones.temporal_blend = 1.0;
    let mut sun = |to_sun: [f32; 3]| {
        gi.settings.set_sun(to_sun, [1.0; 3]);
        let mut encoder = device.create_command_encoder(&Default::default());
        gi.encode(&queue, &mut encoder, 1);
        let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 32, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        encoder.copy_buffer_to_buffer(gi.lighting_buffer(), 0, &staging, 0, 32);
        queue.submit(Some(encoder.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let light: Vec<f32> = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
        light[3]
    };
    let (behind, beside) = (sun([1.0, 0.0, 0.0]), sun([-1.0, 0.0, 0.0]));
    eprintln!("sun behind the slab: {behind}, on the particle's side: {beside}");
    assert!(behind < 0.02 && beside > 0.99);
}
