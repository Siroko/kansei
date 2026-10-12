//! `render=forward`'s reflections: one probe of the room, captured at start. Forward traces
//! nothing, so chrome and lacquer had only the flat fill to reflect; the probe is the room seen
//! from a point (six 90° views into a GBuffer, lit as forward lights them, copied into the
//! layers of a 2D array), box-filtered into mips that stand for rougher surfaces, and read by the
//! room's PBR material (`PROBE_WGSL`) along the reflected ray, its mip by the roughness.
//!
//! The faces are as `glam::Mat4::look_at_rh` points a camera along each axis (±Y looking a hair
//! toward -z, so the up vector is defined); `PROBE_WGSL` rebuilds the same bases to look them up.

use glam::Vec3 as GVec3;

use kansei_core::buffers::Sampler;
use kansei_core::cameras::Camera;
use kansei_core::math::Vec3;
use kansei_core::objects::Scene;
use kansei_core::renderers::{GBuffer, Renderer};

/// Texels a face side, and the mips (256 down to 1).
pub const SIZE: u32 = 256;
pub const MIPS: u32 = 9;

/// The faces' directions, in the order of the layers.
const FACES: [[f32; 3]; 6] = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, -1e-3], [0.0, -1.0, -1e-3], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];

/// The probe's lookup, for a material that binds the array at `@group(0) @binding(5)` and its
/// sampler at 6: `probe_radiance(direction, roughness)`.
pub const PROBE_WGSL: &str = r#"
@group(0) @binding(5) var probe_map: texture_2d_array<f32>;
@group(0) @binding(6) var probe_sampler: sampler;
const PROBE_MIPS: f32 = 9.0;

fn probe_face_dir(k: u32) -> vec3f {
    switch k {
        case 0u: { return vec3f(1.0, 0.0, 0.0); }
        case 1u: { return vec3f(-1.0, 0.0, 0.0); }
        case 2u: { return normalize(vec3f(0.0, 1.0, -1e-3)); }
        case 3u: { return normalize(vec3f(0.0, -1.0, -1e-3)); }
        case 4u: { return vec3f(0.0, 0.0, 1.0); }
        default: { return vec3f(0.0, 0.0, -1.0); }
    }
}

fn probe_radiance(d: vec3f, roughness: f32) -> vec3f {
    let a = abs(d);
    var k = select(5u, 4u, d.z > 0.0);
    if (a.x >= a.y && a.x >= a.z) {
        k = select(1u, 0u, d.x > 0.0);
    } else if (a.y >= a.z) {
        k = select(3u, 2u, d.y > 0.0);
    }
    // the face's camera (look_at_rh: s = f x up, u = s x f)
    let f = probe_face_dir(k);
    let s = normalize(cross(f, vec3f(0.0, 1.0, 0.0)));
    let u = cross(s, f);
    let z = max(dot(d, f), 1e-4);
    let uv = vec2f(dot(d, s) / z, -dot(d, u) / z) * 0.5 + 0.5;
    let mip = sqrt(clamp(roughness, 0.0, 1.0)) * (PROBE_MIPS - 1.0);
    return textureSampleLevel(probe_map, probe_sampler, clamp(uv, vec2f(0.001), vec2f(0.999)), i32(k), mip).rgb;
}
"#;

const DOWNSAMPLE_WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let size = textureDimensions(dst);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let p = vec2i(gid.xy) * 2;
    let l = i32(gid.z);
    let c = textureLoad(src, p, l, 0) + textureLoad(src, p + vec2i(1, 0), l, 0) + textureLoad(src, p + vec2i(0, 1), l, 0) + textureLoad(src, p + vec2i(1, 1), l, 0);
    textureStore(dst, gid.xy, l, c * 0.25);
}
"#;

/// The probe's texture (a 6-layer 2D array with mips) and what binds it.
pub struct Probe {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Probe {
    /// An empty probe (black until `capture`), to bind before the scene is built.
    pub fn new(renderer: &Renderer) -> Self {
        let texture = renderer.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("Room/Probe"),
            size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 6 },
            mip_level_count: MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        Self { texture, view }
    }

    /// The texture and its array view, for `pbr::set_probe`.
    pub fn parts(&self) -> (wgpu::Texture, wgpu::TextureView) {
        (self.texture.clone(), self.view.clone())
    }

    /// Its sampler (binding 6).
    pub fn sampler() -> Sampler {
        Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear)
    }

    /// Render the room from `at` into the probe: the six faces, then the mips. `hidden`: the
    /// renderables left out of it (the character, which moves).
    pub fn capture(&self, renderer: &mut Renderer, scene: &mut Scene, at: GVec3, hidden: &[usize]) {
        let was: Vec<bool> = hidden.iter().map(|&i| scene.get_renderable(i).is_some_and(|r| r.visible)).collect();
        for &i in hidden {
            if let Some(r) = scene.get_renderable_mut(i) {
                r.visible = false;
            }
        }
        let gbuffer = GBuffer::new(renderer.device(), SIZE, SIZE, 1);
        let mut camera = Camera::new(90.0, 0.05, 200.0, 1.0);
        camera.update_projection_matrix();
        camera.set_position(at.x, at.y, at.z);
        for (layer, f) in FACES.iter().enumerate() {
            let target = at + GVec3::from(*f);
            camera.look_at(&Vec3::new(target.x, target.y, target.z));
            renderer.render_scene_offscreen(scene, &mut camera, &gbuffer);
            let mut encoder = renderer.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Room/Probe/Copy") });
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo { texture: &gbuffer.color_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyTextureInfo { texture: &self.texture, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: layer as u32 }, aspect: wgpu::TextureAspect::All },
                wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            );
            renderer.queue().submit(std::iter::once(encoder.finish()));
        }
        self.mips(renderer.device(), renderer.queue());
        for (&i, v) in hidden.iter().zip(was) {
            if let Some(r) = scene.get_renderable_mut(i) {
                r.visible = v;
            }
        }
        gbuffer.destroy();
    }

    fn mips(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Room/Probe/Downsample"), source: wgpu::ShaderSource::Wgsl(DOWNSAMPLE_WGSL.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Room/Probe/Downsample"), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Room/Probe/Mips") });
        for mip in 1..MIPS {
            let view = |level: u32| self.texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), base_mip_level: level, mip_level_count: Some(1), ..Default::default() });
            let (src, dst) = (view(mip - 1), view(mip));
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Room/Probe/Downsample"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dst) }],
            });
            let size = (SIZE >> mip).max(1);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Room/Probe/Downsample"), timestamp_writes: None });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(size.div_ceil(8), size.div_ceil(8), 6);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
}
