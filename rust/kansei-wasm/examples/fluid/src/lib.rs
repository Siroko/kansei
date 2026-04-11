use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::cameras::Camera;
use kansei_core::controls::{CameraControls, MouseVectors};
use kansei_core::geometries::{BoxGeometry, Geometry};
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::loaders::GLTFLoader;
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Mat4, Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{PostProcessingVolume, effects::{
    DepthOfFieldEffect, DepthOfFieldOptions,
    FluidSurfaceEffect, FluidSurfaceOptions,
}};
use kansei_core::renderers::{Renderer, RendererConfig};
use wgpu::util::DeviceExt;
use kansei_core::simulations::fluid::{
    DensityFieldOptions, FluidDensityField, FluidMarchingCubes, FluidSimulation, FluidSimulationOptions,
    FluidSurfaceRenderer, MarchingCubesOptions,
};

const BASIC_LIT_WGSL: &str = include_str!("../../../../kansei-core/src/shaders/basic_lit.wgsl");

// ── Op-art stripe shader (matches engine bind group layout) ──
const STRIPE_WGSL: &str = r#"
// Group 1: Camera (engine layout — separate bindings)
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;

// Group 2: Mesh (engine layout — separate bindings with dynamic offsets)
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

// Group 0: Material params
struct StripeParams {
    color_a: vec4<f32>,
    color_b: vec4<f32>,
    thickness_a: f32,
    thickness_b: f32,
    _pad0: f32,
    _pad1: f32,
};
@group(0) @binding(0) var<uniform> params: StripeParams;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
};

@vertex
fn vertex_main(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> VOut {
    let world_pos = (world_matrix * vec4<f32>(position.xyz, 1.0)).xyz;
    let world_normal = normalize((normal_matrix * vec4<f32>(normal, 0.0)).xyz);
    var out: VOut;
    out.clip_pos = projection_matrix * view_matrix * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.world_normal = world_normal;
    return out;
}

@fragment
fn fragment_main(v: VOut) -> @location(0) vec4<f32> {
    let period = params.thickness_a + params.thickness_b;
    let t = ((v.world_pos.x % period) + period) % period;
    let in_a = t < params.thickness_a;
    let base = select(params.color_b.rgb, params.color_a.rgb, in_a);

    let light = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let ndotl = max(dot(normalize(v.world_normal), light), 0.0);
    let lit = base * (0.3 + ndotl * 0.7);
    return vec4<f32>(lit, 1.0);
}
"#;

// ── Particle billboard shader (custom — not a standard Renderable) ──
const PARTICLE_WGSL: &str = r#"
struct P { view: mat4x4<f32>, proj: mat4x4<f32>, size: f32, }
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> p: P;
struct V { @builtin(position) pos: vec4<f32>, @location(0) col: vec3<f32>, }
const Q: array<vec2<f32>,6> = array(vec2(-1.,-1.),vec2(1.,-1.),vec2(1.,1.),vec2(-1.,-1.),vec2(1.,1.),vec2(-1.,1.));
@vertex fn vs(@builtin(vertex_index) vi: u32) -> V {
    let pid=vi/6u; let c=Q[vi%6u]; let pos=positions[pid]; let s=p.size;
    let r=vec3<f32>(p.view[0][0],p.view[1][0],p.view[2][0]);
    let u=vec3<f32>(p.view[0][1],p.view[1][1],p.view[2][1]);
    let wp=pos.xyz+r*c.x*s+u*c.y*s;
    var o:V; o.pos=p.proj*p.view*vec4<f32>(wp,1.);
    let t=clamp((pos.y+8.)/16.,0.,1.);
    o.col=mix(vec3<f32>(0.1,0.3,0.8),vec3<f32>(0.8,0.95,1.0),t); return o;
}
@fragment fn fs(v:V)->@location(0) vec4<f32>{return vec4<f32>(v.col,1.);}
"#;

// ── Blit shader (fullscreen triangle) ──
const BLIT_WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
    return vec4<f32>(pos[vi], 0.0, 1.0);
}
struct FOut { @location(0) color: vec4<f32>, }
@fragment fn fs(@builtin(position) frag: vec4<f32>) -> FOut {
    let dims = vec2<f32>(textureDimensions(src));
    var out: FOut; out.color = textureSample(src, samp, frag.xy / dims); return out;
}
"#;

// ── Transmission/refraction shader ──
const TRANSMISSION_WGSL: &str = r#"
// Group 0: Material
struct TransmissionParams {
    color: vec4<f32>,       // fluid tint color
    ior: f32,               // index of refraction
    chromatic_aberration: f32,
    tint_strength: f32,
    fresnel_power: f32,
    roughness: f32,
    thickness: f32,         // refraction offset scale
    _pad0: f32,
    _pad1: f32,
};
@group(0) @binding(0) var<uniform> params: TransmissionParams;
@group(0) @binding(1) var background_tex: texture_2d<f32>;
@group(0) @binding(2) var background_sampler: sampler;

// Group 1: Camera
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;

// Group 2: Mesh
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

// Group 3: Shadow (required by pipeline layout, unused here)
@group(3) @binding(0) var shadow_depth_tex: texture_depth_2d;
@group(3) @binding(1) var shadow_sampler: sampler_comparison;
@group(3) @binding(2) var<uniform> shadow_uniforms: vec4<f32>;
@group(3) @binding(3) var cube_shadow_tex: texture_2d_array<f32>;
@group(3) @binding(4) var cube_shadow_sampler: sampler;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
};

@vertex
fn vertex_main(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> VOut {
    let wp = (world_matrix * vec4<f32>(position.xyz, 1.0)).xyz;
    let wn = normalize((normal_matrix * vec4<f32>(normal, 0.0)).xyz);
    var out: VOut;
    out.clip_pos = projection_matrix * view_matrix * vec4<f32>(wp, 1.0);
    out.world_pos = wp;
    out.world_normal = wn;
    return out;
}

@fragment
fn fragment_main(v: VOut) -> @location(0) vec4<f32> {
    let N = normalize(v.world_normal);
    let dims = vec2<f32>(textureDimensions(background_tex));
    let screen_uv = v.clip_pos.xy / dims;

    // View direction (camera is at the origin of view space; extract from inverse view)
    let cam_pos = vec3<f32>(
        view_matrix[0][3] + view_matrix[3][0],
        view_matrix[1][3] + view_matrix[3][1],
        view_matrix[2][3] + view_matrix[3][2],
    );
    // Simpler: reconstruct from view matrix columns
    let eye = -vec3<f32>(
        dot(view_matrix[0].xyz, view_matrix[3].xyz),
        dot(view_matrix[1].xyz, view_matrix[3].xyz),
        dot(view_matrix[2].xyz, view_matrix[3].xyz),
    );
    let V = normalize(eye - v.world_pos);

    // Fresnel (Schlick approximation)
    let f0 = pow((1.0 - params.ior) / (1.0 + params.ior), 2.0);
    let ndotv = max(dot(N, V), 0.0);
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - ndotv, params.fresnel_power);

    // Refraction offset in screen space
    let refract_strength = params.thickness * (1.0 - 1.0 / params.ior);
    let offset = N.xy * refract_strength * 0.05;

    // Chromatic aberration — scale offset differently per channel
    let ca = params.chromatic_aberration;
    let bg_r = textureSample(background_tex, background_sampler, screen_uv + offset * (1.0 + ca)).r;
    let bg_g = textureSample(background_tex, background_sampler, screen_uv + offset).g;
    let bg_b = textureSample(background_tex, background_sampler, screen_uv + offset * (1.0 - ca)).b;
    var refracted = vec3<f32>(bg_r, bg_g, bg_b);

    // Apply tint
    refracted *= mix(vec3<f32>(1.0), params.color.rgb, params.tint_strength);

    // GGX specular (physically-based)
    let alpha = params.roughness * params.roughness;
    let a2 = alpha * alpha;

    // Key light
    let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let H = normalize(V + light_dir);
    let ndoth = max(dot(N, H), 0.0);
    let ndotl = max(dot(N, light_dir), 0.0);

    // GGX distribution
    let denom = ndoth * ndoth * (a2 - 1.0) + 1.0;
    let D = a2 / (3.14159 * denom * denom + 0.0001);

    // Geometric attenuation (Smith-Schlick)
    let k = alpha * 0.5;
    let G = (ndotv / (ndotv * (1.0 - k) + k)) * (ndotl / (ndotl * (1.0 - k) + k));

    let spec_color = vec3<f32>(fresnel);
    let specular = spec_color * D * G * ndotl;

    // Rim light (subtle edge glow from environment)
    let rim = pow(1.0 - ndotv, 3.0) * 0.15;

    // Blend: refracted color + specular + rim via Fresnel
    let result = refracted * (1.0 - fresnel) + specular + vec3<f32>(rim);

    return vec4<f32>(result, 1.0);
}
"#;

// ── MC surface shader: outputs color + world-space normal to GBuffer ──
const MC_SURFACE_WGSL: &str = r#"
struct Params {
    color: vec4<f32>,
    specular: vec4<f32>,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
@group(3) @binding(0) var shadow_depth_tex: texture_depth_2d;
@group(3) @binding(1) var shadow_sampler: sampler_comparison;
@group(3) @binding(2) var<uniform> shadow_uniforms: vec4<f32>;
@group(3) @binding(3) var cube_shadow_tex: texture_2d_array<f32>;
@group(3) @binding(4) var cube_shadow_sampler: sampler;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
};

@vertex
fn vertex_main(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> VOut {
    let wp = (world_matrix * vec4<f32>(position.xyz, 1.0)).xyz;
    let wn = normalize((normal_matrix * vec4<f32>(normal, 0.0)).xyz);
    var out: VOut;
    out.clip_pos = projection_matrix * view_matrix * vec4<f32>(wp, 1.0);
    out.world_pos = wp;
    out.world_normal = wn;
    return out;
}

struct FragOut {
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) albedo: vec4<f32>,
};

@fragment
fn fragment_main(v: VOut) -> FragOut {
    let n = normalize(v.world_normal);
    let light = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let ndotl = max(dot(n, light), 0.0);
    let base = params.color.rgb;
    let lit = base * (0.2 + ndotl * 0.8);

    var out: FragOut;
    out.color = vec4<f32>(lit, 1.0);
    out.emissive = vec4<f32>(0.0);
    // Write world normal (normalized, with small epsilon so refraction detection works)
    out.normal = vec4<f32>(n, 1.0);
    out.albedo = vec4<f32>(base, 1.0);
    return out;
}
"#;

// ── Cubemap capture shader: renders dome stripes to a single color target ──
const CUBEMAP_DOME_WGSL: &str = r#"
struct CaptureParams {
    view_proj: mat4x4<f32>,
    world: mat4x4<f32>,
};
struct StripeParams {
    color_a: vec4<f32>,
    color_b: vec4<f32>,
    thickness_a: f32,
    thickness_b: f32,
    _pad0: f32,
    _pad1: f32,
};
@group(0) @binding(0) var<uniform> capture: CaptureParams;
@group(0) @binding(1) var<uniform> stripes: StripeParams;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@vertex
fn vs(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> VOut {
    let wp = (capture.world * vec4<f32>(position.xyz, 1.0)).xyz;
    var out: VOut;
    out.clip_pos = capture.view_proj * vec4<f32>(wp, 1.0);
    out.world_pos = wp;
    return out;
}

@fragment
fn fs(v: VOut) -> @location(0) vec4<f32> {
    let period = stripes.thickness_a + stripes.thickness_b;
    let t = ((v.world_pos.x % period) + period) % period;
    let in_a = t < stripes.thickness_a;
    let base = select(stripes.color_b.rgb, stripes.color_a.rgb, in_a);
    return vec4<f32>(base, 1.0);
}
"#;

const PARTICLE_MSAA_SAMPLES: u32 = 4;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
    log::info!("Kansei WASM initialized");
}

// ── Helper: create a basic lit material ──
fn make_lit_material(name: &str, color: [f32; 4], opts: MaterialOptions) -> Material {
    let uniform: [f32; 8] = [
        color[0], color[1], color[2], color[3],
        0.15, 0.15, 0.15, 0.5, // specular
    ];
    let mut mat = Material::new(
        name, BASIC_LIT_WGSL,
        vec![Binding::uniform(0, ShaderStages::FRAGMENT)],
        opts,
    );
    mat.set_uniform_bindable(0, &format!("{name}/Color"), &uniform);
    mat
}

// ── Helper: build cornell box walls as standard renderables ──
fn front_cull_opts() -> MaterialOptions {
    let mut opts = MaterialOptions::default();
    opts.cull_mode = CullMode::Front;
    opts
}

fn build_cornell_box(scene: &mut Scene, bounds_min: [f32; 3], bounds_max: [f32; 3]) {
    let [x0, y0, z0] = bounds_min;
    let [x1, y1, z1] = bounds_max;
    let sx = x1 - x0;
    let sy = y1 - y0;
    let sz = z1 - z0;
    let cx = (x0 + x1) * 0.5;
    let cy = (y0 + y1) * 0.5;
    let cz = (z0 + z1) * 0.5;

    let mut floor = Renderable::new(BoxGeometry::new(sx, 0.1, sz),
        make_lit_material("Floor", [0.7, 0.7, 0.7, 1.0], front_cull_opts()));
    floor.object.set_position(cx, y0 - 0.05, cz);
    scene.add(SceneNode::Renderable(floor));

    let mut ceil = Renderable::new(BoxGeometry::new(sx, 0.1, sz),
        make_lit_material("Ceiling", [0.7, 0.7, 0.7, 1.0], front_cull_opts()));
    ceil.object.set_position(cx, y1 + 0.05, cz);
    scene.add(SceneNode::Renderable(ceil));

    let mut back = Renderable::new(BoxGeometry::new(sx, sy, 0.1),
        make_lit_material("BackWall", [0.7, 0.7, 0.7, 1.0], front_cull_opts()));
    back.object.set_position(cx, cy, z0 - 0.05);
    scene.add(SceneNode::Renderable(back));

    let mut left = Renderable::new(BoxGeometry::new(0.1, sy, sz),
        make_lit_material("LeftWall", [0.8, 0.15, 0.1, 1.0], front_cull_opts()));
    left.object.set_position(x0 - 0.05, cy, cz);
    scene.add(SceneNode::Renderable(left));

    let mut right = Renderable::new(BoxGeometry::new(0.1, sy, sz),
        make_lit_material("RightWall", [0.15, 0.8, 0.1, 1.0], front_cull_opts()));
    right.object.set_position(x1 + 0.05, cy, cz);
    scene.add(SceneNode::Renderable(right));

    let mut front = Renderable::new(BoxGeometry::new(sx, sy, 0.1),
        make_lit_material("FrontWall", [0.7, 0.7, 0.7, 1.0], front_cull_opts()));
    front.object.set_position(cx, cy, z1 + 0.05);
    scene.add(SceneNode::Renderable(front));
}

// ── Helper: create MC surface renderable (placeholder — no buffer ptrs yet) ──
// Writes color + world normal to GBuffer so FluidSurfaceEffect can detect and refract it.
fn build_mc_renderable_placeholder() -> Renderable {
    let geo = Geometry::new_indirect_placeholder("MC/Surface");
    let mut opts = MaterialOptions::default();
    opts.cull_mode = CullMode::None;
    opts.mrt_output_count = Some(4); // shader writes to all 4 GBuffer targets

    let uniform_data: [f32; 8] = [
        0.77, 0.96, 1.0, 1.0, // color
        0.15, 0.15, 0.15, 0.5, // specular
    ];
    let mut mat = Material::new(
        "MC/Surface", MC_SURFACE_WGSL,
        vec![Binding::uniform(0, ShaderStages::FRAGMENT)],
        opts,
    );
    mat.set_uniform_bindable(0, "MC/SurfaceParams", &uniform_data);

    let mut r = Renderable::new(geo, mat);
    r.visible = false;
    r
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = canvas_id;
        return Err(JsValue::from_str("kansei-wasm start() is only supported on wasm32"));
    }

    #[cfg(target_arch = "wasm32")]
    {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document.get_element_by_id(canvas_id)
        .ok_or("Canvas not found")?.dyn_into::<web_sys::HtmlCanvasElement>()?;

    let width = canvas.client_width() as u32;
    let height = canvas.client_height() as u32;
    canvas.set_width(width);
    canvas.set_height(height);

    let mut renderer = Renderer::new(RendererConfig {
        width, height,
        sample_count: 4,
        ..Default::default()
    });
    renderer.initialize_with_canvas(canvas.clone()).await;
    let format = renderer.presentation_format();

    // ── Particles ──
    // Spread in an ellipsoid centered on the sim bounds (~70% of bounds extent).
    let count = 75_000usize;
    let center = [0.0f32, 11.0, 0.0]; // (min+max)/2 of bounds
    let half = [22.0f32, 17.0, 14.0]; // ~90% of bounds half-extent (more spread → less pressure)
    let mut positions = vec![0.0f32; count * 4];
    let mut rng: u64 = 12345;
    for i in 0..count {
        loop {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let ux = (rng as f32 / u64::MAX as f32) * 2.0 - 1.0;
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let uy = (rng as f32 / u64::MAX as f32) * 2.0 - 1.0;
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let uz = (rng as f32 / u64::MAX as f32) * 2.0 - 1.0;
            if ux*ux + uy*uy + uz*uz <= 1.0 {
                positions[i*4]   = center[0] + ux * half[0];
                positions[i*4+1] = center[1] + uy * half[1];
                positions[i*4+2] = center[2] + uz * half[2];
                positions[i*4+3] = 1.0;
                break;
            }
        }
    }

    // ── Sim ──
    let mut sim = FluidSimulation::new(&renderer, FluidSimulationOptions {
        max_particles: count as u32, dimensions: 3, smoothing_radius: 1.0,
        pressure_multiplier: 46.5, near_pressure_multiplier: 20.0, density_target: 8.6,
        viscosity: 1.0, damping: 0.997, gravity: [0.0, -12.8, 0.0],
        mouse_force: 950.0, substeps: 2, world_bounds_padding: 0.3,
        ..kansei_core::simulations::fluid::DEFAULT_OPTIONS
    }, &positions);
    sim.world_bounds_min = [-25.0, -8.0, -16.0];
    sim.world_bounds_max = [25.0, 30.0, 16.0];
    sim.rebuild_grid();

    // ── Density field + surface renderer (for raymarch mode) ──
    let density_field = FluidDensityField::new(&renderer, sim.positions_buffer().unwrap(),
        sim.world_bounds_min, sim.world_bounds_max,
        DensityFieldOptions { resolution: 128, kernel_scale: 0.6 });
    let surface_renderer = FluidSurfaceRenderer::new(&renderer);

    // ── Marching cubes (compute only — render via standard Renderable) ──
    let marching_cubes = FluidMarchingCubes::new(&renderer, MarchingCubesOptions {
        max_triangles: 1_000_000,
        iso_level: 0.05,
    });
    let marching_cubes_bg = marching_cubes.create_bind_group(&renderer, &density_field.density_view);

    // ── Build scene with standard Renderables ──
    let mut scene = Scene::new();

    // Dome mesh (GLB) — loaded via GLTFLoader, added as standard Renderable
    let dome_loaded = async {
        let window = web_sys::window().unwrap();
        let resp = wasm_bindgen_futures::JsFuture::from(
            window.fetch_with_str("assets/dome.glb"),
        ).await.ok()?;
        let resp: web_sys::Response = resp.dyn_into().ok()?;
        let buf = wasm_bindgen_futures::JsFuture::from(resp.array_buffer().ok()?).await.ok()?;
        let bytes = js_sys::Uint8Array::new(&buf).to_vec();
        GLTFLoader::load_glb(&bytes).ok()
    }.await;
    let mut dome_scene_index = None;
    if let Some(result) = dome_loaded {
        log::info!("Loaded dome: {} renderables", result.renderables.len());
        for gr in result.renderables {
            let s = 20.0;
            let mut opts = MaterialOptions::default();
            opts.cull_mode = CullMode::Front;

            let stripe_data: [f32; 12] = [
                0.0, 0.0, 0.0, 1.0,   // color_a (black)
                1.0, 1.0, 1.0, 1.0,   // color_b (white)
                1.0, 1.0,             // thickness_a, thickness_b
                0.0, 0.0,             // padding
            ];
            let mut mat = Material::new(
                "Dome/Stripes", STRIPE_WGSL,
                vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)],
                opts,
            );
            mat.set_uniform_bindable(0, "Dome/StripeParams", &stripe_data);

            let mut r = Renderable::new(gr.geometry, mat);
            r.object.position = Vec3::new(gr.position.x, gr.position.y + 3.0, gr.position.z);
            r.object.rotation = gr.rotation;
            r.object.scale = Vec3::new(gr.scale.x * s, gr.scale.y * s, gr.scale.z * s);
            r.object.update_model_matrix();
            r.object.update_world_matrix(None);
            dome_scene_index = Some(scene.add(SceneNode::Renderable(r)));
        }
    }

    // MC surface renderable — placeholder geometry, pointers set after State is stable
    let mc_renderable = build_mc_renderable_placeholder();
    let mc_scene_index = scene.add(SceneNode::Renderable(mc_renderable));

    // Light
    let sun = DirectionalLight::new(
        Vec3::new(0.3, -1.0, 0.5).normalize(),
        Vec3::new(1.0, 1.0, 1.0), 2.0,
    );
    let light_scene_index = scene.add(SceneNode::Light(Light::Directional(sun)));

    // ── Camera ──
    let mut camera = Camera::new(45.0, 0.1, 1000.0, width as f32 / height as f32);
    camera.set_position(0.0, 20.0, 75.0);
    camera.look_at(&Vec3::new(0.0, 3.0, 0.0));
    camera.update_projection_matrix();
    camera.update_view_matrix();

    // ── Particle pipeline (must be created BEFORE sim moves into effect) ──
    let particle_shader = renderer.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Particles"), source: wgpu::ShaderSource::Wgsl(PARTICLE_WGSL.into()),
    });
    let particle_params_buf = renderer.device().create_buffer(&wgpu::BufferDescriptor {
        label: None, size: 144,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let particle_pipeline = renderer.device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Particles"), layout: None,
        vertex: wgpu::VertexState { module: &particle_shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState { module: &particle_shader, entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default() }),
        primitive: Default::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual, stencil: Default::default(), bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState { count: PARTICLE_MSAA_SAMPLES, ..Default::default() },
        multiview: None, cache: None,
    });
    let particle_bg = renderer.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &particle_pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: sim.positions_buffer().unwrap().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: particle_params_buf.as_entire_binding() },
        ],
    });

    // ── Post-processing: FluidSurface (compute + refraction) → DoF ──
    // Must be created after particle_bg (which references sim.positions_buffer)
    // but before offscreen textures (which reference density_field.density_view).
    // Create surface_bg first since it needs density_field.density_view.

    // ── Blit pipeline (for raymarch mode) ──
    let blit_shader = renderer.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Blit"), source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
    });
    let blit_bgl = renderer.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None, entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
        ],
    });
    let blit_pipeline = renderer.device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Blit"),
        layout: Some(&renderer.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&blit_bgl], push_constant_ranges: &[] })),
        vertex: wgpu::VertexState { module: &blit_shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState { module: &blit_shader, entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default() }),
        primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
    });
    let blit_sampler = renderer.device().create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default()
    });

    // ── Offscreen textures (for raymarch + particle modes) ──
    let mk_tex = |label: &str, fmt: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
        let tex = renderer.device().create_texture(&wgpu::TextureDescriptor {
            label: Some(label), size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: fmt, usage, view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        (tex, view)
    };
    let cu = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;
    let (_color_tex, color_view) = mk_tex("Color", wgpu::TextureFormat::Rgba16Float, cu);
    let (_depth_tex, depth_view) = mk_tex("Depth", wgpu::TextureFormat::Depth32Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
    let (_output_tex, output_view) = mk_tex("Output", wgpu::TextureFormat::Rgba16Float, cu);
    let particle_msaa_tex = renderer.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("ParticleMSAA"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: PARTICLE_MSAA_SAMPLES,
        dimension: wgpu::TextureDimension::D2, format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
    });
    let particle_msaa_view = particle_msaa_tex.create_view(&Default::default());
    let particle_depth_tex = renderer.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("ParticleDepth"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: PARTICLE_MSAA_SAMPLES,
        dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
    });
    let particle_depth_view = particle_depth_tex.create_view(&Default::default());

    let surface_bg = surface_renderer.create_bind_group(&color_view, &depth_view, &output_view, &density_field.density_view);
    let blit_bg = renderer.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &blit_bgl, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&output_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&blit_sampler) },
        ],
    });

    // Now move sim, density_field, marching_cubes into the FluidSurfaceEffect
    let volume = PostProcessingVolume::new(
        &renderer,
        vec![
            Box::new(FluidSurfaceEffect::new(
                sim, density_field, marching_cubes, marching_cubes_bg,
                FluidSurfaceOptions::default(),
            )),
            Box::new(DepthOfFieldEffect::new(DepthOfFieldOptions {
                focus_distance: 42.0,
                focus_range: 37.0,
                max_blur: 7.0,
            })),
        ],
    );

    // Camera: back view aligned with long X axis (azimuth = π), radius 30
    let mut controls = CameraControls::from_canvas(&canvas, Vec3::new(0.0, 3.0, 0.0), 30.0);
    controls.set_azimuth(std::f32::consts::PI);
    let mouse = MouseVectors::from_canvas(&canvas);

    let perf_now = window.performance().map(|p| p.now()).unwrap_or(0.0);
    let state = Rc::new(RefCell::new(State {
        renderer, scene, camera, controls, mouse, volume,
        surface_renderer,
        mc_scene_index, dome_scene_index, light_scene_index,
        particle_pipeline, particle_bg, particle_params_buf,
        blit_pipeline, blit_bg, surface_bg,
        color_view, depth_view, output_view,
        particle_msaa_view, particle_depth_view,
        count: count as u32, width, height,
        particle_size: 0.15, show_particles: true, render_mode: 0, mc_iso_level: 0.05,
        use_batched_sim: true,
        sim_accumulator: 0.0, sim_dt_step: 1.0 / 60.0, sim_time_scale: 1.0, max_sim_steps: 4,
        max_render_fps: 0.0, render_accumulator: 0.0,
        frame_count: 0, frame_time_sum: 0.0, last_perf_time: perf_now,
        current_fps: 0.0, current_frame_ms: 0.0,
    }));

    GLOBAL_STATE.with(|gs| { *gs.borrow_mut() = Some(state.clone()); });

    // Now State is at its final heap address — fix up MC buffer pointers.
    // Access MC through the FluidSurfaceEffect in the volume.
    {
        let st = state.borrow();
        let fse = st.volume.effects[0].as_any().downcast_ref::<FluidSurfaceEffect>().unwrap();
        let vb = fse.marching_cubes.vertex_buffer() as *const wgpu::Buffer;
        let ib = fse.marching_cubes.index_buffer() as *const wgpu::Buffer;
        let ab = fse.marching_cubes.indirect_args_buffer() as *const wgpu::Buffer;
        let idx = st.mc_scene_index;
        drop(st);
        let mut st = state.borrow_mut();
        if let Some(r) = st.scene.get_renderable_mut(idx) {
            unsafe { r.geometry.set_external_buffers(vb, ib, Some(ab)); }
        }
    }

    // Animation loop
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone(); let s = state.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        s.borrow_mut().render_frame();
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());

    log::info!("Kansei WASM — {} particles, [toggle via tweakpane]", count);
    Ok(())
    }
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window().unwrap().request_animation_frame(f.as_ref().unchecked_ref()).unwrap();
}

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    mouse: MouseVectors,
    volume: PostProcessingVolume,
    surface_renderer: FluidSurfaceRenderer,
    mc_scene_index: usize,
    dome_scene_index: Option<usize>,
    light_scene_index: usize,
    // Custom pipelines for particle + raymarch modes
    particle_pipeline: wgpu::RenderPipeline, particle_bg: wgpu::BindGroup, particle_params_buf: wgpu::Buffer,
    blit_pipeline: wgpu::RenderPipeline, blit_bg: wgpu::BindGroup, surface_bg: wgpu::BindGroup,
    color_view: wgpu::TextureView, depth_view: wgpu::TextureView, output_view: wgpu::TextureView,
    particle_msaa_view: wgpu::TextureView, particle_depth_view: wgpu::TextureView,
    count: u32, width: u32, height: u32,
    particle_size: f32, show_particles: bool, render_mode: u32, mc_iso_level: f32,
    use_batched_sim: bool,
    sim_accumulator: f64, sim_dt_step: f32, sim_time_scale: f32, max_sim_steps: u32,
    max_render_fps: f64, render_accumulator: f64,
    frame_count: u32, frame_time_sum: f64, last_perf_time: f64,
    current_fps: f64, current_frame_ms: f64,
}

impl State {
    fn render_frame(&mut self) {
        let perf = web_sys::window().unwrap().performance().unwrap();
        let now = perf.now();
        let frame_ms = (now - self.last_perf_time).max(0.0);
        self.last_perf_time = now;

        self.render_accumulator += frame_ms * 0.001;
        let target_render_dt = if self.max_render_fps > 0.0 { 1.0 / self.max_render_fps } else { 0.0 };
        if target_render_dt > 0.0 && self.render_accumulator < target_render_dt { return; }
        let render_dt = if target_render_dt > 0.0 {
            let dt = self.render_accumulator; self.render_accumulator = 0.0; dt
        } else { frame_ms * 0.001 };

        self.frame_time_sum += render_dt * 1000.0;
        self.frame_count += 1;
        if self.frame_count % 60 == 0 {
            let avg = self.frame_time_sum / 60.0;
            self.current_frame_ms = avg;
            self.current_fps = 1000.0 / avg;
            self.frame_time_sum = 0.0;
        }

        let frame_dt = render_dt.max(1.0 / 1000.0);
        self.controls.update(&mut self.camera, 0.0);
        self.mouse.update(frame_dt as f32);
        self.camera.aspect = self.width as f32 / self.height as f32;
        self.camera.update_projection_matrix();

        let eye = self.camera.position();
        let view = self.camera.view_matrix.to_glam();
        let proj = self.camera.projection_matrix.to_glam();
        let inv_view = self.camera.inverse_view_matrix.to_glam();
        let inv_vp = (proj * view).inverse();

        let mouse_ndc = [self.mouse.position.x, self.mouse.position.y];
        let mouse_dir = [self.mouse.direction.x, self.mouse.direction.y];
        let mouse_strength = self.mouse.strength.min(1.0);

        // Step fluid simulation (owned by FluidSurfaceEffect in the volume).
        //
        // True framerate-independent sim via a fixed-step accumulator: drain
        // the accumulator in chunks of exactly `sim_dt_step * sim_time_scale`
        // so the sim evolves identically at 30, 33, 60, 90, 144 fps.
        //
        // `update_batched` already encodes all substeps into a single submit
        // with no CPU-GPU sync, so running 2-3 steps on a slow frame costs
        // ~2-3× compute but almost zero CPU overhead. The `max_sim_steps`
        // cap prevents the classic spiral of death — extra leftover time is
        // discarded rather than accumulated forever.
        let identity = glam::Mat4::IDENTITY.to_cols_array();
        if let Some(fse) = self.volume.effects.get_mut(0)
            .and_then(|e| e.as_any_mut().downcast_mut::<FluidSurfaceEffect>())
        {
            fse.sim.set_camera_matrices(&view.to_cols_array(), &proj.to_cols_array(), &inv_view.to_cols_array(), &identity);
            let step_dt  = self.sim_dt_step.clamp(1.0 / 240.0, 1.0 / 20.0);
            let scale    = self.sim_time_scale.clamp(0.1, 4.0);
            let scaled_dt = step_dt * scale;
            self.sim_accumulator += frame_dt * scale as f64;
            let mut steps = 0u32;
            while self.sim_accumulator >= step_dt as f64 && steps < self.max_sim_steps {
                fse.step_simulation(scaled_dt, mouse_strength, mouse_ndc, mouse_dir, self.use_batched_sim);
                self.sim_accumulator -= step_dt as f64;
                steps += 1;
            }
            // Drop excess if we're running slower than the sim needs — keeps
            // the next frame from inheriting a huge backlog.
            if self.sim_accumulator > step_dt as f64 {
                self.sim_accumulator = step_dt as f64;
            }
        }

        // ── Render mode 0: Particles (custom pipeline) ──
        if self.render_mode == 0 {
            let mut data = [0.0f32; 36];
            data[..16].copy_from_slice(&view.to_cols_array());
            data[16..32].copy_from_slice(&proj.to_cols_array());
            data[32] = self.particle_size;
            self.renderer.queue().write_buffer(&self.particle_params_buf, 0, bytemuck::cast_slice(&data));

            let output = self.renderer.surface().unwrap().get_current_texture().unwrap();
            let canvas_view = output.texture.create_view(&Default::default());
            let mut encoder = self.renderer.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("WasmFluid/Particles"),
            });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.particle_msaa_view, resolve_target: Some(&canvas_view),
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.04, a: 1.0 }),
                            store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.particle_depth_view,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                }).forget_lifetime();
                pass.set_pipeline(&self.particle_pipeline);
                pass.set_bind_group(0, &self.particle_bg, &[]);
                pass.draw(0..self.count * 6, 0..1);
            }
            self.renderer.submit(std::iter::once(encoder.finish()));
            output.present();

        // ── Render mode 1: Raymarch (custom compute + blit) ──
        } else if self.render_mode == 1 {
            let output = self.renderer.surface().unwrap().get_current_texture().unwrap();
            let canvas_view = output.texture.create_view(&Default::default());
            let mut encoder = self.renderer.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("WasmFluid/Raymarch"),
            });

            let fse = self.volume.effects[0].as_any_mut().downcast_mut::<FluidSurfaceEffect>().unwrap();
            fse.density_field.update_with_encoder(&mut encoder,
                fse.sim.world_bounds_min, fse.sim.world_bounds_max,
                fse.sim.particle_count(), fse.sim.params.smoothing_radius);

            // Clear offscreen color + depth (raymarch reads depth to know geometry)
            {
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.color_view, resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.04, a: 1.0 }),
                            store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }), ..Default::default()
                });
            }

            let inv_vp = Mat4::from(inv_vp);
            let fse = self.volume.effects[0].as_any_mut().downcast_mut::<FluidSurfaceEffect>().unwrap();
            let bounds_min = fse.sim.world_bounds_min;
            let bounds_max = fse.sim.world_bounds_max;
            self.surface_renderer.render(&mut encoder, &self.surface_bg,
                &inv_vp, [eye.x, eye.y, eye.z],
                bounds_min, bounds_max,
                self.width, self.height);

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &canvas_view, resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })], ..Default::default()
                }).forget_lifetime();
                pass.set_pipeline(&self.blit_pipeline);
                pass.set_bind_group(0, &self.blit_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            self.renderer.submit(std::iter::once(encoder.finish()));
            output.present();

        // ── Render mode 2/3: MC surface via standard Renderer ──
        // FluidSurfaceEffect handles density + MC compute + refraction composite.
        // DoF runs after. All orchestrated by render_with_postprocessing.
        } else {
            self.renderer.render_with_postprocessing(
                &mut self.scene, &mut self.camera, &mut self.volume,
            );
        }
    }
}

// ── JS interop ──
thread_local! { static GLOBAL_STATE: RefCell<Option<Rc<RefCell<State>>>> = RefCell::new(None); }
fn with_state<F: FnOnce(&mut State)>(f: F) { GLOBAL_STATE.with(|gs| { if let Some(ref rc) = *gs.borrow() { f(&mut rc.borrow_mut()); } }); }
fn with_fluid<F: FnOnce(&mut FluidSurfaceEffect)>(f: F) {
    with_state(|s| {
        if let Some(fse) = s.volume.effects.get_mut(0)
            .and_then(|e| e.as_any_mut().downcast_mut::<FluidSurfaceEffect>()) { f(fse); }
    });
}

#[wasm_bindgen] pub fn get_fps() -> f64 {
    GLOBAL_STATE.with(|gs| {
        if let Some(ref rc) = *gs.borrow() {
            rc.try_borrow().map(|s| s.current_fps).unwrap_or(0.0)
        } else { 0.0 }
    })
}
#[wasm_bindgen] pub fn get_frame_time() -> f64 {
    GLOBAL_STATE.with(|gs| {
        if let Some(ref rc) = *gs.borrow() {
            rc.try_borrow().map(|s| s.current_frame_ms).unwrap_or(0.0)
        } else { 0.0 }
    })
}

#[wasm_bindgen] pub fn set_pressure(v: f32) { with_fluid(|f| f.sim.params.pressure_multiplier = v); }
#[wasm_bindgen] pub fn set_near_pressure(v: f32) { with_fluid(|f| f.sim.params.near_pressure_multiplier = v); }
#[wasm_bindgen] pub fn set_density_target(v: f32) { with_fluid(|f| f.sim.params.density_target = v); }
#[wasm_bindgen] pub fn set_viscosity(v: f32) { with_fluid(|f| f.sim.params.viscosity = v); }
#[wasm_bindgen] pub fn set_damping(v: f32) { with_fluid(|f| f.sim.params.damping = v); }
#[wasm_bindgen] pub fn set_gravity_y(v: f32) { with_fluid(|f| f.sim.params.gravity[1] = v); }
#[wasm_bindgen] pub fn set_radial_gravity(enabled: bool) {
    with_fluid(|f| f.sim.params.radial_gravity = enabled);
}
#[wasm_bindgen] pub fn set_gravity_center(x: f32, y: f32, z: f32) {
    with_fluid(|f| f.sim.params.gravity_center = [x, y, z]);
}
#[wasm_bindgen] pub fn set_mouse_force(v: f32) { with_fluid(|f| f.sim.params.mouse_force = v); }
#[wasm_bindgen] pub fn set_mouse_radius(v: f32) { with_fluid(|f| f.sim.params.mouse_radius = v); }
#[wasm_bindgen] pub fn set_particle_size(v: f32) { with_state(|s| s.particle_size = v); }
#[wasm_bindgen] pub fn set_substeps(v: u32) { with_fluid(|f| f.sim.params.substeps = v); }
#[wasm_bindgen] pub fn set_show_particles(v: bool) {
    with_state(|s| { s.show_particles = v; s.render_mode = if v { 0 } else { 2 }; });
}
#[wasm_bindgen] pub fn set_render_mode(v: u32) {
    with_state(|s| {
        s.render_mode = v.min(3);
        s.show_particles = s.render_mode == 0;
        if let Some(fse) = s.volume.effects.get_mut(0)
            .and_then(|e| e.as_any_mut().downcast_mut::<FluidSurfaceEffect>()) {
            fse.marching_cubes.set_use_classic(s.render_mode == 3);
        }
        if let Some(r) = s.scene.get_renderable_mut(s.mc_scene_index) {
            r.visible = s.render_mode >= 2;
        }
    });
}
#[wasm_bindgen] pub fn set_mc_iso_level(v: f32) {
    with_state(|s| {
        let iso = v.max(0.0);
        s.mc_iso_level = iso;
        s.surface_renderer.density_threshold = iso;
        if let Some(fse) = s.volume.effects.get_mut(0)
            .and_then(|e| e.as_any_mut().downcast_mut::<FluidSurfaceEffect>()) {
            fse.marching_cubes.set_iso_level(iso);
        }
    });
}
#[wasm_bindgen] pub fn set_mc_resolution(v: u32) {
    with_fluid(|f| {
        use kansei_core::simulations::fluid::MarchingCubesGridSizing;
        let mut p = f.marching_cubes.params();
        p.grid_sizing = if v == 0 { MarchingCubesGridSizing::FromSource } else { MarchingCubesGridSizing::MaxAxis(v) };
        f.marching_cubes.set_params(p);
    });
}
#[wasm_bindgen] pub fn set_use_batched_sim(v: bool) { with_state(|s| s.use_batched_sim = v); }
#[wasm_bindgen] pub fn set_sim_dt_step(v: f32) { with_state(|s| s.sim_dt_step = v.max(1.0 / 1000.0)); }
#[wasm_bindgen] pub fn set_sim_time_scale(v: f32) { with_state(|s| s.sim_time_scale = v.max(0.01)); }
#[wasm_bindgen] pub fn set_max_render_fps(v: f32) { with_state(|s| s.max_render_fps = v.max(0.0) as f64); }
#[wasm_bindgen] pub fn set_density_scale(v: f32) { with_state(|s| s.surface_renderer.density_scale = v); }
#[wasm_bindgen] pub fn set_density_threshold(v: f32) {
    with_state(|s| { s.surface_renderer.density_threshold = v; s.mc_iso_level = v.max(0.0); });
}
#[wasm_bindgen] pub fn set_absorption(v: f32) { with_state(|s| s.surface_renderer.absorption = v); }
#[wasm_bindgen] pub fn set_step_count(v: u32) { with_state(|s| s.surface_renderer.step_count = v); }
#[wasm_bindgen] pub fn set_kernel_scale(v: f32) { with_fluid(|f| f.density_field.kernel_scale = v); }
#[wasm_bindgen] pub fn set_density_resolution(v: u32) {
    // Density resolution change requires rebuilding density field + MC bind group.
    // This is complex since the effect owns both — access through with_fluid.
    with_fluid(|f| {
        let device_ptr = &f.sim as *const _ as *const (); // placeholder — need renderer access
        // For now, just update kernel scale. Full resolution change needs renderer access.
        log::warn!("set_density_resolution: not yet supported with FluidSurfaceEffect architecture");
    });
}
#[wasm_bindgen] pub fn set_bounds(min_x: f32, min_y: f32, min_z: f32, max_x: f32, max_y: f32, max_z: f32) {
    with_fluid(|f| {
        f.sim.world_bounds_min = [min_x, min_y, min_z];
        f.sim.world_bounds_max = [max_x, max_y, max_z];
        f.sim.rebuild_grid();
    });
}
#[wasm_bindgen] pub fn set_stripe_params(
    r1: f32, g1: f32, b1: f32,
    r2: f32, g2: f32, b2: f32,
    thick_a: f32, thick_b: f32,
) {
    with_state(|s| {
        if let Some(idx) = s.dome_scene_index {
            if let Some(r) = s.scene.get_renderable_mut(idx) {
                let data: [f32; 12] = [
                    r1, g1, b1, 1.0,
                    r2, g2, b2, 1.0,
                    thick_a, thick_b,
                    0.0, 0.0,
                ];
                r.material.set_uniform_bindable(0, "Dome/StripeParams", &data);
                r.material_dirty = true;
            }
        }
    });
}
#[wasm_bindgen] pub fn set_light_direction(x: f32, y: f32, z: f32) {
    with_state(|s| {
        if let Some(Light::Directional(dl)) = s.scene.get_light_mut(s.light_scene_index) {
            dl.direction = Vec3::new(x, y, z).normalize();
        }
    });
}
#[wasm_bindgen] pub fn set_light_intensity(v: f32) {
    with_state(|s| {
        if let Some(Light::Directional(dl)) = s.scene.get_light_mut(s.light_scene_index) {
            dl.intensity = v;
        }
    });
}
#[wasm_bindgen] pub fn set_light_color(r: f32, g: f32, b: f32) {
    with_state(|s| {
        if let Some(Light::Directional(dl)) = s.scene.get_light_mut(s.light_scene_index) {
            dl.color = Vec3::new(r, g, b);
        }
    });
}
#[wasm_bindgen] pub fn set_dof_focus_distance(v: f32) {
    with_state(|s| {
        if let Some(d) = s.volume.effects.get_mut(1).and_then(|e| e.as_any_mut().downcast_mut::<DepthOfFieldEffect>()) {
            d.options.focus_distance = v;
        }
    });
}
#[wasm_bindgen] pub fn set_dof_focus_range(v: f32) {
    with_state(|s| {
        if let Some(d) = s.volume.effects.get_mut(1).and_then(|e| e.as_any_mut().downcast_mut::<DepthOfFieldEffect>()) {
            d.options.focus_range = v;
        }
    });
}
#[wasm_bindgen] pub fn set_dof_max_blur(v: f32) {
    with_state(|s| {
        if let Some(d) = s.volume.effects.get_mut(1).and_then(|e| e.as_any_mut().downcast_mut::<DepthOfFieldEffect>()) {
            d.options.max_blur = v;
        }
    });
}
#[wasm_bindgen] pub fn set_transmission_params(
    ior: f32, chromatic_aberration: f32, tint_strength: f32,
    fresnel_power: f32, roughness: f32, thickness: f32,
    r: f32, g: f32, b: f32,
) {
    with_fluid(|f| {
        f.options.ior = ior;
        f.options.chromatic_aberration = chromatic_aberration;
        f.options.tint_strength = tint_strength;
        f.options.fresnel_power = fresnel_power;
        f.options.roughness = roughness;
        f.options.thickness = thickness;
        f.options.color = [r, g, b, 1.0];
    });
}
