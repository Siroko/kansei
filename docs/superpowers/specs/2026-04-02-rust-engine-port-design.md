# Kansei Engine — Full Rust Port Design

## Goal

Port the Kansei TypeScript WebGPU engine to Rust targeting both native (via winit) and WASM (via web-sys). Primary motivation: performance comparison between TS and Rust implementations using the fluid simulation as the first benchmark.

## Approach

Vertical slice first. Get a minimal but real render pipeline working end-to-end (Renderer draws a box through Scene -> Camera -> Material -> Geometry -> draw call), then widen horizontally across all modules.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Math types | Own `repr(C)` types, glam internally | GPU-compatible Pod types that double as buffer resources, matching TS pattern |
| Shader composition | Hybrid: `include_str!` for built-ins, runtime `parse_includes()` for user materials | Static perf for engine shaders, dynamic flexibility for user code |
| Bind group layout | Same 4-group structure as TS, extensible via enum | Apples-to-apples comparison, room for path tracer / post-process groups later |
| Platform | `kansei-core` is platform-agnostic, `kansei-native` and `kansei-wasm` are entry points | Core takes a `wgpu::Surface`, never creates one |

## Current State

- **Working**: Math types (Vec2/3/4, Mat4), fluid SPH simulation, density field, surface renderer
- **Stub/broken**: Renderer (has TODOs where draw calls should be), lights, shadows, post-processing, loaders
- **Missing**: Quaternion, BufferGeometry, InstancedGeometry, shader includes, pipeline cache, render bundles, all post-processing effects, GLTF loader, texture loader

## Workspace Structure (unchanged)

```
rust/
  Cargo.toml              # workspace
  kansei-core/             # platform-agnostic engine library
    src/
      lib.rs
      math/                # Vec2, Vec3, Vec4, Mat4, Quaternion, Float
      buffers/             # BufferBase, ComputeBuffer, StructuredBuffer, Texture, Sampler
      geometries/          # Geometry, InstancedGeometry, BoxGeometry, PlaneGeometry, SphereGeometry
      materials/           # Material, MaterialOptions, ComputePass, ShaderChunks
      objects/             # Object3D, Renderable, Scene
      cameras/             # Camera
      renderers/           # Renderer, RendererConfig, GBuffer
      lights/              # DirectionalLight, PointLight, AreaLight
      shadows/             # ShadowMap, CubeMapShadowMap
      postprocessing/      # PostProcessingEffect trait, PostProcessingVolume, SSAO, Bloom, DoF, GodRays, ColorGrading, VolumetricFog
      loaders/             # TextureLoader, GLTFLoader
      controls/            # CameraControls
      simulations/fluid/   # (already ported) FluidSimulation, FluidDensityField, FluidSurfaceRenderer
  kansei-native/           # winit entry point + examples
  kansei-wasm/             # web-sys entry point + JS interop
```

---

## Module Designs

### 1. Math & Buffer Primitives

#### GpuBindable Trait

Every math type that can live in a GPU buffer implements this trait:

```rust
pub trait GpuBindable {
    fn as_bytes(&self) -> &[u8];
    fn gpu_buffer(&self) -> Option<&wgpu::Buffer>;
    fn ensure_buffer(&mut self, device: &wgpu::Device, usage: wgpu::BufferUsages);
    fn upload(&self, queue: &wgpu::Queue);
    fn version(&self) -> u64;
}
```

Mirrors the TS `IBindable` pattern. A `Mat4` owns both a CPU-side value and optionally a GPU buffer. Version counter tracks dirtiness — only upload when changed.

#### Math Types

- `Vec2`, `Vec3`, `Vec4`, `Mat4`: existing `#[repr(C)]` Pod types, implement `GpuBindable`
- `Quaternion`: new, `#[repr(C)]` Pod, for rotation without gimbal lock
- `Float`: existing wrapper for f32 uniforms
- All heavy math (inverse, perspective, decompose) delegates to glam internally

#### Buffer Types

- `BufferBase`: rename current `GpuBuffer`. CPU staging + GPU buffer, lazy init, `write_buffer` on dirty flag
- `ComputeBuffer`: storage buffer with attribute descriptors for instancing (`shader_location`, `offset`, `stride`, `format`)
- `StructuredBuffer`: field-offset calculator for structured uniform data
- `Texture`, `Sampler`: unchanged

### 2. Geometry System

#### Geometry

```rust
pub struct Geometry {
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    vertex_layouts: Vec<wgpu::VertexBufferLayout<'static>>,
    vertex_count: u32,
    initialized: bool,
}
```

Standard vertex layout: position (vec4f, location 0) + normal (vec3f, location 1) + uv (vec2f, location 2) = 36 bytes stride. Matches TS exactly.

Methods: `initialize(device)`, `set_vertices(data)`, `set_indices(data)`.

#### InstancedGeometry

```rust
pub struct InstancedGeometry {
    pub base: Geometry,
    pub extra_buffers: Vec<ComputeBuffer>,
    pub instance_count: u32,
}
```

Extra buffers appended as additional vertex buffer layouts with `step_mode: Instance` at higher shader locations. This is how the TS engine handles per-instance transforms, colors, etc.

#### Built-in Geometries

`BoxGeometry`, `PlaneGeometry`, `SphereGeometry` — construct a `Geometry` with pre-computed vertices/indices. Same as today but on the new base.

### 3. Material System

#### Material

```rust
pub struct Material {
    pub shader_code: String,
    pub options: MaterialOptions,
    bindings: Vec<Binding>,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    pipeline_cache: HashMap<String, wgpu::RenderPipeline>,
}

pub struct MaterialOptions {
    pub transparent: bool,
    pub depth_write: bool,
    pub cull_mode: Option<wgpu::Face>,
    pub topology: wgpu::PrimitiveTopology,
    pub blend: Option<wgpu::BlendState>,
}
```

Pipeline cache key: `"{color_format}:{sample_count}:{depth_format}"`. For GBuffer MRT, key includes all color format names.

`get_pipeline_for_config()` creates pipeline on first access, returns cached on subsequent calls.

Transparent materials: disable depth write, set cull mode to none, enable alpha blend on first color target.

#### Shader Composition

- `ShaderChunks`: `HashMap<String, String>` of reusable WGSL snippets
- `parse_includes(code: &str, chunks: &ShaderChunks) -> String`: recursive `#include <name>` replacement
- Built-in shaders loaded via `include_str!("shaders/basic.wgsl")` at compile time
- User-defined materials use runtime `parse_includes()`

#### ComputePass

Unchanged — shader code + bind groups + dispatch dimensions.

### 4. Objects & Scene Graph

#### Object3D

```rust
pub struct Object3D {
    pub position: Vec3,
    pub rotation: Vec3,
    pub quaternion: Quaternion,
    pub scale: Vec3,
    pub model_matrix: Mat4,
    pub world_matrix: Mat4,
    pub normal_matrix: Mat4,
    parent: Option<usize>,
    children: Vec<usize>,
    dirty: bool,
    use_quaternion: bool,
}
```

`update_model_matrix()`: walks up to parent if dirty, builds T x R x S, then `update_world_matrix()`.
`update_normal_matrix()`: inverse-transpose of world matrix.
`traverse(callback)`: depth-first walk of hierarchy.

Children inherit parent transforms — currently broken, this fixes it.

#### Renderable

```rust
pub struct Renderable {
    pub transform: Object3D,
    pub geometry: Geometry,       // or InstancedGeometry via enum
    pub material: Material,
    pub instance_count: u32,
    pub cast_shadow: bool,
    pub receive_shadow: bool,
    pub render_order: i32,
    pub visible: bool,
    material_dirty: bool,
}
```

#### Scene

```rust
pub struct Scene {
    pub renderables: Vec<Renderable>,
    pub lights: Vec<Light>,
    opaque_objects: Vec<usize>,
    transparent_objects: Vec<usize>,
}
```

`sort(camera_position)`: separates opaque from transparent, sorts transparent back-to-front by squared distance. Returns `ordered_objects()` = opaque + sorted transparent.

### 5. Renderer

Three-phase architecture matching the TS engine.

#### Phase 1 — Update (matrix upload)

```rust
world_matrices_staging: Vec<f32>,
normal_matrices_staging: Vec<f32>,
world_matrices_buf: wgpu::Buffer,
normal_matrices_buf: wgpu::Buffer,
matrix_alignment: u32,  // device.limits.min_uniform_buffer_offset_alignment
```

Per frame: iterate ordered objects, copy each world/normal matrix into staging at `index * alignment`, then two `write_buffer` calls.

#### Phase 2 — Bundle (record draw commands)

```rust
for each renderable in ordered_objects:
    pipeline = material.get_pipeline_for_config(format, samples, depth_format)
    set_pipeline (if changed)
    set_index_buffer (if changed)
    set_vertex_buffers (if changed, including instanced extra buffers)
    set_bind_group(0, material_bind_group)
    set_bind_group(1, shared_mesh_bind_group, &[offset, offset])
    draw_indexed(vertex_count, instance_count)
```

Bundle cached, rebuilt on: object count change, material swap, `invalidate_bundle()`.

#### Phase 3 — Execute

Single `render_pass.execute_bundles(&[bundle])`.

#### Bind Group Slots (extensible)

```rust
pub enum BindGroupSlot {
    Material = 0,
    Mesh = 1,        // dynamic offsets into bulk matrix buffers
    Camera = 2,
    Shadow = 3,
    // future extensions
}
```

Groups 1-3 created/owned by Renderer. Group 0 owned by Material.

#### GBuffer Path

Same three phases targeting MRT: color (rgba16float), emissive (rgba16float), normal (rgba16float), albedo (rgba8unorm) + depth. Pipeline cache keys include all target formats.

#### Depth Copy Pass

Fullscreen triangle resolving MSAA depth to non-MSAA texture for compute shader sampling.

#### Public API

```rust
impl Renderer {
    pub fn render(&mut self, scene: &mut Scene, camera: &mut Camera);
    pub fn render_to_gbuffer(&mut self, scene: &mut Scene, camera: &mut Camera, gbuffer: &GBuffer);
    pub fn resize(&mut self, width: u32, height: u32);
    pub fn invalidate_bundle(&mut self);
    pub fn device(&self) -> &wgpu::Device;
    pub fn queue(&self) -> &wgpu::Queue;
}
```

### 6. Lights & Shadows

#### Lights

```rust
pub enum Light {
    Directional(DirectionalLight),
    Point(PointLight),
    Area(AreaLight),
}
```

- `DirectionalLight`: direction, color, intensity, cast_shadow, optional `ShadowMap`
- `PointLight`: position, color, intensity, radius, cast_shadow, optional `CubeMapShadowMap`
- `AreaLight`: position, color, intensity, width, height

Light data packed into uniform buffer, uploaded per frame.

#### Shadows

- `ShadowMap`: directional light depth texture. Renders scene from light's view/projection using depth-only material override.
- `CubeMapShadowMap`: point light, renders 6 cube faces.
- Shadow pass reuses the Renderer's three-phase loop with the light as camera and depth-only output.
- Shadow textures + sampler bound to group 3.

### 7. Post-Processing

#### PostProcessingEffect Trait

```rust
pub trait PostProcessingEffect {
    fn initialize(&mut self, device: &wgpu::Device, gbuffer: &GBuffer);
    fn render(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue,
              input: &wgpu::TextureView, output: &wgpu::TextureView,
              gbuffer: &GBuffer, camera: &Camera);
    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32);
}
```

#### PostProcessingVolume

```rust
pub struct PostProcessingVolume {
    effects: Vec<Box<dyn PostProcessingEffect>>,
    ping: wgpu::Texture,
    pong: wgpu::Texture,
}
```

Reads from GBuffer, runs each effect in sequence alternating ping/pong. Final result blitted to screen.

#### Effects (each a standalone struct implementing the trait)

- **SSAO**: screen-space AO from depth + normals
- **Bloom**: threshold, downsample/blur chain, composite
- **DepthOfField**: circle-of-confusion from depth, blur
- **GodRays**: radial blur from light screen position
- **ColorGrading**: LUT-based or parametric tone mapping
- **VolumetricFog**: density raymarching through depth

Shaders are `.wgsl` files via `include_str!`. TS WGSL shaders port nearly 1:1.

### 8. Loaders

#### TextureLoader

```rust
pub struct TextureLoader;
impl TextureLoader {
    pub fn load(device: &wgpu::Device, queue: &wgpu::Queue, path: &str) -> Texture;
    pub fn load_bytes(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8], format: wgpu::TextureFormat) -> Texture;
}
```

Uses `image` crate for decoding (PNG, JPG, HDR). WASM reads via fetch + web-sys. Native reads from filesystem.

#### GLTFLoader

```rust
pub struct GLTFLoader;
impl GLTFLoader {
    pub fn load(device: &wgpu::Device, queue: &wgpu::Queue, path: &str) -> GLTFResult;
}
pub struct GLTFResult {
    pub renderables: Vec<Renderable>,
    pub textures: Vec<Texture>,
}
```

Uses `gltf` crate. Extracts meshes into `Geometry`, creates `Material` per glTF material, applies node transforms.

### 9. Controls

#### CameraControls

```rust
pub struct CameraControls {
    pub target: Vec3,
    pub distance: f32,
    pub azimuth: f32,
    pub elevation: f32,
    pub min_distance: f32,
    pub max_distance: f32,
}
impl CameraControls {
    pub fn update(&mut self, camera: &mut Camera);
    pub fn on_mouse_move(&mut self, dx: f32, dy: f32);
    pub fn on_scroll(&mut self, delta: f32);
}
```

Takes deltas, not raw platform events. Platform layer (winit / web-sys) translates events into deltas.

---

## Vertical Slice Order

1. **Math + GpuBindable**: Quaternion, version tracking, ensure_buffer/upload
2. **BufferBase + ComputeBuffer**: Rename GpuBuffer, add attribute descriptors
3. **Geometry + BoxGeometry**: New Geometry base, standard vertex layout
4. **Material + pipeline cache**: Shader module creation, pipeline caching, bind group 0
5. **Object3D + Renderable + Scene**: Transform hierarchy, sort, ordered objects
6. **Renderer core**: Shared matrix buffers, bind groups 1-3, three-phase render loop
7. **Validate**: Render a spinning box — proves the full pipeline works
8. **InstancedGeometry**: Extra vertex buffers with instance step mode
9. **Lights + shadows**: Uniform packing, shadow passes
10. **Post-processing**: Effect trait, volume, port each effect
11. **Loaders**: Texture + GLTF
12. **Fluid example rewrite**: Use engine abstractions instead of raw wgpu
13. **WASM build**: Wire kansei-wasm to use the real engine

## Dependencies

- `wgpu = "24"` (already in workspace)
- `glam = "0.29"` (already in workspace)
- `bytemuck = "1"` (already in workspace)
- `image = "0.25"` (new — texture loading)
- `gltf = "1"` (new — glTF parsing)
