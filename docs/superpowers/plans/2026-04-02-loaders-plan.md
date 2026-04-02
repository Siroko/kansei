# Loaders Implementation Plan (Plan 2e)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add TextureLoader (PNG/JPG → wgpu::Texture) and GLTFLoader (glTF 2.0 → Scene with Geometry + Material + transforms) so the engine can load external assets.

**Architecture:** TextureLoader uses the `image` crate to decode images into raw RGBA8 pixels, then uploads to a wgpu texture. GLTFLoader uses the `gltf` crate to parse .gltf/.glb files, extracts mesh primitives into kansei `Geometry` (interleaved pos+normal+uv matching `Vertex` layout), builds a scene graph with transforms, and extracts PBR material properties.

**Tech Stack:** Rust, wgpu 24, `image` 0.25, `gltf` 1

---

## File Structure

### Files to create:
- `rust/kansei-core/src/loaders/texture_loader.rs` — TextureLoader
- `rust/kansei-core/src/loaders/gltf_loader.rs` — GLTFLoader
- `rust/kansei-native/examples/gltf_viewer.rs` — Validation example

### Files to modify:
- `rust/kansei-core/src/loaders/mod.rs` — Export loaders
- `rust/kansei-core/Cargo.toml` — Add `image` and `gltf` dependencies
- `rust/Cargo.toml` — Add workspace dependencies

---

### Task 1: Add dependencies and create TextureLoader

**Files:**
- Modify: `rust/Cargo.toml` — Add `image` and `gltf` to workspace deps
- Modify: `rust/kansei-core/Cargo.toml` — Add `image` and `gltf` deps
- Create: `rust/kansei-core/src/loaders/texture_loader.rs`
- Modify: `rust/kansei-core/src/loaders/mod.rs`

- [ ] **Step 1: Add dependencies to workspace Cargo.toml**

In `rust/Cargo.toml`, add to `[workspace.dependencies]`:
```toml
image = "0.25"
gltf = { version = "1", features = ["utils"] }
```

In `rust/kansei-core/Cargo.toml`, add to `[dependencies]`:
```toml
image = { workspace = true }
gltf = { workspace = true }
```

- [ ] **Step 2: Create texture_loader.rs**

```rust
// rust/kansei-core/src/loaders/texture_loader.rs
use crate::buffers::Texture;

/// Loads images from disk into GPU textures.
pub struct TextureLoader;

impl TextureLoader {
    /// Load a texture from a file path. Returns a wgpu Texture + TextureView.
    pub fn load(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &str,
    ) -> Result<LoadedTexture, String> {
        let img = image::open(path)
            .map_err(|e| format!("Failed to load image '{}': {}", path, e))?;
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();

        Self::from_rgba8(device, queue, &rgba, width, height, Some(path))
    }

    /// Load a texture from raw bytes (PNG, JPG, etc).
    pub fn load_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        label: Option<&str>,
    ) -> Result<LoadedTexture, String> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| format!("Failed to decode image: {}", e))?;
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();

        Self::from_rgba8(device, queue, &rgba, width, height, label)
    }

    /// Create a GPU texture from raw RGBA8 pixel data.
    pub fn from_rgba8(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
        width: u32,
        height: u32,
        label: Option<&str>,
    ) -> Result<LoadedTexture, String> {
        let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: label.map(|l| l),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            size,
        );

        let view = texture.create_view(&Default::default());

        Ok(LoadedTexture { texture, view, width, height })
    }
}

/// A loaded GPU texture with its view and dimensions.
pub struct LoadedTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}
```

- [ ] **Step 3: Update loaders/mod.rs**

```rust
mod texture_loader;
pub use texture_loader::{TextureLoader, LoadedTexture};
```

- [ ] **Step 4: Verify and commit**

```bash
cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo check -p kansei-core 2>&1 | tail -5
git add Cargo.toml kansei-core/Cargo.toml kansei-core/src/loaders/
git commit -m "feat: add TextureLoader for PNG/JPG → GPU texture loading"
```

---

### Task 2: Create GLTFLoader

**Files:**
- Create: `rust/kansei-core/src/loaders/gltf_loader.rs`
- Modify: `rust/kansei-core/src/loaders/mod.rs`

The GLTFLoader parses .gltf/.glb files using the `gltf` crate and builds kansei engine objects.

- [ ] **Step 1: Create gltf_loader.rs**

```rust
// rust/kansei-core/src/loaders/gltf_loader.rs
use crate::geometries::{Geometry, Vertex};
use crate::objects::{Object3D, Renderable, Scene};
use crate::materials::{Material, MaterialOptions, Binding};
use crate::math::Vec3;

/// Material properties extracted from glTF PBR metallic-roughness.
pub struct GLTFMaterialInfo {
    pub name: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub double_sided: bool,
}

/// Result of loading a glTF file.
pub struct GLTFResult {
    pub renderables: Vec<GLTFRenderable>,
    pub materials: Vec<GLTFMaterialInfo>,
}

/// A loaded renderable with its transform.
pub struct GLTFRenderable {
    pub geometry: Geometry,
    pub material_index: usize,
    pub position: Vec3,
    pub rotation: Vec3,
    pub scale: Vec3,
}

/// Loads glTF 2.0 files into engine objects.
pub struct GLTFLoader;

impl GLTFLoader {
    /// Load a glTF or glb file from disk.
    pub fn load(path: &str) -> Result<GLTFResult, String> {
        let (document, buffers, _images) = gltf::import(path)
            .map_err(|e| format!("Failed to load glTF '{}': {}", path, e))?;

        let materials = Self::parse_materials(&document);
        let renderables = Self::parse_scene(&document, &buffers);

        Ok(GLTFResult { renderables, materials })
    }

    /// Load from in-memory glb bytes.
    pub fn load_glb(bytes: &[u8]) -> Result<GLTFResult, String> {
        let (document, buffers, _images) = gltf::import_slice(bytes)
            .map_err(|e| format!("Failed to parse glb: {}", e))?;

        let materials = Self::parse_materials(&document);
        let renderables = Self::parse_scene(&document, &buffers);

        Ok(GLTFResult { renderables, materials })
    }

    fn parse_materials(doc: &gltf::Document) -> Vec<GLTFMaterialInfo> {
        doc.materials().map(|mat| {
            let pbr = mat.pbr_metallic_roughness();
            GLTFMaterialInfo {
                name: mat.name().unwrap_or("Unnamed").to_string(),
                base_color: pbr.base_color_factor(),
                metallic: pbr.metallic_factor(),
                roughness: pbr.roughness_factor(),
                double_sided: mat.double_sided(),
            }
        }).collect()
    }

    fn parse_scene(doc: &gltf::Document, buffers: &[gltf::buffer::Data]) -> Vec<GLTFRenderable> {
        let mut renderables = Vec::new();

        let scene = doc.default_scene().or_else(|| doc.scenes().next());
        if let Some(scene) = scene {
            for node in scene.nodes() {
                Self::process_node(&node, buffers, &glam::Mat4::IDENTITY, &mut renderables);
            }
        }

        renderables
    }

    fn process_node(
        node: &gltf::Node,
        buffers: &[gltf::buffer::Data],
        parent_transform: &glam::Mat4,
        renderables: &mut Vec<GLTFRenderable>,
    ) {
        let local = glam::Mat4::from_cols_array_2d(&node.transform().matrix());
        let world = *parent_transform * local;

        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                if let Some(geo) = Self::parse_primitive(&primitive, buffers) {
                    // Decompose world transform
                    let (scale, rotation, translation) = world.to_scale_rotation_translation();
                    let euler = rotation.to_euler(glam::EulerRot::YXZ);

                    renderables.push(GLTFRenderable {
                        geometry: geo,
                        material_index: primitive.material().index().unwrap_or(0),
                        position: Vec3::new(translation.x, translation.y, translation.z),
                        rotation: Vec3::new(euler.1, euler.0, euler.2), // YXZ → (y, x, z)
                        scale: Vec3::new(scale.x, scale.y, scale.z),
                    });
                }
            }
        }

        for child in node.children() {
            Self::process_node(&child, buffers, &world, renderables);
        }
    }

    fn parse_primitive(
        primitive: &gltf::Primitive,
        buffers: &[gltf::buffer::Data],
    ) -> Option<Geometry> {
        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

        // Read positions (required)
        let positions: Vec<[f32; 3]> = reader.read_positions()?.collect();
        let vertex_count = positions.len();

        // Read normals (optional, default up)
        let normals: Vec<[f32; 3]> = reader.read_normals()
            .map(|n| n.collect())
            .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; vertex_count]);

        // Read UVs (optional, default zero)
        let uvs: Vec<[f32; 2]> = reader.read_tex_coords(0)
            .map(|tc| tc.into_f32().collect())
            .unwrap_or_else(|| vec![[0.0, 0.0]; vertex_count]);

        // Build interleaved vertices matching Vertex layout
        let vertices: Vec<Vertex> = (0..vertex_count).map(|i| {
            Vertex {
                position: [positions[i][0], positions[i][1], positions[i][2], 1.0],
                normal: normals[i],
                uv: uvs[i],
            }
        }).collect();

        // Read indices
        let indices: Vec<u32> = reader.read_indices()
            .map(|idx| idx.into_u32().collect())
            .unwrap_or_else(|| (0..vertex_count as u32).collect());

        let name = format!("GLTF/Primitive");
        Some(Geometry::new(&name, vertices, indices))
    }
}
```

- [ ] **Step 2: Update loaders/mod.rs**

```rust
mod texture_loader;
mod gltf_loader;

pub use texture_loader::{TextureLoader, LoadedTexture};
pub use gltf_loader::{GLTFLoader, GLTFResult, GLTFRenderable, GLTFMaterialInfo};
```

- [ ] **Step 3: Verify and commit**

```bash
cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo check -p kansei-core 2>&1 | tail -10
git add kansei-core/src/loaders/
git commit -m "feat: add GLTFLoader for loading glTF 2.0 scenes"
```

---

### Task 3: Create glTF viewer validation example

**Files:**
- Create: `rust/kansei-native/examples/gltf_viewer.rs`

A simple viewer that loads a glTF file passed as a command-line argument and renders it with the basic_lit material, directional light, and orbit camera.

- [ ] **Step 1: Create a test .glb file or use a built-in fallback**

Since we need a glTF file to test, the example should handle two cases:
- If a path is passed as arg: load that file
- If no arg: create a simple procedural scene (box + sphere) as fallback

- [ ] **Step 2: Create gltf_viewer.rs**

The example should:
1. Parse command line args for an optional glTF file path
2. If path provided: load with GLTFLoader, create Materials from GLTFMaterialInfo (using basic_lit.wgsl), create Renderables with transforms from GLTFRenderable
3. If no path: create a fallback scene with a box and sphere (same as lit_scene)
4. Add directional + point lights
5. Enable shadows
6. Orbit camera that auto-frames the loaded model (compute bounding box, set distance)
7. render() each frame

Key API for loading:
```rust
use kansei_core::loaders::{GLTFLoader, GLTFResult};

let result = GLTFLoader::load(path)?;
for gr in &result.renderables {
    let mat_info = &result.materials[gr.material_index.min(result.materials.len() - 1)];
    // Create material with base_color from mat_info
    let mat_data: [f32; 8] = [
        mat_info.base_color[0], mat_info.base_color[1], mat_info.base_color[2], mat_info.base_color[3],
        0.5, 0.5, 0.5, 0.3, // specular
    ];
    // ... create Material, Renderable with gr.position/rotation/scale, add to scene
}
```

- [ ] **Step 3: Build and test**

```bash
cargo build --example gltf_viewer 2>&1 | tail -5
# Test with no args (fallback scene):
cargo run --example gltf_viewer
# Test with a glTF file (if available):
# cargo run --example gltf_viewer -- path/to/model.glb
```

- [ ] **Step 4: Commit**

```bash
git add kansei-native/examples/gltf_viewer.rs
git commit -m "feat: add glTF viewer example with fallback scene"
```

---

## Post-Plan Notes

### What this plan produces:
- TextureLoader: load PNG/JPG from disk or bytes → wgpu Texture
- GLTFLoader: parse glTF/glb → GLTFResult with geometries, materials, transforms
- glTF viewer example that loads and renders glTF files
- `image` and `gltf` crate dependencies added to workspace

### Limitations:
- TextureLoader doesn't generate mipmaps (single mip level)
- GLTFLoader doesn't load textures from glTF (baseColorTexture etc.) — only base color factor
- GLTFLoader uses a single default material shader, not PBR
- No async loading (all synchronous/blocking)

### What comes next:
- **Plan 3: Fluid Integration** — Rewrite fluid example using engine abstractions + WASM build
