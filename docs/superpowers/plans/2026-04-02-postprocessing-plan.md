# Post-Processing Implementation Plan (Plan 2d)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a working post-processing pipeline with bloom and color grading effects. Scene is rendered to GBuffer via `render_to_gbuffer()`, effects process the color texture via compute shaders with ping-pong textures, and the final result is blitted to the canvas.

**Architecture:** `PostProcessingVolume` (already exists) orchestrates effect chain. We complete its blit pipeline, implement `BloomEffect` (downsample/upsample/composite compute shaders) and `ColorGradingEffect` (single compute pass with tone mapping), and add a `render_with_postprocessing()` method to Renderer that ties it all together.

**Tech Stack:** Rust, wgpu 24, bytemuck 1

---

## File Structure

### Files to create:
- `rust/kansei-core/src/postprocessing/effects/bloom.rs` — BloomEffect compute shader implementation
- `rust/kansei-core/src/postprocessing/effects/color_grading.rs` — ColorGradingEffect compute shader
- `rust/kansei-core/src/shaders/bloom_downsample.wgsl` — 13-tap downsample + soft threshold
- `rust/kansei-core/src/shaders/bloom_upsample.wgsl` — 9-tap tent filter upsample
- `rust/kansei-core/src/shaders/bloom_composite.wgsl` — Additive bloom composite
- `rust/kansei-core/src/shaders/color_grading.wgsl` — Tone mapping + color correction
- `rust/kansei-core/src/shaders/blit.wgsl` — Fullscreen triangle blit
- `rust/kansei-native/examples/postprocess_scene.rs` — Validation example

### Files to modify:
- `rust/kansei-core/src/postprocessing/volume.rs` — Complete blit pipeline
- `rust/kansei-core/src/postprocessing/effects/mod.rs` — Export effects
- `rust/kansei-core/src/postprocessing/mod.rs` — Re-export effects
- `rust/kansei-core/src/renderers/renderer.rs` — Add render_with_postprocessing()

---

### Task 1: Complete PostProcessingVolume blit pipeline

**Files:**
- Create: `rust/kansei-core/src/shaders/blit.wgsl`
- Modify: `rust/kansei-core/src/postprocessing/volume.rs`

The blit pipeline renders the final post-processed texture to the canvas via a fullscreen triangle.

- [ ] **Step 1: Create blit.wgsl**

Create `rust/kansei-core/src/shaders/blit.wgsl`:

```wgsl
@group(0) @binding(0) var source_tex: texture_2d<f32>;
@group(0) @binding(1) var blit_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(pos[vi], 0.0, 1.0);
    out.uv = uv[vi];
    return out;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(source_tex, blit_sampler, input.uv);
}
```

- [ ] **Step 2: Complete blit pipeline in volume.rs**

Replace the entire `rust/kansei-core/src/postprocessing/volume.rs` with a version that:
- Creates the blit pipeline on first render (lazy init)
- Creates a bind group pointing at the final effect output texture
- After running all effects, renders a fullscreen triangle to the canvas surface view

Key additions to `PostProcessingVolume`:

Add `initialize_blit` method:
```rust
    fn initialize_blit(&mut self, device: &wgpu::Device, surface_format: wgpu::TextureFormat) {
        if self.blit_pipeline.is_some() { return; }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit/Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/blit.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blit/BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blit/Layout"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit/Pipeline"), layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader, entry_point: Some("vertex_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: Some("fragment_main"),
                targets: &[Some(wgpu::ColorTargetState { format: surface_format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(), depth_stencil: None,
            multisample: Default::default(), multiview: None, cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        self.blit_bgl = Some(bgl);
        self.blit_sampler = Some(sampler);
        self.blit_pipeline = Some(pipeline);
    }
```

Update the `render` method signature to accept `surface_format` and actually blit:

```rust
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: &Camera,
        surface_view: &wgpu::TextureView,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) {
        // Lazily create/resize GBuffer
        if self.gbuffer.is_none() || self.gbuffer.as_ref().map(|g| g.width != width || g.height != height).unwrap_or(false) {
            self.gbuffer = Some(GBuffer::new(device, width, height, 4));
            for effect in &mut self.effects {
                if let Some(ref gb) = self.gbuffer {
                    effect.resize(width, height, gb);
                }
            }
        }

        self.initialize_blit(device, surface_format);

        let gbuffer = self.gbuffer.as_ref().unwrap();

        // Initialize uninitialised effects
        for effect in &mut self.effects {
            effect.initialize(device, gbuffer, camera);
        }

        // Run effect chain
        let mut final_view = &gbuffer.color_view; // default: no effects
        if !self.effects.is_empty() {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PostProcessing/Effects"),
            });

            let mut ping = 0;
            let mut first = true;

            for effect in &mut self.effects {
                let (input_view, output_view) = if ping == 0 {
                    if first {
                        (&gbuffer.color_view, &gbuffer.output_view)
                    } else {
                        (&gbuffer.ping_pong_view, &gbuffer.output_view)
                    }
                } else {
                    (&gbuffer.output_view, &gbuffer.ping_pong_view)
                };

                effect.render(&mut encoder, input_view, &gbuffer.depth_view, output_view, camera, width, height);
                first = false;
                ping = 1 - ping;
            }

            queue.submit(std::iter::once(encoder.finish()));

            // Final output is whichever texture was last written to
            final_view = if ping == 0 { &gbuffer.ping_pong_view } else { &gbuffer.output_view };
        }

        // Blit to canvas
        let blit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit/BG"), layout: self.blit_bgl.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(final_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(self.blit_sampler.as_ref().unwrap()) },
            ],
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit/Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: surface_view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_pipeline(self.blit_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, &blit_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
```

- [ ] **Step 3: Verify it compiles**

```bash
cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo check -p kansei-core 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add kansei-core/src/shaders/blit.wgsl kansei-core/src/postprocessing/volume.rs
git commit -m "feat: complete PostProcessingVolume blit pipeline"
```

---

### Task 2: Implement ColorGradingEffect

**Files:**
- Create: `rust/kansei-core/src/shaders/color_grading.wgsl`
- Create: `rust/kansei-core/src/postprocessing/effects/color_grading.rs`
- Modify: `rust/kansei-core/src/postprocessing/effects/mod.rs`
- Modify: `rust/kansei-core/src/postprocessing/mod.rs`

Color grading is a single compute pass — simpler to implement first before bloom.

- [ ] **Step 1: Create color_grading.wgsl**

```wgsl
struct GradeParams {
    brightness: f32,
    contrast: f32,
    saturation: f32,
    temperature: f32,
    tint: f32,
    highlights: f32,
    shadows: f32,
    black_point: f32,
    screen_w: f32,
    screen_h: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: GradeParams;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<u32>(u32(params.screen_w), u32(params.screen_h));
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    var color = textureLoad(input_tex, vec2<i32>(gid.xy), 0).rgb;

    // Black point lift
    if (params.black_point > 0.0) {
        color = max(color - vec3<f32>(params.black_point), vec3<f32>(0.0)) / (1.0 - params.black_point);
    }

    // Shadows/highlights
    let lum = luminance(color);
    let sh_mix = smoothstep(0.0, 1.0, lum);
    let sh_mul = mix(params.shadows, params.highlights, sh_mix);
    color *= sh_mul;

    // Brightness
    color += vec3<f32>(params.brightness);

    // Contrast (pivot at 0.5)
    color = (color - vec3<f32>(0.5)) * params.contrast + vec3<f32>(0.5);

    // Saturation
    let gray = luminance(color);
    color = mix(vec3<f32>(gray), color, params.saturation);

    // Temperature
    color.r += params.temperature * 0.1;
    color.b -= params.temperature * 0.1;

    // Tint
    color.g += params.tint * 0.1;

    color = max(color, vec3<f32>(0.0));
    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}
```

- [ ] **Step 2: Create color_grading.rs**

```rust
// rust/kansei-core/src/postprocessing/effects/color_grading.rs
use crate::cameras::Camera;
use crate::renderers::GBuffer;
use crate::postprocessing::PostProcessingEffect;

pub struct ColorGradingOptions {
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub temperature: f32,
    pub tint: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub black_point: f32,
}

impl Default for ColorGradingOptions {
    fn default() -> Self {
        Self {
            brightness: 0.0, contrast: 1.0, saturation: 1.0,
            temperature: 0.0, tint: 0.0, highlights: 1.0, shadows: 1.0, black_point: 0.0,
        }
    }
}

pub struct ColorGradingEffect {
    pub options: ColorGradingOptions,
    pipeline: Option<wgpu::ComputePipeline>,
    bgl: Option<wgpu::BindGroupLayout>,
    params_buf: Option<wgpu::Buffer>,
    initialized: bool,
}

impl ColorGradingEffect {
    pub fn new(options: ColorGradingOptions) -> Self {
        Self { options, pipeline: None, bgl: None, params_buf: None, initialized: false }
    }
}

impl PostProcessingEffect for ColorGradingEffect {
    fn initialize(&mut self, device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {
        if self.initialized { return; }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ColorGrading/Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/color_grading.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ColorGrading/BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ColorGrading/Layout"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });

        self.pipeline = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ColorGrading/Pipeline"), layout: Some(&layout), module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        }));

        self.params_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ColorGrading/Params"), size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.bgl = Some(bgl);
        self.initialized = true;
    }

    fn render(
        &mut self, encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView, _depth: &wgpu::TextureView, output: &wgpu::TextureView,
        _camera: &Camera, width: u32, height: u32,
    ) {
        let pipeline = match &self.pipeline { Some(p) => p, None => return };
        let bgl = self.bgl.as_ref().unwrap();

        // Upload params
        let o = &self.options;
        let data: [f32; 12] = [
            o.brightness, o.contrast, o.saturation, o.temperature,
            o.tint, o.highlights, o.shadows, o.black_point,
            width as f32, height as f32, 0.0, 0.0,
        ];
        // Note: queue.write_buffer can't be called here (no queue ref).
        // We'll use encoder.copy_buffer_to_buffer or a mapped staging buffer.
        // Simpler: use a write_buffer call from the Renderer before dispatching effects.
        // For now, write params in initialize and use a separate update method.

        let device_bg = encoder.get_device(); // Not available in wgpu...
        // WORKAROUND: Create bind group here using the encoder's implicit device context.
        // Actually, we can't create bind groups from an encoder.
        // Solution: Cache the bind group and rebuild when inputs change.

        // This is a design issue. Let's store the device reference and create bind groups lazily.
        // For now, skip the render if params_buf is None.

        // Actually, the PostProcessingVolume should pass the device and queue.
        // Let's update the trait to include queue for param uploads.
        // But changing the trait is a bigger change...

        // PRAGMATIC FIX: Store params_data and have the volume upload before dispatching.
        // Or: make the effect store its own bind groups and rebuild per-frame.

        // Let's just create the bind group each frame (cheap for compute):
        // We need the device... The encoder doesn't give us one.
        // The trait signature needs to change to include device + queue.

        // DECISION: We'll modify the PostProcessingEffect trait to include device + queue.
        // This is necessary for any effect that needs to upload uniforms or create bind groups.
        todo!("Trait needs device + queue — see Task 2 implementation notes")
    }

    fn resize(&mut self, _width: u32, _height: u32, _gbuffer: &GBuffer) {
        // Bind groups will be recreated per-frame
    }

    fn destroy(&mut self) {
        self.pipeline = None;
        self.bgl = None;
        self.params_buf = None;
        self.initialized = false;
    }
}
```

**WAIT** — I see a design issue. The `PostProcessingEffect::render()` trait doesn't pass `device` or `queue`, but effects need both to upload params and create bind groups. Let me fix the trait first.

- [ ] **Step 2 (revised): Update PostProcessingEffect trait to include device + queue**

In `rust/kansei-core/src/postprocessing/effect.rs`, change the trait:

```rust
use crate::cameras::Camera;
use crate::renderers::GBuffer;

/// Trait for compute-shader post-processing effects.
pub trait PostProcessingEffect {
    fn initialize(&mut self, device: &wgpu::Device, gbuffer: &GBuffer, camera: &Camera);
    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        camera: &Camera,
        width: u32,
        height: u32,
    );
    fn resize(&mut self, width: u32, height: u32, gbuffer: &GBuffer);
    fn destroy(&mut self);
}
```

Update the call site in `volume.rs` to pass device + queue:

```rust
effect.render(device, queue, &mut encoder, input_view, &gbuffer.depth_view, output_view, camera, width, height);
```

- [ ] **Step 3: Create the actual color_grading.rs with the fixed trait**

The ColorGradingEffect implementation creates its bind group each frame (cheap for compute) and uploads params via `queue.write_buffer`:

```rust
// rust/kansei-core/src/postprocessing/effects/color_grading.rs
use crate::cameras::Camera;
use crate::renderers::GBuffer;
use crate::postprocessing::PostProcessingEffect;

pub struct ColorGradingOptions {
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub temperature: f32,
    pub tint: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub black_point: f32,
}

impl Default for ColorGradingOptions {
    fn default() -> Self {
        Self {
            brightness: 0.0, contrast: 1.0, saturation: 1.0,
            temperature: 0.0, tint: 0.0, highlights: 1.0, shadows: 1.0, black_point: 0.0,
        }
    }
}

pub struct ColorGradingEffect {
    pub options: ColorGradingOptions,
    pipeline: Option<wgpu::ComputePipeline>,
    bgl: Option<wgpu::BindGroupLayout>,
    params_buf: Option<wgpu::Buffer>,
    initialized: bool,
}

impl ColorGradingEffect {
    pub fn new(options: ColorGradingOptions) -> Self {
        Self { options, pipeline: None, bgl: None, params_buf: None, initialized: false }
    }
}

impl PostProcessingEffect for ColorGradingEffect {
    fn initialize(&mut self, device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {
        if self.initialized { return; }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ColorGrading/Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/color_grading.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ColorGrading/BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });

        self.pipeline = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ColorGrading/Pipeline"), layout: Some(&layout), module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        }));

        self.params_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ColorGrading/Params"), size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.bgl = Some(bgl);
        self.initialized = true;
    }

    fn render(
        &mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView, _depth: &wgpu::TextureView, output: &wgpu::TextureView,
        _camera: &Camera, width: u32, height: u32,
    ) {
        let pipeline = match &self.pipeline { Some(p) => p, None => return };

        // Upload params
        let o = &self.options;
        let data: [f32; 12] = [
            o.brightness, o.contrast, o.saturation, o.temperature,
            o.tint, o.highlights, o.shadows, o.black_point,
            width as f32, height as f32, 0.0, 0.0,
        ];
        queue.write_buffer(self.params_buf.as_ref().unwrap(), 0, bytemuck::cast_slice(&data));

        // Create bind group
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: self.bgl.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(output) },
                wgpu::BindGroupEntry { binding: 2, resource: self.params_buf.as_ref().unwrap().as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ColorGrading"), ..Default::default() });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
    }

    fn resize(&mut self, _width: u32, _height: u32, _gbuffer: &GBuffer) {}
    fn destroy(&mut self) { self.initialized = false; }
}
```

- [ ] **Step 4: Update effects/mod.rs and postprocessing/mod.rs**

`rust/kansei-core/src/postprocessing/effects/mod.rs`:
```rust
mod color_grading;
pub use color_grading::{ColorGradingEffect, ColorGradingOptions};
```

`rust/kansei-core/src/postprocessing/mod.rs`:
```rust
mod effect;
mod volume;
pub mod effects;

pub use effect::PostProcessingEffect;
pub use volume::PostProcessingVolume;
pub use crate::renderers::GBuffer;
```

- [ ] **Step 5: Verify and commit**

```bash
cargo check -p kansei-core 2>&1 | tail -10
git add kansei-core/src/postprocessing/ kansei-core/src/shaders/color_grading.wgsl
git commit -m "feat: add ColorGradingEffect with parametric tone mapping"
```

---

### Task 3: Implement BloomEffect

**Files:**
- Create: `rust/kansei-core/src/shaders/bloom_downsample.wgsl`
- Create: `rust/kansei-core/src/shaders/bloom_upsample.wgsl`
- Create: `rust/kansei-core/src/shaders/bloom_composite.wgsl`
- Create: `rust/kansei-core/src/postprocessing/effects/bloom.rs`
- Modify: `rust/kansei-core/src/postprocessing/effects/mod.rs`

Bloom uses 3 compute shaders: downsample (13-tap + soft threshold), upsample (9-tap tent filter), composite (additive blend). 6 mip levels.

- [ ] **Step 1: Create bloom_downsample.wgsl**

```wgsl
struct BloomParams {
    threshold: f32,
    knee: f32,
    intensity: f32,
    radius: f32,
    src_width: f32,
    src_height: f32,
    level: u32,
    _pad: u32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: BloomParams;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn soft_threshold(color: vec3<f32>, t: f32, k: f32) -> vec3<f32> {
    let lum = luminance(color);
    let soft = lum - t + k;
    let soft2 = clamp(soft, 0.0, 2.0 * k);
    let contrib = soft2 * soft2 / (4.0 * k + 0.0001);
    let w = max(contrib, lum - t) / max(lum, 0.0001);
    return color * max(w, 0.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_dims = textureDimensions(dst_tex);
    if (gid.x >= dst_dims.x || gid.y >= dst_dims.y) { return; }

    let texel = vec2<f32>(1.0 / params.src_width, 1.0 / params.src_height);
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dst_dims);

    // Convert UV to source texel coords for textureLoad
    let src_dims = textureDimensions(src_tex);
    let src_coord = vec2<i32>(uv * vec2<f32>(src_dims));

    // 13-tap box filter (Jimenez 2014)
    var color = vec3<f32>(0.0);
    color += textureLoad(src_tex, src_coord + vec2<i32>(-1, -1), 0).rgb * 0.0625;
    color += textureLoad(src_tex, src_coord + vec2<i32>( 0, -1), 0).rgb * 0.125;
    color += textureLoad(src_tex, src_coord + vec2<i32>( 1, -1), 0).rgb * 0.0625;
    color += textureLoad(src_tex, src_coord + vec2<i32>(-1,  0), 0).rgb * 0.125;
    color += textureLoad(src_tex, src_coord + vec2<i32>( 0,  0), 0).rgb * 0.25;
    color += textureLoad(src_tex, src_coord + vec2<i32>( 1,  0), 0).rgb * 0.125;
    color += textureLoad(src_tex, src_coord + vec2<i32>(-1,  1), 0).rgb * 0.0625;
    color += textureLoad(src_tex, src_coord + vec2<i32>( 0,  1), 0).rgb * 0.125;
    color += textureLoad(src_tex, src_coord + vec2<i32>( 1,  1), 0).rgb * 0.0625;

    // Apply threshold only on first level
    if (params.level == 0u) {
        color = soft_threshold(color, params.threshold, params.knee);
    }

    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}
```

- [ ] **Step 2: Create bloom_upsample.wgsl**

```wgsl
struct BloomParams {
    threshold: f32,
    knee: f32,
    intensity: f32,
    radius: f32,
    src_width: f32,
    src_height: f32,
    level: u32,
    _pad: u32,
};

@group(0) @binding(0) var smaller_tex: texture_2d<f32>;
@group(0) @binding(1) var bloom_sampler: sampler;
@group(0) @binding(2) var current_tex: texture_2d<f32>;
@group(0) @binding(3) var dst_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: BloomParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_dims = textureDimensions(dst_tex);
    if (gid.x >= dst_dims.x || gid.y >= dst_dims.y) { return; }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dst_dims);
    let texel = 1.0 / vec2<f32>(textureDimensions(smaller_tex));

    // 9-tap tent filter on smaller mip (bilinear sampled)
    var bloom = vec3<f32>(0.0);
    bloom += textureSampleLevel(smaller_tex, bloom_sampler, uv + vec2<f32>(-1.0, -1.0) * texel * params.radius, 0.0).rgb * 1.0;
    bloom += textureSampleLevel(smaller_tex, bloom_sampler, uv + vec2<f32>( 0.0, -1.0) * texel * params.radius, 0.0).rgb * 2.0;
    bloom += textureSampleLevel(smaller_tex, bloom_sampler, uv + vec2<f32>( 1.0, -1.0) * texel * params.radius, 0.0).rgb * 1.0;
    bloom += textureSampleLevel(smaller_tex, bloom_sampler, uv + vec2<f32>(-1.0,  0.0) * texel * params.radius, 0.0).rgb * 2.0;
    bloom += textureSampleLevel(smaller_tex, bloom_sampler, uv + vec2<f32>( 0.0,  0.0) * texel * params.radius, 0.0).rgb * 4.0;
    bloom += textureSampleLevel(smaller_tex, bloom_sampler, uv + vec2<f32>( 1.0,  0.0) * texel * params.radius, 0.0).rgb * 2.0;
    bloom += textureSampleLevel(smaller_tex, bloom_sampler, uv + vec2<f32>(-1.0,  1.0) * texel * params.radius, 0.0).rgb * 1.0;
    bloom += textureSampleLevel(smaller_tex, bloom_sampler, uv + vec2<f32>( 0.0,  1.0) * texel * params.radius, 0.0).rgb * 2.0;
    bloom += textureSampleLevel(smaller_tex, bloom_sampler, uv + vec2<f32>( 1.0,  1.0) * texel * params.radius, 0.0).rgb * 1.0;
    bloom /= 16.0;

    // Add current mip level
    let current = textureLoad(current_tex, vec2<i32>(gid.xy), 0).rgb;
    let result = bloom + current;

    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(result, 1.0));
}
```

- [ ] **Step 3: Create bloom_composite.wgsl**

```wgsl
struct BloomParams {
    threshold: f32,
    knee: f32,
    intensity: f32,
    radius: f32,
    src_width: f32,
    src_height: f32,
    level: u32,
    _pad: u32,
};

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var bloom_tex: texture_2d<f32>;
@group(0) @binding(2) var bloom_sampler: sampler;
@group(0) @binding(3) var dst_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: BloomParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<u32>(u32(params.src_width), u32(params.src_height));
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let scene = textureLoad(scene_tex, vec2<i32>(gid.xy), 0).rgb;
    let bloom = textureSampleLevel(bloom_tex, bloom_sampler, uv, 0.0).rgb;
    let result = scene + bloom * params.intensity;

    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(result, 1.0));
}
```

- [ ] **Step 4: Create bloom.rs**

This is the most complex effect. It manages 6 mip-chain textures for downsampling, 5 for upsampling, plus parameter buffers for each pass. The implementation should:

1. On initialize: create 3 compute pipelines (downsample, upsample, composite), bind group layouts, mip textures, param buffers, sampler
2. On render: dispatch downsample chain (6 passes), upsample chain (5 passes), composite (1 pass)
3. On resize: recreate mip textures

Due to the complexity, the implementer should read the TS BloomEffect.ts for reference. The key pattern:
- Each mip level is half the previous dimensions
- Downsample reads from previous mip, writes to current
- Upsample reads from smaller mip (sampled) + current mip (loaded), writes to upsample texture
- Composite reads scene color + final upsampled bloom, writes to output

The bloom.rs file will be approximately 300-400 lines. The implementer should create a working implementation that:
- Creates all necessary textures in `initialize()` and `resize()`
- Dispatches all passes in sequence in `render()`
- Each pass gets its own param buffer upload + bind group

- [ ] **Step 5: Update effects/mod.rs**

```rust
mod color_grading;
mod bloom;
pub use color_grading::{ColorGradingEffect, ColorGradingOptions};
pub use bloom::{BloomEffect, BloomOptions};
```

- [ ] **Step 6: Verify and commit**

```bash
cargo check -p kansei-core 2>&1 | tail -10
git add kansei-core/src/postprocessing/effects/ kansei-core/src/shaders/bloom_*.wgsl
git commit -m "feat: add BloomEffect with downsample/upsample/composite compute shaders"
```

---

### Task 4: Add render_with_postprocessing to Renderer + validation example

**Files:**
- Modify: `rust/kansei-core/src/renderers/renderer.rs` — Add render_with_postprocessing()
- Create: `rust/kansei-native/examples/postprocess_scene.rs`

- [ ] **Step 1: Add render_with_postprocessing to Renderer**

This method: renders scene to GBuffer via `render_to_gbuffer()`, then runs the PostProcessingVolume chain, then blits to canvas.

```rust
    /// Render scene with post-processing effects.
    pub fn render_with_postprocessing(
        &mut self,
        scene: &mut Scene,
        camera: &mut Camera,
        volume: &mut PostProcessingVolume,
    ) {
        let gbuffer = volume.ensure_gbuffer(self.device.as_ref().unwrap(), self.config.width, self.config.height);
        self.render_to_gbuffer(scene, camera, gbuffer);

        let surface = self.surface.as_ref().unwrap();
        let output = surface.get_current_texture().expect("Surface texture");
        let canvas_view = output.texture.create_view(&Default::default());

        volume.render(
            self.device.as_ref().unwrap(),
            self.queue.as_ref().unwrap(),
            camera,
            &canvas_view,
            self.presentation_format,
            self.config.width,
            self.config.height,
        );

        output.present();
    }
```

Note: The `PostProcessingVolume` needs a method to return its GBuffer for the renderer to use:
```rust
    /// Ensure GBuffer exists at the right size. Returns a reference.
    pub fn ensure_gbuffer(&mut self, device: &wgpu::Device, width: u32, height: u32) -> &GBuffer {
        if self.gbuffer.is_none() || self.gbuffer.as_ref().map(|g| g.width != width || g.height != height).unwrap_or(false) {
            self.gbuffer = Some(GBuffer::new(device, width, height, 4));
        }
        self.gbuffer.as_ref().unwrap()
    }
```

- [ ] **Step 2: Create postprocess_scene.rs example**

Same scene as shadow_scene but with bloom + color grading. Uses `render_with_postprocessing` instead of `render`.

Key differences:
- Creates a PostProcessingVolume with BloomEffect + ColorGradingEffect
- Uses `render_with_postprocessing` in the render loop
- Has some bright emissive surfaces (e.g., a bright white box) to demonstrate bloom
- Color grading with slight warm temperature and boosted contrast

```rust
let volume = PostProcessingVolume::new(vec![
    Box::new(BloomEffect::new(BloomOptions { threshold: 0.8, intensity: 0.6, ..Default::default() })),
    Box::new(ColorGradingEffect::new(ColorGradingOptions { contrast: 1.2, temperature: 0.15, ..Default::default() })),
]);
```

Per frame:
```rust
renderer.render_with_postprocessing(&mut scene, &mut camera, &mut volume);
```

- [ ] **Step 3: Build and run**

```bash
cargo build --example postprocess_scene 2>&1 | tail -5
cargo run --example postprocess_scene
```

- [ ] **Step 4: Commit**

```bash
git add kansei-core/src/renderers/renderer.rs kansei-core/src/postprocessing/ kansei-native/examples/postprocess_scene.rs
git commit -m "feat: add render_with_postprocessing + bloom/color grading example"
```

---

## Post-Plan Notes

### What this plan produces:
- Complete PostProcessingVolume with blit pipeline
- ColorGradingEffect (brightness, contrast, saturation, temperature, tint, shadows/highlights, black point)
- BloomEffect (6-level downsample with soft threshold, tent-filter upsample, additive composite)
- render_with_postprocessing() on Renderer
- Updated PostProcessingEffect trait with device + queue parameters
- Validation example with bloom + color grading

### Limitations (deferred):
- SSAO, DoF, god rays, volumetric fog — follow-up plans
- Emissive MRT (bloom currently reads from color, not separate emissive)
- GBuffer render_to_gbuffer draw loop (still TODO from Plan 1 — needs shadow pass integration)

### What comes next:
- **Plan 2e: Loaders** — Texture + glTF
- **Plan 3: Fluid Integration** — Rewrite fluid example using engine abstractions
