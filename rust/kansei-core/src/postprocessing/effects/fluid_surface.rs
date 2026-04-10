use crate::cameras::Camera;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;
use crate::simulations::fluid::{
    FluidDensityField, FluidMarchingCubes, FluidSimulation,
    SurfaceContractVersion, SurfaceExtractionSourceContract,
};

pub struct FluidSurfaceOptions {
    pub ior: f32,
    pub chromatic_aberration: f32,
    pub tint_strength: f32,
    pub fresnel_power: f32,
    pub roughness: f32,
    pub thickness: f32,
    pub color: [f32; 4],
}

impl Default for FluidSurfaceOptions {
    fn default() -> Self {
        Self {
            ior: 1.41, chromatic_aberration: 0.05, tint_strength: 0.3,
            fresnel_power: 2.3, roughness: 0.28, thickness: 2.4,
            color: [0.77, 0.96, 1.0, 1.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeParams {
    view_matrix: [f32; 16],   // 64 bytes — transform world normal to view space
    color: [f32; 4],
    ior: f32,
    chromatic_aberration: f32,
    tint_strength: f32,
    fresnel_power: f32,
    roughness: f32,
    thickness: f32,
    screen_width: f32,
    screen_height: f32,
}

const COMPOSITE_SHADER: &str = r#"
struct Params {
    view_matrix: mat4x4<f32>,
    color: vec4<f32>,
    ior: f32,
    chromatic_aberration: f32,
    tint_strength: f32,
    fresnel_power: f32,
    roughness: f32,
    thickness: f32,
    screen_width: f32,
    screen_height: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var scene_color: texture_2d<f32>;    // input (MC surface already rendered)
@group(0) @binding(2) var background_tex: texture_2d<f32>; // opaque scene before MC
@group(0) @binding(3) var normal_tex: texture_2d<f32>;     // GBuffer normals
@group(0) @binding(4) var output_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let coord = gid.xy;
    let w = u32(params.screen_width);
    let h = u32(params.screen_height);
    if (coord.x >= w || coord.y >= h) { return; }

    let scene = textureLoad(scene_color, coord, 0);
    let normal_data = textureLoad(normal_tex, coord, 0).rgb;
    let normal_len = length(normal_data);

    // No MC surface here → pass through scene color
    if (normal_len < 0.01) {
        textureStore(output_tex, coord, scene);
        return;
    }

    let N_world = normalize(normal_data);
    // Transform world normal to view space (use upper 3x3 of view matrix)
    let view3 = mat3x3<f32>(
        params.view_matrix[0].xyz,
        params.view_matrix[1].xyz,
        params.view_matrix[2].xyz,
    );
    // Flip back-facing normals so they always face the camera (fluid is double-sided)
    var N_view = normalize(view3 * N_world);
    if (N_view.z < 0.0) { N_view = -N_view; }
    let screen_uv = (vec2<f32>(f32(coord.x), f32(coord.y)) + 0.5) / vec2<f32>(f32(w), f32(h));

    // Refraction offset — use view-space normal's xy (screen-aligned)
    let refract_strength = params.thickness * (1.0 - 1.0 / params.ior);
    let offset = N_view.xy * refract_strength * 0.05;

    // Chromatic aberration
    let ca = params.chromatic_aberration;
    let dims_f = vec2<f32>(f32(w), f32(h));
    let uv_r = clamp(screen_uv + offset * (1.0 + ca), vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_g = clamp(screen_uv + offset, vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_b = clamp(screen_uv + offset * (1.0 - ca), vec2<f32>(0.0), vec2<f32>(1.0));

    let bg_r = textureLoad(background_tex, vec2u(dims_f * uv_r), 0).r;
    let bg_g = textureLoad(background_tex, vec2u(dims_f * uv_g), 0).g;
    let bg_b = textureLoad(background_tex, vec2u(dims_f * uv_b), 0).b;
    var refracted = vec3<f32>(bg_r, bg_g, bg_b);

    // Tint refracted light by fluid color (absorption)
    refracted *= mix(vec3<f32>(1.0), params.color.rgb, params.tint_strength);

    // Fresnel (view-space: N_view.z is proper NdotV since view dir is (0,0,1) in view space)
    let f0 = pow((1.0 - params.ior) / (1.0 + params.ior), 2.0);
    let ndotv = N_view.z;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - ndotv, params.fresnel_power);

    // Screen-space reflection: reflect the view ray off the view-space normal.
    // In view space, the view direction is (0,0,1) (looking toward the surface).
    let reflect_dir = reflect(vec3<f32>(0.0, 0.0, -1.0), N_view);
    let reflect_offset = reflect_dir.xy * 0.3;
    let reflect_uv = clamp(screen_uv + reflect_offset, vec2<f32>(0.0), vec2<f32>(1.0));
    let reflected = textureLoad(background_tex, vec2u(dims_f * reflect_uv), 0).rgb;

    // Rim light for edge glow
    let rim = pow(1.0 - ndotv, 3.0) * 0.15;

    // Final: mix refracted (transmitted) and reflected (environment) via Fresnel, + rim
    let result = mix(refracted, reflected, fresnel) + vec3<f32>(rim);

    textureStore(output_tex, coord, vec4<f32>(result, 1.0));
}
"#;

/// Fluid surface post-processing effect.
///
/// Encapsulates: density field update → MC extract → screen-space refraction composite.
/// The MC renderable must still be in the scene (for GBuffer depth/normals).
/// This effect reads the opaque background and composites the refractive surface.
pub struct FluidSurfaceEffect {
    pub options: FluidSurfaceOptions,
    pub sim: FluidSimulation,
    pub density_field: FluidDensityField,
    pub marching_cubes: FluidMarchingCubes,
    pub marching_cubes_bg: wgpu::BindGroup,
    // Composite pipeline
    composite_pipeline: Option<wgpu::ComputePipeline>,
    composite_bgl: Option<wgpu::BindGroupLayout>,
    params_buf: Option<wgpu::Buffer>,
    composite_bg: Option<wgpu::BindGroup>,
    cached_input_ptr: usize,
    initialized: bool,
}

impl FluidSurfaceEffect {
    pub fn new(
        sim: FluidSimulation,
        density_field: FluidDensityField,
        marching_cubes: FluidMarchingCubes,
        marching_cubes_bg: wgpu::BindGroup,
        options: FluidSurfaceOptions,
    ) -> Self {
        Self {
            options, sim, density_field, marching_cubes, marching_cubes_bg,
            composite_pipeline: None, composite_bgl: None, params_buf: None,
            composite_bg: None, cached_input_ptr: 0, initialized: false,
        }
    }

    /// Run one simulation step. Call from the animation loop before render.
    pub fn step_simulation(&mut self, dt: f32, mouse_strength: f32, mouse_ndc: [f32; 2], mouse_dir: [f32; 2], batched: bool) {
        if batched {
            self.sim.update_batched(dt, mouse_strength, mouse_ndc, mouse_dir);
        } else {
            self.sim.update(dt, mouse_strength, mouse_ndc, mouse_dir);
        }
    }
}

impl PostProcessingEffect for FluidSurfaceEffect {
    fn initialize(&mut self, device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {
        if self.initialized { return; }

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FluidSurface/BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
            ],
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FluidSurface/Composite"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FluidSurface/Layout"),
            bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });
        self.composite_pipeline = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FluidSurface/Pipeline"), layout: Some(&layout),
            module: &module, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None,
        }));
        self.params_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FluidSurface/Params"),
            size: std::mem::size_of::<CompositeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.composite_bgl = Some(bgl);
        self.initialized = true;
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        gbuffer: &GBuffer,
        input: &wgpu::TextureView,
        _depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        camera: &Camera,
        width: u32,
        height: u32,
    ) {
        if !self.initialized { return; }

        // 1. Density field + MC extract compute passes
        self.density_field.update_with_encoder(encoder,
            self.sim.world_bounds_min, self.sim.world_bounds_max,
            self.sim.particle_count(), self.sim.params.smoothing_radius);

        let source = SurfaceExtractionSourceContract {
            version: SurfaceContractVersion::V1,
            field_dims: self.density_field.tex_dims(),
            world_bounds_min: self.sim.world_bounds_min,
            world_bounds_max: self.sim.world_bounds_max,
            iso_value: self.marching_cubes.iso_level(),
        };
        self.marching_cubes.update_with_encoder_and_queue(
            encoder, queue, &self.marching_cubes_bg, source,
        );

        // 2. Upload composite params
        let mut view_matrix = [0.0f32; 16];
        view_matrix.copy_from_slice(camera.view_matrix.as_slice());
        let params = CompositeParams {
            view_matrix,
            color: self.options.color,
            ior: self.options.ior,
            chromatic_aberration: self.options.chromatic_aberration,
            tint_strength: self.options.tint_strength,
            fresnel_power: self.options.fresnel_power,
            roughness: self.options.roughness,
            thickness: self.options.thickness,
            screen_width: width as f32,
            screen_height: height as f32,
        };
        queue.write_buffer(self.params_buf.as_ref().unwrap(), 0, bytemuck::bytes_of(&params));

        // 3. Rebuild bind group if input changed
        let input_ptr = input as *const _ as usize;
        if self.composite_bg.is_none() || input_ptr != self.cached_input_ptr {
            self.cached_input_ptr = input_ptr;
            self.composite_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("FluidSurface/BG"),
                layout: self.composite_bgl.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(input) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&gbuffer.background_view) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&gbuffer.normal_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(output) },
                ],
            }));
        }

        // 4. Composite pass
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FluidSurface/Composite"), timestamp_writes: None,
        });
        pass.set_pipeline(self.composite_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, self.composite_bg.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
    }

    fn resize(&mut self, _width: u32, _height: u32, _gbuffer: &GBuffer) {
        self.composite_bg = None;
        self.cached_input_ptr = 0;
    }

    fn destroy(&mut self) {
        self.initialized = false;
        self.composite_bg = None;
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
