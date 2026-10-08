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
    /// Key light: the direction the light travels (engine `DirectionalLight`
    /// convention), its intensity, and color. Drives the specular highlight
    /// and tints the rim.
    pub light_direction: [f32; 3],
    pub light_intensity: f32,
    pub light_color: [f32; 3],
    /// Strength of the rim glow at grazing angles, tinted by the key light.
    pub rim: f32,
    /// The sky's colour, reflected where the reflected view ray points up (open water under the
    /// sky), mixed in by `sky_reflection` (0: screen-space reflection only).
    pub sky_color: [f32; 3],
    pub sky_reflection: f32,
}

impl Default for FluidSurfaceOptions {
    fn default() -> Self {
        Self {
            ior: 1.41, chromatic_aberration: 0.05, tint_strength: 0.3,
            fresnel_power: 2.3, roughness: 0.28, thickness: 2.4,
            color: [0.77, 0.96, 1.0, 1.0],
            light_direction: [0.3, -1.0, 0.5],
            light_intensity: 2.0,
            light_color: [1.0, 1.0, 1.0],
            rim: 0.15,
            sky_color: [1.0, 1.0, 1.0],
            sky_reflection: 0.0,
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
    light_dir: [f32; 4],   // xyz = direction light travels (world), w = intensity
    light_color: [f32; 4], // rgb, w = rim strength
    sky: [f32; 4],         // rgb, w = sky reflection
    mask: u32,             // 0: any GBuffer normal is fluid; 1: only where emissive alpha >= 0.5
    _pad: [u32; 3],
}

const COMPOSITE_SHADER: &str = include_str!("../../shaders/fluid_surface_composite.wgsl");

/// How [`FluidSurfaceEffect`] finds the fluid's surface in the GBuffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FluidMask {
    /// Any pixel with a normal: for scenes where only the fluid writes the GBuffer's normals.
    #[default]
    AnyNormal,
    /// Pixels with a normal whose emissive alpha is at least 0.5, which the fluid's surface
    /// material writes (and other materials leave at 0): for scenes whose other materials write
    /// normals too (for screen-space GI and AO).
    EmissiveAlpha,
}

/// The fluid surface's GBuffer material (`FluidSurfaceEffect::surface_renderable`).
const SURFACE_MESH_WGSL: &str = include_str!("../../shaders/fluid_surface_mesh.wgsl");

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
    /// Splat radius for the surface density field. `None` = the sim's smoothing
    /// radius. Set it explicitly when the sim radius is smaller than a field voxel,
    /// otherwise the field is sparse and the surface shatters into shards.
    pub splat_radius: Option<f32>,
    /// Whether the surface is extracted from the particles each frame (the default). Off, the
    /// last surface extracted keeps drawing: for a simulation that is not stepping (see
    /// `simulations::fluid::FluidSleep`).
    pub extract: bool,
    /// Whether the effect runs at all (the default): off, it costs nothing and composites nothing,
    /// for a fluid out of view.
    pub active: bool,
    /// How the composite finds the fluid's pixels (any normal, by default).
    pub mask: FluidMask,
}

impl FluidSurfaceEffect {
    /// Show a fluid's [`FluidActivity`](crate::simulations::fluid::FluidActivity): its surface
    /// extracted while it runs, the last one drawn while it sleeps, nothing while it is culled.
    pub fn set_activity(&mut self, activity: crate::simulations::fluid::FluidActivity) {
        use crate::simulations::fluid::FluidActivity;
        self.extract = activity == FluidActivity::Running;
        self.active = activity != FluidActivity::Culled;
    }

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
            composite_bg: None, cached_input_ptr: 0, initialized: false, splat_radius: None,
            extract: true, active: true, mask: FluidMask::AnyNormal,
        }
    }

    /// Run one simulation step. Call from the animation loop before render.
    /// The renderable that puts the fluid's surface (the marching-cubes mesh this effect
    /// extracts) into the GBuffer, which the composite then refracts: add it to the scene. It
    /// writes `color`, the world normal (as the composite reads it, unencoded) and the
    /// `FluidMask::EmissiveAlpha` mark, double-sided, casting no shadow.
    pub fn surface_renderable(&self, color: [f32; 4]) -> crate::objects::Renderable {
        use crate::materials::{Binding, CullMode, Material, MaterialOptions};
        let mc = &self.marching_cubes;
        let geometry = crate::geometries::Geometry::from_gpu_buffers(
            "FluidSurface/Mesh",
            mc.vertex_buffer().clone(),
            mc.index_buffer().clone(),
            Some(mc.indirect_args_buffer().clone()),
        );
        let options = MaterialOptions { cull_mode: CullMode::None, mrt_output_count: Some(4), ..Default::default() };
        let mut material = Material::new("FluidSurface/Mesh", SURFACE_MESH_WGSL, vec![Binding::uniform(0, wgpu::ShaderStages::FRAGMENT)], options);
        material.set_uniform_bindable(0, "FluidSurface/Color", &color);
        let mut renderable = crate::objects::Renderable::new(geometry, material);
        renderable.cast_shadow = false;
        renderable
    }

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
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
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
        if self.extract {
            self.density_field.update_with_encoder(encoder,
                self.sim.world_bounds_min, self.sim.world_bounds_max,
                self.sim.particle_count(), self.splat_radius.unwrap_or(self.sim.params.smoothing_radius));

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
        }

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
            light_dir: [
                self.options.light_direction[0], self.options.light_direction[1],
                self.options.light_direction[2], self.options.light_intensity,
            ],
            light_color: [
                self.options.light_color[0], self.options.light_color[1],
                self.options.light_color[2], self.options.rim,
            ],
            sky: [
                self.options.sky_color[0], self.options.sky_color[1],
                self.options.sky_color[2], self.options.sky_reflection,
            ],
            mask: (self.mask == FluidMask::EmissiveAlpha) as u32,
            _pad: [0; 3],
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
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&gbuffer.emissive_view) },
                ],
            }));
        }

        // 4. Composite pass
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FluidSurface/Composite"), timestamp_writes: crate::profiling::gpu_pass("FluidSurface/Composite").as_ref().map(crate::profiling::PassStamp::compute),
        });
        pass.set_pipeline(self.composite_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, self.composite_bg.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
    }

    fn is_active(&self) -> bool {
        self.active
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shader_validates_and_the_params_layout_matches() {
        let surface = naga::front::wgsl::parse_str(SURFACE_MESH_WGSL).unwrap_or_else(|e| panic!("{}", e.emit_to_string(SURFACE_MESH_WGSL)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&surface).unwrap();
        let module = naga::front::wgsl::parse_str(COMPOSITE_SHADER).unwrap_or_else(|e| panic!("{}", e.emit_to_string(COMPOSITE_SHADER)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
        let span = module.types.iter().find_map(|(_, t)| match (&t.name, &t.inner) {
            (Some(n), naga::TypeInner::Struct { span, .. }) if n == "Params" => Some(*span as usize),
            _ => None,
        });
        assert_eq!(span, Some(std::mem::size_of::<CompositeParams>()));
    }
}
