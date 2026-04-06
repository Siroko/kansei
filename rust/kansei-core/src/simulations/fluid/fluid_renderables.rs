use crate::math::Mat4;
use crate::objects::Object3D;
use crate::renderers::Renderer;
use crate::buffers::Texture;

use super::{
    FluidDensityField, FluidMarchingCubes, FluidParticleRenderer, FluidSurfaceRenderer, FullscreenBlit,
    SimulationRenderable, SimulationRenderableInputContract, SurfaceContractVersion,
};

pub struct ParticlesRenderable {
    pub object: Object3D,
    pub render_order: i32,
    pub visible: bool,
    pub particle_size: f32,
    renderer: FluidParticleRenderer,
}

impl ParticlesRenderable {
    pub fn new(
        renderer: &Renderer,
        positions_buffer: &wgpu::Buffer,
        count: u32,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
        depth_format: Option<wgpu::TextureFormat>,
        particle_size: f32,
    ) -> Self {
        Self {
            object: Object3D::new(),
            render_order: 0,
            visible: true,
            particle_size,
            renderer: FluidParticleRenderer::new(
                renderer,
                positions_buffer,
                count,
                color_format,
                sample_count,
                depth_format,
            ),
        }
    }

    pub fn render(&self, renderer: &Renderer, pass: &mut wgpu::RenderPass<'_>, view: &Mat4, proj: &Mat4) {
        self.renderer.upload(renderer, view, proj, self.particle_size);
        pass.set_pipeline(&self.renderer.pipeline);
        pass.set_bind_group(0, &self.renderer.bind_group, &[]);
        pass.draw(0..self.renderer.count * 6, 0..1);
    }
}

pub struct MarchingCubesRenderable {
    pub object: Object3D,
    pub render_order: i32,
    pub visible: bool,
    pub color: [f32; 4],
    renderer: SimulationRenderable,
    mesh_vertex: wgpu::Buffer,
    mesh_index: wgpu::Buffer,
    mesh_indirect: wgpu::Buffer,
    mesh_stride: u64,
}

impl MarchingCubesRenderable {
    pub fn new(
        renderer: &Renderer,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
        marching_cubes: &FluidMarchingCubes,
    ) -> Self {
        let mesh = marching_cubes.mesh_contract();
        Self {
            object: Object3D::new(),
            render_order: 0,
            visible: true,
            color: [0.77, 0.96, 1.0, 1.0],
            renderer: SimulationRenderable::new(renderer, color_format, sample_count),
            mesh_vertex: mesh.vertex_buffer.clone(),
            mesh_index: mesh.index_buffer.clone(),
            mesh_indirect: mesh.indirect_args_buffer.clone(),
            mesh_stride: mesh.vertex_stride,
        }
    }

    pub fn sync_source(&mut self, marching_cubes: &FluidMarchingCubes) {
        let mesh = marching_cubes.mesh_contract();
        self.mesh_vertex = mesh.vertex_buffer.clone();
        self.mesh_index = mesh.index_buffer.clone();
        self.mesh_indirect = mesh.indirect_args_buffer.clone();
        self.mesh_stride = mesh.vertex_stride;
    }

    pub fn render(&self, renderer: &Renderer, pass: &mut wgpu::RenderPass<'_>, view: &Mat4, proj: &Mat4) {
        self.renderer.render_surface_mesh(
            renderer,
            pass,
            SimulationRenderableInputContract {
                version: SurfaceContractVersion::V1,
                mesh: super::SurfaceMeshGpuContract {
                    version: SurfaceContractVersion::V1,
                    vertex_buffer: &self.mesh_vertex,
                    index_buffer: &self.mesh_index,
                    indirect_args_buffer: &self.mesh_indirect,
                    vertex_stride: self.mesh_stride,
                },
            },
            view,
            proj,
            self.color,
        );
    }
}

pub struct RaymarchingRenderable {
    pub object: Object3D,
    pub render_order: i32,
    pub visible: bool,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    surface_renderer: FluidSurfaceRenderer,
    blit: FullscreenBlit,
    density_view: wgpu::TextureView,
    color_texture: Texture,
    color_view: wgpu::TextureView,
    depth_texture: Texture,
    depth_view: wgpu::TextureView,
    output_texture: Texture,
    output_view: wgpu::TextureView,
    surface_bg: wgpu::BindGroup,
    blit_bg: wgpu::BindGroup,
}

impl RaymarchingRenderable {
    pub fn new(
        renderer: &Renderer,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        density_field: &FluidDensityField,
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
    ) -> Self {
        let surface_renderer = FluidSurfaceRenderer::new(renderer.raw_device(), renderer.raw_queue());
        let blit = FullscreenBlit::new(renderer, surface_format);
        let density_view = density_field.density_view.clone();
        let (color_texture, color_view, depth_texture, depth_view, output_texture, output_view) =
            Self::create_offscreen_textures(renderer, width, height);
        let surface_bg = surface_renderer.create_bind_group(
            &color_view,
            &depth_view,
            &output_view,
            &density_view,
        );
        let blit_bg = blit.create_bind_group(renderer, &output_view);
        Self {
            object: Object3D::new(),
            render_order: 0,
            visible: true,
            bounds_min,
            bounds_max,
            surface_renderer,
            blit,
            density_view,
            color_texture,
            color_view,
            depth_texture,
            depth_view,
            output_texture,
            output_view,
            surface_bg,
            blit_bg,
        }
    }

    fn create_offscreen_textures(
        renderer: &Renderer,
        width: u32,
        height: u32,
    ) -> (
        Texture,
        wgpu::TextureView,
        Texture,
        wgpu::TextureView,
        Texture,
        wgpu::TextureView,
    ) {
        let mut color_texture = Texture::new_2d(
            "FluidRaymarch/Color",
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        color_texture.initialize(renderer.raw_device());
        let color_view = color_texture
            .view()
            .expect("FluidRaymarch/Color view should exist after initialization")
            .clone();

        let mut depth_texture = Texture::new_2d(
            "FluidRaymarch/Depth",
            width,
            height,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        depth_texture.initialize(renderer.raw_device());
        let depth_view = depth_texture
            .view()
            .expect("FluidRaymarch/Depth view should exist after initialization")
            .clone();

        let mut output_texture = Texture::new_2d(
            "FluidRaymarch/Output",
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        output_texture.initialize(renderer.raw_device());
        let output_view = output_texture
            .view()
            .expect("FluidRaymarch/Output view should exist after initialization")
            .clone();

        (color_texture, color_view, depth_texture, depth_view, output_texture, output_view)
    }

    pub fn resize(&mut self, renderer: &Renderer, width: u32, height: u32) {
        let (color_texture, color_view, depth_texture, depth_view, output_texture, output_view) =
            Self::create_offscreen_textures(renderer, width, height);
        self.color_texture = color_texture;
        self.color_view = color_view;
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
        self.output_texture = output_texture;
        self.output_view = output_view;
        self.surface_bg = self.surface_renderer.create_bind_group(
            &self.color_view,
            &self.depth_view,
            &self.output_view,
            &self.density_view,
        );
        self.blit_bg = self.blit.create_bind_group(renderer, &self.output_view);
    }

    pub fn set_density_source(&mut self, density_field: &FluidDensityField) {
        self.density_view = density_field.density_view.clone();
        self.surface_bg = self.surface_renderer.create_bind_group(
            &self.color_view,
            &self.depth_view,
            &self.output_view,
            &self.density_view,
        );
    }

    pub fn input_color_view(&self) -> &wgpu::TextureView {
        &self.color_view
    }

    pub fn input_depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        canvas_view: &wgpu::TextureView,
        inv_view_proj: &Mat4,
        camera_pos: [f32; 3],
        width: u32,
        height: u32,
    ) {
        self.surface_renderer.render(
            encoder,
            &self.surface_bg,
            inv_view_proj,
            camera_pos,
            self.bounds_min,
            self.bounds_max,
            width,
            height,
        );
        let mut pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: canvas_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            })
            .forget_lifetime();
        pass.set_pipeline(&self.blit.pipeline);
        pass.set_bind_group(0, &self.blit_bg, &[]);
        pass.draw(0..3, 0..1);
    }

    pub fn surface_renderer_mut(&mut self) -> &mut FluidSurfaceRenderer {
        &mut self.surface_renderer
    }
}

pub enum FluidRenderable {
    Particles(ParticlesRenderable),
    Raymarching(RaymarchingRenderable),
    MarchingCubes(MarchingCubesRenderable),
}

impl FluidRenderable {
    pub fn visible(&self) -> bool {
        match self {
            FluidRenderable::Particles(r) => r.visible,
            FluidRenderable::Raymarching(r) => r.visible,
            FluidRenderable::MarchingCubes(r) => r.visible,
        }
    }

    pub fn render_order(&self) -> i32 {
        match self {
            FluidRenderable::Particles(r) => r.render_order,
            FluidRenderable::Raymarching(r) => r.render_order,
            FluidRenderable::MarchingCubes(r) => r.render_order,
        }
    }

    pub fn position(&self) -> crate::math::Vec3 {
        match self {
            FluidRenderable::Particles(r) => r.object.position,
            FluidRenderable::Raymarching(r) => r.object.position,
            FluidRenderable::MarchingCubes(r) => r.object.position,
        }
    }
}
