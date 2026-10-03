use bytemuck::{Pod, Zeroable};

use crate::lights::Light;
use crate::math::{Mat4, Vec3};
use crate::shadows::{CascadedShadowMap, CubeMapShadowMap, ShadowMap, SpotShadowAtlas};

/// A point light without a cube shadow (`PointLightData::shadowLayer`).
pub(crate) const NO_SHADOW: u32 = u32::MAX;

/// The WGSL `DirLightData`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct DirLightGpu {
    pub direction: [f32; 3],
    pub shadowed: u32,
    pub color: [f32; 3],
    pub _pad: f32,
}

/// The WGSL `PointLightData`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct PointLightGpu {
    pub position: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    pub shadow_layer: u32,
}

/// The stand-ins bound where a source is missing, the light buffers and the spot atlas' sampler.
struct Gpu {
    dir_lights: wgpu::Buffer,
    point_lights: wgpu::Buffer,
    dummy_depth: wgpu::TextureView,
    dummy_atlas: wgpu::TextureView,
    dummy_vp: wgpu::Buffer,
    dummy_spot_lights: wgpu::Buffer,
    dummy_spot_atlas: wgpu::TextureView,
    spot_sampler: wgpu::Sampler,
}

/// The renderer's lights and shadow maps as compute passes see them: the compute-visible copy of
/// group 3 (`compute_shadows.wgsl`, group 0 bindings 1 and 3-9), shared by the volumetric fog
/// and voxel GI's light injection.
///
/// - Directional lights: the first casts its shadow through the renderer's `ShadowMap`, or the
///   widest cascade of its `CascadedShadowMap`.
/// - Point lights (area lights as points at their position): the first shadow-casting one
///   through the cube shadow atlas.
/// - Spot lights: the renderer's buffer and shadow atlas, read as they are.
///
/// A pass lays its group out with `layout_entries` next to its own bindings, calls `prepare`
/// before recording (it rebuilds its bind group when that returns true) and binds `entries`.
pub(crate) struct ComputeShadows {
    pub(crate) dir: Vec<DirLightGpu>,
    pub(crate) point: Vec<PointLightGpu>,
    shadow_map: Option<(wgpu::TextureView, wgpu::Buffer)>,
    point_shadows: Option<wgpu::TextureView>,
    spot_lights: Option<wgpu::Buffer>,
    spot_shadows: Option<wgpu::TextureView>,
    lights_dirty: bool,
    bindings_dirty: bool,
    gpu: Option<Gpu>,
}

impl ComputeShadows {
    pub(crate) fn new() -> Self {
        Self {
            dir: Vec::new(),
            point: Vec::new(),
            shadow_map: None,
            point_shadows: None,
            spot_lights: None,
            spot_shadows: None,
            lights_dirty: true,
            bindings_dirty: true,
            gpu: None,
        }
    }

    /// Collect the directional, point and area lights (spot lights come from the renderer's
    /// buffer). The first directional light casts through the shadow map if it `cast_shadow`s
    /// (that is the light the renderer's maps follow), the first shadow-casting point light
    /// through the cube atlas. `volumetric_only` keeps the fog's rule: directional lights that
    /// aren't `volumetric` are left out and such point lights scatter nothing. The lights are
    /// uploaded again only if they changed.
    pub(crate) fn update_lights<'a>(&mut self, lights: impl IntoIterator<Item = &'a Light>, volumetric_only: bool) {
        let (old_dir, old_point) = (std::mem::take(&mut self.dir), std::mem::take(&mut self.point));
        let mut seen_directional = false;
        let mut point_shadow_assigned = false;
        for light in lights {
            match light {
                Light::Directional(l) => {
                    let first = !seen_directional;
                    seen_directional = true;
                    if l.volumetric || !volumetric_only {
                        let c = l.effective_color();
                        self.dir.push(DirLightGpu {
                            direction: [l.direction.x, l.direction.y, l.direction.z],
                            shadowed: (first && l.cast_shadow) as u32,
                            color: [c.x, c.y, c.z],
                            _pad: 0.0,
                        });
                    }
                }
                Light::Point(l) => {
                    let shadow_layer = if l.cast_shadow && !point_shadow_assigned {
                        point_shadow_assigned = true;
                        0 // CubeMapShadowMap renders the first shadow-casting light into layers 0..6
                    } else {
                        NO_SHADOW
                    };
                    let c = if l.volumetric || !volumetric_only { l.effective_color() } else { Vec3::ZERO };
                    self.point.push(PointLightGpu {
                        position: [l.position.x, l.position.y, l.position.z],
                        radius: l.radius,
                        color: [c.x, c.y, c.z],
                        shadow_layer,
                    });
                }
                Light::Area(l) => {
                    let c = l.effective_color();
                    self.point.push(PointLightGpu {
                        position: [l.position.x, l.position.y, l.position.z],
                        radius: l.radius,
                        color: [c.x, c.y, c.z],
                        shadow_layer: NO_SHADOW,
                    });
                }
                Light::Spot(_) => {}
            }
        }
        self.lights_dirty |= self.dir != old_dir || self.point != old_point;
    }

    /// The directional shadow from a `ShadowMap` (its view-projection read from its own buffer).
    pub(crate) fn set_shadow_map(&mut self, shadow_map: Option<&ShadowMap>) {
        self.shadow_map = shadow_map.and_then(|sm| Some((sm.depth_view.clone()?, sm.light_vp_buf.clone()?)));
        self.bindings_dirty = true;
    }

    /// The directional shadow from a `CascadedShadowMap`'s widest cascade.
    pub(crate) fn set_cascaded_shadow_map(&mut self, csm: Option<&CascadedShadowMap>) {
        self.shadow_map = csm.map(|c| (c.far_view.clone(), c.far_view_proj.clone()));
        self.bindings_dirty = true;
    }

    pub(crate) fn set_point_shadows(&mut self, cube: Option<&CubeMapShadowMap>) {
        self.point_shadows = cube.map(|c| c.distance_view.clone());
        self.bindings_dirty = true;
    }

    /// The renderer's spot lights (`Renderer::spot_lights_buffer`) and their atlas.
    pub(crate) fn set_spot_lights(&mut self, lights: Option<&wgpu::Buffer>, shadow_atlas: Option<&SpotShadowAtlas>) {
        self.spot_lights = lights.cloned();
        self.spot_shadows = shadow_atlas.map(|a| a.array_view.clone());
        self.bindings_dirty = true;
    }

    /// Spot lights and an atlas given as they are (tests).
    #[cfg(test)]
    pub(crate) fn set_spot_light_views(&mut self, lights: Option<wgpu::Buffer>, atlas: Option<wgpu::TextureView>) {
        self.spot_lights = lights;
        self.spot_shadows = atlas;
        self.bindings_dirty = true;
    }

    pub(crate) fn has_shadow_map(&self) -> bool {
        self.shadow_map.is_some()
    }

    pub(crate) fn has_point_shadows(&self) -> bool {
        self.point_shadows.is_some()
    }

    pub(crate) fn has_spot_lights(&self) -> bool {
        self.spot_lights.is_some()
    }

    /// The layout entries of bindings 1 and 3-9, visible to compute.
    pub(crate) fn layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
        let compute = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty, count: None };
        let storage = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None };
        vec![
            entry(1, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }),
            entry(3, storage),
            entry(4, storage),
            entry(5, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false }),
            entry(6, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
            entry(7, storage),
            entry(8, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false }),
            entry(9, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison)),
        ]
    }

    /// Make the stand-ins and upload the lights (growing their buffers). True when the bind
    /// groups made from `entries` must be rebuilt.
    pub(crate) fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        if self.gpu.is_none() {
            self.gpu = Some(Self::init_gpu(device, queue));
            self.lights_dirty = true;
            self.bindings_dirty = true;
        }
        if self.lights_dirty {
            let gpu = self.gpu.as_mut().unwrap();
            fn fit<T: Pod>(device: &wgpu::Device, buf: &mut wgpu::Buffer, data: &[T], label: &str) -> bool {
                let needed = (std::mem::size_of_val(data) as u64).max(std::mem::size_of::<T>() as u64);
                let grown = needed > buf.size();
                if grown {
                    *buf = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(label),
                        size: needed.next_power_of_two(),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                }
                grown
            }
            let grew_dir = fit(device, &mut gpu.dir_lights, &self.dir, "ComputeShadows/DirLights");
            let grew_point = fit(device, &mut gpu.point_lights, &self.point, "ComputeShadows/PointLights");
            if grew_dir || grew_point {
                self.bindings_dirty = true;
            }
            if !self.dir.is_empty() {
                queue.write_buffer(&gpu.dir_lights, 0, bytemuck::cast_slice(&self.dir));
            }
            if !self.point.is_empty() {
                queue.write_buffer(&gpu.point_lights, 0, bytemuck::cast_slice(&self.point));
            }
            self.lights_dirty = false;
        }
        std::mem::take(&mut self.bindings_dirty)
    }

    /// Bindings 1 and 3-9 (after `prepare`).
    pub(crate) fn entries(&self) -> Vec<wgpu::BindGroupEntry<'_>> {
        let gpu = self.gpu.as_ref().expect("ComputeShadows::prepare first");
        let (depth, vp) = match &self.shadow_map {
            Some((view, buf)) => (view, buf),
            None => (&gpu.dummy_depth, &gpu.dummy_vp),
        };
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
            wgpu::BindGroupEntry { binding: 3, resource: gpu.dir_lights.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: gpu.point_lights.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(self.point_shadows.as_ref().unwrap_or(&gpu.dummy_atlas)) },
            wgpu::BindGroupEntry { binding: 6, resource: vp.as_entire_binding() },
        ];
        entries.extend(self.spot_entries());
        entries
    }

    /// Bindings 7-9 alone: the spot lights, their atlas and its sampler.
    pub(crate) fn spot_entries(&self) -> [wgpu::BindGroupEntry<'_>; 3] {
        let gpu = self.gpu.as_ref().expect("ComputeShadows::prepare first");
        [
            wgpu::BindGroupEntry { binding: 7, resource: self.spot_lights.as_ref().unwrap_or(&gpu.dummy_spot_lights).as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(self.spot_shadows.as_ref().unwrap_or(&gpu.dummy_spot_atlas)) },
            wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&gpu.spot_sampler) },
        ]
    }

    fn init_gpu(device: &wgpu::Device, queue: &wgpu::Queue) -> Gpu {
        let buffer = |label: &str, size: usize, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size as u64, usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
        };
        // Fallbacks bound when no shadow map is set: a 1x1 depth texture (never sampled, the
        // lookups are gated by flags), a 1x1x6 distance atlas, and an identity light VP.
        let dummy_depth = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("ComputeShadows/DummyShadowDepth"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let dummy_atlas = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("ComputeShadows/DummyPointShadow"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let dummy_vp = buffer("ComputeShadows/DummyLightVP", 64, wgpu::BufferUsages::UNIFORM);
        queue.write_buffer(&dummy_vp, 0, bytemuck::cast_slice(Mat4::identity().as_slice()));
        // no spot lights: a buffer whose count is 0 (zero-initialised), and a 1x1 atlas
        let dummy_spot_lights = buffer(
            "ComputeShadows/DummySpotLights",
            16 + std::mem::size_of::<crate::lights::spot_lights_gpu::SpotLightGpu>(),
            wgpu::BufferUsages::STORAGE,
        );
        let dummy_spot_atlas = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("ComputeShadows/DummySpotShadow"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: SpotShadowAtlas::FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let spot_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ComputeShadows/SpotShadowSampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Gpu {
            dir_lights: buffer("ComputeShadows/DirLights", std::mem::size_of::<DirLightGpu>(), wgpu::BufferUsages::STORAGE),
            point_lights: buffer("ComputeShadows/PointLights", std::mem::size_of::<PointLightGpu>(), wgpu::BufferUsages::STORAGE),
            dummy_depth,
            dummy_atlas,
            dummy_vp,
            dummy_spot_lights,
            dummy_spot_atlas,
            spot_sampler,
        }
    }
}
