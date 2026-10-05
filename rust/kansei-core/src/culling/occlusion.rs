use super::{CullPipeline, CullView, DepthPyramid, DepthReduction, OcclusionView};
use crate::cameras::Camera;

/// The renderer's occlusion-culling state: the switch, a pending history reset, the frozen main
/// view (debugging), and per view culled in two phases the depth pyramid its second phase tests
/// against.
pub(crate) struct Occlusion {
    pub enabled: bool,
    pub freeze: bool,
    reset: bool,
    frozen: Option<MainView>,
    pyramids: Vec<(usize, DepthPyramid, wgpu::BindGroup)>,
}

/// The main view to cull with this frame.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MainView {
    pub cull: CullView,
    pub lod_origin: glam::Vec3,
    pub view: glam::Mat4,
    /// the projection as rasterized (jittered)
    pub proj: glam::Mat4,
}

impl MainView {
    pub fn occlusion(&self, depth_size: (u32, u32)) -> OcclusionView {
        OcclusionView { view: self.view, proj: self.proj, depth_size, reverse_z: false, linear_depth: false }
    }
}

impl Occlusion {
    pub fn new() -> Self {
        Self { enabled: true, freeze: false, reset: false, frozen: None, pyramids: Vec::new() }
    }

    pub fn request_reset(&mut self) {
        self.reset = true;
    }

    /// Whether to forget which instances were visible this frame: on request, or on a camera cut.
    pub fn take_reset(&mut self, camera_cut: bool) -> bool {
        std::mem::take(&mut self.reset) || camera_cut
    }

    /// The camera's view, or while frozen the one it had when frozen.
    pub fn main_view(&mut self, camera: &Camera) -> MainView {
        let live = MainView {
            cull: CullView { view_proj: camera.projection_matrix.to_glam() * camera.view_matrix.to_glam(), casters_only: false, reflection: false, gi: false, layer_mask: None, lod_distance_scale: 1.0 },
            lod_origin: camera.inverse_view_matrix.to_glam().w_axis.truncate(),
            view: camera.view_matrix.to_glam(),
            proj: camera.jittered_projection().to_glam(),
        };
        if !self.freeze {
            self.frozen = None;
            return live;
        }
        *self.frozen.get_or_insert(live)
    }

    /// Whether the main view is frozen: its first phase draws what was visible when it froze,
    /// and there is no second phase.
    pub fn frozen(&self) -> bool {
        self.frozen.is_some()
    }

    /// Cull view `view`'s pyramid for a depth buffer of `size` (created or resized as needed),
    /// and the `late` pipeline's bind group for it.
    pub fn pyramid_for(&mut self, view: usize, device: &wgpu::Device, size: (u32, u32), pipeline: &CullPipeline) -> (&DepthPyramid, &wgpu::BindGroup) {
        let k = match self.pyramids.iter().position(|(v, _, _)| *v == view) {
            Some(k) if self.pyramids[k].1.source_size() == size => k,
            found => {
                let pyramid = DepthPyramid::new(device, size.0, size.1, DepthReduction::Max);
                let bind_group = pipeline.pyramid_bind_group(device, &pyramid);
                match found {
                    Some(k) => {
                        self.pyramids[k] = (view, pyramid, bind_group);
                        k
                    }
                    None => {
                        self.pyramids.push((view, pyramid, bind_group));
                        self.pyramids.len() - 1
                    }
                }
            }
        };
        let (_, pyramid, bind_group) = &self.pyramids[k];
        (pyramid, bind_group)
    }

    /// Cull view `view`'s pyramid, once built.
    pub fn pyramid(&self, view: usize) -> Option<&DepthPyramid> {
        self.pyramids.iter().find(|(v, _, _)| *v == view).map(|(_, p, _)| p)
    }
}
