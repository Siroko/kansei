use glam::{IVec3, UVec3, Vec3};

use super::clipmap::{ClipmapLayout, VoxelClipmap};
use super::clipmap_inject::{ClipmapGiSettings, ClipmapInjection};
use super::clipmap_voxelize::{ClipRegion, ClipmapVoxelizer};
use super::cones::gradient_sky_lighting;
use crate::renderers::SharedLayouts;

/// What `Renderer::enable_voxel_clipmap` builds.
#[derive(Clone, Copy, Debug)]
pub struct SceneVoxelClipmapOptions {
    /// Levels, at most `MAX_CLIPMAP_LEVELS`: each covers twice the extent of the one before.
    pub levels: u32,
    /// Voxels across each level in x and z (a multiple of 8).
    pub resolution: u32,
    /// Voxels of each level in y (a multiple of 8).
    pub height_resolution: u32,
    /// The finest level's voxels, metres.
    pub voxel_size: f32,
    /// A level's window moves in steps of this many of its voxels, once the camera is that far
    /// from its centre: larger steps move it less often, each a thicker slab.
    pub snap_voxels: u32,
    /// The reference radiance is stored against (see `VoxelClipmap`).
    pub radiance_scale: f32,
    /// The finest levels dynamic renderables (`Renderable::dynamic`) are voxelized into, every
    /// frame (coarser levels leave them out).
    pub dynamic_levels: u32,
    /// Regions (a slab a level's window moved into, or a whole window) voxelized a frame at most.
    pub jobs_per_frame: u32,
}

impl Default for SceneVoxelClipmapOptions {
    fn default() -> Self {
        Self { levels: 5, resolution: 64, height_resolution: 32, voxel_size: 0.5, snap_voxels: 4, radiance_scale: 1.0, dynamic_levels: 3, jobs_per_frame: 1 }
    }
}

impl SceneVoxelClipmapOptions {
    /// The layout these options give.
    pub fn layout(&self) -> ClipmapLayout {
        ClipmapLayout::new(self.levels, [self.resolution, self.height_resolution, self.resolution], self.voxel_size)
    }
}

/// A job a frame may run: voxelize `region` (cleared first), after which its level's window is at
/// `origin`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ClipJob {
    pub region: ClipRegion,
    pub origin: IVec3,
}

/// Voxel GI for an open scene's meshes (the renderer's, `Renderer::enable_voxel_clipmap`): a
/// voxel clipmap (`VoxelClipmap`) around the camera instead of `SceneVoxelGi`'s one fixed box.
/// Each frame, after the shadow maps and before the GBuffer:
/// 1. each level's window follows the camera in steps (`snap_voxels`); the static renderables with
///    a `Renderable::gi` surface are voxelized a region at a time (`jobs_per_frame`): the slab a
///    window moved into, or a whole window when it is first filled, when one of them changed, or
///    on `invalidate`; a window moves when its slab is done, so the levels always hold what their
///    windows cover. The dynamic ones are voxelized into the finest `dynamic_levels` every frame;
/// 2. the voxels are lit level by level (`settings.levels_per_frame`) by the renderer's lights
///    through their shadow maps, or cones through the clipmap where the maps don't reach, plus
///    their emission and a bounce of last frame's light.
///
/// Read it with `VoxelGIEffect::with_clipmap` (screen-space cones), or with `CLIPMAP_WGSL` from any
/// compute pass.
pub struct SceneVoxelClipmap {
    pub settings: ClipmapGiSettings,
    options: SceneVoxelClipmapOptions,
    clipmap: VoxelClipmap,
    voxelizer: ClipmapVoxelizer,
    pub(crate) injection: ClipmapInjection,
    sky: wgpu::Buffer,
    /// Levels to voxelize anew over their whole window (first fill, invalidation).
    stale: Vec<bool>,
    /// This frame's jobs, by slot.
    jobs: Vec<ClipJob>,
    /// The level the round of `levels_per_frame` lights next.
    next_level: u32,
    frame: u64,
}

impl SceneVoxelClipmap {
    pub(crate) fn new(device: &wgpu::Device, shared: &SharedLayouts, light_buf: &wgpu::Buffer, options: SceneVoxelClipmapOptions) -> Self {
        let layout = options.layout();
        let clipmap = VoxelClipmap::new(device, layout, options.radiance_scale);
        let voxelizer = ClipmapVoxelizer::new(device, shared, light_buf, layout, options.jobs_per_frame.max(1), options.dynamic_levels);
        use wgpu::util::DeviceExt;
        // no sky past the clipmap until one is set
        let sky = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VoxelClipmap/Sky"),
            contents: bytemuck::cast_slice(&gradient_sky_lighting([0.0; 3], [0.0; 3])),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let injection = ClipmapInjection::new(device, &clipmap, &sky);
        Self {
            settings: ClipmapGiSettings::default(),
            options,
            stale: vec![true; layout.levels as usize],
            clipmap,
            voxelizer,
            injection,
            sky,
            jobs: Vec::new(),
            next_level: 1,
            frame: 0,
        }
    }

    pub fn options(&self) -> &SceneVoxelClipmapOptions {
        &self.options
    }

    /// The clipmap of the scene's light, for its consumers.
    pub fn clipmap(&self) -> &VoxelClipmap {
        &self.clipmap
    }

    pub fn voxelizer(&self) -> &ClipmapVoxelizer {
        &self.voxelizer
    }

    pub(crate) fn voxelizer_mut(&mut self) -> &mut ClipmapVoxelizer {
        &mut self.voxelizer
    }

    /// Voxelize every level's whole window again, the finest first (after changing something the
    /// voxelizer can't see, such as a material's texture).
    pub fn invalidate(&mut self) {
        self.stale.iter_mut().for_each(|s| *s = true);
    }

    /// The sky past the clipmap, from `down` to `up` (scene radiance): see `gradient_sky_lighting`.
    /// Black until set. Ignored after `use_sky_lighting`.
    pub fn set_sky_gradient(&self, queue: &wgpu::Queue, up: [f32; 3], down: [f32; 3]) {
        queue.write_buffer(&self.sky, 0, bytemuck::cast_slice(&gradient_sky_lighting(up, down)));
    }

    /// Take the sky from `sky_lighting` (a `SkyLighting` uniform such as
    /// `SkyAtmosphereBindings::sky_lighting`) instead of the gradient.
    pub fn use_sky_lighting(&mut self, sky_lighting: &wgpu::Buffer) {
        self.injection.set_sky(sky_lighting);
    }

    /// Bytes on the GPU: the levels' radiance and the surface buffers.
    pub fn memory_bytes(&self) -> u64 {
        self.clipmap.memory_bytes() + self.voxelizer.memory_bytes()
    }

    /// Whether some level is still to be filled for the first time, or again (`invalidate`).
    pub fn filling(&self) -> bool {
        self.stale.iter().any(|&s| s)
    }

    /// This frame's jobs (after `plan`).
    pub(crate) fn jobs(&self) -> &[ClipJob] {
        &self.jobs
    }

    /// Plan this frame's jobs for a camera at `eye`, with `static_changed` when the static
    /// renderables changed (every level is voxelized again): up to `jobs_per_frame` regions, the
    /// most urgent first: a level never filled or stale before any that moved, then the level
    /// whose window the eye is furthest out of (as a share of the window), finer levels first.
    pub(crate) fn plan(&mut self, queue: &wgpu::Queue, eye: Vec3, static_changed: bool) {
        self.jobs.clear();
        if !self.settings.enabled {
            return;
        }
        if static_changed {
            self.invalidate();
        }
        let layout = *self.clipmap.layout();
        let origins: Vec<Option<IVec3>> = (0..layout.levels).map(|level| self.clipmap.origin(level)).collect();
        let mut taken = vec![false; layout.levels as usize];
        for slot in 0..self.voxelizer.job_slots() as usize {
            let Some(job) = next_job(&layout, &origins, &self.stale, &taken, eye, self.options.snap_voxels) else { break };
            taken[job.region.level as usize] = true;
            self.voxelizer.set_job(queue, slot, job.region);
            self.jobs.push(job);
        }
    }

    /// After the jobs are recorded: their levels' windows move (the shaders read the new origins
    /// from this frame on), and the dynamic views follow.
    pub(crate) fn commit_jobs(&mut self, queue: &wgpu::Queue) {
        for job in &self.jobs {
            let level = job.region.level;
            self.clipmap.set_origin(level, Some(job.origin));
            if job.region.size == UVec3::from(self.clipmap.layout().dims) {
                self.stale[level as usize] = false;
            }
        }
        let origins: Vec<Option<IVec3>> = (0..self.voxelizer.dynamic_levels()).map(|level| self.clipmap.origin(level)).collect();
        self.voxelizer.set_dynamic_windows(queue, &origins);
        self.clipmap.upload(queue);
    }

    /// The levels lit this frame: all, or the finest and the next `levels_per_frame - 1` in turn.
    pub(crate) fn levels_to_light(&mut self) -> Vec<u32> {
        let levels = self.clipmap.layout().levels;
        let per_frame = self.settings.levels_per_frame;
        self.frame += 1;
        if per_frame == 0 || per_frame >= levels {
            return (0..levels).collect();
        }
        let mut lit = vec![0];
        for _ in 1..per_frame {
            lit.push(self.next_level);
            self.next_level = if self.next_level + 1 >= levels { 1 } else { self.next_level + 1 };
        }
        lit
    }

    /// Record the injection of `levels`, `any_dynamic` when dynamic renderables were voxelized
    /// this frame.
    pub(crate) fn encode_lighting(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, levels: &[u32], any_dynamic: bool) {
        let dynamic_levels = self.voxelizer.dynamic_levels();
        let has_dynamic = move |level: u32| any_dynamic && level < dynamic_levels;
        self.injection.encode(device, queue, encoder, &self.clipmap, &self.voxelizer, levels, &has_dynamic, &self.settings);
    }
}

/// The most urgent job for a clipmap laid out as `layout` whose levels' windows are at `origins`
/// (None: never filled), for an eye at `eye`, among the levels not `taken`: a level never filled
/// or `stale` (a whole window, at the eye) before any that moved, finer first; then the level whose
/// window the eye is furthest out of, as a share of the window, its move along the axis it moved
/// most (the slab of the moved window outside the old one). None when no level needs one.
pub(crate) fn next_job(layout: &ClipmapLayout, origins: &[Option<IVec3>], stale: &[bool], taken: &[bool], eye: Vec3, snap: u32) -> Option<ClipJob> {
    let dims = UVec3::from(layout.dims).as_ivec3();
    let mut best: Option<(f32, ClipJob)> = None;
    for level in 0..layout.levels {
        if taken[level as usize] {
            continue;
        }
        let (urgency, job) = match origins[level as usize] {
            Some(origin) if !stale[level as usize] => {
                let target = layout.follow(level, origin, eye, snap);
                let delta = target - origin;
                if delta == IVec3::ZERO {
                    continue;
                }
                let axis = (0..3).max_by_key(|&a| (delta[a].abs(), std::cmp::Reverse(a))).unwrap();
                let mut moved = origin;
                moved[axis] = target[axis];
                let d = delta[axis];
                let region = if d.abs() >= dims[axis] {
                    ClipRegion { level, lo: moved, size: dims.as_uvec3() }
                } else {
                    let mut lo = moved;
                    let mut size = dims;
                    if d > 0 {
                        lo[axis] = origin[axis] + dims[axis];
                    }
                    size[axis] = d.abs();
                    ClipRegion { level, lo, size: size.as_uvec3() }
                };
                (d.abs() as f32 / dims[axis] as f32, ClipJob { region, origin: moved })
            }
            current => {
                let origin = match current {
                    Some(origin) => layout.follow(level, origin, eye, snap),
                    None => layout.centred_origin(level, eye, snap),
                };
                (f32::INFINITY, ClipJob { region: ClipRegion { level, lo: origin, size: dims.as_uvec3() }, origin })
            }
        };
        if best.as_ref().is_none_or(|(u, _)| urgency > *u) {
            best = Some((urgency, job));
        }
    }
    best.map(|(_, job)| job)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_levels_fill_finest_first_then_slabs_follow_the_eye() {
        let layout = ClipmapLayout::new(3, [32, 16, 32], 1.0);
        let mut origins = vec![None; 3];
        let stale = [false; 3];
        let free = [false; 3];
        let eye = Vec3::new(0.5, 1.0, 0.5);
        for expect in 0..3 {
            let job = next_job(&layout, &origins, &stale, &free, eye, 4).unwrap();
            assert_eq!(job.region.level, expect);
            assert_eq!(job.region.size, UVec3::new(32, 16, 32));
            assert_eq!(job.region.lo, job.origin);
            origins[expect as usize] = Some(job.origin);
        }
        assert!(next_job(&layout, &origins, &stale, &free, eye, 4).is_none(), "nothing to do while the eye stays");
        // the eye moves 9 m along +x: level 0 (furthest out of its window) moves 8 voxels, its
        // new slab the 8 voxels past its old window
        let moved = eye + Vec3::new(9.0, 0.0, 0.0);
        let job = next_job(&layout, &origins, &stale, &free, moved, 4).unwrap();
        let old = origins[0].unwrap();
        assert_eq!(job.region.level, 0);
        assert_eq!(job.origin, old + IVec3::new(8, 0, 0));
        assert_eq!(job.region.lo, IVec3::new(old.x + 32, old.y, old.z));
        assert_eq!(job.region.size, UVec3::new(8, 16, 32));
        // level 0 taken (another slot has it): level 1 (2 m voxels, 4.75 out) moves one step
        let job = next_job(&layout, &origins, &stale, &[true, false, false], moved, 4).unwrap();
        assert_eq!((job.region.level, job.region.size), (1, UVec3::new(4, 16, 32)));
        // the eye moves back past the start: the slab is on the low side
        let back = eye - Vec3::new(6.0, 0.0, 0.0);
        let job = next_job(&layout, &origins, &stale, &free, back, 4).unwrap();
        assert_eq!(job.region.level, 0);
        assert_eq!(job.origin, old - IVec3::new(4, 0, 0));
        assert_eq!(job.region.lo, job.origin);
        assert_eq!(job.region.size, UVec3::new(4, 16, 32));
        // a stale level is redone whole before any slab
        let job = next_job(&layout, &origins, &[false, false, true], &free, moved, 4).unwrap();
        assert_eq!((job.region.level, job.region.size), (2, UVec3::new(32, 16, 32)));
        // a jump past a window: the whole window at once
        let far = eye + Vec3::new(100.0, 0.0, 0.0);
        let job = next_job(&layout, &origins, &stale, &free, far, 4).unwrap();
        assert_eq!(job.region.size, UVec3::new(32, 16, 32));
        assert_eq!(job.region.lo, job.origin);
    }
}
