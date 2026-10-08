//! Ray tracing a near-field grid of triangles, in compute shaders (WebGPU has no hardware ray
//! tracing): a uniform grid of world triangles in a box round the camera (or a fixed box, a
//! room), rebuilt on the GPU, for sharp and glossy reflections and short rays the voxels are too
//! coarse for. See `docs/plans/2026-10-05-rt-grid-design.md`.
//!
//! The technique combines Hector Arellano's two posts: triangles treated as particles, small
//! enough to fit the grid's cells, with the few big ones kept apart (miaumiau.cat/?p=1457), and a
//! grid of what each voxel holds walked by a 3D DDA, each listed item tested exactly
//! (miaumiau.cat/?p=1476's ray tracing). On compute the lists need no fixed slots: a count, a
//! prefix sum and a fill give each cell an exact list.
//!
//! - [`RtGrid`] holds the grid and builds it: each rebuild, between `begin` and `finish`, one
//!   `gather` appends the triangles of [`RtSource`]s (a mesh placed once per instance record, by
//!   an [`RtPlacement`]) that meet the box; `finish` lists them in the cells they overlap (the
//!   separating-axis test, cells widened by `RtGridOptions::epsilon`): a thread a triangle, a
//!   workgroup a triangle spanning more than 32 columns of cells, and room-sized ones in a short
//!   list every ray tests (`big_triangle_cells`). Its buffers grow to what a build needed, read
//!   back (`read_back`, then `stats`).
//! - [`SceneRtGrid`] (`Renderer::enable_rt_grid`) is the renderer's grid round the camera, fed on
//!   the GPU from the renderables with `Renderable::rt`: its box is a cull view, so instances are
//!   culled for it and cluster LOD cut there at about a cell of error, and the gather reads those
//!   views' draw lists and records directly.
//! - [`RtScene`] places meshes ([`RtMesh`]) on the CPU for a grid with no renderer;
//!   [`split_large_triangles`] splits a mesh's big triangles at load.
//! - [`RT_GRID_WGSL`]'s `kansei_rt_trace` traces it from any compute pass, with the buffers
//!   `RtGrid::bindings_wgsl` declares.
//! - [`RtReflectionsEffect`] traces sharp and glossy reflections through it on the surfaces whose
//!   material writes an F0 (`materials::GBUFFER_OUT_WGSL`'s `kansei_gbuffer_out_specular`), the
//!   hits lit by the voxel GI (the light leaving the surface there), the voxel cone past it.
//! - [`RtDiffuseGiEffect`] traces diffuse GI through it: one ray a pixel (by default one for each
//!   2 x 2) from the GBuffer's surfaces, the hits lit by their exact direct light and one voxel
//!   cone, the voxels past the grid's box, denoised by SVGF; an opt-in GI path beside voxel cones
//!   (`gi::VoxelGIEffect`) and screen-space GI, with a reference path tracer through the grid.

mod diffuse;
mod effect;
mod grid;
mod mesh;
mod scene;
mod scene_grid;

pub use diffuse::{RtDiffuseGiEffect, RtDiffuseGiOptions, RtGiDenoise, RtGiHitLighting, RtGiKernel, RtGiMode, RtGiResolution, RtGiShadows, RtGiStats, RtGiView};
pub use effect::{RtReflectionStats, RtReflectionsEffect, RtReflectionsOptions, RtReflectionsView, RtTraceResolution};
pub use grid::{RtGrid, RtGridHandle, RtGridOptions, RtGridStats, RtPlacement, RtSource, RtSurface, RT_MAX_CELLS, RT_TRIANGLE_BYTES};
pub use mesh::{split_large_triangles, transform_box, RtMesh};
pub use scene::{RtInstance, RtScene};
pub use scene_grid::{SceneRtGrid, SceneRtGridOptions, SceneRtGridStats};

/// Ray tracing through an `RtGrid` from a compute pass: `KanseiRtGrid`, `KanseiRtHit` and
/// `kansei_rt_trace(origin, dir, t_min, t_max, flags)`, the closest hit of a ray among the grid's
/// triangles (`KANSEI_RT_ANY_HIT`: any hit, for shadows; `KANSEI_RT_SOLID`: no alpha test), with
/// `kansei_rt_exit` (where a ray leaves the box), `kansei_rt_contains`, and a hit triangle's
/// `kansei_rt_albedo`, `kansei_rt_uv`, `kansei_rt_source` and `kansei_rt_record`. Declare the
/// buffers with `RtGrid::bindings_wgsl(group, first)` (bind `RtGrid::bind_group_entries`), and
/// define `fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool`, whether an alpha-tested
/// triangle (`RtSurface::with_alpha_layer`) is there at `uv` (`RT_OPAQUE_WGSL`: everywhere).
pub const RT_GRID_WGSL: &str = concat!(include_str!("shaders/rt_types.wgsl"), include_str!("shaders/rt_grid.wgsl"));

/// A `kansei_rt_covered` for `RT_GRID_WGSL` that keeps every hit (no alpha test).
pub const RT_OPAQUE_WGSL: &str = "fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool { return true; }\n";

#[cfg(test)]
mod tests {
    use super::grid::{gather_wgsl, RtGridGpu, RtSourceGpu, BUILD_WGSL};
    use super::*;
    use crate::clusters::InstanceTransform;

    fn validate(name: &str, code: &str) -> naga::Module {
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
        // the strictest capabilities, closest to core WebGPU
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
        module
    }

    fn struct_span(module: &naga::Module, name: &str) -> usize {
        module
            .types
            .iter()
            .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == name => Some(*span as usize),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no struct {name}"))
    }

    #[test]
    fn shaders_validate_rt() {
        let build = validate("build", BUILD_WGSL);
        assert_eq!(struct_span(&build, "KanseiRtGrid"), std::mem::size_of::<RtGridGpu>());
        let placements = [
            RtPlacement::None,
            RtPlacement::Instance(InstanceTransform::Matrix { offset: 16 }),
            RtPlacement::Instance(InstanceTransform::Placement { position: 0, scale: Some(12), yaw: Some(16), yaw_scale: -1.0, rotation: Some(20) }),
            RtPlacement::Instance(InstanceTransform::Placement { position: 4, scale: None, yaw: None, yaw_scale: 1.0, rotation: None }),
        ];
        for p in &placements {
            let gather = validate("gather", &gather_wgsl(&p.wgsl()));
            assert_eq!(struct_span(&gather, "RtSource"), std::mem::size_of::<RtSourceGpu>());
            assert_eq!(struct_span(&gather, "KanseiRtGrid"), std::mem::size_of::<RtGridGpu>());
        }
        // the library as a caller uses it
        let trace = validate(
            "trace library",
            &format!(
                "{RT_GRID_WGSL}\n{}\n{RT_OPAQUE_WGSL}\n@group(0) @binding(0) var<storage, read_write> out: array<vec4f>;\n\
                 @compute @workgroup_size(1) fn main() {{\n\
                     let h = kansei_rt_trace(vec3f(0.0), vec3f(0.0, 0.0, 1.0), 0.0, 100.0, KANSEI_RT_ANY_HIT);\n\
                     out[0] = vec4f(h.normal * h.t, kansei_rt_exit(vec3f(0.0), vec3f(1.0, 0.0, 0.0)));\n\
                     out[1] = vec4f(kansei_rt_albedo(h.triangle), f32(kansei_rt_source(h.triangle) + kansei_rt_record(h.triangle)));\n\
                     out[2] = vec4f(kansei_rt_uv(h.triangle, h.bary), select(0.0, 1.0, kansei_rt_contains(vec3f(1.0))), 0.0);\n\
                 }}",
                RtGrid::bindings_wgsl(1, 4)
            ),
        );
        assert_eq!(struct_span(&trace, "KanseiRtGrid"), std::mem::size_of::<RtGridGpu>());
    }

    #[test]
    fn shaders_validate_reflections() {
        let custom = "fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool { if (uv.x > 1.5) { return true; } return textureSampleLevel(kansei_rt_alpha_texture, kansei_rt_alpha_sampler, uv, 0.0).a >= 0.5; }";
        for clipmap in [true, false] {
            for covered in [super::effect::trace_wgsl(clipmap, "fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool { return true; }"), super::effect::trace_wgsl(clipmap, custom)] {
                let trace = validate(&format!("reflection trace (clipmap {clipmap})"), &covered);
                assert_eq!(struct_span(&trace, "RtReflectParams"), std::mem::size_of::<super::effect::RtReflectParamsGpu>());
                assert_eq!(struct_span(&trace, "KanseiRtGrid"), std::mem::size_of::<RtGridGpu>());
            }
        }
        let resolve = validate("reflection resolve", &super::effect::resolve_wgsl());
        assert_eq!(struct_span(&resolve, "RtReflectParams"), std::mem::size_of::<super::effect::RtReflectParamsGpu>());
        // the material hook
        validate(
            "gbuffer out specular",
            &format!("{}\n@fragment fn main() -> KanseiGBufferOut {{ return kansei_gbuffer_out_specular(vec3f(1.0), vec3f(0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.5), 0.04, 0.2); }}", crate::materials::GBUFFER_OUT_WGSL),
        );
    }

    #[test]
    fn shaders_validate_diffuse_gi() {
        for clipmap in [true, false] {
            let trace = validate(&format!("gi trace (clipmap {clipmap})"), &super::diffuse::trace_wgsl(clipmap, RT_OPAQUE_WGSL));
            assert_eq!(struct_span(&trace, "RtGiParams"), std::mem::size_of::<super::diffuse::RtGiParamsGpu>());
        }
        let svgf = validate("gi svgf", &super::diffuse::svgf_wgsl());
        assert_eq!(struct_span(&svgf, "RtGiParams"), std::mem::size_of::<super::diffuse::RtGiParamsGpu>());
        assert_eq!(struct_span(&svgf, "AtrousParams"), std::mem::size_of::<super::diffuse::AtrousParamsGpu>());
        let composite = validate("gi composite", &super::diffuse::composite_wgsl());
        assert_eq!(struct_span(&composite, "RtGiParams"), std::mem::size_of::<super::diffuse::RtGiParamsGpu>());
    }

    #[test]
    fn placements_place_as_the_cluster_cull_does() {
        // a record: position (1, 2, 3), scale 2, yaw 0.5 (times yaw_scale -1), rotation a quarter
        // turn about x; evaluated by a CPU twin of the generated WGSL's arithmetic
        let q = glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
        let p = glam::Vec3::new(0.3, -0.7, 1.1);
        let scaled = p * 2.0;
        let r = glam::Vec3::new(q.x, q.y, q.z);
        let rotated = scaled + 2.0 * r.cross(r.cross(scaled) + q.w * scaled);
        let a = -0.5f32;
        let yawed = glam::Vec3::new(a.cos() * rotated.x + a.sin() * rotated.z, rotated.y, -a.sin() * rotated.x + a.cos() * rotated.z);
        let placed = yawed + glam::Vec3::new(1.0, 2.0, 3.0);
        // cluster_cull.wgsl: yaw matrix (columns (c, 0, -s), (0, 1, 0), (s, 0, c)) * rotation * scale
        let yaw = glam::Mat3::from_cols(glam::Vec3::new(a.cos(), 0.0, -a.sin()), glam::Vec3::Y, glam::Vec3::new(a.sin(), 0.0, a.cos()));
        let expect = yaw * glam::Mat3::from_quat(q) * (p * 2.0) + glam::Vec3::new(1.0, 2.0, 3.0);
        assert!(placed.distance(expect) < 1e-5, "{placed} vs {expect}");
        // and the WGSL names the record's words
        let code = RtPlacement::Instance(InstanceTransform::Placement { position: 0, scale: Some(12), yaw: Some(16), yaw_scale: -1.0, rotation: Some(20) }).wgsl();
        assert!(code.contains("record, 3u") && code.contains("record, 4u") && code.contains("record, 5u") && code.contains("-1.0"), "{code}");
    }
}
