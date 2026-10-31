//! Ray tracing a near-field grid of triangles, in compute shaders (WebGPU has no hardware ray
//! tracing): a uniform grid of world triangles in a box round the camera (or a fixed box, a
//! room), rebuilt on the GPU, for sharp and glossy reflections and short rays the voxels are too
//! coarse for. See `docs/plans/2026-10-05-rt-grid-design.md`. On native, where the device has
//! hardware ray queries, the same triangles can go into an acceleration structure instead
//! ([`RtBackend`], `docs/rt-hardware.md`): the same `kansei_rt_trace` and hits, traced in hardware.
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
//!   `RtGrid::bindings_wgsl` declares; [`RT_QUERY_WGSL`]'s with hardware ray queries, where
//!   `RtGrid::backend` is `RtBackend::Hardware` (`RtBackend::wgsl` and `bindings_wgsl` pick).
//! - [`RtReflectionsEffect`] traces sharp and glossy reflections through it on the surfaces whose
//!   material writes an F0 (`materials::GBUFFER_OUT_WGSL`'s `kansei_gbuffer_out_specular`), the
//!   hits lit by the voxel GI (the light leaving the surface there), the voxel cone past it.
//! - [`RtDiffuseGiEffect`] traces diffuse GI through it: one ray a pixel (by default one for each
//!   2 x 2) from the GBuffer's surfaces, the hits lit by their exact direct light and one voxel
//!   cone, the voxels past the grid's box, denoised by SVGF; an opt-in GI path beside voxel cones
//!   (`gi::VoxelGIEffect`) and screen-space GI, with a reference path tracer through the grid.
//! - [`RtShadowsEffect`] lights the surfaces that ask for it (`RT_SHADOWS_GBUFFER_WGSL`) by the
//!   sun and the spot lights with ray-traced shadows: any-hit shadow rays through the grid toward
//!   a point of each emitter (soft penumbrae, sharp contacts), screen-space contact rays, denoised.
//! - [`RtCausticsEffect`] adds what glass changes of that direct light (its shadow rays pass
//!   through glass): photons from the sun and the spot lights through each glass caster, into a
//!   tile of texels round it, the light it focuses less the light that passes straight through.

mod caustics;
mod diffuse;
mod effect;
mod grid;
mod mesh;
mod query;
mod scene;
mod scene_grid;
mod shadows;

pub use caustics::{RtCausticReceivers, RtCausticsEffect, RtCausticsOptions, RtCausticsView, RT_CAUSTICS_LOOKUP_WGSL, RT_CAUSTICS_MAX_CASTERS, RT_CAUSTICS_MAX_LIGHTS};
pub use diffuse::{RtDiffuseGiEffect, RtDiffuseGiOptions, RtGiDenoise, RtGiHitLighting, RtGiKernel, RtGiMode, RtGiResolution, RtGiShadows, RtGiStats, RtGiView};
pub use effect::{RtGlass, RtReflectionStats, RtReflectionsEffect, RtReflectionsOptions, RtReflectionsView, RtTraceResolution};
pub use grid::{RtBackend, RtGrid, RtGridBindings, RtGridHandle, RtGridOptions, RtGridStats, RtPlacement, RtSource, RtSurface, RT_MAX_CELLS, RT_TRIANGLE_BYTES};
pub use mesh::{split_large_triangles, transform_box, RtGpuMesh, RtMesh};
pub use scene::{RtInstance, RtScene};
pub use scene_grid::{SceneRtGrid, SceneRtGridOptions, SceneRtGridStats};
pub use shadows::{RtShadowVisibility, RtShadowsEffect, RtShadowsView, RT_SHADOWS_GBUFFER_WGSL, RT_SHADOW_MAX_LIGHTS};

/// Ray tracing through an `RtGrid` from a compute pass: `KanseiRtGrid`, `KanseiRtHit` and
/// `kansei_rt_trace(origin, dir, t_min, t_max, flags)`, the closest hit of a ray among the grid's
/// triangles (`KANSEI_RT_ANY_HIT`: any hit, for shadows; `KANSEI_RT_SOLID`: no alpha test), with
/// `kansei_rt_exit` (where a ray leaves the box), `kansei_rt_contains`, and a hit triangle's
/// `kansei_rt_albedo`, `kansei_rt_uv`, `kansei_rt_source` and `kansei_rt_record`. Declare the
/// buffers with `RtGrid::bindings_wgsl(group, first)` (bind `RtGrid::bind_group_entries`), and
/// define `fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool`, whether an alpha-tested
/// triangle (`RtSurface::with_alpha_layer`) is there at `uv` (`RT_OPAQUE_WGSL`: everywhere).
pub const RT_GRID_WGSL: &str = concat!(include_str!("shaders/rt_types.wgsl"), include_str!("shaders/rt_hit.wgsl"), include_str!("shaders/rt_grid.wgsl"));

/// `RT_GRID_WGSL` with hardware ray queries (`RtBackend::Hardware`, native only): the same
/// functions and `KanseiRtHit`, over the grid's acceleration structure. Declare its bindings with
/// `RtBackend::Hardware.bindings_wgsl(group, first)` (bind `RtGrid::bind_group_entries`). It
/// starts `enable wgpu_ray_query;`, which must open the module: pass the assembled module through
/// [`hoist_wgsl_enables`].
pub const RT_QUERY_WGSL: &str = concat!(include_str!("shaders/rt_types.wgsl"), include_str!("shaders/rt_hit.wgsl"), include_str!("shaders/rt_query.wgsl"));

/// A WGSL module with its `enable` directives (`RT_QUERY_WGSL`'s, spliced mid-module) moved to
/// its start, once each, as WGSL requires; the rest as it was.
pub fn hoist_wgsl_enables(code: &str) -> String {
    let mut enables: Vec<&str> = Vec::new();
    let mut rest = String::with_capacity(code.len());
    for line in code.split_inclusive('\n') {
        let t = line.trim();
        if t.starts_with("enable ") && t.ends_with(';') {
            if !enables.contains(&t) {
                enables.push(t);
            }
        } else {
            rest.push_str(line);
        }
    }
    if enables.is_empty() {
        return code.to_string();
    }
    format!("{}\n{rest}", enables.join("\n"))
}

/// Light packed in one word for `RtSurface::record_emission` (RGBE): `kansei_rt_pack_emission(radiance)`
/// for the pass that writes the instance records, `kansei_rt_unpack_emission(word)` for a material
/// that reads them back; [`pack_emission`] is its CPU twin. A hit's is `RT_GRID_WGSL`'s
/// `kansei_rt_emission`.
pub const RT_EMISSION_WGSL: &str = include_str!("shaders/rt_emission.wgsl");

/// `RT_EMISSION_WGSL`'s `kansei_rt_pack_emission` on the CPU: `radiance` (cd/m², linear rgb) as
/// RGBE, r | g << 8 | b << 16 | e << 24, worth (m + 0.5) x 2^(e - 136); 0 is black.
pub fn pack_emission(radiance: [f32; 3]) -> u32 {
    let c = radiance.map(|x| x.max(0.0));
    let peak = c[0].max(c[1]).max(c[2]);
    if peak < 1e-30 {
        return 0;
    }
    let x = (peak.log2().floor() as i32 + 1).clamp(-127, 127);
    let m = c.map(|v| ((v * 2f32.powi(8 - x)) as u32).min(255));
    m[0] | m[1] << 8 | m[2] << 16 | ((x + 128) as u32) << 24
}

/// `RT_EMISSION_WGSL`'s `kansei_rt_unpack_emission` on the CPU.
pub fn unpack_emission(word: u32) -> [f32; 3] {
    let e = word >> 24;
    if e == 0 {
        return [0.0; 3];
    }
    [word & 255, (word >> 8) & 255, (word >> 16) & 255].map(|m| (m as f32 + 0.5) * 2f32.powi(e as i32 - 136))
}

/// A `kansei_rt_covered` for `RT_GRID_WGSL` that keeps every hit (no alpha test).
pub const RT_OPAQUE_WGSL: &str = "fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool { return true; }\n";

#[cfg(test)]
mod tests {
    use super::grid::{gather_wgsl, RtGridGpu, RtLayerGpu, RtSourceGpu, BUILD_WGSL};
    use super::*;

    const BACKENDS: [RtBackend; 2] = [RtBackend::Software, RtBackend::Hardware];
    use crate::clusters::InstanceTransform;

    fn validate(name: &str, code: &str) -> naga::Module {
        validate_for(RtBackend::Software, name, code)
    }

    /// Validated for `backend`: the strictest capabilities, closest to core WebGPU (whose
    /// unpack2x16float naga 30 counts as f16-in-f32, a capability wgpu grants every WebGPU
    /// device), plus ray queries for the hardware backend (native only).
    fn validate_for(backend: RtBackend, name: &str, code: &str) -> naga::Module {
        let name = format!("{name} ({backend:?})");
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
        let mut capabilities = naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32;
        if backend == RtBackend::Hardware {
            capabilities |= naga::valid::Capabilities::RAY_QUERY;
        }
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
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
        assert_eq!(struct_span(&build, "KanseiRtLayer"), std::mem::size_of::<RtLayerGpu>());
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
        // the library as a caller uses it, through either backend
        for backend in BACKENDS {
            let trace = validate_for(
                backend,
                "trace library",
                &hoist_wgsl_enables(&format!(
                    "{}\n{}\n{RT_OPAQUE_WGSL}\n@group(0) @binding(0) var<storage, read_write> out: array<vec4f>;\n\
                     @compute @workgroup_size(1) fn main() {{\n\
                         let h = kansei_rt_trace(vec3f(0.0), vec3f(0.0, 0.0, 1.0), 0.0, 100.0, KANSEI_RT_ANY_HIT);\n\
                         out[0] = vec4f(h.normal * h.t, kansei_rt_exit(vec3f(0.0), vec3f(1.0, 0.0, 0.0)));\n\
                         out[1] = vec4f(kansei_rt_albedo(h.triangle), f32(kansei_rt_source(h.triangle) + kansei_rt_record(h.triangle)));\n\
                         out[2] = vec4f(kansei_rt_uv(h.triangle, h.bary), select(0.0, 1.0, kansei_rt_contains(vec3f(1.0))), 0.0);\n\
                     }}",
                    backend.wgsl(),
                    backend.bindings_wgsl(1, 4)
                )),
            );
            assert_eq!(struct_span(&trace, "KanseiRtGrid"), std::mem::size_of::<RtGridGpu>());
        }
        // the hardware backend's BLAS input
        let build = validate("query build", super::query::BUILD_WGSL);
        assert_eq!(struct_span(&build, "KanseiRtQueryParams"), std::mem::size_of::<super::query::QueryParamsGpu>());
    }

    #[test]
    fn enables_open_the_module() {
        let code = hoist_wgsl_enables(&format!("struct A {{ x: f32 }}\n{RT_QUERY_WGSL}\nenable wgpu_ray_query;\nfn f() {{}}\n"));
        assert!(code.starts_with("enable wgpu_ray_query;\n"), "{}", &code[..80]);
        assert_eq!(code.matches("enable ").count(), 1, "once");
        assert!(code.contains("struct A { x: f32 }\n") && code.ends_with("fn f() {}\n"));
        // a module without enables is left as it was
        assert_eq!(hoist_wgsl_enables(RT_GRID_WGSL), RT_GRID_WGSL);
        assert!(!RT_GRID_WGSL.contains("enable "), "the software library stays core WebGPU");
    }

    /// Buffers and acceleration structures a stage binds, which Metal draws from one table of 31
    /// (wgpu 30 keeps 2: `max_buffers_and_acceleration_structures_per_shader_stage` is 29).
    fn metal_buffer_slots(groups: &[Vec<wgpu::BindGroupLayoutEntry>]) -> usize {
        groups.iter().flatten().filter(|e| matches!(e.ty, wgpu::BindingType::Buffer { .. } | wgpu::BindingType::AccelerationStructure { .. })).count()
    }

    #[test]
    fn rt_passes_fit_metals_buffer_slots() {
        for backend in BACKENDS {
            for clipmap in [true, false] {
                let gi = metal_buffer_slots(&super::diffuse::trace_layout_entries(clipmap, backend));
                let reflections = metal_buffer_slots(&super::effect::trace_layout_entries(clipmap, backend)).max(metal_buffer_slots(&super::effect::glass_layout_entries(clipmap, backend)));
                assert!(gi <= 29 && reflections <= 29, "{backend:?}, clipmap {clipmap}: GI trace {gi}, reflections {reflections} of 29");
            }
            let shadows = metal_buffer_slots(&super::shadows::trace_layout_entries(backend));
            assert!(shadows <= 29, "{backend:?}: shadows trace {shadows} of 29");
            let caustics = metal_buffer_slots(&super::caustics::photon_layout_entries(backend));
            assert!(caustics <= 29, "{backend:?}: caustics photons {caustics} of 29");
        }
    }

    /// Sampled textures a stage binds: core WebGPU's limit is 16 (`maxSampledTexturesPerShaderStage`).
    fn sampled_textures(groups: &[Vec<wgpu::BindGroupLayoutEntry>]) -> usize {
        groups.iter().flatten().filter(|e| matches!(e.ty, wgpu::BindingType::Texture { .. })).count()
    }

    #[test]
    fn rt_passes_fit_webgpus_sampled_textures() {
        for backend in BACKENDS {
            for clipmap in [true, false] {
                let gi = sampled_textures(&super::diffuse::trace_layout_entries(clipmap, backend));
                let reflections = sampled_textures(&super::effect::trace_layout_entries(clipmap, backend));
                let glass = sampled_textures(&super::effect::glass_layout_entries(clipmap, backend));
                assert!(gi <= 16 && reflections <= 16 && glass <= 16, "{backend:?}, clipmap {clipmap}: GI trace {gi}, reflections {reflections}, glass {glass} of 16");
            }
        }
    }

    #[test]
    fn shaders_validate_reflections() {
        let custom = "fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool { if (uv.x > 1.5) { return true; } return textureSampleLevel(kansei_rt_alpha_texture, kansei_rt_alpha_sampler, uv, 0.0).a >= 0.5; }";
        for clipmap in [true, false] {
            let albedo = super::effect::DEFAULT_HIT_ALBEDO_WGSL;
            // the albedo hook as a surface would write it: the source tells the floor apart
            let hook = "fn kansei_rt_hit_albedo(triangle: u32, p: vec3f, bary: vec2f) -> vec3f { if (kansei_rt_source(triangle) == 3u) { return vec3f(fract(p.x)); } return kansei_rt_albedo(triangle) * (1.0 + 0.0 * bary.x) + 0.0 * rp.invView[3].xyz; }\n";
            for backend in BACKENDS {
                for covered in [
                    super::effect::trace_wgsl(clipmap, backend, &format!("fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool {{ return true; }}\n{albedo}")),
                    super::effect::trace_wgsl(clipmap, backend, &format!("{custom}\n{albedo}")),
                    super::effect::trace_wgsl(clipmap, backend, &format!("{custom}\n{hook}")),
                ] {
                    let trace = validate_for(backend, &format!("reflection trace (clipmap {clipmap})"), &covered);
                    assert_eq!(struct_span(&trace, "RtReflectParams"), std::mem::size_of::<super::effect::RtReflectParamsGpu>());
                    assert_eq!(struct_span(&trace, "KanseiRtGrid"), std::mem::size_of::<RtGridGpu>());
                }
                let glass = validate_for(backend, &format!("glass (clipmap {clipmap})"), &super::effect::glass_wgsl(clipmap, backend, &format!("{custom}\n{albedo}")));
                assert_eq!(struct_span(&glass, "RtReflectParams"), std::mem::size_of::<super::effect::RtReflectParamsGpu>());
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
            for backend in BACKENDS {
                let trace = validate_for(backend, &format!("gi trace (clipmap {clipmap})"), &super::diffuse::trace_wgsl(clipmap, backend, RT_OPAQUE_WGSL));
                assert_eq!(struct_span(&trace, "RtGiParams"), std::mem::size_of::<super::diffuse::RtGiParamsGpu>());
            }
        }
        let svgf = validate("gi svgf", &super::diffuse::svgf_wgsl());
        assert_eq!(struct_span(&svgf, "RtGiParams"), std::mem::size_of::<super::diffuse::RtGiParamsGpu>());
        assert_eq!(struct_span(&svgf, "AtrousParams"), std::mem::size_of::<super::diffuse::AtrousParamsGpu>());
        let composite = validate("gi composite", &super::diffuse::composite_wgsl());
        assert_eq!(struct_span(&composite, "RtGiParams"), std::mem::size_of::<super::diffuse::RtGiParamsGpu>());
    }

    #[test]
    fn shaders_validate_rt_shadows() {
        for backend in BACKENDS {
            let trace = validate_for(backend, "shadows trace", &super::shadows::trace_wgsl(backend));
            assert_eq!(struct_span(&trace, "RtShadowParams"), super::shadows::params_size());
            assert_eq!(struct_span(&trace, "RtShadowLight"), super::shadows::light_size());
            assert_eq!(struct_span(&trace, "KanseiRtGrid"), std::mem::size_of::<RtGridGpu>());
        }
        for (name, code) in [("shadows filter", super::shadows::filter_wgsl()), ("shadows composite", super::shadows::composite_wgsl())] {
            let module = validate(name, &code);
            assert_eq!(struct_span(&module, "RtShadowParams"), super::shadows::params_size());
            assert_eq!(struct_span(&module, "RtShadowLight"), super::shadows::light_size());
        }
        // the material hook
        validate(
            "gbuffer out rt lit",
            &format!(
                "{}\n{RT_SHADOWS_GBUFFER_WGSL}\n@fragment fn main() -> KanseiGBufferOut {{ return kansei_gbuffer_out_rt_lit(vec3f(0.0), vec3f(0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.5), 0.6, 0.04, false); }}",
                crate::materials::GBUFFER_OUT_WGSL
            ),
        );
    }

    #[test]
    fn shaders_validate_rt_caustics() {
        let sizes = super::caustics::struct_sizes();
        let check = |module: &naga::Module, names: &[&str]| {
            for (name, size) in sizes.iter().filter(|(n, _)| names.contains(n)) {
                assert_eq!(struct_span(module, name), *size, "{name}");
            }
        };
        for backend in BACKENDS {
            let photons = validate_for(backend, "caustics photons", &super::caustics::photons_wgsl(backend));
            check(&photons, &["CausticsParams", "CausticTask", "KanseiRtCausticTile", "KanseiRtCaustics"]);
            assert_eq!(struct_span(&photons, "KanseiRtGrid"), std::mem::size_of::<RtGridGpu>());
        }
        let resolve = validate("caustics resolve", &super::caustics::resolve_wgsl());
        check(&resolve, &["KanseiRtCausticTile", "KanseiRtCaustics", "CompositeParams"]);
        // the lookup as a pass that lights surfaces itself reads it
        validate(
            "caustics lookup",
            &format!(
                "{RT_CAUSTICS_LOOKUP_WGSL}\n@group(0) @binding(0) var map: texture_2d_array<f32>;\n@group(0) @binding(1) var smp: sampler;\n@group(0) @binding(2) var<uniform> c: KanseiRtCaustics;\n\
                 @group(0) @binding(3) var<storage, read_write> out: array<vec4f>;\n@compute @workgroup_size(1) fn main() {{ out[0] = vec4f(kansei_rt_caustics_lookup(map, smp, c, out[1].xyz, vec3f(0.0, 1.0, 0.0)), 1.0); }}"
            ),
        );
    }

    #[test]
    fn emission_packs_within_a_percent_of_its_peak() {
        validate(
            "emission",
            &format!("{RT_EMISSION_WGSL}\n@group(0) @binding(0) var<storage, read_write> out: array<vec4f>;\n@compute @workgroup_size(1) fn main() {{ out[0] = vec4f(kansei_rt_unpack_emission(kansei_rt_pack_emission(out[0].rgb)), 0.0); }}"),
        );
        assert_eq!(pack_emission([0.0; 3]), 0);
        assert_eq!(unpack_emission(0), [0.0; 3]);
        for radiance in [[1.0, 0.5, 0.25], [60.0, 22.0, 4.0], [0.003, 0.0, 0.002], [12000.0, 9000.0, 20000.0], [255.0, 256.0, 0.5]] {
            let back = unpack_emission(pack_emission(radiance));
            let peak = radiance.iter().cloned().fold(0.0, f32::max);
            for k in 0..3 {
                assert!((back[k] - radiance[k]).abs() <= 0.01 * peak, "{radiance:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn surface_words_carry_their_flags() {
        assert_eq!(RtSurface::new([1.0; 3]).surface_word(), 0);
        assert_eq!(RtSurface::new([1.0; 3]).with_smooth_normals().surface_word(), 4);
        assert_eq!(RtSurface::new([1.0; 3]).with_alpha_layer(2).with_smooth_normals().surface_word(), 1 | 2 << 8 | 4);
        // glass (KANSEI_RT_GLASS) is smooth too
        assert_eq!(RtSurface::glass([0.9; 3]).surface_word(), 4 | 8);
        // the albedo's alpha is 1 - F0: opaque for a matte surface
        assert_eq!(RtSurface::new([1.0, 0.0, 0.0]).albedo_word(), 255 | 255 << 24);
        assert_eq!(RtSurface::new([0.0; 3]).with_specular(0.04).albedo_word() >> 24, 245);
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
