# Impostors

A lake ringed by a forest of spruces (40 000 by default), mirrored in the water by a planar
reflection. The spruces have two mesh LODs and, beyond `far` metres, an octahedral impostor baked
at start-up from the nearest LOD: a billboard per tree that reads the albedo, normal and depth of
the three baked views nearest its view direction and shades them as the mesh's material does. The
impostor is baked from the whole sphere of directions, so the reflection sees the far shore from
below.

Engine API: `Renderer::bake_impostor` (`ImpostorOptions`), `impostors::IMPOSTOR_WGSL` and
`billboard_geometry`, `InstanceCulling` (`with_lod_range`, `with_crossfade`,
`culling::LOD_FADE_WGSL`), `PlanarReflection` (`PLANAR_REFLECTION_WGSL`),
`Renderer::culling_stats`, `TemporalAAEffect`, `ToneMapEffect`, and for the diagnostics
`pacing::FrameTimer` and `profiling::AbBench`.

| URL parameter | Effect |
|---|---|
| `cam=shore\|low\|high\|fly\|forest` | starting camera (default `shore`); `fly` circles the lake, `forest` rises and falls over the water by the south shore |
| `trees=<n>` | spruces (default 40 000) |
| `far=<metres>` | where the impostors start (default 120; the coarser mesh LOD covers 50 m to there) |
| `impostors=0` | start with the impostors off: the coarser mesh LOD reaches the horizon |
| `depth=1` | the impostors write the depth of the surface they find (`frag_depth`, which costs) |
| `frames=<n>` | the bake: views per side of the atlas (default 12, so 12 x 12) |
| `frame=<texels>` | the bake: texels per side of a view, a power of two (default 128) |
| `fade=<metres>` | dithered crossfades that wide between the LODs and into the impostors (default 0: none) |
| `showfade=1` | tint the fading instances: red fading out, blue fading in |
| `taa=0` | no temporal anti-aliasing |
| `t=<seconds>` | freeze the `fly` and `forest` paths at that time |
| `size=<w>x<h>` | draw at exactly that many pixels whatever the window's size |
| `bench=1` | diagnostic: alternate the impostors on and off every 3 s, 8 times, and report the mean GPU time and frame interval of each |
| `bench=fade` | diagnostic: as `bench=1`, alternating the crossfades (`fade=`, default 20 m) on and off |
| `bench=bands` | diagnostic: alternate the crossfade width between `fade=` (default 20 m) and a millimetre |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: I (or the panel's checkbox) toggles the impostors, C cycles the camera. The HUD shows
the instances drawn for the camera and the reflection, the bake time and the GPU time.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
