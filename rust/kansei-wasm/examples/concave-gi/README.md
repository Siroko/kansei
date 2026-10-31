# Concave GI

A U channel, an arch and a bowl of wood, a few centimetres each, on a desk under a warm key spot
light and a studio HDRI, to compare a mid tier's diffuse GI on hollows only a few voxels across
(the voxel volume at `Medium`: 96 voxels over the desk's 1.3 m, 1.35 cm each). The materials have
no sky ambient of their own: the GI brings the sky, so where it is wrong the image shows it.

- `gi=probes`: the voxels' irradiance probes (`SdfProbes`, one every 4 voxels, `spacing=`) read by
  `VoxelGIEffect::set_probes`, with the distance field's AO at 0.8 on their light (`sdf_ao=`). The
  probes hold the light where they sit, up to a spacing off the surface: in a hollow narrower than
  that they see more sky than its floor does (without the AO, the channel's inner walls 1.8 times
  as bright as the RT GI has them, the desk at the bowl's foot 1.5 times), and the AO, which can't
  tell what they already see, darkens what they do see a second time (with it, the desk under the
  arch at 0.35 of its light, the tunnel's walls 0.5, the pieces' feet and the channel's walls
  0.7-0.8). Denser probes don't fix it (one every 2 voxels measures the same): the hollows are a
  few voxels across, and the probes sit off them.
- `gi=rt` (the default): the hybrid RT GI (`RtDiffuseGiEffect`), rays through the grid of the
  pieces' triangles (1.5 cm cells), the voxels for the hits' further bounces and past 12 cm. At
  `res=quarter` (the default) a ray for each 4 x 4 pixels, `RtGiResolution::Quarter`; at
  `res=half` one for each 2 x 2, what a high tier runs.

`kansei-native`'s `concave_gi` example renders the same scene headless and measures each against
the half-resolution RT GI, by the mean colour of each piece's faces:

```text
cargo run --release -p kansei-native --example concave_gi -- --modes probes,rt-quarter,rt,reference --out <dir>
```

| URL parameter | Effect |
|---|---|
| `gi=probes\|rt` | the probes, or the RT GI (default) |
| `res=quarter\|half` | the RT GI's rays: one for each 4 x 4 pixels (default) or 2 x 2 |
| `cam=wide\|channel\|arch\|bowl` | the starting view (default wide) |
| `view=lit\|indirect` | the lit image (default), or the GI's light alone |
| `sdf_ao=<0..1>` | the probes' AO from the distance field (default 0.8) |
| `spacing=<voxels>` | voxels between probes (default 4) |
| `cone_steps=`, `cone_tan=`, `hit_cone_steps=`, `rt_shadows=rays\|maps` | the RT GI's voxel cone past 12 cm, its hits' cone and their shadows |
| `env=hdri\|gradient` | the studio HDRI (default; a studio gradient where it isn't found) |
| `ev=<EV100>` | the exposure (default 3.2) |
| `perf=1` | the performance overlay: each pass's GPU time (`window.kanseiPerf.report()`) |

Controls: drag to orbit, wheel or pinch to zoom, right-drag or shift-drag to pan.

The HDRI is gi-box's (Poly Haven's CC0 `studio_small_09`), fetched from beside this page on the
site (`../gi-box/assets/`) or from the source tree when `rust/kansei-wasm/examples/` is served
(`../../gi-box/www/assets/`).

Build: `wasm-pack build --target web --release` here, serve `rust/kansei-wasm/examples/` (for the
HDRI) and open `concave-gi/www/`.
