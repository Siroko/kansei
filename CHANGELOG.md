# Changelog

The `kansei` npm package is the TypeScript engine in `src/`. The Rust engine in `rust/` is not
published to npm or crates.io (see the README for depending on it from git).

## 0.1.0

The first release since 0.0.11 (January 2025). The TypeScript engine was rebuilt alongside the
Rust engine: the two now share their bind-group layout, depth convention and much of their WGSL,
which changes the contract custom shaders are written against. Read
[Breaking changes](#breaking-changes-and-how-to-migrate) before upgrading.

### Highlights

**Rendering**
- Stock materials: `Material.basicLit` (forward Blinn-Phong), `Material.standardLit` (GBuffer:
  scene lights, shadows, sky hemisphere, emission, instancing, mirror and glass via
  `mirrorOptions` and `glassOptions`), `Material.emissive` and `Material.gradientSky`.
- Deferred rendering through `PostProcessingVolume` and its `GBuffer` (single-sample with motion
  vectors and `TemporalAAEffect`, or 4x MSAA with `{ msaaSampleCount: 4 }`).
- New geometry: `CylinderGeometry`, `IcosphereGeometry`, `HeightfieldGeometry`,
  `SpruceGeometry`, `Geometry.fromArrays`; 32-bit indices by default.
- Web plumbing shared by the examples: `Canvas` (sizes the drawing buffer to the CSS box and
  device pixel ratio), `run` (frame loop with a cap on frames in flight), `Keys`, `Gamepad`,
  `param`/`paramOr`/`flag`.
- Pacing and profiling: `FixedStep`, `FrameTimer`, `Renderer.setProfiling` with per-pass GPU
  times (`gpuPass`) and CPU sections, `AbBench`.

**Lighting and shadows**
- Directional, point, spot and area lights in a scene light uniform every camera binds; clustered
  light lists for many spot lights.
- Directional shadow maps, cascaded shadows, point-light cube shadows and a spot shadow atlas
  (`Renderer.enableShadows`, `enableCascadedShadows`, `enablePointShadows`, ...), plus sky
  occlusion and compute-pass shadow lookups (`ComputeShadows`).
- Physically based `SkyAtmosphere` with `AtmosphereEffect`, `HeightFogEffect`,
  `VolumetricCloudsEffect`, and froxel `VolumetricFogEffect` with local fog volumes.

**Global illumination and ray tracing**
- Voxel GI of meshes (`Renderer.enableVoxelGI`, `VoxelGIEffect`), a voxel clipmap round the
  camera with probes (`SceneVoxelClipmap`), and particle voxel GI (`ParticleGi`).
- `ScreenSpaceGIEffect`.
- A GPU-built grid of the scene's triangles (`SceneRtGrid`, `RtGrid`) with hybrid ray-traced
  diffuse GI (`RtDiffuseGiEffect`, SVGF, a reference mode), `RtReflectionsEffect`, and mirror and
  glass materials.
- BVH path tracer (`PathTracerEffect`).

**Post-processing**
- `ToneMapEffect` (EV100 exposure, film curves, white balance, local exposure),
  `ColorGradingEffect`, `BloomEffect`, `TemporalAAEffect`, `MotionBlurEffect`, `SSAOEffect`,
  `GodRaysEffect`, `DepthOfFieldEffect` and the physical-lens `CinematicDepthOfFieldEffect`.

**Visibility, LOD and reflections**
- GPU instance culling per view (`InstanceCulling`): frustum, LOD bands with crossfades and
  two-phase occlusion against a `DepthPyramid`.
- Cluster LOD (`buildClusterMesh`, `ClusterLod`): the graph built on the CPU, cut per view on the
  GPU, with card clusters for foliage.
- Octahedral impostors (`Impostor`, `bakeImpostor`) and `PlanarReflection` (also screen-space).

**Simulation**
- GPU particle fluids (`FluidSimulation`): SPH or Position Based Fluids on a shared
  `NeighbourGrid`, containers, colliders, nozzles, sleeping at rest (`FluidStepper`).
- Fluid surfaces: `FluidDensityField`, `FluidMarchingCubes`, `FluidSurfaceEffect`,
  `FluidRaymarchEffect`.

**Animation and collision**
- Skeletal skinning in the materials' vertex shaders (`SkinnedGltf`, `skinnedLitMaterial`, ...).
- Motion matching (`MotionMatcher`, `CharacterController`), inertialization, foot IK, warping and
  traversal planning; `CollisionWorld` with ray and sphere casts.

**Loaders and text**
- `GLTFLoader` (glTF/GLB, `KHR_texture_basisu`) and `KTX2Loader` (Basis Universal ETC1S and
  UASTC, transcoded to BC, ASTC or ETC2 by what the device samples). The transcoder's
  WebAssembly is inlined in the package and loaded only when a page first decodes a KTX2 file.
- `FontLoader` parses `.arfont` atlases in TypeScript (no WebAssembly any more), with
  `parseArFont` for bytes already in memory.

**Shared WGSL**
- Many shaders come from the Rust engine's sources (validated there by naga) and are exported
  for custom materials: `LIGHTS_WGSL`, `SHADOW_MAP_WGSL`, `GBUFFER_OUT_WGSL`, `TONEMAP_WGSL`,
  the GI, RT and atmosphere chunks, and `assemble` to build a shader the way the Rust engine
  does. `ShaderChunks` gains `shadows`, `lights`, `gbufferOut` and `spotLights`.

### Packaging

- `exports` map (`import` and `types` for `kansei`), `sideEffects: false`, and the build runs on
  `npm pack`/`npm publish` (`prepack`).
- `gl-matrix` is a dependency (its types appear in the declarations); `@webgpu/types` stays a
  peer dependency, and the declarations reference it, so a TypeScript project needs no
  `"types": ["@webgpu/types"]` entry to compile against kansei.
- The Basis Universal transcoder's licence ships in `dist/loaders/ktx2/basis/LICENSE`.

### Breaking changes and how to migrate

Code that only uses the engine's classes and stock materials mostly keeps working. What changes
is the contract custom WGSL is written against, and a few defaults.

#### Bind groups: the camera is group 1, the mesh is group 2

0.0.11 bound the mesh at group 1 and the camera at group 2. Both engines now use one layout
(`BindGroupSlot`, `src/renderers/SharedLayouts.ts`):

| Group | 0.0.11 | 0.1.0 |
|---|---|---|
| 0 | the material's bindings | the material's bindings |
| 1 | mesh: normal matrix @0, world matrix @1 | camera: view @0, projection @1, scene lights @2 (fragment), temporal data @3 |
| 2 | camera: view @0, projection @1 | mesh: normal matrix @0, world + previous world @1 |
| 3 | (none) | shadows and lights, fragment stage only |

Before (0.0.11):

```wgsl
@group(1) @binding(0) var<uniform> normalMatrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> worldMatrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> viewMatrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> projectionMatrix: mat4x4<f32>;
```

After (0.1.0): swap the group numbers.

```wgsl
@group(1) @binding(0) var<uniform> viewMatrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projectionMatrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normalMatrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> worldMatrix: mat4x4<f32>;
```

Mesh binding 1 now holds the world matrix followed by the previous frame's (128 bytes, for
motion vectors); a shader that declares only `mat4x4<f32>` there reads the current world
matrix. To read both, prepend `MOTION_VECTORS_WGSL`, which declares `KanseiMeshTransforms`
(`world`, `prevWorld`) and the camera's temporal uniform:

```wgsl
@group(2) @binding(1) var<uniform> mesh: KanseiMeshTransforms;
```

Vertex stages must not read group 3: shadow, reflection and velocity passes redraw each
material through its own `vertex_main` without it. Prefer the shared chunks
(`LIGHTS_WGSL`, `SHADOW_MAP_WGSL`, `GBUFFER_OUT_WGSL`) to declaring group 1 binding 2 or group 3
yourself.

#### Depth is [0, 1] (WebGPU's convention)

`Matrix4.perspective` (and so `Camera`) used gl-matrix's OpenGL `mat4.perspective`, whose clip
depth runs from -1 to 1; WebGPU clips depth to [0, 1], so the near half of that range was lost.
It now uses `perspectiveZO`, and `Matrix4.ortho` (new) uses `orthoZO`. Depth is cleared to 1.0,
so a depth of 1.0 means sky.

Code that uses the camera's matrices needs no change. Code that linearises a depth-buffer value,
or builds its own projection, does.

Before (0.0.11, OpenGL range):

```wgsl
fn linear_depth(d: f32, near: f32, far: f32) -> f32 {
    let z = d * 2.0 - 1.0;
    return 2.0 * near * far / (far + near - z * (far - near));
}
```

After (0.1.0, [0, 1] range):

```wgsl
fn linear_depth(d: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - d * (far - near));
}
```

And in TypeScript, build projections with the ZO variants:

```ts
// before
mat4.perspective(out, fovy, aspect, near, far);
mat4.ortho(out, left, right, bottom, top, near, far);
// after
mat4.perspectiveZO(out, fovy, aspect, near, far);   // or new Matrix4().perspective(...)
mat4.orthoZO(out, left, right, bottom, top, near, far); // or new Matrix4().ortho(...)
```

#### Geometry indices are 32-bit by default

`Geometry.indexFormat` now defaults to `'uint32'` (it was `'uint16'`), as the stock geometries
and the Rust engine build. A custom `Geometry` subclass that fills `indices` with a
`Uint16Array` must say so, or switch to 32-bit indices:

```ts
// before
this.indices = new Uint16Array(indexList);
// after: either
this.indices = new Uint32Array(indexList);
// or keep 16-bit indices
this.indices = new Uint16Array(indexList);
this.indexFormat = 'uint16';
// or build it in one go
const geometry = Geometry.fromArrays('MyMesh', vertices, new Uint32Array(indexList));
```

#### Smaller changes

- `FontLoader` no longer instantiates the Artery Font WebAssembly: `.arfont` files are parsed in
  TypeScript. `FontInfo` is now `ArFont` plus `sdfTexture`; its image `data` is a `Uint8Array`
  (was `number[]`), and metrics gain `line_height`, `underline_y` and `underline_thickness`.
- `Object3D` no longer has the protected `rotationMatrix`, `translationMatrix` and `scaleMatrix`
  fields; subclasses read `worldMatrix` (or `matrixNeedsUpdate` to force a rebuild).
- Material fragment shaders drawn into a `PostProcessingVolume` write the GBuffer's four
  targets (`GBUFFER_OUT_WGSL`), or set `mrtOutputCount` to the number they write. Forward
  rendering with `Renderer.render` is unchanged.
- The package now has an `exports` map, so deep imports such as `kansei/dist/math/Vector3.js` no
  longer resolve: import everything from `kansei`.
