# Simulation Renderer Contract

This document standardizes the GPU contract between simulation extractors (for example, marching cubes) and `SimulationRenderable`.

## Versioning

- `SurfaceContractVersion::V1` is the initial contract.
- Producers and consumers must check version compatibility before use.

## Producer / Consumer Roles

- **Producer**: `MarchingCubesSimulation` (or another extractor) converts simulation data into a GPU mesh.
- **Consumer**: `SimulationRenderable` draws that mesh without CPU readback.

## Surface Extraction Source Contract

`SurfaceExtractionSourceContract` fields:

- `version`
- `field_dims` (voxel grid dimensions)
- `world_bounds_min`
- `world_bounds_max`
- `iso_value`

This contract is simulation-agnostic. Any simulation that can provide a scalar field and bounds can be used.

## Surface Mesh GPU Contract

`SurfaceMeshGpuContract` fields:

- `version`
- `vertex_buffer`
- `index_buffer`
- `indirect_args_buffer` (`DrawIndexedIndirect` layout: 5 x `u32`)
- `vertex_stride`

### Vertex layout (V1)

- Position: `float32x4`
- Normal: `float32x4`
- Rust type: `MarchingCubesVertex`

## Runtime Guarantees

- GPU-only flow: no readback (`map_async`) required for rendering.
- Renderer consumes mesh via `draw_indexed_indirect`.
- Producer owns allocation and update of mesh buffers.

## Parameterization Requirements

`MarchingCubesSimulationParams` includes:

- `grid_sizing` (`FromSource`, `Fixed([x,y,z])`, `MaxAxis(u32)`)
- `iso_level`
- `max_triangles`
- `normal_sample_step`
- `surface_smoothing`

Implementations must clamp unsafe values to GPU-safe ranges.

## Extension Guidance

Future extractors (marching tetrahedra, dual contouring, etc.) should:

- either emit `SurfaceMeshGpuContract` V1 directly, or
- introduce `V2` with a documented compatibility adapter path.
