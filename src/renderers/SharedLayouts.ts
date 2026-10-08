/**
 * Bind group slots shared by every render pipeline, the same as the Rust engine's
 * (`rust/kansei-core/src/renderers/shared_layouts.rs`), so material WGSL works in both:
 *
 * - group 0: the material's own bindings
 * - group 1: the camera: view @0, projection @1, scene lights @2, temporal data @3
 * - group 2: the mesh: normal matrix @0, world + previous world matrix @1 (dynamic offsets)
 * - group 3: shadows
 */
export const BindGroupSlot = {
    Material: 0,
    Camera: 1,
    Mesh: 2,
    Shadow: 3,
} as const;

/** Size of the scene light uniform at camera binding 2 (`KanseiLights` in `light_uniforms.wgsl`). */
export const LIGHT_UNIFORM_BYTES = 400;

/** Size of the camera temporal uniform at camera binding 3 (`KanseiCameraTemporal` in `motion_vectors.wgsl`). */
export const CAMERA_TEMPORAL_BYTES = 160;

/** Size of the window bound at mesh binding 1: world then previous world (`KanseiMeshTransforms`). */
export const MESH_TRANSFORMS_BYTES = 128;

/** Group 1 layout entries: view, projection, scene lights, temporal data. */
export function cameraBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
    const vertexFragment = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    return [
        { binding: 0, visibility: vertexFragment, buffer: { type: 'uniform' } },
        { binding: 1, visibility: vertexFragment, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 3, visibility: vertexFragment, buffer: { type: 'uniform' } },
    ];
}

/** Group 2 layout entries: normal matrix and world + previous world, at per-object dynamic offsets. */
export function meshBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
    return [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } },
    ];
}

/** Bytes between two objects' slots in the mesh buffers: room for the 128-byte window, aligned for dynamic offsets. */
export function meshSlotStride(device: GPUDevice): number {
    const alignment = device.limits.minUniformBufferOffsetAlignment ?? 256;
    return Math.ceil(MESH_TRANSFORMS_BYTES / alignment) * alignment;
}
