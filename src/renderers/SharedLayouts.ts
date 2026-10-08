/**
 * Bind group slots shared by every render pipeline, the same as the Rust engine's
 * (`rust/kansei-core/src/renderers/shared_layouts.rs`), so material WGSL works in both:
 *
 * - group 0: the material's own bindings
 * - group 1: the camera: view @0, projection @1, scene lights @2, temporal data @3
 * - group 2: the mesh: normal matrix @0, world + previous world matrix @1 (dynamic offsets)
 * - group 3: shadows and lights, fragment-only (`shadowBindGroupLayoutEntries`)
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

/** Size of the shadow uniform at group 3 binding 2 (`KanseiShadowMap` in `shadow_map.wgsl`). */
export const SHADOW_UNIFORM_BYTES = 96;

/** Size of the spot light buffer at group 3 binding 6: a 16-byte header and 128 lights of 144 bytes (`KanseiSpotLights`). */
export const SPOT_LIGHTS_BUFFER_BYTES = 16 + 128 * 144;

/** Size of the light-cluster parameters at group 3 binding 8 (`KanseiClusterParams`). */
export const CLUSTER_PARAMS_BYTES = 160;

/** Size of the cascades uniform at group 3 binding 11 (`KanseiCascades`). */
export const CASCADES_BYTES = 384;

/**
 * Group 3 layout entries, the Rust engine's 13 (`shared_layouts.rs`), all fragment-only: vertex
 * stages must not read group 3, since depth pipelines (`Material.getDepthPipeline`) leave it out
 * and shadow passes render into the textures it samples. It only grows by additive bindings; the
 * renderer binds 1x1 dummies (or zeroed buffers) for whatever is not enabled.
 *
 * - 0-2: the directional shadow map: depth texture, comparison sampler, `KanseiShadowMap` uniform
 * - 3-4: the point-light cube shadow (r32float distances, 6 layers a light), non-filtering sampler
 * - 5-7: the spot shadow atlas (depth 2d-array), spot lights (storage), comparison sampler
 * - 8-9: light-cluster parameters (uniform) and per-cluster light lists (storage)
 * - 10-12: the cascaded shadow map (depth 2d-array), cascades (uniform), comparison sampler
 */
export function shadowBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
    const fragment = GPUShaderStage.FRAGMENT;
    return [
        { binding: 0, visibility: fragment, texture: { sampleType: 'depth' } },
        { binding: 1, visibility: fragment, sampler: { type: 'comparison' } },
        { binding: 2, visibility: fragment, buffer: { type: 'uniform' } },
        { binding: 3, visibility: fragment, texture: { sampleType: 'unfilterable-float', viewDimension: '2d-array' } },
        { binding: 4, visibility: fragment, sampler: { type: 'non-filtering' } },
        { binding: 5, visibility: fragment, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
        { binding: 6, visibility: fragment, buffer: { type: 'read-only-storage' } },
        { binding: 7, visibility: fragment, sampler: { type: 'comparison' } },
        { binding: 8, visibility: fragment, buffer: { type: 'uniform' } },
        { binding: 9, visibility: fragment, buffer: { type: 'read-only-storage' } },
        { binding: 10, visibility: fragment, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
        { binding: 11, visibility: fragment, buffer: { type: 'uniform' } },
        { binding: 12, visibility: fragment, sampler: { type: 'comparison' } },
    ];
}

/** Bytes between two objects' slots in the mesh buffers: room for the 128-byte window, aligned for dynamic offsets. */
export function meshSlotStride(device: GPUDevice): number {
    const alignment = device.limits.minUniformBufferOffsetAlignment ?? 256;
    return Math.ceil(MESH_TRANSFORMS_BYTES / alignment) * alignment;
}
