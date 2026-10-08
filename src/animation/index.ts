/**
 * Skeletal animation (Rust `animation`): skeletons, poses and clips on quaternion joint
 * transforms, glTF import of skins and animations, and vertex-shader skinning.
 *
 * - `Skeleton` (joints parents first, rest pose), `Pose` (local transforms; `toModel` is the
 *   forward kinematics), `Clip` (every joint's transform at a fixed rate, sampled with nlerp).
 * - `SkinnedGltf` imports a glTF's skeleton, skinned meshes and animations.
 * - Skinning runs in the vertex shader by vertex pulling (`SKINNING_WGSL`): the geometry keeps
 *   the standard vertex layout, and a skinned material reads the bone palette and each vertex's
 *   joints and weights from its own storage buffers (group 0). Shadow and velocity passes redraw
 *   materials through their `vertex_main`, so they see the skinned mesh too; the palette also
 *   holds last frame's matrices, for motion vectors. Mark skinned renderables `dynamic` and
 *   update their `BonePalette` each frame.
 */
export { Transform, nlerp, quatAbs, quatLog, quatExp, quatToScaledAngleAxis, quatFromScaledAngleAxis, angularVelocity } from "./Transform";
export { Skeleton } from "./Skeleton";
export { Pose } from "./Pose";
export { Clip } from "./Clip";
export { SkinnedMesh, MAX_INFLUENCES, strongestInfluences } from "./SkinnedMesh";
export { SkinnedGltf } from "./SkinnedGltf";
export {
    SKINNING_WGSL, SKINNED_LIT_WGSL, SKINNED_LIT_TEXTURED_WGSL, PALETTE_BINDING, SKIN_BINDING, SKINNED_LIT_PARAMS_BYTES,
    BonePalette, skinBuffer, skinnedMaterial, skinnedLitMaterial, skinnedLitTexturedMaterial, packSkinnedLitParams,
} from "./Skinning";
export type { SkinnedLitParams, SkinTextures } from "./Skinning";
