// The voxelizer's fragment stage for materials without a `voxel_fragment_entry`: the
// renderable's constant GiSurface (voxel_write.wgsl). It reads only builtins, a subset of any
// material's vertex outputs.
@fragment
fn voxel_fragment(@builtin(position) fragPos: vec4f, @builtin(front_facing) front: bool) {
    kansei_voxel_write(fragPos, front, kansei_voxel_draw.albedo, kansei_voxel_draw.emission);
}
