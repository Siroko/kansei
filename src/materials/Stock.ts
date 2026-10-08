/**
 * Stock materials: ready-made shaders for scenes that don't need their own.
 * Rust: `materials/stock.rs`.
 */
import { Material, MaterialOptions } from "./Material";
import { uniformBindable } from "./StandardLit";
import { BASIC_INSTANCED_WGSL, BASIC_LIT_WGSL } from "./shaders/SharedWGSL";

/**
 * WGSL for instances placed by a base point, a uniform scale and a yaw about +y (as
 * `mat4.fromYRotation`, then scale, then the base): `kansei_place(local, inst, yaw)` takes a
 * point from the mesh to the world, with `inst` the base (xyz) and scale (w); `kansei_turn(v,
 * yaw)` turns a direction (a normal); `kansei_unplace(world, inst, yaw)` undoes `kansei_place`
 * (an impostor reads the camera in the tree's frame). Rust: `materials::INSTANCE_PLACEMENT_WGSL`.
 */
export const INSTANCE_PLACEMENT_WGSL = /* wgsl */`
fn kansei_turn(v: vec3<f32>, yaw: f32) -> vec3<f32> {
    let c = cos(yaw);
    let s = sin(yaw);
    return vec3<f32>(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn kansei_place(local: vec3<f32>, inst: vec4<f32>, yaw: f32) -> vec3<f32> {
    return kansei_turn(local, yaw) * inst.w + inst.xyz;
}

fn kansei_unplace(world: vec3<f32>, inst: vec4<f32>, yaw: f32) -> vec3<f32> {
    return kansei_turn((world - inst.xyz) / inst.w, -yaw);
}
`;

/** See `Material.basicLit`. */
export function basicLit(label: string, color: [number, number, number, number], specular: [number, number, number, number], options: MaterialOptions = {}): Material {
    return new Material(BASIC_LIT_WGSL, {
        mrtOutputCount: 1,
        ...options,
        label,
        bindings: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, value: uniformBindable(new Float32Array([...color, ...specular])) }],
    });
}

/** See `Material.basicInstanced`. */
export function basicInstanced(label: string, color: [number, number, number, number], options: MaterialOptions = {}): Material {
    return new Material(BASIC_INSTANCED_WGSL, {
        mrtOutputCount: 1,
        ...options,
        label,
        bindings: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, value: uniformBindable(new Float32Array(color)) }],
    });
}
