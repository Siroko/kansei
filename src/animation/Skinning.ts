import skinning from '../../rust/kansei-core/src/shaders/skinning.wgsl?raw';
import motionVectors from '../../rust/kansei-core/src/shaders/motion_vectors.wgsl?raw';
import skinnedLit from '../../rust/kansei-core/src/shaders/skinned_lit.wgsl?raw';
import skinnedLitTextured from '../../rust/kansei-core/src/shaders/skinned_lit_textured.wgsl?raw';
import { BufferBase } from "../buffers/BufferBase";
import { ComputeBuffer } from "../buffers/ComputeBuffer";
import { Sampler } from "../buffers/Sampler";
import { Texture } from "../buffers/Texture";
import { Material, MaterialOptions } from "../materials/Material";
import { BindGroupDescriptor } from "../materials/BindableGroup";
import { SHADOW_MAP_WGSL } from "../materials/shaders/SharedWGSL";
import { assemble } from "../materials/shaders/ShaderUtils";
import { SkinnedMesh } from "./SkinnedMesh";
import { Transform } from "./Transform";

/**
 * WGSL for skinned materials: the bone palette (group 0 binding 1), the vertices' joints and
 * weights (group 0 binding 2) and `kansei_skin(vertex_index, position, normal)`, the skinned
 * position and normal in model space plus last frame's position. Imported from the Rust engine
 * (`animation::SKINNING_WGSL`); see the file's header for use.
 */
export const SKINNING_WGSL: string = skinning;

/**
 * The sun's shadow for the skinned lit shaders (`kansei_sun_shadow`, 1 lit and 0 shadowed).
 * Rust's is the renderer's cascaded shadow map (`shadows::CASCADED_SHADOWS_WGSL`, group 3
 * bindings 10-12; TS `Renderer.enableCascadedShadows`). Here it is still the directional
 * `ShadowMap` (`SHADOW_MAP_WGSL`, group 3 bindings 0-3), lit everywhere when there is none; with
 * the cascades on, that binding holds their widest cascade.
 */
const SUN_SHADOW_WGSL = /* wgsl */`${SHADOW_MAP_WGSL}
fn kansei_sun_shadow(worldPos: vec3f, N: vec3f, pixel: vec2f) -> f32 {
    return kansei_shadow_map(worldPos, N);
}
`;

/**
 * A skinned surface lit by a sun (shadowed) and a sky, writing motion vectors: the shader
 * `skinnedLitMaterial` builds on. Its uniform (group 0 binding 0) is `SkinnedLitParams`. The
 * Rust engine's `SKINNED_LIT_WGSL` with the sun's shadow from `SUN_SHADOW_WGSL`.
 */
export const SKINNED_LIT_WGSL: string = assemble([skinning, motionVectors, SUN_SHADOW_WGSL, skinnedLit]);

/**
 * `SKINNED_LIT_WGSL` with textures: colour (sRGB), tangent-space normal (+Y up, no stored
 * tangents needed) and occlusion/roughness/metallic, read at the mesh's uv, with a GGX specular.
 * Its uniform is `SkinnedLitParams` (`baseColor` tints the colour texture); see
 * `skinnedLitTexturedMaterial`.
 */
export const SKINNED_LIT_TEXTURED_WGSL: string = assemble([skinning, motionVectors, SUN_SHADOW_WGSL, skinnedLitTextured]);

/** Group 0 binding of the palette and of the per-vertex skin records. */
export const PALETTE_BINDING = 1;
export const SKIN_BINDING = 2;

/**
 * The joint matrices a skinned material reads, this frame's then last frame's (for motion
 * vectors), in one buffer. Rust: `animation::BonePalette`.
 *
 * Its `buffer` is shared by every material made with it: `set` (or `update`) marks it changed,
 * and the renderer uploads it with one `writeBuffer` when it next prepares those materials.
 */
export class BonePalette {
    /** `joints` current matrices, then `joints` previous ones (column-major, 16 floats each). */
    public readonly matrices: Float32Array;
    private primed: boolean = false;
    private scratch: Float32Array;
    private _buffer?: ComputeBuffer;

    /** A palette of `joints` identity matrices (no motion). */
    constructor(public readonly joints: number) {
        const count = 2 * Math.max(joints, 1);
        this.matrices = new Float32Array(count * 16);
        for (let i = 0; i < count; i++) {
            this.matrices[16 * i] = this.matrices[16 * i + 5] = this.matrices[16 * i + 10] = this.matrices[16 * i + 15] = 1;
        }
        this.scratch = new Float32Array(joints * 16);
    }

    /**
     * This frame's matrices (16 floats per joint); the previous frame's become last frame's.
     * The first call (and the first after `resetMotion`) sets both, so a new or teleported mesh
     * has no motion.
     */
    public set(current: Float32Array): void {
        const n = this.joints * 16;
        if (current.length !== n) throw new Error(`BonePalette.set: ${current.length / 16} matrices for ${this.joints} joints`);
        if (this.primed) {
            this.matrices.copyWithin(n, 0, n);
        } else {
            this.matrices.set(current, n);
            this.primed = true;
        }
        this.matrices.set(current, 0);
        if (this._buffer) this._buffer.needsUpdate = true;
    }

    /** `set` from a model-space pose of `mesh`'s skeleton. */
    public update(mesh: SkinnedMesh, model: Transform[]): void {
        this.set(mesh.palette(model, this.scratch));
    }

    /** Forget last frame's matrices: the next `set` has no motion (after a teleport or cut). */
    public resetMotion(): void {
        this.primed = false;
    }

    public current(): Float32Array {
        return this.matrices.subarray(0, this.joints * 16);
    }

    public previous(): Float32Array {
        return this.matrices.subarray(this.joints * 16, 2 * this.joints * 16);
    }

    /** The read-only storage buffer holding the palette, for a material's binding 1. */
    public get buffer(): ComputeBuffer {
        this._buffer ??= new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_READ_ONLY_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE | BufferBase.BUFFER_USAGE_COPY_DST,
            buffer: this.matrices,
        });
        return this._buffer;
    }

    /** Write the palette into its GPU buffer now, rather than when the renderer next prepares its materials. */
    public upload(device: GPUDevice): void {
        const buffer = this.buffer;
        if (buffer.initialized) buffer.update(device);
    }
}

/** A read-only storage buffer of `mesh`'s per-vertex joints and weights, for a material's binding 2. */
export function skinBuffer(mesh: SkinnedMesh): ComputeBuffer {
    return new ComputeBuffer({
        type: BufferBase.BUFFER_TYPE_READ_ONLY_STORAGE,
        usage: BufferBase.BUFFER_USAGE_STORAGE,
        buffer: mesh.skinWords(),
    });
}

/** A uniform buffer of `params`, or `params` itself when it is a buffer already. */
function uniform(params: Float32Array | Uint32Array | ComputeBuffer): ComputeBuffer {
    if (params instanceof ComputeBuffer) return params;
    return new ComputeBuffer({
        type: BufferBase.BUFFER_TYPE_UNIFORM,
        usage: BufferBase.BUFFER_USAGE_UNIFORM | BufferBase.BUFFER_USAGE_COPY_DST,
        buffer: params,
    });
}

/** Group 0 bindings 0-2 of a skinned material. */
function skinnedBindings(params: ComputeBuffer, mesh: SkinnedMesh, palette: BonePalette): BindGroupDescriptor[] {
    if (palette.joints !== mesh.skinJoints.length) {
        throw new Error(`skinned material: the palette has ${palette.joints} matrices for ${mesh.skinJoints.length} skin joints`);
    }
    return [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, value: params },
        { binding: PALETTE_BINDING, visibility: GPUShaderStage.VERTEX, value: palette.buffer },
        { binding: SKIN_BINDING, visibility: GPUShaderStage.VERTEX, value: skinBuffer(mesh) },
    ];
}

/**
 * A material drawing `mesh` skinned (Rust `skinned_material`): `shader` (which includes
 * `SKINNING_WGSL`), its uniform `params` at binding 0 (visible to both stages; pass a
 * `ComputeBuffer` to update it later), the palette's buffer at binding 1 and the skin records at
 * binding 2. Update the palette each frame (`BonePalette.update`) and mark the renderable
 * `dynamic`. `options.bindings` adds bindings after these.
 */
export function skinnedMaterial(
    label: string,
    shader: string,
    params: Float32Array | Uint32Array | ComputeBuffer,
    mesh: SkinnedMesh,
    palette: BonePalette,
    options: MaterialOptions = {},
): Material {
    return new Material(shader, {
        ...options,
        label,
        bindings: [...skinnedBindings(uniform(params), mesh, palette), ...(options.bindings ?? [])],
    });
}

/** `SKINNED_LIT_WGSL`'s uniform: albedo, sun and sky (physical units in the Rust engine). */
export interface SkinnedLitParams {
    /** Linear albedo (a unused); with textures, the colour texture's tint. */
    baseColor: [number, number, number, number?];
    /** The direction sunlight travels. */
    sunDirection: [number, number, number];
    /** Colour times illuminance (lux, under a `ToneMapEffect` exposure). */
    sun: [number, number, number];
    /** Zenith luminance (cd/m²). */
    sky: [number, number, number];
}

/** Bytes of `SkinnedLitParams` on the GPU (`KanseiSkinnedSurface`: four vec4f). */
export const SKINNED_LIT_PARAMS_BYTES = 64;

/** `params` laid out as `KanseiSkinnedSurface`, into `out`. */
export function packSkinnedLitParams(params: SkinnedLitParams, out: Float32Array = new Float32Array(SKINNED_LIT_PARAMS_BYTES / 4)): Float32Array {
    out.set([params.baseColor[0], params.baseColor[1], params.baseColor[2], params.baseColor[3] ?? 1], 0);
    out.set([...params.sunDirection, 0], 4);
    out.set([...params.sun, 0], 8);
    out.set([...params.sky, 0], 12);
    return out;
}

/** Options of the skinned lit materials: the colour pass writes target 0 (and velocity at @location(4)). */
const SKINNED_LIT_OPTIONS: MaterialOptions = { outputsVelocity: true, mrtOutputCount: 1 };

/**
 * A `SKINNED_LIT_WGSL` material for `mesh`, writing motion vectors (Rust `skinned_lit_material`).
 * Pass `params` as a `ComputeBuffer` (of `packSkinnedLitParams`) to change it later.
 */
export function skinnedLitMaterial(label: string, params: SkinnedLitParams | ComputeBuffer, mesh: SkinnedMesh, palette: BonePalette): Material {
    const buffer = params instanceof ComputeBuffer ? params : packSkinnedLitParams(params);
    return skinnedMaterial(label, SKINNED_LIT_WGSL, buffer, mesh, palette, SKINNED_LIT_OPTIONS);
}

/**
 * The textures of `skinnedLitTexturedMaterial`: colour (sRGB), normal map and
 * occlusion/roughness/metallic (both linear), e.g. `Texture.fromImage` or a glTF's
 * `loadTexture`.
 */
export interface SkinTextures {
    baseColor: Texture;
    normal: Texture;
    orm: Texture;
}

/**
 * A `SKINNED_LIT_TEXTURED_WGSL` material for `mesh`, writing motion vectors, its textures
 * sampled trilinearly with 8x anisotropy (Rust `skinned_lit_textured_material`).
 */
export function skinnedLitTexturedMaterial(
    label: string,
    params: SkinnedLitParams | ComputeBuffer,
    mesh: SkinnedMesh,
    palette: BonePalette,
    textures: SkinTextures,
): Material {
    const buffer = params instanceof ComputeBuffer ? params : packSkinnedLitParams(params);
    return skinnedMaterial(label, SKINNED_LIT_TEXTURED_WGSL, buffer, mesh, palette, {
        ...SKINNED_LIT_OPTIONS,
        bindings: [
            { binding: 3, visibility: GPUShaderStage.FRAGMENT, value: textures.baseColor },
            { binding: 4, visibility: GPUShaderStage.FRAGMENT, value: textures.normal },
            { binding: 5, visibility: GPUShaderStage.FRAGMENT, value: textures.orm },
            { binding: 6, visibility: GPUShaderStage.FRAGMENT, value: new Sampler('linear', 'linear', 'repeat', 8) },
        ],
    });
}
