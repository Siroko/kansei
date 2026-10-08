/**
 * The standard lit material: a physically based surface lit by everything the scene has, for
 * scenes drawn through a `PostProcessingVolume` (it writes the GBuffer's targets).
 * Rust: `materials/standard.rs`.
 */
import { Material } from "./Material";
import { ComputeBuffer } from "../buffers/ComputeBuffer";
import { BufferBase } from "../buffers/BufferBase";
import { assemble } from "./shaders/ShaderUtils";
import {
    CASCADED_SHADOWS_WGSL, GBUFFER_OUT_WGSL, GRADIENT_SKY_BODY_WGSL, LIGHTS_WGSL, MOTION_VECTORS_WGSL,
    SHADOW_MAP_WGSL, SPOT_LIGHTS_WGSL, STANDARD_LIT_BODY_WGSL,
} from "./shaders/SharedWGSL";

/**
 * How an instanced `Material.standardLit` places each instance: a vec4 per instance at vertex
 * location 3 (a `ComputeBuffer` with `shaderLocation: 3`, `format: 'float32x4'`, 16-byte stride).
 * - `'offsetScale'`: xyz the offset, w a uniform scale.
 * - `'offsetHeight'`: xyz the offset, w a scale of the height (y) only, for trunks and posts.
 */
export type StandardInstancing = 'offsetScale' | 'offsetHeight';

/**
 * What `Material.standardLit` draws. Radiances are in cd/m² and the lights' colours in the units
 * the scene's lights use, so the result goes through `ToneMapEffect` like the rest of an HDR
 * scene. Every field is optional; the defaults are Rust's `StandardLitOptions::default()`.
 */
export interface StandardLitOptions {
    /** Diffuse albedo (linear rgb), or the specular colour of a metal. Default 0.5 grey. */
    baseColor?: [number, number, number];
    /** GGX roughness, 0 (mirror) to 1. Default 0.6. */
    roughness?: number;
    /** 0 (dielectric) to 1 (metal). Default 0. */
    metallic?: number;
    /** Radiance the surface emits (cd/m²), also written to the GBuffer's emissive target. Default none. */
    emissive?: [number, number, number];
    /**
     * Sky radiance from straight up (cd/m²): a hemisphere of ambient light, reflected by the
     * albedo. Default none.
     */
    skyUp?: [number, number, number];
    /** Radiance from straight down (the ground's bounce), blended with `skyUp` by the normal. Default none. */
    skyDown?: [number, number, number];
    /** Instanced placement (unset: one mesh at its transform). */
    instancing?: StandardInstancing;
    /**
     * Also write screen-space motion (@location(4)) for temporal effects: sets
     * `MaterialOptions.outputsVelocity`. Instances must not move between frames.
     */
    outputsVelocity?: boolean;
}

/** A sky of three radiances for `Material.gradientSky` (cd/m², linear rgb). */
export interface GradientSkyOptions {
    zenith?: [number, number, number];
    horizon?: [number, number, number];
    /** Below the horizon. */
    ground?: [number, number, number];
    /**
     * How fast the horizon gives way to the zenith: the height's exponent (0.5 by default: a
     * wide horizon band).
     */
    curve?: number;
}

const INSTANCE_INPUT = '@location(3) instance : vec4f,';
const INSTANCE_PLACE: Record<StandardInstancing, string> = {
    offsetScale: 'local = local * v.instance.w + v.instance.xyz;',
    offsetHeight: 'local = vec3f(local.x, local.y * v.instance.w, local.z) + v.instance.xyz;',
};

/** The shader of `options`: the chunks it uses, then the material with its placeholders filled. */
export function standardLitShader(options: StandardLitOptions = {}): string {
    const velocity = options.outputsVelocity ?? false;
    const instancing = options.instancing;
    const body = assemble([STANDARD_LIT_BODY_WGSL], {
        KANSEI_INSTANCE_INPUT: instancing ? INSTANCE_INPUT : '',
        KANSEI_INSTANCE_PLACE: instancing ? INSTANCE_PLACE[instancing] : '',
        KANSEI_WORLD_BINDING: velocity
            ? '@group(2) @binding(1) var<uniform> mesh : KanseiMeshTransforms;'
            : '@group(2) @binding(1) var<uniform> world_matrix : mat4x4f;',
        KANSEI_WORLD: velocity ? 'mesh.world' : 'world_matrix',
        KANSEI_VELOCITY_VARYINGS: velocity ? '@location(2) currClip : vec4f,\n    @location(3) prevClip : vec4f,' : '',
        KANSEI_VELOCITY_OUTPUT: velocity ? '@location(4) velocity : vec2f,' : '',
        KANSEI_VELOCITY_VERTEX: velocity
            ? 'out.currClip = kansei_camera_temporal.viewProj * world;\n    out.prevClip = kansei_camera_temporal.prevViewProj * (mesh.prevWorld * vec4f(local, 1.0));'
            : '',
        KANSEI_VELOCITY_FRAGMENT: velocity ? 'out.velocity = kansei_motion_vector(in.currClip, in.prevClip);' : '',
    });
    return [LIGHTS_WGSL, SHADOW_MAP_WGSL, CASCADED_SHADOWS_WGSL, SPOT_LIGHTS_WGSL, velocity ? MOTION_VECTORS_WGSL : '', body].join('\n');
}

/** The material's group 0 uniform, as `KanseiStandardSurface` lays it out (80 bytes). */
export function standardLitUniform(options: StandardLitOptions = {}): Float32Array {
    const [r, g, b] = options.baseColor ?? [0.5, 0.5, 0.5];
    const [er, eg, eb] = options.emissive ?? [0, 0, 0];
    const [ur, ug, ub] = options.skyUp ?? [0, 0, 0];
    const [dr, dg, db] = options.skyDown ?? [0, 0, 0];
    return new Float32Array([
        r, g, b, 1,
        er, eg, eb, 0,
        ur, ug, ub, 0,
        dr, dg, db, 0,
        options.roughness ?? 0.6, options.metallic ?? 0, 0, 0,
    ]);
}

/** A uniform buffer holding `data`, for a stock material's group 0 binding 0. */
export function uniformBindable(data: Float32Array): ComputeBuffer {
    return new ComputeBuffer({
        type: BufferBase.BUFFER_TYPE_UNIFORM,
        usage: BufferBase.BUFFER_USAGE_UNIFORM | BufferBase.BUFFER_USAGE_COPY_DST,
        buffer: data,
    });
}

/** See `Material.standardLit`. */
export function standardLit(label: string, options: StandardLitOptions = {}): Material {
    return new Material(standardLitShader(options), {
        label,
        bindings: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, value: uniformBindable(standardLitUniform(options)) }],
        mrtOutputCount: 4,
        outputsVelocity: options.outputsVelocity ?? false,
    });
}

/** See `Material.emissive`. */
export function emissive(label: string, radiance: [number, number, number]): Material {
    return standardLit(label, { baseColor: [0, 0, 0], emissive: radiance });
}

/** See `Material.gradientSky`. */
export function gradientSky(label: string, options: GradientSkyOptions = {}): Material {
    const [zr, zg, zb] = options.zenith ?? [3000, 5000, 9000];
    const [hr, hg, hb] = options.horizon ?? [9000, 9500, 10500];
    const [gr, gg, gb] = options.ground ?? [1500, 1500, 1400];
    const uniform = new Float32Array([zr, zg, zb, options.curve ?? 0.5, hr, hg, hb, 0, gr, gg, gb, 0]);
    return new Material(`${GBUFFER_OUT_WGSL}\n${GRADIENT_SKY_BODY_WGSL}`, {
        label,
        bindings: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, value: uniformBindable(uniform) }],
        mrtOutputCount: 4,
        cullMode: 'none',
    });
}
