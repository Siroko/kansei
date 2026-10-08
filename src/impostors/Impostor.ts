import { vec3 } from 'gl-matrix';
import { Geometry } from '../buffers/Geometry';
import { Texture } from '../buffers/Texture';
import impostorWgsl from '../../rust/kansei-core/src/shaders/impostor.wgsl?raw';

/**
 * Octahedral impostors: a far LOD of two triangles per instance for instanced renderables. A
 * port of the Rust engine's `impostors` module, sharing its WGSL.
 *
 * `Renderer.bakeImpostor` renders renderables of the scene (the parts of one object: bark and
 * foliage, say) with their own materials from N x N directions of an octahedral grid (or a
 * hemi-octahedral one, for things only seen from above), orthographically, into two atlases of
 * N x N frames: the albedo with the coverage, and the object-space normal with the depth. At run
 * time a material written with `IMPOSTOR_WGSL` draws `billboardGeometry` per instance, facing the
 * camera, and reads the three frames nearest the view direction with one step of parallax: the
 * albedo and normal to shade as the mesh's material does, and the surface point to write depth
 * from. The shadow map draws it facing the light.
 *
 * Its instances are culled like any instanced renderable's (`InstanceCulling`, sharing the
 * meshes' instances, with the far LOD band).
 *
 * ```ts
 * // bake from the tree's LOD0 renderables, with one instance record placing it at the origin
 * const impostor = renderer.bakeImpostor([bark, foliage], {
 *     instance: new Float32Array([0, 0, 0, 1, 0, 0, 0, 0]),
 * });
 * const material = new Material(`${IMPOSTOR_WGSL}\n${shader}`, { bindings: [
 *     { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, value: uniform(impostor.params()) },
 *     { binding: 1, visibility: GPUShaderStage.FRAGMENT, value: impostor.albedoTexture() },
 *     { binding: 2, visibility: GPUShaderStage.FRAGMENT, value: impostor.normalDepthTexture() },
 *     { binding: 3, visibility: GPUShaderStage.FRAGMENT, value: new Sampler('linear', 'linear', 'clamp-to-edge') },
 * ] });
 * const quad = new InstancedGeometry(billboardGeometry('Trees/Impostor'), count, [instances]);
 * ```
 */

/**
 * WGSL for drawing an `Impostor` in a material: `KanseiImpostor` (its `params`),
 * `kansei_impostor_corner` (the billboard, vertex stage), `kansei_impostor_sample` (albedo,
 * coverage, normal and surface point) and the octahedral mapping. See the file's header
 * (`rust/kansei-core/src/shaders/impostor.wgsl`). Rust: `impostors::IMPOSTOR_WGSL`.
 */
export const IMPOSTOR_WGSL: string = impostorWgsl;

/**
 * Which directions an impostor's frames are baked from: `octahedral`, the whole sphere (for
 * things seen from below too: in a mirror, from a slope); `hemi-octahedral`, the upper
 * hemisphere only, twice the frames' density there, for things only ever seen from above the
 * horizon.
 */
export type ImpostorLayout = 'octahedral' | 'hemi-octahedral';

type Vec3 = [number, number, number];

/** How to bake an impostor (Rust `ImpostorOptions`). */
export interface ImpostorOptions {
    /** Frames per side of the atlases (N x N views). Default 12. */
    frames?: number;
    /**
     * Texels per side of a frame, a power of two (the atlases are `frames * frameSize` square,
     * with a mip chain down to a texel per frame; two RGBA8 atlases of 12 x 128 take ~25 MB).
     * Default 128.
     */
    frameSize?: number;
    /** Default `octahedral`. */
    layout?: ImpostorLayout;
    /** Render texels per atlas texel side, averaged (antialiasing, and softer coverage). Default 2. */
    supersample?: number;
    /**
     * The object's bounding box (min and max corners), object space; by default the one round
     * the parts' vertex positions (right when `instance` leaves them in place). The frames are
     * baked round the sphere through the box's corners (or round the vertices), and the billboard
     * covers the box's outline.
     */
    bounds?: [Vec3, Vec3];
    /**
     * One instance record, in the layout of the parts' instance buffer, that places the object at
     * the origin unrotated and unscaled (ignored for parts without instances).
     */
    instance?: Float32Array | Uint32Array;
}

/** Bytes of `KanseiImpostor` in `IMPOSTOR_WGSL` (`Impostor.params`). */
export const IMPOSTOR_PARAMS_BYTES = 48;

/** A baked impostor: its atlases and what `KanseiImpostor` needs to read them. */
export class Impostor {
    /** The atlases' format: albedo and coverage; normal (`n * 0.5 + 0.5`) and depth. */
    static readonly FORMAT: GPUTextureFormat = 'rgba8unorm';

    constructor(
        public readonly frames: number,
        public readonly frameSize: number,
        public readonly layout: ImpostorLayout,
        /** The bounds' centre (object space). */
        public readonly center: Vec3,
        /** The radius of the sphere the frames were baked round. */
        public readonly radius: number,
        /** The box's half size (object space). */
        public readonly extent: Vec3,
        /** Albedo (rgb) and coverage (a). */
        public readonly albedoAtlas: GPUTexture,
        /** Object-space normal (rgb, `n * 0.5 + 0.5`) and depth (a). */
        public readonly normalDepthAtlas: GPUTexture,
    ) {}

    /** `KanseiImpostor` in `IMPOSTOR_WGSL` (`IMPOSTOR_PARAMS_BYTES`): bind it as a uniform. */
    params(): Float32Array {
        const f = new Float32Array(IMPOSTOR_PARAMS_BYTES / 4);
        const u = new Uint32Array(f.buffer);
        f.set(this.center, 0);
        f[3] = this.radius;
        f.set(this.extent, 4);
        u[7] = this.frames;
        u[8] = this.layout === 'hemi-octahedral' ? 1 : 0;
        return f;
    }

    /** Albedo (rgb) and coverage (a), to bind as a texture_2d<f32>. */
    albedoTexture(): Texture {
        return Texture.fromView('Impostor/Albedo', this.albedoAtlas);
    }

    /**
     * Object-space normal (rgb, `n * 0.5 + 0.5`) and depth (a: 0 at the frame's near side of the
     * bounding sphere, 1 at its far side), to bind as a texture_2d<f32>.
     */
    normalDepthTexture(): Texture {
        return Texture.fromView('Impostor/NormalDepth', this.normalDepthAtlas);
    }

    /** Frees the atlases. */
    destroy(): void {
        this.albedoAtlas.destroy();
        this.normalDepthAtlas.destroy();
    }
}

/**
 * The billboard an impostor material draws per instance: four corners at `position.xy` in
 * {-1, 1} (for `kansei_impostor_corner`), two triangles facing +z.
 */
export function billboardGeometry(label: string): Geometry {
    const vertices: number[] = [];
    for (const [x, y] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
        vertices.push(x, y, 0, 1, 0, 0, 1, x * 0.5 + 0.5, 0.5 - y * 0.5);
    }
    return Geometry.fromArrays(label, new Float32Array(vertices), new Uint32Array([0, 1, 2, 0, 2, 3]));
}

// The octahedral mapping and the frames' axes, as in IMPOSTOR_WGSL.

const sign = (v: number) => (v >= 0 ? 1 : -1);

/** The grid position ([-1, 1] squared) of a direction, y up (`kansei_impostor_encode`). */
export function impostorEncode(dir: Vec3, hemi: boolean): [number, number] {
    const l = Math.abs(dir[0]) + Math.abs(dir[1]) + Math.abs(dir[2]);
    const [x, y, z] = [dir[0] / l, dir[1] / l, dir[2] / l];
    if (hemi) {
        const d = Math.max(Math.abs(x) + Math.abs(z), y > 0 ? 1 : 1e-12);
        const [hx, hz] = [x / d, z / d];
        return [hx + hz, hx - hz];
    }
    if (y >= 0) return [x, z];
    return [(1 - Math.abs(z)) * sign(x), (1 - Math.abs(x)) * sign(z)];
}

/** The unit direction at a grid position (`kansei_impostor_decode`). */
export function impostorDecode(g: [number, number], hemi: boolean): Vec3 {
    const out = vec3.create();
    if (hemi) {
        const x = (g[0] + g[1]) * 0.5;
        const z = (g[0] - g[1]) * 0.5;
        vec3.set(out, x, 1 - Math.abs(x) - Math.abs(z), z);
    } else {
        const y = 1 - Math.abs(g[0]) - Math.abs(g[1]);
        if (y < 0) vec3.set(out, (1 - Math.abs(g[1])) * sign(g[0]), y, (1 - Math.abs(g[0])) * sign(g[1]));
        else vec3.set(out, g[0], y, g[1]);
    }
    vec3.normalize(out, out);
    return [out[0], out[1], out[2]];
}

/**
 * The up reference of a view from `dir`: y, or z looking straight down or up
 * (`kansei_impostor_basis` takes right = up x dir, up = dir x right).
 */
export function impostorUpReference(dir: Vec3): Vec3 {
    return Math.abs(dir[1]) > 0.999 ? [0, 0, 1] : [0, 1, 0];
}

/** The direction frame (column, row) of an N x N grid is baked from. */
export function impostorFrameDirection(frames: number, layout: ImpostorLayout, column: number, row: number): Vec3 {
    const g: [number, number] = [(column + 0.5) / frames * 2 - 1, (row + 0.5) / frames * 2 - 1];
    return impostorDecode(g, layout === 'hemi-octahedral');
}
