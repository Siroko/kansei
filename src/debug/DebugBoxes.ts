/**
 * Debug drawing: `DebugBoxes`, a set of flat-coloured boxes the app places by matrix each frame
 * (one instanced draw), for trajectories, bones, contact points and other markers. A port of the
 * Rust engine's `debug.rs`.
 */
import { mat4, quat, vec3 } from "gl-matrix";
import { ComputeBuffer } from "../buffers/ComputeBuffer";
import { BufferBase } from "../buffers/BufferBase";
import { BoxGeometry } from "../geometries/BoxGeometry";
import { InstancedGeometry } from "../geometries/InstancedGeometry";
import { Material } from "../materials/Material";
import { Renderable } from "../objects/Renderable";
import { Object3D } from "../objects/Object3D";
import type { Renderer } from "../renderers/Renderer";
import { Vector4 } from "../math/Vector4";

/**
 * Unit boxes placed by a per-instance matrix, one flat colour lit a little from above. The Rust
 * shader's vertex stage; its fragment stage also fills the GBuffer's other three targets
 * (emissive, normal, albedo), because a TS material's pipeline writes every target a pass has.
 * Drawn straight to the canvas, the extra outputs are dropped.
 */
export const DEBUG_BOXES_WGSL = /* wgsl */ `
struct Marker { color: vec4<f32> };
@group(0) @binding(0) var<uniform> marker: Marker;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
struct VIn {
    @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>,
    @location(3) m0: vec4<f32>, @location(4) m1: vec4<f32>, @location(5) m2: vec4<f32>, @location(6) m3: vec4<f32>,
};
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) normal: vec3<f32> };
@vertex
fn vertex_main(v: VIn) -> VOut {
    let m = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    var out: VOut;
    out.clip = projection_matrix * view_matrix * m * v.position;
    out.normal = normalize((m * vec4<f32>(v.normal, 0.0)).xyz);
    return out;
}
struct FOut {
    @location(0) color: vec4<f32>, @location(1) emissive: vec4<f32>,
    @location(2) normal: vec4<f32>, @location(3) albedo: vec4<f32>,
};
@fragment
fn fragment_main(in: VOut) -> FOut {
    let shade = 0.65 + 0.35 * max(in.normal.y, 0.0);
    return FOut(
        vec4<f32>(marker.color.rgb * shade, 1.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
        vec4<f32>(in.normal * 0.5 + 0.5, 1.0),
        vec4<f32>(marker.color.rgb, 1.0),
    );
}
`;

/**
 * `count` unit boxes (centred, 1 across) in one colour, each placed by its matrix in `matrices`
 * (16 floats each, column-major): set them (`setMatrix`, `segment`), then `upload` once a frame.
 * A zero matrix hides its box. They cast no shadow and draw after the opaque scene; `xRay` draws
 * them over everything (no depth test).
 */
export class DebugBoxes {
    /** One column-major matrix per box (unit box to world), 16 floats each. */
    public readonly matrices: Float32Array;
    /** The boxes' renderable, added to the scene (or the parent) given to the constructor. */
    public readonly renderable: Renderable;
    private readonly instances: ComputeBuffer;

    /**
     * Add `count` boxes of `color` (linear rgb, in the scene's units of radiance) to `parent`
     * (a scene), all hidden (zero matrices).
     */
    constructor(parent: Object3D, count: number, color: [number, number, number], xRay: boolean = false) {
        this.matrices = new Float32Array(count * 16);
        this.instances = new ComputeBuffer({
            usage: BufferBase.BUFFER_USAGE_VERTEX | BufferBase.BUFFER_USAGE_COPY_DST,
            buffer: this.matrices,
            stride: 64,
            attributes: [0, 1, 2, 3].map((k) => ({ shaderLocation: 3 + k, offset: 16 * k, format: 'float32x4' as GPUVertexFormat })),
        });
        const material = new Material(DEBUG_BOXES_WGSL, {
            bindings: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, value: new Vector4(color[0], color[1], color[2], 1) }],
            ...(xRay ? { transparent: true, depthWriteEnabled: false, depthCompare: 'always' as GPUCompareFunction } : {}),
        });
        this.renderable = new Renderable(new InstancedGeometry(new BoxGeometry(1, 1, 1), count, [this.instances]), material);
        this.renderable.dynamic = true;
        this.renderable.castShadow = false;
        this.renderable.receiveShadow = false;
        this.renderable.renderOrder = 10;
        parent.add(this.renderable);
    }

    /** Whether the boxes are drawn. */
    public get visible(): boolean {
        return this.renderable.visible;
    }
    public set visible(visible: boolean) {
        this.renderable.visible = visible;
    }

    /** Place box `index` by `matrix` (column-major, a gl-matrix `mat4` or 16 numbers). */
    public setMatrix(index: number, matrix: ArrayLike<number>): void {
        this.matrices.set(matrix, index * 16);
    }

    /** Write `matrices` to the GPU (one write); before the first frame they go up with the buffer. */
    public upload(renderer: Renderer): void {
        if (this.instances.initialized) this.instances.update(renderer.gpuDevice);
    }

    /** Hide every box (zero matrices; upload to show it). */
    public clear(): void {
        this.matrices.fill(0);
    }
}

/**
 * A box's matrix from `a` to `b`, `thickness` across (zero, hidden, when they meet): a bone, a
 * ray, an edge. Written into `out` when given.
 */
export function segment(a: ArrayLike<number>, b: ArrayLike<number>, thickness: number, out: mat4 = mat4.create()): mat4 {
    const d = vec3.fromValues(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    const length = vec3.length(d);
    if (length < 1e-5) return mat4.set(out, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const rotation = quat.rotationTo(quat.create(), [0, 1, 0], vec3.scale(d, d, 1 / length));
    const centre = vec3.fromValues((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5);
    return mat4.fromRotationTranslationScale(out, rotation, centre, [thickness, length, thickness]);
}
