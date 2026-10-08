import { CLUSTER_MESH_WGSL } from './vertexStage';
import { clusterMeshBindGroupLayoutEntries } from '../renderers/SharedLayouts';
import { gpuPass } from '../profiling/Profiler';
import type { ClusterCulling, ClusterGpu, Cut } from './ClusterLod';

/**
 * The cluster LOD debug view (TS only): `Renderer.setClusterDebug` draws the camera's cluster
 * cuts (`Renderable.clusters`) coloured by what made them, through a pipeline generated around
 * each material's `vertex_main` as the cluster path's is (`Material.getClusterDebugPipeline`), and
 * counts the cut's clusters and triangles per level on the GPU, read back a few frames late
 * (`levels`). Off (the default), nothing of it runs or is made.
 *
 * The debug pipeline draws a cut without its index buffer (`drawIndirect` over the same draw
 * arguments: the indexed draw's first five words read as a plain draw's), reading each index from
 * the buffer itself, so each corner knows its place in its triangle for the edges.
 */

/**
 * What the colours show. `cluster`: each cluster its own. `level`: the build round (level 0 the
 * full mesh; the higher, the coarser). `group`: the clusters simplified together into their
 * parent (siblings share a colour; the coarsest level, which has no parent, by cluster). `error`:
 * the cluster's error as the cut measures it for the camera, over the budget
 * (`Renderer.setClusterErrorThreshold`): blue at 0 (the full mesh), red at the budget, magenta past
 * it. `instance`: each instance its own. `triangle`: each triangle its own.
 */
export type ClusterDebugMode = 'cluster' | 'level' | 'group' | 'error' | 'instance' | 'triangle';

export const CLUSTER_DEBUG_MODES: readonly ClusterDebugMode[] = ['cluster', 'level', 'group', 'error', 'instance', 'triangle'];

export interface ClusterDebugOptions {
    mode?: ClusterDebugMode;
    /** Draw each triangle's edges (off by default). */
    edges?: boolean;
    /** The colours' radiance in the colour target (1.5 by default), for the scene's exposure. */
    brightness?: number;
}

/** A level's share of the camera's cut: its clusters and their triangles. */
export interface ClusterLevelCount {
    clusters: number;
    triangles: number;
}

/** Levels the per-level counts keep apart (deeper ones count as the last). */
export const DEBUG_LEVELS = 32;
/** Bytes of `KanseiClusterDebug`. */
export const CLUSTER_DEBUG_BYTES = 16;
/** The debug draw's entry points. */
export const CLUSTER_DEBUG_VERTEX_ENTRY = 'kansei_cluster_debug_vertex';
export const CLUSTER_DEBUG_FRAGMENT_ENTRY = 'kansei_cluster_debug_fragment';
const STATS_WORKGROUPS = 256;

/**
 * The debug draw's group 2: the cluster draw's (`clusterMeshBindGroupLayoutEntries`), then the
 * cut's indices (5), its cull parameters (6), the frame's cluster views (7) and the debug view's
 * settings (8).
 */
export function clusterDebugBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
    return [
        ...clusterMeshBindGroupLayoutEntries(),
        { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 6, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 7, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 8, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ];
}

/**
 * The debug draw's stages, after the generated cluster vertex function (`clusterVertexFunction`),
 * whose output's clip position is member `position` (null: the output is the position), into `targets` colour targets (four: the GBuffer's colour, emissive, normal and
 * albedo). The cut's placement and errors are cluster_cull.wgsl's, read from its parameters.
 */
export function clusterDebugWgsl(position: string | null, targets: number): string {
    const outputs = ['color', 'emissive', 'normal', 'albedo'];
    const members = Array.from({ length: Math.max(targets, 1) }, (_, i) => `    @location(${i}) ${outputs[i] ?? `extra${i}`}: vec4<f32>,`).join('\n');
    const writes = Array.from({ length: Math.max(targets, 1) }, (_, i) => [
        '    out.color = vec4<f32>(color, 1.0);',
        '    out.emissive = vec4<f32>(0.0);',
        '    out.normal = vec4<f32>(n * 0.5 + 0.5, 1.0);',
        '    out.albedo = vec4<f32>(base, 1.0);',
    ][i] ?? `    out.extra${i} = vec4<f32>(0.0);`).join('\n');
    return /* wgsl */`
struct KanseiDebugCull {
    world: mat4x4<f32>,
    kind: u32,
    position_word: u32,
    scale_word: u32,
    yaw_word: u32,
    rotation_word: u32,
    stride_words: u32,
    first_record: u32,
    instance_count: u32,
    count_word: u32,
    capacity: u32,
    vertex_count: u32,
    flags: u32,
    yaw_scale: f32,
    stretch: f32,
    view: u32,
    triangle_capacity: u32,
}

struct KanseiDebugView {
    planes: array<vec4<f32>, 6>,
    eye: vec3<f32>,
    pixels_per_radian: f32,
    near: f32,
    threshold: f32,
    orthographic: u32,
    pad: f32,
}

struct KanseiClusterDebug {
    mode: u32,
    edges: u32,
    brightness: f32,
    pad: f32,
}

@group(2) @binding(5) var<storage, read> kansei_debug_indices: array<u32>;
@group(2) @binding(6) var<uniform> kansei_debug_cull: KanseiDebugCull;
@group(2) @binding(7) var<storage, read> kansei_debug_views: array<KanseiDebugView>;
@group(2) @binding(8) var<uniform> kansei_debug: KanseiClusterDebug;

struct KanseiClusterDebugOut {
    @builtin(position) clip: vec4<f32>,
    // the cluster, its level, its group's hash, the instance record
    @location(0) @interpolate(flat) info: vec4<u32>,
    // the cluster's projected error over the budget, the triangle
    @location(1) @interpolate(flat) ratio: f32,
    @location(2) @interpolate(flat) triangle: u32,
    @location(3) normal: vec3<f32>,
    @location(4) corner: vec3<f32>,
}

fn kansei_debug_record_f32(record: u32, word: u32) -> f32 {
    return bitcast<f32>(kansei_cluster_records[record * kansei_debug_cull.stride_words + word]);
}

fn kansei_debug_record_vec4(record: u32, word: u32) -> vec4<f32> {
    return vec4<f32>(kansei_debug_record_f32(record, word), kansei_debug_record_f32(record, word + 1u), kansei_debug_record_f32(record, word + 2u), kansei_debug_record_f32(record, word + 3u));
}

// cluster_cull.wgsl's \`placement\`: where the record puts the mesh, in the renderable's space
fn kansei_debug_placement(record: u32) -> mat4x4<f32> {
    let p = kansei_debug_cull;
    if (p.kind == 2u) {
        let w = p.position_word;
        return mat4x4<f32>(kansei_debug_record_vec4(record, w), kansei_debug_record_vec4(record, w + 4u), kansei_debug_record_vec4(record, w + 8u), kansei_debug_record_vec4(record, w + 12u));
    }
    var m = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
    if (p.kind != 1u) {
        return mat4x4<f32>(vec4<f32>(m[0], 0.0), vec4<f32>(m[1], 0.0), vec4<f32>(m[2], 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    }
    if (p.rotation_word != 0xffffffffu) {
        let q = kansei_debug_record_vec4(record, p.rotation_word);
        let x2 = q.x + q.x;
        let y2 = q.y + q.y;
        let z2 = q.z + q.z;
        m = mat3x3<f32>(
            vec3<f32>(1.0 - (q.y * y2 + q.z * z2), q.x * y2 + q.w * z2, q.x * z2 - q.w * y2),
            vec3<f32>(q.x * y2 - q.w * z2, 1.0 - (q.x * x2 + q.z * z2), q.y * z2 + q.w * x2),
            vec3<f32>(q.x * z2 + q.w * y2, q.y * z2 - q.w * x2, 1.0 - (q.x * x2 + q.y * y2)),
        );
    }
    if (p.yaw_word != 0xffffffffu) {
        let a = kansei_debug_record_f32(record, p.yaw_word) * p.yaw_scale;
        m = mat3x3<f32>(vec3<f32>(cos(a), 0.0, -sin(a)), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(sin(a), 0.0, cos(a))) * m;
    }
    if (p.scale_word != 0xffffffffu) {
        m = m * kansei_debug_record_f32(record, p.scale_word);
    }
    let t = vec3<f32>(kansei_debug_record_f32(record, p.position_word), kansei_debug_record_f32(record, p.position_word + 1u), kansei_debug_record_f32(record, p.position_word + 2u));
    return mat4x4<f32>(vec4<f32>(m[0], 0.0), vec4<f32>(m[1], 0.0), vec4<f32>(m[2], 0.0), vec4<f32>(t, 1.0));
}

// cluster_cull.wgsl's \`projected\`: a mesh sphere's error, placed by \`model\`, in pixels
fn kansei_debug_projected(error: f32, sphere: vec4<f32>, model: mat4x4<f32>, scale: f32, view: KanseiDebugView) -> f32 {
    let center = (model * vec4<f32>(sphere.xyz, 1.0)).xyz;
    let radius = (sphere.w + (1.0 - 1.0 / kansei_debug_cull.stretch) * length(sphere.xyz)) * scale;
    if (view.orthographic != 0u) {
        return error * scale * view.pixels_per_radian;
    }
    return error * scale / max(distance(view.eye, center) - radius, view.near) * view.pixels_per_radian;
}

fn kansei_debug_hash(x: u32) -> u32 {
    // PCG
    let state = x * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

@vertex
fn ${CLUSTER_DEBUG_VERTEX_ENTRY}(@builtin(vertex_index) kansei_corner: u32, @builtin(instance_index) kansei_instance: u32) -> KanseiClusterDebugOut {
    let index = kansei_debug_indices[kansei_corner];
    let shaded = kansei_cluster_vertex(index, kansei_instance);
    var out: KanseiClusterDebugOut;
    out.clip = shaded${position ? `.${position}` : ''};
    let draw = kansei_cluster_draws[index >> 8u];
    let c = draw.y;
    let record = draw.x;
    let level = kansei_cluster_word(c, 3u);
    // siblings share their parent's sphere (the coarsest level has none: by cluster)
    var group = kansei_debug_hash(c);
    if (kansei_cluster_f32(c, 24u) >= 0.0) {
        group = kansei_debug_hash(kansei_cluster_word(c, 20u) ^ kansei_debug_hash(kansei_cluster_word(c, 21u) ^ kansei_debug_hash(kansei_cluster_word(c, 22u) ^ kansei_debug_hash(kansei_cluster_word(c, 23u) + level))));
    }
    out.info = vec4<u32>(c, level, group, record);
    // the cut's measure of the cluster (cluster_cull.wgsl), over the budget
    let model = kansei_debug_cull.world * kansei_debug_placement(record);
    let m = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
    let g = transpose(m) * m;
    let scale = sqrt(max(dot(abs(g[0]), vec3<f32>(1.0)), max(dot(abs(g[1]), vec3<f32>(1.0)), dot(abs(g[2]), vec3<f32>(1.0))))) * kansei_debug_cull.stretch;
    let view = kansei_debug_views[kansei_debug_cull.view];
    let error = kansei_debug_projected(kansei_cluster_f32(c, 15u), kansei_cluster_vec4(c, 16u), model, scale, view);
    out.ratio = select(select(0.0, 2.0, error > 0.0), error / view.threshold, view.threshold > 0.0);
    out.triangle = kansei_corner / 3u;
    // the mesh normal through the placement's cofactors (any scale), facing out
    let vertex = kansei_cluster_local_vertex(c, index & 0xffu);
    let cofactors = mat3x3<f32>(cross(m[1], m[2]), cross(m[2], m[0]), cross(m[0], m[1]));
    out.normal = cofactors * kansei_cluster_attribute(vertex, 4u, 3u).xyz * sign(determinant(m));
    let k = kansei_corner % 3u;
    out.corner = vec3<f32>(f32(k == 0u), f32(k == 1u), f32(k == 2u));
    return out;
}

fn kansei_debug_hue(h: u32) -> vec3<f32> {
    let hue = f32(h & 0xffffu) / 65536.0;
    let value = 0.65 + 0.35 * f32((h >> 16u) & 0xffu) / 255.0;
    let rgb = clamp(abs(fract(hue + vec3<f32>(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, vec3<f32>(0.0), vec3<f32>(1.0));
    return mix(vec3<f32>(1.0), rgb, 0.75) * value;
}

// a level's colour: twelve far apart, then round again
fn kansei_debug_level(level: u32) -> vec3<f32> {
    var palette = array<vec3<f32>, 12>(
        vec3<f32>(0.12, 0.47, 0.71), vec3<f32>(1.0, 0.5, 0.05), vec3<f32>(0.17, 0.63, 0.17), vec3<f32>(0.84, 0.15, 0.16),
        vec3<f32>(0.58, 0.4, 0.74), vec3<f32>(0.55, 0.34, 0.29), vec3<f32>(0.89, 0.47, 0.76), vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.74, 0.74, 0.13), vec3<f32>(0.09, 0.75, 0.81), vec3<f32>(0.68, 0.78, 0.91), vec3<f32>(1.0, 0.73, 0.47),
    );
    return palette[level % 12u];
}

// 0 (blue) through the budget (red); past it magenta
fn kansei_debug_ramp(t: f32) -> vec3<f32> {
    if (t > 1.0) {
        return vec3<f32>(1.0, 0.0, 1.0);
    }
    let x = clamp(t, 0.0, 1.0);
    return clamp(vec3<f32>(1.5 * x - 0.1, 1.0 - 2.4 * abs(x - 0.5), 1.0 - 1.6 * x), vec3<f32>(0.03), vec3<f32>(1.0));
}

struct KanseiClusterDebugTargets {
${members}
}

@fragment
fn ${CLUSTER_DEBUG_FRAGMENT_ENTRY}(in: KanseiClusterDebugOut, @builtin(front_facing) front: bool) -> KanseiClusterDebugTargets {
    // (derivatives before any branch)
    let width = fwidth(in.corner);
    var base: vec3<f32>;
    switch kansei_debug.mode {
        case 0u: { base = kansei_debug_hue(kansei_debug_hash(in.info.x)); }
        case 1u: { base = kansei_debug_level(in.info.y); }
        case 2u: { base = kansei_debug_hue(in.info.z); }
        case 3u: { base = kansei_debug_ramp(in.ratio); }
        case 4u: { base = kansei_debug_hue(kansei_debug_hash(in.info.w + 0x9e3779b9u)); }
        default: { base = kansei_debug_hue(kansei_debug_hash(in.triangle ^ (in.info.x << 12u))); }
    }
    if (kansei_debug.edges != 0u) {
        let inside = smoothstep(vec3<f32>(0.0), width * 1.2, in.corner);
        base = mix(vec3<f32>(0.02), base, min(inside.x, min(inside.y, inside.z)));
    }
    var n = normalize(in.normal);
    if (!front) {
        n = -n;
    }
    let sun = normalize(vec3<f32>(0.4, 0.8, 0.3));
    let light = 0.35 + 0.15 * n.y + 0.6 * max(dot(n, sun), 0.0);
    let color = base * light * kansei_debug.brightness;
    var out: KanseiClusterDebugTargets;
${writes}
    return out;
}
`;
}

/** The per-level counts' compute pass: the camera cut's listed clusters binned by level. */
const STATS_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> kansei_cluster_mesh: array<u32>;
@group(0) @binding(1) var<storage, read> draws: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> draw: array<u32, 12>;
// clusters per level, then triangles per level
@group(0) @binding(3) var<storage, read_write> bins: array<atomic<u32>, ${2 * DEBUG_LEVELS}>;
${CLUSTER_MESH_WGSL}
var<workgroup> kansei_local_bins: array<atomic<u32>, ${2 * DEBUG_LEVELS}>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32, @builtin(num_workgroups) groups: vec3<u32>) {
    // the entries claimed, as far as the list holds them (those that didn't fit are NONE)
    let count = min(draw[6], arrayLength(&draws));
    for (var i = id.x; i < count; i += groups.x * 64u) {
        let entry = draws[i];
        if (entry.y == 0xffffffffu) {
            continue;
        }
        let level = min(kansei_cluster_word(entry.y, 3u), ${DEBUG_LEVELS - 1}u);
        atomicAdd(&kansei_local_bins[level], 1u);
        atomicAdd(&kansei_local_bins[${DEBUG_LEVELS}u + level], kansei_cluster_word(entry.y, 2u));
    }
    workgroupBarrier();
    let n = atomicLoad(&kansei_local_bins[lane]);
    if (n > 0u) {
        atomicAdd(&bins[lane], n);
    }
}
`;

/**
 * The cluster LOD debug view's settings and its per-level counts (`Renderer.setClusterDebug`).
 * Changing the settings rewrites a uniform: no pipeline or bundle is remade.
 */
export class ClusterDebug {
    private _mode: ClusterDebugMode;
    private _edges: boolean;
    private _brightness: number;
    private dirty = true;
    private device: GPUDevice | null = null;
    private _uniform: GPUBuffer | null = null;
    private layout: GPUBindGroupLayout | null = null;
    private draws = new WeakMap<Cut, { key: (GPUBuffer | null)[]; group: GPUBindGroup }>();
    private stats: { pipeline: GPUComputePipeline; bins: GPUBuffer; staging: GPUBuffer; groups: WeakMap<Cut, { key: GPUBuffer[]; group: GPUBindGroup }> } | null = null;
    private pending: 'copied' | 'mapping' | 'mapped' | 'failed' | null = null;
    private _levels: ClusterLevelCount[] | null = null;

    constructor(options: ClusterDebugOptions = {}) {
        this._mode = options.mode ?? 'cluster';
        this._edges = options.edges ?? false;
        this._brightness = options.brightness ?? 1.5;
    }

    get mode(): ClusterDebugMode { return this._mode; }
    set mode(mode: ClusterDebugMode) {
        if (!CLUSTER_DEBUG_MODES.includes(mode)) throw new Error(`ClusterDebug: no mode ${mode}`);
        this._mode = mode;
        this.dirty = true;
    }

    get edges(): boolean { return this._edges; }
    set edges(on: boolean) {
        this._edges = on;
        this.dirty = true;
    }

    get brightness(): number { return this._brightness; }
    set brightness(value: number) {
        this._brightness = value;
        this.dirty = true;
    }

    /**
     * The camera's last cut read back, per level (level 0 the full mesh): its clusters and their
     * triangles, summed over the clustered renderables; null until a reading arrives.
     */
    get levels(): ClusterLevelCount[] | null {
        return this._levels;
    }

    /** The settings' uniform, written when they changed. */
    uniform(device: GPUDevice): GPUBuffer {
        if (this.device !== device) this.destroy();
        this.device = device;
        this._uniform ??= device.createBuffer({ label: 'ClusterDebug/Settings', size: CLUSTER_DEBUG_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        if (this.dirty) {
            const bytes = new ArrayBuffer(CLUSTER_DEBUG_BYTES);
            new Uint32Array(bytes, 0, 2).set([CLUSTER_DEBUG_MODES.indexOf(this._mode), this._edges ? 1 : 0]);
            new Float32Array(bytes, 8, 1)[0] = this._brightness;
            device.queue.writeBuffer(this._uniform, 0, bytes);
            this.dirty = false;
        }
        return this._uniform;
    }

    /**
     * Draw `cut` (of `gpu`, bound and drawn this frame) with `pipeline`
     * (`Material.getClusterDebugPipeline`): its debug group 2 at matrix `offset`, then the cull's
     * indirect draw read as a non-indexed one (the world matrices bound `worldBytes` at a time).
     * False when the cut has no draw group yet.
     */
    encodeDraw(encoder: GPURenderPassEncoder | GPURenderBundleEncoder, culling: ClusterCulling, gpu: ClusterGpu, cut: Cut, pipeline: GPURenderPipeline, offset: number, worldBytes: number): boolean {
        const device = this.device!;
        const draw = cut.draw;
        if (!draw) return false;
        // the cluster draw's buffers (normal, world, draws, records), then the indices
        const key = [...draw.key, cut.indices];
        let bound = this.draws.get(cut);
        if (!bound || bound.key.length !== key.length || !bound.key.every((b, i) => b === key[i])) {
            this.layout ??= device.createBindGroupLayout({ label: 'ClusterDebug BindGroupLayout', entries: clusterDebugBindGroupLayoutEntries() });
            const [normal, world, , records] = draw.key as [GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer | null];
            const entry = (binding: number, buffer: GPUBuffer, size?: number): GPUBindGroupEntry => ({ binding, resource: { buffer, size } });
            bound = {
                key,
                group: device.createBindGroup({
                    label: 'ClusterDebug/Draw',
                    layout: this.layout,
                    entries: [
                        entry(0, normal, 64), entry(1, world, worldBytes), entry(2, gpu.mesh), entry(3, cut.draws),
                        entry(4, records ?? gpu.mesh), entry(5, cut.indices), entry(6, cut.params), entry(7, culling.viewsBuffer), entry(8, this._uniform!),
                    ],
                }),
            };
            this.draws.set(cut, bound);
        }
        encoder.setPipeline(pipeline);
        encoder.setBindGroup(2, bound.group, [offset, offset]);
        encoder.drawIndirect(cut.args, 0);
        return true;
    }

    /**
     * Count the camera's `cuts` (bound this frame, after the cull) per level, and copy the counts
     * for the readback unless one is in flight (call `submitted` once `encoder` is submitted).
     */
    encodeStats(encoder: GPUCommandEncoder, cuts: [ClusterGpu, Cut][]): void {
        const device = this.device;
        if (!device || cuts.length === 0) return;
        const bytes = 2 * DEBUG_LEVELS * 4;
        const stats = this.stats ??= {
            pipeline: device.createComputePipeline({
                label: 'ClusterDebug/Levels',
                layout: 'auto',
                compute: { module: device.createShaderModule({ label: 'ClusterDebug/Levels', code: STATS_WGSL }), entryPoint: 'main' },
            }),
            bins: device.createBuffer({ label: 'ClusterDebug/Levels', size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }),
            staging: device.createBuffer({ label: 'ClusterDebug/LevelsReadback', size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
            groups: new WeakMap(),
        };
        if (this.pending === 'mapped') {
            const words = new Uint32Array(stats.staging.getMappedRange().slice(0));
            stats.staging.unmap();
            let deepest = DEBUG_LEVELS;
            while (deepest > 0 && words[deepest - 1] === 0) deepest--;
            this._levels = Array.from({ length: deepest }, (_, l) => ({ clusters: words[l], triangles: words[DEBUG_LEVELS + l] }));
            this.pending = null;
        } else if (this.pending === 'failed') {
            this.pending = null;
        }
        encoder.clearBuffer(stats.bins);
        const pass = encoder.beginComputePass({ label: 'ClusterDebug/Levels', timestampWrites: gpuPass('ClusterDebug/Levels') });
        pass.setPipeline(stats.pipeline);
        for (const [gpu, cut] of cuts) {
            const key = [gpu.mesh, cut.draws, cut.args];
            let bound = stats.groups.get(cut);
            if (!bound || !bound.key.every((b, i) => b === key[i])) {
                bound = {
                    key,
                    group: device.createBindGroup({
                        label: 'ClusterDebug/Levels',
                        layout: stats.pipeline.getBindGroupLayout(0),
                        entries: key.concat(stats.bins).map((buffer, binding) => ({ binding, resource: { buffer } })),
                    }),
                };
                stats.groups.set(cut, bound);
            }
            pass.setBindGroup(0, bound.group);
            pass.dispatchWorkgroups(STATS_WORKGROUPS);
        }
        pass.end();
        if (this.pending === null) {
            encoder.copyBufferToBuffer(stats.bins, 0, stats.staging, 0, bytes);
            this.pending = 'copied';
        }
    }

    /** The encoder `encodeStats` recorded into was submitted: map its copy. */
    submitted(): void {
        if (this.pending !== 'copied' || !this.stats) return;
        this.pending = 'mapping';
        this.stats.staging.mapAsync(GPUMapMode.READ).then(
            () => { this.pending = 'mapped'; },
            () => { this.pending = 'failed'; },
        );
    }

    /** Frees the GPU state (made again when drawn). */
    destroy(): void {
        this._uniform?.destroy();
        this._uniform = null;
        this.stats?.bins.destroy();
        this.stats?.staging.destroy();
        this.stats = null;
        this.pending = null;
        this.layout = null;
        this.draws = new WeakMap();
        this.dirty = true;
    }
}
