// Cluster LOD: a renderable's cut for one view (clusters/gpu.rs). `prepare` resets the draw and
// sizes `cull`'s dispatch from the visible instances. `cull` runs a workgroup per visible
// instance over the clusters of the levels its cut can reach (clusters::LevelBounds), testing
// the cut rule, the frustum and the backface cone, and appends the drawn ones to the draw list.
// `finish` draws at most the list's capacity. Included with cluster_mesh.wgsl.

struct ClusterCull {
    world: mat4x4<f32>,
    // 0: none (the renderable's transform), 1: placement, 2: a matrix
    kind: u32,
    position_word: u32,
    scale_word: u32,
    yaw_word: u32,
    rotation_word: u32,
    stride_words: u32,
    first_record: u32,
    // the visible instances without a count word
    instance_count: u32,
    count_word: u32,
    capacity: u32,
    vertex_count: u32,
    flags: u32,
    // the record's yaw times this (radians); how much further the material may stretch an
    // instance (spheres and errors grow by it)
    yaw_scale: f32,
    stretch: f32,
    pad0: u32,
    pad1: u32,
}

struct ClusterView {
    planes: array<vec4<f32>, 6>,
    eye: vec3<f32>,
    pixels_per_radian: f32,
    near: f32,
    threshold: f32,
    pad: vec2<f32>,
}

// DrawIndirect's four words, then the visible instances, the clusters claimed and the triangles
// drawn
struct ClusterDraw {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
    visible: u32,
    claimed: atomic<u32>,
    triangles: atomic<u32>,
    pad: u32,
}

const NONE: u32 = 0xffffffffu;
const KIND_PLACEMENT: u32 = 1u;
const KIND_MATRIX: u32 = 2u;
const FLAG_CONE: u32 = 1u;
const MAX_GROUPS: u32 = 65535u;
// clusters::WINDOW_SLACK
const WINDOW_SLACK: f32 = 1e-3;

@group(0) @binding(0) var<uniform> params: ClusterCull;
@group(0) @binding(1) var<storage, read> kansei_cluster_mesh: array<u32>;
@group(0) @binding(2) var<storage, read> records: array<u32>;
@group(0) @binding(3) var<storage, read_write> draws: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> draw: ClusterDraw;
@group(1) @binding(0) var<uniform> view: ClusterView;

// prepare and finish
@group(0) @binding(10) var<uniform> prepare_params: ClusterCull;
@group(0) @binding(11) var<storage, read> instance_args: array<u32>;
@group(0) @binding(12) var<storage, read_write> prepare_draw: array<u32, 8>;
@group(0) @binding(13) var<storage, read_write> dispatch: array<u32, 4>;

@compute @workgroup_size(1)
fn prepare() {
    var visible = prepare_params.instance_count;
    if (prepare_params.count_word != NONE) {
        visible = instance_args[prepare_params.count_word];
    }
    prepare_draw = array<u32, 8>(prepare_params.vertex_count, 0u, 0u, 0u, visible, 0u, 0u, 0u);
    let x = min(visible, MAX_GROUPS);
    dispatch = array<u32, 4>(x, select(0u, (visible + x - 1u) / x, x > 0u), 1u, 0u);
}

@compute @workgroup_size(1)
fn finish() {
    prepare_draw[1] = min(prepare_draw[5], prepare_params.capacity);
}

fn record_f32(record: u32, word: u32) -> f32 {
    return bitcast<f32>(records[record * params.stride_words + word]);
}

fn record_vec3(record: u32, word: u32) -> vec3<f32> {
    return vec3<f32>(record_f32(record, word), record_f32(record, word + 1u), record_f32(record, word + 2u));
}

fn record_vec4(record: u32, word: u32) -> vec4<f32> {
    return vec4<f32>(record_vec3(record, word), record_f32(record, word + 3u));
}

// a unit quaternion (x y z w) as a rotation (glam's Mat3::from_quat)
fn rotation(q: vec4<f32>) -> mat3x3<f32> {
    let x2 = q.x + q.x;
    let y2 = q.y + q.y;
    let z2 = q.z + q.z;
    let xx = q.x * x2;
    let xy = q.x * y2;
    let xz = q.x * z2;
    let yy = q.y * y2;
    let yz = q.y * z2;
    let zz = q.z * z2;
    let wx = q.w * x2;
    let wy = q.w * y2;
    let wz = q.w * z2;
    return mat3x3<f32>(
        vec3<f32>(1.0 - (yy + zz), xy + wz, xz - wy),
        vec3<f32>(xy - wz, 1.0 - (xx + zz), yz + wx),
        vec3<f32>(xz + wy, yz - wx, 1.0 - (xx + yy)),
    );
}

// where the record puts the mesh, in the renderable's space
fn placement(record: u32) -> mat4x4<f32> {
    if (params.kind == KIND_MATRIX) {
        let w = params.position_word;
        return mat4x4<f32>(record_vec4(record, w), record_vec4(record, w + 4u), record_vec4(record, w + 8u), record_vec4(record, w + 12u));
    }
    var m = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
    if (params.kind != KIND_PLACEMENT) {
        return mat4x4<f32>(vec4<f32>(m[0], 0.0), vec4<f32>(m[1], 0.0), vec4<f32>(m[2], 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    }
    if (params.rotation_word != NONE) {
        m = rotation(record_vec4(record, params.rotation_word));
    }
    if (params.yaw_word != NONE) {
        let a = record_f32(record, params.yaw_word) * params.yaw_scale;
        m = mat3x3<f32>(vec3<f32>(cos(a), 0.0, -sin(a)), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(sin(a), 0.0, cos(a))) * m;
    }
    if (params.scale_word != NONE) {
        m = m * record_f32(record, params.scale_word);
    }
    return mat4x4<f32>(vec4<f32>(m[0], 0.0), vec4<f32>(m[1], 0.0), vec4<f32>(m[2], 0.0), vec4<f32>(record_vec3(record, params.position_word), 1.0));
}

// clusters::projected_error_at: `error` (world units) seen `distance` away, in pixels
fn projected_at(error: f32, distance: f32) -> f32 {
    return error / max(distance, view.near) * view.pixels_per_radian;
}

// the error of a mesh sphere (xyz, radius) placed by `model`, `scale` a bound on its largest scale
fn projected(error: f32, sphere: vec4<f32>, model: mat4x4<f32>, scale: f32) -> f32 {
    let center = (model * vec4<f32>(sphere.xyz, 1.0)).xyz;
    return projected_at(error * scale, distance(view.eye, center) - sphere.w * scale);
}

// clusters::LevelBounds::may_draw, from the eye's distance to the mesh's origin
fn level_may_draw(level: u32, origin_distance: f32, scale: f32) -> bool {
    let min_error = bitcast<f32>(kansei_level_word(level, 2u));
    let max_parent_error = bitcast<f32>(kansei_level_word(level, 3u));
    let near_reach = bitcast<f32>(kansei_level_word(level, 4u));
    let far_reach = bitcast<f32>(kansei_level_word(level, 5u));
    let fine_enough = projected_at(min_error * scale, origin_distance + near_reach * scale) <= view.threshold * (1.0 + WINDOW_SLACK);
    let parent_over = max_parent_error < 0.0 || projected_at(max_parent_error * scale, origin_distance - far_reach * scale) > view.threshold * (1.0 - WINDOW_SLACK);
    return fine_enough && parent_over;
}

fn cluster_drawn(c: u32, model: mat4x4<f32>, scale: f32, eye_mesh: vec3<f32>, cone: bool) -> bool {
    // the cut rule: fine enough, and its parent not
    if (projected(kansei_cluster_f32(c, 15u), kansei_cluster_vec4(c, 16u), model, scale) > view.threshold) {
        return false;
    }
    let parent_error = kansei_cluster_f32(c, 24u);
    if (parent_error >= 0.0 && projected(parent_error, kansei_cluster_vec4(c, 20u), model, scale) <= view.threshold) {
        return false;
    }
    // the frustum
    let bounds = kansei_cluster_vec4(c, 4u);
    let center = (model * vec4<f32>(bounds.xyz, 1.0)).xyz;
    for (var p = 0u; p < 6u; p++) {
        if (dot(view.planes[p].xyz, center) + view.planes[p].w < -bounds.w * scale) {
            return false;
        }
    }
    // every triangle facing away (clusters::Cluster::backfacing, in the mesh's space)
    if (cone) {
        let apex = kansei_cluster_vec4(c, 8u);
        if (dot(normalize(apex.xyz - eye_mesh), kansei_cluster_vec4(c, 12u).xyz) >= apex.w) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn cull(@builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let slot = group.y * groups.x + group.x;
    if (slot >= draw.visible) {
        return;
    }
    let record = params.first_record + slot;
    let model = params.world * placement(record);
    let m = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
    // a bound on the largest scale in any direction: the square root of the largest absolute
    // row sum of mᵀm (the columns' lengths when they are orthogonal; more under shear, where the
    // columns' lengths fall short and the level window would skip levels the cut needs)
    let g = transpose(m) * m;
    let scale = sqrt(max(dot(abs(g[0]), vec3<f32>(1.0)), max(dot(abs(g[1]), vec3<f32>(1.0)), dot(abs(g[2]), vec3<f32>(1.0))))) * params.stretch;
    let det = determinant(m);
    // a mirroring transform turns the winding over: no cone test
    let cone = (params.flags & FLAG_CONE) != 0u && det > 0.0;
    var eye_mesh = vec3<f32>(0.0);
    if (cone) {
        let inverse = transpose(mat3x3<f32>(cross(m[1], m[2]), cross(m[2], m[0]), cross(m[0], m[1]))) * (1.0 / det);
        eye_mesh = inverse * (view.eye - model[3].xyz);
    }
    let origin_distance = distance(view.eye, model[3].xyz);
    let levels = kansei_cluster_mesh[6];
    for (var level = 0u; level < levels; level++) {
        if (!level_may_draw(level, origin_distance, scale)) {
            continue;
        }
        let first = kansei_level_word(level, 0u);
        let count = kansei_level_word(level, 1u);
        for (var i = lane; i < count; i += 64u) {
            let c = first + i;
            if (cluster_drawn(c, model, scale, eye_mesh, cone)) {
                let at = atomicAdd(&draw.claimed, 1u);
                if (at < params.capacity) {
                    draws[at] = vec2<u32>(record, c);
                    atomicAdd(&draw.triangles, kansei_cluster_word(c, 2u));
                }
            }
        }
    }
}
