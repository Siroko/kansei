/* tslint:disable */
/* eslint-disable */

export function get_fps(): number;

export function get_frame_time(): number;

export function init(): void;

export function set_absorption(v: number): void;

export function set_bounds(min_x: number, min_y: number, min_z: number, max_x: number, max_y: number, max_z: number): void;

export function set_damping(v: number): void;

export function set_density_resolution(v: number): void;

export function set_density_scale(v: number): void;

export function set_density_target(v: number): void;

export function set_density_threshold(v: number): void;

export function set_dof_focus_distance(v: number): void;

export function set_dof_focus_range(v: number): void;

export function set_dof_max_blur(v: number): void;

export function set_gravity_center(x: number, y: number, z: number): void;

export function set_gravity_y(v: number): void;

export function set_kernel_scale(v: number): void;

export function set_light_color(r: number, g: number, b: number): void;

export function set_light_direction(x: number, y: number, z: number): void;

export function set_light_intensity(v: number): void;

export function set_max_render_fps(v: number): void;

export function set_mc_iso_level(v: number): void;

export function set_mc_resolution(v: number): void;

export function set_mouse_force(v: number): void;

export function set_mouse_radius(v: number): void;

export function set_near_pressure(v: number): void;

export function set_particle_size(v: number): void;

export function set_pressure(v: number): void;

export function set_radial_gravity(enabled: boolean): void;

export function set_render_mode(v: number): void;

export function set_show_particles(v: boolean): void;

export function set_sim_dt_step(v: number): void;

export function set_sim_time_scale(v: number): void;

export function set_step_count(v: number): void;

export function set_stripe_params(r1: number, g1: number, b1: number, r2: number, g2: number, b2: number, thick_a: number, thick_b: number): void;

export function set_substeps(v: number): void;

export function set_transmission_params(ior: number, chromatic_aberration: number, tint_strength: number, fresnel_power: number, roughness: number, thickness: number, r: number, g: number, b: number): void;

export function set_use_batched_sim(v: boolean): void;

export function set_viscosity(v: number): void;

export function start(canvas_id: string): Promise<void>;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly set_bounds: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly set_damping: (a: number) => void;
    readonly set_density_target: (a: number) => void;
    readonly set_dof_focus_distance: (a: number) => void;
    readonly set_dof_focus_range: (a: number) => void;
    readonly set_dof_max_blur: (a: number) => void;
    readonly set_gravity_center: (a: number, b: number, c: number) => void;
    readonly set_gravity_y: (a: number) => void;
    readonly set_kernel_scale: (a: number) => void;
    readonly set_light_color: (a: number, b: number, c: number) => void;
    readonly set_light_direction: (a: number, b: number, c: number) => void;
    readonly set_mc_iso_level: (a: number) => void;
    readonly set_mc_resolution: (a: number) => void;
    readonly set_mouse_force: (a: number) => void;
    readonly set_mouse_radius: (a: number) => void;
    readonly set_near_pressure: (a: number) => void;
    readonly set_pressure: (a: number) => void;
    readonly set_radial_gravity: (a: number) => void;
    readonly set_render_mode: (a: number) => void;
    readonly set_show_particles: (a: number) => void;
    readonly set_stripe_params: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly set_substeps: (a: number) => void;
    readonly set_transmission_params: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
    readonly set_use_batched_sim: (a: number) => void;
    readonly set_viscosity: (a: number) => void;
    readonly start: (a: number, b: number) => any;
    readonly set_light_intensity: (a: number) => void;
    readonly set_density_threshold: (a: number) => void;
    readonly get_fps: () => number;
    readonly get_frame_time: () => number;
    readonly init: () => void;
    readonly set_absorption: (a: number) => void;
    readonly set_density_scale: (a: number) => void;
    readonly set_particle_size: (a: number) => void;
    readonly set_step_count: (a: number) => void;
    readonly set_max_render_fps: (a: number) => void;
    readonly set_density_resolution: (a: number) => void;
    readonly set_sim_dt_step: (a: number) => void;
    readonly set_sim_time_scale: (a: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__hd699ab66e63e7540: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h13a80c2f90f1693f: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h26b5814b833ec8e0: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h26b5814b833ec8e0_2: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h26b5814b833ec8e0_3: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h26b5814b833ec8e0_4: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h2e15691cdd54a2b9: (a: number, b: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
