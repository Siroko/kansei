/**
 * The fluid's `SimParams` uniform and its grid helpers, imported from the Rust engine (Vite
 * `?raw`) so both run the same layout: `PARAMS` in `FluidSimulationParams.ts` packs it.
 * Rust: `simulations::fluid::SIM_PARAMS_WGSL`.
 */
import simParams from '../../../../rust/kansei-core/src/simulations/fluid/shaders/sim-params.wgsl?raw';

export const simParamsStruct: string = simParams;
