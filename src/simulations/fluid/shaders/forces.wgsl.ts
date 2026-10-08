/**
 * SPH forces (pressure, near pressure, viscosity, gravity, mouse), imported from the Rust
 * engine (Vite `?raw`) and assembled as its `with_params` does: `SimParams` first, the
 * neighbour-search workgroup size substituted.
 */
import forces from '../../../../rust/kansei-core/src/simulations/fluid/shaders/forces.wgsl?raw';
import { assemble } from '../../../materials/shaders/ShaderUtils';
import { simParamsStruct } from './sim-params.wgsl';

/** Workgroup size of the neighbour-search passes (`__NEIGHBOR_WG__`; dispatches use 64). */
const NEIGHBOR_WG = 64;

export const shaderCode: string = assemble([simParamsStruct, forces], { __NEIGHBOR_WG__: String(NEIGHBOR_WG) });
