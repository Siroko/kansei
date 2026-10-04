//! Spatial grids for particle simulations on the GPU.
//!
//! [`NeighbourGrid`] counting-sorts points into uniform cells every step, so a pass can find the
//! points near one by visiting the cells around it: the fluid's neighbour search
//! ([`FluidSimulation::grid`](crate::simulations::fluid::FluidSimulation::grid)) and any other
//! particle system (flocking, repulsion, ray walks through particles) share it.

mod neighbour_grid;

pub use neighbour_grid::{GpuNeighbourGrid, GridLayout, NeighbourGrid, NeighbourGridOptions, NEIGHBOUR_GRID_WGSL};
