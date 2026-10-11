//! The mouse's push on the water: a [`FluidSubstepPass`] that makes the particles in a column round
//! a point follow a velocity (the mouse's motion over the water), softly, for either solver.

use bytemuck::{Pod, Zeroable};

use kansei_core::simulations::fluid::{FluidSimulation, FluidSubstepPass, SIM_PARAMS_WGSL};

const PUSH_WGSL: &str = include_str!("push.wgsl");

/// The WGSL `Push` (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct PushGpu {
    pub center: [f32; 2],
    pub bottom: f32,
    pub top: f32,
    pub velocity: [f32; 3],
    pub radius: f32,
    pub strength: f32,
    pub on: u32,
    pub _pad: [f32; 2],
}

pub struct FluidPush {
    queue: wgpu::Queue,
    uniform: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    current: PushGpu,
}

impl FluidPush {
    pub fn new(sim: &FluidSimulation) -> Self {
        let (device, queue) = sim.gpu();
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FluidPush/Uniform"),
            size: std::mem::size_of::<PushGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (pipeline, bind_group) = sim.substep_pipeline("FluidPush", &format!("{SIM_PARAMS_WGSL}\n{PUSH_WGSL}"), &[(&uniform, false)]);
        let push = Self { queue: queue.clone(), uniform, pipeline, bind_group, current: PushGpu::default() };
        push.queue.write_buffer(&push.uniform, 0, bytemuck::bytes_of(&push.current));
        push
    }

    /// The push for the next steps (simulation units), or none.
    pub fn set(&mut self, push: Option<PushGpu>) {
        let next = push.map_or(PushGpu::default(), |p| PushGpu { on: 1, ..p });
        if next != self.current {
            self.current = next;
            self.queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&next));
        }
    }

    pub fn active(&self) -> bool {
        self.current.on != 0
    }
}

impl FluidSubstepPass for FluidPush {
    fn dispatch(&self, pass: &mut wgpu::ComputePass<'_>, particle_count: u32) {
        if self.current.on == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(particle_count.div_ceil(64), 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_push_shader_validates_and_its_uniform_matches() {
        let source = format!("{SIM_PARAMS_WGSL}\n{PUSH_WGSL}");
        let module = naga::front::wgsl::parse_str(&source).expect("push.wgsl parses");
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).expect("push.wgsl validates");
        let push = module.types.iter().find(|(_, t)| t.name.as_deref() == Some("Push")).unwrap().1;
        let naga::TypeInner::Struct { span, .. } = push.inner else { panic!() };
        assert_eq!(span as usize, std::mem::size_of::<PushGpu>());
    }
}
