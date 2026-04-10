use crate::materials::ComputePass;

/// Dispatch multiple compute passes in sequence with barriers between them.
pub struct ComputeBatch;

impl ComputeBatch {
    /// Encode and submit a batch of compute dispatches.
    /// Each pass gets its own beginComputePass/end for implicit barriers.
    pub fn submit(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        passes: &[(&ComputePass, u32, u32, u32)], // (pass, workgroups_x, y, z)
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ComputeBatch"),
        });

        for (pass, wx, wy, wz) in passes {
            let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ComputeBatch/Pass"),
                timestamp_writes: None,
            });
            pass.dispatch(&mut compute, *wx, *wy, *wz);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Encode passes into an existing command encoder (no submit).
    pub fn encode(
        encoder: &mut wgpu::CommandEncoder,
        passes: &[(&ComputePass, u32, u32, u32)],
    ) {
        for (pass, wx, wy, wz) in passes {
            let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ComputeBatch/Pass"),
                timestamp_writes: None,
            });
            pass.dispatch(&mut compute, *wx, *wy, *wz);
        }
    }
}
