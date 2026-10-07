//! How many frames may be on the GPU at once.
//!
//! A browser calls `requestAnimationFrame` whether or not the GPU has finished the frames before:
//! a frame that costs more than a refresh queues behind the last, and the queue grows until
//! Chrome holds back the canvas (about five frames: half a second behind the input at 10 fps).
//! `FramesInFlight` lets the loop skip a refresh while `cap` frames are still on the GPU, so the
//! frame it renders next starts from fresh input.
//!
//! wgpu 24's WebGPU backend does not implement `Queue::on_submitted_work_done`, so the end of a
//! frame is a fence of its own: a 4-byte buffer cleared after the frame's last submit and then
//! mapped, which resolves once the GPU has passed it (the queue runs in order). It uses only the
//! device and queue, nothing of the page, so it works the same from a worker.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

/// Counts the frames still on the GPU; see the module docs.
pub struct FramesInFlight {
    device: wgpu::Device,
    queue: wgpu::Queue,
    cap: u32,
    in_flight: Arc<AtomicU32>,
    /// One per frame in flight (a fence is reused once its map resolves)
    fences: Vec<(wgpu::Buffer, Arc<AtomicBool>)>,
}

impl FramesInFlight {
    /// At most `cap` frames on the GPU; 0 never holds a frame back.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, cap: u32) -> Self {
        Self { device: device.clone(), queue: queue.clone(), cap, in_flight: Arc::new(AtomicU32::new(0)), fences: Vec::new() }
    }

    pub fn cap(&self) -> u32 {
        self.cap
    }

    /// Frames submitted whose GPU work is not done yet.
    pub fn in_flight(&self) -> u32 {
        self.in_flight.load(Ordering::Acquire)
    }

    /// Whether a frame may start now.
    pub fn ready(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::Maintain::Poll);
        self.cap == 0 || self.in_flight() < self.cap
    }

    /// After the frame's last submit: fence it.
    pub fn frame_submitted(&mut self) {
        if self.cap == 0 {
            return;
        }
        let k = match self.fences.iter().position(|(_, busy)| !busy.load(Ordering::Acquire)) {
            Some(k) => k,
            None => {
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("FramesInFlight"),
                    size: 4,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.fences.push((buffer, Arc::new(AtomicBool::new(false))));
                self.fences.len() - 1
            }
        };
        let (buffer, busy) = &self.fences[k];
        busy.store(true, Ordering::Release);
        self.in_flight.fetch_add(1, Ordering::AcqRel);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("FramesInFlight") });
        encoder.clear_buffer(buffer, 0, None);
        self.queue.submit(Some(encoder.finish()));
        let (in_flight, busy, mapped) = (self.in_flight.clone(), busy.clone(), buffer.clone());
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            if result.is_ok() {
                mapped.unmap();
            }
            // (a failed map, as on a lost device, still ends the frame: the loop must not stall)
            in_flight.fetch_sub(1, Ordering::AcqRel);
            busy.store(false, Ordering::Release);
        });
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::Maintain::Poll);
    }
}
