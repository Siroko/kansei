use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

#[derive(Default)]
struct MapState {
    result: Option<Result<(), wgpu::BufferAsyncError>>,
    waker: Option<Waker>,
}

/// A GPU buffer's contents on their way to the CPU (`Renderer::read_buffer_async`). It owns its
/// staging buffer and borrows nothing, so it can be awaited after the renderer is released.
///
/// On the web the browser maps the buffer and wakes the future. Natively nothing polls the
/// device on its own, so the first poll waits for the GPU (`device.poll(Maintain::Wait)`).
pub struct BufferReadback<T> {
    staging: wgpu::Buffer,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    device: wgpu::Device,
    state: Arc<Mutex<MapState>>,
    _data: PhantomData<fn() -> T>,
}

impl<T: bytemuck::Pod> BufferReadback<T> {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Self {
        let size = buffer.size();
        assert!(size % std::mem::size_of::<T>().max(1) as u64 == 0, "a {size}-byte buffer is not a whole number of {}", std::any::type_name::<T>());
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback/Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Readback/Encoder") });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        queue.submit(std::iter::once(encoder.finish()));

        let state = Arc::new(Mutex::new(MapState::default()));
        let mapped = state.clone();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let mut s = mapped.lock().unwrap();
            s.result = Some(result);
            if let Some(waker) = s.waker.take() {
                waker.wake();
            }
        });
        Self { staging, device: device.clone(), state, _data: PhantomData }
    }
}

impl<T: bytemuck::Pod> Future for BufferReadback<T> {
    type Output = Result<Vec<T>, wgpu::BufferAsyncError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.state.lock().unwrap().result.is_none() {
                let _ = self.device.poll(wgpu::Maintain::Wait);
            }
        }
        let mut s = self.state.lock().unwrap();
        match s.result.take() {
            Some(Ok(())) => {
                drop(s);
                let data = bytemuck::pod_collect_to_vec(&self.staging.slice(..).get_mapped_range()[..]);
                self.staging.unmap();
                Poll::Ready(Ok(data))
            }
            Some(Err(e)) => Poll::Ready(Err(e)),
            None => {
                s.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}
