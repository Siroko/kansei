//! The `requestAnimationFrame` loop.

use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::cameras::Camera;
use kansei_core::renderers::Renderer;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::Canvas;

/// One animation frame, as [`run`] hands it to the example.
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    /// Seconds since [`run`] started.
    pub time: f64,
    /// Seconds since the previous frame (0 on the first).
    pub dt: f32,
    /// Frames before this one.
    pub index: u64,
    /// The canvas's drawing-buffer size in pixels.
    pub size: (u32, u32),
    /// The new drawing-buffer size when the canvas changed size since the previous frame.
    pub resized: Option<(u32, u32)>,
}

impl Frame {
    /// Apply a resize, if this frame has one, to the renderer (surface and depth targets) and
    /// the camera's aspect ratio. Call it before rendering; an example with size-dependent
    /// resources of its own checks [`Frame::resized`] as well.
    pub fn resize(&self, renderer: &mut Renderer, camera: &mut Camera) {
        if let Some((width, height)) = self.resized {
            renderer.resize(width, height);
            camera.aspect = width as f32 / height as f32;
            camera.update_projection_matrix();
        }
    }
}

/// Call `frame` on every animation frame from now on, with the frame's timing and any canvas
/// resize. The closure owns (or shares) the example's state.
pub fn run(canvas: &Canvas, mut frame: impl FnMut(&Frame) + 'static) {
    let canvas = canvas.clone();
    let start = crate::now();
    let mut last = start;
    let mut index = 0u64;
    let next: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let first = next.clone();
    *first.borrow_mut() = Some(Closure::new(move || {
        let now = crate::now();
        let resized = canvas.poll_resize();
        frame(&Frame {
            time: now - start,
            dt: if index == 0 { 0.0 } else { (now - last) as f32 },
            index,
            size: canvas.size(),
            resized,
        });
        last = now;
        index += 1;
        request_animation_frame(next.borrow().as_ref().unwrap());
    }));
    request_animation_frame(first.borrow().as_ref().unwrap());
}

fn request_animation_frame(callback: &Closure<dyn FnMut()>) {
    web_sys::window()
        .expect("no window")
        .request_animation_frame(callback.as_ref().unchecked_ref())
        .expect("requestAnimationFrame failed");
}
