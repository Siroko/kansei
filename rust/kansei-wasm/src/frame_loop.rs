//! The `requestAnimationFrame` loop.

use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::cameras::Camera;
use kansei_core::renderers::Renderer;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::frame_gate::FramesInFlight;
use crate::Canvas;

/// One animation frame, as [`run`] hands it to the example.
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    /// Seconds since [`run`] started.
    pub time: f64,
    /// Seconds since the previous rendered frame (0 on the first), refreshes skipped between
    /// them included: advance animation by it and a skipped refresh loses no time.
    pub dt: f32,
    /// Frames rendered before this one.
    pub index: u64,
    /// Refreshes skipped since the previous rendered frame, while the GPU was still busy with
    /// the frames before (see [`RunOptions::max_frames_in_flight`]).
    pub skipped: u32,
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

/// The frames in flight [`run`] allows unless [`RunOptions`] or `?inflight=` say otherwise.
/// Two keep the frame rate of a scene that keeps up only by encoding a frame while the GPU
/// draws the one before (one halved it, on a 120 Hz display), and still hold a scene far over
/// budget to two frames behind the input.
pub const DEFAULT_MAX_FRAMES_IN_FLIGHT: u32 = 2;

/// How [`run_with`] drives the loop.
#[derive(Debug, Clone, Copy)]
pub struct RunOptions {
    /// The most frames on the GPU at once: on a refresh where this many rendered frames are
    /// still on the GPU, the frame is skipped (the next refresh is still requested) and its time
    /// goes to the next rendered frame's `dt`. Without a cap the browser keeps calling the loop
    /// while the GPU falls behind, and the frames queue (some five deep in Chrome: input shows
    /// half a second late at 10 fps). 0: no cap. `?inflight=` on the page's URL overrides it.
    ///
    /// The cap needs the renderer's queue: it applies to the renderer made by
    /// [`crate::Canvas::renderer`] (the frame's work counts from that queue's submits).
    pub max_frames_in_flight: u32,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self { max_frames_in_flight: DEFAULT_MAX_FRAMES_IN_FLIGHT }
    }
}

/// Call `frame` on every animation frame from now on (the page's `requestAnimationFrame`, or in
/// a worker the worker's), with the frame's timing and any canvas resize. The closure owns (or shares) the example's state. At most
/// [`DEFAULT_MAX_FRAMES_IN_FLIGHT`] frames go to the GPU at once (see
/// [`RunOptions::max_frames_in_flight`]; `?inflight=0` turns the cap off).
pub fn run(canvas: &Canvas, frame: impl FnMut(&Frame) + 'static) {
    run_with(canvas, RunOptions::default(), frame);
}

/// [`run`], with options.
///
/// With a `kansei_core::pacing::FramePacer` the two compose: the pacer is asked on rendered
/// frames only (pass it [`Frame::time`] in ms), and sees a refresh the cap skipped as one the
/// browser called late, as Chrome does while frames are on the GPU; it slows to a cadence the
/// GPU keeps up with, where the cap seldom skips. At the default cap of two that is what the
/// pacer already expects of the browser. At one the cap also skips refreshes the pacer would
/// have declined, and it may learn a longer refresh interval than the display's; it still renders
/// the same steady cadence (`pacing`'s
/// `a_frame_loop_that_caps_frames_in_flight_renders_the_same_cadence`). `dt` stays the time
/// between rendered frames either way.
pub fn run_with(canvas: &Canvas, options: RunOptions, mut frame: impl FnMut(&Frame) + 'static) {
    let canvas = canvas.clone();
    let cap = crate::param_or("inflight", options.max_frames_in_flight);
    let start = crate::now();
    let mut last = start;
    let mut index = 0u64;
    let mut skipped = 0u32;
    let mut gate: Option<FramesInFlight> = None;
    let next: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let first = next.clone();
    *first.borrow_mut() = Some(Closure::new(move || {
        if gate.is_none() && cap > 0 {
            gate = canvas.gpu().map(|(device, queue)| FramesInFlight::new(&device, &queue, cap));
        }
        if gate.as_ref().is_some_and(|g| !g.ready()) {
            skipped += 1;
            request_animation_frame(next.borrow().as_ref().unwrap());
            return;
        }
        let now = crate::now();
        let resized = canvas.poll_resize();
        frame(&Frame {
            time: now - start,
            dt: if index == 0 { 0.0 } else { (now - last) as f32 },
            index,
            skipped,
            size: canvas.size(),
            resized,
        });
        if let Some(gate) = gate.as_mut() {
            gate.frame_submitted();
        }
        last = now;
        index += 1;
        skipped = 0;
        request_animation_frame(next.borrow().as_ref().unwrap());
    }));
    request_animation_frame(first.borrow().as_ref().unwrap());
}

/// The page's `requestAnimationFrame`, or in a worker the worker's (see `js/worker.js`
/// `requestFrame`: where a browser has none, the page's frames are posted to the worker).
fn request_animation_frame(callback: &Closure<dyn FnMut()>) {
    match web_sys::window() {
        Some(window) => {
            window.request_animation_frame(callback.as_ref().unchecked_ref()).expect("requestAnimationFrame failed");
        }
        None => crate::worker::request_frame(callback.as_ref().unchecked_ref()),
    }
}
