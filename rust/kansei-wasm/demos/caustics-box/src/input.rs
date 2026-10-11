//! The pointer on the canvas: what each drag does is the page's choice (`Pointer::take`), so the
//! left button can push the water while the right one orbits (`CameraControls::from_canvas`
//! binds the left to orbiting). Pointer events, so a pen or a finger drags as the mouse does.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

/// Which drag is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Drag {
    /// the left button (or a finger)
    Primary,
    /// the right or middle button, or the left with alt
    Orbit,
    /// the left with shift
    Light,
}

#[derive(Default)]
struct State {
    drag: Option<Drag>,
    /// the pointer (CSS px from the canvas's corner) and the canvas's size
    at: [f32; 2],
    size: [f32; 2],
    /// motion (px) and wheel since the last `take`
    moved: [f32; 2],
    wheel: f32,
}

/// What the pointer did since the last frame.
pub struct PointerFrame {
    pub drag: Option<Drag>,
    /// the pointer in normalized device coordinates (x right, y up)
    pub ndc: [f32; 2],
    /// its motion (px)
    pub moved: [f32; 2],
    pub wheel: f32,
}

pub struct Pointer {
    state: Rc<RefCell<State>>,
}

impl Pointer {
    pub fn listen(canvas: &web_sys::HtmlCanvasElement) -> Self {
        let state = Rc::new(RefCell::new(State::default()));
        let place = |s: &mut State, canvas: &web_sys::HtmlCanvasElement, e: &web_sys::MouseEvent| {
            s.at = [e.offset_x() as f32, e.offset_y() as f32];
            s.size = [canvas.client_width().max(1) as f32, canvas.client_height().max(1) as f32];
        };
        let on = |name: &str, f: Box<dyn FnMut(web_sys::PointerEvent)>| {
            let cb = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(f);
            canvas.add_event_listener_with_callback(name, cb.as_ref().unchecked_ref()).ok();
            cb.forget();
        };
        {
            let (state, canvas) = (state.clone(), canvas.clone());
            on(
                "pointerdown",
                Box::new(move |e: web_sys::PointerEvent| {
                    let mut s = state.borrow_mut();
                    place(&mut s, &canvas, &e);
                    s.drag = Some(match e.button() {
                        0 if e.shift_key() => Drag::Light,
                        0 if e.alt_key() => Drag::Orbit,
                        0 => Drag::Primary,
                        _ => Drag::Orbit,
                    });
                    canvas.set_pointer_capture(e.pointer_id()).ok();
                    canvas.focus().ok();
                }),
            );
        }
        {
            let (state, canvas) = (state.clone(), canvas.clone());
            on(
                "pointermove",
                Box::new(move |e: web_sys::PointerEvent| {
                    let mut s = state.borrow_mut();
                    let before = s.at;
                    place(&mut s, &canvas, &e);
                    if s.drag.is_some() {
                        s.moved[0] += s.at[0] - before[0];
                        s.moved[1] += s.at[1] - before[1];
                    }
                }),
            );
        }
        for name in ["pointerup", "pointercancel"] {
            let state = state.clone();
            on(name, Box::new(move |_| state.borrow_mut().drag = None));
        }
        {
            let state = state.clone();
            let cb = Closure::<dyn FnMut(web_sys::WheelEvent)>::new(move |e: web_sys::WheelEvent| {
                e.prevent_default();
                state.borrow_mut().wheel += e.delta_y() as f32;
            });
            let options = web_sys::AddEventListenerOptions::new();
            options.set_passive(false);
            canvas.add_event_listener_with_callback_and_add_event_listener_options("wheel", cb.as_ref().unchecked_ref(), &options).ok();
            cb.forget();
        }
        {
            let cb = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |e: web_sys::MouseEvent| e.prevent_default());
            canvas.add_event_listener_with_callback("contextmenu", cb.as_ref().unchecked_ref()).ok();
            cb.forget();
        }
        Self { state }
    }

    /// The drag, the pointer and what moved since the last call.
    pub fn take(&self) -> PointerFrame {
        let mut s = self.state.borrow_mut();
        let [w, h] = [s.size[0].max(1.0), s.size[1].max(1.0)];
        let frame = PointerFrame { drag: s.drag, ndc: [s.at[0] / w * 2.0 - 1.0, 1.0 - s.at[1] / h * 2.0], moved: s.moved, wheel: s.wheel };
        s.moved = [0.0; 2];
        s.wheel = 0.0;
        frame
    }
}
