use crate::math::Vec3;
use crate::cameras::Camera;

#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(target_arch = "wasm32")]
use std::rc::Rc;

/// Shared inner state for WASM event closures.
#[cfg(target_arch = "wasm32")]
struct ControlState {
    azimuth: f32,
    elevation: f32,
    radius: f32,
    dragging: bool,
    last_mouse: Option<(f32, f32)>,
    dirty: bool,
}

/// Orbit camera controls (mirrors TS CameraControls).
/// On WASM, can auto-attach mouse events via `from_canvas()`.
pub struct CameraControls {
    pub target: Vec3,
    pub radius: f32,
    azimuth: f32,
    elevation: f32,
    #[cfg(target_arch = "wasm32")]
    shared: Option<Rc<RefCell<ControlState>>>,
}

impl CameraControls {
    /// Create controls for native — caller forwards input events manually via rotate()/zoom().
    pub fn new(target: Vec3, radius: f32) -> Self {
        Self {
            target,
            radius,
            azimuth: 0.0,
            elevation: 0.3,
            #[cfg(target_arch = "wasm32")]
            shared: None,
        }
    }

    /// Create controls and auto-attach mouse/wheel events to an HTML canvas (WASM only).
    #[cfg(target_arch = "wasm32")]
    pub fn from_canvas(canvas: &web_sys::HtmlCanvasElement, target: Vec3, radius: f32) -> Self {
        use wasm_bindgen::prelude::*;
        use wasm_bindgen::JsCast;

        let shared = Rc::new(RefCell::new(ControlState {
            azimuth: 0.0,
            elevation: 0.3,
            radius,
            dragging: false,
            last_mouse: None,
            dirty: false,
        }));

        // mousemove
        { let s = shared.clone();
          let cb = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |e: web_sys::MouseEvent| {
            let mut st = s.borrow_mut();
            if st.dragging {
                if let Some((lx, ly)) = st.last_mouse {
                    let dx = (e.offset_x() as f32 - lx) * -0.005;
                    let dy = (e.offset_y() as f32 - ly) * 0.005;
                    st.azimuth += dx;
                    st.elevation = (st.elevation + dy).clamp(-1.5, 1.5);
                    st.dirty = true;
                }
            }
            st.last_mouse = Some((e.offset_x() as f32, e.offset_y() as f32));
          });
          canvas.add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref()).ok();
          cb.forget(); }

        // mousedown
        { let s = shared.clone();
          let cb = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |e: web_sys::MouseEvent| {
            s.borrow_mut().dragging = e.buttons() & 1 != 0;
          });
          canvas.add_event_listener_with_callback("mousedown", cb.as_ref().unchecked_ref()).ok();
          cb.forget(); }

        // mouseup
        { let s = shared.clone();
          let cb = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |_: web_sys::MouseEvent| {
            s.borrow_mut().dragging = false;
          });
          canvas.add_event_listener_with_callback("mouseup", cb.as_ref().unchecked_ref()).ok();
          cb.forget(); }

        // wheel
        { let s = shared.clone();
          let cb = Closure::<dyn FnMut(web_sys::WheelEvent)>::new(move |e: web_sys::WheelEvent| {
            let mut st = s.borrow_mut();
            st.radius = (st.radius + e.delta_y() as f32 * 0.01).clamp(1.0, 200.0);
            st.dirty = true;
          });
          canvas.add_event_listener_with_callback("wheel", cb.as_ref().unchecked_ref()).ok();
          cb.forget(); }

        Self {
            target,
            radius,
            azimuth: 0.0,
            elevation: 0.3,
            shared: Some(shared),
        }
    }

    /// Sync state from WASM event closures (call before update).
    #[cfg(target_arch = "wasm32")]
    fn sync_from_shared(&mut self) {
        if let Some(ref shared) = self.shared {
            let st = shared.borrow();
            self.azimuth = st.azimuth;
            self.elevation = st.elevation;
            self.radius = st.radius;
        }
    }

    /// Returns true if the camera moved since last check (WASM only).
    pub fn is_dirty(&mut self) -> bool {
        #[cfg(target_arch = "wasm32")]
        if let Some(ref shared) = self.shared {
            let mut st = shared.borrow_mut();
            let d = st.dirty;
            st.dirty = false;
            return d;
        }
        false
    }

    pub fn update(&mut self, camera: &mut Camera, _dt: f32) {
        #[cfg(target_arch = "wasm32")]
        self.sync_from_shared();

        let x = self.target.x + self.radius * self.azimuth.sin() * self.elevation.cos();
        let y = self.target.y + self.radius * self.elevation.sin();
        let z = self.target.z + self.radius * self.azimuth.cos() * self.elevation.cos();
        camera.set_position(x, y, z);
        camera.look_at(&self.target);
    }

    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.azimuth += dx;
        self.elevation = (self.elevation + dy).clamp(-1.5, 1.5);
        #[cfg(target_arch = "wasm32")]
        if let Some(ref shared) = self.shared {
            let mut st = shared.borrow_mut();
            st.azimuth = self.azimuth;
            st.elevation = self.elevation;
        }
    }

    pub fn zoom(&mut self, delta: f32) {
        self.radius = (self.radius - delta).clamp(1.0, 200.0);
        #[cfg(target_arch = "wasm32")]
        if let Some(ref shared) = self.shared {
            shared.borrow_mut().radius = self.radius;
        }
    }

    pub fn set_azimuth(&mut self, a: f32) { self.azimuth = a; }
    pub fn set_elevation(&mut self, e: f32) { self.elevation = e.clamp(-1.5, 1.5); }
    pub fn azimuth(&self) -> f32 { self.azimuth }
    pub fn elevation(&self) -> f32 { self.elevation }
}
