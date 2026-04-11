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
    /// Cumulative target offset from touch/drag panning. Added to `target`
    /// when computing the camera position — lets touch handlers pan without
    /// needing direct access to the `CameraControls` struct.
    pan_offset: [f32; 3],
    // Mouse drag state
    dragging: bool,
    last_mouse: Option<(f32, f32)>,
    // Touch state
    /// Number of active touch points (0, 1, or 2).
    touch_count: u32,
    /// Last-known positions of up to 2 tracked touches (page coords).
    touch_positions: [(f32, f32); 2],
    /// Last-known distance between two touch points (for pinch zoom).
    touch_distance: f32,
    dirty: bool,
}

/// Orbit camera controls (mirrors TS CameraControls).
/// On WASM, can auto-attach mouse / wheel / touch events via `from_canvas()`.
///
/// Touch gestures (matches Three.js OrbitControls):
/// - 1 finger drag: orbit (azimuth + elevation)
/// - 2 finger pinch: dolly (radius)
/// - 2 finger drag: pan (target offset)
pub struct CameraControls {
    pub target: Vec3,
    pub radius: f32,
    azimuth: f32,
    elevation: f32,
    /// Pan offset accumulated via touch or mouse drag. Added to `target` in `update()`.
    pan_offset: Vec3,
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
            pan_offset: Vec3::new(0.0, 0.0, 0.0),
            #[cfg(target_arch = "wasm32")]
            shared: None,
        }
    }

    /// Create controls and auto-attach mouse / wheel / touch events to an HTML canvas (WASM only).
    #[cfg(target_arch = "wasm32")]
    pub fn from_canvas(canvas: &web_sys::HtmlCanvasElement, target: Vec3, radius: f32) -> Self {
        use wasm_bindgen::prelude::*;
        use wasm_bindgen::JsCast;

        let shared = Rc::new(RefCell::new(ControlState {
            azimuth: 0.0,
            elevation: 0.3,
            radius,
            pan_offset: [0.0, 0.0, 0.0],
            dragging: false,
            last_mouse: None,
            touch_count: 0,
            touch_positions: [(0.0, 0.0); 2],
            touch_distance: 0.0,
            dirty: false,
        }));

        // ── mousemove ─────────────────────────────────────────
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

        // ── mousedown ─────────────────────────────────────────
        { let s = shared.clone();
          let cb = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |e: web_sys::MouseEvent| {
            s.borrow_mut().dragging = e.buttons() & 1 != 0;
          });
          canvas.add_event_listener_with_callback("mousedown", cb.as_ref().unchecked_ref()).ok();
          cb.forget(); }

        // ── mouseup ───────────────────────────────────────────
        { let s = shared.clone();
          let cb = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |_: web_sys::MouseEvent| {
            s.borrow_mut().dragging = false;
          });
          canvas.add_event_listener_with_callback("mouseup", cb.as_ref().unchecked_ref()).ok();
          cb.forget(); }

        // ── wheel ─────────────────────────────────────────────
        { let s = shared.clone();
          let cb = Closure::<dyn FnMut(web_sys::WheelEvent)>::new(move |e: web_sys::WheelEvent| {
            let mut st = s.borrow_mut();
            st.radius = (st.radius + e.delta_y() as f32 * 0.01).clamp(1.0, 200.0);
            st.dirty = true;
          });
          canvas.add_event_listener_with_callback("wheel", cb.as_ref().unchecked_ref()).ok();
          cb.forget(); }

        // ── touchstart ────────────────────────────────────────
        // Capture initial touch positions + midpoint + pinch distance. A new
        // gesture resets the baseline so deltas in `touchmove` are relative
        // to the start of the current gesture, not the previous one.
        { let s = shared.clone();
          let cb = Closure::<dyn FnMut(web_sys::TouchEvent)>::new(move |e: web_sys::TouchEvent| {
            e.prevent_default();
            let touches = e.touches();
            let count = touches.length().min(2);
            let mut st = s.borrow_mut();
            st.touch_count = count;
            for i in 0..count {
                if let Some(t) = touches.get(i) {
                    st.touch_positions[i as usize] = (t.page_x() as f32, t.page_y() as f32);
                }
            }
            if count == 2 {
                let (x0, y0) = st.touch_positions[0];
                let (x1, y1) = st.touch_positions[1];
                st.touch_distance = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
            }
          });
          canvas.add_event_listener_with_callback("touchstart", cb.as_ref().unchecked_ref()).ok();
          cb.forget(); }

        // ── touchmove ─────────────────────────────────────────
        // 1 finger: orbit. 2 fingers: pinch zoom + pan (drag midpoint).
        { let s = shared.clone();
          let cb = Closure::<dyn FnMut(web_sys::TouchEvent)>::new(move |e: web_sys::TouchEvent| {
            e.prevent_default();
            let touches = e.touches();
            let count = touches.length().min(2);
            if count == 0 { return; }
            let mut st = s.borrow_mut();

            if count == 1 {
                // Orbit: same speed as mouse drag.
                if let Some(t) = touches.get(0) {
                    let (lx, ly) = st.touch_positions[0];
                    let x = t.page_x() as f32;
                    let y = t.page_y() as f32;
                    let dx = (x - lx) * -0.005;
                    let dy = (y - ly) * 0.005;
                    st.azimuth += dx;
                    st.elevation = (st.elevation + dy).clamp(-1.5, 1.5);
                    st.touch_positions[0] = (x, y);
                    st.dirty = true;
                }
            } else if count == 2 {
                let t0 = touches.get(0);
                let t1 = touches.get(1);
                if let (Some(t0), Some(t1)) = (t0, t1) {
                    let x0 = t0.page_x() as f32;
                    let y0 = t0.page_y() as f32;
                    let x1 = t1.page_x() as f32;
                    let y1 = t1.page_y() as f32;

                    // ── Pinch zoom ────────────────────────────
                    // Radius proportional to distance ratio so the pinch
                    // feels continuous and cancels out when fingers move
                    // parallel (pan only).
                    let new_dist = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
                    if st.touch_distance > 0.0 && new_dist > 0.0 {
                        let ratio = st.touch_distance / new_dist;
                        st.radius = (st.radius * ratio).clamp(1.0, 500.0);
                    }
                    st.touch_distance = new_dist;

                    // ── Pan (midpoint drag) ───────────────────
                    // Translate pan_offset along camera-local right/up
                    // axes so a horizontal finger drag moves the target
                    // horizontally on screen regardless of orbit angle.
                    let (mx_old, my_old) = (
                        (st.touch_positions[0].0 + st.touch_positions[1].0) * 0.5,
                        (st.touch_positions[0].1 + st.touch_positions[1].1) * 0.5,
                    );
                    let mx = (x0 + x1) * 0.5;
                    let my = (y0 + y1) * 0.5;
                    // Scale pan to feel natural relative to orbit radius:
                    // bigger radius = larger world-space pan per pixel.
                    let pan_scale = st.radius * 0.0025;
                    let dx = (mx - mx_old) * -pan_scale;
                    let dy = (my - my_old) * pan_scale;

                    // Camera-local right and up from azimuth + elevation.
                    let (sa, ca) = (st.azimuth.sin(), st.azimuth.cos());
                    let (se, ce) = (st.elevation.sin(), st.elevation.cos());
                    // Forward (target - camera) = -(sin(az)cos(el), sin(el), cos(az)cos(el))
                    // Right = normalize(cross(forward, world_up))
                    let right = (ca, 0.0, -sa);
                    // Up = cross(right, forward)
                    let up = (-sa * se, ce, -ca * se);

                    st.pan_offset[0] += dx * right.0 + dy * up.0;
                    st.pan_offset[1] += dx * right.1 + dy * up.1;
                    st.pan_offset[2] += dx * right.2 + dy * up.2;

                    st.touch_positions[0] = (x0, y0);
                    st.touch_positions[1] = (x1, y1);
                    st.dirty = true;
                }
            }
          });
          canvas.add_event_listener_with_callback("touchmove", cb.as_ref().unchecked_ref()).ok();
          cb.forget(); }

        // ── touchend / touchcancel ────────────────────────────
        // Reset to match the remaining touch count so the next move is a
        // clean delta (otherwise lifting one finger of a pinch would teleport).
        let make_end_cb = |s: Rc<RefCell<ControlState>>| {
            Closure::<dyn FnMut(web_sys::TouchEvent)>::new(move |e: web_sys::TouchEvent| {
                e.prevent_default();
                let touches = e.touches();
                let count = touches.length().min(2);
                let mut st = s.borrow_mut();
                st.touch_count = count;
                for i in 0..count {
                    if let Some(t) = touches.get(i) {
                        st.touch_positions[i as usize] = (t.page_x() as f32, t.page_y() as f32);
                    }
                }
                if count == 2 {
                    let (x0, y0) = st.touch_positions[0];
                    let (x1, y1) = st.touch_positions[1];
                    st.touch_distance = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
                } else {
                    st.touch_distance = 0.0;
                }
            })
        };
        {
            let cb = make_end_cb(shared.clone());
            canvas.add_event_listener_with_callback("touchend", cb.as_ref().unchecked_ref()).ok();
            cb.forget();
        }
        {
            let cb = make_end_cb(shared.clone());
            canvas.add_event_listener_with_callback("touchcancel", cb.as_ref().unchecked_ref()).ok();
            cb.forget();
        }

        Self {
            target,
            radius,
            azimuth: 0.0,
            elevation: 0.3,
            pan_offset: Vec3::new(0.0, 0.0, 0.0),
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
            self.pan_offset = Vec3::new(st.pan_offset[0], st.pan_offset[1], st.pan_offset[2]);
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

        // Effective target = user-set target + accumulated pan from touch/mouse.
        let tx = self.target.x + self.pan_offset.x;
        let ty = self.target.y + self.pan_offset.y;
        let tz = self.target.z + self.pan_offset.z;
        let x = tx + self.radius * self.azimuth.sin() * self.elevation.cos();
        let y = ty + self.radius * self.elevation.sin();
        let z = tz + self.radius * self.azimuth.cos() * self.elevation.cos();
        camera.set_position(x, y, z);
        camera.look_at(&Vec3::new(tx, ty, tz));
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

    pub fn set_azimuth(&mut self, a: f32) {
        self.azimuth = a;
        #[cfg(target_arch = "wasm32")]
        if let Some(ref shared) = self.shared { shared.borrow_mut().azimuth = a; }
    }
    pub fn set_elevation(&mut self, e: f32) {
        let e = e.clamp(-1.5, 1.5);
        self.elevation = e;
        #[cfg(target_arch = "wasm32")]
        if let Some(ref shared) = self.shared { shared.borrow_mut().elevation = e; }
    }
    pub fn azimuth(&self) -> f32 { self.azimuth }
    pub fn elevation(&self) -> f32 { self.elevation }
}
