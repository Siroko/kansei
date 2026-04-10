use crate::math::Vec2;

#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(target_arch = "wasm32")]
use std::rc::Rc;

/// Shared target for WASM mousemove closure.
#[cfg(target_arch = "wasm32")]
struct MouseTarget {
    x: f32,
    y: f32,
}

/// Mouse position/direction tracker (mirrors TS MouseVectors).
pub struct MouseVectors {
    pub speed: f32,
    pub position: Vec2,
    pub direction: Vec2,
    pub strength: f32,
    end: Vec2,
    #[cfg(target_arch = "wasm32")]
    shared: Option<Rc<RefCell<MouseTarget>>>,
}

impl MouseVectors {
    pub fn new() -> Self {
        Self {
            speed: 8.0,
            position: Vec2::ZERO,
            direction: Vec2::ZERO,
            strength: 0.0,
            end: Vec2::ZERO,
            #[cfg(target_arch = "wasm32")]
            shared: None,
        }
    }

    /// Create and auto-attach a mousemove listener to an HTML canvas (WASM only).
    /// Converts pixel coordinates to NDC [-1, 1] and feeds them as interpolation targets.
    #[cfg(target_arch = "wasm32")]
    pub fn from_canvas(canvas: &web_sys::HtmlCanvasElement) -> Self {
        use wasm_bindgen::prelude::*;
        use wasm_bindgen::JsCast;

        // Use CSS dimensions (client_width/height) since mouse events report CSS pixels
        let w = canvas.client_width().max(1) as f32;
        let h = canvas.client_height().max(1) as f32;

        let shared = Rc::new(RefCell::new(MouseTarget { x: 0.0, y: 0.0 }));

        { let s = shared.clone();
          let cb = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |e: web_sys::MouseEvent| {
            let mut t = s.borrow_mut();
            t.x = (e.offset_x() as f32 / w - 0.5) * 2.0;
            t.y = (e.offset_y() as f32 / h - 0.5) * 2.0;
          });
          canvas.add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref()).ok();
          cb.forget();
        }

        Self {
            speed: 8.0,
            position: Vec2::ZERO,
            direction: Vec2::ZERO,
            strength: 0.0,
            end: Vec2::ZERO,
            shared: Some(shared),
        }
    }

    /// Set an immediate position (and target), avoiding interpolation jumps.
    pub fn set_position(&mut self, x: f32, y: f32) {
        self.position = Vec2::new(x, y);
        self.end = self.position;
        self.direction = Vec2::ZERO;
        self.strength = 0.0;
    }

    /// Set the interpolation target in NDC space [-1, 1].
    pub fn set_target(&mut self, x: f32, y: f32) {
        self.end = Vec2::new(x, y);
    }

    /// Set the interpolation target from pixel coordinates.
    pub fn set_target_from_screen(&mut self, x: f32, y: f32, width: f32, height: f32) {
        let w = width.max(1.0);
        let h = height.max(1.0);
        self.set_target((x / w - 0.5) * 2.0, (y / h - 0.5) * 2.0);
    }

    /// Set immediate position from pixel coordinates.
    pub fn set_position_from_screen(&mut self, x: f32, y: f32, width: f32, height: f32) {
        let w = width.max(1.0);
        let h = height.max(1.0);
        self.set_position((x / w - 0.5) * 2.0, (y / h - 0.5) * 2.0);
    }

    /// Sync from WASM event closure target, then interpolate.
    pub fn update(&mut self, dt: f32) {
        #[cfg(target_arch = "wasm32")]
        if let Some(ref shared) = self.shared {
            let t = shared.borrow();
            self.end = Vec2::new(t.x, t.y);
        }

        let prev = self.position;
        let t = self.speed * dt.max(0.0);
        self.position = Vec2::new(
            self.position.x + (self.end.x - self.position.x) * t,
            self.position.y + (self.end.y - self.position.y) * t,
        );
        self.direction = Vec2::new(prev.x - self.position.x, prev.y - self.position.y);
        self.strength = self.direction.length();
    }
}

impl Default for MouseVectors {
    fn default() -> Self { Self::new() }
}
