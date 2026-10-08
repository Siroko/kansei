//! The canvas an example draws to: its drawing buffer sized from its CSS box and the device
//! pixel ratio, kept in step when the page resizes.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

#[cfg(target_arch = "wasm32")]
use kansei_core::renderers::{Renderer, RendererConfig};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

/// The canvas's drawing buffer is its CSS size times `devicePixelRatio`, at most this much
/// unless [`Canvas::with_max_pixel_ratio`] or `?dpr=` says otherwise.
const DEFAULT_MAX_PIXEL_RATIO: f32 = 2.0;

/// The page's canvas. Its drawing buffer (`width`/`height` attributes) is the CSS box times the
/// device pixel ratio; [`crate::run`] re-measures it when the box or the ratio changes and
/// reports the new size in [`crate::Frame::resized`].
///
/// `?dpr=<ratio>` on the page's URL fixes the ratio (`?dpr=1` draws one pixel per CSS pixel).
/// The page must give the canvas a CSS size (`width: 100vw; height: 100vh`, say): its box is
/// measured, and a box sized by the drawing buffer would grow with it.
#[derive(Clone)]
pub struct Canvas {
    inner: Rc<Inner>,
}

struct Inner {
    element: HtmlCanvasElement,
    max_pixel_ratio: Cell<f32>,
    /// `?dpr=`, read once.
    pixel_ratio_param: Option<f64>,
    /// A drawing-buffer size that ignores the page (`with_size`).
    fixed_size: Cell<Option<(u32, u32)>>,
    /// Set by the resize observer; the frame loop re-measures on the next frame.
    box_changed: Rc<Cell<bool>>,
    /// The ratio last measured, to notice a move to a screen with another ratio.
    pixel_ratio: Cell<f64>,
    /// The device and queue of the renderer made by [`Canvas::renderer`], for the frame loop's
    /// frames-in-flight cap.
    gpu: RefCell<Option<(wgpu::Device, wgpu::Queue)>>,
    _observer: Option<(web_sys::ResizeObserver, Closure<dyn FnMut()>)>,
}

impl Canvas {
    /// The `<canvas>` with this id, sized for the screen. Also routes panics and logs to the
    /// console ([`crate::init`]).
    pub fn find(id: &str) -> Result<Canvas, JsValue> {
        crate::init();
        let element = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id(id))
            .ok_or_else(|| JsValue::from_str(&format!("no element #{id}")))?
            .dyn_into::<HtmlCanvasElement>()?;
        let box_changed = Rc::new(Cell::new(false));
        let observer = {
            let flag = box_changed.clone();
            let callback = Closure::<dyn FnMut()>::new(move || flag.set(true));
            web_sys::ResizeObserver::new(callback.as_ref().unchecked_ref()).ok().map(|o| {
                o.observe(&element);
                (o, callback)
            })
        };
        let canvas = Canvas {
            inner: Rc::new(Inner {
                element,
                max_pixel_ratio: Cell::new(DEFAULT_MAX_PIXEL_RATIO),
                pixel_ratio_param: crate::param("dpr").and_then(|v| v.parse::<f64>().ok()).filter(|r| *r > 0.0),
                fixed_size: Cell::new(None),
                box_changed,
                pixel_ratio: Cell::new(0.0),
                gpu: RefCell::new(None),
                _observer: observer,
            }),
        };
        canvas.measure();
        Ok(canvas)
    }

    /// Cap the device pixel ratio at `ratio` (2 by default): a costly example can draw fewer
    /// pixels on high-density screens. `?dpr=` still overrides it.
    pub fn with_max_pixel_ratio(self, ratio: f32) -> Self {
        self.inner.max_pixel_ratio.set(ratio.max(0.1));
        self.measure();
        self
    }

    /// Draw at exactly `width` x `height` pixels whatever the page's size (the CSS box still
    /// stretches it), e.g. for a benchmark at a fixed resolution.
    pub fn with_size(self, width: u32, height: u32) -> Self {
        let size = (width.max(1), height.max(1));
        self.inner.fixed_size.set(Some(size));
        self.set_size(size);
        self
    }

    /// The `<canvas>` element, for input listeners (`CameraControls::from_canvas`, ...).
    pub fn element(&self) -> &HtmlCanvasElement {
        &self.inner.element
    }

    /// The drawing buffer's size in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.inner.element.width().max(1), self.inner.element.height().max(1))
    }

    /// Width over height, for a camera's projection.
    pub fn aspect(&self) -> f32 {
        let (w, h) = self.size();
        w as f32 / h as f32
    }

    /// A renderer drawing to this canvas, at its size: `config`'s other fields (sample count,
    /// clear colour, limits, ...) as given. [`crate::run`] keeps its frames in flight in check.
    #[cfg(target_arch = "wasm32")]
    pub async fn renderer(&self, config: RendererConfig) -> Renderer {
        let (width, height) = self.size();
        let device_pixel_ratio = self.inner.pixel_ratio.get() as f32;
        let mut renderer = Renderer::new(RendererConfig { width, height, device_pixel_ratio, ..config });
        renderer.initialize_with_canvas(self.inner.element.clone()).await;
        *self.inner.gpu.borrow_mut() = Some((renderer.device().clone(), renderer.queue().clone()));
        renderer
    }

    /// The device and queue of the renderer [`Canvas::renderer`] made, if it made one.
    pub(crate) fn gpu(&self) -> Option<(wgpu::Device, wgpu::Queue)> {
        self.inner.gpu.borrow().clone()
    }

    /// Re-measure if the CSS box or the pixel ratio changed; the new drawing-buffer size when it
    /// differs from the current one.
    pub(crate) fn poll_resize(&self) -> Option<(u32, u32)> {
        if self.inner.fixed_size.get().is_some() {
            return None;
        }
        let ratio = self.pixel_ratio();
        if !self.inner.box_changed.replace(false) && ratio == self.inner.pixel_ratio.get() {
            return None;
        }
        let before = self.size();
        self.measure();
        let after = self.size();
        (after != before).then_some(after)
    }

    fn pixel_ratio(&self) -> f64 {
        if let Some(ratio) = self.inner.pixel_ratio_param {
            return ratio.clamp(0.1, 4.0);
        }
        let device = web_sys::window().map_or(1.0, |w| w.device_pixel_ratio());
        device.min(self.inner.max_pixel_ratio.get() as f64)
    }

    fn measure(&self) {
        let ratio = self.pixel_ratio();
        self.inner.pixel_ratio.set(ratio);
        if let Some(size) = self.inner.fixed_size.get() {
            self.set_size(size);
            return;
        }
        let element = &self.inner.element;
        let width = (element.client_width().max(1) as f64 * ratio).round() as u32;
        let height = (element.client_height().max(1) as f64 * ratio).round() as u32;
        self.set_size((width, height));
    }

    fn set_size(&self, (width, height): (u32, u32)) {
        let element = &self.inner.element;
        if element.width() != width {
            element.set_width(width);
        }
        if element.height() != height {
            element.set_height(height);
        }
    }
}
