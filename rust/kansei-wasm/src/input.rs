//! Keyboard and gamepad input for pages that play: the keys held and pressed, and the first
//! gamepad's sticks and buttons.

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[derive(Default)]
struct KeyState {
    held: HashSet<String>,
    pressed: Vec<String>,
}

/// The keyboard, from the window's key events: which keys are held, and which went down since
/// the last [`Keys::take_pressed`]. Keys are `KeyboardEvent.key` in lower case: `"w"`,
/// `"arrowup"`, `" "`, `"shift"`.
///
/// Keys typed into a text field (a panel's number box) are left to it. Otherwise the arrows
/// and space don't scroll the page. A key released while the page is in the background never
/// sends its keyup, so leaving the page releases every key.
#[derive(Clone, Default)]
pub struct Keys(Rc<RefCell<KeyState>>);

impl Keys {
    /// Listen to the window's key events from now on.
    pub fn listen() -> Self {
        let keys = Self::default();
        if let Some(window) = web_sys::window() {
            let down = keys.0.clone();
            let on_down = Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(move |e: web_sys::KeyboardEvent| {
                if typed_into_a_field(&e) {
                    return;
                }
                let key = e.key().to_lowercase();
                if key.starts_with("arrow") || key == " " {
                    e.prevent_default();
                }
                let mut keys = down.borrow_mut();
                if !e.repeat() {
                    keys.pressed.push(key.clone());
                }
                keys.held.insert(key);
            });
            let up = keys.0.clone();
            let on_up = Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(move |e: web_sys::KeyboardEvent| {
                up.borrow_mut().held.remove(&e.key().to_lowercase());
            });
            let blur = keys.0.clone();
            let on_blur = Closure::<dyn FnMut()>::new(move || blur.borrow_mut().held.clear());
            let _ = window.add_event_listener_with_callback("keydown", on_down.as_ref().unchecked_ref());
            let _ = window.add_event_listener_with_callback("keyup", on_up.as_ref().unchecked_ref());
            let _ = window.add_event_listener_with_callback("blur", on_blur.as_ref().unchecked_ref());
            on_down.forget();
            on_up.forget();
            on_blur.forget();
        }
        keys
    }

    /// Whether `key` is held down.
    pub fn held(&self, key: &str) -> bool {
        self.0.borrow().held.contains(key)
    }

    /// Whether any of `keys` is held down.
    pub fn any_held(&self, keys: &[&str]) -> bool {
        let state = self.0.borrow();
        keys.iter().any(|k| state.held.contains(*k))
    }

    /// 1 while one of `positive` is held, -1 for `negative`, 0 for both or neither: an axis
    /// such as `axis(&["a", "arrowleft"], &["d", "arrowright"])`.
    pub fn axis(&self, negative: &[&str], positive: &[&str]) -> f32 {
        self.any_held(positive) as i32 as f32 - self.any_held(negative) as i32 as f32
    }

    /// The keys that went down since the last call, in order (a held key's auto-repeat is not
    /// a press).
    pub fn take_pressed(&self) -> Vec<String> {
        std::mem::take(&mut self.0.borrow_mut().pressed)
    }
}

/// Whether `e` is typing into a text field, which keeps its keys.
fn typed_into_a_field(e: &web_sys::KeyboardEvent) -> bool {
    let Some(element) = e.target().and_then(|t| t.dyn_into::<web_sys::HtmlElement>().ok()) else { return false };
    if element.is_content_editable() {
        return true;
    }
    match element.tag_name().as_str() {
        "TEXTAREA" | "SELECT" => true,
        "INPUT" => !matches!(
            element.get_attribute("type").unwrap_or_default().to_lowercase().as_str(),
            "checkbox" | "radio" | "button" | "submit" | "reset" | "range" | "color" | "file"
        ),
        _ => false,
    }
}

/// The first connected gamepad in the standard mapping, read once a frame with
/// [`Gamepad::poll`]: its sticks past a dead zone, its buttons held, and those pressed since the
/// poll before.
#[derive(Default)]
pub struct Gamepad {
    connected: bool,
    left: [f32; 2],
    right: [f32; 2],
    buttons: Vec<(bool, f32)>,
    was: Vec<bool>,
}

impl Gamepad {
    /// Bottom face button (A, cross).
    pub const A: usize = 0;
    /// Right face button (B, circle).
    pub const B: usize = 1;
    /// Left face button (X, square).
    pub const X: usize = 2;
    /// Top face button (Y, triangle).
    pub const Y: usize = 3;
    pub const LEFT_BUMPER: usize = 4;
    pub const RIGHT_BUMPER: usize = 5;
    pub const LEFT_TRIGGER: usize = 6;
    pub const RIGHT_TRIGGER: usize = 7;

    /// No gamepad read yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Read the first connected gamepad; false (sticks at rest, no button held) when there is
    /// none.
    pub fn poll(&mut self) -> bool {
        self.was = self.buttons.iter().map(|b| b.0).collect();
        let pad = web_sys::window()
            .and_then(|w| w.navigator().get_gamepads().ok())
            .and_then(|pads| (0..pads.length()).find_map(|i| pads.get(i).dyn_into::<web_sys::Gamepad>().ok()));
        let Some(pad) = pad else {
            *self = Self { was: std::mem::take(&mut self.was), ..Self::default() };
            return false;
        };
        let axes: Vec<f32> = pad.axes().iter().map(|a| a.as_f64().unwrap_or(0.0) as f32).collect();
        let axis = |i: usize| axes.get(i).copied().unwrap_or(0.0);
        self.left = dead_zone(axis(0), axis(1));
        self.right = dead_zone(axis(2), axis(3));
        self.buttons = pad
            .buttons()
            .iter()
            .map(|b| b.dyn_into::<web_sys::GamepadButton>().map_or((false, 0.0), |b| (b.pressed(), b.value() as f32)))
            .collect();
        self.connected = true;
        true
    }

    /// Whether the last poll found a gamepad.
    pub fn connected(&self) -> bool {
        self.connected
    }

    /// The left stick, x right and y down, each -1 to 1, zero within the dead zone.
    pub fn left_stick(&self) -> [f32; 2] {
        self.left
    }

    /// The right stick, as [`Gamepad::left_stick`].
    pub fn right_stick(&self) -> [f32; 2] {
        self.right
    }

    /// Whether `button` is held.
    pub fn held(&self, button: usize) -> bool {
        self.buttons.get(button).is_some_and(|b| b.0)
    }

    /// How far `button` is pressed, 0 to 1 (a trigger's travel).
    pub fn value(&self, button: usize) -> f32 {
        self.buttons.get(button).map_or(0.0, |b| b.1)
    }

    /// Whether `button` went down between the last two polls.
    pub fn pressed(&self, button: usize) -> bool {
        self.held(button) && !self.was.get(button).copied().unwrap_or(false)
    }
}

/// A stick past a radial dead zone of 0.15, rescaled so the zone's edge reads 0 and full tilt 1.
fn dead_zone(x: f32, y: f32) -> [f32; 2] {
    const DEAD: f32 = 0.15;
    let m = (x * x + y * y).sqrt();
    if m < DEAD {
        return [0.0, 0.0];
    }
    let s = ((m - DEAD) / (1.0 - DEAD)).min(1.0) / m;
    [x * s, y * s]
}

#[cfg(test)]
mod tests {
    use super::dead_zone;

    #[test]
    fn the_dead_zone_rescales_the_tilt_and_keeps_the_direction() {
        assert_eq!(dead_zone(0.1, -0.1), [0.0, 0.0]);
        let [x, y] = dead_zone(0.0, -1.0);
        assert!(x == 0.0 && (y + 1.0).abs() < 1e-6);
        let [x, y] = dead_zone(0.3, 0.4);
        let m = (x * x + y * y).sqrt();
        assert!((m - (0.5 - 0.15) / 0.85).abs() < 1e-6 && (x / y - 0.75).abs() < 1e-6);
        let [x, y] = dead_zone(2.0, 0.0);
        assert!((x - 1.0).abs() < 1e-6 && y == 0.0);
    }
}
