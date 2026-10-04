//! The exports a page calls between frames: the lake's tweak panel (P), the cannon's prompt and
//! trigger, and a debug readback. They act on the world of whichever page registered itself
//! ([`register`]): the lake page's, or the motion-matching example's, whose module carries these
//! exports too (a dependency's `#[wasm_bindgen]` functions end up in the final module).

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

use kansei_core::postprocessing::effects::FluidSurfaceEffect;
use kansei_core::postprocessing::PostProcessingVolume;
use kansei_core::renderers::Renderer;
use kansei_core::simulations::fluid::FluidSolver;

use crate::lake::{Lake, SurfaceSettings};
use crate::world::World;

/// A page's state that holds a [`World`]: what the exports act on.
pub trait Host {
    /// The world, the post-processing volume with the lake's surface effect in it, and the
    /// renderer.
    fn parts(&mut self) -> (&mut World, &mut PostProcessingVolume, &Renderer);
}

thread_local! {
    /// The page's state, for the exports the tweak panel calls between frames.
    static HOST: RefCell<Option<Rc<RefCell<dyn Host>>>> = const { RefCell::new(None) };
}

/// Make `host` the state the exports act on (call once the page has started).
pub fn register(host: Rc<RefCell<dyn Host>>) {
    HOST.with(|h| *h.borrow_mut() = Some(host));
}

/// Run `f` on the world, the volume and the renderer, once a page has registered.
fn with_world<R>(f: impl FnOnce(&mut World, &mut PostProcessingVolume, &Renderer) -> R) -> Option<R> {
    let host = HOST.with(|h| h.borrow().clone())?;
    let mut host = host.borrow_mut();
    let (world, volume, renderer) = host.parts();
    Some(f(world, volume, renderer))
}

/// Run `f` on the lake, its surface effect and the renderer, when there is a lake.
fn with_lake<R>(f: impl FnOnce(&mut Lake, &mut FluidSurfaceEffect, &Renderer) -> R) -> Option<R> {
    with_world(|world, volume, renderer| {
        let lake = world.lake.as_mut()?;
        let surface = volume.effect_mut::<FluidSurfaceEffect>()?;
        Some(f(lake, surface, renderer))
    })
    .flatten()
}

fn surface_js(s: SurfaceSettings) -> JsValue {
    let o = js_sys::Object::new();
    let set = |k: &str, v: JsValue| {
        let _ = js_sys::Reflect::set(&o, &k.into(), &v);
    };
    set("surfaceField", s.surface_field.into());
    set("resolution", s.resolution.into());
    set("kernel", s.kernel.into());
    set("particleRadius", s.particle_radius.into());
    set("iso", s.iso.into());
    set("interpolate", s.interpolate.into());
    o.into()
}

/// The lake's current settings, for the tweak panel: the simulation's and the surface's.
#[wasm_bindgen]
pub fn lake_settings() -> JsValue {
    with_lake(|lake, surface, _| {
        let p = &surface.sim.params;
        let o: js_sys::Object = surface_js(lake.surface_settings()).into();
        for (k, v) in [
            ("viscosity", p.viscosity),
            ("negativePressure", p.negative_pressure_scale),
            ("pressure", p.pressure_multiplier),
            ("nearPressure", p.near_pressure_multiplier),
            ("restDensity", p.density_target),
            ("substeps", p.substeps as f32),
            ("timeScale", lake.time_scale()),
            ("drag", lake.drag()),
            ("splash", lake.splash_push),
            ("friction", lake.friction()),
            ("rest", lake.rest() as u32 as f32),
            ("solver", if p.solver == FluidSolver::Pbf { 1.0 } else { 0.0 }),
            ("pbfIterations", p.pbf.iterations as f32),
            ("pbfRelaxation", p.pbf.relaxation),
            ("pbfScorrK", p.pbf.scorr_k),
            ("pbfScorrN", p.pbf.scorr_n),
            ("pbfXsph", p.pbf.xsph),
            ("pbfVorticity", p.pbf.vorticity),
        ] {
            let _ = js_sys::Reflect::set(&o, &k.into(), &v.into());
        }
        o.into()
    })
    .unwrap_or(JsValue::NULL)
}

/// Whether the lake's water is "running", "culled" (out of view, not stepped or drawn) or
/// "asleep" (settled with nothing near it: not stepped, its surface drawn as it was); null without
/// a lake.
#[wasm_bindgen]
pub fn lake_state() -> Option<String> {
    with_lake(|lake, _, _| lake.state().name().to_string())
}

/// Set one of the lake's simulation settings by name (see `lake::Lake::set`), or the mill's
/// (`mill`: 1 turning, 0 stopped; `millRpm`: its speed in turns a minute).
#[wasm_bindgen]
pub fn lake_set(key: &str, value: f32) -> bool {
    if let Some(done) = with_world(|world, _, _| {
        let mill = world.mill.as_mut()?;
        match key {
            "mill" => mill.on = value > 0.5,
            "millRpm" => mill.rpm = value.clamp(0.0, 60.0),
            _ => return None,
        }
        Some(true)
    })
    .flatten()
    {
        return done;
    }
    with_lake(|lake, surface, _| lake.set(surface, key, value)).unwrap_or(false)
}

/// The mill's settings for the tweak panel ({ mill: 0 or 1, millRpm }), or null without one.
#[wasm_bindgen]
pub fn mill_settings() -> JsValue {
    with_world(|world, _, _| {
        let mill = world.mill.as_ref()?;
        let o = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&o, &"mill".into(), &(mill.on as u32).into());
        let _ = js_sys::Reflect::set(&o, &"millRpm".into(), &mill.rpm.into());
        Some(JsValue::from(o))
    })
    .flatten()
    .unwrap_or(JsValue::NULL)
}

/// What the page should prompt near the cannon ("E / X — fire water …", or that the lake is
/// full), or "" when it may not fire (the character is not by it): for a page that shows its own
/// overlay.
#[wasm_bindgen]
pub fn cannon_prompt() -> String {
    with_world(|world, _, _| world.prompt().to_string()).unwrap_or_default()
}

/// The cannon's trigger from the page (a click or touch on the prompt): `down` fires (a burst,
/// and it pours while held), `false` releases it. In the motion-matching example it only fires
/// with the character by the cannon.
#[wasm_bindgen]
pub fn cannon_fire(down: bool) {
    with_world(|world, _, _| world.fire(down));
}

/// How full the lake is: { particles, capacity, fill (0 at the start's level, 1 full), level (m) },
/// or null without a lake.
#[wasm_bindgen]
pub fn lake_fill() -> JsValue {
    with_world(|world, _, _| {
        let lake = world.lake.as_ref()?;
        let o = js_sys::Object::new();
        for (k, v) in [("particles", lake.particles() as f32), ("capacity", lake.capacity() as f32), ("fill", lake.fill()), ("level", lake.level())] {
            let _ = js_sys::Reflect::set(&o, &k.into(), &v.into());
        }
        Some(JsValue::from(o))
    })
    .flatten()
    .unwrap_or(JsValue::NULL)
}

/// Put the lake's water back as it started.
#[wasm_bindgen]
pub fn lake_reset() {
    with_lake(|lake, surface, _| lake.reset(surface));
}

/// Extract the lake's surface with these settings.
#[wasm_bindgen]
pub fn lake_surface(surface_field: bool, resolution: u32, kernel: f32, particle_radius: f32, iso: f32, interpolate: bool) {
    let settings = SurfaceSettings { surface_field, resolution: resolution.clamp(32, 384), kernel: kernel.max(0.5), particle_radius, iso, interpolate };
    with_lake(|lake, surface, renderer| lake.set_surface(renderer, surface, settings));
}

/// Apply a surface preset ("droplets", "smooth" or "performance") and return its settings.
#[wasm_bindgen]
pub fn lake_surface_preset(name: &str) -> JsValue {
    let settings = match name {
        "smooth" => SurfaceSettings::SMOOTH,
        "performance" => SurfaceSettings::PERFORMANCE,
        _ => SurfaceSettings::DROPLETS,
    };
    with_lake(|lake, surface, renderer| lake.set_surface(renderer, surface, settings));
    surface_js(settings)
}

/// Debugging the lake (with `debug=1` on the page's URL, else null): its particles by region,
/// read back from the GPU ("lake n (y) · bank · wall band · outside").
#[wasm_bindgen]
pub async fn lake_regions() -> JsValue {
    if !kansei_wasm::flag("debug", false) {
        log::warn!("lake_regions reads the lake back from the GPU: open the page with debug=1");
        return JsValue::NULL;
    }
    let Some(positions) = read_lake_positions().await else { return JsValue::NULL };
    with_lake(|lake, _, _| {
        let r = lake.regions(&positions);
        JsValue::from_str(&format!("lake {} ({:.3}) · bank {} ({:.3}) · wall band {} ({:.3}) · outside {} ({:.3})", r[0].0, r[0].1, r[1].0, r[1].1, r[2].0, r[2].1, r[3].0, r[3].1))
    })
    .unwrap_or(JsValue::NULL)
}

/// The lake's particle positions (simulation space, 4 floats each), read back from the GPU.
async fn read_lake_positions() -> Option<Vec<f32>> {
    let readback = with_lake(|_, surface, renderer| surface.sim.positions_buffer().map(|b| renderer.read_buffer_async::<f32>(b)))??;
    readback.await.ok()
}
