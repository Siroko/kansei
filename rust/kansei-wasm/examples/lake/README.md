# Lake

A small lake of SPH water beside a course of boxes, with a water cannon on its bank and a water
mill turning in it, under the sun with cascaded shadows and TAA. Nothing to download: no animation
pack, no character. Orbit the camera, fire the cannon, watch the mill stir the water and the lake
rest when nothing moves it.

It is also the world of the [motion-matching example](../motion-matching/README.md), which puts a
character on it to wade into the water. That crate depends on this one as a library (see
[As a library](#as-a-library)).

## What it shows

- **The water** is the engine's fluid (`simulations::fluid`), the one the fluid clock uses: about
  128K particles 5 cm apart, less viscous than the clock's. It is held by a `FluidContainer` whose
  walls follow the lake's irregular outline, a strip of shore outside it, and whose floor is the
  lake bed, shelving from the waterline to about 0.6 m deep. Walls, floor and colliders act on it
  as `FluidSubstepPass`es run every substep (`update_batched_with`).
- **The scale:** the simulation runs at 11 times the world's size and √11 times real time, so waves
  and splashes move at their real pace. It steps on a fixed step (`pacing::FixedStep`).
- **The surface** is marching cubes over a surface field (`DensityFieldOptions::particle_radius`):
  smooth over the bulk, with spray as small droplets. It refracts the bed and reflects the sky
  (`FluidSurfaceEffect`, its mesh drawn into the GBuffer with `surface_renderable`).
- **Resting** (`FluidSleep`): out of view for 1.5 s (time for the last waves to die out) the water
  is **culled**: neither stepped nor its surface extracted or composited. In view, once no particle
  has moved faster than 5 cm/s for a second (a GPU reduction read back a few frames late,
  `FluidSpeedProbe`) and nothing is within 2 m of the lake, it is **asleep**: not stepped, its last
  surface drawn as it was. A stream poured in, a turning mill, a changed setting or a reset wake it
  at once. The HUD's `water` line shows the state and the last speed read; `lake_state()` (a wasm
  export) returns `"running"`, `"culled"` or `"asleep"`.
- **The bed** is in the collision world (`CollisionWorld`, a triangle mesh), as are the course's
  boxes and the props, for a character to walk on (the motion-matching example). Its wet line
  follows the water's level.

### The water cannon

A little cannon stands on the lake's west bank, aimed in an arc into the lake.
- **Firing:** with no character here, it is always ready and its prompt always shows
  (`E / X — fire water`). E, the gamepad's X, or a click on the prompt fires a burst; holding pours.
  It fires as it stands: the barrel's aim is fixed. The water leaves the muzzle at 5 m/s from a
  disc 20 cm across, about 2,000 particles a second, flies its arc and splashes in.
- **New particles:** the water is new particles (`FluidNozzle` into `FluidSimulation::with_capacity`'s
  spare room), laid a particle spacing apart so they start at the lake's density.
- **The level rises:** from −0.10 m to the brim (0.00 m), about 48K particles more (128K to 176K),
  in about 20 s of pouring. The bed's wet line follows the level.
- **When full:** the prompt says so and R (the gamepad's Y) drains it back to the start, as the P
  panel's reset does (`FluidSimulation::reset_particles`).
- **The cap's cost:** the brim, not higher, because the full lake costs its particles: at the brim
  a frame took about 14 ms against 13 ms at the start on the test Mac; filled to +0.04 m (half as
  many particles again) it took about 16 ms against 10.

In the motion-matching example the cannon fires only with the character within 2.5 m of it.

### The water mill

A paddle wheel turns in the lake's north shallows.
- **The paddles:** they are colliders that move with the wheel (`FluidCapsule::rigid`). They lift
  the water, throw it off as they rise and push a current along the shore. Its frame's legs are
  still colliders, kept clear of the bed (water squeezed between a collider and the floor jitters
  and never sleeps).
- **Resting:** while it turns, the lake stays awake in view and still culls out of view. Stopped,
  the lake sleeps once its waves die down (about 30 s).
- **The P panel** turns it on and off and sets its speed (16 rpm by default); `mill=0` starts it
  stopped.

### Tweaking the water

P shows a panel (Tweakpane), hidden at first:
- **solver:** SPH (the default) or PBF, Position Based Fluids (`FluidSolver::Pbf`: a density
  constraint projected on the positions, Macklin & Müller 2013), with its iterations, relaxation,
  tensile correction (`s_corr` k and n), XSPH viscosity and vorticity confinement. PBF steps 2
  substeps to SPH's 4;
- **water:** substeps, time scale, the bed's friction, the legs' drag and the landing splash (for
  the motion-matching character), whether it may rest (culled, asleep), a reset (which drains what
  the cannon poured in) and how full it is; and for SPH its viscosity, tensile correction (how much
  of the pull under the rest density acts), stiffness and near stiffness, and rest density;
- **water mill:** turning or not, and its speed;
- **surface:** presets (*Surface field, droplets*, the default; *Density iso, smooth*;
  *Performance*), and each setting: surface field or density, iso level, kernel radius, particle
  radius, grid resolution, interpolation.

## Build and run

```sh
cd rust/kansei-wasm/examples/lake
wasm-pack build --target web --release
python3 -m http.server 8080   # then open http://localhost:8080/www/
```

It is published at kansei.graphics/examples/lake/ (not in the curated list).

## Controls

| | keyboard / mouse | gamepad |
|---|---|---|
| fire the water cannon (hold to pour) | E, or click the prompt | X |
| drain the lake | R | Y |
| orbit, pan, zoom | drag, right drag (or shift drag), wheel | right stick (orbit) |
| tweak panel | P | |

URL parameters:
- `course=0` leaves the boxes out;
- `lake=0` leaves the lake out (a flat ground plane);
- `rest=0` never rests the water (always stepped and drawn);
- `mill=0` starts the water mill stopped;
- `taa=0` turns TAA off;
- `profile=1` logs each labelled GPU pass's time (the fluid's included) to the console every 3 s
  (`Renderer::set_profiling`);
- `debug=1` allows `lake_regions()` (on `window` as `lakeRegions`): it reads the particles back
  from the GPU and counts them in the lake, on the bank, against the walls and outside them, with
  their mean height, to check water drains back. Without `debug=1` it returns null.

A page with its own overlay can show the cannon's prompt and trigger:
- `cannon_prompt()` returns the prompt's text (`""` when it may not fire). The page also sets it
  into a `#prompt` element when it has one.
- `cannon_fire(down)` presses and releases the trigger.
- `lake_fill()` returns `{ particles, capacity, fill, level }`.

## As a library

The crate (`kansei-wasm-lake`) builds the world for another page; the motion-matching example
depends on it with `default-features = false`, which leaves out this page's `start` (the `demo`
feature) so its own `start` is the only one in its module.

- `World::new(renderer, scene, &WorldOptions)` builds the course, the sky, the ground or the lake
  with its cannon and mill, and the sun. `WorldOptions::from_url()` reads `course=`, `lake=`,
  `rest=`, `mill=` and `taa=`; `cannon_reach` makes the cannon fire only with a character within
  reach.
- `World::post_processing(renderer, scene)` returns the chain: the lake's surface effect, TAA and
  the tone map. `renderer(canvas)` makes the renderer it expects (cascaded shadows to 60 m).
- `World::update(WorldInput { legs, landing, at, fire, drain }, dt, scene, volume, renderer,
  view_proj)` steps it: capsules in the water (a character's legs), a landing's splash, where the
  character stands (for the cannon's reach), the cannon's trigger and the drain.
- `World::status(place)` gives the HUD's lake, water, cannon and mill lines; `World::collision` is
  the collision world to walk on.
- A page's state implements `Host` (its world, post-processing volume and renderer) and calls
  `register` once started: the panel's exports (`lake_settings`, `lake_set`, `lake_state`,
  `lake_fill`, `lake_reset`, `lake_surface`, `lake_surface_preset`, `mill_settings`,
  `cannon_prompt`, `cannon_fire`, `lake_regions`) then act on its world. A dependency's
  `#[wasm_bindgen]` exports end up in the final module, so the motion-matching page imports them
  from its own `pkg/`.
