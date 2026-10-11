# Motion matching

A skinned character driven by motion matching (`kansei_core::animation::motion_matching`):
walking, running, starting, stopping, turning and strafing under keyboard or gamepad control, on
a ground plane with cascaded shadows and TAA, and over a course of boxes: hurdling, vaulting,
mantling and climbing on request, falling off edges and landing, and wading into a lake.

**No animation data ships with Kansei.** The page loads a motion-matching pack (`.kmm`) from
`www/pack/locomotion.kmm`, or from the URL in `?pack=<url>`. Without one it shows how to make
one, and renders the empty scene. So the example is not on kansei.graphics: its world without
the character (the course, the lake, the cannon and the mill) is the [lake example](../lake/README.md),
which is. This crate depends on that one for the world (`kansei_wasm_lake::World`) and keeps the
character: the controller, the bodies and retargeting, the follow camera, the HUD, the packs and
the clip tools.

`www/kimodo.html` is a second page on the same module, for packs of generated clips (see
[Generated animation (Kimodo)](#generated-animation-kimodo)).

`www/room.html` is a third, the same character in a furnished room lit by ray-traced direct
light and global illumination, with mirrors, a glass dragon in a pond, furniture to vault and
climb, fog and dust (see
[The room](#the-room)). The character and its controls are shared by the pages (`src/character.rs`,
`src/player.rs`); the room is `src/room/`.

## A pack

Bake one from glTF clips and a skinned mesh with [`kansei-anim-bake`](../../../kansei-anim-bake/README.md).
It can export from an Unreal Engine project, headless.

Keep packs outside the repository and link them in. `*.kmm` is gitignored, and so is `www/pack/`:

```sh
ln -s /path/to/private/packs rust/kansei-wasm/demos/motion-matching/www/pack
```

An app that stores its packs another way (encrypted, say) can depend on this crate and call
`start_with_loader(canvas_id, load)` instead of `start`: `load` gets each pack's URL and returns
its `.kmm` bytes, so it can fetch and decode them itself.

A pack carries the licence of the animation it was baked from, in its `source` and `license`
notes. Don't put a pack built from licensed third-party data at a public URL unless
that licence allows it.

The clips should cover:
- idle;
- walk and run loops;
- starts, stops, turns and pivots;
- for the course: traversals (hurdles, vaults, mantles, climbs), a fall loop and landings, named
  as `actions` in the bake config. Without them the course only blocks the way.

Tag them `idle`, `walk` and `run` for the gait filter (see the bake config). The walk and run
paces default to 2 and 5 m/s; set `walk=` and `run=` in the URL to match your data.

## A second body

A character pack (`www/pack/hero.kmm`, or `?hero=<url>`) is optional. It holds a mesh rigged to the
same skeleton as the motion pack, with its own proportions and textures. `kansei-anim-bake` makes
one from a glTF mesh and its textures (`character` config).

- Loaded, it is shown by default. C switches between it and the motion pack's own mesh, and
  `?char=hero` or `?char=mannequin` picks one.
- A motion pack may then ship without a mesh of its own (no `MESH` section): the character pack's
  body is the only one.
- The pose is retargeted onto its skeleton (`animation::retarget`: rotations as they are,
  translations oriented and scaled to its bones), and foot locking works on its legs.
- Its colour, normal (tangent space, +Y up) and occlusion/roughness/metallic textures are
  decoded from the pack and mipmapped on load.

## The course

Boxes stand around the start, some turned so you can take them at an angle:
- low rails (0.5 to 1 m) to hurdle, and a thin wall;
- boxes (about 1 m high, 1 m deep) to vault;
- blocks (1.2 to 1.5 m) to mantle onto, and walls (2 and 2.4 m) to climb;
- long narrow beams, too narrow to stand along;
- two stacks of blocks, one on top of another: mantle onto the first, then onto the second;
- a 0.3 m step, walked up without a traversal;
- two 1.2 m platforms 1.5 m apart: a running jump crosses the gap, a walking one falls short.

Space (A) traverses what is ahead, along the way the character moves (else where it faces)
(`motion_matching::traversal`):
- the kind comes from the obstacle's shape: its height, its depth and whether there is room to
  stand on it or beyond it;
- the clip and its start frame come from the pace and the distance to the ledge, among the
  clips that leave room where they end;
- the clip's root motion is warped so its ledge lands on the real one, at the real height.

Pressed a little early, Space waits up to a second for the obstacle to come in reach. With
nothing to traverse ahead (nothing there, or something too high, too narrow or with no room on
it), Space jumps:
- a jump clip whose pace and pose fit plays up to its take-off;
- from there the flight is ballistic, with the run-up's momentum and the controller's gravity,
  kept out of the boxes and landing wherever it comes down, box tops included;
- the jump clip, then the fall loop, animate it in the air, and the landing is chosen by the
  height of the fall (light or heavy) and the pace (into a stand, a walk or a run).

Walking off a top falls the same way. The HUD's `state` line shows what the character is doing,
and why the last Space was refused ("too high", "no room to land"…).

## The lake

East of the course lies a small lake (`lake=0` leaves it out; `at=14,-1,90` starts on its
shore): the [lake example](../lake/README.md)'s SPH water, with its water cannon and water mill.
Everything about the water, the cannon, the mill and the P panel is in that README; here the
character is in it:

- The character wades: the bed is in the collision world, shelving from the waterline to about
  0.6 m deep, so it walks down into the water and out again.
- Its thighs, shins, feet and hips are capsules (`FluidColliders`). Walking pushes a wake and
  ripples ahead of the legs, running throws the water up, and water pushed onto the shore drains
  back.
- Landing in the water (a jump or a fall) throws a crown of spray, scaled by how fast the
  character came down: a sphere at its feet pushes the water out for a moment
  (`FluidCapsule::expansion`).
- Legs within 2 m of the lake wake its water from rest.
- **The cannon** (`at=14.4,-3.9,10` starts beside it) fires only with the character within 2.5 m
  of it: E, the gamepad's X, or a click on its prompt; R (Y) by it drains the lake.
- **The mill** (`at=18,2.6,110` looks along the shore at it) turns in the north shallows.

## The room

`www/room.html` (`start_room`) puts the character in a 40 x 40 m room, 9 m high (`src/room/`). It
loads the same packs as `www/index.html` (`www/pack/`, `pack=`, `hero=`). Without one a capsule
stands in for the character (drawn by the same skinned material, in the mirrors and the shadows)
under the HUD's "no motion pack" message, and the room renders as it does with one.

- **The room.** A white marble floor of 1.8 m slabs (each its own piece of the stone, turned and
  offset, with seams) or a lacquered oak herringbone (`floor=wood`), plaster walls (a terracotta
  one to the south, grey limewash to the north), three tall steel-framed windows to the west with
  deep reveals, a 6 x 12 m skylight in a concrete ceiling crossed by oak beams and a steel grid,
  four concrete columns, oak baseboards (`layout.rs`). Surfaces are box-mapped PBR (`pbr.rs`:
  colour, normal, occlusion/roughness/metallic) with a slow noise over metres against the
  tiling.
- **The light.** A low sun (9000 lx at 24°, 3600 K) through the windows, the skylight (a cool
  7500 K spot light whose emitter is the 6 x 12 m opening, 32000 cd), four warm 2700 K lamps
  under glowing shades (three floor lamps and the dining table's pendant), a narrow display spot
  on the dragon from a beam 12 m behind it (1.5 Mcd, 11°, about 7000 lx on the plinth: a
  theatre's follow spot, else its pool and the dragon's caustics drown in the skylight's; a 3 cm
  emitter: sharp shadows and caustics; `display_spot`, `display_cd`), low enough that the dragon's
  light falls forward on the plinth's top, and the sky the
  windows show, which also lights the rays leaving through them (`SkyLighting`, its SH built on
  the CPU from the gradient). Exposure EV100 9, AgX.
- **Ray-traced direct light** (`kansei_core::rt::RtShadowsEffect`, new with this page). Surfaces
  that leave their direct light to it (`kansei_gbuffer_out_rt_lit`: the emissive alpha holds their
  roughness) are lit by the sun and every spot light with shadow rays through the grid of
  triangles: toward a point of the sun's disk (`sun_soft`, 1.2° by default), of a spot light's disk
  or of a rectangle (the skylight, `set_rect_emitter`), so penumbras widen with distance from the
  occluder as an area light's do. One ray per light per 2 x 2 pixels (`shadow_res=full` for every
  pixel), accumulated over frames and filtered (an à-trous wavelet with depth and normal stops),
  upsampled by depth and normal; then Lambert plus a GGX specular toward the emitter's point
  nearest the reflected ray (on surfaces the traced reflections don't cover). Short screen-space
  rays (`contact_length`, 25 cm) add the contacts the 30 cm grid misses: chair legs, the feet.
  `shadows=maps` goes back to shadow maps (cascades and the spot lights' PCSS atlas) in the
  materials. Its debug views (`rt_view=visibility|direct|mask`) show one light's visibility, the
  direct light alone and which pixels it lights.
- **The GI.** The hybrid by default (`RtDiffuseGiEffect`: a ray for each 2 x 2 pixels through the
  grid, the hits lit by the lights, shadowed by the maps, `gi_shadows=rays` for rays, and a voxel
  cone, SVGF 3 x 3), over voxel GI's volume of the room. G (or `gi=`) cycles the others: voxel
  cones (`voxel`), voxel cones with screen-space GI in front for the near field (`voxel+ssgi`,
  `VoxelGIOptions::near_field`), irradiance probes traced in the volume's distance field
  (`probes`, `SceneVoxelGi::enable_probes`; `probe_spacing` in voxels, 6 by default, and
  `probes_per_frame`, 0 for all, in the panel), the probes with the screen-space near field
  (`probes+ssgi`), screen space alone (`ssgi`, dark here: it bounces only what is on screen) and
  none. All read the same exposure. The volume sees the window sky past it
  (`use_sky_lighting`), for the cones and the probes' rays that leave through the windows. The
  probe lattice puts probes in the volume's half-metre margin outside the one-sided walls and
  above the ceiling, where they see the sky; at the default backface limit (0.25) the lookups
  reached them through the one-voxel shell and lit a band along the top of the walls. Those
  probes see the shell's back faces, so the room drops them at 0.1. The room's materials add no
  sky ambient of their own, so the effects take none out (`ambient: 0`). In the probe modes the
  volume is lit every third frame (`SceneVoxelGi::lighting_interval`: the probes blend its light
  over tens of frames anyway) and the probes are looked up at half resolution and upsampled
  (`Probes/Apply`; their update is `Probes/Update`): the GI's passes 2.0-2.5 to 1.1 ms with
  `probes`, 3.8-4.1 to 3.0-3.2 with `probes+ssgi` (the screen-space trace 1.2 of it).
- **Reflections and glass.** The marble, the lacquered wood, a mirror on the north wall and a
  brushed one (roughness 0.1) on the east wall reflect through `RtReflectionsEffect`, and the glass
  Stanford dragon (index of refraction 1.5) through its glass pass. Only glossy pixels (roughness
  under 0.45) take traced reflections: a rough lobe's few rays come out as speckle, and its blur is
  what the lights' GGX and the GI give anyway. A reflected point the camera also sees takes the
  lit image there (sharp, with every light and the GI); the effect checks it is the same surface
  (position and normal), and where something thin in front hides it (a leg over the floor) it
  takes the same surface a few pixels aside, so the floor in the mirror doesn't streak behind
  each leg. The other hits are lit by the lights (shadow rays) and voxel GI, textured: the
  surfaces' colour maps and the hero's base colour are drawn into a 256-texel array
  (`assets::albedo_array`), each surface's grid triangles carry their box-mapped uvs and a layer
  (`RtSurface::with_albedo_layer`), and `RtReflectionsEffect::set_albedo_texture` reads it, a
  hit's albedo the surface's mean times the texel over the layer's mean. The surfaces' means are
  measured from the textures at start, for the grid and voxel GI alike.
- **The water, ray traced** (`fluid_render=rt`, the default; `screen` draws it with the old
  screen-space composite). The pond's surface (its marching-cubes mesh) goes into the GBuffer as
  glass (`FluidSurfaceEffect::surface_renderable_glass`, index of refraction 1.33, a turquoise tint
  a metre, small wind ripples on its normal, `fluid_ripples`), and the glass pass refracts through
  it like the dragon: Beer-Lambert in the water to the bed, the plinth or the legs in the grid.
  What it finds there lights itself (the bed and the plinth, `RtSurface::with_self_lit`), so the
  pass reads it from the GBuffer's background image (the scene before the water was drawn) plus a
  wide voxel cone for its indirect light, not tracing its lights. The water is in the grid too, as
  glass: `FluidHeightfield` finds the top of the fluid in each column of its density field (spray
  left out) and writes a mesh 30 cm apart over it (`Renderable::rt_mesh`), coarse because the
  grid's cost is triangles a cell, marked `RtSurface::with_approximate` so the glass pass doesn't
  bend twice at it; the mirrors show the water through it, and the caustics' photons refract at
  the fine height field. Its reflection is traced with the other surfaces' (`RtGlass::
  reflections_in_trace`: half resolution, accumulated), and the glass pass traces half the glass
  pixels a frame (`RtGlass::checkerboard`, the others from the traced ones beside them):
  together they took the glass pass from 34 to 4.6 ms in `dragon`. The bed is pale marble
  (`fluid_bed`: `gravel`, or `blackmarble` for the reflecting pool it was), where the caustics
  show.
- **Caustics** (`kansei_core::rt::RtCausticsEffect`, new with this page; `caustics`). Every frame
  each light (the sun, the spot lights) sends a grid of photons (`caustics_photons` a side) at the
  dragon; those that hit glass first are refracted through it (Fresnel, absorption, total internal
  reflection) and add their light where they land on a 3 cm map over the pond, and the same
  photons taken straight on take it away where the direct light lands, so the map holds what the
  glass changes: the light it focuses on the plinth and the shadow it leaves. It is blended over
  frames (`caustics_blend`) and added to the surfaces under it that face up (on screen, and at the
  reflections' and the glass's hits). Physical, the glass's light is a faint dappling (10-30% over
  the spot's own): `game` takes it 3x as bright with its splats half as wide
  (`caustics_glass_gain`, `caustics_sharpen`; `RtCausticsEffect::focus_gain`, `focus_sharpen`).
  The water's caustics are a game's pool's (`kansei_water_caustic_lines`, `kansei_rt_caustics_focus`):
  a net of bright curved lines over what lies under the water, two animated cellular layers bent
  by slow waves, the light between them at 0.45 and on them up to 4x (the mean about kept); its
  ripples' own focus (the refraction's Jacobian, `kansei_water_focus`) was too soft to read on the
  pale bed. `caustics_water_photons` sends photons through the water instead, which under the
  skylight's 6 x 12 m emitter blur to almost nothing (physically). `cam=caustics` looks down on
  the plinth's top and the bed with both. The map costs 0.5 ms (fed every other frame,
  `caustics_every`; four times as seldom with the pond out of view), the lines up to 0.7 in the
  glass pass where the bed fills the screen.
- **The water's surface and box** (the panel's Fluid → surface resolution & box; `room.info().fluid`
  reads back what it is built at). The surface is interpolated marching cubes over a surface field
  of `fluid_res` voxels along the box's longest side (256: 193 x 32 x 256 voxels of 5.7 cm over
  the 11 x 1.8 x 14.6 m box), at most `fluid_mc_max` triangles (500000), a lone particle a droplet
  of `fluid_drop` (4.1 cm) and each splatted over `fluid_splat` (13.6 cm, the smoothing); the
  ray tracing grid's copy is a height-field mesh `fluid_rt_spacing` apart (30 cm, 37 x 49
  vertices). The box (`fluid_min_x` ... `fluid_max_z`, world metres, the pond's own unless the URL
  names it; `fluid_bounds_show` draws it) is the simulation's bounds and the surface's. A change
  makes the field, the marching cubes, the height field and the grid's copy again, live (a slider
  sends when let go); a new box also refits the simulation's neighbour grid and fills the pond
  again with the water within it (squeezed in instead, the water outside blew up). The droplet
  and splat radii are live. Cost with the water running (`dragon`, `fluid_rest=0`): 128 voxels
  24.8 ms, 192 25.3, 256 27.1, 320 28.7, 384 32.7; the grid's mesh at 15 cm +0.6 ms (its triangles
  in the reflections' and the glass's traces), 50 cm the same as 30; the triangles' cap costs
  nothing until the surface needs them. At rest (the default) the surface isn't extracted, so
  its resolution costs nothing per frame.
- **Glass and water quality** (the panel's preset, `quality=`). `game` (the default): the
  checkerboard, the glass's reflections at half resolution, the water's caustic lines, the glass's
  caustics 3x and sharper, photons from the lights' centres. `high`: every glass pixel and its own reflection, photons
  through the water from the lights' emitters, four times the photons.
- **Furniture** from Poly Haven, CC0: a sofa with throw pillows, two lounge chairs, an arm chair,
  an ottoman, a coffee table, a cabinet, a round dining table with four chairs, a vase, steel and
  wooden shelves with books, plants, a picture frame, a pendant lamp. Each model is drawn as it
  comes, and traced as a simplified stand-in (`assets.rs`: welded, simplified to 1.5 cm, at most
  1500 triangles, in the grid only): the chairs alone put thousands of triangles in a 30 cm cell,
  which made the rays' traversal five to eight times slower (the room's grid went from 196k
  triangles to 24k). The solid pieces are boxes in the collision world.
- **Parkour.** A crate, a rail, a sideboard, an island, dark marble blocks, a two-block stack, a 2.2 m platform behind a
  step, two ledges with a gap, a bench: Space vaults, hurdles, mantles and climbs them as on the
  course. `room_check()` probes each piece as the traversal does and says whether it gets the
  traversal meant (vault: the crate, the sideboard, the island; hurdle: the rail, the bench;
  mantle: the block, the stack, its top, the ledge; climb: the platform from its step). All ten
  pass.
- **One-sided walls.** The walls, the floor and the ceiling face in and discard their back faces,
  so from outside (O, or `cam=outside`) the camera looks through them into the room. The grid hits
  a triangle from either side, so the rays from inside meet the walls as they should. Thick boxes
  outside keep the character in.
- **The character in the rays.** The grid takes geometry as built, not as the vertex shader skins
  it, so a low-poly copy of the body shown stands in for it there (`look::Proxy`, `grid_only`):
  each body's mesh (the hero's, or the pack's mannequin) simplified to 1500 triangles
  (`proxy_tris=` another count), keeping original vertices so their joints and weights still
  apply, and skinned on the CPU with the shown body's palette whenever a bone has moved 3 cm
  (`Geometry::revision` tells the grid). C switches the copy with the body. The hero's copy
  carries its uvs and the base colour's layer, so the mirrors show its jacket, face and jeans: it
  is simplified with its uv seams kept (`assets::simplify_seamed`; welded by position, the
  triangles along the seams read the texture across the atlas and the mirrored face and jacket
  came out mangled), letting a seam move only where the uvs on both sides agree, which reaches
  1500 at the same cost (kept fixed, the seams stop it at about 2250, 1.5 ms more). The
  mannequin's is welded, one colour with smooth normals. `proxy_view=1` draws the copy for the
  camera beside the body, textured as the rays see it. They cast the ray-traced shadow, show in the
  mirrors and the glass, and occlude the GI's rays. Walking, the grid rebuilds every frame
  (about 0.8-1.6 ms); standing, never. `rt_body=0` leaves the copy out.
- **The pond.** A rectangular pool with a pale marble bed round the marble plinth, the
  lake's SPH water at its scale and tuning, 180k particles. The character wades in as it does in
  the lake, and the water rests: culled out of view, asleep once its fastest particle is under
  0.15 m/s. The plinth is a still capsule collider, not part of the container's floor (as a floor
  that steep it kept the water churning).
- **Fog and dust.** A faint haze fills the room (a box `LocalFogVolume`), lit by the sun through
  the cascades and the skylight through its map, so the windows throw shafts. 65536 dust motes
  fill the room and drift in a slow curl flow (a compute shader), pushed by the character, drawn
  from a sprite atlas a compute shader makes at startup (soft motes, fibres, flecks and bokeh
  discs, with mips), tumbling, blurred into bokeh by the depth of field. They are lit only by
  direct light through its shadows, so they show in the shafts and vanish in shade: the sun's
  through the windows and through the skylight (the brightest, looking toward the sun), and the
  core of each spot light's cone (the skylight's column, the lamps' pools; a wide spot lighting
  every mote read as snow). `dust_bright` (2.5) is radiance per lux before the phase function,
  calibrated at EV 9; `dust_count=` sets how many.
- **Post.** TAA, depth of field focused on the character (or the screen's centre, or a distance),
  bloom, AgX tone mapping with white balance, contrast, saturation, lift/gain, vignette, grain and
  chromatic aberration, motion blur off by default.

The panel (Tweakpane, P hides it, collapsed on small screens) holds every setting, its starting
values from the URL: presets for the look (golden hour, midday, overcast, night lamps, gallery),
the camera, the depth of field and the post-processing (neutral, warm cinematic, cool moody,
filmic contrast, bleach bypass, vintage); a preset sets several controls and editing one makes it
"custom". Its folders: Scene (floor, roughness, mirrors, the dragon's glass), Lights, Shadows / GI
/ reflections, Fog and dust, Fluid (the pond: simulate, show, sleep, SPH or PBF, the fill with a
Reset button, time scale, substeps, the solvers' parameters, gravity, the legs' push, the
surface's look; all live but the fill), Depth of field, Post-processing, Resolution (the canvas
height, native to 540p, the pixel ratio and the scene's scale, TAA upscaling; kept in the URL)
and Stats. `src/room/settings.rs` lists them all with their defaults: each is also a URL
parameter of the same name (`?floor=wood&gi=voxel&ev=8.5`), and `room.set(name, value)` sets one
live. `cam=` names a view: `outside`, `hall`, `dragon`, `mirror`, `windows`, `parkour`, `living`,
`dining`, `library`, the shadow close-ups `contact`, `feet`, `plinth`, and `caustics` (any other follows the
character). `panel=0` and `hud=0` start without the panel and the text, for captures; `refl_screen=0` lights every reflected hit by the traced path (no lit-image lookups) and `refl_indirect=` scales the voxels' light at those hits with the hybrid GI on; `vox_sky=0` leaves the window sky out of the voxel volume; `pond=0`,
`dragon=0`, `rt_body=0` leave those out; `lod_px=` is the dragon's cluster LOD error on screen (pixels, 1 by default; 0 the whole mesh) and `grid_lod_cm=` its cut's in the grid the glass refracts through (3 cm: about 3 900 of its 19 300 triangles); `perf=1` (or the backquote key, or Stats → performance overlay) shows kansei_wasm's performance overlay (fps, CPU and GPU ms with graphs, draws, triangles, pipelines, GPU memory; click it for every pass, the draws and triangles by pass and by material, memory by label and pipelines by material; `kanseiPerf.report()` gives them as data); `stats=1` shows each pass's GPU time and the CPU sections
(also in `room.info()`), `profile=1` logs the profile every 3 s.

Keys, on top of the shared ones: O inside / outside, G the GI, T the shadows, F the fog, N the dust,
P the panel. On `window.room`: `info()`, `settings()`, `set(name, value)` and `check()`.

### The assets

`www/assets/room/` (25 MB: 10 surfaces, 18 models) is made by `tools/room_assets.py`, which
downloads them and encodes every texture to KTX2 (Basis Universal: colour as ETC1S, normals and
occlusion/roughness/metallic as UASTC) with `rust/tools/ktx2`; the models become .glb files whose
textures use `KHR_texture_basisu`. Each texture is uploaded once and shared by the materials that
use it (`assets.rs`). All CC0:

- surfaces from [ambientCG](https://ambientcg.com): Marble012 (the floor), WoodFloor016 (the
  herringbone), Plaster001, Concrete031 (the columns), Concrete046 (the ceiling), Wood049 (oak),
  Carpet012, Marble016 (the pond's bed, the blocks), Gravel043, Metal032;
- models from [Poly Haven](https://polyhaven.com): sofa_02, mid_century_lounge_chair,
  modern_arm_chair_01, Ottoman_01, modern_coffee_table_01, modern_wooden_cabinet,
  wooden_display_shelves_01, steel_frame_shelves_03, round_wooden_table_01, dining_chair_02,
  potted_plant_02, potted_plant_04, ceramic_vase_01, ceramic_vase_03, throw_pillows_01,
  hanging_picture_frame_02, book_encyclopedia_set_01, modern_ceiling_lamp_01.

The dragon is the GI box's (`www/assets/` links to `examples/gi-box/www/assets/`): "Stanford
Dragon (Vrip)" by 3D graphics 101, CC-BY-NC-4.0. The page credits all three.

### What a frame costs

1080p (`dpr=1`), M4 Pro, headless Chrome without vsync (`stats=1`), the defaults (hybrid GI,
ray-traced shadows and reflections, fog, dust, depth of field, TAA, bloom), the water asleep,
measured with the GPU otherwise idle (other sessions sharing it doubled every pass):

| | frame interval | GPU span |
|---|---|---|
| `hall` / `living` / `dining` / `mirror`, no pack | 16.7 / 15.9 / 16.4 / 15.9 ms | 16.3 / 14.9 / 15.8 / 15.1 ms |
| `dragon` (the glass fills the screen) | 20.6 ms | 20.1 ms |
| `outside` | 14.7 ms | 13.1 ms |
| GASP character in `hall`, standing / walking | 17.1 / 18.5 ms | 16.5 / 17.6 ms |
| the follow camera, walking | 20.3 ms | 19.8 ms |
| `hall`, GI hits shadowed by rays (`gi_shadows=rays`) | 20.1 ms | 19.8 ms |
| `hall`, shadow maps (`shadows=maps`) | 16.9 ms | 16.2 ms |
| `hall`, voxel GI / screen-space GI / no GI | 14.7 / 14.3 / 11.7 ms | 14.0 / 13.3 / 11.0 ms |
| `hall`, interleaved in one page with the GPU shared (every pass about 1.5x): rt / voxel / voxel+ssgi / probes / probes+ssgi / ssgi / off | 26.5-27.1 / 24.0-25.4 / 25.8-26.1 / 22.4-23.7 / 24.1-25.1 / 24.3-25.1 / 21.0-23.0 ms | |
| `hall`, the water running (`fluid_rest=0`) | 21.2 ms | 20.7 ms |
| `hall`, every ray at full resolution | 31.0 ms | 30.5 ms |
| GASP hero beside the north mirror (`at=-2,-17,230`): capsules / 1500-triangle copy / 3000 | 17.0 / 17.9 / 19.1 ms | |
| `hall` at 720p | 10.8 ms | 10.2 ms |

The ray-traced water and the caustics (`game` quality, the defaults), against the build before
them (screen-space water, no caustics, black bed), each camera preset in one session, frame
interval in ms, no pack:

| | outside | hall | dragon | mirror | windows | parkour | living | dining | library | contact | feet | plinth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| before | 14.1 | 16.7 | 19.1 | 14.9 | 14.8 | 14.1 | 15.0 | 15.7 | 14.1 | 17.8 | 18.9 | 22.3 |
| after | 14.5 | 17.6 | 21.8 | 15.8 | 14.9 | 15.4 | 15.9 | 16.4 | 15.2 | 19.1 | 19.6 | 24.5 |

What it adds: the caustics 0.3-0.5 ms (the photons every other frame, the composite every frame),
the water's reflection in the reflections' trace (up to 1.5 ms where the water fills the screen),
its refraction in the glass pass (0.5 ms in `dragon`), the display spot's light. `quality=high`
(every glass pixel, its own reflection, photons through the water) lands at 34 ms in `dragon`
(the glass pass 16.8 ms, the photons 1.8 against `game`'s 22 ms and 4.7); the first version,
before the approximations, had the glass pass at 34 ms there. The water's caustic lines, the glass's gain and the
follow spot (later, caustics on / off in one page): `dragon` 21.9 / 21.4 ms, `plinth` 24.3-25.4
/ 24.7, `caustics` 25.9 / 24.7, `hall` 17.5 / 17.2.

By effect in `hall` (GPU): the hybrid GI 4.6 ms (its trace 3.0), the renderer's passes 2.2 (the
GBuffer, the cascades, the skylight's shadow map, velocity), the ray-traced direct light 2.1
(trace 0.7-1.4, temporal and wavelet 0.6, composite 0.8), the reflections 1.4, the glass 1.3, depth
of field 0.7, TAA 0.6, voxel GI's upkeep 0.5, fog 0.5, bloom 0.3, the dust 0.01 of compute and a few tenths of a millisecond to draw. On the
CPU the character takes about 2.4 ms and the renderer about 2.

What it took to get there, from a first pass at 92 ms: the furniture's stand-ins in the grid (the
GI's trace 22 to 3 ms, the reflections' 31 to 1.5), the lamps without shadow maps when the
shadows are ray traced (they had redrawn the room four more times a frame, 3 ms), the GI's hits
shadowed by the maps (6.2 to 3.0 ms), and a shallower pond that sleeps. The Resolution folder's
scene scale (0.75: 15.2 ms in `hall`) or 900p (16.4 ms) buy the rest on slower GPUs.

Then the draws. The overlay's breakdown showed 2251 draws and 4.0 million triangles a frame, 85%
of the draws and 82% of the triangles in the shadow maps: the sun's four cascades and two spot
lights each redrew every caster (317 draws, 550 000 triangles), and Poly Haven's book set, 68 000
triangles in 40 meshes, four on the shelves, was 1120 of the draws and 1.9 million of the
triangles on its own (the dragon is 19 300 triangles, drawn once, casting no shadow map). Every
part was its own material with its own shader and pipelines: 1411 pipelines. Now each model is a
mesh per material, the architecture's boxes and panels a mesh per surface (`Builder::batch`), the
furniture casts its shadows as a simplified copy (`Renderable::shadow_geometry`, within 6 mm),
each shadow map draws only the static casters in its frustum, and materials with the same WGSL
share their program and pipelines (`SharedLayouts::programs`): 315 draws, 0.92 million triangles
(the GBuffer's 0.7 million of them) and 130 pipelines, `hall` 18.2-18.5 to 14.7-15.9 ms and the
user's view (`gi=probes+ssgi`) 12.0-15.4 to 10.2-10.5. Cluster LOD for the dense furniture was
slower (17.7 against 13.2 ms: 300 more cull dispatches a frame, 93 MB of cut buffers), so the
camera draws the full meshes. The dragon in the grid at a cell's error was 300 triangles, a lump
through the glass: within 3 cm it is 3 900, 0.5-0.7 ms more glass.

## Build and run

```sh
cd rust/kansei-wasm/demos/motion-matching
wasm-pack build --target web --release
python3 -m http.server 8080   # then open http://localhost:8080/www/ (or www/kimodo.html, www/room.html)
```

The crate builds with WASM SIMD (`.cargo/config.toml`), since the search runs on the CPU.
`scripts/build-wasm-examples.sh` skips it: its packs never ship.

## Controls

| | keyboard / mouse | gamepad |
|---|---|---|
| move (relative to the camera) | WASD, arrows | left stick (tilt sets the pace) |
| run | Shift | B, right trigger |
| jump, or traverse what is ahead | Space | A |
| strafe (face the camera's way) | Q | left bumper |
| fire the water cannon (by it; hold to pour) | E, or click the prompt | X |
| drain the lake (by the cannon) | R | Y |
| orbit, zoom | drag, wheel | right stick |
| overlay (trajectory, feet, HUD) | B | |
| skeleton, mesh, foot locking | K, M, L | |
| character (with a character pack) | C | |

In the overlay:
- blue boxes are the simulated character now and at ⅓, ⅔ and 1 s ahead;
- the small boxes under the feet grow while a foot is planted and locked;
- the red bar and post mark the last ledge found, and its height.

URL parameters:
- `hero=<url>` loads that character pack, `hero=none` none (the default is `pack/hero.kmm`);
- `gait=0` searches every clip, instead of idle + walk or idle + run by tag;
- `taa=0` turns TAA off;
- `course=0` leaves the boxes out;
- `lake=0` leaves the lake out;
- `rest=0` never rests the lake's water (always stepped and drawn, as before resting);
- `mill=0` starts the water mill stopped;
- `debug=1` allows `lake_regions()` (see the lake's README);
- `at=<x>,<z>,<degrees>` starts the character there, facing that way (0 is +Z).
- `drive=1` drives a fixed route instead of the player (starts, walks, a turn, stops, a run, a
  turn and a stop at a run, pivots, strafes, walking backwards), round and round: two packs get the
  same input, to record them side by side. `drive_restart()` (on `window` as `driveRestart`) starts
  it over.
- `circle=<radius>` runs round a circle of that radius (metres; negative turns right) at the run
  pace instead, as the foot-slide course does (see [Foot slide](#foot-slide)).
- `play=<pattern>` plays the pack's clips whose names start with the pattern (`*` any run of
  characters, e.g. `play=Parkour/*_00`; several patterns separated by commas) one after another, as they are, each from the start point:
  for clips the search never picks (generated ones, see
  [`genanim`](../../../kansei-anim-bake/genanim/README.md)). `drive_restart()` starts them over.
- A page can drive these at runtime: `clip_names()` lists the pack's clips, `play_clips(pattern)`
  plays them from where the character stands (`""` gives it back to the player), and
  `set_drive(on)` starts or stops the `drive=1` route there (`www/kimodo.html` builds its clip
  browser on them).
- `view=<degrees>` turns the camera round the character from behind it (90: its left side).
- `profile=1` logs each labelled GPU pass's time (the fluid's included) to the console every
  3 s (`Renderer::set_profiling`).
- The lake's exports (the P panel's, `cannon_prompt()`, `cannon_fire(down)`, `lake_fill()`,
  `lake_regions()`) come from the lake crate and are in this module too: see the
  [lake's README](../lake/README.md).

## Foot slide

Foot locking (L) pins a planted foot where it touched down and bends the leg to it with two-joint
IK: on its ankle while the heel is down, on its ball once the heel lifts (a running foot lands
on its ball and rolls off it). A foot counts as planted while the pack's contacts say so: its
ankle or its ball low and under 1 m/s in the clip (`ContactThresholds`; packs baked before that
rule get their contacts found again on load). The pin lets go when the contact ends or the
animated foot strays 0.3 m from it, and pins again where the animation has the foot while the
contact lasts.

To measure it, headless, on a pack:

```sh
cd rust
cargo run -p kansei-core --release --example foot_slide -- <pack.kmm> [hero=<character.kmm>]
```

It plays a scripted course (straight walk and run, run circles of 2, 3.5 and 5 m and walk circles
of 1.5 and 3 m both ways, starts and stops, 180° turns; `animation::motion_matching::foot_slide`)
and prints each scenario's planted-foot slide (cm per second planted, cm per plant) and how far
the character faces from the simulation. `circle=<radius>` on this page runs the same circles,
to watch them. The tightest circles and 180° turns still slide where the database has no clip
for the turn: the character follows the simulation through clips that turn less.

## Timing against the TS port

The TS engine has a pure TypeScript port of this runtime (`src/animation/motion_matching`,
`src/collision`) and a page like this one on it, `examples/index_motion_matching.html` (packs in
the gitignored `examples/pack/`, or `?pack=<url>`). Both pages time the same things the same way
(`src/timing.rs` here), for the same pack:

- the HUD's `update` line: the character's update per frame (search, pose, inertialization,
  foot locking, collision), mean, 95th percentile and largest over the last 3000 frames;
  `window.motionTimings()` returns them as JSON (the TS page also splits off the searches);
- `window.benchSearch(step)`: milliseconds per `Database::search` over every `step`th frame's
  features, nudged, as queries;
- `window.benchPose(count)`: milliseconds to sample a pose between two frames, run its forward
  kinematics and fill the first body's bone palette, `count` times.

Run both with `drive=1` (the same route, `driveRestart()` starts it over) and `lake=0` here (the TS
page has the course and no lake), alternating the two pages: other sessions share the machine.
Serve them cross-origin isolated (COOP `same-origin`, COEP `credentialless`), or Chrome rounds
`performance.now()` to 0.1 ms.

## Generated animation (Kimodo)

`www/kimodo.html` plays packs of generated clips: NVIDIA Kimodo, baked by
[`genanim`](../../../kansei-anim-bake/genanim/README.md), to feel how they play under the stick.
It plays like `www/index.html` (keyboard or gamepad, the course, the lake; see the controls) and
adds a panel (P hides it) to switch packs and to watch the custom clips. It differs in one thing:
it loads a character pack only when the URL names one (`hero=<url>`; else it adds `hero=none`),
since the generated packs carry their own SOMA body.

### The packs

They are private and stay out of git (`www/pack/` is gitignored). Link them in:

```sh
rust/kansei-wasm/demos/motion-matching/link-packs.sh [data]   # default ~/Documents/dev/kansei-private-data
```

That links `<data>/genanim/pack` as `www/pack/gen` and `<data>/gasp/pack` as `www/pack/gasp`
(inside `www/pack/`: if that is itself a link to your packs folder, the links land in that
folder). The panel marks the packs that aren't linked, and opening one shows how to link it.

| `?pack=` | pack | body | paces (walk, run m/s) |
|---|---|---|---|
| `dance` (the default) | `gen/gen-dance.kmm`: the dance card | SOMA | 2, 5 |
| `all` | `gen/gen-all.kmm`: the dance card and the custom clips | SOMA | 2, 5 |
| `limp` | `gen/gen-limp.kmm`: limping on the left leg | SOMA | 1.2, 3 |
| `gasp-gen` | `gen/gasp-plus-gen.kmm`: GASP with the generated clips retargeted | mannequin (C: hero) | 2, 5 |
| `gasp` | `gasp/gasp-locomotion.kmm`: GASP only, for reference | hero (C: mannequin) | 2, 5 |

A pack id is written out into the pack's URL and its parameters (`walk=`, `run=`, `hero=`,
`char=`); switching packs reloads the page with them and keeps the rest of the URL (`course=0`,
`lake=0`, `view=`, `at=`…). `pack=<url>` loads any pack, as `www/index.html` does.

GASP packs (and their renders) are under GASP's licence: keep screenshots and recordings of them
private.

### The clip browser

Under **clips**, the custom clips (push, drag, car and parkour, in `all` and `gasp-gen`) by kind:
pick a clip and a sample (or all of its samples) and play it, or every clip of the kind. They play
as they are, root motion included, one after another from where the character stands, with the
clip's name on the HUD (`play_clips`, as `play=<pattern>` does at start). A `play=` field takes any
pattern (`*` any run, several separated by commas). **drive the route** drives `drive=1`'s route
from there; **back to the stick** (or Escape) gives the character back to the player.
