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
light and global illumination, with mirrors, a glass dragon, an aquarium of SPH water, furniture to vault and
climb, fog and dust (see
[The room](#the-room)). The character and its controls are shared by the pages (`src/character.rs`,
`src/player.rs`); the room is `src/room/`.

`www/showroom.html` is a fourth, the room's renderer in a white gallery of furniture pieces, the
character shown as a mannequin (see [The showroom](#the-showroom)). The two share `www/room.js`
(the panel) and `src/room/` (`Venue`).

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

## More bodies

Character packs are optional: the hero (`www/pack/hero.kmm`, or `?hero=<url>`) and the mannequin
(`www/pack/mannequin.kmm`, or `?mannequin=<url>`; [The mannequin](#the-mannequin)), `none` for
neither. Each holds a mesh rigged to the same skeleton as the motion pack, with its own
proportions and textures. `kansei-anim-bake` makes one from a glTF mesh and its textures
(`character` config).

- The hero is shown first (the showroom: the mannequin). C cycles through the bodies and the
  motion pack's own mesh; `?char=hero`, `?char=mannequin` or `?char=pack` picks one
  (`char=mannequin` without a mannequin pack is the motion pack's mesh, as it was).
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

`www/room.html` (`start_room`) puts the character in a 32 x 28 m loft gallery, 8 m high
(`src/room/`), authored in Blender and loaded from `www/assets/room/room.glb` (see
[The room's source](#the-rooms-source)). It loads the same packs as `www/index.html`
(`www/pack/`, `pack=`, `hero=`). Without one a capsule stands in for the character (drawn by the
same skinned material, in the mirrors and the shadows) under the HUD's "no motion pack" message,
and the room renders as it does with one.

- **The room.** A polished concrete floor, white plaster walls and a red brick one to the east,
  four tall steel-framed windows to the west, a 6 x 12 m skylight in a board-formed concrete
  ceiling crossed by steel I-beams, four concrete columns; a lounge in the windows' light, a
  dining table under a pendant, steel shelves, a kitchen island, sculptures on white plinths, three
  canvases, the glass dragon on a marble drum under the skylight, the aquarium by the windows and
  a parkour course (`layout.rs` reads it all from the glb). Its 107k triangles are drawn as a mesh
  per material (47 draws), the surfaces PBR (`pbr.rs`: colour, normal, occlusion/roughness/
  metallic) on the uvs Blender gives them (tiling ones box-mapped there by a geometry-nodes
  modifier), with a slow noise over metres against the tiling.
- **The light.** A low sun (9000 lx at 24°, 3600 K) through the windows, the skylight (a cool
  7500 K spot light whose emitter is the 6 x 12 m opening, 32000 cd), four warm 2700 K lamps
  under glowing shades (three floor lamps and the dining table's pendant), a narrow display spot
  on the dragon from a beam 10 m behind it (1.5 Mcd, 11°, about 7000 lx on the plinth: a
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
  screen-space composite). The aquarium's water surface (its marching-cubes mesh) goes into the GBuffer as
  glass (`FluidSurfaceEffect::surface_renderable_glass`, index of refraction 1.33, a turquoise tint
  a metre, small wind ripples on its normal, `fluid_ripples`), and the glass pass refracts through
  it like the dragon: Beer-Lambert in the water to the tank's stones and the room behind it in
  the grid. Surfaces marked `kansei_self_lit` light themselves there (`RtSurface::with_self_lit`):
  the pass reads them from the GBuffer's background image (the scene before the water was drawn)
  plus a wide voxel cone for its indirect light, not tracing its lights. The water is in the grid too, as
  glass: `FluidHeightfield` finds the top of the fluid in each column of its density field (spray
  left out) and writes a mesh 30 cm apart over it (`Renderable::rt_mesh`), coarse because the
  grid's cost is triangles a cell, marked `RtSurface::with_approximate` so the glass pass doesn't
  bend twice at it; the mirrors show the water through it, and the caustics' photons refract at
  the fine height field. Its reflection is traced with the other surfaces' (`RtGlass::
  reflections_in_trace`: half resolution, accumulated), and the glass pass traces half the glass
  pixels a frame (`RtGlass::checkerboard`, the others from the traced ones beside them):
  together they took the glass pass from 34 to 4.6 ms in `dragon`. The tank's panes are not
  traced (two more interfaces a pixel: the glass pass took 27 ms with the tank filling the screen):
  the water's own surface refracts at its walls, a thin steel frame round it.
- **Caustics** (`kansei_core::rt::RtCausticsEffect`, new with this page; `caustics`). Every frame
  each light (the sun, the spot lights) sends a grid of photons (`caustics_photons` a side) at the
  dragon; those that hit glass first are refracted through it (Fresnel, absorption, total internal
  reflection) and add their light where they land on a 3 cm map over the floor (the glb's
  `CAUSTICS_Area`: round the aquarium, where the sun through it lands, and the dragon), and the same
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
  pale bed. `caustics_water_photons` (on by default, both presets) sends photons through the
  water: the low sun through the aquarium throws its caustics across the floor east of it
  (`cam=caustics`). The map costs 0.7-1.0 ms with the water's photons (0.5 without; fed every
  other frame, `caustics_every`; four times as seldom with the tank and the dragon out of view).
- **The water's surface and box** (the panel's Fluid → surface resolution & box; `room.info().fluid`
  reads back what it is built at). The surface is interpolated marching cubes over a surface field
  of `fluid_res` voxels along the box's longest side (256: 72 x 91 x 256 voxels of 1.6 cm over
  the tank's 1.15 x 1.45 x 4.1 m box), at most `fluid_mc_max` triangles (500000), a lone particle a droplet
  of `fluid_drop` (4.1 cm) and each splatted over `fluid_splat` (13.6 cm, the smoothing); the
  ray tracing grid's copy is a height-field mesh `fluid_rt_spacing` apart (30 cm, 4 x 14
  vertices). The box (`fluid_min_x` ... `fluid_max_z`, world metres, the tank's own unless the URL
  names it; `fluid_bounds_show` draws it) is the simulation's bounds and the surface's. A change
  makes the field, the marching cubes, the height field and the grid's copy again, live (a slider
  sends when let go); a new box also refits the simulation's neighbour grid and fills the tank
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
- **Furniture** from Poly Haven, CC0 (lounge and arm chairs, an ottoman, coffee and side tables,
  a pendant, stools, steel shelves, a cabinet, plants, vases, pillows, five sculptures), decimated
  in Blender (a live Decimate modifier each), and pieces modelled there (the sofa, the tables, the
  floor lamps, the aquarium, the plinths). Each model with more than 500 triangles has a
  `_SHADOW` copy of about 400: it casts the shadow maps' shadows and is what the ray tracing grid
  traces (24.8k triangles in all: the furniture's full meshes in the grid had cost the rays half
  again). Colliders are the glb's `_COL` boxes.
- **Parkour.** The north-east blocks (a crate, a rail, a block, a two-block stack, a 2.2 m
  platform behind a step, two ledges with a gap), the island, a bench and the dragon's drum:
  Space vaults, hurdles, mantles and climbs them as on the course. The south-east is the
  [parkour course](#parkour), tagged in the glb (`kansei_traversal`): four platforms with
  1.0-1.4 m gaps to jump (`platform`, a chain in order), a 2.8 m tower whose top edges are ledges
  to hang from and shimmy round (`ledge`), a 5 m wall to run along (`wallrun`) and three bars to
  swing from (`bar`). The page reads them into the controller's parkour
  (`layout::parkour_pieces`, `ParkourFeatures::from_pieces`; the console logs how many edges,
  bars, wall faces and platforms it found). `room_check()` (`room.check()`) probes each piece as
  the traversal does and says whether it gets the traversal meant (the course's platforms:
  mantle, the tower: climb; walls and bars not checked); all pass. `room_parkour()`
  (`room.parkour()`) lists every tagged piece with its bounds and properties.
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
- **The aquarium.** A 1 x 4 m tank on an oak and steel stand by the west windows, its water the
  lake's SPH at its scale and tuning (30 780 particles, 85% full), the container the glb's
  `AQUA_Water` box. It rests: culled out of view, asleep once its fastest particle is under
  0.15 m/s. Running (`fluid_rest=0`) it costs about 1 ms of simulation a frame and its surface
  extraction.
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
"custom". Its folders: Scene (the floor's roughness, mirrors, the dragon's glass), Lights, Shadows / GI
/ reflections, Fog and dust, Fluid (the aquarium: simulate, show, sleep, SPH or PBF, the fill with a
Reset button, time scale, substeps, the solvers' parameters, gravity, the legs' push, the
surface's look; all live but the fill), Depth of field, Post-processing, Resolution (the canvas
height, native to 540p, the pixel ratio and the scene's scale, TAA upscaling; kept in the URL)
and Stats. `src/room/settings.rs` lists them all with their defaults: each is also a URL
parameter of the same name (`?gi=voxel&ev=8.5`), and `room.set(name, value)` sets one
live. `cam=` names a view, the glb's `CAM_<name>` cameras: `hall`, `dragon`, `aquarium`, `mirror`,
`windows`, `lounge`, `dining`, `gallery`, `parkour`, `course`, `outside` and `caustics` (any other
follows the character). `panel=0` and `hud=0` start without the panel and the text, for captures; `refl_screen=0` lights every reflected hit by the traced path (no lit-image lookups) and `refl_indirect=` scales the voxels' light at those hits with the hybrid GI on; `vox_sky=0` leaves the window sky out of the voxel volume; `aquarium=0`,
`dragon=0`, `rt_body=0` leave those out; `lod_px=` is the dragon's cluster LOD error on screen (pixels, 1 by default; 0 the whole mesh) and `grid_lod_cm=` its cut's in the grid the glass refracts through (3 cm: about 3 900 of its 19 300 triangles); `perf=1` (or the backquote key, or Stats → performance overlay) shows kansei_wasm's performance overlay (fps, CPU and GPU ms with graphs, draws, triangles, pipelines, GPU memory; click it for every pass, the draws and triangles by pass and by material, memory by label and pipelines by material; `kanseiPerf.report()` gives them as data); `stats=1` shows each pass's GPU time and the CPU sections
(also in `room.info()`), `profile=1` logs the profile every 3 s.

Keys, on top of the shared ones: O inside / outside, G the GI, T the shadows, F the fog, N the dust,
P the panel. On `window.room`: `info()`, `settings()`, `set(name, value)` and `check()`.

### The room's source

The room is a Blender scene kept outside this repository (its .blend, its textures and the
scripts that made it: `room_build.py` builds it, `export_glb.py` exports it; the README there
lists the contract below in full). Only the exported glb lands here, packed by
`tools/room_glb.py`: each material's occlusion and metallic-roughness maps as one ORM image, every
image encoded to KTX2 (Basis Universal: colour as ETC1S, normals and ORM as UASTC) with
`rust/tools/ktx2`, through `KHR_texture_basisu` (encoded images cached by content, so a re-export
encodes only what changed):

```sh
/Applications/Blender.app/Contents/MacOS/Blender -b room.blend --python export_glb.py
python3 tools/room_glb.py <kansei-assets>/room/export/room.glb   # writes www/assets/room/room.glb
```

What the page reads from the glb (`layout.rs`), by names and custom properties (glTF extras):

- every mesh, merged with the others of its material (and one-sided or not) into one renderable;
- `<name>_COL`: a box in the collision world (the floor and thick walls outside are added);
- `<name>_SHADOW` (a child): what its parent casts into the shadow maps and what the grid traces;
- `<name>_CUT` (Blender's boolean cutters) and `kansei: preview` (Blender-only stand-ins): skipped;
- `kansei`: `shell` (one-sided), `mirror` (`kansei_roughness`), `glass`, `aquarium` (the water's
  box, `fill`), `caustics` (the caustics map's floor rectangle), `dragon` (an empty: where, yaw,
  `length`), `spawn` (an arrow: the character's start and heading);
- materials: base colour × factor, normal map × strength, the ORM image, roughness factor, alpha
  clip for cut-outs, emission; `kansei_reflective`, `kansei_macro`, `kansei_shade` (a lamp shade
  that follows the lamp settings), `kansei_self_lit` on objects;
- lights by `kansei_light`: `sun` (its direction sets `sun_elev`/`sun_azim` unless the URL does),
  `skylight` (an area light: its `width`/`length` are the RT shadows' and caustics' emitter, the
  first spot light), `lamp` (spot lights), `display` (the dragon's spot, the last); their
  intensities and colours are the page's settings;
- cameras `CAM_<name>`: the fixed views (`distance` the orbit's);
- `kansei_traversal` on parkour pieces: `vault`, `hurdle`, `mantle`, `climb` (checked by
  `room_check()` from `kansei_from` m south of the piece at `kansei_from_y`), and the course's
  `platform` (`kansei_chain`, `kansei_chain_index`), `ledge` (`kansei_grab`, `kansei_corners`),
  `wallrun` (`kansei_run_faces`, `kansei_run_axis`) and `bar` (`kansei_bar_height`,
  `kansei_bar_axis`), listed by `room_parkour()`.

The room: 107 347 triangles (architecture 1 200, furniture 105 463, parkour 384, mirrors 48,
aquarium 252), 47 materials, 111 KTX2 images (25.7 MB; the glb 29.8 MB), 24.8k triangles in the
grid. All CC0, listed in [`www/assets/room/LICENSE.md`](www/assets/room/LICENSE.md): models from
[Poly Haven](https://polyhaven.com), surfaces from [ambientCG](https://ambientcg.com), the
canvases and the rest modelled or made for it. The dragon is the GI box's (`www/assets/` links to
`examples/gi-box/www/assets/`): "Stanford Dragon (Vrip)" by 3D graphics 101, CC-BY-NC-4.0. The
page credits them.

### What a frame costs

1080p (`dpr=1`), M4 Pro, headless Chrome without vsync (`stats=1`, `perf=1`), the defaults
(`game`: hybrid GI, ray-traced shadows, reflections and glass, caustics with the water's photons,
fog, dust, depth of field, TAA, bloom), no pack, the water asleep. The Blender room against the
procedural one it replaced, alternated in one session (the GPU shared with other sessions: tens
of percent between runs), GPU ms per frame:

| view | procedural room | Blender room |
|---|---|---|
| `hall` | 16.9-19.2 | 19.6-19.8 |
| `dragon` | 22.4-26.8 | 22.0-24.2 |
| `mirror` | 14.3-17.1 | 18.0-18.9 |
| `windows` | 13.1-14.5 | 17.6-19.1 |
| `outside` | 12.8-12.9 | 13.4-15.1 |
| draws / triangles a frame (`hall`) | 336 / 937k | 293 / 556k |
| the GBuffer's triangles | 705k | 265k |

The triangles drawn fell by 40% (the GBuffer's by 62%), but the grid is denser per cell (the room
is smaller: 24.8k triangles in 110 x 36 x 100 cells against 29.4k in 140 x 36 x 140), so the
rays cost more: the hybrid GI's trace 3.1-3.6 against 4.0-5.1 ms, the RT shadows' 0.8-0.9
against 1.4-2.4. The aquarium is most of the rest: without it (`aquarium=0`) `mirror` is
15.5 ms; in its own view (`aquarium`) it costs about 7 ms asleep (its glass pass, its photons,
the rays through the grid's water) and 15-20 ms running (`fluid_rest=0`: the SPH steps about
2-3 ms, the surface field and marching cubes, a moving grid). Its surface is 128 voxels along
the tank (3.2 cm): at 256 the marching cubes' draw alone was 3-9 ms. Before the `_SHADOW`
proxies stood in for the furniture in the grid it traced 43k triangles and `hall` was 21.6 ms;
with the tank's panes traced as glass, its own view 54 ms.

## The showroom

`www/showroom.html` (`start_showroom`) is a 20 x 14 m white gallery, 5 m high: a polished concrete
floor, plaster walls (the north one a warm grey backdrop), three tall windows to the west, a long
skylight over the aisle, two rows of low white plinths and an oak bench. On the plinths stand the
furniture pieces modelled in Blender from the owner's photos (the GAME version of each: the
Karuselli, Togo and Panton to the north, the MG5, Paimio, 406 and Wave cantilever to the south),
and a green Aalto vase of ray-traced glass on a pedestal under a narrow display spot, throwing
green caustics. Five track spots (3500 K) light five pieces; daylight comes through the skylight
and the windows. Everything else is the room's: every mode works (`gi=rt|voxel|voxel+ssgi|probes|
probes+ssgi|ssgi|off`, `shadows=rt|maps`, `quality=game|high`, `render=forward`), the panel, the
perf overlay, the character and its parkour (the bench takes a hurdle, the plinths are steps).
`cam=` names a view: `hall`, `aisle`, `outside` and one a piece (`karuselli`, `togo`, `panton`,
`vase`, `mg5`, `paimio`, `a406`, `wave`). The looks: `showroom` (the default), `overcast`, `spots
only`, `golden hour`.

**The furniture is private.** The pieces depict branded designs and are derived from the owner's
photos, so they are not in this repository: the page loads the architecture from
`www/assets/showroom/showroom.glb` (CC0, committed) and the pieces from
`www/assets/showroom/private/pieces.glb`, a gitignored link to the owner's exports
(`link-packs.sh` makes it). Without it the plinths stand empty; `pieces=0` leaves the pieces out
even when linked, for captures that may be published. Don't commit the pieces, a render of them or
anything made from them unless the owner says so.

The scene is authored in Blender outside the repository (`kansei-assets/showroom/`: its
`showroom_build.py` builds `showroom.blend`, `export_glb.py` writes the two glbs), on the room's
contract ([The room's source](#the-rooms-source)) plus:

- each piece is its asset's GAME collection, linked from the asset's `.blend` (or its
  `export/*_game.glb` imported while that has none) as a collection instance; adding one is a
  line in `PIECES` and a rebuild, and a piece whose asset has no `export/*_game.glb` yet is left
  out. Pieces without `_SHADOW` or `_COL` get them made (a ~500-triangle decimated copy, a box);
  a `_SHADOW` on the instance stands for all its meshes (`layout.rs`);
- a transmissive material (`KHR_materials_transmission`) is ray-traced glass whatever its
  object's `kansei`, tinted by its volume's attenuation (`KHR_materials_volume`) brought to a
  metre: the vase's green;
- the skylight's `inner`/`outer` (degrees) set its cone (the room keeps 62 and 86);
- the page reads several glbs as one room (`layout::Source`), the missing optional ones left out.

```sh
B=/Applications/Blender.app/Contents/MacOS/Blender
$B -b --factory-startup --python ~/Documents/dev/kansei-assets/showroom/showroom_build.py
python3 tools/room_glb.py ~/Documents/dev/kansei-assets/showroom/export/showroom.glb --out www/assets/showroom/showroom.glb
python3 tools/room_glb.py ~/Documents/dev/kansei-assets/showroom/export/pieces.glb --out ~/Documents/dev/kansei-assets/showroom/export/web/pieces.glb
sh link-packs.sh
```

GPU per frame at 1080p (`dpr=1`, `cam=hall`, no pack, M4 Pro, headless Chrome without vsync, the
GPU shared with other sessions), without the pieces / with them (the grid's 1650 / 7168
triangles): `gi=rt` 14.1 / 16.9 ms, `voxel` 12.7 / 14.2, `voxel+ssgi` 14.6 / 16.4, `probes`
9.9 / 11.4, `probes+ssgi` 12.0 / 13.6, `ssgi` 11.1 / 13.1, `off` 9.4 / 11.0, `shadows=maps`
10.7 / 13.7, `quality=high` 12.8 / 20.0, `render=forward` 4.3 / 4.5. The mannequin walking (the
grid rebuilt every frame for its copy) added about 10 ms in `gi=rt`.

### The mannequin

A neutral artist's mannequin, modelled for the showroom (CC0): an egg head with no face, a
seamless torso, rigid tapered limbs between graphite ball joints, three-segment fingers, 26k
triangles, a 1024² colour and AO atlas baked in Cycles. `kansei-assets/mannequin/`
`mannequin_build.py` builds it from the joints of the packs' skeleton (read from the hero
character's glb) and `mannequin_export.py` bakes it and writes the glb with that skeleton's own
nodes and inverse bind matrices, so the packs' rotations play on it unchanged; the torso is
weighted along the spine (and the clavicles at the shoulders), every other part rigidly to its
bone. `bake-config.json` there makes `export/mannequin.kmm` with `kansei-anim-bake`; `link-packs.sh`
links it as `www/pack/mannequin.kmm`. The mesh is original, but the pack carries the rig's joint
frames, so it stays local with the packs.

## Build and run

```sh
cd rust/kansei-wasm/demos/motion-matching
wasm-pack build --target web --release
python3 -m http.server 8080   # then open http://localhost:8080/www/ (or www/kimodo.html, www/room.html, www/showroom.html)
```

The crate builds with WASM SIMD (`.cargo/config.toml`), since the search runs on the CPU.
`scripts/build-wasm-examples.sh` skips it: its packs never ship.

## Controls

| | keyboard / mouse | gamepad |
|---|---|---|
| move (relative to the camera) | WASD, arrows | left stick (tilt sets the pace) |
| run | Shift | B, right trigger |
| jump, or traverse what is ahead; by a wall, run along it; hanging, climb up; on a wall, jump off; on a bar, let go | Space | A |
| let go (of a ledge or a bar), drop off a wall | X | right bumper |
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

## Parkour

On top of the traversals, the controller (`motion_matching::traversal::parkour`) does the moves a
level tags for it (`traversal::features`: a glb node's `kansei_traversal`, read by
`ParkourPiece::from_extras`; the room's course has them all):

- **Platform to platform.** Space toward a `platform` (the next of its `kansei_chain` first) or any
  top across a gap jumps at it: the run-up turns toward it and the take-off's speed is solved so
  the arc lands a little in from its near edge, upward at the jump clip's own take-off speed (so
  the body's rise carries on). The landing is picked by the pace the player asks for and must fit
  on what it lands on (a run-out that would walk off a platform's far edge gives way to a
  shorter one).
- **Hanging from a ledge.** Climbing a tagged `ledge` stops hanging from it, and a fall or a jump
  that brings the body to an edge catches it. The hang is the climb clip's own hanging frame
  (`parkour::hang_poses`), held and breathing a frame either way, the hands pinned on the top and
  the feet on the face below by IK (`MotionMatcher::reach`); with a `hangidle` clip in the pack
  (generated, braced on the face or free, by its name's `Braced`) the hang settles into that loop
  once still, placed by its hips. A/D (the stick to either side)
  shimmy along the edge and round its outer corners, hands and feet stepping in turn, each pinned
  where it lands; Space climbs up (the climb clip on from its hang, warped onto the edge); X lets
  go. The climbs' and mantles' ledges are refined from the hands (`parkour::refine_grips`) so the
  wrists grip the face rather than sink into the top, and hands that still sink into what they
  vault or climb are pushed out of it. With air catches in the pack (`catch` actions: GASP's
  `Traversal/Catch/*` clips, which fall onto a 3 m top and catch it), a fall onto an edge plays
  the captured catch from just before the hands meet it into its hang (`Catch_Cliff`), and Space
  climbs up with that clip's own climb.
- **Air mantle.** Falling against the face of a top within a reach of the feet (0.1-0.9 m above
  them, room to stand on it, not a ledge to hang from, not on a jump aimed elsewhere) catches it
  and mantles up (`Catch_Mantle_*`: the one whose depth at the catch and pace fit, and whose
  run-out stays on that top).
- **Wall runs.** Running along a `wallrun` face (or jumping into it along it) and pressing Space
  runs along the wall: the run loop at the wall-run speed, the body leaning away from the wall
  about its feet, the planted feet pinned flat on it, the wall-side hand kept out of it; light
  gravity at first, then more, up to 1.3 s; the lean comes off before the run ends. Space jumps
  off, away from the wall (into another wall run, or a catch); X drops off.
- **Bars.** A jump at a `bar` aims the hips through the place hanging from it puts them, and the
  hands catch it as the body passes: the body swings about the bar as a damped pendulum, pumped
  by W/S, the hands pinned to it. With bar clips in the pack (`barhang`, `barswing` actions:
  generated, see [Generated animation](#generated-animation-kimodo)) the pose is the hang clip
  still or the swing clip phase-locked to the pendulum (its frame from the swing's phase, the body
  turned about the bar by what the pendulum swings beyond the clip); without them, the climb's
  hanging frame swung as a whole. Space lets go at the swing's speed (hanging still, with a
  `pullup` clip in the pack, it pulls up instead and hangs again after), X straight down.

Every change of move keeps what shows continuous: the root's velocity (`MotionMatcher::
blend_root`), its place and turn (`blend_position`, `blend_rotation`) ease out over a few frames
instead of jumping, a catch onto a clip whose root sits elsewhere under the body re-roots it
(`reroot`), and the pose is refreshed the frame the move starts (`refresh`).

**The packs.** The moves work with the GASP pack as it is (`gasp-locomotion.kmm`): jumps, the
hang, shimmy, corners and climbs come from its jump and climb clips; the wall run from its run
loop; the bars from the climb's hang. The bar clips come from Kimodo
(`kansei-anim-bake/genanim/prompts/parkour.json`, retargeted onto the mannequin, route B) and go
into a private pack with the GASP clips, e.g. the GASP bake config plus:

```json
"export": ["export", "../genanim/export-parkour-b"],
"include": ["...", "Jump/M_Neutral_Jump_F_Start_Across_*", "Jump/M_Neutral_Jump_F_Start_Sprint_*", "Parkour/Gen_*"],
"actions": { "...": [], "barhang": ["Parkour/Gen_Bar_Hang_*"], "barswing": ["Parkour/Gen_Bar_Swing_*"],
             "pullup": ["Parkour/Gen_Bar_PullUp_*"], "hangidle": ["Parkour/Gen_Hang_Idle_*"],
             "shimmy": ["Parkour/Gen_Shimmy_*"], "wallrun": ["Parkour/Gen_Wall_Run_*"] }
```

GASP's air catches and its sprint and roll landings join the same pack (export them with
`Traversal/Catch/*/M_Neutral_*` and `Jump/M_Neutral_*` in the export config):

```json
"export": ["export", "export-extras", "../genanim/export-parkour-b"],
"actions": { "...": [], "land": ["...", "Jump/M_Neutral_Jump_F_Land_Sprint_*", "Jump/M_Neutral_Jump_F_Land_Roll_*"],
             "catch": ["Traversal/Catch/Cliff/M_Neutral_*", "Traversal/Catch/Mantle/M_Neutral_*"] }
```

Captured hang clips from Mixamo (downloaded by hand, put on the mannequin by
`kansei-anim-bake/mixamo`; Mixamo's terms forbid AI/ML use, so never into Kimodo) go into the same
private pack:
- `hangidle` takes its braced idle (`Hang_Idle_Braced_Against_Wall`) and free one (`Hang_Idle_Ledge`);
- `shimmy` takes its braced and free shimmies. With shimmy clips in the pack, the shimmy plays the
  one for the hang (braced or not, by `Braced` in its name) and the way it goes (`Left`/`Right`),
  at the stick's pace, the hang moving along the edge at the clip's own speed.

Their other clips (catches, hops, drops, climb-ups, wall runs, swings) are baked but not wired yet.

(`wallrun` is baked but not played yet: the generated wall runs read as leaping beside the wall,
so the GASP-based one stays; Kimodo's shimmies, which paddle their legs, are left out of `shimmy`;
see the PR.) Packs with the new action kinds don't load in older builds or
the TS port.

**Testing it.** `render=forward` (the panel's render, top) draws the room with its direct lights
and shadow maps and the post-processing only, no ray tracing and no GI (a flat fill in their
place): the frame rate a character's animation is tuned at. `parkour_eval` measures the moves on a
pack against the course (the room's, built in code), at 30, 60 and 120 fps and with 100 ms frame
hitches: limbs off their targets, planted-foot slide, hands and feet inside the level, and pops
(the largest jump in the hips', spine's and head's velocity over 1/30 s windows, against the
run's 99th percentile; `clips=1` prints the captured clips' own, the floor: their take-offs reach
1-3 m/s and their landings 4-9):

```sh
cargo run -p kansei-core --release --example parkour_eval -- <pack.kmm> [only=<name part>] [trace=1] [csv=<dir>] [clips=1]
cargo run -p kansei-core --release --example pack_actions -- <pack.kmm> [track=<clip>] [clip=<clip>]
```

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
