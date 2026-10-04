# Sky atmosphere

A physically based sky (Hillaire 2020) from noon to dusk over a clearing in a ring of dark spruce
proxies, with hills out to 15 km fading into the aerial perspective. Surfaces are lit by the sun
the sky is rendered with and by the sky itself (SH), in lux and cd/m² exposed by EV100; a chrome
ball, a rough metal ball and a pond reflect the sky's prefiltered environment cubemap. Volumetric
clouds are on by default; fog, a local fog volume and screen-space GI are optional.

Engine API: `SkyAtmosphere` (`sun_illuminance_at`, `capture_fog`),
`atmosphere::direction_from_elevation_bearing`, `SKY_LIGHTING_WGSL`, `SKY_ENVIRONMENT_WGSL`,
`CLOUD_SHADOW_WGSL`, `AtmosphereEffect`, `VolumetricCloudsEffect` (`CloudLayer`),
`VolumetricFogEffect` with `LocalFogVolume`, `HeightFogEffect`, `ScreenSpaceGIEffect`,
`ToneMapEffect` (`ToneMapper::AcesFitted`, `exposure_from_ev100_lens`), `Renderer::enable_shadows`.

| URL parameter | Effect |
|---|---|
| `elevation=<degrees>` | sun elevation, negative below the horizon (default: a 60 s cycle between 32.5 and -7.5) |
| `bearing=<degrees>` | sun bearing, clockwise from north (default 140) |
| `look=<degrees>` | view bearing (default: the sun's) |
| `pitch=<degrees>` | view pitch (default 6) |
| `height=<metres>` | camera height (default 1.7; above about 30 it sees over the trees) |
| `ev=<EV100>` | exposure (default: follows the sun's elevation) |
| `haze=<x>` | scales the Mie scattering |
| `ozone=<x>` | scales the absorbing layer |
| `moon=1` | add a moon (any value turns it on) |
| `clouds=<0..1>` | cloud coverage (default 0.45; 0: no clouds) |
| `cloudtype=<0..1>` | 0 stratus to 1 cumulus (default 0.7) |
| `cloudbase=<metres>` | the cloud layer's base (default 1500) |
| `cloudthick=<metres>` | the cloud layer's thickness (default 2500) |
| `cloudshadows=0` | the clouds cast no shadows on the scene |
| `cloudquality=low\|medium\|high` | the clouds' quality (default: the effect's own) |
| `fog=<density>` | froxel height fog, base density per metre (e.g. 0.01), lit by the sun and the sky |
| `mist=1` | a local fog volume in the clearing (any value but `box`; `mist=box`: a box-shaped one) |
| `gi=low\|medium\|high\|ultra` | screen-space global illumination (`gi=1`: medium; default off) |
| `preset=midsommar` | the Unreal intro's light block: sun -2.5°, haze 1.7, ozone 0.8, EV100 3.9, analytic height fog past 120 m and volumetric fog inside it; `elevation=` and `ev=` still apply, `haze=`, `ozone=`, `fog=` and `mist=` do not |
| `capturefog=1` | with the preset: the sky lighting and the reflections see the sky through its height fog (`capturefog=ue`: the fog below the horizon too, rather than the lit ground) |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: none; the page shows its URL parameters at the bottom left.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
