// The irradiance a surface receives through a VoxelVolume: six 60-degree cones over the
// hemisphere around its normal, one along it and five tilted 60 degrees from it, weighted by
// their share of the cosine-weighted hemisphere (pi / 4 and 3 pi / 20: they sum to pi), the
// arrangement of Crassin et al. 2011. Light past the volume is the sky's. Needs
// voxel_volume.wgsl, voxel_cones.wgsl and SKY_LIGHTING_WGSL.

const VOXEL_GI_PI : f32 = 3.14159265;

// Returns the irradiance (rgb, scene units) and the share of the cosine-weighted hemisphere that
// sees past the volume (a). The tilted cones turn by `angle` about the normal (jitter it per
// pixel and frame and let a temporal filter integrate); `skyScale` scales the sky.
fn voxelIrradiance(
    vol: VoxelVolume,
    radiance: texture_3d<f32>,
    linearClamp: sampler,
    sky: SkyLighting,
    skyScale: f32,
    origin: vec3f,
    n: vec3f,
    angle: f32,
    startDist: f32,
    maxDist: f32,
    maxSteps: u32,
) -> vec4f {
    // a frame around the normal (Duff et al. 2017)
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3f(b, s + n.y * n.y * a, -n.y);
    let tanHalf = 0.57735027;   // 30 degrees
    var e = vec3f(0.0);
    var open = 0.0;
    for (var k = 0u; k < 6u; k++) {
        var dir = n;
        var w = 0.25 * VOXEL_GI_PI;
        if (k > 0u) {
            let phi = angle + f32(k - 1u) * (2.0 * VOXEL_GI_PI / 5.0);
            dir = normalize(n * 0.5 + (t * cos(phi) + bt * sin(phi)) * 0.8660254);
            w = 0.15 * VOXEL_GI_PI;
        }
        let c = voxelConeTrace(vol, radiance, linearClamp, origin, dir, tanHalf, startDist, maxDist, maxSteps);
        e += w * (c.rgb + c.a * skyScale * skyRadiance(sky, dir));
        open += w * c.a;
    }
    return vec4f(e, open / VOXEL_GI_PI);
}
