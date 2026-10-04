// The spheres' shading (after room_spheres.wgsl), drawn testing for the depth that
// room_spheres_depth.wgsl laid: it runs for the nearest sphere of each pixel alone, not for every
// sphere of the pile behind it.
@fragment
fn fragment_main(in: VOut) -> @location(0) vec4f {
    let pv = eyeHit(in);
    let ray = normalize(in.viewPos);
    let r = in.sphere.w;
    let nv = (pv - in.sphere.xyz) / r;

    // to the world, on the room's side of the floor
    let right = vec3f(view_matrix[0][0], view_matrix[1][0], view_matrix[2][0]);
    let up = vec3f(view_matrix[0][1], view_matrix[1][1], view_matrix[2][1]);
    let back = vec3f(view_matrix[0][2], view_matrix[1][2], view_matrix[2][2]);
    var n = normalize(right * nv.x + up * nv.y + back * nv.z);
    var d = normalize(right * ray.x + up * ray.y + back * ray.z);
    if (particles.mirrored > 0.5) {
        n.y = -n.y;
        d.y = -d.y;
    }
    let i = in.index;
    let p = in.world + n * r;
    let cosV = dot(-d, n);

    var color: vec3f;
    let kind = particleKind(i);
    if (scene.view > 0.5) {
        let emission = lighting[2u * i + 1u].rgb;
        color = particleAlbedo(i, any(emission > vec3f(0.0))) * lighting[2u * i].rgb;
    } else if (kind != MATTE) {
        let rays = sphereRays(kind, p, n, d, in.world, r);
        color = vec3f(0.0);
        for (var k = 0u; k < rays.count; k++) {
            let first = k == 0u;
            color += select(rays.w1, rays.w0, first) * trace(select(rays.o1, rays.o0, first), select(rays.d1, rays.d0, first), i);
        }
    } else {
        // matte, with a soft sheen of the panel (shadowed as its light is)
        color = shadeMatte(i, p, n, -d) + fresnel(0.04, cosV) * panelSeen(scene, p, reflect(d, n)) * lighting[2u * i].a;
    }
    if (particles.mirrored > 0.5) {
        color *= floorReflection(scene, p.y);
    }
    return vec4f(tonemap(scene, color), 1.0);
}
