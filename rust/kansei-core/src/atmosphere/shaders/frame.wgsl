// Per-frame sky state, written by SkyAtmosphere::update. Positions are in the planet frame (km).

struct SkyFrame {
    invViewProj        : mat4x4f,
    cameraPos          : vec3f,   // clamped inside the atmosphere
    _pad0              : f32,
    worldOrigin        : vec3f,   // the world origin (metres (0, 0, 0)) in the planet frame
    _pad1              : f32,
    sunDirection       : vec3f,   // unit vector toward the sun
    sunAngularRadius   : f32,     // radians
    sunIlluminance     : vec3f,   // at the top of the atmosphere
    sunDiskLuminance   : f32,     // disk luminance per unit illuminance; 0 hides the disk
    moonDirection      : vec3f,
    moonAngularRadius  : f32,
    moonIlluminance    : vec3f,   // zero disables the moon
    moonDiskLuminance  : f32,
    skyLuminanceFactor : vec3f,
    _pad2              : f32,
}
