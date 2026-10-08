// The planet's atmosphere and its sun and moon, as the Rust engine's `atmosphere/params.rs`.

export type Vec3 = [number, number, number];

/**
 * The planet's atmosphere. The fields and their defaults follow Unreal's `SkyAtmosphereComponent`
 * (an Earth-like atmosphere from Bruneton 2017 and Hillaire 2020), so a scene exported from
 * Unreal can copy its values across: each coefficient is a colour times a scale, per kilometre.
 */
export interface AtmosphereParams {
    /** Radius of the planet's surface, km. */
    bottomRadiusKm: number;
    /** Thickness of the atmosphere above the surface, km. */
    atmosphereHeightKm: number;
    /** Diffuse reflectance of the planet's surface; it bounces light back into the sky. */
    groundAlbedo: Vec3;
    /** Air molecules: scattering per km at sea level is `rayleighScattering * rayleighScatteringScale`. */
    rayleighScatteringScale: number;
    rayleighScattering: Vec3;
    /** Altitude at which Rayleigh density falls to 1/e, km. */
    rayleighExponentialDistributionKm: number;
    /** Aerosols (haze): scattering and absorption per km at sea level. */
    mieScatteringScale: number;
    mieScattering: Vec3;
    mieAbsorptionScale: number;
    mieAbsorption: Vec3;
    /** Cornette-Shanks g: 0 isotropic, toward 1 a tight halo around the sun. */
    mieAnisotropy: number;
    /** Altitude at which aerosol density falls to 1/e, km. */
    mieExponentialDistributionKm: number;
    /** An absorbing layer (ozone): absorption per km at the tent's tip. */
    otherAbsorptionScale: number;
    otherAbsorption: Vec3;
    /**
     * The absorbing layer's density is a tent: `otherTentTipValue` at `otherTentTipAltitudeKm`,
     * falling linearly to zero `otherTentWidthKm` above and below it.
     */
    otherTentTipAltitudeKm: number;
    otherTentTipValue: number;
    otherTentWidthKm: number;
    /** Scales the multiple-scattering contribution (1 is physical). */
    multiScatteringFactor: number;
    /** Artistic tint of the sky's luminance (not of the aerial perspective); 1 is physical. */
    skyLuminanceFactor: Vec3;
    /** Aerial perspective as if the scene were this many times farther away; 1 is physical. */
    aerialPerspectiveViewDistanceScale: number;
    /** Distance from the camera before which there is no aerial perspective, km. */
    aerialPerspectiveStartDepthKm: number;
}

/** Earth, with Unreal's `SkyAtmosphereComponent` defaults. */
export function earthAtmosphere(): AtmosphereParams {
    return {
        bottomRadiusKm: 6360,
        atmosphereHeightKm: 100,
        groundAlbedo: [0.402, 0.402, 0.402],
        rayleighScatteringScale: 0.0331,
        rayleighScattering: [0.175287, 0.409607, 1.0],
        rayleighExponentialDistributionKm: 8,
        mieScatteringScale: 0.003996,
        mieScattering: [1, 1, 1],
        mieAbsorptionScale: 0.000444,
        mieAbsorption: [1, 1, 1],
        mieAnisotropy: 0.8,
        mieExponentialDistributionKm: 1.2,
        otherAbsorptionScale: 0.001881,
        otherAbsorption: [0.345561, 1.0, 0.045188],
        otherTentTipAltitudeKm: 25,
        otherTentTipValue: 1,
        otherTentWidthKm: 15,
        multiScatteringFactor: 1,
        skyLuminanceFactor: [1, 1, 1],
        aerialPerspectiveViewDistanceScale: 1,
        aerialPerspectiveStartDepthKm: 0.1,
    };
}

export function atmosphereTopRadiusKm(a: AtmosphereParams): number {
    return a.bottomRadiusKm + Math.max(a.atmosphereHeightKm, 1e-3);
}

const scaled = (c: Vec3, s: number): Vec3 => [c[0] * s, c[1] * s, c[2] * s];
const clamp = (x: number, lo: number, hi: number) => Math.min(Math.max(x, lo), hi);

/** The WGSL `Atmosphere` struct (common.wgsl), 96 bytes. */
export const ATMOSPHERE_BYTES = 96;

/** `params` as the WGSL `Atmosphere` struct (Rust `AtmosphereParams::gpu_layout`). */
export function atmosphereGpu(a: AtmosphereParams): Float32Array {
    const rayleigh = scaled(a.rayleighScattering, a.rayleighScatteringScale);
    const mieScattering = scaled(a.mieScattering, a.mieScatteringScale);
    const mieAbsorption = scaled(a.mieAbsorption, a.mieAbsorptionScale);
    return new Float32Array([
        a.bottomRadiusKm, atmosphereTopRadiusKm(a),
        -1 / Math.max(a.rayleighExponentialDistributionKm, 1e-3), -1 / Math.max(a.mieExponentialDistributionKm, 1e-3),
        ...rayleigh, clamp(a.mieAnisotropy, -0.999, 0.999),
        ...mieScattering, a.otherTentTipAltitudeKm,
        mieScattering[0] + mieAbsorption[0], mieScattering[1] + mieAbsorption[1], mieScattering[2] + mieAbsorption[2], a.otherTentTipValue,
        ...scaled(a.otherAbsorption, a.otherAbsorptionScale), Math.max(a.otherTentWidthKm, 1e-3),
        ...a.groundAlbedo, a.multiScatteringFactor,
    ]);
}

/** Extinction per km at `altitudeKm`, as the shaders' `sampleMedium`. */
export function extinctionAt(a: AtmosphereParams, altitudeKm: number): Vec3 {
    const h = Math.max(altitudeKm, 0);
    const rayleigh = Math.exp(-h / Math.max(a.rayleighExponentialDistributionKm, 1e-3));
    const mie = Math.exp(-h / Math.max(a.mieExponentialDistributionKm, 1e-3));
    const absorption = Math.max(a.otherTentTipValue - Math.abs(h - a.otherTentTipAltitudeKm) / Math.max(a.otherTentWidthKm, 1e-3), 0);
    const out: Vec3 = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
        const mieExt = a.mieScattering[i] * a.mieScatteringScale + a.mieAbsorption[i] * a.mieAbsorptionScale;
        out[i] = a.rayleighScattering[i] * a.rayleighScatteringScale * rayleigh
            + mieExt * mie
            + a.otherAbsorption[i] * a.otherAbsorptionScale * absorption;
    }
    return out;
}

/**
 * Transmittance from `altitudeKm` above the surface to space along a ray whose cosine with the
 * local zenith is `cosZenith`; zero if the ray hits the planet. The CPU twin of the transmittance
 * LUT, for lighting that has to agree with the sky (the sun's colour at the ground, say).
 */
export function transmittanceToSpace(a: AtmosphereParams, altitudeKm: number, cosZenith: number): Vec3 {
    const bottom = a.bottomRadiusKm, top = atmosphereTopRadiusKm(a);
    const r = bottom + clamp(altitudeKm, 0, a.atmosphereHeightKm);
    const mu = clamp(cosZenith, -1, 1);
    const horizon = -Math.sqrt(Math.max((r - bottom) * (r + bottom), 0)) / r;
    if (mu < horizon) return [0, 0, 0];
    const d = -r * mu + Math.sqrt(Math.max((top - r) * (top + r) + r * r * mu * mu, 0));
    const STEPS = 256;
    const dt = d / STEPS;
    const depth = [0, 0, 0];
    for (let i = 0; i < STEPS; i++) {
        const t = (i + 0.5) * dt;
        const h = Math.sqrt(r * r + 2 * r * mu * t + t * t) - bottom;
        const e = extinctionAt(a, h);
        for (let c = 0; c < 3; c++) depth[c] += e[c] * dt;
    }
    return [Math.exp(-depth[0]), Math.exp(-depth[1]), Math.exp(-depth[2])];
}

/** A light far outside the atmosphere with a visible disk: the sun or the moon. */
export interface CelestialLight {
    /** Unit vector from the scene toward the light, world space (Y up). */
    direction: Vec3;
    /**
     * Illuminance at the top of the atmosphere (colour times intensity). In lux for a physically
     * exposed scene (the sun is about 100 000, a full moon about 0.25), or any unit the scene's
     * other lights share. Zero switches the light off.
     */
    illuminance: Vec3;
    /** Apparent diameter, degrees (the sun 0.5357 as Unreal's default, the moon about 0.52). */
    angularDiameterDeg: number;
    /** Scales the disk's luminance; 0 hides the disk but keeps the light in the sky. */
    diskLuminanceScale: number;
}

export function sunLight(): CelestialLight {
    return { direction: directionFromElevationBearing(30, 180), illuminance: [1, 1, 1], angularDiameterDeg: 0.5357, diskLuminanceScale: 1 };
}

/** A moon that is off until given an illuminance. */
export function moonLight(): CelestialLight {
    return { direction: directionFromElevationBearing(20, 0), illuminance: [0, 0, 0], angularDiameterDeg: 0.52, diskLuminanceScale: 1 };
}

export function celestialAngularRadius(l: CelestialLight): number {
    return Math.max(l.angularDiameterDeg * 0.5 * Math.PI / 180, 1e-5);
}

/** Disk luminance per unit illuminance: 1 / (solid angle * mean limb darkening). */
export function celestialDiskLuminance(l: CelestialLight, limbDarkening: number): number {
    const solidAngle = 2 * Math.PI * (1 - Math.cos(celestialAngularRadius(l)));
    return Math.max(l.diskLuminanceScale, 0) / (solidAngle * (1 - limbDarkening / 3));
}

/**
 * Unit vector toward a light at `elevationDeg` above the horizon (negative below it) and a
 * compass `bearingDeg` clockwise from north, in kansei's world frame (Y up, north = -Z, east =
 * +X). Unreal's sun rotator (pitch, yaw) maps to elevation = -pitch, bearing = yaw.
 */
export function directionFromElevationBearing(elevationDeg: number, bearingDeg: number): Vec3 {
    const e = elevationDeg * Math.PI / 180, b = bearingDeg * Math.PI / 180;
    return [Math.sin(b) * Math.cos(e), Math.sin(e), -Math.cos(b) * Math.cos(e)];
}
