import { mat4, vec3 } from "gl-matrix";
import { Light } from "./Light";
import { Vector3 } from "../math/Vector3";

const DEG = Math.PI / 180;

/**
 * A spot light in photometric units (Rust `lights::SpotLight`): a point source whose luminous
 * intensity (candela) is full inside `innerAngle` of `direction` and fades to zero at
 * `outerAngle`, falling off with the inverse square of distance until `range`, where a smooth
 * window takes it to zero.
 *
 * Angles are half-angles from the axis in radians, as UE's inner/outer cone angles. The
 * illuminance it gives a surface facing it at distance d is `intensity / d²` lux, so lit
 * surfaces come out in cd/m², ready for `ToneMapEffect`'s EV100 exposure.
 *
 * The light sits at its world position (`position`, under its parents); `direction` is in world
 * space (its rotation is unused). Materials shade with it through `SPOT_LIGHTS_WGSL`
 * (`kansei_spot_lights_radiance`); `castShadow` gives it a layer of the renderer's spot shadow
 * atlas (`Renderer.enableSpotShadows`).
 */
class SpotLight extends Light {
    /** The direction the light points, in world space (normalized when packed). */
    public direction: [number, number, number];
    /** Metres; the light reaches exactly zero here (UE's attenuation radius). */
    public range: number;
    /** Half-angle of the full-intensity cone, radians. */
    public innerAngle: number;
    /** Half-angle where the light reaches zero, radians. */
    public outerAngle: number;
    /**
     * Radius of the emitter in metres. Shadows are contact-hardening (PCSS): sharp where the
     * caster touches the receiver, softer with distance. 0 gives a fixed small PCF kernel.
     */
    public sourceRadius: number = 0.05;
    /**
     * Scattering in volumetric fog, relative to the surface lighting (UE's volumetric
     * scattering intensity); 0 keeps the light out of the fog.
     */
    public volumetricScale: number = 1.0;
    /** Receiver offset along the surface normal for shadow lookups, in shadow-map texels. */
    public shadowNormalBias: number = 1.5;

    /**
     * @param direction - The direction the light points, in world space.
     * @param color - Linear RGB tint; multiplied by `intensity`.
     * @param intensity - Luminous intensity on the axis, in candela.
     * @param range - Metres; the light reaches zero here.
     * @param innerAngle - Half-angle of the full-intensity cone, radians.
     * @param outerAngle - Half-angle where the light reaches zero, radians.
     */
    constructor(
        direction: [number, number, number] = [0, 0, -1],
        color: [number, number, number] = [1, 1, 1],
        intensity: number = 1000,
        range: number = 20,
        innerAngle: number = 20 * DEG,
        outerAngle: number = 30 * DEG,
    ) {
        super('spot', color, intensity);
        this.direction = direction;
        this.range = range;
        this.innerAngle = innerAngle;
        this.outerAngle = outerAngle;
    }

    /** The light's position in world space. */
    get worldPosition(): [number, number, number] {
        this.updateModelMatrix();
        const m = this.worldMatrix.internalMat4;
        return [m[12], m[13], m[14]];
    }

    /** Points the light at `target` (world space), from its world position. */
    lookAt(target: Vector3) {
        const [x, y, z] = this.worldPosition;
        const d: [number, number, number] = [target.x - x, target.y - y, target.z - z];
        const len = Math.hypot(d[0], d[1], d[2]);
        if (len > 1e-8) this.direction = [d[0] / len, d[1] / len, d[2] / len];
    }

    /** Outer and inner half-angles clamped to a valid cone (outer in (0, 89°], inner <= outer). */
    cone(): [number, number] {
        const outer = Math.min(Math.max(this.outerAngle, 1e-3), 89 * DEG);
        return [outer, Math.min(Math.max(this.innerAngle, 0), outer)];
    }

    /** The view matrix of the light's shadow map: at its position, looking along `direction`. */
    shadowView(out: mat4 = mat4.create()): mat4 {
        const eye = this.worldPosition as vec3;
        const dir = vec3.normalize(vec3.create(), this.direction as vec3);
        if (vec3.length(dir) === 0) vec3.set(dir, 0, 0, -1);
        const up: vec3 = Math.abs(dir[1]) > 0.99 ? [0, 0, 1] : [0, 1, 0];
        return mat4.lookAt(out, eye, vec3.add(vec3.create(), eye, dir), up);
    }

    /**
     * The perspective projection of the light's shadow map: the outer cone plus a margin for the
     * filter kernel, `[0, 1]` depth from `near` to `range`.
     */
    shadowProjection(near: number, out: mat4 = mat4.create()): mat4 {
        const [outer] = this.cone();
        const fov = Math.min(outer * 2 * 1.05 + 2 * DEG, 178 * DEG);
        return mat4.perspectiveZO(out, fov, 1, near, Math.max(this.range, near * 2));
    }

    /** `shadowProjection(near) * shadowView()`. */
    shadowViewProjection(near: number, out: mat4 = mat4.create()): mat4 {
        return mat4.multiply(out, this.shadowProjection(near), this.shadowView());
    }
}

export { SpotLight };
