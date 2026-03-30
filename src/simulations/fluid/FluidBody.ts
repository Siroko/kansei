import { Vector3 } from '../../math/Vector3';

export interface FluidBodyPrimitive {
    type: 'circle' | 'box' | 'capsule' | 'ellipse';
    radius?: number;
    halfW?: number;
    halfH?: number;
    halfLength?: number;
    radiusX?: number;
    radiusY?: number;
    localX?: number;
    localY?: number;
    localAngle?: number;
}

export interface FluidBodyOptions {
    primitives: FluidBodyPrimitive[];
    position?: Vector3;
    velocity?: Vector3;
    angle?: number;
    angularVelocity?: number;
    mass?: number;
    restitution?: number;
    reactionMultiplier?: number;
    maxPushDist?: number;
    forceClampFactor?: number;
    rightingStrength?: number;
    linearDamping?: number;
    angularDamping?: number;
    density?: number;
    mouseScale?: number;
}

const PRIMITIVE_TYPE_MAP: Record<string, number> = {
    circle: 0,
    box: 1,
    capsule: 2,
    ellipse: 3,
};

class FluidBody {
    public position: Vector3;
    public velocity: Vector3;
    public angle: number;
    public angularVelocity: number;
    public mass: number;
    public restitution: number;
    public reactionMultiplier: number;
    public maxPushDist: number;
    public forceClampFactor: number;
    public rightingStrength: number;
    public linearDamping: number;
    public angularDamping: number;
    public density: number;
    public mouseScale: number;
    public primitives: FluidBodyPrimitive[];

    readonly index: number;
    readonly primitiveStart: number;
    readonly primitiveCount: number;

    constructor(options: FluidBodyOptions, index: number, primitiveStart: number) {
        this.position = options.position ?? new Vector3();
        this.velocity = options.velocity ?? new Vector3();
        this.angle = options.angle ?? 0;
        this.angularVelocity = options.angularVelocity ?? 0;
        this.mass = options.mass ?? 1.0;
        this.restitution = options.restitution ?? 0.3;
        this.reactionMultiplier = options.reactionMultiplier ?? 2000.0;
        this.maxPushDist = options.maxPushDist ?? 0.5;
        this.forceClampFactor = options.forceClampFactor ?? 3.0;
        this.rightingStrength = options.rightingStrength ?? 10.0;
        this.linearDamping = options.linearDamping ?? 0.999;
        this.angularDamping = options.angularDamping ?? 0.98;
        this.density = options.density ?? 0.5;
        this.mouseScale = options.mouseScale ?? 0.1;
        this.primitives = options.primitives;
        this.index = index;
        this.primitiveStart = primitiveStart;
        this.primitiveCount = options.primitives.length;
    }

    computeInertia(): number {
        let inertia = 0;
        let totalArea = 0;
        for (const prim of this.primitives) {
            const ox = prim.localX ?? 0;
            const oy = prim.localY ?? 0;
            const offsetSq = ox * ox + oy * oy;
            let area = 0;
            switch (prim.type) {
                case 'circle': {
                    const r = prim.radius ?? 1;
                    area = Math.PI * r * r;
                    inertia += area * (0.5 * r * r + offsetSq);
                    break;
                }
                case 'box': {
                    const w = (prim.halfW ?? 1) * 2;
                    const h = (prim.halfH ?? 1) * 2;
                    area = w * h;
                    inertia += area * ((w * w + h * h) / 12 + offsetSq);
                    break;
                }
                case 'capsule': {
                    const r = prim.radius ?? 0.5;
                    const l = (prim.halfLength ?? 1) * 2;
                    area = Math.PI * r * r + 2 * r * l;
                    inertia += area * (r * r * 0.5 + l * l / 12 + offsetSq);
                    break;
                }
                case 'ellipse': {
                    const a = prim.radiusX ?? 1;
                    const b = prim.radiusY ?? 1;
                    area = Math.PI * a * b;
                    inertia += area * ((a * a + b * b) / 4 + offsetSq);
                    break;
                }
            }
            totalArea += area;
        }
        return totalArea > 0 ? inertia * (this.mass / totalArea) : this.mass;
    }

    static packPrimitive(prim: FluidBodyPrimitive, f32: Float32Array, u32: Uint32Array, offset: number): void {
        u32[offset] = PRIMITIVE_TYPE_MAP[prim.type] ?? 0;
        switch (prim.type) {
            case 'circle':
                f32[offset + 1] = prim.radius ?? 1;
                f32[offset + 2] = 0;
                break;
            case 'box':
                f32[offset + 1] = prim.halfW ?? 1;
                f32[offset + 2] = prim.halfH ?? 1;
                break;
            case 'capsule':
                f32[offset + 1] = prim.radius ?? 0.5;
                f32[offset + 2] = prim.halfLength ?? 1;
                break;
            case 'ellipse':
                f32[offset + 1] = prim.radiusX ?? 1;
                f32[offset + 2] = prim.radiusY ?? 1;
                break;
        }
        f32[offset + 3] = prim.localX ?? 0;
        f32[offset + 4] = prim.localY ?? 0;
        f32[offset + 5] = prim.localAngle ?? 0;
    }

    packState(f32: Float32Array, u32: Uint32Array, offset: number): void {
        f32[offset + 0] = this.position.x;
        f32[offset + 1] = this.position.y;
        f32[offset + 2] = this.position.z;
        f32[offset + 3] = 0;
        f32[offset + 4] = this.velocity.x;
        f32[offset + 5] = this.velocity.y;
        f32[offset + 6] = this.velocity.z;
        f32[offset + 7] = 0;
        f32[offset + 8] = this.angle;
        f32[offset + 9] = this.angularVelocity;
        f32[offset + 10] = this.mass;
        f32[offset + 11] = this.computeInertia();
        f32[offset + 12] = this.restitution;
        u32[offset + 13] = this.primitiveStart;
        u32[offset + 14] = this.primitiveCount;
        f32[offset + 15] = this.reactionMultiplier;
        f32[offset + 16] = this.maxPushDist;
        f32[offset + 17] = this.forceClampFactor;
        f32[offset + 18] = this.rightingStrength;
        f32[offset + 19] = this.linearDamping;
        f32[offset + 20] = this.angularDamping;
        f32[offset + 21] = this.density;
        f32[offset + 22] = this.mouseScale;
        f32[offset + 22] = 0;
        f32[offset + 23] = 0;
    }
}

export { FluidBody, PRIMITIVE_TYPE_MAP };
