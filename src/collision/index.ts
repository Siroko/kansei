/** CPU collision (Rust `collision`): boxes and triangle meshes, casts, overlaps, capsule push-out. */
export { Obb, TriangleMesh, CollisionWorld, ALL_LAYERS, rayCapsule, raySphere, rayTriangle, closestPointTriangle } from "./CollisionWorld";
export type { Triangle, Shape, Collider, Hit, CastResult } from "./CollisionWorld";
