/**
 * Motion matching (Rust `animation::motion_matching`): animate a character by searching a
 * database of animation frames, every few frames, for the one whose pose and future trajectory
 * best match where the character is and where the input says it is going, then playing on from
 * there.
 *
 * Clean-room, after Simon Clavet's "Motion Matching and The Road to Next-Gen Animation" (GDC
 * 2016), Kristjan Zadziuk's (GDC 2016), and Holden, Kanoun, Perepichka and Popa's "Learned
 * Motion Matching" (SIGGRAPH 2020) with Daniel Holden's MIT-licensed reference implementation
 * (<https://github.com/orangeduck/Motion-Matching>). A pure TypeScript port of the Rust module,
 * to measure against it (its search has no SIMD here).
 *
 * - `DatabaseBuilder` turns clips (on one skeleton, at one rate) into a `Database`: poses with
 *   quantized rotations, the character root's motion (the root joint's position and heading),
 *   foot contacts, and 27 search features per frame (feet positions and velocities, hips
 *   velocity, the root's future positions and facings at 1/3, 2/3 and 1 s), normalized per group
 *   and weighted, with bounding boxes over runs of 16 and 64 frames.
 * - `Database.search` is a brute-force nearest neighbour over the features that skips runs whose
 *   box is already too far.
 * - `MotionMatcher` is the runtime character: spring-simulated trajectory, periodic and forced
 *   searches, root motion, inertialized transitions, the animated character kept near the
 *   simulation, and foot locking with two-joint IK.
 * - `MotionPack` and `CharacterPack` read (and write) `.kmm` packs. None ships with Kansei.
 * - `FootSlide` measures how far planted feet slide, over a scripted course (`measureFootSlide`).
 * - `CharacterController` (`Traversal`) keeps a matcher out of a `CollisionWorld`, on its ground,
 *   falling and landing, and hurdles, vaults, mantles, climbs or jumps on request.
 */
export { FORWARD, yawOf, yawRotation, wrapAngle } from "./Heading";
export {
    FEATURES, STRIDE, BOUND_SMALL, BOUND_LARGE, TRAJECTORY_TIMES, ACTION_TAG,
    findJointRoles, defaultFeatureWeights, defaultContactThresholds, defaultSearchFilter, filterAllows,
    ClipInfo, Database, DatabaseBuilder,
} from "./Database";
export type { JointRoles, FeatureWeights, ContactThresholds, SearchFilter, Match, RootSample } from "./Database";
export { MotionMatcher, defaultMotionMatchingSettings } from "./MotionMatcher";
export type { MotionMatchingSettings, MotionInput, Simulation, SearchInfo, RootPath, Action, Constrain } from "./MotionMatcher";
export { ActionKind, ActionClip, actionKindName, actionKindFromName, crosses } from "./ActionClip";
export type { ActionClipFields } from "./ActionClip";
export { MotionPack, CharacterPack, PackError } from "./Pack";
export { FootSlide, moveVelocity, footSlideCourse, measureFootSlide } from "./FootSlide";
export type { FootSlideReport, Move, Scenario } from "./FootSlide";
export type { PackMesh, PackImage } from "./Pack";
export {
    detectObstacle, defaultDetectionSettings, Refusal, isRefusal, defaultTraversalRules, standsAt, traversalKind, planTraversal, planJump,
    CharacterController,
} from "./Traversal";
export type { Obstacle, DetectionSettings, TraversalResult, TraversalRules, CharacterState } from "./Traversal";
