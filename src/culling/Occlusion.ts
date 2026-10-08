import { mat4 } from 'gl-matrix';
import type { Camera } from '../cameras/Camera';
import { DepthPyramid } from './DepthPyramid';
import { CullPipeline, CullView, OcclusionView, cullView } from './InstanceCulling';

/** The main view to cull with this frame. */
export interface MainView {
    cull: CullView;
    lodOrigin: [number, number, number];
    view: mat4;
    /** the projection as rasterized (jittered) */
    proj: mat4;
}

/** What occlusion culling of `main` projects with, against a depth buffer of `depthSize`. */
export function mainOcclusionView(main: MainView, depthSize: [number, number]): OcclusionView {
    return { view: main.view, proj: main.proj, depthSize, reverseZ: false, linearDepth: false };
}

/**
 * The renderer's occlusion-culling state: the switch, a pending history reset, the frozen main
 * view (debugging), and per view culled in two phases the depth pyramid its second phase tests
 * against. Rust: `culling::Occlusion`.
 */
export class Occlusion {
    enabled = true;
    freeze = false;
    private reset = false;
    private frozenView: MainView | null = null;
    private pyramids: { view: number; pyramid: DepthPyramid; bindGroup: GPUBindGroup }[] = [];

    requestReset(): void {
        this.reset = true;
    }

    /** Whether to forget which instances were visible this frame: on request, or on a camera cut. */
    takeReset(cameraCut: boolean): boolean {
        const reset = this.reset;
        this.reset = false;
        return reset || cameraCut;
    }

    /** The camera's view, or while frozen the one it had when frozen. */
    mainView(camera: Camera): MainView {
        if (this.freeze && this.frozenView) return this.frozenView;
        const eye = camera.inverseViewMatrix.internalMat4;
        const live: MainView = {
            // the unjittered projection: the frustum, not where pixels sample
            cull: cullView(camera.viewProjection()),
            lodOrigin: [eye[12], eye[13], eye[14]],
            view: mat4.clone(camera.viewMatrix.internalMat4),
            proj: camera.jitteredProjection(),
        };
        this.frozenView = this.freeze ? live : null;
        return live;
    }

    /**
     * Whether the main view is frozen: its first phase draws what was visible when it froze, and
     * there is no second phase.
     */
    get frozen(): boolean {
        return this.frozenView !== null;
    }

    /**
     * Cull view `view`'s pyramid for a depth buffer of `size` (created or recreated as needed),
     * and the `late` pipeline's bind group for it.
     */
    pyramidFor(view: number, device: GPUDevice, size: [number, number], pipeline: CullPipeline): { pyramid: DepthPyramid; bindGroup: GPUBindGroup } {
        let entry = this.pyramids.find((p) => p.view === view);
        if (entry && (entry.pyramid.sourceSize[0] !== size[0] || entry.pyramid.sourceSize[1] !== size[1])) {
            entry.pyramid.destroy();
            this.pyramids.splice(this.pyramids.indexOf(entry), 1);
            entry = undefined;
        }
        if (!entry) {
            const pyramid = new DepthPyramid(device, size[0], size[1], 'max');
            entry = { view, pyramid, bindGroup: pipeline.pyramidBindGroup(pyramid) };
            this.pyramids.push(entry);
        }
        return entry;
    }

    /** Cull view `view`'s pyramid, once built. */
    pyramid(view: number): DepthPyramid | null {
        return this.pyramids.find((p) => p.view === view)?.pyramid ?? null;
    }
}
