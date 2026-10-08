import { RtGrid, RtPlacement, RtSource, RtSurface, effectiveEpsilon, rtAlbedoWord, rtSurfaceWord } from './RtGrid';
import { RtMesh, boxesMeet, transformBox } from './RtMesh';

/** An instance of one of an `RtScene`'s meshes. Rust: `rt::RtInstance`. */
export interface RtInstance {
    mesh: number;
    /** Mesh to world, column-major. */
    transform: ArrayLike<number>;
    surface: RtSurface;
}

const MATRIX = RtPlacement.instance({ kind: 'matrix', offset: 0 });

/**
 * Meshes and their instances for an `RtGrid`, with no renderer: each gather picks the instances
 * whose world box meets the grid's on the CPU and places their triangles on the GPU. A source per
 * mesh and surface; its id (`kansei_rt_source`) is the mesh's index, its record
 * (`kansei_rt_record`) the instance's rank among that source's in the box. Rust: `rt::RtScene`.
 */
export class RtScene {
    private readonly meshes: { mesh: RtMesh; buffer: GPUBuffer | null }[] = [];
    public readonly instances: RtInstance[] = [];
    private records: GPUBuffer | null = null;
    /** Instances in the box at the last gather, and their triangles. */
    public inBox = 0;
    public inBoxTriangles = 0;

    /** Add a mesh; its index. */
    addMesh(mesh: RtMesh): number {
        this.meshes.push({ mesh, buffer: null });
        return this.meshes.length - 1;
    }

    mesh(index: number): RtMesh {
        return this.meshes[index].mesh;
    }

    get meshCount(): number {
        return this.meshes.length;
    }

    /** Add an instance; its index. */
    addInstance(instance: RtInstance): number {
        if (instance.mesh >= this.meshes.length) throw new Error(`RtScene: no mesh ${instance.mesh}`);
        this.instances.push(instance);
        return this.instances.length - 1;
    }

    /**
     * Gather the instances whose world box meets `grid`'s into it: between `RtGrid.begin` and
     * `RtGrid.finish`, as its one gather this build.
     */
    gather(device: GPUDevice, encoder: GPUCommandEncoder, grid: RtGrid): void {
        const [lo, hi] = grid.bounds();
        const eps = effectiveEpsilon(grid.options);
        // the instances in the box, by (mesh, surface): their matrices, a source's run each
        const chosen: [number, number, number, number][] = [];
        this.instances.forEach((i, k) => {
            const m = this.meshes[i.mesh].mesh;
            const [a, b] = transformBox(i.transform, m.min, m.max);
            if (boxesMeet(a, b, lo, hi, eps)) chosen.push([i.mesh, rtSurfaceWord(i.surface), rtAlbedoWord(i.surface), k]);
        });
        chosen.sort((x, y) => x[0] - y[0] || x[1] - y[1] || x[2] - y[2] || x[3] - y[3]);
        this.inBox = chosen.length;
        this.inBoxTriangles = chosen.reduce((sum, c) => sum + this.meshes[c[0]].mesh.triangleCount, 0);
        const bytes = Math.max(chosen.length, 1) * 64;
        if (!this.records || this.records.size < bytes) {
            this.records?.destroy();
            let size = 64;
            while (size < bytes) size *= 2;
            this.records = device.createBuffer({ label: 'RtScene/Records', size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        }
        if (chosen.length > 0) {
            const matrices = new Float32Array(chosen.length * 16);
            chosen.forEach((c, k) => matrices.set(this.instances[c[3]].transform, k * 16));
            device.queue.writeBuffer(this.records, 0, matrices);
        }
        for (const c of chosen) {
            const m = this.meshes[c[0]];
            m.buffer ??= m.mesh.createBuffer(device);
        }
        grid.reserveTriangles(Math.min(this.inBoxTriangles, 0xffffffff));
        // a source per run of one mesh and surface
        const sources: RtSource[] = [];
        let start = 0;
        while (start < chosen.length) {
            let end = start + 1;
            while (end < chosen.length && chosen[end][0] === chosen[start][0] && chosen[end][1] === chosen[start][1] && chosen[end][2] === chosen[start][2]) end++;
            const mesh = this.meshes[chosen[start][0]];
            sources.push({
                mesh: mesh.buffer!,
                triangles: mesh.mesh.triangleCount,
                records: { buffer: this.records, stride: 64 },
                firstRecord: start,
                recordCount: end - start,
                world: IDENTITY,
                placement: MATRIX,
                surface: this.instances[chosen[start][3]].surface,
                id: chosen[start][0],
            });
            start = end;
        }
        grid.gather(encoder, sources);
    }

    destroy(): void {
        this.records?.destroy();
        for (const m of this.meshes) m.buffer?.destroy();
    }
}

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
