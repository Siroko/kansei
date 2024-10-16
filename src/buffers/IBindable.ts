export interface IBindable {
    type?: string;
    initialized: boolean;
    needsUpdate: boolean;

    get resource(): GPUBindingResource;
    initialize(gpuDevice: GPUDevice): void;
    update(gpuDevice: GPUDevice): void;
}