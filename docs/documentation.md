# kansei Engine Documentation

kansei is an open-source 3D engine for WebGPU, providing a modular architecture for creating 3D graphics applications in the browser.

## Table of Contents

1. [Renderer](#renderer)
2. [Texture](#texture)
3. [Geometry](#geometry)
4. [Material](#material)
5. [Mesh](#mesh)
6. [Scene](#scene)
7. [Camera](#camera)
8. [VideoTexture](#videotexture)
9. [Vector4](#vector4)
10. [TextureLoader](#textureloader)
11. [Sampler](#sampler)
12. [Compute](#compute)
13. [ComputeBuffer](#computebuffer)

## Renderer

The `Renderer` class is responsible for handling WebGPU rendering.

### Public API

- `constructor(options: RendererOptions)`: Creates a new Renderer instance with the given options.
- `async initialize(): Promise<void>`: Initializes the WebGPU device and context.
- `setSize(width: number, height: number): void`: Sets the size of the rendering canvas.
- `render(scene: Scene, camera: Camera): void`: Renders the given scene using the specified camera.
- `async compute(compute: Compute, workgroupsX: number, workgroupsY: number, workgroupsZ: number): Promise<void>`: Executes a compute shader.
- `async readBackBuffer<T>(buffer: ComputeBuffer, ArrayType: new (buffer: ArrayBuffer) => T): Promise<T>`: Reads data back from a compute buffer.

## Texture

The `Texture` class handles 2D textures for materials.

### Public API

- `constructor(imageBitmap: ImageBitmap)`: Creates a new Texture instance from an ImageBitmap.
- `async initialize(gpuDevice: GPUDevice): Promise<void>`: Initializes the texture on the GPU.
- `get resource(): GPUBindingResource`: Returns the GPU binding resource for this texture.

## Geometry

The `Geometry` class defines vertex data for meshes.

### Public API

- `constructor()`: Creates a new Geometry instance.
- `createVertexBuffer(gpuDevice: GPUDevice): void`: Creates a vertex buffer on the GPU.

## Material

The `Material` class manages shaders and bindings for rendering.

### Public API

- `constructor(shaderCode: string, bindings: BindGroupDescriptor[])`: Creates a new Material instance with the given shader code and bindings.
- `async initialize(gpuDevice: GPUDevice, vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null>, presentationFormat: GPUTextureFormat): Promise<void>`: Initializes the material on the GPU.
- `async getBindGroup(gpuDevice: GPUDevice, bindingGroupLayoutPosition: number): Promise<GPUBindGroup>`: Gets the bind group for this material.

## Mesh

The `Mesh` class combines geometry and material to create renderable objects.

### Public API

- `constructor(geometry: Geometry, material: Material)`: Creates a new Mesh instance with the given geometry and material.

## Scene

The `Scene` class manages the 3D scene graph.

### Public API

- `constructor()`: Creates a new Scene instance.
- `add(object: Object3D): void`: Adds an object to the scene.

## Camera

The `Camera` class defines view and projection matrices.

### Public API

- `constructor(fov: number, near: number, far: number, aspect: number)`: Creates a new Camera instance with the given parameters.

## VideoTexture

The `VideoTexture` class supports video textures for dynamic content.

### Public API

- `constructor(videoElement: HTMLVideoElement)`: Creates a new VideoTexture instance from an HTML video element.
- `async initialize(gpuDevice: GPUDevice): Promise<void>`: Initializes the video texture on the GPU.
- `get resource(): GPUBindingResource`: Returns the GPU binding resource for this video texture.

## Vector4

The `Vector4` class represents 4D vectors and provides mathematical operations.

### Public API

- `constructor(x: number, y: number, z: number, w: number)`: Creates a new Vector4 instance with the given components.
- `get/set x(): number`: Gets or sets the x component.
- `get/set y(): number`: Gets or sets the y component.
- `get/set z(): number`: Gets or sets the z component.
- `get/set w(): number`: Gets or sets the w component.

## TextureLoader

The `TextureLoader` class loads textures from URLs.

### Public API

- `async load(url: string): Promise<ImageBitmap>`: Loads an image from the given URL and returns it as an ImageBitmap.

## Sampler

The `Sampler` class defines texture sampling behavior.

### Public API

- `constructor(magFilter: GPUFilterMode, minFilter: GPUFilterMode, repeatMode: GPUAddressMode)`: Creates a new Sampler instance with the given filtering and repeat modes.
- `initialize(gpuDevice: GPUDevice): void`: Initializes the sampler on the GPU.
- `get resource(): GPUBindingResource`: Returns the GPU binding resource for this sampler.

## Compute

The `Compute` class manages compute shader pipelines and execution.

### Public API

- `constructor(shaderCode: string, bindings: BindGroupDescriptor[])`: Creates a new Compute instance with the given shader code and bindings.
- `initialize(gpuDevice: GPUDevice): void`: Initializes the compute pipeline on the GPU.
- `async getBindGroup(gpuDevice: GPUDevice, bindingGroupLayoutPosition: number): Promise<GPUBindGroup>`: Gets the bind group for this compute pipeline.

## ComputeBuffer

The `ComputeBuffer` class represents a buffer for use in compute shaders.

### Public API

- `constructor()`: Creates a new ComputeBuffer instance.
- `setBuffer(buffer: Float32Array, usage: GPUFlagsConstant): void`: Sets the buffer data and usage flags.

These classes form the core of the kansei engine, providing a comprehensive set of tools for WebGPU-based 3D rendering and computation.
