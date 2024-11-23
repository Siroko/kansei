# kansei

Kansei is a Toy WebGPU engine built with TypeScript, inspired by old school 3D frameworks like Papervision3D, Flash API or some more modern ones like Pixi.js, Three.js or Unity compute pipeline, this library is not intended to cover a generalist use case but will be specifically targeting WebGPU and provide tools to render and compute simulations by using native WGSL. 

Note that the library is highly experimental and WIP, so it will take some time to be production-ready 😅 

## Features

- WebGPU-based rendering
- Modular architecture
- 3D primitives and custom geometries
- Material system with shader customization
- Texture and video texture support
- Scene graph management
- Camera controls
- Compute shader support
- MSDF text rendering

## Installation

```bash
npm install kansei
```

## Basic Usage

```typescript
import {
  Renderer,
  Scene,
  Camera,
  Mesh,
  BoxGeometry,
  Material,
} from 'kansei';

const renderer = new Renderer({
  antialias: true,
  alphaMode: 'premultiplied'
});
await renderer.initialize();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new Scene();
const camera = new Camera(75, 0.1, 100, window.innerWidth / window.innerHeight);
camera.position.z = 5;

const geometry = new BoxGeometry(1, 1, 1);
const material = new Material(shaderCode, {
  uniforms: [
    {
      binding: 0,
      visibility: GPUShaderStage.FRAGMENT,
      value: texture
    }
  ],
  transparent: false,
  depthWriteEnabled: true,
  cullMode: 'back'
});
const mesh = new Mesh(geometry, material);
scene.add(mesh);

function animate() {
  mesh.rotation.x += 0.01;
  mesh.rotation.y += 0.01;
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();
```

## Core Components

- **Renderer**: Handles WebGPU rendering and compute operations
- **Scene**: Manages the 3D scene graph
- **Camera**: Defines view and projection
- **Mesh**: Combines geometry and material
- **Geometry**: Defines vertex data
- **Material**: Manages shaders and uniforms
- **Texture**: Handles 2D textures
- **VideoTexture**: Supports video textures
- **Vector3** and **Vector4**: Represent 3D and 4D vectors
- **Matrix4**: Handles 4x4 matrix operations
- **TextureLoader**: Loads textures from URLs
- **Compute**: Manages compute shader operations
- **ComputeBuffer**: Handles data for compute shaders

## Font generation

1. Visit [msdf.kansei.graphics](https://msdf.kansei.graphics/)
2. Upload your TTF/OTF font file
3. Download the generated `.arfont` file containing:
   - SDF texture atlas
   - Font metrics
   - Glyph information

## Text Rendering

Kansei includes a text rendering system using Multi-channel Signed Distance Field (MSDF) technology.

### Text Usage

```typescript
import { 
  FontLoader, 
  TextGeometry, 
  Material, 
  Mesh, 
  Sampler,
  Vector4 
} from 'kansei';

// Load the font
const fontLoader = new FontLoader();
const fontInfo = await fontLoader.load('path/to/font.arfont');

// Create text geometry
const geometry = new TextGeometry({
  text: 'Hello World',
  fontInfo: fontInfo,
  width: 40,
  height: 100,
  fontSize: 25,
  color: new Vector4(1, 1, 1, 1)
});

// Create material with SDF shader
const material = new Material(shaderCode, {
  uniforms: [
    {
      binding: 0,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
      value: fontInfo.sdfTexture
    },
    {
      binding: 1,
      visibility: GPUShaderStage.FRAGMENT,
      value: new Sampler('linear', 'linear')
    }
  ],
  transparent: true
});

const textMesh = new Mesh(geometry, material);
scene.add(textMesh);
```

### Features

- High-quality text rendering at any scale
- Support for custom fonts via MSDF generation
- Efficient GPU-based rendering
- Full Unicode support
- Customizable text properties:
  - Font size
  - Line width
  - Line height
  - Color

## Compute Shader Support

```typescript
import { Compute, ComputeBuffer, BufferBase } from 'kansei';

// Create compute shader with storage buffer
const computeBuffer = new ComputeBuffer({
  usage: BufferBase.BUFFER_USAGE_STORAGE | BufferBase.BUFFER_USAGE_VERTEX,
  type: ComputeBuffer.BUFFER_TYPE_STORAGE,
  buffer: data,
  shaderLocation: 0
});

const compute = new Compute(shaderCode, {
  uniforms: [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      value: computeBuffer
    }
  ]
});

// Dispatch compute shader
compute.dispatch(workgroupCount);
```

## Development

1. Clone the repository
2. Install dependencies: `npm install`
3. Build: `npm run build`

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Documentation

For detailed documentation on each component, please refer to the [documentation file](docs/documentation.md).

## Examples

Check out our example implementations:

1. [Basic scene rendering](examples/index.html)
2. [Compute shader usage](examples/index_compute.html)
3. [Text rendering](examples/index_text.html)

## License

MIT License

This project uses [gl-matrix](https://github.com/toji/gl-matrix) by Brandon Jones and Colin MacKenzie IV, which is also under the MIT License.
