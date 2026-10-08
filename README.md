# kansei

<!-- Gallery: two cells per <tr>, each image a ~1600 px wide JPEG under 300 KB in docs/media/readme/
     linking to its live example. To add an entry, add a <td> (start a new <tr> after every two);
     an odd last entry can take colspan="2" as a wide hero. -->
<table>
  <tr>
    <td width="50%" valign="top">
      <a href="https://kansei.graphics/examples/instancing/"><img src="docs/media/readme/instancing.jpg" alt="Instancing" width="100%"></a><br>
      <a href="https://kansei.graphics/examples/instancing/"><b>Instancing</b></a><br>
      <sub>A ball pushes through a carpet of instanced cubes, simulated in compute</sub>
    </td>
    <td width="50%" valign="top">
      <a href="https://kansei.graphics/examples/voxel-gi-particles/?rt=on&amp;dof=1"><img src="docs/media/readme/voxel-gi-particles.jpg" alt="Voxel GI on particles" width="100%"></a><br>
      <a href="https://kansei.graphics/examples/voxel-gi-particles/?rt=on&amp;dof=1"><b>Voxel GI on particles</b></a><br>
      <sub>An SPH pile lit by voxel cone tracing, with ray-traced mirror and glass spheres and depth of field</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="https://kansei.graphics/examples/depth-of-field/"><img src="docs/media/readme/depth-of-field.jpg" alt="Depth of field" width="100%"></a><br>
      <a href="https://kansei.graphics/examples/depth-of-field/"><b>Depth of field</b></a><br>
      <sub>Physical lens depth of field with bokeh on a scrolling field of glowing columns</sub>
    </td>
    <td width="50%" valign="top">
      <a href="https://kansei.graphics/examples/fluid/"><img src="docs/media/readme/fluid.jpg" alt="Fluid" width="100%"></a><br>
      <a href="https://kansei.graphics/examples/fluid/"><b>Fluid</b></a><br>
      <sub>A 3D particle fluid splashing in a box, meshed with marching cubes and refracting the room</sub>
    </td>
  </tr>
</table>

Live examples: [kansei.graphics](https://kansei.graphics/)

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
  bindings: [
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
- **Material**: Manages shaders and bindings
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
  bindings: [
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
  bindings: [
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

This project uses:
- [gl-matrix](https://github.com/toji/gl-matrix) by Brandon Jones and Colin MacKenzie IV
- [artery-font](https://github.com/sidit77/artery-font/) by sidit77 - A pure Rust parser for Artery Atlas font files (compiled to WebAssembly)

All dependencies are under the MIT License.
