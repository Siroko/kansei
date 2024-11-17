# kansei

Kansei is a Toy WebGPU engine built with TypeScript, designed to bring a bit more feeling into digital experiences. Inspired by kansei engineering, it’s not just about achieving visual polish or performance; it’s about building interactions that resonate on an emotional level.

## Features

- WebGPU-based rendering
- Modular architecture
- 3D primitives and custom geometries
- Material system with shader customization
- Texture and video texture support
- Scene graph management
- Camera controls

## Installation

```bash
npm install kansei
```

## Basic Usage

```javascript
import {
  Renderer,
  Scene,
  Camera,
  Mesh,
  BoxGeometry,
  Material,
} from 'kansei';

const renderer = new Renderer();
await renderer.initialize();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new Scene();
const camera = new Camera(75, 0.1, 100, window.innerWidth / window.innerHeight);
camera.position.z = 5;

const geometry = new BoxGeometry(1, 1, 1);
const material = new Material(shaderCode);
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

## Text Rendering

Kansei includes a high-quality text rendering system using Multi-channel Signed Distance Field (MSDF) technology.

### Font Generation

1. Visit [msdf.kansei.graphics](https://msdf.kansei.graphics/)
2. Upload your TTF/OTF font file
3. Download the generated `.arfont` file containing:
   - SDF texture atlas
   - Font metrics
   - Glyph information

### Basic Text Usage

```javascript
import { FontLoader, TextGeometry, Material, Mesh } from 'kansei';

// Load the font
const fontLoader = new FontLoader();
const fontInfo = await fontLoader.load('path/to/your-font.arfont');

// Create text geometry
const geometry = new TextGeometry('Hello World', fontInfo, {
  fontSize: 40,
  lineWidth: 100,
  lineHeight: 25
});

// Create material with SDF shader
const material = new Material(shaderCode, [
  {
    binding: 0,
    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
    value: fontInfo.sdfTexture
  }
]);

// Create and add mesh to scene
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
  - Alignment

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

## License

MIT License

This project uses [gl-matrix](https://github.com/toji/gl-matrix) by Brandon Jones and Colin MacKenzie IV, which is also under the MIT License.
