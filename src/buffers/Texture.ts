import { IBindable } from "./IBindable";
import type { BindingLayout } from "../materials/Binding";
import { isDepthFormat, textureFormatBlock, textureFormatSampleType } from "./TextureFormats";

/** An image the texture copies from with `copyExternalImageToTexture`. */
export type TextureSource = ImageBitmap | HTMLCanvasElement;

/** One mip level's texels, tightly packed rows (or block rows), layer after layer. */
export type TextureLevelData = ArrayBufferView | ArrayBuffer;

/**
 * A texture of any format and shape, created on first use and filled with `levels` if given
 * (the Rust engine's `Texture::new_2d`/`new_3d`/`new_2d_array`/`from_levels`).
 */
export interface TextureOptions {
    label?: string;
    width: number;
    height: number;
    /** Depth of a 3D texture or layer count of a 2D one (6 for a cube). Defaults to 1. */
    depthOrArrayLayers?: number;
    /** Defaults to `rgba8unorm`; compressed formats need their device feature. */
    format?: GPUTextureFormat;
    /** Defaults to `2d`. */
    dimension?: GPUTextureDimension;
    /**
     * The bound view's dimension. Defaults to `3d` for a 3D texture, `2d-array` for a 2D texture
     * of several layers and `2d` otherwise; pass `cube` or `cube-array` for cubemaps.
     */
    viewDimension?: GPUTextureViewDimension;
    /** Defaults to the number of `levels`, or 1. */
    mipLevelCount?: number;
    /** Defaults to `TEXTURE_BINDING | COPY_DST`; add `STORAGE_BINDING` to bind it as a storage texture. */
    usage?: GPUTextureUsageFlags;
    /** Initial data, one entry per mip level from level 0, uploaded on initialization. */
    levels?: TextureLevelData[];
}

/**
 * Represents a WebGPU texture that can be bound to a shader: an image (optionally with GPU-built
 * mips), a texture of any format, dimension and layer count filled level by level, or a texture
 * created elsewhere (`Texture.fromView`).
 * @implements {IBindable}
 */
class Texture implements IBindable {
    /** Flag indicating if the texture has been initialized */
    public initialized: boolean = false;
    /** Magnification filter mode for the texture */
    public magFilter: GPUFilterMode = 'linear';
    /** Minification filter mode for the texture */
    public minFilter: GPUFilterMode = 'linear';
    /** Unique identifier for the texture */
    public uuid: string;
    /** Type identifier for the texture */
    type: string = 'texture';
    /** Flag indicating if the texture needs to be updated */
    public needsUpdate: boolean = false;

    public label: string;
    public format: GPUTextureFormat = 'rgba8unorm';
    public width: number;
    public height: number;
    public depthOrArrayLayers: number = 1;
    public dimension: GPUTextureDimension = '2d';
    public viewDimension: GPUTextureViewDimension = '2d';
    public mipLevelCount: number = 1;
    public usage: GPUTextureUsageFlags = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;

    /** The underlying WebGPU texture object */
    private texture?: GPUTexture;
    private view?: GPUTextureView;
    /** The format the bound view reads, when it differs from the texture's (sRGB images). */
    private viewFormat?: GPUTextureFormat;
    private source?: TextureSource;
    private levels?: TextureLevelData[];

    /**
     * Creates a new Texture instance.
     * @param source - The image to copy, or the texture's shape, format and data
     * @param mipmaps - For an image: build its mip chain on the GPU
     */
    constructor(
        source: TextureSource | TextureOptions,
        private mipmaps: boolean = false
    ) {
        this.uuid = crypto.randomUUID();
        if (!isTextureOptions(source)) {
            this.label = 'Texture';
            this.source = source;
            this.width = source.width;
            this.height = source.height;
            this.usage =
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.STORAGE_BINDING;
            this.mipLevelCount = mipmaps ? Texture.fullMipCount(source.width, source.height) : 1;
        } else {
            this.label = source.label ?? 'Texture';
            this.width = source.width;
            this.height = source.height;
            this.depthOrArrayLayers = Math.max(1, source.depthOrArrayLayers ?? 1);
            this.format = source.format ?? this.format;
            this.dimension = source.dimension ?? '2d';
            this.viewDimension = source.viewDimension
                ?? (this.dimension === '3d' ? '3d' : this.depthOrArrayLayers > 1 ? '2d-array' : this.dimension);
            this.levels = source.levels;
            this.mipLevelCount = Math.max(1, source.mipLevelCount ?? source.levels?.length ?? 1);
            this.usage = source.usage ?? this.usage;
        }
    }

    /** An empty 2D texture, filled by a render or compute pass (give it the usages that need). */
    public static new2D(label: string, width: number, height: number, format: GPUTextureFormat, usage?: GPUTextureUsageFlags): Texture {
        return new Texture({ label, width, height, format, usage });
    }

    /** An empty 3D texture, bound as `texture_3d` or `texture_storage_3d`. */
    public static new3D(label: string, width: number, height: number, depth: number, format: GPUTextureFormat, usage?: GPUTextureUsageFlags): Texture {
        return new Texture({ label, width, height, depthOrArrayLayers: depth, dimension: '3d', format, usage });
    }

    /** An empty 2D array of `layers` equal layers, bound as `texture_2d_array`. */
    public static new2DArray(label: string, width: number, height: number, layers: number, format: GPUTextureFormat, usage?: GPUTextureUsageFlags): Texture {
        return new Texture({ label, width, height, depthOrArrayLayers: layers, viewDimension: '2d-array', format, usage });
    }

    /** An empty cubemap of six `size` x `size` faces (+X, -X, +Y, -Y, +Z, -Z), bound as `texture_cube`. */
    public static newCube(label: string, size: number, format: GPUTextureFormat, usage?: GPUTextureUsageFlags): Texture {
        return new Texture({ label, width: size, height: size, depthOrArrayLayers: 6, viewDimension: 'cube', format, usage });
    }

    /** An RGBA8 2D texture from tightly packed rows. */
    public static fromRGBA(label: string, width: number, height: number, data: TextureLevelData): Texture {
        return new Texture({ label, width, height, levels: [data] });
    }

    /** An RGBA8 2D texture array from one `width` x `height` image per layer. */
    public static fromRGBALayers(label: string, width: number, height: number, layers: Uint8Array[]): Texture {
        return Texture.fromArrayLevels(label, 'rgba8unorm', width, height, layers.length, [concatBytes(layers)]);
    }

    /**
     * A 2D texture with its whole mip chain given, level 0 first, in any format including the
     * block-compressed ones (each level's blocks tightly packed, rounded up to whole blocks).
     */
    public static fromLevels(label: string, format: GPUTextureFormat, width: number, height: number, levels: TextureLevelData[]): Texture {
        return new Texture({ label, width, height, format, levels });
    }

    /**
     * A 2D array of `layers` layers with its whole mip chain given, level 0 first, each level
     * holding every layer in turn, bound as `texture_2d_array`.
     */
    public static fromArrayLevels(label: string, format: GPUTextureFormat, width: number, height: number, layers: number, levels: TextureLevelData[]): Texture {
        return new Texture({ label, width, height, depthOrArrayLayers: layers, viewDimension: '2d-array', format, levels });
    }

    /**
     * An image as an RGBA8 texture, its colour sRGB-encoded (read back linear, through an
     * `rgba8unorm-srgb` view) or linear data (`srgb: false`: normal maps, roughness).
     */
    public static fromImage(source: TextureSource, options: { label?: string; srgb?: boolean; mipmaps?: boolean } = {}): Texture {
        const texture = new Texture(source, options.mipmaps ?? false);
        if (options.label) texture.label = options.label;
        if (options.srgb) texture.viewFormat = 'rgba8unorm-srgb';
        return texture;
    }

    /**
     * Wrap a texture created elsewhere (a render target, a LUT, a cubemap) with the view to bind,
     * so it can be attached to a material like any other Texture. `view` defaults to the whole
     * texture seen as `viewDimension`.
     */
    public static fromView(label: string, texture: GPUTexture, view?: GPUTextureView, viewDimension?: GPUTextureViewDimension): Texture {
        const wrapped = new Texture({
            label,
            width: texture.width,
            height: texture.height,
            depthOrArrayLayers: texture.depthOrArrayLayers,
            format: texture.format,
            dimension: texture.dimension,
            viewDimension,
            mipLevelCount: texture.mipLevelCount,
            usage: texture.usage,
        });
        wrapped.texture = texture;
        wrapped.view = view ?? texture.createView({ label, dimension: wrapped.viewDimension, aspect: wrapped.viewAspect() });
        wrapped.initialized = true;
        return wrapped;
    }

    /** The mips down to 1 texel on the largest axis: `floor(log2(max side)) + 1`. */
    public static fullMipCount(width: number, height: number, depth: number = 1): number {
        return Math.floor(Math.log2(Math.max(width, height, depth, 1))) + 1;
    }

    /**
     * Builder: give the texture `levels` mips (at least 1), for a chain a pass fills and binds a
     * level at a time through `mipView`.
     */
    public withMipLevels(levels: number): Texture {
        this.mipLevelCount = Math.max(1, levels);
        return this;
    }

    /**
     * Updates the texture. Currently a placeholder for future implementation.
     * @returns {Promise<void>}
     */
    public async update(): Promise<void> {
    }

    /**
     * Initializes the texture with the given WebGPU device.
     * @param {GPUDevice} gpuDevice - The WebGPU device to create the texture with
     */
    public initialize(gpuDevice: GPUDevice) {
        if (this.texture) {
            this.initialized = true;
            return;
        }
        if (this.source) {
            this.texture = this.webGPUTextureFromImageBitmapOrCanvas(gpuDevice, this.source);
            if (this.mipmaps) this.createMipmaps(gpuDevice);
        } else {
            this.texture = gpuDevice.createTexture({
                label: this.label,
                size: { width: this.width, height: this.height, depthOrArrayLayers: this.depthOrArrayLayers },
                mipLevelCount: this.mipLevelCount,
                dimension: this.dimension,
                format: this.format,
                usage: this.usage,
            });
            this.levels?.forEach((data, level) => this.writeLevel(gpuDevice, level, data));
            this.levels = undefined;
        }
        this.view = this.texture.createView({
            label: this.label,
            dimension: this.viewDimension,
            format: this.viewFormat,
            aspect: this.viewAspect(),
        });
        this.initialized = true;
    }

    /**
     * Upload one mip level (every layer, or every slice of a 3D texture), tightly packed rows of
     * texels or blocks. Compressed copies cover whole blocks: the level's physical size.
     */
    public writeLevel(gpuDevice: GPUDevice, level: number, data: TextureLevelData) {
        const block = textureFormatBlock(this.format);
        const blocksX = Math.ceil(Math.max(1, this.width >> level) / block.width);
        const blocksY = Math.ceil(Math.max(1, this.height >> level) / block.height);
        const depth = this.dimension === '3d' ? Math.max(1, this.depthOrArrayLayers >> level) : this.depthOrArrayLayers;
        gpuDevice.queue.writeTexture(
            { texture: this.texture!, mipLevel: level },
            data,
            { offset: 0, bytesPerRow: blocksX * block.bytes, rowsPerImage: blocksY },
            { width: blocksX * block.width, height: blocksY * block.height, depthOrArrayLayers: depth },
        );
    }

    /**
     * A view of mip `level` alone (after initialization), to bind (through `Texture.fromView`) as a
     * storage texture to write that level or as a sampled texture to read it.
     */
    public mipView(level: number): GPUTextureView | undefined {
        if (!this.texture || level >= this.mipLevelCount) return undefined;
        return this.texture.createView({
            label: this.label,
            dimension: this.viewDimension,
            baseMipLevel: level,
            mipLevelCount: 1,
            aspect: this.viewAspect(),
        });
    }

    /** The GPU texture, once initialized. */
    get gpuTexture(): GPUTexture | undefined {
        return this.texture;
    }

    /**
     * Gets the binding resource for this texture.
     * @returns {GPUBindingResource} The texture view that can be used for binding
     */
    get resource(): GPUBindingResource {
        return this.view!;
    }

    /**
     * Sampled as its view dimension and format's sample type, unless the binding asks for a
     * storage texture (of this format and view dimension unless it names its own).
     */
    public getBindingLayout(gpuDevice: GPUDevice, requested?: BindingLayout): BindingLayout {
        if (requested && 'storageTexture' in requested) {
            return { storageTexture: { access: 'write-only', format: this.format, viewDimension: this.viewDimension, ...requested.storageTexture } };
        }
        const texture: GPUTextureBindingLayout = {
            sampleType: textureFormatSampleType(this.viewFormat ?? this.format, gpuDevice),
            viewDimension: this.viewDimension,
        };
        if (requested && 'texture' in requested) return { texture: { ...texture, ...requested.texture } };
        return requested ?? { texture };
    }

    /** Depth-stencil textures are sampled through their depth aspect. */
    private viewAspect(): GPUTextureAspect {
        return isDepthFormat(this.format) && this.format.includes('stencil') ? 'depth-only' : 'all';
    }

    /**
     * Creates a WebGPU texture from an ImageBitmap or Canvas source.
     * @param {GPUDevice} gpuDevice - The WebGPU device to create the texture with
     * @param {ImageBitmap | HTMLCanvasElement} source - The source image or canvas
     * @returns {GPUTexture} The created WebGPU texture
     * @private
     */
    private webGPUTextureFromImageBitmapOrCanvas(gpuDevice: GPUDevice, source: TextureSource) {
        const textureDescriptor: GPUTextureDescriptor = {
            label: this.label,
            size: { width: source.width, height: source.height },
            format: this.format,
            usage: this.usage,
            mipLevelCount: this.mipLevelCount,
            viewFormats: this.viewFormat ? [this.viewFormat] : [],
        };
        const texture = gpuDevice.createTexture(textureDescriptor);

        gpuDevice.queue.copyExternalImageToTexture({ source }, { texture }, textureDescriptor.size);

        return texture;
    }

    public createMipmaps(gpuDevice: GPUDevice) {
        const computePipeline = gpuDevice.createComputePipeline({
            layout: 'auto',
            compute: {
                module: gpuDevice.createShaderModule({ code: this.mipmapShader }),
                entryPoint: 'main',
            },
        });

        let width = this.texture!.width;
        let height = this.texture!.height;

        for (let level = 0; level < this.texture!.mipLevelCount - 1; level++) {
            const inputView = this.texture!.createView({
                baseMipLevel: level,
                mipLevelCount: 1,
            });
            const outputView = this.texture!.createView({
                baseMipLevel: level + 1,
                mipLevelCount: 1,
            });

            const bindGroup = gpuDevice.createBindGroup({
                layout: computePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: inputView },
                    { binding: 1, resource: outputView },
                ],
            });

            const commandEncoder = gpuDevice.createCommandEncoder();
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(computePipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(Math.ceil(width / 2), Math.ceil(height / 2));
            pass.end();
            gpuDevice.queue.submit([commandEncoder.finish()]);

            width /= 2;
            height /= 2;
        }

    }

    private mipmapShader: string = /* wgsl */`
        @group(0) @binding(0) var inputTexture: texture_2d<f32>;
        @group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let outCoord = vec2<u32>(global_id.xy);
            
            // Sample in a 6x3 pattern for enhanced anisotropic filtering
            let color = (
                // First (top) row - weight 0.1
                textureLoad(inputTexture, vec2<i32>(outCoord * 2u) + vec2<i32>(-2, -1), 0) * 0.01 +
                textureLoad(inputTexture, vec2<i32>(outCoord * 2u) + vec2<i32>(-1, -1), 0) * 0.02 +
                textureLoad(inputTexture, vec2<i32>(outCoord * 2u) + vec2<i32>(0, -1), 0) * 0.03 +
                textureLoad(inputTexture, vec2<i32>(outCoord * 2u) + vec2<i32>(1, -1), 0) * 0.02 +
                textureLoad(inputTexture, vec2<i32>(outCoord * 2u) + vec2<i32>(2, -1), 0) * 0.01 +
                textureLoad(inputTexture, vec2<i32>(outCoord * 2u) + vec2<i32>(3, -1), 0) * 0.01 +

                // Middle row (primary sampling) - weight 0.5
                textureLoad(inputTexture, vec2<i32>(outCoord * 2u) + vec2<i32>(-2, 0), 0) * 0.05 +
                textureLoad(inputTexture, vec2<i32>(outCoord * 2u) + vec2<i32>(-1, 0), 0) * 0.1 +
                textureLoad(inputTexture, vec2<i32>(outCoord * 2u) + vec2<i32>(0, 0), 0) * 0.15 +
                textureLoad(inputTexture, vec2<i32>(outCoord * 2u) + vec2<i32>(1, 0), 0) * 0.1 +
                textureLoad(inputTexture, vec2<i32>(outCoord * 2u) + vec2<i32>(2, 0), 0) * 0.05 +
                textureLoad(inputTexture, vec2<i32>(outCoord * 2u) + vec2<i32>(3, 0), 0) * 0.05 +

                // Bottom row - weight 0.4
                textureLoad(inputTexture, vec2<i32>(outCoord * 2u) + vec2<i32>(-2, 1), 0) * 0.04 +
                textureLoad(inputTexture, vec2<i32>(outCoord * 2u) + vec2<i32>(-1, 1), 0) * 0.08 +
                textureLoad(inputTexture, vec2<i32>(outCoord * 2u) + vec2<i32>(0, 1), 0) * 0.12 +
                textureLoad(inputTexture, vec2<i32>(outCoord * 2u) + vec2<i32>(1, 1), 0) * 0.08 +
                textureLoad(inputTexture, vec2<i32>(outCoord * 2u) + vec2<i32>(2, 1), 0) * 0.04 +
                textureLoad(inputTexture, vec2<i32>(outCoord * 2u) + vec2<i32>(3, 1), 0) * 0.04
            );

            textureStore(outputTexture, vec2<i32>(outCoord), color);
        }
    `;
}

/** Options are a plain object; images and canvases are platform objects. */
function isTextureOptions(source: TextureSource | TextureOptions): source is TextureOptions {
    const prototype = Object.getPrototypeOf(source);
    return prototype === Object.prototype || prototype === null;
}

function concatBytes(parts: Uint8Array[]): Uint8Array {
    const out = new Uint8Array(parts.reduce((sum, part) => sum + part.byteLength, 0));
    let offset = 0;
    for (const part of parts) {
        out.set(part, offset);
        offset += part.byteLength;
    }
    return out;
}

export { Texture }
